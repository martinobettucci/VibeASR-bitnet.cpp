#!/usr/bin/env python3
"""Evaluate real long-form recordings: whisper native vs this stack chunked.

    python3 bench/fetch_longform.py --dataset tedlium -n 6
    python3 bench/lf_eval.py --item tedlium_05 --chunk 60

Real long-form means whole recordings (TED talks here, 8-25 min). Two facts shape
the comparison and both are stated in the output:

  * whisper.cpp handles arbitrary length natively -- it slides a 30 s encoder
    window and carries decoder state across windows.
  * VibeVoice-ASR does not. Its usable single-pass range is ~80 s (past that the
    decoder emits its end token early and drops the tail), and this runtime's
    encoder arena costs ~110 MB per second of audio, capping a pass near 135 s on
    a 16 GB host. So this stack is evaluated the way it would actually be
    deployed: fixed-length chunks, transcribed independently, concatenated.

Chunking is a real cost, not a footnote -- it loses cross-chunk context and can
clip words at boundaries. It is reported as what it is.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from scoring import score  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RX_W_TOTAL = re.compile(r"total time =\s+([\d.]+) ms")
RX_W_LOAD = re.compile(r"load time =\s+([\d.]+) ms")


def vibe_chunked(binary, vae, lm, wav, threads, chunk_sec, timeout, scratch):
    """Transcribe in fixed chunks; returns (text, compute_ms, n_chunks)."""
    data, sr = sf.read(wav, dtype="int16")
    n = int(chunk_sec * sr)
    parts, total = [], 0.0
    os.makedirs(scratch, exist_ok=True)
    for k in range(0, len(data), n):
        seg = data[k: k + n]
        if len(seg) < sr:            # ignore a sub-second tail
            break
        p = os.path.join(scratch, "chunk.wav")
        sf.write(p, seg, sr, subtype="PCM_16")
        r = subprocess.run([binary, "--vae-model", vae, "--lm-model", lm,
                            "--audio", p, "-t", str(threads), "--greedy"],
                           capture_output=True, timeout=timeout)
        if r.returncode != 0:
            return None, None, 0
        err = r.stderr.decode("utf-8", "replace")
        for key in ("VAE acoustic encode", "VAE semantic encode", "Prompt build",
                    "LM prefill", "LM decode"):
            m = re.search(key + r":\s+([\d.]+) ms", err)
            if m:
                total += float(m.group(1))
        parts.append(" ".join(l.strip() for l in
                              r.stdout.decode("utf-8", "replace").splitlines() if l.strip()))
    return " ".join(parts), total, len(parts)


def whisper_native(binary, model, wav, iso, threads, timeout):
    r = subprocess.run([binary, "-m", model, "-f", wav, "-t", str(threads),
                        "-l", iso, "--no-timestamps"], capture_output=True, timeout=timeout)
    err = r.stderr.decode("utf-8", "replace")
    mt, ml = RX_W_TOTAL.search(err), RX_W_LOAD.search(err)
    if not mt or r.returncode != 0:
        return None, None
    compute = float(mt.group(1)) - (float(ml.group(1)) if ml else 0.0)
    return " ".join(l.strip() for l in
                    r.stdout.decode("utf-8", "replace").splitlines() if l.strip()), compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "bench", "data", "lf_tedlium"))
    ap.add_argument("--item", required=True)
    ap.add_argument("--chunk", type=float, default=60.0)
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--iso", default="en")
    ap.add_argument("--vae", default="models/vibeasr/vibeasr-vae-encoder-i8_s.gguf")
    ap.add_argument("--lm", default="models/vibeasr/vibeasr-lm-i2_s-tied.gguf")
    ap.add_argument("--bin", default="build/bin/asr_infer")
    ap.add_argument("--upstream-bin", default="/home/user/upstream-vibeasr/build/bin/asr_infer")
    ap.add_argument("--upstream-lm", default="models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf")
    ap.add_argument("--whisper-bin", default="/home/user/whisper.cpp/build/bin/whisper-cli")
    ap.add_argument("--whisper-small", default="/home/user/whisper.cpp/models/ggml-small-q5_1.bin")
    ap.add_argument("--whisper-turbo", default="/home/user/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin")
    ap.add_argument("--scratch", default="/tmp/lf_chunks")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results"))
    args = ap.parse_args()

    index = json.load(open(os.path.join(args.data, "index.json"), encoding="utf-8"))
    it = next(x for x in index if x["name"] == args.item)
    wav = os.path.join(args.data, it["name"] + ".wav")
    ref, dur = it["text"], it["duration"]
    print("%s: %.1f s (%.1f min), %d reference words\n"
          % (it["name"], dur, dur / 60.0, len(ref.split())))

    runs = [
        ("this fork (chunked %.0fs)" % args.chunk,
         lambda: vibe_chunked(args.bin, args.vae, args.lm, wav, args.threads,
                              args.chunk, args.timeout, args.scratch)),
        ("upstream (chunked %.0fs)" % args.chunk,
         lambda: vibe_chunked(args.upstream_bin, args.vae, args.upstream_lm, wav,
                              args.threads, args.chunk, args.timeout, args.scratch)),
        ("whisper-small (native)",
         lambda: whisper_native(args.whisper_bin, args.whisper_small, wav, args.iso,
                                args.threads, args.timeout) + (0,)),
        ("whisper-turbo (native)",
         lambda: whisper_native(args.whisper_bin, args.whisper_turbo, wav, args.iso,
                                args.threads, args.timeout) + (0,)),
    ]

    rows = []
    for label, fn in runs:
        t0 = time.time()
        try:
            hyp, compute, nchunk = fn()
        except subprocess.TimeoutExpired:
            hyp, compute, nchunk = None, None, 0
        wall = time.time() - t0
        if hyp is None:
            print("%-26s FAILED" % label, flush=True)
            rows.append({"engine": label, "error": True})
            continue
        we, ww, _, _ = score(ref, hyp, args.iso, True)
        wer = 100.0 * we / ww
        rows.append({"engine": label, "wer": wer, "compute_ms": compute,
                     "rtf": compute / 1000.0 / dur, "wall_s": wall,
                     "chunks": nchunk, "words": len(hyp.split()), "hyp": hyp})
        print("%-26s WER %6.2f   compute %7.1f s   RTF %5.2f   words %5d%s"
              % (label, wer, compute / 1000.0, compute / 1000.0 / dur, len(hyp.split()),
                 "   chunks %d" % nchunk if nchunk else ""), flush=True)

    ok = [r for r in rows if "wer" in r]
    if ok:
        base = next((r for r in ok if r["engine"].startswith("upstream")), ok[0])
        print("\nspeed vs %s:" % base["engine"])
        for r in ok:
            print("  %-26s %.2fx" % (r["engine"], base["compute_ms"] / r["compute_ms"]))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "lf_%s.json" % it["name"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"item": it["name"], "duration": dur, "ref_words": len(ref.split()),
                   "chunk_sec": args.chunk, "rows": rows}, f, ensure_ascii=False, indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
