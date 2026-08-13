#!/usr/bin/env python3
"""Long-form benchmark: one recording, every engine, interleaved.

    python3 bench/longform_bench.py --wav bench/data/longform/conv_2spk_2min.wav

Every other benchmark here runs 5-25 s utterances, which is whisper's ideal case
(fixed 30 s encoder window, fully used) and this architecture's worst (a 3200x
audio-compressing LLM decoder that emits speaker-labelled segments has nothing to
show at that length). This one runs whole recordings.

Reports, per engine: compute time, RTF, WER against the stitched reference, and
whether the engine produced speaker labels at all. Engines run back to back on
one recording, so the timings share a noise window (see interleaved_speed.py for
why that matters).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import score  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RX_W_TOTAL = re.compile(r"total time =\s+([\d.]+) ms")
RX_W_LOAD = re.compile(r"load time =\s+([\d.]+) ms")
STAGES = ("vae_acoustic", "vae_semantic", "prompt_build", "prefill", "decode")


def run_vibe(binary, vae, lm, wav, threads, json_out, timeout):
    cmd = [binary, "--vae-model", vae, "--lm-model", lm, "--audio", wav,
           "-t", str(threads), "--greedy"]
    if json_out:
        cmd += ["--prompt-format", "json"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    wall = time.time() - t0
    if p.returncode != 0:
        return None, {"error": "rc=%d" % p.returncode}, wall
    out = p.stdout.decode("utf-8", "replace")
    err = p.stderr.decode("utf-8", "replace")
    g = lambda k: (float(re.search(k + r":\s+([\d.]+) ms", err).group(1))
                   if re.search(k + r":\s+([\d.]+) ms", err) else 0.0)
    stats = {"vae_acoustic": g("VAE acoustic encode"),
             "vae_semantic": g("VAE semantic encode"),
             "prompt_build": g("Prompt build"), "prefill": g("LM prefill"),
             "decode": g("LM decode")}
    stats["compute_ms"] = sum(stats[s] for s in STAGES)
    hyp = "\n".join(l for l in out.splitlines() if l.strip())
    return hyp, stats, wall


def run_whisper(binary, model, wav, iso, threads, timeout):
    t0 = time.time()
    p = subprocess.run([binary, "-m", model, "-f", wav, "-t", str(threads),
                        "-l", iso, "--no-timestamps"],
                       capture_output=True, timeout=timeout)
    wall = time.time() - t0
    err = p.stderr.decode("utf-8", "replace")
    mt, ml = RX_W_TOTAL.search(err), RX_W_LOAD.search(err)
    if not mt or p.returncode != 0:
        return None, {"error": "rc=%d" % p.returncode}, wall
    compute = float(mt.group(1)) - (float(ml.group(1)) if ml else 0.0)
    hyp = " ".join(l.strip() for l in p.stdout.decode("utf-8", "replace").splitlines()
                   if l.strip())
    return hyp, {"compute_ms": compute}, wall


def plain(hyp):
    """Strip our segment/speaker decoration down to words, for WER."""
    if not hyp:
        return ""
    # "[0.00 - 3.20] Speaker 0: text"  ->  "text"
    lines = []
    for l in hyp.splitlines():
        m = re.match(r"^\[[^\]]*\]\s*(?:Speaker\s+\d+:\s*)?(.*)$", l.strip())
        lines.append(m.group(1) if m else l.strip())
    return " ".join(x for x in lines if x)


def has_speakers(hyp):
    return bool(hyp) and bool(re.search(r"Speaker\s+\d+", hyp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--iso", default="fr")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--vae", default="models/vibeasr/vibeasr-vae-encoder-i8_s.gguf")
    ap.add_argument("--lm", default="models/vibeasr/vibeasr-lm-i2_s-tied.gguf")
    ap.add_argument("--bin", default="build/bin/asr_infer")
    ap.add_argument("--upstream-bin", default="/home/user/upstream-vibeasr/build/bin/asr_infer")
    ap.add_argument("--upstream-lm", default="models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf")
    ap.add_argument("--whisper-bin", default="/home/user/whisper.cpp/build/bin/whisper-cli")
    ap.add_argument("--whisper-small", default="/home/user/whisper.cpp/models/ggml-small-q5_1.bin")
    ap.add_argument("--whisper-turbo", default="/home/user/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results"))
    args = ap.parse_args()

    meta = json.load(open(os.path.splitext(args.wav)[0] + ".json", encoding="utf-8"))
    ref, dur = meta["text"], meta["duration"]
    print("%s: %.1f s, %d speakers, %d turns\n"
          % (os.path.basename(args.wav), dur, meta["n_speakers"], meta["n_turns"]))

    engines = [
        ("this fork (JSON segments)", lambda: run_vibe(args.bin, args.vae, args.lm,
                                                       args.wav, args.threads, True, args.timeout)),
        ("upstream runtime", lambda: run_vibe(args.upstream_bin, args.vae, args.upstream_lm,
                                              args.wav, args.threads, True, args.timeout)),
        ("whisper small q5_1", lambda: run_whisper(args.whisper_bin, args.whisper_small,
                                                   args.wav, args.iso, args.threads, args.timeout)),
        ("whisper large-v3-turbo q5", lambda: run_whisper(args.whisper_bin, args.whisper_turbo,
                                                          args.wav, args.iso, args.threads, args.timeout)),
    ]

    rows = []
    for label, fn in engines:
        try:
            hyp, stats, wall = fn()
        except subprocess.TimeoutExpired:
            hyp, stats, wall = None, {"error": "timeout"}, args.timeout
        if hyp is None:
            print("%-28s FAILED (%s)" % (label, stats.get("error")), flush=True)
            rows.append({"engine": label, "error": stats.get("error")})
            continue
        we, ww, _, _ = score(ref, plain(hyp), args.iso, True)
        wer = 100.0 * we / ww
        rows.append({"engine": label, "compute_ms": stats["compute_ms"], "wall_s": wall,
                     "rtf": stats["compute_ms"] / 1000.0 / dur, "wer": wer,
                     "speakers": has_speakers(hyp), "hyp": hyp, **stats})
        print("%-28s compute %7.1f s   RTF %5.2f   WER %5.2f   speakers: %s"
              % (label, stats["compute_ms"] / 1000.0,
                 stats["compute_ms"] / 1000.0 / dur, wer,
                 "yes" if has_speakers(hyp) else "no"), flush=True)

    ok = [r for r in rows if "compute_ms" in r]
    if ok:
        base = ok[0]
        print("\nspeed relative to %s:" % base["engine"])
        for r in ok[1:]:
            print("  %-28s %.2fx %s" % (r["engine"],
                                        r["compute_ms"] / base["compute_ms"],
                                        "slower" if r["compute_ms"] > base["compute_ms"] else "faster"))

    os.makedirs(args.out, exist_ok=True)
    tag = "longform_" + os.path.splitext(os.path.basename(args.wav))[0]
    path = os.path.join(args.out, tag + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"wav": os.path.basename(args.wav), "duration": dur,
                   "n_speakers": meta["n_speakers"], "n_turns": meta["n_turns"],
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
