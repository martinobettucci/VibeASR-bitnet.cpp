#!/usr/bin/env python3
"""Short (<30 s) vs long (>30 s) comparison: every engine, every language.

    python3 bench/make_shortlong.py
    python3 bench/shortlong_bench.py -t 4

14 items (7 languages x 2 regimes). Engines are interleaved PER ITEM -- all
engines transcribe an item back to back before moving on -- so each speed ratio
has both halves inside one noise window; sequential phase-per-engine timing
drifts enough on shared hosts to invert known ratios.

Reports WER and compute per engine per regime, plus the speed ratio against the
upstream runtime baseline. The two regimes are reported separately because they
rank the engines differently, which is the entire point of the exercise.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import score  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RX_W_TOTAL = re.compile(r"total time =\s+([\d.]+) ms")
RX_W_LOAD = re.compile(r"load time =\s+([\d.]+) ms")
STAGES = ("vae_acoustic", "vae_semantic", "prompt_build", "prefill", "decode")


def run_vibe(binary, vae, lm, wav, threads, timeout):
    p = subprocess.run([binary, "--vae-model", vae, "--lm-model", lm, "--audio", wav,
                        "-t", str(threads), "--greedy"], capture_output=True, timeout=timeout)
    if p.returncode != 0:
        return None, None
    err = p.stderr.decode("utf-8", "replace")

    def g(k):
        m = re.search(k + r":\s+([\d.]+) ms", err)
        return float(m.group(1)) if m else 0.0

    compute = sum(g(k) for k in ("VAE acoustic encode", "VAE semantic encode",
                                 "Prompt build", "LM prefill", "LM decode"))
    hyp = " ".join(l.strip() for l in p.stdout.decode("utf-8", "replace").splitlines()
                   if l.strip())
    return hyp, compute


def run_whisper(binary, model, wav, iso, threads, timeout):
    p = subprocess.run([binary, "-m", model, "-f", wav, "-t", str(threads),
                        "-l", iso, "--no-timestamps"], capture_output=True, timeout=timeout)
    err = p.stderr.decode("utf-8", "replace")
    mt, ml = RX_W_TOTAL.search(err), RX_W_LOAD.search(err)
    if not mt or p.returncode != 0:
        return None, None
    compute = float(mt.group(1)) - (float(ml.group(1)) if ml else 0.0)
    hyp = " ".join(l.strip() for l in p.stdout.decode("utf-8", "replace").splitlines()
                   if l.strip())
    return hyp, compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--vae", default="models/vibeasr/vibeasr-vae-encoder-i8_s.gguf")
    ap.add_argument("--lm", default="models/vibeasr/vibeasr-lm-i2_s-tied.gguf")
    ap.add_argument("--bin", default="build/bin/asr_infer")
    ap.add_argument("--upstream-bin", default="/home/user/upstream-vibeasr/build/bin/asr_infer")
    ap.add_argument("--upstream-lm", default="models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf")
    ap.add_argument("--whisper-bin", default="/home/user/whisper.cpp/build/bin/whisper-cli")
    ap.add_argument("--whisper-small", default="/home/user/whisper.cpp/models/ggml-small-q5_1.bin")
    ap.add_argument("--whisper-turbo", default="/home/user/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin")
    ap.add_argument("--data", default=os.path.join(ROOT, "bench", "data", "shortlong"))
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results"))
    args = ap.parse_args()

    items = json.load(open(os.path.join(args.data, "index.json"), encoding="utf-8"))

    def engines(wav, iso):
        return [
            ("this fork", lambda: run_vibe(args.bin, args.vae, args.lm, wav,
                                           args.threads, args.timeout)),
            ("upstream", lambda: run_vibe(args.upstream_bin, args.vae, args.upstream_lm,
                                          wav, args.threads, args.timeout)),
            ("whisper-small", lambda: run_whisper(args.whisper_bin, args.whisper_small,
                                                  wav, iso, args.threads, args.timeout)),
            ("whisper-turbo", lambda: run_whisper(args.whisper_bin, args.whisper_turbo,
                                                  wav, iso, args.threads, args.timeout)),
        ]

    recs = []
    print("%-14s %-6s %6s | %s" % ("item", "regime", "audio",
                                   "  ".join("%-14s" % e for e in
                                             ("this fork", "upstream", "whisper-small", "whisper-turbo"))))
    for it in items:
        wav = os.path.join(args.data, it["name"] + ".wav")
        iso = it["lang"].split("_")[0]
        row = {"item": it["name"], "lang": it["lang"], "regime": it["regime"],
               "duration": it["duration"], "engines": {}}
        cells = []
        for label, fn in engines(wav, iso):
            try:
                hyp, compute = fn()
            except subprocess.TimeoutExpired:
                hyp, compute = None, None
            if hyp is None:
                row["engines"][label] = {"error": True}
                cells.append("%-14s" % "FAIL")
                continue
            we, ww, _, _ = score(it["text"], hyp, iso, True)
            wer = 100.0 * we / ww
            row["engines"][label] = {"wer": wer, "compute_ms": compute, "hyp": hyp}
            cells.append("%5.1f/%6.1fs" % (wer, compute / 1000.0))
        recs.append(row)
        print("%-14s %-6s %5.1fs | %s" % (it["name"], it["regime"], it["duration"],
                                          "  ".join(cells)), flush=True)

    # aggregate per regime: corpus WER (word-weighted) and speed vs upstream
    print("\n%-8s %-16s %8s %10s %12s" % ("regime", "engine", "WER", "compute", "vs upstream"))
    summary = {}
    for regime in ("short", "long"):
        rows = [r for r in recs if r["regime"] == regime]
        for label in ("upstream", "this fork", "whisper-small", "whisper-turbo"):
            we = ww = 0.0
            tot = 0.0
            base = 0.0
            ok = 0
            for r in rows:
                e = r["engines"].get(label, {})
                b = r["engines"].get("upstream", {})
                if "wer" not in e:
                    continue
                it = next(x for x in items if x["name"] == r["item"])
                n = len(it["text"].split())
                we += e["wer"] * n
                ww += n
                tot += e["compute_ms"]
                if "compute_ms" in b:
                    base += b["compute_ms"]
                ok += 1
            if not ok:
                continue
            ratio = base / tot if tot else float("nan")
            summary["%s/%s" % (regime, label)] = {"wer": we / ww, "compute_s": tot / 1000.0,
                                                  "vs_upstream": ratio, "n": ok}
            print("%-8s %-16s %8.2f %9.1fs %11.2fx" %
                  (regime, label, we / ww, tot / 1000.0, ratio))

    path = os.path.join(args.out, "shortlong_t%d.json" % args.threads)
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": recs}, f, ensure_ascii=False, indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
