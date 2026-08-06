#!/usr/bin/env python3
"""RTF as a function of thread count, on a fixed set of clips.

    python3 bench/thread_scaling.py --threads 1,2,3,4 --lang en_us -n 4

Prints one row per thread count with the compute-only RTF and its breakdown, so the
scaling limit is visible per stage rather than only in the total.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_asr import run_clip  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ["vae_acoustic", "vae_semantic", "prefill", "decode"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,2,3,4")
    ap.add_argument("--lang", default="en_us")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--isa", default=None)
    ap.add_argument("--vae", default="models/vibeasr/vibeasr-vae-encoder-i8_s.gguf")
    ap.add_argument("--lm", default="models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf")
    ap.add_argument("--bin", default="build/bin/asr_infer")
    ap.add_argument("--data", default=os.path.join(ROOT, "bench", "data"))
    args = ap.parse_args()

    d = os.path.join(args.data, args.lang)
    refs = [json.loads(l) for l in open(os.path.join(d, "refs.jsonl"), encoding="utf-8")][: args.n]
    audio = sum(r["duration"] for r in refs)
    extra_env = {"VIBEASR_ISA": args.isa} if args.isa else {}

    print("%s, %d clips, %.1f s audio, isa=%s\n" %
          (args.lang, len(refs), audio, args.isa or "auto"))
    print("%8s %8s %10s %10s %9s %9s" %
          ("threads", "RTF", "VAE-ac ms", "VAE-sem ms", "prefill", "decode"))
    for t in [int(x) for x in args.threads.split(",")]:
        totals = {k: 0.0 for k in STAGES}
        for r in refs:
            _, stats, _ = run_clip(args.bin, args.vae, args.lm,
                                   os.path.join(d, r["wav"]), t, extra_env, 900)
            for k in STAGES:
                totals[k] += stats.get(k) or 0.0
        compute = sum(totals.values()) / 1000.0
        print("%8d %8.3f %10.0f %10.0f %9.0f %9.0f" %
              (t, compute / audio, totals["vae_acoustic"], totals["vae_semantic"],
               totals["prefill"], totals["decode"]))


if __name__ == "__main__":
    main()
