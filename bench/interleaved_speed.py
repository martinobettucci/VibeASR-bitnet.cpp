#!/usr/bin/env python3
"""Interleaved multi-engine speed measurement: engines cycle PER CLIP.

    python3 bench/interleaved_speed.py -n 40 -t 4 --tag inter40 \
        --engine fork="build/bin/asr_infer|models/vibeasr/vibeasr-lm-i2_s-tied.gguf|" \
        --engine exact="build/bin/asr_infer|models/vibeasr/vibeasr-lm-i2_s-tied.gguf|VIBEASR_RES_FUSE=0" \
        --engine upstream="/home/user/upstream-vibeasr/build/bin/asr_infer|models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf|"

Sequential phase-per-engine sweeps proved inadequate on shared cloud hosts: the
host's throughput drifted 30%+ BETWEEN phases, large enough to invert a known
1.15x ratio. Here every clip is measured on all engines back to back within a
couple of minutes, so each per-clip ratio shares its noise window; the median of
those ratios is robust to drift on any longer timescale.

Whisper engines: pass --whisper label="bin|model" -- language codes and timing
parsing are handled per engine kind. WER is NOT computed here (transcripts are
deterministic per engine; take them from the full sweeps).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speed_test import SUPPORTED, draw, pct  # noqa: E402
from run_asr import run_clip  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RX_TOTAL = re.compile(r"total time =\s+([\d.]+) ms")
RX_LOAD  = re.compile(r"load time =\s+([\d.]+) ms")


def run_vibe(binary, lm, env_kv, wav, threads):
    env = {}
    if env_kv:
        k, _, v = env_kv.partition("=")
        env[k] = v
    hyp, stats, _ = run_clip(binary, "models/vibeasr/vibeasr-vae-encoder-i8_s.gguf",
                             lm, wav, threads, env, 900)
    if hyp is None:
        return None
    vae = stats.get("vae_parallel") or ((stats.get("vae_acoustic") or 0) +
                                        (stats.get("vae_semantic") or 0))
    return vae + (stats.get("prompt_build") or 0) + (stats.get("prefill") or 0) + \
        (stats.get("decode") or 0)


def run_whisper(binary, model, wav, iso, threads):
    p = subprocess.run([binary, "-m", model, "-f", wav, "-t", str(threads),
                        "-l", iso, "--no-timestamps"],
                       capture_output=True, timeout=900)
    err = p.stderr.decode("utf-8", "replace")
    mt, ml = RX_TOTAL.search(err), RX_LOAD.search(err)
    if not mt or p.returncode != 0:
        return None
    return float(mt.group(1)) - (float(ml.group(1)) if ml else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--engine", action="append", default=[],
                    help='label="bin|lm-gguf|ENV=V" (vibeasr-style CLI)')
    ap.add_argument("--whisper", action="append", default=[],
                    help='label="bin|model"')
    ap.add_argument("--langs", default=",".join(SUPPORTED))
    ap.add_argument("--data", default=os.path.join(ROOT, "bench", "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results"))
    args = ap.parse_args()

    engines = []
    for spec in args.engine:
        label, _, rest = spec.partition("=")
        b, lm, envkv = (rest.strip('"').split("|") + ["", ""])[:3]
        engines.append((label, "vibe", (b, lm, envkv)))
    for spec in args.whisper:
        label, _, rest = spec.partition("=")
        b, model = rest.strip('"').split("|")[:2]
        engines.append((label, "whisper", (b, model)))
    if len(engines) < 2:
        sys.exit("need at least 2 engines")

    clips = draw(args.data, args.langs.split(","), args.n)
    labels = [e[0] for e in engines]
    print("%d clips x %s" % (len(clips), "/".join(labels)))

    # one warmup clip per engine, discarded (page-cache/pack effects)
    for label, kind, cfg in engines:
        lang, r = clips[0]
        wav = os.path.join(args.data, lang, r["wav"])
        if kind == "vibe":
            run_vibe(cfg[0], cfg[1], cfg[2], wav, args.threads)
        else:
            run_whisper(cfg[0], cfg[1], wav, lang.split("_")[0], args.threads)
    print("warmup done", flush=True)

    recs = []
    for k, (lang, r) in enumerate(clips):
        wav = os.path.join(args.data, lang, r["wav"])
        row = {"lang": lang, "wav": r["wav"], "duration": r["duration"], "ms": {}}
        for label, kind, cfg in engines:
            if kind == "vibe":
                ms = run_vibe(cfg[0], cfg[1], cfg[2], wav, args.threads)
            else:
                ms = run_whisper(cfg[0], cfg[1], wav, lang.split("_")[0], args.threads)
            row["ms"][label] = ms
        recs.append(row)
        print("%-3d %-7s " % (k, lang) +
              "  ".join("%s %7.0f" % (l, row["ms"][l] or -1) for l in labels),
              flush=True)

    base = labels[0]
    print("\npaired per-clip ratios vs %s (ratio > 1 = slower than %s):" % (base, base))
    summary = {}
    for l in labels[1:]:
        ratios = sorted(r["ms"][l] / r["ms"][base] for r in recs
                        if r["ms"][l] and r["ms"][base])
        corpus = (sum(r["ms"][l] for r in recs if r["ms"][l] and r["ms"][base])
                  / sum(r["ms"][base] for r in recs if r["ms"][l] and r["ms"][base]))
        summary[l] = {"median": pct(ratios, 50), "p10": pct(ratios, 10),
                      "p90": pct(ratios, 90), "corpus": corpus, "n": len(ratios)}
        print("  %-24s median %.2fx   p10 %.2fx  p90 %.2fx   corpus %.2fx   n=%d"
              % (l, pct(ratios, 50), pct(ratios, 10), pct(ratios, 90), corpus,
                 len(ratios)))

    path = os.path.join(args.out, args.tag + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"base": base, "summary": summary, "records": recs}, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
