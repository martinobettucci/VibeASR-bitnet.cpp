#!/usr/bin/env python3
"""Relative WER + speed table across engines, from speed_test/whisper_bench JSONs.

    python3 bench/relative_report.py \
        --ours bench/results/rel_ours.json \
        --ref  bench/results/rel_upstream.json=upstream \
        --ref  bench/results/rel_whisper_turbo.json=whisper-large-v3-turbo-q5 \
        [--ref ...]

Why relative: absolute RTF depends on whatever host the benchmark landed on (cloud
VMs migrate, neighbours steal cycles). Two things transfer across hosts: WER (a
property of model + decoder alone) and the per-clip speed RATIO between engines run
back-to-back on one host. Everything printed here is one of those two.

Speed ratio per clip = other_compute / ours_compute; the table reports the median
of the paired per-clip ratios (robust to a noise spike hitting one clip) plus the
corpus-level ratio (total compute / total compute, weighting long clips more).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import score  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["records"]


def wer_by_lang(recs):
    agg = defaultdict(lambda: [0, 0])
    for r in recs:
        iso = r["lang"].split("_")[0]
        we, ww, _, _ = score(r["ref"], r["hyp"], iso, True)
        agg[r["lang"]][0] += we
        agg[r["lang"]][1] += ww
    total = [sum(v[0] for v in agg.values()), sum(v[1] for v in agg.values())]
    out = {k: 100.0 * v[0] / v[1] for k, v in agg.items() if v[1]}
    out["ALL"] = 100.0 * total[0] / total[1]
    return out


def median(xs):
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])) if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", action="append", default=[],
                    help="path=json label, repeatable")
    args = ap.parse_args()

    ours = load(args.ours)
    okey = {(r["lang"], r["wav"]): r for r in ours}
    engines = [("this fork", ours)]
    for spec in args.ref:
        path, _, label = spec.partition("=")
        engines.append((label or os.path.basename(path), load(path)))

    print("%-28s %8s | per-language WER" % ("engine", "WER ALL"))
    langs = sorted({r["lang"] for r in ours})
    for label, recs in engines:
        w = wer_by_lang(recs)
        cells = "  ".join("%s %.1f" % (l, w[l]) for l in langs if l in w)
        print("%-28s %8.2f | %s" % (label, w["ALL"], cells))

    print("\n%-28s %14s %14s %8s" %
          ("engine", "speed vs ours", "corpus ratio", "paired"))
    print("%-28s %14s %14s %8s" % ("this fork", "1.00x", "1.00x", "n=%d" % len(ours)))
    for label, recs in engines[1:]:
        pairs = [(okey[(r["lang"], r["wav"])], r) for r in recs
                 if (r["lang"], r["wav"]) in okey]
        ratios = [b["compute_ms"] / a["compute_ms"] for a, b in pairs]
        corpus = (sum(b["compute_ms"] for _, b in pairs)
                  / sum(a["compute_ms"] for a, _ in pairs))
        print("%-28s %13.2fx %13.2fx %8s" %
              (label, median(ratios), corpus, "n=%d" % len(pairs)))
    print("\n(ratio > 1: that engine is slower than this fork on the same clip,")
    print(" same host, same threads. Absolute RTF intentionally not shown.)")


if __name__ == "__main__":
    main()
