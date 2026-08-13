#!/usr/bin/env python3
"""Final publication table: WER + speed-over-baseline for every engine, markdown.

    python3 bench/final_table.py \
        --baseline bench/results/fin_upstream.json="upstream engine, original weights" \
        --engine bench/results/fin_fork.json="this fork (residual fusion, default)" \
        --engine bench/results/fin_fork_exact.json="this fork (bit-exact mode)" \
        --engine bench/results/fin_whisper_turbo.json="whisper.cpp large-v3-turbo q5" \
        --engine bench/results/fin_whisper_small.json="whisper.cpp small q5"

Speed is expressed OVER THE BASELINE (upstream original code). With --speed, the
ratios come from an interleaved run (all engines per clip) rather than from the
WER runs, because cross-phase timing on a shared host is not trustworthy.

"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import score  # noqa: E402


def load(spec):
    path, _, label = spec.partition("=")
    with open(path, encoding="utf-8") as f:
        return label or os.path.basename(path), json.load(f)["records"]


def wer_by_lang(recs):
    agg = defaultdict(lambda: [0, 0])
    for r in recs:
        we, ww, _, _ = score(r["ref"], r["hyp"], r["lang"].split("_")[0], True)
        agg[r["lang"]][0] += we
        agg[r["lang"]][1] += ww
    out = {k: 100.0 * v[0] / v[1] for k, v in agg.items() if v[1]}
    out["ALL"] = 100.0 * sum(v[0] for v in agg.values()) / sum(v[1] for v in agg.values())
    return out


def median(xs):
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])) if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--engine", action="append", default=[])
    ap.add_argument("--speed", default=None,
                    help="interleaved raw JSON (list of {lang,<engine>:ms}); speed "
                         "ratios come from there instead of from the WER runs")
    ap.add_argument("--speed-key", action="append", default=[],
                    help="table-label=interleaved-key, repeatable")
    args = ap.parse_args()

    # Speed from a separate interleaved run. Ratios taken ACROSS PHASES of a
    # sequential sweep are not trustworthy on shared hosts -- measured drift
    # between phases was large enough to invert a known 1.15x ratio -- so the
    # published speed column comes from a run where every engine measures each
    # clip back to back. WER stays on the big sweeps: it is deterministic per
    # engine and immune to host noise.
    speed_rows = json.load(open(args.speed)) if args.speed else None
    # rsplit: table labels themselves contain '=' (e.g. "VIBEASR_RES_FUSE=0")
    skey = dict(s.rsplit("=", 1) for s in args.speed_key)

    blabel, base = load(args.baseline)
    bkey = {(r["lang"], r["wav"]): r for r in base}
    rows = [(blabel, base, True)] + [(l, r, False) for l, r in map(load, args.engine)]

    langs = sorted({r["lang"] for r in base})
    head = "| Engine | " + " | ".join(l.replace("_", "-") for l in langs) + \
           " | **WER all** | **speed × over baseline** |"
    sep = "|:--|" + "--:|" * (len(langs) + 2)
    print(head)
    print(sep)
    for label, recs, is_base in rows:
        w = wer_by_lang(recs)
        cells = " | ".join("%.1f" % w[l] if l in w else "-" for l in langs)
        if is_base:
            speed = "1.00× (baseline)"
        elif speed_rows is not None:
            k = skey.get(label) or skey.get(label.replace("**", ""))
            bk = skey.get(blabel) or skey.get(blabel.replace("**", ""))
            if not k or not bk:
                speed = "—"
            else:
                ratios = [r[bk] / r[k] for r in speed_rows if r.get(k) and r.get(bk)]
                speed = "**%.2f×** (n=%d)" % (median(ratios), len(ratios))
        else:
            pairs = [(bkey[(r["lang"], r["wav"])], r) for r in recs
                     if (r["lang"], r["wav"]) in bkey]
            ratios = [b["compute_ms"] / e["compute_ms"] for b, e in pairs]
            corpus = (sum(b["compute_ms"] for b, _ in pairs)
                      / sum(e["compute_ms"] for _, e in pairs))
            speed = "**%.2f×** (corpus %.2f×, n=%d)" % (median(ratios), corpus, len(pairs))
        print("| %s | %s | **%.2f** | %s |" % (label, cells, w["ALL"], speed))


if __name__ == "__main__":
    main()
