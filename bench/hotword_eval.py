#!/usr/bin/env python3
"""A/B the decoder-level hotword biasing on FLEURS-French.

    python3 bench/hotword_eval.py [-t 4] [--lang fr_fr] [-n 24]

Four conditions per clip:

  baseline   no hotwords
  oracle     capitalised terms from the clip's own reference (mechanism ceiling --
             deployments know their domain terms, not the transcript)
  poison     the *next* clip's oracle terms, at the strongest lambda tried. The
             hallucination control: a biasing scheme that helps on oracle but does
             not stay flat under poison is worse than nothing.

Writes bench/results/hotword_<lang>.json with every hypothesis, so re-scoring under
a different normalisation needs no re-run.
"""

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import score  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def oracle_terms(raw):
    toks = re.findall(r"\b[A-ZÀ-Ž][\w'’-]+\b", raw)
    first = raw.split()[0] if raw.split() else ""
    out = []
    for t in toks:
        if t != first and t not in out:
            out.append(t)
    return out[:8]


def run(binary, vae, lm, wav, threads, hotwords, lam):
    cmd = [binary, "--vae-model", vae, "--lm-model", lm, "--audio", wav,
           "-t", str(threads), "--greedy"]
    if hotwords:
        cmd += ["--hotwords", ",".join(hotwords), "--hotword-boost", str(lam)]
    p = subprocess.run(cmd, capture_output=True)
    out = p.stdout.decode("utf-8", "replace")
    return "\n".join(l for l in out.splitlines()
                     if l.strip() and not l.lstrip().startswith(("Audio:", "VAE:", "["))).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="fr_fr")
    ap.add_argument("-n", type=int, default=24)
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--lams", default="2.5,5")
    ap.add_argument("--vae", default="models/vibeasr/vibeasr-vae-encoder-i8_s.gguf")
    ap.add_argument("--lm", default="models/vibeasr/vibeasr-lm-i2_s-tied.gguf")
    ap.add_argument("--bin", default="build/bin/asr_infer")
    args = ap.parse_args()

    iso = args.lang.split("_")[0]
    d = os.path.join(ROOT, "bench", "data", args.lang)
    refs = [json.loads(l) for l in open(os.path.join(d, "refs.jsonl"), encoding="utf-8")][: args.n]
    lams = [float(x) for x in args.lams.split(",")]

    conds = [("baseline", None, 0.0)]
    for lam in lams:
        conds.append(("oracle l=%g" % lam, "own", lam))
    conds.append(("poison l=%g" % lams[-1], "next", lams[-1]))

    records = []
    print("%-14s %8s %8s   (%d clips, %s)" % ("condition", "WER%", "CER%", len(refs), args.lang))
    for name, mode, lam in conds:
        we = ww = ce = cc = 0
        for i, r in enumerate(refs):
            if mode == "own":
                hw = oracle_terms(r["raw_transcription"])
            elif mode == "next":
                hw = oracle_terms(refs[(i + 1) % len(refs)]["raw_transcription"])
            else:
                hw = None
            hyp = run(args.bin, args.vae, args.lm, os.path.join(d, r["wav"]),
                      args.threads, hw, lam)
            a, b, c, dd = score(r["text"], hyp, iso, True)
            we += a; ww += b; ce += c; cc += dd
            records.append({"cond": name, "wav": r["wav"], "hotwords": hw,
                            "ref": r["text"], "hyp": hyp})
        print("%-14s %8.2f %8.2f" % (name, 100.0 * we / ww, 100.0 * ce / cc), flush=True)

    out = os.path.join(ROOT, "bench", "results", "hotword_%s.json" % args.lang)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
