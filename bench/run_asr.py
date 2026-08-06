#!/usr/bin/env python3
"""Run asr_infer over a fetched FLEURS slice and report WER/CER and RTF.

    python3 bench/run_asr.py --langs eu19 -t 4 --tag baseline

Writes bench/results/<tag>.json (per-clip records) and prints a per-language table.
RTF here is compute-only -- VAE encode + prompt build + prefill + decode -- excluding
model load and audio decode, because model load is a one-off that a server pays once
and would otherwise dominate on short clips. asr_infer's own "RTF" line includes it;
the two are both reported so the difference is visible.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from languages import EU24, resolve  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TIMINGS = {
    "vae_acoustic":  re.compile(r"VAE acoustic encode:\s+([\d.]+) ms"),
    "vae_semantic":  re.compile(r"VAE semantic encode:\s+([\d.]+) ms"),
    "prompt_build":  re.compile(r"Prompt build:\s+([\d.]+) ms"),
    "prefill":       re.compile(r"LM prefill:\s+([\d.]+) ms"),
    "decode":        re.compile(r"LM decode:\s+([\d.]+) ms"),
    "vae_load":      re.compile(r"VAE model loading:\s+([\d.]+) ms"),
    "lm_load":       re.compile(r"LM model loading:\s+([\d.]+) ms"),
    "total":         re.compile(r"Total inference:\s+([\d.]+) ms"),
    "rtf_reported":  re.compile(r"RTF:\s+([\d.]+)"),
}

PUNCT = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)


def normalise(s):
    """Lowercase, strip punctuation, collapse whitespace -- the usual FLEURS recipe."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.translate(PUNCT)
    return " ".join(s.split())


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def run_clip(binary, vae, lm, wav, threads, extra_env, timeout):
    env = dict(os.environ)
    env.update(extra_env)
    cmd = [binary, "--vae-model", vae, "--lm-model", lm, "--audio", wav,
           "-t", str(threads), "--greedy"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    wall = time.time() - t0
    if p.returncode != 0:
        return None, {"error": p.stderr[-400:]}, wall

    stats = {}
    for key, rx in TIMINGS.items():
        m = rx.search(p.stderr)
        stats[key] = float(m.group(1)) if m else None

    # The transcript is everything asr_infer put on stdout, minus its summary lines.
    text = "\n".join(
        ln for ln in p.stdout.splitlines()
        if ln.strip() and not ln.lstrip().startswith(("Audio:", "VAE:", "["))
    ).strip()
    return text, stats, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="eu19")
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--vae", default="models/vibeasr/vibeasr-vae-encoder-i8_s.gguf")
    ap.add_argument("--lm", default="models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf")
    ap.add_argument("--bin", default="build/bin/asr_infer")
    ap.add_argument("--isa", default=None, help="force VIBEASR_ISA (avx2|avx512|vnni|amx)")
    ap.add_argument("-n", type=int, default=0, help="cap clips per language (0 = all)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--data", default=os.path.join(ROOT, "bench", "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results"))
    args = ap.parse_args()

    extra_env = {"VIBEASR_ISA": args.isa} if args.isa else {}
    os.makedirs(args.out, exist_ok=True)
    langs = resolve(args.langs)

    records = []
    print("%-6s %-12s %6s %6s %7s %7s   %s" %
          ("lang", "name", "WER%", "CER%", "RTF", "RTFall", "clips"))
    print("-" * 72)

    for lang in langs:
        d = os.path.join(args.data, lang)
        refs_path = os.path.join(d, "refs.jsonl")
        if not os.path.exists(refs_path):
            print("%-6s (no data -- run fetch_fleurs.py)" % lang)
            continue
        refs = [json.loads(l) for l in open(refs_path, encoding="utf-8")]
        if args.n:
            refs = refs[: args.n]

        werr = wtot = cerr = ctot = 0
        audio_s = compute_s = total_s = 0.0
        used = 0
        for r in refs:
            hyp, stats, wall = run_clip(args.bin, args.vae, args.lm,
                                        os.path.join(d, r["wav"]), args.threads,
                                        extra_env, args.timeout)
            if hyp is None:
                records.append({"lang": lang, "wav": r["wav"], **stats})
                continue
            ref_n, hyp_n = normalise(r["text"]), normalise(hyp)
            rw, hw = ref_n.split(), hyp_n.split()
            werr += edit_distance(rw, hw); wtot += len(rw)
            cerr += edit_distance(ref_n, hyp_n); ctot += len(ref_n)

            compute = sum(v or 0.0 for k, v in stats.items()
                          if k in ("vae_acoustic", "vae_semantic", "prompt_build", "prefill", "decode"))
            audio_s += r["duration"]
            compute_s += compute / 1000.0
            total_s += (stats.get("total") or 0.0) / 1000.0
            used += 1
            records.append({"lang": lang, "wav": r["wav"], "duration": r["duration"],
                            "ref": r["text"], "hyp": hyp, "compute_ms": compute, **stats})

        if used and wtot:
            print("%-6s %-12s %6.2f %6.2f %7.3f %7.3f   %d" %
                  (lang, EU24.get(lang, ("", lang))[1], 100.0 * werr / wtot,
                   100.0 * cerr / max(ctot, 1), compute_s / audio_s, total_s / audio_s, used))

    meta = {"tag": args.tag, "threads": args.threads, "isa": args.isa or "auto",
            "vae": args.vae, "lm": args.lm, "langs": langs}
    with open(os.path.join(args.out, args.tag + ".json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "records": records}, f, ensure_ascii=False, indent=1)
    print("\nwrote %s" % os.path.join(args.out, args.tag + ".json"))


if __name__ == "__main__":
    main()
