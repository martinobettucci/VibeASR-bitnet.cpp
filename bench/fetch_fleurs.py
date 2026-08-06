#!/usr/bin/env python3
"""Materialise a fixed FLEURS test slice per language for the EU benchmark.

Writes bench/data/<lang>/NNNN.wav plus a refs.jsonl carrying the reference text and
duration, so the WER/RTF sweep is reproducible and does not re-hit the network.

    python3 bench/fetch_fleurs.py --langs eu19 -n 24

Needs HF_TOKEN in the environment (FLEURS itself is public, but the token lifts the
anonymous rate limit).
"""

import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq  # noqa: E402

from hf_range import RangeFile, fleurs_url  # noqa: E402
from languages import EU24, resolve  # noqa: E402

# Keep clips long enough that per-call fixed costs do not dominate RTF, and short
# enough that a sweep finishes: FLEURS is 16 kHz mono.
MIN_SEC = 5.0
MAX_SEC = 25.0
SR = 16000


def fetch_lang(lang, n, outdir, token):
    dest = os.path.join(outdir, lang)
    refs_path = os.path.join(dest, "refs.jsonl")
    if os.path.exists(refs_path) and len(open(refs_path).readlines()) >= n:
        print("  %-8s cached" % lang)
        return
    os.makedirs(dest, exist_ok=True)

    t0 = time.time()
    rf = RangeFile(fleurs_url(lang), token)
    pf = pq.ParquetFile(rf)
    cols = ["id", "num_samples", "audio", "transcription", "raw_transcription"]

    rows = []
    for batch in pf.iter_batches(batch_size=64, columns=cols):
        d = batch.to_pydict()
        for i in range(batch.num_rows):
            dur = d["num_samples"][i] / float(SR)
            if not (MIN_SEC <= dur <= MAX_SEC):
                continue
            rows.append({
                "id": d["id"][i],
                "dur": dur,
                "wav": d["audio"][i]["bytes"],
                "text": d["transcription"][i],
                "raw": d["raw_transcription"][i],
            })
        if len(rows) >= n:
            break
    rf.close()

    # Deterministic pick: lowest FLEURS ids among the duration-filtered rows.
    rows.sort(key=lambda r: r["id"])
    rows = rows[:n]

    with open(refs_path, "w", encoding="utf-8") as fh:
        for k, r in enumerate(rows):
            wav = os.path.join(dest, "%04d.wav" % k)
            with open(wav, "wb") as w:
                w.write(r["wav"])
            fh.write(json.dumps({
                "wav": os.path.basename(wav),
                "id": r["id"],
                "duration": round(r["dur"], 3),
                "text": r["text"],
                "raw_transcription": r["raw"],
            }, ensure_ascii=False) + "\n")

    total = sum(r["dur"] for r in rows)
    print("  %-8s %2d clips, %6.1fs audio, fetched in %5.1fs" % (lang, len(rows), total, time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="eu19", help="eu19 | eu24 | comma-separated FLEURS configs")
    ap.add_argument("-n", type=int, default=24, help="clips per language")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    langs = resolve(args.langs)
    print("Fetching %d clips x %d languages -> %s" % (args.n, len(langs), args.out))
    for lang in langs:
        name = EU24.get(lang, ("?", lang))[1]
        try:
            fetch_lang(lang, args.n, args.out, token)
        except Exception as e:  # keep going; a missing language should not kill the sweep
            print("  %-8s FAILED (%s): %s" % (lang, name, e))


if __name__ == "__main__":
    main()
