#!/usr/bin/env python3
"""Materialise an MLS (Multilingual LibriSpeech) test slice as a benchmark language.

    python3 bench/fetch_mls.py --config french -n 24

Writes bench/data/<iso>_mls/ in the same refs.jsonl format as fetch_fleurs.py, so
run_asr.py and summarize.py work unchanged. The directory is named <iso>_mls (e.g.
fr_mls) so the scorer resolves the ISO code -- and with it the number speller -- from
the prefix.

Why this corpus exists in the harness: the upstream tech report scores French on
"MLC", an unpublished internal corpus. MLS is the closest public sibling in register
-- read audiobook speech -- and serves as the anchor for French work that FLEURS'
encyclopedic register cannot provide. Same-name comparability with the paper is NOT
claimed; same-pipeline comparability across our own configurations is the point.

Audio arrives as Ogg Opus at 16 kHz and is transcoded to 16-bit WAV, which is what
the runtime's dr_wav loader reads.
"""

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402

from hf_range import RangeFile  # noqa: E402

ISO = {"french": "fr", "german": "de", "italian": "it", "portuguese": "pt",
       "spanish": "es", "dutch": "nl", "polish": "pl"}

MIN_SEC, MAX_SEC = 5.0, 25.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="french")
    ap.add_argument("-n", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()

    iso = ISO[args.config]
    dest = os.path.join(args.out, "%s_mls" % iso)
    refs_path = os.path.join(dest, "refs.jsonl")
    if os.path.exists(refs_path) and len(open(refs_path).readlines()) >= args.n:
        print("cached: %s" % dest)
        return
    os.makedirs(dest, exist_ok=True)

    url = ("https://huggingface.co/api/datasets/facebook/multilingual_librispeech"
           "/parquet/%s/test/0.parquet" % args.config)
    rf = RangeFile(url, os.environ.get("HF_TOKEN"))
    pf = pq.ParquetFile(rf)

    rows = []
    for batch in pf.iter_batches(batch_size=16, columns=["id", "audio", "transcript", "audio_duration"]):
        d = batch.to_pydict()
        for i in range(batch.num_rows):
            dur = d["audio_duration"][i]
            if not (MIN_SEC <= dur <= MAX_SEC):
                continue
            rows.append({"id": d["id"][i], "dur": dur,
                         "opus": d["audio"][i]["bytes"], "text": d["transcript"][i]})
        if len(rows) >= args.n:
            break
    rf.close()

    rows.sort(key=lambda r: r["id"])   # deterministic pick, as in fetch_fleurs
    rows = rows[: args.n]

    with open(refs_path, "w", encoding="utf-8") as fh:
        for k, r in enumerate(rows):
            data, sr = sf.read(io.BytesIO(r["opus"]))
            pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
            wav = os.path.join(dest, "%04d.wav" % k)
            sf.write(wav, pcm, sr, subtype="PCM_16")
            fh.write(json.dumps({"wav": os.path.basename(wav), "id": r["id"],
                                 "duration": round(r["dur"], 3), "text": r["text"],
                                 "raw_transcription": r["text"]}, ensure_ascii=False) + "\n")

    print("wrote %d clips, %.1f s audio -> %s"
          % (len(rows), sum(r["dur"] for r in rows), dest))


if __name__ == "__main__":
    main()
