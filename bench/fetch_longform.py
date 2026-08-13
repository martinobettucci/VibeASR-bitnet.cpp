#!/usr/bin/env python3
"""Fetch a REAL long-form ASR dataset (whole recordings, not stitched clips).

    python3 bench/fetch_longform.py --dataset tedlium -n 6

Earlier revisions of this harness built "long-form" audio by concatenating short
utterances. That is not long-form speech: splicing independently recorded
sentences creates abrupt speaker/channel changes every few seconds, which sends
some engines into repetition loops and makes others skip whole spans. Every
engine degraded, which is the signature of a broken corpus, not a model result.

These are the sets whisper's own long-form evaluation uses -- single continuous
recordings, minutes long, with full reference transcripts:

  tedlium    distil-whisper/tedlium-long-form   TED talks, ~10-20 min, English
  earnings21 distil-whisper/earnings21          earnings calls, ~1 h, English
  meanwhile  distil-whisper/meanwhile           late-night monologues, English

All are English; that is a limitation of what exists publicly for long-form and
is called out where the results are reported. Audio is written as 16-bit WAV,
which is what the runtime's dr_wav loader reads, and each recording gets a JSON
sidecar in the same shape as the rest of this harness.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETS = {
    "tedlium": ("distil-whisper/tedlium-long-form", "default", "test"),
    "earnings21": ("distil-whisper/earnings21", "full", "test"),
    "meanwhile": ("distil-whisper/meanwhile", "default", "test"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tedlium", choices=sorted(SETS))
    ap.add_argument("-n", type=int, default=6, help="recordings to keep")
    ap.add_argument("--max-sec", type=float, default=0.0,
                    help="if > 0, truncate each recording to this many seconds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo, config, split = SETS[args.dataset]
    out = args.out or os.path.join(ROOT, "bench", "data", "lf_" + args.dataset)
    os.makedirs(out, exist_ok=True)

    url = ("https://huggingface.co/api/datasets/%s/parquet/%s/%s/0.parquet"
           % (repo, config, split))
    rf = RangeFile(url, os.environ.get("HF_TOKEN"))
    pf = pq.ParquetFile(rf)
    cols = pf.schema_arrow.names
    text_col = next(c for c in ("text", "transcription", "sentence") if c in cols)

    index = []
    for batch in pf.iter_batches(batch_size=1, columns=["audio", text_col]):
        if len(index) >= args.n:
            break
        d = batch.to_pydict()
        for i in range(batch.num_rows):
            if len(index) >= args.n:
                break
            data, sr = sf.read(io.BytesIO(d["audio"][i]["bytes"]), dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]
            if args.max_sec > 0:
                data = data[: int(args.max_sec * sr)]
            pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
            name = "%s_%02d" % (args.dataset, len(index))
            sf.write(os.path.join(out, name + ".wav"), pcm, sr, subtype="PCM_16")
            index.append({"name": name, "lang": "en_us", "regime": "long",
                          "duration": round(len(pcm) / float(sr), 2),
                          "text": d[text_col][i], "n_clips": 1})
            print("%-14s %7.1f s  %5d ref words"
                  % (name, len(pcm) / float(sr), len(d[text_col][i].split())), flush=True)
    rf.close()

    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print("\nwrote %d recordings, %.1f min total -> %s"
          % (len(index), sum(x["duration"] for x in index) / 60.0, out))


if __name__ == "__main__":
    main()
