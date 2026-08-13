#!/usr/bin/env python3
"""Build one short (<30 s) and one long (>30 s) item per language.

    python3 bench/make_shortlong.py

The clip suites in this repo top out around 25 s, which never crosses whisper's
fixed 30 s encoder window and never gives this model the context its 3200x
compression is designed to exploit. The two regimes behave completely differently
(short: whisper-small wins; long: this model wins by 3x), so the comparison needs
one item on each side of the boundary.

Short item = the longest single clip under 30 s in that language's set.
Long item  = consecutive clips concatenated to just over 35 s.

The long items are stitched, and for FLEURS that means a speaker change at each
join (FLEURS sentences are independently recorded). MLS French is the one set
where consecutive utterances come from the same speaker and book, so its long
item is genuinely continuous -- treat fr_mls as the clean long-form case and the
FLEURS long items as a harder, speaker-switching variant.
"""

import json
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_pcm16(path):
    """Read any WAV subtype as int16 PCM.

    fetch_fleurs writes 32-bit FLOAT wavs (samples in [-1, 1]); fetch_mls writes
    PCM_16. Reading a float file with dtype='int16' truncates every sample to 0
    or +-1 -- silence that transcribes as "you" and scores 100 WER on every
    engine at once, which is what a broken harness looks like. Read as float and
    scale explicitly instead.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    return (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16), sr
LANGS = ["en_us", "fr_fr", "it_it", "pt_br", "es_419", "de_de", "fr_mls"]
LONG_TARGET = 35.0


def main():
    data = os.path.join(ROOT, "bench", "data")
    out = os.path.join(data, "shortlong")
    os.makedirs(out, exist_ok=True)
    index = []

    for lang in LANGS:
        d = os.path.join(data, lang)
        refs = [json.loads(l) for l in open(os.path.join(d, "refs.jsonl"), encoding="utf-8")]

        # short: longest clip still under 30 s
        cand = [r for r in refs if r["duration"] < 30.0]
        short = max(cand, key=lambda r: r["duration"])
        pcm, sr = read_pcm16(os.path.join(d, short["wav"]))
        name = "%s_short" % lang
        sf.write(os.path.join(out, name + ".wav"), pcm, sr, subtype="PCM_16")
        index.append({"name": name, "lang": lang, "regime": "short",
                      "duration": round(len(pcm) / sr, 2), "text": short["text"],
                      "n_clips": 1})

        # long: consecutive clips (file order = id order) past the target
        chunks, texts, dur = [], [], 0.0
        for r in refs:
            if dur >= LONG_TARGET:
                break
            p, s2 = read_pcm16(os.path.join(d, r["wav"]))
            assert s2 == sr
            chunks.append(p)
            texts.append(r["text"])
            dur += len(p) / sr
        if dur < 30.0:
            print("skip %s: only %.1f s available" % (lang, dur))
            continue
        name = "%s_long" % lang
        sf.write(os.path.join(out, name + ".wav"), np.concatenate(chunks), sr, subtype="PCM_16")
        index.append({"name": name, "lang": lang, "regime": "long",
                      "duration": round(dur, 2), "text": " ".join(texts),
                      "n_clips": len(chunks)})

    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    for it in index:
        print("%-14s %-6s %6.1fs  %2d clip(s)  %4d ref words"
              % (it["name"], it["regime"], it["duration"], it["n_clips"],
                 len(it["text"].split())))
    print("\nwrote", os.path.join(out, "index.json"))


if __name__ == "__main__":
    main()
