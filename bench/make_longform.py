#!/usr/bin/env python3
"""Stitch MLS utterances into multi-speaker long-form recordings.

    python3 bench/fetch_mls.py --config french -n 200 --out /tmp/mlsbig
    python3 bench/make_longform.py --src /tmp/mlsbig/fr_mls --minutes 8 --speakers 2

Writes bench/data/longform/<name>.wav plus <name>.json holding the reference
transcript, the speaker turn boundaries and their timings.

Why this corpus exists: every clip-level benchmark in this repo runs 5-25 s
utterances, which is whisper's ideal case (its encoder always processes a fixed
30 s window, so short clips waste most of it -- and long ones get chopped into
many). It is also this architecture's worst case: VibeVoice-ASR compresses audio
3200x into an LLM context specifically so that a long recording is ONE encode and
ONE prefill, and it emits speaker-labelled segments from the same decoding pass.
None of that is exercised below 30 s. This builds the recordings that do.

Turns alternate between speakers, each turn being 1-3 consecutive utterances from
one MLS book (so the prosody within a turn is continuous), with a short silence
between turns as a real conversation would have.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GAP_SEC = 0.35        # inter-turn silence
UTTS_PER_TURN = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a fetch_mls.py output dir")
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--speakers", type=int, default=2)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "data", "longform"))
    args = ap.parse_args()

    refs = [json.loads(l) for l in
            open(os.path.join(args.src, "refs.jsonl"), encoding="utf-8")]
    by_spk = defaultdict(list)
    for r in refs:
        by_spk[r["id"].split("_")[0]].append(r)
    # deterministic: most-material speakers first, utterances in file order
    spks = sorted(by_spk, key=lambda s: (-len(by_spk[s]), s))[: args.speakers]
    if len(spks) < args.speakers:
        sys.exit("only %d speakers available in %s" % (len(by_spk), args.src))
    for s in spks:
        by_spk[s].sort(key=lambda r: r["id"])

    target = args.minutes * 60.0
    sr = None
    audio, turns, cursor, idx = [], [], 0.0, {s: 0 for s in spks}
    turn_no = 0
    while cursor < target:
        spk = spks[turn_no % len(spks)]
        take = by_spk[spk][idx[spk]: idx[spk] + UTTS_PER_TURN]
        if not take:
            break
        idx[spk] += len(take)
        turn_start = cursor
        texts = []
        for r in take:
            data, this_sr = sf.read(os.path.join(args.src, r["wav"]), dtype="int16")
            sr = sr or this_sr
            assert this_sr == sr
            audio.append(data)
            cursor += len(data) / float(sr)
            texts.append(r["text"])
        turns.append({"speaker": spks.index(spk), "speaker_id": spk,
                      "start": round(turn_start, 3), "end": round(cursor, 3),
                      "text": " ".join(texts)})
        audio.append(np.zeros(int(GAP_SEC * sr), dtype=np.int16))
        cursor += GAP_SEC
        turn_no += 1

    name = args.name or "conv_%dspk_%dmin" % (len(spks), int(round(cursor / 60.0)))
    os.makedirs(args.out, exist_ok=True)
    wav = os.path.join(args.out, name + ".wav")
    sf.write(wav, np.concatenate(audio), sr, subtype="PCM_16")
    meta = {"wav": os.path.basename(wav), "duration": round(cursor, 2),
            "n_speakers": len(spks), "n_turns": len(turns),
            "text": " ".join(t["text"] for t in turns), "turns": turns}
    with open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("%s: %.1f min, %d turns, %d speakers"
          % (wav, cursor / 60.0, len(turns), len(spks)))


if __name__ == "__main__":
    main()
