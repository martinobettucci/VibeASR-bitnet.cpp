#!/usr/bin/env python3
"""Run whisper.cpp over the same clip draw as speed_test.py, same JSON schema.

    python3 bench/whisper_bench.py --bin ../whisper.cpp/build/bin/whisper-cli \
        --model ../whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin -n 100 -t 4 \
        --tag whisper_turbo_n100

Whisper is the external yardstick for the relative benchmark: absolute RTF numbers
are a property of whatever VM the run landed on, but ratios between engines measured
back-to-back on one host transfer. Each clip gets the true language code (--l): our
model receives no language hint, so this is the generous setting for whisper -- if
the comparison holds under it, it holds.

compute_ms = whisper's own timing total minus model load (load is paid once per
process here, as in speed_test). Note whisper's encoder always processes a fixed
30 s mel window, so its compute is roughly flat in clip length where ours is
proportional -- visible in the per-clip spread.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RX_TOTAL = re.compile(r"total time =\s+([\d.]+) ms")
RX_LOAD  = re.compile(r"load time =\s+([\d.]+) ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--langs", default=",".join(SUPPORTED))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--data", default=os.path.join(ROOT, "bench", "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results"))
    args = ap.parse_args()

    clips = draw(args.data, args.langs.split(","), args.n)
    print("%d clips, %d threads, %s" % (len(clips), args.threads, os.path.basename(args.model)))

    recs = []
    t_start = time.time()
    for k, (lang, r) in enumerate(clips):
        wav = os.path.join(args.data, lang, r["wav"])
        iso = lang.split("_")[0]
        cmd = [args.bin, "-m", args.model, "-f", wav, "-t", str(args.threads),
               "-l", iso, "--no-timestamps"]
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print("%-4d %-7s TIMEOUT" % (k, lang), flush=True)
            continue
        out = p.stdout.decode("utf-8", "replace")
        err = p.stderr.decode("utf-8", "replace")
        mt = RX_TOTAL.search(err)
        ml = RX_LOAD.search(err)
        if not mt or p.returncode != 0:
            print("%-4d %-7s FAILED rc=%d" % (k, lang, p.returncode), flush=True)
            continue
        compute = float(mt.group(1)) - (float(ml.group(1)) if ml else 0.0)
        hyp = " ".join(l.strip() for l in out.splitlines() if l.strip())
        dur = r["duration"]
        rtf = compute / 1000.0 / dur
        recs.append({"lang": lang, "wav": r["wav"], "duration": dur,
                     "compute_ms": compute, "rtf": rtf, "hyp": hyp, "ref": r["text"],
                     "load_ms": float(ml.group(1)) if ml else None})
        print("%-4d %-7s %6.2fs %8.0fms %7.3f  %s" %
              (k, lang, dur, compute, rtf, hyp[:40]), flush=True)

    rtfs = [x["rtf"] for x in recs]
    audio = sum(x["duration"] for x in recs)
    comp = sum(x["compute_ms"] for x in recs) / 1000.0
    summary = {"tag": args.tag, "threads": args.threads, "clips": len(recs),
               "model": args.model, "audio_sec": round(audio, 2),
               "compute_sec": round(comp, 2),
               "rtf_corpus": comp / audio if audio else None,
               "rtf_median": pct(rtfs, 50), "rtf_p10": pct(rtfs, 10),
               "rtf_p90": pct(rtfs, 90),
               "wall_sec": round(time.time() - t_start, 1)}
    print("\ncorpus RTF %.3f  median %.3f" % (summary["rtf_corpus"], summary["rtf_median"]))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.tag + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": recs}, f, ensure_ascii=False, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
