#!/usr/bin/env python3
"""Speed-only benchmark: an RTF *distribution* over many clips, not one clip's number.

    python3 bench/speed_test.py -n 100 -t 4

Every speed figure quoted so far came from repeated runs of a single 8.38 s clip.
That controls variance beautifully and tells you nothing about how RTF moves with
clip length, language, or how many tokens the decoder chooses to emit -- which is
the thing that actually varies in deployment. This runs N distinct clips once each
and reports the spread.

Clips are drawn round-robin across the model's supported languages (so the mix is
not one language's register) and the draw is deterministic: same -n, same clips.

Reported per clip:
  compute = vae_acoustic + vae_semantic + prompt_build + prefill + decode
  RTF     = compute / audio_duration          (model load excluded -- paid once)
  decode rate = generated tokens / decode seconds

Model load is measured once and reported separately. Nothing else should be running
on the host: check `uptime` first, the numbers are meaningless under contention.

Warm-up matters more than it looks. The harness spawns one process per clip, so every
clip re-mmaps 1.2 GB of weights; on a cold page cache the faults land *inside* the
timed VAE section (measured on this VM: 9110 ms acoustic cold vs 1352 ms warm, same
clip, same binary). --warmup clips are run and discarded before the measured set, and
the run also reports a trimmed RTF that drops the worst 10% so a mid-run cache
eviction is visible as the gap between trimmed and untrimmed rather than silently
inflating the mean.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_asr import TIMINGS, run_clip  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The languages the weights actually support, plus the MLS French anchor for a
# second register. Deliberately not the EU19 sweep.
SUPPORTED = ["en_us", "fr_fr", "it_it", "pt_br", "es_419", "de_de", "fr_mls"]

TOKENS_RX = re.compile(r"Generated tokens:\s+(\d+)")


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def draw(data_dir, langs, n):
    """Round-robin across languages, clips in file order -- deterministic."""
    pools = []
    for lang in langs:
        p = os.path.join(data_dir, lang, "refs.jsonl")
        if not os.path.exists(p):
            continue
        refs = [json.loads(l) for l in open(p, encoding="utf-8")]
        pools.append((lang, refs))
    out, i = [], 0
    while len(out) < n and any(i < len(r) for _, r in pools):
        for lang, refs in pools:
            if i < len(refs) and len(out) < n:
                out.append((lang, refs[i]))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100, help="number of clips")
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--langs", default=",".join(SUPPORTED))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--vae", default="models/vibeasr/vibeasr-vae-encoder-i8_s.gguf")
    ap.add_argument("--lm", default="models/vibeasr/vibeasr-lm-i2_s-tied.gguf")
    ap.add_argument("--bin", default="build/bin/asr_infer")
    ap.add_argument("--warmup", type=int, default=2,
                    help="clips run and discarded first, to fault the weights in")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--data", default=os.path.join(ROOT, "bench", "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results"))
    args = ap.parse_args()

    tag = args.tag or "speed_t%d_n%d" % (args.threads, args.n)
    clips = draw(args.data, args.langs.split(","), args.n)
    if not clips:
        sys.exit("no clips found under %s" % args.data)

    print("%d clips, %d threads, %s" % (len(clips), args.threads, os.path.basename(args.lm)))

    for w in range(args.warmup):
        lang, r = clips[w % len(clips)]
        _, st, _ = run_clip(args.bin, args.vae, args.lm,
                            os.path.join(args.data, lang, r["wav"]),
                            args.threads, {}, args.timeout)
        print("warmup %d: acoustic %.0f ms (discarded)" % (w, st.get("vae_acoustic") or 0.0),
              flush=True)

    print("%-4s %-7s %7s %8s %8s %7s %7s" %
          ("#", "lang", "audio", "compute", "RTF", "tok", "tok/s"), flush=True)

    recs = []
    t_start = time.time()
    for k, (lang, r) in enumerate(clips):
        wav = os.path.join(args.data, lang, r["wav"])
        hyp, stats, wall = run_clip(args.bin, args.vae, args.lm, wav,
                                    args.threads, {}, args.timeout)
        if hyp is None:
            print("%-4d %-7s  FAILED: %s" % (k, lang, stats.get("error", "?")[:60]), flush=True)
            continue
        compute = sum(stats.get(x) or 0.0 for x in
                      ("vae_acoustic", "vae_semantic", "prompt_build", "prefill", "decode"))
        dur = r["duration"]
        rtf = compute / 1000.0 / dur
        # asr_infer does not print a token count; approximate from the decode budget
        # only if it ever does. Until then tok/s is derived from the hypothesis length
        # in whitespace words, which is a stable proxy for relative decode cost.
        words = len(hyp.split())
        dec_s = (stats.get("decode") or 0.0) / 1000.0
        rec = {"lang": lang, "wav": r["wav"], "duration": dur, "compute_ms": compute,
               "rtf": rtf, "words": words, "wall_s": wall, **stats}
        recs.append(rec)
        print("%-4d %-7s %6.2fs %7.0fms %8.3f %7d %7.1f" %
              (k, lang, dur, compute, rtf, words, words / dec_s if dec_s else 0.0),
              flush=True)
    elapsed = time.time() - t_start

    rtfs = [r["rtf"] for r in recs]
    audio = sum(r["duration"] for r in recs)
    comp = sum(r["compute_ms"] for r in recs) / 1000.0
    loads = [(r.get("vae_load") or 0) + (r.get("lm_load") or 0) for r in recs]

    summary = {
        "tag": tag, "threads": args.threads, "clips": len(recs), "lm": args.lm,
        "audio_sec": round(audio, 2), "compute_sec": round(comp, 2),
        "rtf_corpus": comp / audio if audio else float("nan"),
        "rtf_median": pct(rtfs, 50), "rtf_mean": sum(rtfs) / len(rtfs) if rtfs else float("nan"),
        "rtf_p10": pct(rtfs, 10), "rtf_p90": pct(rtfs, 90),
        "rtf_min": min(rtfs) if rtfs else None, "rtf_max": max(rtfs) if rtfs else None,
        # trimmed: worst 10% of clips dropped. A big gap to rtf_mean means the run hit
        # host noise (cache eviction, a noisy neighbour), not that the model is slow.
        "rtf_trimmed_mean": (sum(sorted(rtfs)[: max(1, int(len(rtfs) * 0.9))])
                             / max(1, int(len(rtfs) * 0.9))) if rtfs else float("nan"),
        "model_load_ms_median": pct(loads, 50),
        "wall_sec": round(elapsed, 1),
    }
    stages = ("vae_acoustic", "vae_semantic", "prompt_build", "prefill", "decode")
    summary["stage_share"] = {s: round(100.0 * sum(r.get(s) or 0 for r in recs)
                                       / (comp * 1000.0), 2) for s in stages}

    print("\n%d clips, %.1f s audio, %.1f s compute" % (len(recs), audio, comp))
    print("  corpus RTF (total compute / total audio) : %.3f" % summary["rtf_corpus"])
    print("  per-clip RTF  median %.3f   mean %.3f   trimmed-mean %.3f" %
          (summary["rtf_median"], summary["rtf_mean"], summary["rtf_trimmed_mean"]))
    print("                p10 %.3f   p90 %.3f   min %.3f   max %.3f" %
          (summary["rtf_p10"], summary["rtf_p90"], summary["rtf_min"], summary["rtf_max"]))
    print("  model load (once, excluded above)        : %.0f ms" % summary["model_load_ms_median"])
    print("  stage share of compute: " +
          "  ".join("%s %.1f%%" % (s.replace("vae_", "vae-"), summary["stage_share"][s])
                    for s in stages))
    print("  wall clock for the whole run             : %.0f s" % elapsed)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, tag + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": recs}, f, indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
