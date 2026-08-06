# VibeASR CPU benchmark harness

Everything here is a runnable script. `run_all.sh` chains them in order; each step can
also be run on its own, and each is safe to interrupt and resume.

```bash
HF_TOKEN=hf_...  ./bench/run_all.sh
```

A full pass is several hours on 4 cores. For a quick check:

```bash
CLIPS=4 LANGS=en_us,fr_fr,de_de ./bench/run_all.sh
```

`SKIP` takes a space-separated list of step names to leave out
(`build models kernels requant fetch sweep scaling`), and `THREADS` overrides the
thread count (defaults to `nproc`).

## The steps

| Script | What it does |
|:--|:--|
| `kernel_bench` (C++, built by CMake) | Checks every SIMD kernel path against an exact int32 scalar reference, then measures single-thread throughput per shape |
| `fetch_fleurs.py` | Materialises a fixed FLEURS test slice per language into `bench/data/<lang>/` |
| `requant_lm_head` (C++, built by CMake) | Re-quantises or drops a single tensor of an already-quantised GGUF |
| `run_asr.py` | Runs `asr_infer` over a slice, computes WER/CER and RTF, writes `bench/results/<tag>/<lang>.json` |
| `thread_scaling.py` | RTF against thread count, broken down by pipeline stage |
| `summarize.py` | Renders the result JSON as the Markdown tables in the top-level README |

`hf_range.py` and `languages.py` are support modules, not entry points.

## Reproducing individual measurements

```bash
# Kernel correctness and throughput, both ISA paths
./build/bin/kernel_bench
VIBEASR_ISA=avx2 ./build/bin/kernel_bench

# Where run time actually goes, per kernel
VIBEASR_KERNEL_STATS=1 ./build/bin/asr_infer \
    --vae-model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
    --lm-model  models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf \
    --audio bench/data/en_us/0000.wav -t 4 --greedy

# WER + RTF for one language
python3 bench/fetch_fleurs.py --langs en_us -n 24
python3 bench/run_asr.py --langs en_us -n 24 -t 4 --tag myrun

# Compare two runs
python3 bench/summarize.py eu19_vnni eu19_head_q6k --diff
```

## Conventions worth knowing

**RTF is compute-only** — VAE encode + prompt build + prefill + decode. Model load and
audio decode are excluded, because load is a one-off a server pays once and would
otherwise dominate on short clips. `asr_infer`'s own `RTF:` line *does* include load;
both are reported (`RTF` and `RTF incl. load`) so the gap stays visible.

**WER is corpus-level** — total word errors over total reference words, not a mean of
per-clip rates. Text is NFKC-normalised, lowercased, stripped of Unicode punctuation,
and whitespace-collapsed before scoring, against FLEURS' own `transcription` field.

**Clip selection is deterministic** — lowest FLEURS ids among clips between 5 s and
25 s, so a re-run scores the same audio. Short clips are excluded because per-call
fixed costs would swamp RTF; very long ones only to keep a sweep finishing.

**Sample size** — the default 24 clips per language is roughly 300-500 reference words.
That resolves large differences between configurations but not one-point WER changes.
Raise `CLIPS` when a comparison looks close.

**ISA selection** — kernels pick the widest supported path at run time. `VIBEASR_ISA`
(`avx2`, `avx512`, `vnni`, `amx`) clamps that downward for A/B runs; it can never
select something the CPU cannot execute. `VIBEASR_ISA_VERBOSE=1` prints the choice.
