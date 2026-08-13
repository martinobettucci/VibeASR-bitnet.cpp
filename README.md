<h1 align="center">VibeASR.cpp — CPU-optimised fork</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://huggingface.co/P2Enjoy/VibeVoice-ASR-BitNet-slim"><img src="https://img.shields.io/badge/🤗-Slim_weights-orange.svg" alt="Slim weights"></a>
  <a href="https://github.com/microsoft/VibeASR.cpp"><img src="https://img.shields.io/badge/upstream-microsoft%2FVibeASR.cpp-lightgrey.svg" alt="Upstream"></a>
</p>

---

A fork of [microsoft/VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp) that makes
VibeVoice-ASR-BitNet fast on a plain x86 CPU. **This README is about the runtime.**
Model accuracy, language coverage and the weights themselves are documented on the
[model card](https://huggingface.co/P2Enjoy/VibeVoice-ASR-BitNet-slim) and not
repeated here.

<p align="center">
  <img src="media/fork_overview.png" width="92%" alt="Fork architecture: dual INT8 VAE encoders into a ternary Qwen2.5-1.5B with tied Q6_K embedding; the 467 MB F16 output head is removed"/>
</p>

**Headline: 3.2–3.6× the upstream runtime at unchanged accuracy**, measured as
paired per-clip ratios against upstream on identical weights (absolute RTF is a
property of the host — this project watched the same binary drift 18% between
sessions and 30% within one). Plus:

- **Deterministic** — identical transcripts at any thread count. Upstream output
  changed with `-t`; every partition-dependent rounding path is gone.
- **Portable** — AVX2 baseline with runtime dispatch up to AVX-512/VNNI. No
  `-march=native` time bombs.
- **A numerical fix** — the upstream AVX2 int8 kernels overflow their int16
  accumulators on full-range activations; the VNNI kernels accumulate in int32.
- **A long-input fix** — inputs beyond ~25 s used to segfault. See *Arena sizing*.
- **Decoder-level hotword biasing** — `--hotwords`, built and verified here.
- **A reproduction harness** — every number below regenerates from `bench/`.

---

## Quick Start

```bash
git clone --recursive -b claude/asr-cpu-optimization-cztnh9 \
    https://github.com/martinobettucci/VibeASR-bitnet.cpp.git
cd VibeASR-bitnet.cpp

pip install -r requirements.txt
python setup_env.py --skip-download     # builds portable binaries, applies patches/
./scripts/download_models.sh            # slim weights from P2Enjoy (1.23 GB)

./build/bin/asr_infer \
    --vae-model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
    --lm-model  models/vibeasr/vibeasr-lm-i2_s-tied.gguf \
    --audio input.wav -t 4 --greedy
```

Gradio demo: `python demo/gradio_asr_demo.py --port 7860 --vae-model ... --lm-model ...`

**Operational limits** (properties of the weights and of graph memory, not of the
optimisations): a single pass is usable to **~80 s of audio** — past that the
decoder ends early and drops the tail — and the encoder arena costs **~110 MB per
second**, capping a pass near 135 s on a 16 GB host. Chunk longer recordings;
`bench/lf_eval.py` shows a 60 s chunker. `--prompt-format json` is rejected on the
1.5B weights (it targets the 7B checkpoint).

---

## Performance

Reference hardware: **4-vCPU Intel Xeon (Cascade Lake class, AVX-512F/BW/DQ/VL +
VNNI) cloud VM**, 4 threads, compute only (model load excluded).

### Engine speed, versus upstream on identical weights

| Workload | this fork | whisper-small q5_1 | whisper-turbo q5_0 |
|:--|--:|--:|--:|
| 14 short clips, 7 languages (`bench/shortlong_bench.py`) | **3.15×** | 3.58× | 1.08× |
| 100-clip suite, 7 languages (`bench/interleaved_speed.py`) | **2.35×** | 3.06× | 0.75× |
| Real 7.7-min TED talk (`bench/lf_eval.py`) | **3.60×** | 4.15× | 1.48× |

Upstream = 1.00× in every row. The three suites disagree by ~1.2× on absolute
ratio and agree completely on ordering, which is the level of precision a shared
cloud host supports.

### Accuracy is unchanged by the optimisations

Short clips, 7 languages: **8.73 WER vs upstream's 8.41** — and the only
numerics-changing optimisation (residual fusion) accounts for it; `VIBEASR_RES_FUSE=0`
is bit-exact with upstream. Real long-form (7.7-min TED talk): **5.67 vs 5.61**.
Per-language model accuracy belongs on the model card.

### Measurement methodology

Speed is published as **paired per-clip ratios from interleaved runs** — every
engine transcribes a given clip back to back before moving to the next — because
sequential phase-per-engine timing on a shared host drifted enough between phases
to invert a ratio known to be 1.15×. WER comes from full suites, being
deterministic per engine and immune to host noise. `bench/README.md` documents
each script.

---

## What was optimised

A per-node graph profiler (`VIBEASR_NODE_PROFILE=1`, sections named per encoder
stage) drove every step. In the order they were found:

1. **Per-call overhead, not arithmetic.** GEMM row blocks were 4, issuing 194M
   `vec_dot` calls per clip at ~2% of int8 peak. Tuned to 32 (`bench/row_block_sweep.sh`).
2. **The AVX2 kernels were numerically wrong.** int16 accumulation of `vpmaddubsw`
   wraps on full-range int8; the AVX-512 VNNI kernels accumulate in int32.
   `kernel_bench --check` validates 1661 points against exact scalar references.
3. **A register-tiled INT8 GEMM** (4×4 int32 accumulator tile, +128 weight-bias
   trick instead of a per-pair sign fold): ~2× kernel throughput.
4. **Scalar epilogues.** After every fused matmul the runtime made two scalar
   passes (dequant+absmax, then `roundf` per element) over multi-megabyte
   activations — vectorised, bit-exact by construction.
5. **Data movement was 41% of graph time** (CONT 23.5% + IM2COL 17.3%). The
   ConvNeXt mixer's `permute→cont→im2col→matmul→cont` chain became one
   layout-native depthwise-conv node: in `[dim, frames]` a causal k-tap conv is k
   fused multiply-adds over contiguous vectors. Data movement fell to 8%.
6. **The GEMM tile was reduction-bound at small K.** Early stages run tall-thin
   matmuls (K = 32–512 bytes, tens of thousands of rows) where 16 horizontal
   reductions per 4×4 tile cost more than the `vpdpbusd` work, and K=32 missed the
   tile gate entirely. The packed-B kernel repacks weights once at model load into
   the layout `vpdpbusd` consumes, accumulates 16 output channels vertically — no
   horizontal reductions, K%4 granularity — and fuses the int32→float scale+bias+absmax
   epilogue into the accumulator store, deleting the int32 buffer, its memset and
   the separate dequant pass. Acoustic encoder **1.26×**, bit-exact.
7. **im2col was never necessary.** In `[C, T]` a strided-conv window of KW frames ×
   IC channels is one contiguous slice of KW·IC bytes, so the packed GEMM reads
   downsample windows in place with row stride s·IC — and the `[C,T]→[T,C]`
   transpose feeding each conv dies with it (12 of 14 im2col nodes, all 14
   per-stage transposes). VAE **1.44×**, bit-exact.
8. **Residual fusion** — the one deliberate numerics change. Each ConvNeXt block
   half ended with ADD_SCALED: quantise the output, re-read, dequantise, apply the
   layer scale, add the residual. Fused into the producer's epilogue in place over
   the cache-warm float buffer, so only the *sum* is quantised — one DRAM round
   trip and one quant→dequant round trip per block half removed. Gated on WER, not
   hashes: **+0.04 corpus WER for acoustic 1.35× / semantic 1.19×**.

### Determinism

Transcripts previously depended on thread count. Bisection with per-stage tensor
dumps traced it to vector/scalar boundaries that moved with the thread partition
while computing different roundings (FMA contraction, reciprocal-vs-divide,
nearest-even vs half-away). All hot paths now use masked full-width AVX-512 with no
scalar edges: **byte-identical output at `-t 1/2/3/4`**, verified by transcript hashes.

### Arena sizing (the long-input segfault)

Inputs beyond ~25 s died with "segfault at 0" minutes into the encode.
`ggml_graph_compute_with_ctx` allocates the thread work buffer *as an object in the
same context* as the graph tensors and sizes it from the largest node, so it scales
with audio length too; the arena reserved for tensors only, the work-buffer
allocation returned NULL, and kernels wrote to address 0. Measured with
`VIBEASR_ARENA_STATS=1`: 3076 B/sample of tensors + 1024 B/sample of work buffer,
flat across lengths. Reserving 4600 B/sample fixes it and raises the ceiling from
~60 s to ~135 s on a 16 GB host. The remaining limit is architectural — a ggml
context arena has no liveness analysis, so every intermediate stays alive for the
whole graph; lifting it means moving the encoder to `ggml_gallocr`.

### Portability

`GGML_NATIVE=OFF` by default (AVX2 baseline + runtime dispatch). Learned the hard
way: a `-march=native` build died with SIGILL when the VM migrated to a host without
AVX512-FP16. One binary serves every x86-64-with-AVX2 host and still lights up VNNI.

### Negative results, kept on record

- **Parallel encoders.** The acoustic and semantic graphs are independent and can
  run concurrently on pinned disjoint cores (without pinning, two ggml pools
  spin-waiting at node barriers measured 3.5× *slower*). On 4 cores it still loses:
  semantic is latency-bound (~1.1 s at any thread count) while acoustic scales to
  every core, so a pinned 2+2 split measured 6.25 s vs 3.44 s sequential. Ships
  default-on only at ≥ 6 threads.
- **Stage-6 truncate-and-project** (ridge regression, H3-style): cosine 0.92–0.98
  to the full encoder, but WER doubles — an autoregressive consumer propagates what
  a one-shot diffusion consumer absorbs. Also targeted parameters, not time.
- **Q8_0 output head**: no accuracy gain over the tied Q6_K embedding for +248 MB.
- **VNNI for the LM's I2_S Nx1 kernel**: AVX2 measured faster (87 vs 45 GMAC/s).

---

## Runtime switches

Every optimisation ships with its kill switch. Defaults are what the benchmarks measure.

| Switch | Default | Effect of overriding | Basis for the default |
|:--|:--|:--|:--|
| `VIBEASR_GEMM_PACKED=0` | on | unfused tiled GEMM + separate dequant pass | packed: acoustic 1.26×, bit-exact |
| `VIBEASR_CONV_GEMM=0` | on | im2col + transpose downsampling | conv-as-GEMM: VAE 1.44×, bit-exact |
| `VIBEASR_RES_FUSE=0` | on | unfused ADD_SCALED chain (bit-exact mode) | fusion: 1.17× corpus at +0.04 WER |
| `VIBEASR_DWCONV=0` | on | permute/im2col mixer chain | dw_direct: 1.40–1.46×, bit-exact |
| `VIBEASR_PAR_ENC=1/0` | on at ≥ 6 threads | force/forbid concurrent encoders | loses on 4 cores (6.25 s vs 3.44 s) |
| `VIBEASR_GEMM_TILE=0` | on | pre-tile blocked vec_dot (fallback path) | tile: ~2× kernel throughput |
| `VIBEASR_I2_NX1_VNNI=1` | off | VNNI for the LM's I2_S Nx1 | AVX2 measured faster (87 vs 45 GMAC/s) |
| `VIBEASR_ALLOW_JSON=1` | off | permit `--prompt-format json` on 1.5B | JSON targets the 7B checkpoint |
| `VIBEASR_ISA=avx2\|avx512\|vnni` | auto | cap the dispatched ISA | — |
| `VIBEASR_ISA_VERBOSE=1` | off | print the dispatched ISA at start | — |
| `VIBEASR_NODE_PROFILE=1` | off | per-node/section graph profiler | run with `VIBEASR_PAR_ENC=0` |
| `VIBEASR_KERNEL_STATS=1` | off | per-kernel time/GMAC table at exit | — |
| `VIBEASR_ARENA_STATS=1` | off | tensors + work-buffer bytes per sample | how to re-derive the reservation |
| `VIBEASR_GEMM_SHAPES=1` | off | print every GEMM call's shape | the tile-tuning census |
| `VIBEASR_STAGE6_PROJ=<f>` | off | stage-6 truncation apparatus | negative result, kept on record |

## Tools

| | |
|:--|:--|
| `tools/requant_lm_head.cpp` | re-quantise or drop one tensor of an I2_S GGUF |
| `bench/kernel_bench` | exact-reference kernel validation + throughput |
| `bench/interleaved_speed.py` | paired speed ratios, engines cycled per clip |
| `bench/shortlong_bench.py` | short vs long regime, all engines, all languages |
| `bench/fetch_longform.py` + `bench/lf_eval.py` | real long-form sets (TED-LIUM, Earnings-21) |
| `bench/hotword_eval.py` | the four-condition hotword table |
| `bench/model_report.py` | per-tensor bit-budget of a GGUF |
| `bench/run_all.sh` | full reproduction: build → models → sweeps → tables |

---

## Notes for Windows

Windows builds require **GCC or Clang** — MSVC is rejected by `src/CMakeLists.txt`.
MinGW-w64 (e.g. [WinLibs](https://winlibs.com/)) is recommended:

```bash
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_MAKE_PROGRAM=mingw32-make
cmake --build build --target asr_infer -j
```

Keep the MinGW `bin` dir on `PATH` at runtime so the executables find their DLLs.

## Rebuilding the slim weights

```bash
./build/bin/requant_lm_head original-lm.gguf slim-lm.gguf --drop
python bench/model_report.py slim-lm.gguf      # verify the bit budget
```

What `--drop` does and why it is lossless is explained on the
[model card](https://huggingface.co/P2Enjoy/VibeVoice-ASR-BitNet-slim).

## Upstream project & citation

```bibtex
@article{xu2025vibeasrbitnet,
    title={VibeVoice-ASR-BitNet Technical Report},
    author={Xu, Songchen and Song, Ting and Huang, Shaohan and Peng, Zhiliang and Xia, Yan and Tu, Yujie and Huang, Xin and Yu, Jianwei and Dong, Li and Wei, Furu},
    journal={arXiv preprint arXiv:2607.21075},
    year={2025}
}
```

## License

MIT, as upstream.
