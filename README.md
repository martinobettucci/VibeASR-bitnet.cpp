<h1 align="center">VibeASR.cpp</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://huggingface.co/microsoft/VibeVoice-ASR-BitNet"><img src="https://img.shields.io/badge/🤗-Models-orange.svg" alt="HuggingFace"></a>
  <a href="https://huggingface.co/spaces/microsoft/vibevoice-asr-bitnet-demo"><img src="https://img.shields.io/badge/✨-Demo-green.svg" alt="Demo"></a>
  <a href="https://arxiv.org/abs/2607.21075"><img src="https://img.shields.io/badge/📄-Tech_Report-red.svg" alt="Tech Report"></a>
</p>

---

**VibeASR.cpp** is the official inference runtime for **VibeVoice-ASR-BitNet** — enabling real-time multilingual speech recognition on CPU through heterogeneous quantization (I8\_S for VAE + I2\_S for LM).

To enable efficient edge CPU deployment, we replace the original Qwen2.5-7B language model with Qwen2.5-1.5B, achieving only modest accuracy degradation (1–4% absolute WER increase) while reducing the total model size from 4.62 GB to 1.58 GB. Combined with custom SIMD kernels and operator fusion in the ggml framework, VibeVoice-ASR-BitNet achieves **1.6–2.3× faster** inference than Whisper.cpp at comparable model sizes, with real-time capability (RTF < 1) on low-resource CPUs.

<p align="center">
  <img src="media/report_overview.png" width="92%"/>
</p>

<p align="center">
  📄 <a href="https://arxiv.org/abs/2607.21075">Tech Report</a> &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/microsoft/VibeVoice-ASR-BitNet">Models</a> &nbsp;|&nbsp;
  ✨ <a href="https://huggingface.co/spaces/microsoft/vibevoice-asr-bitnet-demo">Demo</a> &nbsp;|&nbsp;
  🏠 <a href="https://aka.ms/GeneralAI">GeneralAI</a>
  
</p>

---

## Key Results

### Model Size

<div align="center">

| Component | VibeVoice-ASR-1.5B (FP16) | VibeVoice-ASR-BitNet | Compression |
|:---------:|:-------------------------:|:--------------------:|:-----------:|
| VAE Tokenizer | 1.31 GB | 0.65 GB | 2.0× |
| LM Decoder | 3.32 GB | 0.92 GB | 3.6× |
| **Total** | **4.62 GB** | **1.58 GB** | **2.9×** |

</div>

### Inference Performance

<div align="center">

**AMD EPYC 7V13 (AVX2+FMA, 24C, 216GB)**

| | 1T | 2T | 3T | 4T | 6T | 8T |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RTF | 1.98 | 1.08 | **0.77** | **0.63** | **0.49** | **0.42** |
| vs. Whisper.cpp | 2.28× | 2.12× | 1.86× | 1.86× | 1.71× | 1.55× |

**Apple M4 (ARM NEON, 4P+6E, 16GB)**

| | 1T | 2T | 3T | 4T | 6T | 8T |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RTF | 1.18 | **0.68** | **0.52** | **0.43** | **0.48** | **0.42** |

**Intel Core i7-13700 (AVX2+FMA, 8P+8E, 32GB, Windows 11 / MinGW GCC)**

| | 1T | 2T | 3T | 4T | 6T | 8T |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RTF | 1.55 | **0.97** | **0.78** | **0.71** | **0.67** | **0.69** |

</div>

> RTF (Real-Time Factor) on audio input. **Bold** = RTF < 1 (real-time). EPYC/M4 use 20s audio; i7-13700 uses a 10.3s clip.

### Accuracy (WER%)

<div align="center">

| Benchmark | VibeVoice-ASR-7B (FP16) | VibeVoice-ASR-BitNet | Parakeet | Whisper | SenseVoice | FunASR |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MLC-EN | 7.82 | **8.25** | 8.40 | 13.57 | 12.39 | 11.36 |
| MLC-FR | 16.03 | 17.41 | — | — | — | — |
| MLC-IT | 15.67 | 17.23 | — | — | — | — |
| MLC-KO | 9.83 | 11.15 | — | — | — | — |
| MLC-PT | 22.41 | 24.87 | — | — | — | — |
| MLC-VI | 20.15 | 22.38 | — | — | — | — |
| AISHELL4 | 19.83 | 27.45 | — | — | 22.52 | **20.41** |
| AMI-ihm | 17.42 | **21.36** | 21.92 | 27.07 | 30.81 | 32.07 |
| AMI-sdm | 24.18 | **25.87** | 26.33 | 36.92 | 48.11 | 40.17 |
| AliMeeting | 36.21 | 40.58 | — | — | **38.75** | 39.27 |
| Fleurs-en | 4.73 | 5.21 | 4.09 | **3.99** | 6.84 | 4.93 |
| Fleurs-zh | 7.92 | 8.35 | — | — | **5.56** | 7.00 |
| Libri-clean | 2.17 | 2.41 | **1.49** | 1.98 | 2.78 | 1.58 |
| Libri-other | 5.84 | 6.27 | **3.13** | 3.60 | 6.81 | 4.01 |
| VoxPopuli | 4.92 | **5.18** | 5.26 | 7.19 | 8.63 | 6.46 |

</div>

> **Note:** The accuracy benchmarks above are evaluated on standard-accent speech corpora. Performance on accented or dialectal speech not represented in the training data may degrade more significantly, as is common with ASR models trained on specific data distributions.

---

## Quick Start

### Requirements

- Python ≥ 3.9, CMake ≥ 3.14, GCC/Clang with C++11 support
- ~2 GB disk space (code + quantized models)

> **Windows users:** MSVC is **not** supported — the build requires GCC or Clang (MinGW-w64 recommended). See [Notes for Windows](#notes-for-windows) below.

### One-Command Setup

```bash
git clone --recursive https://github.com/microsoft/VibeASR.cpp.git
cd VibeASR.cpp
pip install -r requirements.txt
python setup_env.py
```

### Manual Build

```bash
git clone --recursive https://github.com/microsoft/VibeASR.cpp.git
cd VibeASR.cpp

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Download pre-quantized models
pip install huggingface_hub
huggingface-cli download microsoft/VibeVoice-ASR-BitNet --local-dir models/vibeasr
```

---

## Usage

### CLI Inference

```bash
./build/bin/asr_infer \
    --vae-model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
    --lm-model models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf \
    --audio input.wav -t 4
```

### Web Demo (Gradio)

```bash
pip install gradio soundfile numpy

python demo/gradio_asr_demo.py --port 7860 \
    --vae-model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
    --lm-model models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf
```

---

## Notes for Windows

Windows builds require **GCC or Clang** — MSVC is rejected by `src/CMakeLists.txt`. MinGW-w64
(e.g. [WinLibs](https://winlibs.com/)) is recommended. Use the *MinGW Makefiles* generator, and
keep the MinGW `bin` dir on your `PATH` at runtime so the executables find their DLLs:

```bash
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_MAKE_PROGRAM=mingw32-make
cmake --build build --target asr_infer -j
```

- Build with the command above rather than `setup_env.py` (its clang probe assumes a POSIX shell).
- Gradio: run `python demo/gradio_asr_demo.py --port 7860` (model paths come from the script's
  `MODEL_CONFIGS`, not `--vae-model`/`--lm-model`; pass `--bin build/bin/asr_infer.exe` if needed).

---

## Model Conversion

For most users, downloading pre-quantized models from [HuggingFace](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) is recommended. To convert from SafeTensors yourself:

### Step 1: SafeTensors → F32 GGUF

```bash
# LM (BitNet) — handles weight preprocessing and config flattening automatically
python utils/convert_lm_to_gguf.py <safetensors-dir>

# VAE Tokenizer
python utils/convert_vae_to_gguf.py <safetensors-dir>
```

### Step 2: F32 GGUF → Quantized GGUF

```bash
# VAE Tokenizer: F32 → I8_S
./build/bin/llama-quantize \
    <safetensors-dir>/vibeasr-vae-encoder-f32.gguf \
    <safetensors-dir>/vibeasr-vae-encoder-i8_s.gguf \
    I8_S 1 1

# LM: F32 → I2_S (with Q6_K embeddings)
./build/bin/llama-quantize --token-embedding-type Q6_K \
    <safetensors-dir>/vibeasr-lm-f32.gguf \
    <safetensors-dir>/vibeasr-lm-i2_s-embed-q6_k.gguf \
    I2_S 1 1
```

---

## CPU optimisation on AVX-512

Work done on this fork, measured on **Intel Xeon @2.8 GHz (Cascade Lake, 4 cores,
AVX-512F/BW/DQ/VL + AVX512_VNNI, 15 GB)** against an 8.38 s FLEURS clip.
Everything below is reproduced by `./bench/run_all.sh` — see [bench/README.md](bench/README.md).

### Speed: 1.59–1.78× end to end

`bench/rtf_compare.sh`, three configurations back to back on the same clips and box.
4 clips, 37.0 s of audio, compute-only RTF (model load excluded):

| Threads | upstream | fork-code | fork-full | speedup |
|--:|--:|--:|--:|--:|
| 1 | 3.463 | 1.944 | 2.025 | 1.78× |
| 2 | 1.785 | 1.032 | 1.031 | 1.73× |
| 4 | 1.021 | **0.641** | 0.649 | **1.59×** |

`upstream` is AVX2 kernels with row blocks at 4 and the released LM; `fork-code` adds
the AVX-512 VNNI kernels, row blocks at 32, the register-tiled INT8 GEMM and the
vectorised fused-op epilogues; `fork-full` also swaps in the slim LM. All fork paths
gate on the ISA at run time, so the `upstream` column genuinely runs the original
code, and one binary serves both.

`fork-code` → `fork-full` is within noise: dropping 467 MB only touches decode,
roughly 11% of compute. **The size work paid for itself in bytes, not in seconds.**
The speedup holding at 2 threads (1.73×) matters for deployment: on a many-core
server the natural shape is many concurrent streams at 2–3 threads each, each one at
or under real time.

The row blocks (`VAE_ROW_BLOCK_SIZE`, `ROW_BLOCK_SIZE` — how many activation rows go
into one `vec_dot` call) are the largest single contributor. At 4, one clip issues
**194 million** `vec_dot` calls averaging ~1200 MACs each. Measured over 7 runs per
value on the 8.38 s clip, total compute:

| Row block | median | min–max |
|--:|--:|:--|
| 4 (upstream) | 8451 ms | 8040–8586 |
| **32** | **7607 ms** | 7300–7875 |

**1.10×.** This is result-preserving — transcripts hashed across 3 clips × 2 thread
counts are byte-identical between block 4 and 32. Re-tune with
`bench/row_block_sweep.sh --macro <NAME>`.

#### Register-tiled INT8 GEMM

Blocking only amortises the call; the loop still re-read one operand for every step
of the other. `ggml_gemm_i8_i8_tiled` holds a 4×4 tile of int32 accumulators in
registers so both operands are loaded once per tile. It also drops the sign fold the
`vec_dot` kernels need — `vpdpbusd` wants one unsigned operand, and folding x's sign
onto y costs work *per operand pair*, which never amortises. Instead the weights get
+128 (a sign-bit flip, making them unsigned) with a `-128·Σy` correction per row,
where the row sums are computed once per row block.

Single thread, tile vs the blocking it replaces, both on VNNI
(`VIBEASR_GEMM_TILE=0` selects the old path without demoting `vec_dot`):

| shape | tiled | vec_dot blocking | |
|:--|--:|--:|--:|
| n=512, 32×32 | 105.7 | 67.3 | 1.57× |
| n=2048, 32×32 | 99.7 | 44.5 | 2.24× |
| n=2048, 128×64 | 106.7 | 44.5 | 2.40× |
| n=8960, 64×32 | 89.9 | 44.5 | 2.02× |

End to end, 7 runs each on the 8.38 s clip: **7510 → 6774 ms median (1.11×)**, ranges
7361–7721 and 6524–6931, non-overlapping. Output is byte-identical (3 clips hashed),
and `kernel_bench` checks the tile against the scalar reference at 1655 points
including the ragged edges where it falls back to `vec_dot`.

This one needs a patch to the pinned submodule, since `ggml_gemm_i8_i8` lives there.
The kernel itself is in `src/`; `patches/0001-ggml-i8s-fast-paths.patch` only swaps
call sites, and `setup_env.py` applies it (idempotently) before building.

#### Vectorised fused-op epilogues

With the tile in place, profiling showed the VAE spending only ~25% of its CPU in
matmul kernels. The biggest remaining block: after every fused I8_S matmul/conv the
runtime made two full **scalar** passes over the output — int32 → float with scale
and bias while tracking the absolute maximum, then float → int8 with clamp and
`roundf`, a libm call per element — on multi-megabyte activations at every layer.

`vibeasr_i8s_dequant_absmax` and `vibeasr_i8s_quant_i8` (AVX-512, in `src/`) replace
eight such loops. Bit-exact by construction: mul-then-add ordering is preserved and
`roundf`'s half-away-from-zero is implemented as `trunc(v + copysign(0.5, v))`;
transcripts hash identically before and after. 7 runs on the 8.38 s clip, 4 threads:
tile-only median 6774 ms → **5579 ms** (1.21×), VAE encode ~5650 → ~3770 ms.

> **A correction.** Earlier revisions of this section claimed 2.08× end to end, 1.55×
> on the VAE and 2.95× on prefill. Those came from single-run sweeps, and this VM has
> roughly 2× transient variance — the slow-configuration rows happened to land in slow
> windows. Repeated measurement (7 reps, non-overlapping ranges) gives the numbers
> above. The sweep script now requires repetitions and reports median and spread, and
> it no longer times with `VIBEASR_KERNEL_STATS` enabled, since that profiler's cost
> is per-call and would itself bias results toward larger blocks.

### Accuracy: the AVX2 kernels were wrong

The AVX2 I8_S kernels accumulate `vpmaddubsw` results in int16 and only flush to
int32 every 32 blocks, so they **wrap** on activations spanning the full ±127 range
that `quantize_i8_s` emits. `vpdpbusd` accumulates in int32 and cannot.
`bench/kernel_bench.cpp` checks every path against an exact int32 scalar reference:

| Path | narrow (\|v\| ≤ 8) | full (\|v\| ≤ 127) |
|:--|--:|--:|
| AVX-512 VNNI | 360/360 | **360/360** |
| AVX2 | 360/360 | **216/360** |

The I2_S kernels are unaffected — ternary weights keep partial sums small.

Kernels select the widest supported path at run time (`VIBEASR_ISA=avx2\|avx512\|vnni\|amx`
forces one; `VIBEASR_KERNEL_STATS=1` reports where time goes). AMX-INT8 is detected,
including the `arch_prctl` tile request, but no tile kernels ship — this hardware
cannot execute them, so they could not be validated.

> On a 1-FMA-unit AVX-512 part, zmm `vpdpbusd` retires 1/cycle and ymm `vpmaddubsw`
> 2/cycle — both 64 int8 products per cycle. VNNI buys correctness here, not raw
> throughput. Parts with two FMA units should see more.

### Size: the LM shipped a duplicate tensor

`output.weight` was **F16, 466.7 MB — 47% of the LM** — while `token_embd.weight` sat
beside it as Q6_K at 191.4 MB. In the source checkpoint `tie_word_embeddings` is true
and `lm_head.weight` is **bit-identical** to `embed_tokens.weight`, so it was the same
matrix twice, the second copy at higher precision. llama.cpp loads
`LLM_TENSOR_OUTPUT` as `TENSOR_NOT_REQUIRED` and falls back to `token_embd`, so it can
simply be dropped:

The repacked model — sizes, bit budget, and the measured WER cost of the trade
(about +0.4 corpus-wide for −47% LM size) — lives at
**[P2Enjoy/VibeVoice-ASR-BitNet-slim](https://huggingface.co/P2Enjoy/VibeVoice-ASR-BitNet-slim)**;
its model card is the authoritative source for those numbers and they are not
repeated here. `llama-quantize` cannot produce it — it has no way to leave the I2_S
body alone — hence `tools/requant_lm_head.cpp`.

Note the ternary body is packed at exactly **2.000** bits/weight, not log₂3 = 1.585:
I2_S stores four ternary values per byte and wastes one of four codes, which is 68 MB
of padding (20.8% of the body). `bench/model_report.py` prints the full budget.

### WER: measured here, reported there

VibeVoice-ASR was trained on **en, zh, fr, it, ko, pt, vi**. Of the EU official
languages that means English, French, Italian and Portuguese are in-distribution;
Spanish and German are not but generalise usably. Others degrade sharply and no
amount of quantisation or kernel work changes that.

The WER methodology lives in this repo — FLEURS slices, corpus-level scoring, number
spelling on both sides (`bench/README.md` documents the conventions, and
`bench/run_asr.py` / `bench/summarize.py` regenerate every figure). The resulting
per-language tables for the released model against the slim repack are published on
the [model card](https://huggingface.co/P2Enjoy/VibeVoice-ASR-BitNet-slim), not
duplicated here.

One calibration point worth keeping in mind when reading any of them: FLEURS French
is much harder than the MLC-FR set the tech report scores 17.41 on — FLEURS Italian
scores *better* here than the report's MLC-IT — so cross-corpus comparisons mislead;
the gap is the corpus, not the pipeline.

### Known issue: output depends on thread count

`en_us/0000` decodes "the periodic table" at 1 and 2 threads and "a priori" at 4 —
same binary, same weights, greedy sampling. Some float reduction in the graph is
partitioned by thread count. Pre-existing, not introduced here, but it means WER is
only comparable between runs at equal `-t`.

---

## Citation

```bibtex
@article{xu2025vibeasrbitnet,
    title={VibeVoice-ASR-BitNet Technical Report},
    author={Xu, Songchen and Song, Ting and Huang, Shaohan and Peng, Zhiliang and Xia, Yan and Tu, Yujie and Huang, Xin and Yu, Jianwei and Dong, Li and Wei, Furu},
    journal={arXiv preprint arXiv:2607.21075},
    year={2025}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
