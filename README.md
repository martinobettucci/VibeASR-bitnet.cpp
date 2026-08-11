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

Work done on this fork. Reference hardware: **4-vCPU Intel Xeon (Cascade Lake class,
AVX-512F/BW/DQ/VL + VNNI) cloud VMs**. Every number is reproduced by
`./bench/run_all.sh`; methodology and per-script docs in [bench/README.md](bench/README.md).
Accuracy tables live on the [model card](https://huggingface.co/P2Enjoy/VibeVoice-ASR-BitNet-slim)
and are not duplicated here.

### Result: ~2.8× end to end, real-time on 4 modest cores

Compute-only time for an 8.38 s FLEURS clip at 4 threads (VAE encode + prefill +
decode, model load excluded), medians of 7 runs, each pair measured back-to-back on
one host:

| Configuration | Compute | RTF |
|:--|--:|--:|
| Upstream (AVX2 kernels, released code) | ~8.8 s | ~1.05 |
| + row-block tuning, VNNI kernels, tiled INT8 GEMM, vector epilogues | 5.6 s | 0.67 |
| + layout-native depthwise conv | **3.1 s** | **0.372** |

Cloud VMs migrate across hosts and absolute times vary between sessions; the
*ratios* were taken as same-host A/B pairs with non-overlapping 7-run ranges, and the
largest single step (the layout-native conv) replicates across two different hosts:
1.46× (4559→3115 ms) and 1.40× (7992→5703 ms). At these speeds a
stream holds real-time on ~1.5 cores, so a 32-core server carries roughly 20
concurrent streams — the deployment shape that replaces an ASR GPU.

### Where the time went, measured

A per-node graph profiler (`VIBEASR_NODE_PROFILE=1`, sections named per encoder
stage) drove every optimisation. The headline findings, in the order they were found:

1. **Per-call overhead, not arithmetic.** The GEMM row blocks were 4, issuing 194M
   `vec_dot` calls per clip at ~2% of int8 peak. Tuned to 32 (`bench/row_block_sweep.sh`).
2. **The AVX2 kernels were numerically wrong.** int16 accumulation of `vpmaddubsw`
   wraps on full-range int8; the AVX-512 VNNI kernels accumulate in int32.
   `kernel_bench --check` validates 1661 points against exact scalar references.
3. **A register-tiled INT8 GEMM** (4×4 int32 accumulator tile, +128 weight-bias trick
   instead of a per-pair sign fold): ~2× kernel throughput over the blocked vec_dot.
4. **Scalar epilogues.** After every fused matmul the runtime made two scalar passes
   (dequant+absmax, then `roundf` per element) over multi-megabyte activations —
   vectorised, bit-exact by construction.
5. **Data movement was 41% of graph time** (CONT transposes 23.5% + IM2COL 17.3%).
   The ConvNeXt mixer's `permute→cont→im2col→matmul→cont` chain is now one
   layout-native depthwise-conv node (`dw_direct`): the [dim, frames] layout makes the
   causal k-tap conv k fused multiply-adds over contiguous vectors. Data movement fell
   to 8%; byte-identical transcripts with the old chain (`VIBEASR_DWCONV=0`).

### Determinism

Transcripts previously depended on thread count. Bisection with per-stage tensor
dumps traced it to vector/scalar boundaries that moved with the thread partition
while computing different roundings (FMA contraction, reciprocal-vs-divide, nearest-
even vs half-away). All hot paths now use masked full-width AVX-512 with no scalar
edges: **output is byte-identical at `-t 1/2/3/4`**, verified by transcript hashes.

### Portability

The build defaults to `GGML_NATIVE=OFF` (AVX2 baseline + runtime dispatch to
AVX-512/VNNI in our kernels). This is learned the hard way: a `-march=native` build
died with SIGILL when the VM migrated to a host without AVX512-FP16. One binary now
serves every x86-64-with-AVX2 host and still lights up VNNI where present.

### Negative results, kept on record

- **Stage-6 truncate-and-project** (ridge regression, H3-style): cosine 0.92–0.98 to
  the full encoder, but WER doubles — an autoregressive consumer propagates what a
  one-shot diffusion consumer absorbs. Also targeted parameters, not time: the last
  stage holds 89% of VAE FFN weights and ~4% of VAE runtime. Apparatus retained
  (`bench/stage6_calib.py`, `VIBEASR_STAGE6_PROJ`).
- **Prompt hotwords** (`--context-info`): oracle proper nouns took French from 36%
  to 90% WER. The slot derails autoregressive decoding; real hotword biasing needs
  decoder-level score boosting, which this runtime does not have.
- **Q8_0 output head**: no accuracy gain over the tied Q6_K embedding (corpus 13.81
  vs 13.65) for +248 MB.

### Tools this added

| | |
|:--|:--|
| `tools/requant_lm_head.cpp` | re-quantise or drop one tensor of an I2_S GGUF |
| `bench/kernel_bench` | exact-reference kernel validation + throughput |
| `bench/run_all.sh` | full reproduction: build → models → sweeps → tables |
| `bench/model_report.py` | per-tensor bit-budget of a GGUF |
| `VIBEASR_NODE_PROFILE` / `VIBEASR_KERNEL_STATS` | graph and kernel profilers |
| `patches/0001-ggml-i8s-fast-paths.patch` | all submodule changes, applied by `setup_env.py` |

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
