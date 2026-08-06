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

### Speed: 2.08× end to end

| Stage | Upstream | This fork | |
|:--|--:|--:|--:|
| VAE encode | 9718 ms | 5643 ms | 1.72× |
| LM prefill | 2727 ms | 754 ms | 3.62× |
| LM decode | ~1600 ms | 878 ms | 1.82× |
| **Compute total** | **15161 ms** | **7275 ms** | **2.08×** |
| **RTF, 4 threads** | **1.81** | **0.868** | real-time |

Most of that came from two integers, not from SIMD. `VAE_ROW_BLOCK_SIZE` and
`ROW_BLOCK_SIZE` set how many activation rows go into one `vec_dot` call in the I8_S
and I2_S GEMMs. Both were 4, so one clip issued **194 million** `vec_dot` calls
averaging ~1200 MACs each, running at roughly **2% of this CPU's int8 peak**. The
arithmetic was never the bottleneck — the per-call prologue, epilogue and horizontal
reduction were.

| Row block | VAE encode | LM prefill | vec_dot calls |
|--:|--:|--:|--:|
| 4 (upstream) | 9718 ms | 2727 ms | 194.8M |
| 16 | 6524 ms | 1025 ms | 52.9M |
| **32** | **6289 ms** | **925 ms** | 31.9M |
| 64 | 6504 ms | 935 ms | 25.1M |

Past 32 the win reverses: call count keeps falling but the activation rows stop
fitting in L1. Re-tune with `bench/row_block_sweep.sh --macro <NAME>`.

This is result-preserving, and that was verified rather than assumed — transcripts
hashed across 3 clips × 2 thread counts are byte-identical between block 4 and 32.

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

| | LM | Total | bits/weight |
|:--|--:|--:|--:|
| As released | 992.9 MB | 1.70 GB | 4.44 |
| `requant_lm_head --drop` | **526.1 MB** | **1.23 GB** | **2.69** |

Published as [P2Enjoy/VibeVoice-ASR-BitNet-slim](https://huggingface.co/P2Enjoy/VibeVoice-ASR-BitNet-slim).
`llama-quantize` cannot do this — it has no way to leave the I2_S body alone — hence
`tools/requant_lm_head.cpp`.

Note the ternary body is packed at exactly **2.000** bits/weight, not log₂3 = 1.585:
I2_S stores four ternary values per byte and wastes one of four codes, which is 68 MB
of padding (20.8% of the body). `bench/model_report.py` prints the full budget.

### WER on the languages the model supports

VibeVoice-ASR was trained on **en, zh, fr, it, ko, pt, vi**. Of the EU official
languages that means English, French, Italian and Portuguese are in-distribution;
Spanish and German are not but generalise usably. Others degrade sharply and no
amount of quantisation or kernel work changes that.

FLEURS, 24 clips per language, greedy, `-t 2`, numbers spelled out on both sides
(the references write `35 mm` where the model says `trente-cinq millimètres`; without
that normalisation WER is inflated by 1–2 points).

| Language | Released | Slim (`--drop`) | Δ |
|:--|--:|--:|--:|
| Spanish | 6.47 | 6.47 | +0.00 |
| English | 8.23 | 8.58 | +0.34 |
| Portuguese | 8.90 | 8.57 | −0.33 |
| Italian | 9.67 | 9.52 | −0.16 |
| French | 34.08 | 35.88 | +1.81 |

Dropping the F16 head costs about **+0.4 WER corpus-wide for −47% LM size**. Small,
but not free — the output projection moves from F16 to Q6_K.

FLEURS French is much harder than the MLC-FR set the tech report scores 17.41 on; for
calibration, FLEURS Italian here (9.67) is *better* than the report's MLC-IT (17.23),
so the gap is the corpus, not the pipeline.

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
