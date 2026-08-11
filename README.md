<h1 align="center">VibeASR.cpp — CPU-optimised fork</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://huggingface.co/P2Enjoy/VibeVoice-ASR-BitNet-slim"><img src="https://img.shields.io/badge/🤗-Slim_weights-orange.svg" alt="Slim weights"></a>
  <a href="https://github.com/microsoft/VibeASR.cpp"><img src="https://img.shields.io/badge/upstream-microsoft%2FVibeASR.cpp-lightgrey.svg" alt="Upstream"></a>
</p>

---

This is a fork of [microsoft/VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp)
focused on one question: **how fast and how small can VibeVoice-ASR-BitNet get on a
plain x86 CPU without losing accuracy?** Everything documented here is about the
fork; for the original project, its paper numbers and its documentation, see the
[upstream README](https://github.com/microsoft/VibeASR.cpp#readme).

<p align="center">
  <img src="media/fork_overview.png" width="92%" alt="Fork architecture: dual INT8 VAE encoders into a ternary Qwen2.5-1.5B with tied Q6_K embedding; the 467 MB F16 output head is removed"/>
</p>

What the fork changes, all measured (see the results section below and the
[model card](https://huggingface.co/P2Enjoy/VibeVoice-ASR-BitNet-slim)):

- **~2.8× faster end to end** than the upstream runtime on a 4-core AVX-512 VM —
  compute RTF ≈ 0.37, real-time with headroom. AVX-512/VNNI kernels with runtime
  dispatch, a register-tiled INT8 GEMM, vectorised quantisation epilogues, and a
  layout-native depthwise convolution that removed 33% of graph time.
- **27% smaller model** (1.70 → 1.23 GB): the LM shipped its tied output projection
  twice; the F16 copy is dropped, at no measured accuracy cost (corpus WER 13.65 vs
  13.95 for the original weights, same build, six languages).
- **Deterministic**: identical transcripts at any thread count. Upstream output
  changed with `-t`; the fork removes every partition-dependent rounding path.
- **Portable binaries**: AVX2 baseline build with runtime dispatch up to VNNI —
  no `-march=native` time bombs.
- **A numerical fix**: the upstream AVX2 int8 kernels overflow their int16
  accumulators on full-range activations; the VNNI kernels accumulate in int32.
- **A reproduction harness**: every number in this README regenerates via
  `bench/run_all.sh`, with kernel-level exact-reference tests (`kernel_bench`).

---

## Quick Start

```bash
git clone --recursive -b claude/asr-cpu-optimization-cztnh9 \
    https://github.com/martinobettucci/VibeASR-bitnet.cpp.git
cd VibeASR-bitnet.cpp

pip install -r requirements.txt
python setup_env.py --skip-download     # builds portable binaries, applies patches/

./scripts/download_models.sh            # slim weights from P2Enjoy (1.23 GB)
```

### Run

```bash
./build/bin/asr_infer \
    --vae-model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
    --lm-model  models/vibeasr/vibeasr-lm-i2_s-tied.gguf \
    --audio input.wav -t 4 --greedy
```

Gradio demo: `python demo/gradio_asr_demo.py --port 7860 --vae-model ... --lm-model ...`

### Weights

| | file | size |
|:--|:--|--:|
| LM (ternary I2_S + tied Q6_K embedding) | `vibeasr-lm-i2_s-tied.gguf` | 526 MB |

> The Qwen2.5-**1.5B** backbone is upstream's choice (they distilled down from 7B);
> this fork does not swap or retrain it. The only weight change here is dropping the
> LM's duplicated F16 output projection — the ternary body is byte-for-byte
> Microsoft's. Backbone swaps to smaller Qwens were investigated and rejected with
> measurements (see the negative-results section).
| VAE encoders (INT8) | `vibeasr-vae-encoder-i8_s.gguf` | 703 MB |

Hosted at [P2Enjoy/VibeVoice-ASR-BitNet-slim](https://huggingface.co/P2Enjoy/VibeVoice-ASR-BitNet-slim)
— the model card there is the authoritative source for accuracy tables and the
weight-surgery details. The original weights remain at
[microsoft/VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet)
and still work with this runtime unchanged.

Supported languages (measured): English, French, Italian, Portuguese from the
training mix; Spanish and German generalise usably. Other languages degrade sharply
— that is a property of the model's training data, not of this runtime.

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

## Notes for Windows

Windows builds require **GCC or Clang** — MSVC is rejected by `src/CMakeLists.txt`.
MinGW-w64 (e.g. [WinLibs](https://winlibs.com/)) is recommended:

```bash
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_MAKE_PROGRAM=mingw32-make
cmake --build build --target asr_infer -j
```

Keep the MinGW `bin` dir on `PATH` at runtime so the executables find their DLLs.

---

## Rebuilding the slim weights yourself

The slim LM is produced from the original release with one tool — no retraining:

```bash
./scripts/download_models.sh            # or start from the microsoft GGUFs
./build/bin/requant_lm_head original-lm.gguf slim-lm.gguf --drop
python bench/model_report.py slim-lm.gguf      # verify the bit budget
```

For full SafeTensors → GGUF conversion, follow the
[upstream instructions](https://github.com/microsoft/VibeASR.cpp#model-conversion);
the conversion scripts in `utils/` are unchanged in this fork.

---

## Upstream project & citation

The original project, its technical report, evaluation on the paper's benchmarks,
and the unmodified documentation live at
[microsoft/VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp). If you use this
work academically, cite their report:

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

MIT, as upstream.
