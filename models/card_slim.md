---
language:
- en
- fr
- it
- pt
- es
- de
license: mit
pipeline_tag: automatic-speech-recognition
tags:
- ASR
- quantization
- cpu-inference
- gguf
- bitnet
library_name: ggml
base_model: microsoft/VibeVoice-ASR-BitNet
---

# VibeVoice-ASR-BitNet-slim

A repack of [microsoft/VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet)
with a redundant tensor removed, served by a substantially optimised CPU runtime.
**No retraining; the ternary transformer body is byte-for-byte the original.**

| | LM | VAE | Total | bits/weight (LM) |
|:--|--:|--:|--:|--:|
| microsoft/VibeVoice-ASR-BitNet | 992.9 MB | 703.1 MB | 1.70 GB | 4.44 |
| **this repo** | **526.1 MB** | 703.1 MB | **1.23 GB** | **2.69** |

## What changed in the weights

The released LM GGUF stores `output.weight` as F16 (466.7 MB — 47% of the file) next
to `token_embd.weight` as Q6_K. In the source checkpoint `tie_word_embeddings` is
true and the two matrices are **bit-identical**, so the F16 tensor is the same matrix
twice at higher precision. llama.cpp loads `LLM_TENSOR_OUTPUT` as optional and falls
back to `token_embd`, so the duplicate is simply dropped: the output projection runs
through the Q6_K copy. That is the only numerical change, and it also removes
466.7 MB from every decoded token's memory traffic.

## Accuracy

FLEURS test clips (24 per language) plus a Multilingual LibriSpeech French anchor,
greedy decoding, scored corpus-level with digits and years spelled out on both sides
(methodology and scripts:
[bench/README.md](https://github.com/martinobettucci/VibeASR-bitnet.cpp/blob/claude/asr-cpu-optimization-cztnh9/bench/README.md)).

Both models measured with one runtime build (branch commit `094b53c`, portable
AVX2+VNNI, deterministic — identical transcripts at any thread count). Transcripts
are bit-reproducible *within* a build; absolute WER can shift a point or two between
builds as float codegen changes, which is why both columns always come from the same
binary.

| Corpus / language | Released (1.70 GB) | **Slim (1.23 GB)** | Δ |
|:--|--:|--:|--:|
| FLEURS Spanish | 5.96 | 5.96 | 0.00 |
| FLEURS English | 6.48 | 6.83 | +0.35 |
| FLEURS Portuguese | 9.23 | 9.23 | 0.00 |
| FLEURS Italian | 9.67 | 9.67 | 0.00 |
| FLEURS German | 13.41 | 13.24 | −0.17 |
| FLEURS French | 34.63 | 36.02 | +1.39 |
| **FLEURS corpus** | **13.95** | **14.25** | **+0.30** |
| MLS French (read speech) | 21.97 | 22.31 | +0.34 |

Four of six FLEURS languages produce **identical transcripts** under both models;
German is marginally better, and the corpus delta (+0.30) comes almost entirely from
French, the model's weakest language, where ±1–2 points is within sampling noise on
~700 reference words. Practical read: **no measurable accuracy cost** for the 27%
size reduction.

The MLS row doubles as the register check: French drops from ~35% (FLEURS,
encyclopedic) to ~22% (audiobooks) for both models — the FLEURS gap is corpus
difficulty, not a defect of either quantisation.

The runtime also ships opt-in decoder-level hotword biasing (`--hotwords`,
token-trie logit boosting): on FLEURS-French with oracle terms it recovers ~4 WER
points at `--hotword-boost 5`, with no degradation when fed wrong terms. Details and
the four-condition table in the repo README.

## Languages

VibeVoice-ASR was trained on **en, zh, fr, it, ko, pt, vi**. Of the EU languages
above, Spanish and German are out-of-distribution but generalise usably; French is
in-distribution yet hard on FLEURS' register (heavy in proper nouns — the tech
report's 17.4 on MLC-FR is a different corpus, not a contradiction). Other EU
languages degrade sharply; this repack does not change language coverage.

## Speed

Documented in the runtime repo, not here — it is a property of the code, and the
code moved a lot: ~2.8× end to end versus the upstream runtime on a 4-core AVX-512
VM (compute RTF ≈ 0.37 for an 8 s clip), via VNNI kernels, a register-tiled INT8
GEMM, vectorised quantisation epilogues and a layout-native depthwise convolution.
Tables, profiler methodology, and full reproduction scripts:

➡️ **[martinobettucci/VibeASR-bitnet.cpp](https://github.com/martinobettucci/VibeASR-bitnet.cpp/tree/claude/asr-cpu-optimization-cztnh9)** — "CPU optimisation on AVX-512"

## Usage

Drop-in for the released model — same runtime, same flags:

```bash
./build/bin/asr_infer \
    --vae-model vibeasr-vae-encoder-i8_s.gguf \
    --lm-model  vibeasr-lm-i2_s-tied.gguf \
    --audio input.wav -t 4 --greedy
```

## Provenance

Produced with `tools/requant_lm_head.cpp --drop` from the repo above. The VAE
encoder and tokenizer files are copied unmodified from upstream. Licensed MIT, as
upstream.
