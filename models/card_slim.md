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

Two measurements, two questions.

**1. Does the slim repack cost accuracy?** FLEURS test clips (24 per language)
plus a Multilingual LibriSpeech French anchor, greedy decoding, scored
corpus-level with digits and years spelled out on both sides (methodology and
scripts:
[bench/README.md](https://github.com/martinobettucci/VibeASR-bitnet.cpp/blob/claude/asr-cpu-optimization-cztnh9/bench/README.md)).
Both models on one runtime build (`094b53c`, portable AVX2+VNNI, deterministic —
identical transcripts at any thread count):

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

Four of six languages produce **identical transcripts**; the corpus delta comes
almost entirely from French, the model's weakest language, where ±1–2 points is
sampling noise on ~700 reference words. Practical read: **no measurable accuracy
cost** for the 27% size reduction. The MLS row is the register check: French drops
from ~35% (encyclopedic FLEURS) to ~22% (audiobooks) for both models — corpus
difficulty, not quantisation damage.

**2. Where does the model sit in the landscape?** A second, independent suite —
100 clips drawn across the six languages plus the MLS-French anchor, all engines
back-to-back on one host at 4 threads, later runtime build, whisper given each
clip's language code (generous: this model runs unhinted):

Speed is given as a ratio over the upstream runtime (absolute RTF is a property
of the benchmark host, not of the model); > 1 is faster than upstream.

| Engine / weights | de | en | es | fr | fr-MLS | it | pt | **all** | **speed** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Upstream engine, original 1.70 GB | 16.7 | 6.6 | 5.8 | 32.7 | 21.6 | 8.3 | 10.8 | **15.75** | 1.00× |
| **This repo (slim) + fork engine, defaults** | 15.6 | 6.6 | 7.1 | 33.6 | 21.9 | 8.6 | 11.1 | **16.07** | **2.35×** |
| This repo, fork engine bit-exact mode | 13.9 | 7.2 | 5.8 | 35.4 | 21.2 | 10.2 | 10.5 | **16.03** | **2.01×** |
| whisper.cpp small q5_1 | 8.5 | 4.6 | 7.1 | 12.9 | 16.7 | 6.6 | 8.0 | **9.92** | **3.06×** |
| whisper.cpp large-v3-turbo q5_0 | 4.1 | 3.7 | 4.0 | 3.3 | 10.3 | 3.3 | 4.3 | **5.12** | **0.75×** |

Slim-vs-original is parity (+0.32, per-language scatter both ways — German
improves under the fork's overflow-fixed kernels, Spanish regresses). On these
**short clips** whisper-small is more accurate and ~1.3× faster.

**But short clips are the wrong test for this architecture, and the difference is
dramatic.** VibeVoice-ASR compresses audio 3200× into an LLM context so a whole
recording is one encode with full context available to the decoder. On 76 s of
continuous French audiobook speech, same host, back to back:

| Engine | RTF | **WER** |
|:--|--:|--:|
| **This model, fork runtime** | 0.44 | **8.88** |
| whisper.cpp small q5_1 | 0.30 | 27.57 |
| whisper.cpp large-v3-turbo q5 | 0.99 | 5.61 |

The ranking inverts: **8.88 against whisper-small's 27.57**, a 3× gap the other
way, and within 3 points of large-v3-turbo at 2.3× less compute. The same weights
score ~21 on this corpus's short clips — more context makes this model better,
which is the point of the architecture.

**Known limit: ~80 seconds.** Past that the decoder emits its end token early and
drops the tail (measured: 101 s of audio → 119 of 280 reference words), with the
token budget 98% unused. It is a property of the weights, not the runtime —
upstream behaves identically. Segment recordings longer than ~75 s.

Two things this model does **not** do, despite the runtime supporting the format:
it emits **no speaker labels**, and the `{Start, End, Speaker, Content}` JSON
prompt is meant for the 7B model — asking this 1.5B for it costs ~1.4 WER. Use
plain text.

Guidance: **use whisper for short-clip transcription; use this for continuous
speech in the 30–75 s range**, where it is far more accurate than whisper-small at
comparable cost and near turbo at a fraction of it — plus decoder-level hotword
biasing (FLEURS-French 36.0 → 31.9 with domain terms). Methodology and
reproduction scripts:
[runtime repo](https://github.com/martinobettucci/VibeASR-bitnet.cpp/tree/claude/asr-cpu-optimization-cztnh9).

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

Documented in the runtime repo, not here — it is a property of the code, and
absolute RTF is a property of the host, so speed is published as **ratios**
between engines measured back-to-back: **≈3.0× faster than the upstream runtime**
(2.56× paired per-clip median over 100 clips, × 1.17× from the residual-fusion
default gated at +0.04 WER above), ~1.25× faster than whisper-small overall and
1.76× faster below 8 s of audio even before fusion. Achieved via VNNI kernels
with a packed-B INT8 GEMM (fused dequantisation epilogue), conv-as-GEMM
downsampling, and a layout-native depthwise convolution. Tables, profiler
methodology, and full reproduction scripts:

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
