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
with a redundant tensor removed. **No retraining; the ternary transformer body is
byte-for-byte the original.**

| | LM | VAE | Total | bits/weight (LM) |
|:--|--:|--:|--:|--:|
| microsoft/VibeVoice-ASR-BitNet | 992.9 MB | 703.1 MB | 1.70 GB | 4.44 |
| **this repo** | **526.1 MB** | 703.1 MB | **1.23 GB** | **2.69** |

Runtime, kernels and speed engineering live in the
[VibeASR.cpp fork](https://github.com/martinobettucci/VibeASR-bitnet.cpp/tree/claude/asr-cpu-optimization-cztnh9)
and are not repeated here. This card covers the weights: what changed, how accurate
they are, and where they break.

## What changed in the weights

The released LM GGUF stores `output.weight` as F16 (466.7 MB — 47% of the file) next
to `token_embd.weight` as Q6_K. In the source checkpoint `tie_word_embeddings` is
true and the two matrices are **bit-identical**, so the F16 tensor is the same matrix
twice at higher precision. llama.cpp loads `LLM_TENSOR_OUTPUT` as optional and falls
back to `token_embd`, so the duplicate is simply dropped: the output projection runs
through the Q6_K copy. That is the only numerical change, and it also removes
466.7 MB from every decoded token's memory traffic.

Verified twice on independent suites: **+0.30 and +0.32 corpus WER** versus the
original weights, with per-language deltas scattering in both directions — i.e. no
measurable accuracy cost for a 27% size reduction.

## Accuracy

Greedy decoding, corpus-level WER with digits and years spelled out on both sides.
Two regimes, because they rank engines differently.

### Short clips (5–25 s), 7 language/register sets

| Engine | WER |
|:--|--:|
| whisper.cpp large-v3-turbo q5_0 | **2.39** |
| whisper.cpp small q5_1 | 5.79 |
| microsoft original weights | 8.41 |
| **this repo** | 8.73 |

On short clips **whisper is more accurate than this model**, and whisper-small is
also faster. If short-clip transcription accuracy is what you need, use whisper.

Per language (100-clip suite, this repo vs the original weights — the comparison
this card is actually about):

| | de | en | es | fr | fr-MLS | it | pt | **all** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| microsoft original | 16.7 | 6.6 | 5.8 | 32.7 | 21.6 | 8.3 | 10.8 | **15.75** |
| **this repo** | 15.6 | 6.6 | 7.1 | 33.6 | 21.9 | 8.6 | 11.1 | **16.07** |

(The absolute level differs from the table above because that one uses a
14-item two-regime suite; both compare engines measured in the same session.)

### Long-form (real TED talks, whole recordings)

| Engine | mode | 7.7 min | 13.9 min |
|:--|:--|--:|--:|
| whisper.cpp small q5_1 | native | **5.10** | 4.78 |
| **this repo** | 60 s chunks | 5.67 | **4.70** |
| microsoft original weights | 60 s chunks | 5.61 | 5.38 |
| whisper.cpp large-v3-turbo q5_0 | native | 29.00 | 39.69 |

On genuine long-form this model is **level with whisper-small** — each wins one
talk — and roughly 5 WER against a 2.4 on short clips, so the gap to whisper
closes substantially as recordings get longer. whisper-turbo collapses into a
repetition loop on both talks under whisper.cpp's default flags (its reference
implementation has temperature-fallback logic that suppresses this).

Measured on real continuous speech (`distil-whisper/tedlium-long-form`), not on
concatenated short clips — splicing independent utterances produces speaker jumps
that break every engine and measure nothing.

## Limits you must design around

- **~80 seconds per pass.** Past that the decoder emits its end token early and
  silently drops the tail (measured: 101 s of audio → 119 of 280 reference words,
  with the token budget 98% unused). This is a property of the weights — the
  original checkpoint behaves identically. **Chunk longer audio**; 60 s chunks give
  the long-form numbers above.
- **No speaker labels.** The runtime can request a `{Start, End, Speaker, Content}`
  format, but that prompt targets the 7B checkpoint and this 1.5B model emits no
  speaker turns and transcribes ~1.4 WER worse when asked for it. The CLI rejects
  it. There is no diarization here.
- **Language coverage.** VibeVoice-ASR was trained on **en, zh, fr, it, ko, pt, vi**.
  Spanish and German are out of distribution but generalise usably; French is
  in-distribution yet weak on FLEURS' proper-noun-heavy register (~33 WER) and much
  better on read speech (~22 on MLS). Other languages degrade sharply. This repack
  does not change coverage.

## Usage

Drop-in for the released model — same runtime, same flags:

```bash
./build/bin/asr_infer \
    --vae-model vibeasr-vae-encoder-i8_s.gguf \
    --lm-model  vibeasr-lm-i2_s-tied.gguf \
    --audio input.wav -t 4 --greedy
```

Domain terms can be biased at decode time with `--hotwords a,b,c --hotword-boost 5`
(token-trie logit boosting): measured on FLEURS-French with oracle terms it recovers
**36.0 → 31.9 WER**, and feeding the *wrong* clip's terms at the same strength does
not degrade the baseline. λ=8 is past the stability knee.

## Provenance

Produced with `tools/requant_lm_head.cpp --drop`. The VAE encoder and tokenizer files
are copied unmodified from upstream. Licensed MIT, as upstream.
