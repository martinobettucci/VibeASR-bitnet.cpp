#!/usr/bin/env bash
#
# Reproduce every number in the "CPU optimisation on AVX-512" section of the README.
#
#   HF_TOKEN=hf_... ./bench/run_all.sh
#
# Steps are individually skippable so a partial re-run is cheap; each writes into
# bench/results/ and is safe to interrupt. The full sweep is several hours on 4 cores
# -- set CLIPS to something small (e.g. 4) for a quick pass.
#
#   CLIPS=4 LANGS=en_us,fr_fr,de_de ./bench/run_all.sh
#   SKIP="fetch requant" ./bench/run_all.sh
#
# Step names: build models kernels requant fetch sweep variants scaling
#
set -euo pipefail

cd "$(dirname "$0")/.."

THREADS="${THREADS:-$(nproc)}"
CLIPS="${CLIPS:-24}"
LANGS="${LANGS:-eu19}"
MODELS="${MODELS:-models/vibeasr}"
SKIP="${SKIP:-}"

VAE="$MODELS/vibeasr-vae-encoder-i8_s.gguf"
LM_BASE="$MODELS/vibeasr-lm-i2_s-embed-q6_k.gguf"   # as released
LM_TIED="$MODELS/vibeasr-lm-i2_s-tied.gguf"         # redundant F16 output.weight dropped
LM_Q5="$MODELS/vibeasr-lm-i2_s-tied-q5k.gguf"       # + tied embedding at Q5_K
LM_Q4="$MODELS/vibeasr-lm-i2_s-tied-q4k.gguf"       # + tied embedding at Q4_K

# Languages the model actually handles, as measured by the full EU sweep. The other
# EU official languages are not in VibeVoice-ASR's training mix and score far worse;
# LANGS=eu19 still runs all of them.
EU_SUPPORTED="${EU_SUPPORTED:-es_419,it_it,de_de,pt_br,en_us,fr_fr}"

skip() { case " $SKIP " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- 0. build -------------------------------------------------------------
if ! skip build; then
  step "build"
  cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build build -j"$THREADS" --target asr_infer kernel_bench requant_lm_head
fi

# --- 1. models ------------------------------------------------------------
if ! skip models; then
  step "download models"
  if [ ! -f "$LM_BASE" ]; then
    python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub
    hf download microsoft/VibeVoice-ASR-BitNet --local-dir "$MODELS" \
        --include '*.gguf' 'tokenizer*' 'config.json'
  else
    echo "already present: $LM_BASE"
  fi
fi

# --- 2. kernel correctness and throughput ---------------------------------
if ! skip kernels; then
  step "kernel correctness + throughput (AVX-512 VNNI vs AVX2)"
  mkdir -p bench/results
  ./build/bin/kernel_bench            | tee bench/results/kernels_auto.txt
  VIBEASR_ISA=avx2 ./build/bin/kernel_bench | tee bench/results/kernels_avx2.txt
fi

# --- 3. squeeze the LM ----------------------------------------------------
if ! skip requant; then
  step "drop the redundant F16 output.weight, then squeeze the tied embedding"
  # output.weight is bit-identical to token_embd (tie_word_embeddings=true) and the
  # loader falls back to token_embd when it is absent, so dropping it is free.
  [ -f "$LM_TIED" ] || ./build/bin/requant_lm_head "$LM_BASE" "$LM_TIED" --drop
  [ -f "$LM_Q5" ]   || ./build/bin/requant_lm_head "$LM_TIED" "$LM_Q5" \
                          --tensor token_embd.weight --type q5_k -t "$THREADS"
  [ -f "$LM_Q4" ]   || ./build/bin/requant_lm_head "$LM_TIED" "$LM_Q4" \
                          --tensor token_embd.weight --type q4_k -t "$THREADS"
  ls -l "$LM_BASE" "$LM_TIED" "$LM_Q5" "$LM_Q4" | awk '{printf "  %8.1f MB  %s\n", $5/1e6, $9}'
  python3 bench/model_report.py "$LM_BASE" "$LM_TIED" "$LM_Q4" "$VAE"
fi

# --- 4. evaluation data ---------------------------------------------------
if ! skip fetch; then
  step "fetch FLEURS slice ($CLIPS clips x $LANGS)"
  python3 bench/fetch_fleurs.py --langs "$LANGS" -n "$CLIPS"
fi

# --- 5a. language coverage: all of EU-19, released model ------------------
if ! skip sweep; then
  step "WER + RTF across $LANGS (released model, AVX-512 VNNI)"
  python3 bench/run_asr.py --langs "$LANGS" -n "$CLIPS" -t "$THREADS" \
      --lm "$LM_BASE" --tag eu19_vnni
fi

# --- 5b. model variants, on the languages the model supports --------------
if ! skip variants; then
  step "WER + RTF per LM variant, on $EU_SUPPORTED"
  python3 bench/run_asr.py --langs "$EU_SUPPORTED" -n "$CLIPS" -t "$THREADS" \
      --lm "$LM_BASE" --tag eu_released
  python3 bench/run_asr.py --langs "$EU_SUPPORTED" -n "$CLIPS" -t "$THREADS" \
      --lm "$LM_TIED" --tag eu_tied
  python3 bench/run_asr.py --langs "$EU_SUPPORTED" -n "$CLIPS" -t "$THREADS" \
      --lm "$LM_Q5"   --tag eu_tied_q5k
  python3 bench/run_asr.py --langs "$EU_SUPPORTED" -n "$CLIPS" -t "$THREADS" \
      --lm "$LM_Q4"   --tag eu_tied_q4k

  step "WER + RTF: AVX2 baseline, for the ISA comparison"
  python3 bench/run_asr.py --langs "$EU_SUPPORTED" -n "$CLIPS" -t "$THREADS" \
      --lm "$LM_BASE" --isa avx2 --tag eu_released_avx2
fi

# --- 6. thread scaling ----------------------------------------------------
if ! skip scaling; then
  step "RTF vs thread count"
  python3 bench/thread_scaling.py --threads "$(seq -s, 1 "$THREADS")" --lang en_us -n 4 \
      | tee bench/results/thread_scaling.txt
fi

# --- 7. tables ------------------------------------------------------------
step "summary"
echo "--- language coverage across $LANGS"
python3 bench/summarize.py eu19_vnni
echo "--- LM variants on the supported subset"
python3 bench/summarize.py eu_released eu_tied eu_tied_q5k eu_tied_q4k --diff
echo "--- AVX-512 VNNI vs AVX2"
python3 bench/summarize.py eu_released eu_released_avx2 --diff
