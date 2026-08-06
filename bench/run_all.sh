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
set -euo pipefail

cd "$(dirname "$0")/.."

THREADS="${THREADS:-$(nproc)}"
CLIPS="${CLIPS:-24}"
LANGS="${LANGS:-eu19}"
MODELS="${MODELS:-models/vibeasr}"
SKIP="${SKIP:-}"

VAE="$MODELS/vibeasr-vae-encoder-i8_s.gguf"
LM_BASE="$MODELS/vibeasr-lm-i2_s-embed-q6_k.gguf"
LM_Q6="$MODELS/vibeasr-lm-i2_s-head-q6_k.gguf"
LM_Q4="$MODELS/vibeasr-lm-i2_s-head-q4_k.gguf"

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

# --- 3. re-quantise the F16 lm_head ---------------------------------------
if ! skip requant; then
  step "re-quantise output.weight (F16 -> Q6_K, Q4_K)"
  [ -f "$LM_Q6" ] || ./build/bin/requant_lm_head "$LM_BASE" "$LM_Q6" --type q6_k -t "$THREADS"
  [ -f "$LM_Q4" ] || ./build/bin/requant_lm_head "$LM_BASE" "$LM_Q4" --type q4_k -t "$THREADS"
  ls -l "$LM_BASE" "$LM_Q6" "$LM_Q4" | awk '{printf "  %8.1f MB  %s\n", $5/1e6, $9}'
fi

# --- 4. evaluation data ---------------------------------------------------
if ! skip fetch; then
  step "fetch FLEURS slice ($CLIPS clips x $LANGS)"
  python3 bench/fetch_fleurs.py --langs "$LANGS" -n "$CLIPS"
fi

# --- 5. WER / RTF sweeps --------------------------------------------------
if ! skip sweep; then
  step "WER + RTF: released model, AVX-512 VNNI"
  python3 bench/run_asr.py --langs "$LANGS" -n "$CLIPS" -t "$THREADS" \
      --lm "$LM_BASE" --tag eu19_vnni

  step "WER + RTF: released model, AVX2 baseline"
  python3 bench/run_asr.py --langs "$LANGS" -n "$CLIPS" -t "$THREADS" \
      --lm "$LM_BASE" --isa avx2 --tag eu19_avx2

  step "WER + RTF: lm_head Q6_K"
  python3 bench/run_asr.py --langs "$LANGS" -n "$CLIPS" -t "$THREADS" \
      --lm "$LM_Q6" --tag eu19_head_q6k

  step "WER + RTF: lm_head Q4_K"
  python3 bench/run_asr.py --langs "$LANGS" -n "$CLIPS" -t "$THREADS" \
      --lm "$LM_Q4" --tag eu19_head_q4k
fi

# --- 6. thread scaling ----------------------------------------------------
if ! skip scaling; then
  step "RTF vs thread count"
  python3 bench/thread_scaling.py --threads "$(seq -s, 1 "$THREADS")" --lang en_us -n 4 \
      | tee bench/results/thread_scaling.txt
fi

# --- 7. tables ------------------------------------------------------------
step "summary"
python3 bench/summarize.py eu19_vnni
python3 bench/summarize.py eu19_vnni eu19_avx2 eu19_head_q6k eu19_head_q4k --diff
