#!/usr/bin/env bash
#
# Measure VAE encode time against VAE_ROW_BLOCK_SIZE.
#
#   ./bench/row_block_sweep.sh [clip.wav] [threads] [block sizes...]
#
# The block size is the number of activation rows per vec_dot call in the I8_S GEMM.
# It does not change any result -- only how much per-call overhead is amortised -- so
# this sweep is purely a speed measurement. Rebuilds ggml once per value.
set -euo pipefail
cd "$(dirname "$0")/.."

WAV="${1:-bench/data/en_us/0000.wav}"
THREADS="${2:-$(nproc)}"
shift 2 2>/dev/null || true
BLOCKS=("${@:-4 8 16 32 64}")
# shellcheck disable=SC2206
BLOCKS=(${BLOCKS[@]})

VAE=models/vibeasr/vibeasr-vae-encoder-i8_s.gguf
LM=models/vibeasr/vibeasr-lm-i2_s-tied.gguf

printf '%-8s %12s %12s %12s %10s\n' block "VAE-ac ms" "VAE-sem ms" "prefill ms" "calls"
for b in "${BLOCKS[@]}"; do
  # The blocking loop lives in ggml-aarch64.c, a C file, so the define has to reach
  # the C compiler too -- setting only CXX flags silently measures the header default.
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-DVAE_ROW_BLOCK_SIZE=$b" \
        -DCMAKE_CXX_FLAGS="-DVAE_ROW_BLOCK_SIZE=$b" >/dev/null 2>&1
  cmake --build build -j"$THREADS" --target asr_infer >/dev/null 2>&1

  out=$(VIBEASR_KERNEL_STATS=1 ./build/bin/asr_infer --vae-model "$VAE" --lm-model "$LM" \
        --audio "$WAV" -t "$THREADS" --greedy 2>&1 >/dev/null)
  ac=$(printf '%s' "$out"  | sed -n 's/.*VAE acoustic encode:\s*\([0-9.]*\).*/\1/p')
  se=$(printf '%s' "$out"  | sed -n 's/.*VAE semantic encode:\s*\([0-9.]*\).*/\1/p')
  pf=$(printf '%s' "$out"  | sed -n 's/.*LM prefill:\s*\([0-9.]*\).*/\1/p')
  calls=$(printf '%s' "$out" | awk '/i8_s /{gsub(",","",$0); c+=$(NF-2)} END{printf "%d", c}')
  printf '%-8s %12s %12s %12s %10s\n' "$b" "$ac" "$se" "$pf" "$calls"
done

# Leave the tree configured at the committed default.
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
cmake --build build -j"$THREADS" --target asr_infer >/dev/null 2>&1
