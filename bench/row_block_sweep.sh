#!/usr/bin/env bash
#
# Measure inference time against a GEMM row-blocking macro.
#
#   ./bench/row_block_sweep.sh [--macro NAME] [--wav FILE] [--threads N] [blocks...]
#
#   ./bench/row_block_sweep.sh                          # VAE_ROW_BLOCK_SIZE, 4..64
#   ./bench/row_block_sweep.sh --macro ROW_BLOCK_SIZE   # the LM's prefill equivalent
#
# The row block is how many activation rows go into one vec_dot call. It changes no
# results -- only how much per-call prologue, epilogue and horizontal reduction get
# amortised -- so this is purely a speed measurement.
#
# The define has to reach the C compiler as well as the C++ one: the blocking loops
# live in ggml-aarch64.c, so setting only CMAKE_CXX_FLAGS silently rebuilds against
# the header default and measures nothing.
#
# ggml is rebuilt once per value, so budget a few minutes per block size.
set -euo pipefail
cd "$(dirname "$0")/.."

MACRO=VAE_ROW_BLOCK_SIZE
WAV=bench/data/en_us/0000.wav
THREADS=$(nproc)
BLOCKS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --macro)   MACRO="$2"; shift 2 ;;
    --wav)     WAV="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    *)         BLOCKS+=("$1"); shift ;;
  esac
done
[ ${#BLOCKS[@]} -gt 0 ] || BLOCKS=(4 16 32 64)

VAE=models/vibeasr/vibeasr-vae-encoder-i8_s.gguf
LM=models/vibeasr/vibeasr-lm-i2_s-tied.gguf

echo "sweeping $MACRO on $WAV, $THREADS threads"
printf '%-8s %12s %12s %12s %12s %12s\n' block "VAE-ac ms" "VAE-sem ms" "prefill ms" "decode ms" calls
for b in "${BLOCKS[@]}"; do
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-D$MACRO=$b" \
        -DCMAKE_CXX_FLAGS="-D$MACRO=$b" >/dev/null 2>&1
  cmake --build build -j"$THREADS" --target asr_infer >/dev/null 2>&1

  out=$(VIBEASR_KERNEL_STATS=1 ./build/bin/asr_infer --vae-model "$VAE" --lm-model "$LM" \
        --audio "$WAV" -t "$THREADS" --greedy 2>&1 >/dev/null)
  get() { printf '%s' "$out" | sed -n "s/.*$1:[[:space:]]*\([0-9.]*\).*/\1/p" | head -1; }
  calls=$(printf '%s' "$out" | awk '/i8_s |i2_s /{c+=$(NF-2)} END{printf "%d", c}')
  printf '%-8s %12s %12s %12s %12s %12s\n' "$b" \
      "$(get 'VAE acoustic encode')" "$(get 'VAE semantic encode')" \
      "$(get 'LM prefill')" "$(get 'LM decode')" "$calls"
done

# Leave the tree configured at the committed defaults.
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
cmake --build build -j"$THREADS" --target asr_infer >/dev/null 2>&1
