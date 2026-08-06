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
#
# Every configuration is measured REPS times and reported as median plus range. This
# is not optional rigour: a single run on a shared VM can be 2x off, and an earlier
# single-run version of this script produced a 1.55x "speedup" that repeated
# measurement cut to 1.10x. If min and max overlap between two blocks, there is no
# result -- raise REPS or move to a quieter machine.
#
# The profiler is deliberately NOT enabled during timing. VIBEASR_KERNEL_STATS adds
# per-call overhead, and per-call overhead is exactly what the row block changes, so
# it would bias every comparison toward larger blocks.
set -euo pipefail
cd "$(dirname "$0")/.."

MACRO=VAE_ROW_BLOCK_SIZE
WAV=bench/data/en_us/0000.wav
THREADS=$(nproc)
REPS=${REPS:-7}
BLOCKS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --macro)   MACRO="$2"; shift 2 ;;
    --wav)     WAV="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --reps)    REPS="$2"; shift 2 ;;
    *)         BLOCKS+=("$1"); shift ;;
  esac
done
[ ${#BLOCKS[@]} -gt 0 ] || BLOCKS=(4 16 32 64)

VAE=models/vibeasr/vibeasr-vae-encoder-i8_s.gguf
LM=models/vibeasr/vibeasr-lm-i2_s-tied.gguf

echo "sweeping $MACRO on $WAV, $THREADS threads, $REPS reps per value"
printf '%-8s %12s %12s %12s %10s\n' block "median ms" "min" "max" "spread"
for b in "${BLOCKS[@]}"; do
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-D$MACRO=$b" \
        -DCMAKE_CXX_FLAGS="-D$MACRO=$b" >/dev/null 2>&1
  cmake --build build -j"$THREADS" --target asr_infer >/dev/null 2>&1

  times=()
  for _ in $(seq "$REPS"); do
    times+=("$(./build/bin/asr_infer --vae-model "$VAE" --lm-model "$LM" \
        --audio "$WAV" -t "$THREADS" --greedy 2>&1 >/dev/null \
      | sed -n 's/.*\(VAE acoustic encode\|VAE semantic encode\|Prompt build\|LM prefill\|LM decode\):[[:space:]]*\([0-9.]*\).*/\2/p' \
      | paste -sd+ | bc)")
  done
  printf '%s\n' "${times[@]}" | sort -n | awk -v b="$b" '
    {v[NR]=$1}
    END {
      med = (NR%2) ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2
      printf "%-8s %12.1f %12.1f %12.1f %9.1f%%\n", b, med, v[1], v[NR], 100*(v[NR]-v[1])/med
    }'
done

# Leave the tree configured at the committed defaults.
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
cmake --build build -j"$THREADS" --target asr_infer >/dev/null 2>&1
