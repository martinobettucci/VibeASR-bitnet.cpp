#!/usr/bin/env bash
#
# Back-to-back RTF comparison: upstream configuration vs this fork, same clips, same
# box, same session. Reports compute-only RTF (VAE encode + prompt + prefill + decode,
# excluding model load) per thread count.
#
#   ./bench/rtf_compare.sh [--clips N] [--lang en_us] [--threads 1,2,4]
#
# Three configurations, to separate what the code changes bought from what the model
# repack bought:
#
#   upstream    AVX2 kernels, row blocks at 4, released LM
#   fork-code   AVX-512 VNNI, row blocks at 32, released LM
#   fork-full   AVX-512 VNNI, row blocks at 32, slim LM (F16 output.weight dropped)
#
# Two ggml builds are needed since the row blocks are compile-time.
set -euo pipefail
cd "$(dirname "$0")/.."

CLIPS=4
LANG_=en_us
THREADS=1,2,4

while [ $# -gt 0 ]; do
  case "$1" in
    --clips)   CLIPS="$2"; shift 2 ;;
    --lang)    LANG_="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
done

VAE=models/vibeasr/vibeasr-vae-encoder-i8_s.gguf
LM_RELEASED=models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf
LM_SLIM=models/vibeasr/vibeasr-lm-i2_s-tied.gguf
DATA="bench/data/$LANG_"

mapfile -t WAVS < <(ls "$DATA"/*.wav | head -"$CLIPS")
AUDIO=$(python3 -c "
import json,sys
rs=[json.loads(l) for l in open('$DATA/refs.jsonl',encoding='utf-8')][:$CLIPS]
print(sum(r['duration'] for r in rs))")

build() {  # build <row_block>
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-DVAE_ROW_BLOCK_SIZE=$1 -DROW_BLOCK_SIZE=$1" \
        -DCMAKE_CXX_FLAGS="-DVAE_ROW_BLOCK_SIZE=$1 -DROW_BLOCK_SIZE=$1" >/dev/null 2>&1
  cmake --build build -j"$(nproc)" --target asr_infer >/dev/null 2>&1
}

measure() {  # measure <label> <lm> <isa> <threads>
  local label=$1 lm=$2 isa=$3 t=$4 total=0
  for w in "${WAVS[@]}"; do
    local out
    out=$(VIBEASR_ISA="$isa" ./build/bin/asr_infer --vae-model "$VAE" --lm-model "$lm" \
          --audio "$w" -t "$t" --greedy 2>&1 >/dev/null)
    local ms
    ms=$(printf '%s' "$out" | sed -n 's/.*\(VAE acoustic encode\|VAE semantic encode\|Prompt build\|LM prefill\|LM decode\):[[:space:]]*\([0-9.]*\).*/\2/p' \
         | paste -sd+ | bc)
    total=$(echo "$total + $ms" | bc)
  done
  printf '%-12s %8s %10.3f\n' "$label" "$t" "$(echo "scale=6; $total / 1000 / $AUDIO" | bc)"
}

echo "$LANG_, $CLIPS clips, ${AUDIO}s audio, compute-only RTF (model load excluded)"
printf '%-12s %8s %10s\n' config threads RTF

build 4
for t in ${THREADS//,/ }; do measure "upstream" "$LM_RELEASED" avx2 "$t"; done

build 32
for t in ${THREADS//,/ }; do measure "fork-code" "$LM_RELEASED" vnni "$t"; done
for t in ${THREADS//,/ }; do measure "fork-full" "$LM_SLIM"     vnni "$t"; done

# Leave the tree at the committed defaults.
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
cmake --build build -j"$(nproc)" --target asr_infer >/dev/null 2>&1
