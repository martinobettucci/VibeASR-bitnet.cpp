#!/usr/bin/env bash
#
# Download the slimmed VibeASR weights from P2Enjoy/VibeVoice-ASR-BitNet-slim.
#
#   ./scripts/download_models.sh [target-dir]     # default: models/vibeasr
#
# Plain curl, no Python dependency; resumable (-C -), so re-running finishes an
# interrupted download instead of starting over. The repo is public — no token.
set -euo pipefail

DEST="${1:-models/vibeasr}"
BASE="https://huggingface.co/P2Enjoy/VibeVoice-ASR-BitNet-slim/resolve/main"

FILES=(
  vibeasr-lm-i2_s-tied.gguf
  vibeasr-vae-encoder-i8_s.gguf
  config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
)

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  echo "==> $f"
  curl -L -C - --fail --progress-bar -o "$DEST/$f" "$BASE/$f"
done

echo
echo "Done. Run:"
echo "  ./build/bin/asr_infer \\"
echo "      --vae-model $DEST/vibeasr-vae-encoder-i8_s.gguf \\"
echo "      --lm-model  $DEST/vibeasr-lm-i2_s-tied.gguf \\"
echo "      --audio input.wav -t 4 --greedy"
