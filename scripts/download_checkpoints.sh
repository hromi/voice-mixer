#!/usr/bin/env bash
# Downloads the official OpenVoice V2 model checkpoints and lays them out
# under ./checkpoints the way voicelab/config.py expects:
#   checkpoints/converter/{config.json,checkpoint.pth}
#   checkpoints/base_speakers/ses/<key>.pth
#
# The original MyShell S3 zip (checkpoints_v2_0417.zip) has gone 404, so we
# pull the same files from MyShell's official Hugging Face mirror instead:
# https://huggingface.co/myshell-ai/OpenVoiceV2
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINTS_DIR="${VOICELAB_CHECKPOINTS_DIR:-$ROOT_DIR/checkpoints}"
BASE_URL="https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main"

FILES=(
  "converter/config.json"
  "converter/checkpoint.pth"
  "base_speakers/ses/en-au.pth"
  "base_speakers/ses/en-br.pth"
  "base_speakers/ses/en-default.pth"
  "base_speakers/ses/en-india.pth"
  "base_speakers/ses/en-newest.pth"
  "base_speakers/ses/en-us.pth"
  "base_speakers/ses/es.pth"
  "base_speakers/ses/fr.pth"
  "base_speakers/ses/jp.pth"
  "base_speakers/ses/kr.pth"
  "base_speakers/ses/zh.pth"
)

for rel_path in "${FILES[@]}"; do
  dest="$CHECKPOINTS_DIR/$rel_path"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then
    echo "skip (already present): $rel_path"
    continue
  fi
  echo "Downloading $rel_path ..."
  curl -L --fail -o "$dest" "$BASE_URL/$rel_path"
done

echo "Done. Now install ML dependencies and MeloTTS's unidic data:"
echo "  pip install -r requirements-ml.txt"
echo "  python -m unidic download"
