#!/usr/bin/env bash
# Sets up an isolated venv for Qwen3-TTS (German/Russian/Portuguese/Italian
# cloning support), separate from the project's main Python environment.
#
# Why: qwen-tts uses `X | None` union-type syntax at module level, which
# requires Python 3.10+. This project's main interpreter may be older
# (e.g. Python 3.9), so Qwen3-TTS runs as a subprocess in its own venv
# instead — see voicelab/engine.py's _QwenWorkerProcess and
# voicelab/qwen_worker.py. Also requires a CUDA GPU; there's no CPU path.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VOICELAB_QWEN_VENV_DIR:-$ROOT_DIR/.venv-qwen}"

PYTHON_BIN="${QWEN_PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "No Python 3.10+ interpreter found (tried python3.12/3.11/3.10)." >&2
  echo "Install one, or set QWEN_PYTHON_BIN to its path, then rerun." >&2
  exit 1
fi

echo "Using $($PYTHON_BIN --version) at $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "Installing torch (CUDA 12.8 build) + qwen-tts into $VENV_DIR ..."
"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu128
"$VENV_DIR/bin/pip" install -q qwen-tts

# qwen-tts's own dependency resolution can occasionally pull a torchaudio
# build mismatched against the CUDA torch build just installed above
# (ABI errors like "undefined symbol" / "cannot open shared object file"
# at import time) — force-reinstall it from the same index to be sure.
"$VENV_DIR/bin/pip" install -q --force-reinstall --no-deps torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "Verifying import..."
"$VENV_DIR/bin/python" -c "from qwen_tts import Qwen3TTSModel; print('qwen_tts OK')"

echo "Done. The Qwen3-TTS checkpoint (~4GB) downloads automatically from"
echo "Hugging Face the first time you clone in German/Russian/Portuguese/Italian."
