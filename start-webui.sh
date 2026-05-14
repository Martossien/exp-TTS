#!/bin/bash
# start-webui.sh — Quick start with auto-env detection
# Set CONDA_ENV or VENV_NAME to override defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-exp-stt}"
VENV_NAME="${VENV_NAME:-venv}"

if [ -d "$VENV_NAME" ]; then
    source "$VENV_NAME/bin/activate"
    python app.py "$@"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    python app.py "$@"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    python app.py "$@"
elif command -v pyenv &>/dev/null && PYENV_PYTHON=$(pyenv which python 2>/dev/null); then
    "$PYENV_PYTHON" app.py "$@"
else
    python app.py "$@"
fi