#!/bin/bash
set -euo pipefail

# ============================================================
# install_conda.sh — Conda-based installation for exp-STT
#
# Usage:
#   ./install_conda.sh                    # defaults
#   ./install_conda.sh --env-name myenv   # custom env name
#   ./install_conda.sh --python 3.12      # custom Python version
#   ./install_conda.sh --skip-models     # skip HF model downloads
#   ./install_conda.sh --dry-run          # print commands without running
#
# Compatible: Linux/macOS with CUDA GPU + conda
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Defaults ---
ENV_NAME="exp-stt"
PYTHON_VER="3.11"
SKIP_MODELS=0
DRY_RUN=0

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)    ENV_NAME="$2"; shift 2 ;;
        --python)      PYTHON_VER="$2"; shift 2 ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --env-name NAME     Conda env name (default: exp-stt)"
            echo "  --python VERSION    Python version (default: 3.11)"
            echo "  --skip-models       Skip HuggingFace model downloads"
            echo "  --dry-run           Print commands without running"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY-RUN] $*"
    else
        "$@"
    fi
}

echo "============================================================"
echo " exp-STT — Conda Installation"
echo " Env: $ENV_NAME  Python: $PYTHON_VER"
echo "============================================================"

# --- 1. Check conda ---
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Anaconda or Miniconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# --- 2. Check CUDA ---
CUDA_VERSION=""
if command -v nvidia-smi &>/dev/null; then
    echo "Detected GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' | head -1)
    echo "CUDA driver version: ${CUDA_VERSION:-unknown}"
else
    echo "WARNING: nvidia-smi not found. No NVIDIA GPU detected."
    echo "  CPU-only mode will be used, but inference will be very slow."
fi

# --- 3. Create conda environment ---
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists. Reusing."
else
    echo "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VER}..."
    run conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
fi

# --- 4. Activate environment ---
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "Python: $(python --version)"
echo "Pip: $(pip --version | head -1)"

# --- 5. Install compatible setuptools ---
echo ""
echo "Installing setuptools (<81, required for jhj0517-whisper)..."
run pip install 'setuptools<81' wheel --quiet

# --- 6. Install PyTorch ---
# Detect CUDA version for PyTorch index URL
if [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    case "$CUDA_MAJOR" in
        12) PYTORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
        13) PYTORCH_INDEX="https://download.pytorch.org/whl/cu130" ;;
        *)  PYTORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
    esac
else
    PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi
echo ""
echo "Installing PyTorch (index: $PYTORCH_INDEX)..."
run pip install torch torchaudio --index-url "$PYTORCH_INDEX" --quiet

# --- 7. Install core dependencies ---
echo ""
echo "Installing core dependencies..."
run pip install faster-whisper==1.1.1 accelerate librosa soundfile jiwer \
    gradio pytubefix 'ruamel.yaml==0.18.6' 'pyannote.audio>=4.0.0' \
    matplotlib sentencepiece --quiet

# --- 8. Install transformers (CRITICAL: must stay <= 4.57.x, 5.x breaks Cohere ASR) ---
echo ""
echo "Installing transformers==4.57.6 (CRITICAL: do NOT upgrade to 5.x)..."
run pip install 'transformers==4.57.6' --quiet

# --- 9. Install git dependencies (with --no-build-isolation for setuptools compat) ---
echo ""
echo "Installing git dependencies..."
run pip install --no-build-isolation 'git+https://github.com/jhj0517/jhj0517-whisper.git'
run pip install --no-build-isolation 'mistral-common[audio]'
run pip install --no-build-isolation 'git+https://github.com/jhj0511/ultimatevocalremover_api.git'
run pip install --no-build-isolation 'git+https://github.com/jhj0511/pyrubberband.git'

# --- 10. Install qwen-asr ---
echo ""
echo "Installing qwen-asr..."
run pip install qwen-asr --quiet

# --- 11. Copy config examples if configs don't exist ---
if ls configs/*.example &>/dev/null; then
    for example_file in configs/*.example; do
        target_file="${example_file%.example}"
        if [ ! -f "$target_file" ]; then
            echo "Copying $example_file -> $target_file"
            run cp "$example_file" "$target_file"
        else
            echo "Config already exists: $target_file"
        fi
    done
fi

# --- 12. Download models (optional) ---
if [ "$SKIP_MODELS" -eq 0 ]; then
    echo ""
    echo "Downloading models (this may take a while)..."
    run python -c "
import torch
from faster_whisper import WhisperModel
print('Downloading faster-whisper large-v3...')
WhisperModel('large-v3', device='cpu', compute_type='int8',
             download_root='models/Whisper/faster-whisper')
print('faster-whisper large-v3 downloaded.')
"
    echo ""
    echo "HuggingFace models (Voxtral, Qwen3-ASR, Cohere, pyannote)"
    echo "will be downloaded on first use. For offline use, run:"
    echo "  python -c \\\"from huggingface_hub import snapshot_download; \\"
    echo "    snapshot_download('mistralai/Voxtral-Mini-3B-2507'); \\"
    echo "    snapshot_download('Qwen/Qwen3-ASR-1.7B'); \\"
    echo "    snapshot_download('CohereLabs/cohere-transcribe-03-2026')\\\""
    echo ""
    echo "For pyannote, set your HF token first:"
    echo "  python -c \\\"from huggingface_hub import HfFolder; HfFolder.save_token('YOUR_TOKEN')\\\""
else
    echo ""
    echo "Skipping model downloads (--skip-models)."
    echo "Models will be downloaded on first use."
fi

# --- 13. Verify installation ---
echo ""
echo "=== Installation Verification ==="
run python -c "
import torch
print(f'torch: {torch.__version__}  CUDA: {torch.version.cuda or \"CPU only\"}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
import transformers; print(f'transformers: {transformers.__version__}')
import faster_whisper; print(f'faster_whisper: {faster_whisper.__version__}')
import pyannote.audio; print(f'pyannote.audio: {pyannote.audio.__version__}')
import gradio; print(f'gradio: {gradio.__version__}')
print('All imports OK!')
" || echo "Verification failed. Check errors above."

echo ""
echo "============================================================"
echo " Installation complete!"
echo " To launch: conda activate $ENV_NAME && ./start.sh"
echo "============================================================"