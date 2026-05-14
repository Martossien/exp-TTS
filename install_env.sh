#!/bin/bash
# ============================================================
# install_env.sh — Installation adaptive de exp-STT
#
# Detecte automatiquement l'environnement et s'adapte :
#   - conda (Linux/macOS)
#   - venv/pyenv (Linux/macOS)
#   - venv (Windows/MSYS2/Git Bash)
#
# Usage :
#   ./install_env.sh                     # auto-detect
#   ./install_env.sh --env conda         # forcer conda
#   ./install_env.sh --env venv           # forcer venv
#   ./install_env.sh --env-name myenv    # nom d'environnement personnalise
#   ./install_env.sh --cuda 12.6         # forcer version CUDA PyTorch
#   ./install_env.sh --skip-models       # ne pas telecharger les modeles
#   ./install_env.sh --dry-run            # afficher sans executer
#
# Compatible avec :
#   - Machine boulot : 2x Xeon E5 / 2x RTX 5090 / venv+pyenv / CUDA 12.6
#   - Machine maison  : EPYC 7532 / 8x RTX 3090 / conda / CUDA 13.0
#   - Autres configs Linux avec GPU NVIDIA
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Defaults ---
ENV_TYPE="auto"
ENV_NAME=""
CUDA_VERSION=""
SKIP_MODELS=0
DRY_RUN=0
PYTHON_VER="3.11"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)         ENV_TYPE="$2"; shift 2 ;;
        --env-name)    ENV_NAME="$2"; shift 2 ;;
        --cuda)        CUDA_VERSION="$2"; shift 2 ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --python)      PYTHON_VER="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --env auto|conda|venv    Force env type (default: auto)"
            echo "  --env-name NAME          Custom env name"
            echo "  --cuda VERSION           PyTorch CUDA version (default: auto-detect)"
            echo "  --python VERSION         Python version for conda (default: 3.11)"
            echo "  --skip-models            Skip model downloads"
            echo "  --dry-run                Show what would be done"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Helper functions ---
info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }
fail()  { echo -e "\033[1;31m[FAIL]\033[0m $*"; exit 1; }

run_cmd() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  DRY-RUN: $*"
    else
        "$@"
    fi
}

# --- Detect environment ---
detect_env() {
    if [ "$ENV_TYPE" != "auto" ]; then
        echo "$ENV_TYPE"
        return
    fi
    if command -v conda &>/dev/null && conda info &>/dev/null; then
        echo "conda"
    elif command -v python3 &>/dev/null || command -v python &>/dev/null; then
        echo "venv"
    else
        echo "venv"
    fi
}

detect_cuda() {
    if [ -n "$CUDA_VERSION" ]; then
        echo "$CUDA_VERSION"
        return
    fi
    local driver_cuda
    driver_cuda=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | grep -oP '[\d.]+' | head -1 || true)
    if [ -z "$driver_cuda" ]; then
        warn "nvidia-smi non trouve — CUDA 12.6 par defaut"
        echo "12.6"
        return
    fi
    local major
    major=$(echo "$driver_cuda" | cut -d. -f1)
    case "$major" in
        12) echo "12.6" ;;  # cu126 covers CUDA 12.x
        13) echo "13.0" ;;  # cu130 for CUDA 13.x (installe par uvr)
        *)   echo "12.6" ;;  # fallback
    esac
}

detect_gpu_info() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=count,name,memory.total --format=csv,noheader 2>/dev/null | head -1
    else
        echo "No NVIDIA GPU detected"
    fi
}

# --- Main ---
ENV_TYPE=$(detect_env)
DETECTED_CUDA=$(detect_cuda)

# Default env name based on type
if [ -z "$ENV_NAME" ]; then
    case "$ENV_TYPE" in
        conda) ENV_NAME="exp-stt" ;;
        venv)  ENV_NAME="venv" ;;
    esac
fi

echo "============================================================"
echo " exp-STT — Installation adaptive"
echo "============================================================"
echo " Env type     : $ENV_TYPE"
echo " Env name     : $ENV_NAME"
echo " Python       : $PYTHON_VER"
echo " CUDA detecte : $DETECTED_CUDA"
echo " GPU          : $(detect_gpu_info)"
echo " Skip models  : $SKIP_MODELS"
echo " Dry run      : $DRY_RUN"
echo "============================================================"

# ============================================================
# STEP 1: Create environment
# ============================================================
info "Etape 1/6: Creation de l'environnement"

case "$ENV_TYPE" in
    conda)
        if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
            ok "Environnement conda '${ENV_NAME}' existant — reutilisation"
        else
            info "Creation conda env '${ENV_NAME}' (Python $PYTHON_VER)..."
            run_cmd conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
        fi
        ACTIVATE_CMD="eval \"\$(conda shell.bash hook)\" && conda activate $ENV_NAME"
        PIP_CMD="conda run -n $ENV_NAME pip"
        PYTHON_CMD="conda run -n $ENV_NAME python"
        ;;
    venv)
        if [ -d "$ENV_NAME" ] && [ -f "$ENV_NAME/bin/python" ]; then
            ok "venv '${ENV_NAME}' existant — reutilisation"
        else
            info "Creation venv '${ENV_NAME}'..."
            run_cmd python3 -m venv "$ENV_NAME"
        fi
        ACTIVATE_CMD="source $ENV_NAME/bin/activate"
        PIP_CMD="$ENV_NAME/bin/pip"
        PYTHON_CMD="$ENV_NAME/bin/python"
        ;;
    *)
        fail "Type d'environnement inconnu: $ENV_TYPE"
        ;;
esac

# ============================================================
# STEP 2: Install setuptools compatible
# ============================================================
info "Etape 2/6: setuptools compatible (pour jhj0517-whisper)"
run_cmd $PIP_CMD install 'setuptools<81' wheel --quiet

# ============================================================
# STEP 3: Install PyTorch
# ============================================================
info "Etape 3/6: PyTorch (CUDA $DETECTED_CUDA)"

if [ "$DETECTED_CUDA" = "13.0" ]; then
    # CUDA 13: torch 2.12+cu130 sera installe par uvr, on pose la base cu126
    # qui sera upgrader automatiquement
    info "CUDA 13.0 detecte — installation torch base (sera upgrader par uvr)"
    run_cmd $PIP_CMD install torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 --quiet
elif [ "$DETECTED_CUDA" = "12.6" ]; then
    run_cmd $PIP_CMD install torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 --quiet
else
    run_cmd $PIP_CMD install torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 --quiet
fi

# ============================================================
# STEP 4: Install Python dependencies
# ============================================================
info "Etape 4/6: Dependances Python"

# Core packages
run_cmd $PIP_CMD install faster-whisper==1.1.1 accelerate librosa soundfile jiwer \
    gradio pytubefix 'ruamel.yaml==0.18.6' matplotlib sentencepiece --quiet

# CRITICAL: transformers fixed — do NOT upgrade to 5.x (breaks Cohere ASR)
run_cmd $PIP_CMD install 'transformers==4.57.6' --quiet

# pyannote.audio — 4.x required for community-1 + torchaudio 2.11+
# (3.3.2 in original requirements.txt was a bug — community-1 needs 4.x)
run_cmd $PIP_CMD install 'pyannote.audio>=4.0.0' --quiet

# Git-based packages (need --no-build-isolation for setuptools)
run_cmd $PIP_CMD install --no-build-isolation \
    'git+https://github.com/jhj0517/jhj0517-whisper.git'

run_cmd $PIP_CMD install --no-build-isolation \
    'mistral-common[audio]'

run_cmd $PIP_CMD install --no-build-isolation \
    'git+https://github.com/jhj0517/ultimatevocalremover_api.git'

run_cmd $PIP_CMD install --no-build-isolation \
    'git+https://github.com/jhj0517/pyrubberband.git'

# Qwen ASR
run_cmd $PIP_CMD install qwen-asr --quiet

# ============================================================
# STEP 5: Config files
# ============================================================
info "Etape 5/6: Fichiers de configuration"

for example_file in configs/*.example; do
    target_file="${example_file%.example}"
    if [ ! -f "$target_file" ]; then
        info "Copie: $(basename "$example_file") -> $(basename "$target_file")"
        run_cmd cp "$example_file" "$target_file"
    else
        ok "Config existante conservee: $(basename "$target_file")"
    fi
done

# ============================================================
# STEP 6: Download models (optional)
# ============================================================
if [ "$SKIP_MODELS" -eq 1 ]; then
    warn "Etape 6/6: Telechargement des modeles --ignore par --skip-models"
else
    info "Etape 6/6: Telechargement des modeles (peut prendre ~30 min)"
    info "Les modeles serontmis en cache dans ~/.cache/huggingface/hub/"

    # HF token check
    HF_TOKEN=$($PYTHON_CMD -c "
from huggingface_hub import HfFolder
t = HfFolder.get_token()
print(t if t else '')
" 2>/dev/null || echo "")

    if [ -z "$HF_TOKEN" ]; then
        warn "Pas de token HuggingFace detecte."
        warn "Le modele pyannote/speaker-diarization-community-1 necessite un token."
        warn "Obtenez-le sur https://huggingface.co/settings/tokens"
        warn "Puis: $PYTHON_CMD -c \"from huggingface_hub import HfFolder; HfFolder.save_token('VOTRE_TOKEN')\""
    fi

    MODELS="Systran/faster-whisper-large-v3
CohereLabs/cohere-transcribe-03-2026
Qwen/Qwen3-ASR-1.7B
mistralai/Voxtral-Mini-3B-2507
facebook/nllb-200-1.3B"

    for MODEL in $MODELS; do
        info "Telechargement: $MODEL"
        run_cmd $PYTHON_CMD -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL')
print('OK: $MODEL')
" || warn "Echec ou partiel: $MODEL (peut necessiter un token HF)"
    done

    # pyannote (requires HF token with accepted terms)
    if [ -n "$HF_TOKEN" ]; then
        info "Telechargement: pyannote/speaker-diarization-community-1"
        run_cmd $PYTHON_CMD -c "
from huggingface_hub import HfFolder
from pyannote.audio import Pipeline
token = HfFolder.get_token()
Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=token)
print('OK: pyannote/speaker-diarization-community-1')
" || warn "Echec pyannote — verifiez que vous avez accepte les termes sur https://huggingface.co/pyannote/speaker-diarization-community-1"
    else
        warn "pyannote/speaker-diarization-community-1 ignore (pas de token HF)"
    fi
fi

# ============================================================
# Verification
# ============================================================
echo ""
echo "============================================================"
info "Verification de l'installation..."
echo "============================================================"

$PYTHON_CMD -c "
import torch, transformers, faster_whisper, pyannote.audio, gradio
print(f'  torch:          {torch.__version__} CUDA: {torch.version.cuda}')
print(f'  transformers:   {transformers.__version__}')
print(f'  faster_whisper: {faster_whisper.__version__}')
print(f'  pyannote.audio: {pyannote.audio.__version__}')
print(f'  gradio:         {gradio.__version__}')
print(f'  GPU count:      {torch.cuda.device_count()}')
print(f'  bf16 support:   {torch.cuda.is_bf16_supported()}')
" 2>&1 || warn "Certains imports echouent — verifiez les messages ci-dessus"

echo ""
echo "============================================================"
ok "Installation terminee!"
echo ""
echo "  Pour lancer l'application :"
case "$ENV_TYPE" in
    conda) echo "    conda activate $ENV_NAME && ./start.sh --whisper_type faster-whisper" ;;
    venv)  echo "    source $ENV_NAME/bin/activate && ./start.sh --whisper_type faster-whisper" ;;
esac
echo ""
echo "  Modeles disponibles :"
echo "    --whisper_type faster-whisper   (large-v3, rapide)"
echo "    --whisper_type cohere-asr       (2B, excellent FR, bf16 required)"
echo "    --whisper_type qwen3-asr        (1.7B, 52 langues)"
echo "    --whisper_type voxtral-mini     (3B, Mistral, lent)"
echo "============================================================"