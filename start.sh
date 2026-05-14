#!/bin/bash

# Start script for Voxtral-WebUI
# This script launches the application on 0.0.0.0:7860 and logs all output to app.log

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project directory
cd "$SCRIPT_DIR"

# Create PID file directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/pids"

# Check if the application is already running
if [ -f "$SCRIPT_DIR/pids/voxtral.pid" ]; then
    PID=$(cat "$SCRIPT_DIR/pids/voxtral.pid")
    if ps -p $PID > /dev/null; then
        echo "Voxtral-WebUI is already running with PID $PID"
        exit 1
    else
        # Remove stale PID file
        rm "$SCRIPT_DIR/pids/voxtral.pid"
    fi
fi

# Select the Python runtime explicitly.
# Set CONDA_ENV or VENV_NAME to override the default environment name.
CONDA_ENV="${CONDA_ENV:-exp-stt}"
VENV_NAME="${VENV_NAME:-venv}"
if [ -d "$VENV_NAME" ]; then
    source "$VENV_NAME/bin/activate"
    PYTHON_BIN="python"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    PYTHON_BIN="python"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    PYTHON_BIN="python"
elif command -v pyenv &>/dev/null && PYENV_PYTHON=$(pyenv which python 2>/dev/null); then
    PYTHON_BIN="$PYENV_PYTHON"
else
    echo "Warning: No virtual environment found. Using python from PATH."
    PYTHON_BIN="python"
fi

# Parse command line arguments
WHISPER_TYPE="voxtral-mini"
SERVER_PORT=7860
CONFIG_FILE="configs/default_parameters.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --whisper_type)
            WHISPER_TYPE="$2"
            shift 2
            ;;
        --server_port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--whisper_type TYPE] [--server_port PORT] [--config CONFIG_FILE]"
            echo "  --whisper_type: voxtral-mini (default), qwen3-asr, faster-whisper, whisper, insanely_fast_whisper"
            echo "  --server_port: Port number (default: 7860)"
            echo "  --config: Configuration file (default: configs/default_parameters.yaml)"
            exit 1
            ;;
    esac
done

# PyTorch 2.11+CUDA 13.0 requires libnvrtc-builtins.so.13.0 and libnvrtc.so.13
# for JIT CUDA kernel compilation (torchaudio vmap+fft, diarization, etc.).
# These ship inside the nvidia/cu13 Python package — add them explicitly.
NVRTC_CU13_LIB=$(python -c "import nvidia.cuda_runtime as _; print(_.__path__[0].replace('cuda_runtime','cu13') + '/lib')" 2>/dev/null)
if [ -z "$NVRTC_CU13_LIB" ] || [ ! -d "$NVRTC_CU13_LIB" ]; then
    NVRTC_CU13_LIB=$(python -c "import site, os; print(os.path.join(site.getsitepackages()[0], 'nvidia', 'cu13', 'lib'))" 2>/dev/null)
fi
if [ -d "$NVRTC_CU13_LIB" ]; then
    export LD_LIBRARY_PATH="${NVRTC_CU13_LIB}:${LD_LIBRARY_PATH}"
    echo "Added NVRTC cu13 libs to LD_LIBRARY_PATH: $NVRTC_CU13_LIB"
fi

# Reduce CUDA allocator fragmentation on long transcription jobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "Using PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# Default to online Hugging Face access so models can be downloaded on first run.
# Set HF_HUB_OFFLINE=1 to prevent any network access (requires pre-downloaded models).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
echo "Using HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

# Log VRAM state before launch so we know what's available.
if command -v nvidia-smi &>/dev/null; then
    echo "GPU VRAM state at startup:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
        --format=csv,noheader,nounits \
    | while IFS=',' read -r idx name used free total; do
        used=$(echo "$used" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
        total=$(echo "$total" | tr -d ' '); name=$(echo "$name" | xargs)
        echo "  GPU $idx ($name): ${used}MiB used / ${total}MiB total (${free}MiB free)"
        if [ "$free" -lt 10240 ]; then
            echo "  WARNING: GPU $idx has less than 10 GB free — inference may fail or be slow"
        fi
    done
fi

# Launch the application in the background. setsid detaches it from the
# caller so it keeps running after automation or a non-interactive shell exits.
echo "Starting Voxtral-WebUI with whisper_type=$WHISPER_TYPE on port $SERVER_PORT..."
echo "Using configuration file: $CONFIG_FILE"
echo "Using Python: $PYTHON_BIN"
setsid "$PYTHON_BIN" app.py --whisper_type "$WHISPER_TYPE" --server_name 0.0.0.0 --server_port "$SERVER_PORT" --inbrowser False >> app.log 2>&1 < /dev/null &

# Save the PID
echo $! > "$SCRIPT_DIR/pids/voxtral.pid"

sleep 2
if ! ps -p "$!" > /dev/null; then
    echo "Voxtral-WebUI failed to stay running. Last app.log lines:"
    tail -n 80 app.log
    rm -f "$SCRIPT_DIR/pids/voxtral.pid"
    exit 1
fi

echo "Voxtral-WebUI started with PID $!"
echo "Logs are being written to app.log"
echo "Access the application at http://0.0.0.0:$SERVER_PORT"
