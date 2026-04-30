#!/bin/bash
# Lancer le serveur vLLM pour Voxtral-Mini-4B-Realtime-2602
# Port 8000 (different du 8080 utilise par la LLM principale)
# GPU unique (le modele fait ~16 Go < 32 Go par GPU)
#
# Correction expert #4 : --served-model-name pour que le client puisse
# envoyer le repo_id meme si le serveur charge un chemin local.

set -euo pipefail

MODEL_PATH="${1:-/root/models/Voxtral-Mini-4B-Realtime-2602}"
PORT="${2:-8000}"
GPU_ID="${3:-1}"
SERVED_NAME="${4:-mistralai/Voxtral-Mini-4B-Realtime-2602}"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

if [ -f /root/proxy.sh ]; then
    source /root/proxy.sh
fi

source /root/vllm-qwen36-env/bin/activate

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERREUR : Modele non trouve dans $MODEL_PATH"
    echo "Telechargez-le d'abord :"
    echo "  source /root/vllm-qwen36-env/bin/activate"
    echo "  huggingface-cli download mistralai/Voxtral-Mini-4B-Realtime-2602 \\"
    echo "    --local-dir /root/models/Voxtral-Mini-4B-Realtime-2602"
    exit 1
fi

if fuser "$PORT/tcp" &>/dev/null; then
    echo "ERREUR : Port $PORT deja occupe."
    echo "Utilisez : bash scripts/stop_voxtral_realtime_vllm.sh"
    exit 1
fi

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null || echo "0")
if [ "$FREE_MB" -lt 18000 ]; then
    echo "⚠️  GPU $GPU_ID : seulement $(( FREE_MB / 1024 )) Go libres (< 18 Go recommandes)"
    echo "    Pensez a liberer la VRAM avant de lancer."
    echo "    Le WebUI gere cela automatiquement via _ensure_vram()."
fi

echo "==> Lancement vLLM Voxtral Realtime sur GPU $GPU_ID, port $PORT"
echo "    Modele      : $MODEL_PATH"
echo "    Served name : $SERVED_NAME"
echo "    Log         : /var/log/vllm_voxtral_realtime.log"

VLLM_DISABLE_COMPILE_CACHE=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --compilation_config '{"cudagraph_mode": "PIECEWISE"}' \
    > /var/log/vllm_voxtral_realtime.log 2>&1 &

PID=$!
echo "    PID         : $PID"
echo "$PID" > /tmp/vllm_voxtral_realtime.pid

echo -n "==> Attente du health check"
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo " OK (${i}s)"
        echo ""
        echo "Serveur pret : ws://localhost:$PORT/v1/realtime"
        exit 0
    fi
    echo -n "."
    sleep 2
done
echo " TIMEOUT"
echo "Consultez les logs : tail -f /var/log/vllm_voxtral_realtime.log"
exit 1
