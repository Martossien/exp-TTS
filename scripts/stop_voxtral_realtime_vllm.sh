#!/bin/bash
# Arreter le serveur vLLM Voxtral Realtime (port 8000)
#
# Correction expert #7 : moins agressif.
# - tue d'abord le PID file
# - tue le port 8000
# - ne tue tous les workers VLLM orphelins qu'avec --kill-orphans

set -euo pipefail

PORT="${1:-8000}"
KILL_ORPHANS=false
if [ "${2:-}" = "--kill-orphans" ]; then
    KILL_ORPHANS=true
fi

echo "==> Arret du serveur vLLM Voxtral Realtime (port $PORT)..."

killed=false

# 1. PID file
if [ -f /tmp/vllm_voxtral_realtime.pid ]; then
    PID=$(cat /tmp/vllm_voxtral_realtime.pid)
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null && echo "    SIGTERM -> PID $PID (PID file)"
        killed=true
    fi
    rm -f /tmp/vllm_voxtral_realtime.pid
fi

sleep 2

# 2. fuser sur le port
fuser -k "$PORT/tcp" 2>/dev/null && echo "    fuser -k $PORT/tcp" && killed=true

# 3. Workers orphelins : uniquement avec --kill-orphans
if $KILL_ORPHANS; then
    pids=$(pgrep -f "VLLM::(EngineCore|Worker)" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "    Workers VLLM orphelins : $pids"
        echo "$pids" | xargs -r kill 2>/dev/null
        sleep 3
        pids2=$(pgrep -f "VLLM::(EngineCore|Worker)" 2>/dev/null || true)
        if [ -n "$pids2" ]; then
            echo "$pids2" | xargs -r kill -9 2>/dev/null
            echo "    SIGKILL -> $pids2"
        fi
    fi
else
    orphans=$(pgrep -f "VLLM::(EngineCore|Worker)" 2>/dev/null || true)
    if [ -n "$orphans" ]; then
        echo "    Workers VLLM orphelins detectes : $orphans"
        echo "    (non touches. Utilisez --kill-orphans pour les tuer)"
    fi
fi

echo "==> Arret termine."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || true
