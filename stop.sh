#!/bin/bash

# Stop script for Voxtral-WebUI
# Kills the application, then checks and cleans up GPU VRAM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/pids/voxtral.pid"
LOG_TAG="[STOP $(date '+%Y-%m-%d %H:%M:%S')]"

# ── helpers ────────────────────────────────────────────────────────────────

log() { echo "$LOG_TAG $*"; }

vram_report() {
    # Print free/used VRAM for every GPU, return total used (MiB)
    if ! command -v nvidia-smi &>/dev/null; then
        log "nvidia-smi not found — skipping VRAM report"
        return
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
        --format=csv,noheader,nounits \
    | while IFS=',' read -r idx name used free total; do
        used=$(echo "$used" | tr -d ' ')
        free=$(echo "$free" | tr -d ' ')
        total=$(echo "$total" | tr -d ' ')
        name=$(echo "$name" | xargs)
        log "  GPU $idx ($name): ${used}MiB used / ${total}MiB total (${free}MiB free)"
    done
}

cleanup_vram() {
    # Kill any processes still holding CUDA device handles after the main process died.
    if ! command -v nvidia-smi &>/dev/null; then
        return
    fi

    # Collect PIDs that still have an open handle on any /dev/nvidia* device.
    local gpu_pids
    gpu_pids=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u)

    if [ -z "$gpu_pids" ]; then
        log "No leftover processes holding GPU memory."
        return
    fi

    log "GPU processes still alive after stop:"
    for pid in $gpu_pids; do
        local cmdline
        cmdline=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")
        log "  PID $pid ($cmdline)"
    done

    log "Sending SIGTERM to leftover GPU processes..."
    for pid in $gpu_pids; do
        kill "$pid" 2>/dev/null && log "  SIGTERM → $pid"
    done

    sleep 2

    # Anything still alive gets SIGKILL
    local survivors
    survivors=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u)
    if [ -n "$survivors" ]; then
        log "Force-killing remaining GPU processes with SIGKILL..."
        for pid in $survivors; do
            kill -9 "$pid" 2>/dev/null && log "  SIGKILL → $pid"
        done
        sleep 1
    fi

    log "VRAM state after cleanup:"
    vram_report
}

# ── main ───────────────────────────────────────────────────────────────────

if [ ! -f "$PID_FILE" ]; then
    log "Voxtral-WebUI is not running (no PID file found)"
    # Still run VRAM cleanup in case of a stale GPU process from a crash.
    log "Checking for stale GPU processes anyway..."
    cleanup_vram
    exit 0
fi

PID=$(cat "$PID_FILE")

log "VRAM state before stop:"
vram_report

if ps -p "$PID" > /dev/null 2>&1; then
    log "Stopping Voxtral-WebUI (PID: $PID)..."
    kill "$PID"

    # Wait up to 10 s for graceful exit
    for i in $(seq 1 10); do
        sleep 1
        if ! ps -p "$PID" > /dev/null 2>&1; then
            log "Process exited gracefully after ${i}s."
            break
        fi
    done

    if ps -p "$PID" > /dev/null 2>&1; then
        log "Process did not stop gracefully — force-killing (SIGKILL)..."
        kill -9 "$PID" 2>/dev/null
        sleep 1
    fi

    rm -f "$PID_FILE"
    log "Voxtral-WebUI stopped (PID: $PID)"
else
    log "PID $PID not found — removing stale PID file"
    rm -f "$PID_FILE"
fi

# Give the OS a moment to release the CUDA context before probing
sleep 2

log "Cleaning up GPU VRAM..."
cleanup_vram

log "Done."
