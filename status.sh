#!/bin/bash

# Status script for Voxtral-WebUI
# This script shows the status of the running application

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project directory
cd "$SCRIPT_DIR"

# Check if PID file exists
if [ ! -f "$SCRIPT_DIR/pids/voxtral.pid" ]; then
    echo "Status: Voxtral-WebUI is not running (no PID file found)"
    echo "Log file size: $(du -h app.log 2>/dev/null | cut -f1 || echo '0B')"
    exit 0
fi

# Read the PID
PID=$(cat "$SCRIPT_DIR/pids/voxtral.pid")

# Check if the process is running
if ps -p $PID > /dev/null; then
    # Try to determine which whisper implementation is being used
    WHISPER_TYPE="unknown"
    if ps -p $PID -o args= | grep -q "faster-whisper"; then
        WHISPER_TYPE="faster-whisper"
    elif ps -p $PID -o args= | grep -q "insanely_fast_whisper"; then
        WHISPER_TYPE="insanely_fast_whisper"
    elif ps -p $PID -o args= | grep -q "whisper"; then
        WHISPER_TYPE="whisper"
    elif ps -p $PID -o args= | grep -q "voxtral-mini"; then
        WHISPER_TYPE="voxtral-mini"
    fi
    
    PORT=7860
    PORT_MATCH=$(ps -p $PID -o args= | grep -o "server_port [0-9]*" | cut -d' ' -f2)
    if [ ! -z "$PORT_MATCH" ]; then
        PORT=$PORT_MATCH
    fi
    
    echo "Status: Voxtral-WebUI is running with PID $PID"
    echo "Whisper implementation: $WHISPER_TYPE"
    echo "Access URL: http://0.0.0.0:$PORT"
    echo "Log file size: $(du -h app.log 2>/dev/null | cut -f1 || echo '0B')"
else
    echo "Status: Voxtral-WebUI is not running (stale PID file)"
    echo "Log file size: $(du -h app.log 2>/dev/null | cut -f1 || echo '0B')"
    echo "Note: You may want to run ./stop.sh to clean up the stale PID file"
fi