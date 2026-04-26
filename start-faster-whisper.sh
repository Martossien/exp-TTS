#!/bin/bash

# Start script for Voxtral-WebUI with faster-whisper implementation
# This allows using standard Whisper models like large-v3

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project directory
cd "$SCRIPT_DIR"

# Launch the application with faster-whisper implementation
./start.sh --whisper_type faster-whisper --server_port 7861 --config configs/default_parameters_faster_whisper.yaml