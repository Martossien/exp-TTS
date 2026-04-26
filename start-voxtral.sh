#!/bin/bash

# Start script for Voxtral-WebUI with voxtral-mini implementation
# This uses the Voxtral-specific model

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project directory
cd "$SCRIPT_DIR"

# Launch the application with voxtral-mini implementation
./start.sh --whisper_type voxtral-mini --server_port 7860 --config configs/default_parameters.yaml