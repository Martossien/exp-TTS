#!/bin/bash

# Start script for Voxtral-WebUI with Qwen3-ASR implementation.
# The 1.7B model is the preferred/default Qwen model; 0.6B remains available in the UI.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

./start.sh --whisper_type qwen3-asr --server_port 7862 --config configs/default_parameters.yaml
