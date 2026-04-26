# Voxtral-WebUI - Quick Start Guide

## Starting the Application

### Option 1: Start Voxtral Implementation (Primary)
```bash
./start-voxtral.sh
```
- Uses: voxtral-mini-3b model
- Port: 7860
- Configuration: configs/default_parameters.yaml

### Option 2: Start Faster Whisper Implementation
```bash
./start-faster-whisper.sh
```
- Uses: large-v3 and other standard models
- Port: 7861
- Configuration: configs/default_parameters_faster_whisper.yaml

### Option 3: Custom Start
```bash
# Generic start with parameters
./start.sh --whisper_type voxtral-mini --server_port 7860
./start.sh --whisper_type faster-whisper --server_port 7861
```

## Stopping the Application
```bash
./stop.sh
```

## Checking Status
```bash
./status.sh
```

## Accessing the Application
- Voxtral: http://localhost:7860/

- Faster Whisper: http://localhost:7861/
## Key Features Available
- File transcription
- YouTube video transcription
- Microphone recording transcription
- Speaker diarization (with HuggingFace token)
- Background music separation
- Multiple subtitle formats (SRT, WebVTT, TXT, LRC)

## Troubleshooting
1. Check logs: `tail -f app.log`
2. Verify status: `./status.sh`
3. Restart if needed: `./stop.sh` then appropriate start command
4. Ensure correct model is selected for implementation

## Important Notes
- Voxtral implementation only works with voxtral-mini-3b model
- Faster Whisper implementation works with standard Whisper models
- HuggingFace token configured for diarization features
- Application logs to app.log file and console