# Voxtral-WebUI - Project Structure Summary

## Critical Files

### Main Application
- `app.py` - Entry point and main Gradio interface
- `TECHNICAL_DOCUMENTATION.md` - This documentation
- `requirements.txt` - Python dependencies

### Configuration Files
- `configs/default_parameters.yaml` - Default settings for Voxtral implementation
- `configs/default_parameters_faster_whisper.yaml` - Default settings for Faster Whisper

### Deployment Scripts
- `start.sh` - Generic start script
- `start-voxtral.sh` - Start Voxtral implementation (port 7860)
- `start-faster-whisper.sh` - Start Faster Whisper implementation (port 7861)
- `stop.sh` - Stop application
- `status.sh` - Check application status

## Modified Files Summary

### Core Fixes
1. `modules/whisper/voxtral_whisper_inference.py` - Fixed typo in method name
2. `modules/diarize/diarize_pipeline.py` - Added error handling for None model
3. `modules/diarize/diarizer.py` - Fixed segment processing and error handling
4. `modules/whisper/whisper_factory.py` - Added model filtering by implementation
5. `modules/utils/logger.py` - Enhanced logging to file

### UI/Configuration
1. `app.py` - Updated to use filtered model list
2. `configs/default_parameters.yaml` - Contains Voxtral-specific settings
3. `configs/default_parameters_faster_whisper.yaml` - Contains Faster Whisper settings

### Deployment
1. All start/stop/status scripts in the root directory

## Key Features by Implementation

### Voxtral Implementation (voxtral-mini)
- Model: voxtral-mini-3b only
- Port: 7860
- Configuration: configs/default_parameters.yaml
- Start: ./start-voxtral.sh

### Faster Whisper Implementation (faster-whisper)
- Models: All standard Whisper models (tiny, base, small, medium, large-v3, etc.)
- Port: 7861
- Configuration: configs/default_parameters_faster_whisper.yaml
- Start: ./start-faster-whisper.sh

## Backup Information
- Full backup: ~/old_backup/Voxtral-WebUI-backup-20250918.tar.gz
- Documentation: TECHNICAL_DOCUMENTATION.md

## HuggingFace Token
- Token is configured in both configuration files
- Required for speaker diarization
- Username: martossien
- Email: sylvain@octopuce.com