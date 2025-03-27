# Test Files Directory

This directory contains sample audio files for testing the speech-to-text and emotion classification functionality.

## Available Test Files

- `test_1.wav` - Sample audio file for testing

## How to Use

You can test the audio processing workflow with these files using:

1. **Python Test Script**:
   ```
   python test_audio_workflow.py
   ```
   This script will send the audio to the API and show the transcription results.

2. **Direct API Call**:
   ```
   curl -X POST "http://localhost:8000/api/v1/audio/process" \
     -F "audio=@test_files/test_1.wav" \
     -F "duration=5.0"
   ```

3. **Swagger UI**:
   - Go to http://localhost:8000/docs
   - Navigate to the `/api/v1/audio/process` endpoint
   - Click "Try it out"
   - Upload a test file and set the duration
   - Click "Execute"

## Adding New Test Files

You can add your own audio files to this directory for testing. Supported formats:
- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)

For best results with the Gemini transcription, use clear audio with minimal background noise and good recording quality. 