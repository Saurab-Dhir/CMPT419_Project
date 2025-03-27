from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
from app.services.audio_service import audio_service
from app.models.audio import AudioProcessingResult

# Create router
router = APIRouter()

@router.post("/process", response_model=AudioProcessingResult)
async def process_audio(
    audio: UploadFile = File(...),
    duration: float = Form(...),
):
    """
    Process an audio file for transcription and emotion classification.
    
    This endpoint initiates two parallel processes:
    1. Sends the audio to Gemini API for speech-to-text transcription
    2. Logs the audio for emotion classification (classifier in development)
    
    Args:
        audio: The audio file (supported formats: WAV, MP3, FLAC)
        duration: Duration of the audio in seconds
        
    Returns:
        Processed audio data with transcription and placeholder for emotion analysis
    """
    # Validate file type
    if audio.content_type not in ["audio/wav", "audio/mpeg", "audio/flac", "audio/mp3"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {audio.content_type}. Supported types: WAV, MP3, FLAC"
        )
    
    try:
        # Process the audio using the audio service
        # This will:
        # 1. Log the audio for the classifier (currently in development)
        # 2. Send the audio to Gemini for transcription
        print(f"Processing audio file: {audio.filename} (duration: {duration}s)")
        result = await audio_service.process_audio(audio.file, duration)
        return result
    except Exception as e:
        # Log the error
        print(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@router.get("/status")
async def audio_status():
    """Get the status of the audio processing service."""
    return {
        "status": "operational",
        "services": {
            "speech_to_text": "active - using Gemini API",
            "emotion_classification": "in development"
        }
    } 