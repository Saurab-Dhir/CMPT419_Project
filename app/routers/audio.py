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
    Process an audio file to extract features, transcribe speech, and predict emotions.
    
    Args:
        audio: The audio file (supported formats: WAV, MP3, FLAC)
        duration: Duration of the audio in seconds
        
    Returns:
        Processed audio data with features, transcription, and emotion prediction
    """
    # Validate file type
    if audio.content_type not in ["audio/wav", "audio/mpeg", "audio/flac", "audio/mp3"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {audio.content_type}. Supported types: WAV, MP3, FLAC"
        )
    
    try:
        # Process the audio using the audio service
        result = await audio_service.process_audio(audio.file, duration)
        return result
    except Exception as e:
        # Log the error in a real application
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@router.get("/status")
async def audio_status():
    """Get the status of the audio processing service."""
    # In a real implementation, this would check if all required services are available
    return {
        "status": "operational",
        "services": {
            "feature_extraction": True,
            "speech_to_text": True,
            "emotion_detection": True
        }
    } 