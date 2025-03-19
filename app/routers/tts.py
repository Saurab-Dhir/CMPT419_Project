from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Optional
from pydantic import BaseModel, Field
from app.services.tts_service import tts_service
import os
import uuid

# Pydantic models for request and response
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice: Optional[str] = Field(None, description="Voice to use for synthesis")
    response_id: Optional[str] = Field(None, description="Optional ID to associate with this response")

class TTSResponse(BaseModel):
    audio_url: str = Field(..., description="URL to the generated audio file")
    text: str = Field(..., description="The text that was converted")
    voice: str = Field(..., description="The voice that was used")
    output_path: Optional[str] = Field(None, description="Path to the saved output file")

# Create router
router = APIRouter()

@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """
    Convert text to speech using TogetherAI TTS API.
    
    Args:
        request: TTSRequest with text to convert and optional voice
        
    Returns:
        TTSResponse with URL to the audio file
    """
    try:
        # Generate a response ID if not provided
        response_id = request.response_id or f"tts_req_{uuid.uuid4().hex[:8]}"
        
        # Generate speech from text
        audio_url, output_path = await tts_service.synthesize(
            text=request.text, 
            voice=request.voice,
            response_id=response_id
        )
        
        if not audio_url:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate speech audio"
            )
        
        # Return the response
        return TTSResponse(
            audio_url=audio_url,
            text=request.text,
            voice=request.voice or "default",
            output_path=output_path
        )
        
    except Exception as e:
        # Log the error in a real application
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@router.get("/status")
async def tts_status():
    """Get the status of the TTS service."""
    return {
        "status": "operational",
        "service": "TogetherAI TTS",
        "voices_available": True
    } 