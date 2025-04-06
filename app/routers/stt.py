from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from typing import Optional
from pydantic import BaseModel, Field
from app.services.stt_service import stt_service
import mimetypes
import uuid

class STTResponse(BaseModel):
    transcript: str = Field(..., description="Speech transcript")

router = APIRouter()

@router.post("/transcribe", response_model=STTResponse)
async def transcribe_speech(
        audio_file: UploadFile = File(...),
        response_id: Optional[str] = Form(None)
    ):
    try:
        new_id = uuid.uuid4().hex[:8]
        response_id = f"stt_req_{response_id or new_id}"
        mime_type, _ = mimetypes.guess_type(audio_file.filename)
        transcript = await stt_service.transcribe(
            audio_file=audio_file.file,
            mime_type=mime_type,
            response_id=response_id
        )
        
        if not transcript:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate speech transcript"
            )
        
        return STTResponse(transcript=transcript.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@router.get("/status")
async def stt_status():
    """Get the status of the STT service."""
    return {
        "status": "operational",
        "service": "Gemini API"
    } 