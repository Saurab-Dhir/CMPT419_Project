from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from typing import Optional
import io
import uuid
from app.services.multimodal_service import multimodal_service
from app.models.response import LLMToTTSResponse

router = APIRouter()

@router.post("/process", response_model=LLMToTTSResponse)
async def process_multimodal_input(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file containing speech"),
    video_file: UploadFile = File(..., description="Video file or image containing face"),
    audio_duration: float = Form(..., description="Duration of the audio in seconds"),
    session_id: str = Form(..., description="Session identifier for tracking"),
    file_extension: str = Form("jpg", description="File extension for the video/image")
):
    """
    Process multimodal input (audio + video) to generate an empathetic response.
    
    This endpoint:
    1. Transcribes speech using Gemini
    2. Detects semantic emotion from words using Gemini
    3. Analyzes vocal tone using a tone classification model
    4. Analyzes facial expressions using DeepFace
    5. Generates a natural response considering all emotion sources
    6. Converts the response to speech using ElevenLabs
    
    Returns a response with both text and audio URL.
    """
    if not audio_file.filename or not video_file.filename:
        raise HTTPException(status_code=400, detail="Both audio and video files are required")
    
    try:
        # Read the files into memory
        audio_data = await audio_file.read()
        video_data = await video_file.read()
        
        # Create file-like objects
        audio_io = io.BytesIO(audio_data)
        video_io = io.BytesIO(video_data)
        
        # Set proper content types based on file extensions
        audio_io.content_type = audio_file.content_type
        video_io.content_type = video_file.content_type
        
        # Process the multimodal input
        result = await multimodal_service.process_multimodal_input(
            audio_file=audio_io,
            video_file=video_io,
            audio_duration=audio_duration,
            session_id=session_id,
            file_extension=file_extension
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing multimodal input: {str(e)}"
        ) 