from fastapi import APIRouter, HTTPException, File, UploadFile, Form, BackgroundTasks
from typing import Optional, List
from pydantic import BaseModel, Field
from app.models.response import GeneratedResponse, CombinedEmotionAnalysis, ResponseWithAudio
from app.services.llm_service import llm_service
from app.services.tts_service import tts_service
from app.services.stt_service import stt_service
import uuid
from datetime import datetime
import os

# Pydantic models for request and response
class ResponseRequest(BaseModel):
    text: str = Field(..., description="User's text input")
    emotion: Optional[str] = Field(None, description="Detected primary emotion")
    emotion_analysis: Optional[CombinedEmotionAnalysis] = Field(None, description="Full emotion analysis if available")
    session_id: str = Field(..., description="Session identifier for context tracking")
    generate_audio: bool = Field(False, description="Whether to generate TTS audio")
    voice: Optional[str] = Field(None, description="Voice to use for TTS if generating audio")

class AudioResponseRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier for context tracking")
    generate_audio: bool = Field(True, description="Whether to generate TTS audio")
    voice: Optional[str] = Field(None, description="Voice to use for TTS if generating audio")

# Create router
router = APIRouter()

@router.post("/generate", response_model=ResponseWithAudio)
async def generate_response(request: ResponseRequest):
    """
    Generate an empathetic response using Gemini with optional TTS.
    
    Args:
        request: ResponseRequest with user input and emotion data
        
    Returns:
        ResponseWithAudio with the generated response and optional audio URL
    """
    try:
        # Generate response text using the LLM service
        response_text, response_id = await llm_service.generate_response(
            prompt=request.text,
            emotion=request.emotion
        )
        
        if not response_text:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate response"
            )
        
        # Create a GeneratedResponse object
        generated_response = GeneratedResponse(
            text=response_text,
            emotion_addressed=request.emotion,
            response_type="validation_with_reframe"  # Default type
        )
        
        # Generate audio if requested
        audio_url = None
        if request.generate_audio:
            audio_url, full_path = await tts_service.synthesize(
                text=response_text, 
                voice=request.voice,
                response_id=response_id
            )
        
        # Use the provided emotion analysis or create a basic one
        emotion_analysis = request.emotion_analysis
        if not emotion_analysis:
            # Create a simple emotion analysis based on the provided emotion
            emotion_analysis = CombinedEmotionAnalysis(
                overall_emotion=request.emotion or "unknown",
                emotion_intensity=0.7,
                emotion_valence=-0.5 if request.emotion in ["anxiety", "sadness", "anger", "fear"] else 0.5,
                emotion_arousal=0.8 if request.emotion in ["anxiety", "anger", "joy"] else 0.3
            )
        
        # Return the complete response
        return ResponseWithAudio(
            id=response_id,
            timestamp=datetime.now(),
            emotion_analysis=emotion_analysis,
            response=generated_response,
            audio_url=audio_url,
            session_id=request.session_id
        )
        
    except Exception as e:
        # Log the error in a real application
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.post("/audio-pipeline", response_model=ResponseWithAudio)
async def audio_to_response_pipeline(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    generate_audio: bool = Form(True),
    voice: Optional[str] = Form(None)
):
    """
    Complete audio-to-text-to-audio pipeline:
    1. Transcribe audio using Gemini
    2. Generate empathetic response using Gemini
    3. Convert response to speech using TogetherAI
    
    Args:
        audio: Audio file with user's speech
        session_id: Session identifier
        generate_audio: Whether to generate TTS audio
        voice: Voice to use for TTS
        
    Returns:
        ResponseWithAudio with the full pipeline results
    """
    try:
        # Step 1: Transcribe audio using Gemini
        transcription = await stt_service.transcribe(audio.file)
        user_text = transcription.text
        
        if not user_text:
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio. Please try again with a clearer recording."
            )
        
        print(f"Transcribed text: {user_text}")
        
        # Step 2: Generate response using Gemini
        response_text, response_id = await llm_service.generate_response(prompt=user_text)
        
        if not response_text:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate response"
            )
        
        print(f"Generated response: {response_text}")
        
        # Create a GeneratedResponse object
        generated_response = GeneratedResponse(
            text=response_text,
            emotion_addressed="unspecified",
            response_type="empathetic_response"
        )
        
        # Step 3: Generate audio response if requested
        audio_url = None
        if generate_audio:
            audio_url, full_path = await tts_service.synthesize(
                text=response_text, 
                voice=voice,
                response_id=response_id
            )
            print(f"Generated audio saved at: {full_path}")
        
        # Create a basic emotion analysis
        emotion_analysis = CombinedEmotionAnalysis(
            overall_emotion="unspecified",
            emotion_intensity=0.5,
            emotion_valence=0.0,
            emotion_arousal=0.5
        )
        
        # Return the complete response
        return ResponseWithAudio(
            id=response_id,
            timestamp=datetime.now(),
            emotion_analysis=emotion_analysis,
            response=generated_response,
            audio_url=audio_url,
            session_id=session_id
        )
    
    except Exception as e:
        print(f"Error in audio pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@router.get("/status")
async def response_status():
    """Get the status of the response generation service."""
    return {
        "status": "operational",
        "llm": "Gemini",
        "models_available": True,
        "audio_pipeline": "enabled"
    } 