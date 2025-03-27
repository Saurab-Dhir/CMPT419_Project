from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any
from app.models.response import LLMInput, LLMToTTSResponse
from app.services.llm_service import llm_service
from app.services.elevenlabs_service import elevenlabs_service
import time

# Create router
router = APIRouter()

@router.post("/process", response_model=LLMToTTSResponse)
async def process_llm_to_tts(llm_input: LLMInput):
    """
    Process text through the LLM and convert the response to speech using ElevenLabs.
    
    Args:
        llm_input: The structured input for the LLM
        
    Returns:
        LLMToTTSResponse object with text and audio URL
    """
    try:
        # Step 1: Generate text response from LLM (Gemini)
        start_time = time.time()
        response_text, response_id = await llm_service.process_llm_input(llm_input)
        llm_time = time.time() - start_time
        
        if not response_text:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate text response from LLM"
            )
        
        # Step 2: Convert text to speech using ElevenLabs
        start_time = time.time()
        audio_url, audio_path = await elevenlabs_service.synthesize_speech(
            text=response_text,
            response_id=response_id
        )
        tts_time = time.time() - start_time
        
        # Step 3: Create and return the combined response
        response = LLMToTTSResponse(
            response_id=response_id,
            llm_text=response_text,
            audio_url=audio_url,
            session_id=llm_input.session_id,
            emotion=llm_input.emotion,
            metadata={
                "llm_processing_time": llm_time,
                "tts_processing_time": tts_time,
                "total_processing_time": llm_time + tts_time
            }
        )
        
        return response
        
    except Exception as e:
        # Log the error and raise a consistent exception
        print(f"Error in LLM-to-TTS workflow: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process LLM-to-TTS workflow: {str(e)}"
        )

@router.post("/process-async", response_model=LLMToTTSResponse)
async def process_llm_to_tts_async(llm_input: LLMInput, background_tasks: BackgroundTasks):
    """
    Process text through LLM and TTS asynchronously in the background.
    Returns immediately with the LLM response and queues the TTS processing.
    
    Args:
        llm_input: The structured input for the LLM
        background_tasks: FastAPI background task manager
        
    Returns:
        LLMToTTSResponse object with text (audio URL will be None initially)
    """
    try:
        # Step 1: Generate text response from LLM (Gemini)
        start_time = time.time()
        response_text, response_id = await llm_service.process_llm_input(llm_input)
        llm_time = time.time() - start_time
        
        if not response_text:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate text response from LLM"
            )
        
        # Create initial response without audio URL
        response = LLMToTTSResponse(
            response_id=response_id,
            llm_text=response_text,
            audio_url=None,  # Will be generated asynchronously
            session_id=llm_input.session_id,
            emotion=llm_input.emotion,
            metadata={
                "llm_processing_time": llm_time,
                "status": "tts_processing_queued"
            }
        )
        
        # Queue TTS processing in the background
        async def process_tts_in_background():
            try:
                await elevenlabs_service.synthesize_speech(
                    text=response_text,
                    response_id=response_id
                )
            except Exception as e:
                print(f"Background TTS processing error: {str(e)}")
        
        background_tasks.add_task(process_tts_in_background)
        
        return response
        
    except Exception as e:
        # Log the error and raise a consistent exception
        print(f"Error in async LLM-to-TTS workflow: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process async LLM-to-TTS workflow: {str(e)}"
        )

@router.get("/status")
async def workflow_status():
    """Simple status endpoint to verify this router is functioning."""
    return {
        "status": "ok",
        "workflows": ["LLM to TTS synchronous", "LLM to TTS asynchronous"]
    } 