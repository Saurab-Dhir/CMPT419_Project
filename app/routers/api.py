from fastapi import APIRouter
from app.core.config import settings
from app.routers import audio, tts, response, stt

# Create the main API router
api_router = APIRouter()

# Include our routers
api_router.include_router(audio.router, prefix="/audio", tags=["Audio Processing"])
api_router.include_router(tts.router, prefix="/tts", tags=["Text-to-Speech"])
api_router.include_router(stt.router, prefix="/stt", tags=["Speech-to-Text"])
api_router.include_router(response.router, prefix="/response", tags=["Response Generation"])

# Create empty stubs for visual/facial processing
@api_router.get("/visual/status")
async def visual_status():
    """Placeholder for visual processing status endpoint."""
    return {"status": "visual processing not implemented"}

@api_router.get("/emotion/status")
async def emotion_status():
    """Placeholder for emotion classification status endpoint."""
    return {"status": "emotion classification not implemented"}

@api_router.get("/response/status")
async def response_status():
    """Placeholder for LLM response generation status endpoint."""
    return {"status": "llm response not implemented"}

@api_router.get("/tts/status")
async def tts_status():
    """Placeholder for text-to-speech status endpoint."""
    return {"status": "tts not implemented"} 

@api_router.get("/stt/status")
async def stt_status():
    """Placeholder for speech-to-text status endpoint."""
    return {"status": "stt not implemented"} 