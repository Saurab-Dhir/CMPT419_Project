from fastapi import APIRouter
from app.core.config import settings
from app.routers import audio, tts, response, face

# Create the main API router
api_router = APIRouter()

# Include our routers
api_router.include_router(audio.router, prefix="/audio", tags=["Audio Processing"])
api_router.include_router(tts.router, prefix="/tts", tags=["Text-to-Speech"])
api_router.include_router(response.router, prefix="/response", tags=["Response Generation"])
api_router.include_router(face.router, prefix="/face", tags=["Facial Emotion Detection"])

# Create empty stubs for visual/facial processing
@api_router.get("/visual/status")
async def visual_status():
    """Placeholder for visual processing status endpoint."""
    return {"status": "not implemented"}

@api_router.get("/emotion/status")
async def emotion_status():
    """Placeholder for emotion classification status endpoint."""
    return {"status": "not implemented"}

@api_router.get("/response/status")
async def response_status():
    """Placeholder for LLM response generation status endpoint."""
    return {"status": "not implemented"}

@api_router.get("/tts/status")
async def tts_status():
    """Placeholder for text-to-speech status endpoint."""
    return {"status": "not implemented"} 