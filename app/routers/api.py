from fastapi import APIRouter
from app.core.config import settings
from app.routers import audio, tts, response, llm_to_tts, visual, multimodal, realtime

# Create the main API router
api_router = APIRouter()

# Include our routers
api_router.include_router(audio.router, prefix="/audio", tags=["Audio Processing"])
api_router.include_router(tts.router, prefix="/tts", tags=["Text-to-Speech"])
api_router.include_router(response.router, prefix="/response", tags=["Response Generation"])
api_router.include_router(llm_to_tts.router, prefix="/llm-to-tts", tags=["LLM to TTS Workflow"])
api_router.include_router(visual.router, prefix="/visual", tags=["Visual Processing"])
api_router.include_router(multimodal.router, prefix="/multimodal", tags=["Multimodal Processing"])
api_router.include_router(realtime.router, prefix="/realtime", tags=["Real-time Processing"])

# Create empty stubs for emotion processing
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