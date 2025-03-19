from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from app.routers.api import api_router
from app.core.config import settings

app = FastAPI(
    title="Empathetic Self-talk Coach API",
    description="API for the 'Mirror mirror on the wall!' project",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Create static directory for audio files if it doesn't exist
os.makedirs("static/audio", exist_ok=True)

# Mount the static directory
app.mount("/audio", StaticFiles(directory="static/audio"), name="audio")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "ok", "message": "API is running"} 