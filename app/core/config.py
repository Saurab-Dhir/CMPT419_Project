import os
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Empathetic Self-talk Coach"
    
    # Security settings - update these for production
    SECRET_KEY: str = Field(default="development_secret_key")
    
    # LLM settings - using Gemini
    LLM_API_KEY: str = Field(default="")
    LLM_MODEL: str = Field(default="gemini-pro")  # Using Gemini Pro
    
    # ElevenLabs API settings for TTS
    ELEVENLABS_API_KEY: str = Field(default="")
    
    # Legacy TogetherAI API settings (kept for backward compatibility)
    TOGETHERAI_API_KEY: str = Field(default="")
    
    # STT and TTS settings
    STT_SERVICE: str = Field(default="gemini")  # Using Gemini for STT
    TTS_SERVICE: str = Field(default="elevenlabs")  # Using ElevenLabs for TTS
    TTS_VOICE: str = Field(default="21m00Tcm4TlvDq8ikWAM")  # Default ElevenLabs "Rachel" voice ID
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 