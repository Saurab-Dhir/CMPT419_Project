from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class AudioFeatures(BaseModel):
    """Model for extracted audio features."""
    volume: float = Field(..., description="Average volume level")
    pitch: float = Field(..., description="Average pitch")
    speaking_rate: float = Field(..., description="Words per minute")
    pauses: int = Field(..., description="Number of pauses detected")
    tonal_variation: float = Field(..., description="Measure of tonal variation")
    
class AudioEmotionPrediction(BaseModel):
    """Model for emotion predictions from audio."""
    emotion: str = Field(..., description="Detected primary emotion")
    confidence: float = Field(..., description="Confidence score for the prediction")
    secondary_emotions: Dict[str, float] = Field(default_factory=dict, description="Secondary emotions with confidence scores")

class AudioTranscription(BaseModel):
    """Model for speech-to-text transcription."""
    text: str = Field(..., description="Transcribed text")
    language: str = Field(default="en", description="Detected language")
    
class AudioProcessingResult(BaseModel):
    """Complete result of audio processing."""
    id: str = Field(..., description="Unique identifier for the processing result")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the processing was completed")
    duration: float = Field(..., description="Duration of the audio in seconds")
    features: AudioFeatures = Field(..., description="Extracted audio features")
    transcription: AudioTranscription = Field(..., description="Speech-to-text result")
    emotion_prediction: Optional[AudioEmotionPrediction] = Field(None, description="Emotion prediction if available")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "audio_1234567890",
                "timestamp": "2023-04-01T12:00:00",
                "duration": 10.5,
                "features": {
                    "volume": 0.75,
                    "pitch": 220.0,
                    "speaking_rate": 150.0,
                    "pauses": 3,
                    "tonal_variation": 0.6
                },
                "transcription": {
                    "text": "I feel a bit stressed about my upcoming presentation.",
                    "language": "en"
                },
                "emotion_prediction": {
                    "emotion": "anxiety",
                    "confidence": 0.85,
                    "secondary_emotions": {
                        "nervousness": 0.75,
                        "fear": 0.45
                    }
                }
            }
        } 