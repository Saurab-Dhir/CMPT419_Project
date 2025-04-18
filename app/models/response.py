from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.audio import AudioEmotionPrediction
from app.models.visual import FacialEmotionPrediction

class CombinedEmotionAnalysis(BaseModel):
    """Combined emotion analysis from audio and visual inputs."""
    audio_emotion: Optional[AudioEmotionPrediction] = Field(None, description="Emotion analysis from audio")
    visual_emotion: Optional[FacialEmotionPrediction] = Field(None, description="Emotion analysis from facial expressions")
    overall_emotion: str = Field(..., description="Overall determined emotion")
    emotion_intensity: float = Field(..., description="Intensity of the emotion (0-1)")
    emotion_valence: float = Field(..., description="Emotional valence (negative to positive, -1 to 1)")
    emotion_arousal: float = Field(..., description="Emotional arousal (calm to excited, 0-1)")

class GeneratedResponse(BaseModel):
    """Model for the empathetic response generated by the LLM."""
    text: str = Field(..., description="The generated response text")
    emotion_addressed: str = Field(..., description="The emotion being addressed")
    response_type: str = Field(..., description="Type of response (validation, reframing, etc.)")
    alternative_responses: Optional[List[str]] = Field(None, description="Alternative responses that were considered")

class ResponseWithAudio(BaseModel):
    """Complete response with optional TTS audio."""
    id: str = Field(..., description="Unique identifier for the response")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated")
    emotion_analysis: CombinedEmotionAnalysis = Field(..., description="Emotion analysis that informed the response")
    response: GeneratedResponse = Field(..., description="The generated empathetic response")
    audio_url: Optional[str] = Field(None, description="URL to the TTS audio file if generated")
    session_id: str = Field(..., description="ID of the session this response belongs to")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "response_1234567890",
                "timestamp": "2023-04-01T12:01:00",
                "emotion_analysis": {
                    "audio_emotion": {
                        "emotion": "anxiety",
                        "secondary_emotions": {
                            "nervousness": 0.75,
                            "fear": 0.45
                        }
                    },
                    "visual_emotion": {
                        "emotion": "concern",
                        "secondary_emotions": {
                            "fear": 0.45,
                            "sadness": 0.30
                        }
                    },
                    "overall_emotion": "anxiety",
                    "emotion_intensity": 0.7,
                    "emotion_valence": -0.6,
                    "emotion_arousal": 0.8
                },
                "response": {
                    "text": "I notice you're feeling anxious about your presentation. That's completely understandable - public speaking can be challenging. Remember that feeling nervous means you care about doing well, which is actually a strength. What specific aspect feels most overwhelming right now?",
                    "emotion_addressed": "anxiety",
                    "response_type": "validation_with_reframe",
                    "alternative_responses": [
                        "It sounds like you're experiencing some anxiety about your presentation. Would you like to talk through some preparation strategies?",
                        "I'm hearing that you're feeling nervous about presenting. What would help you feel more prepared?"
                    ]
                },
                "audio_url": "https://example.com/tts/response_1234567890.mp3",
                "session_id": "session_9876543210"
            }
        }

class LLMInput(BaseModel):
    """Pydantic model for input to the LLM service."""
    text: str = Field(..., description="User's text input to process")
    emotion: Optional[str] = Field(None, description="Detected primary emotion")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    session_id: str = Field(..., description="Session identifier for tracking the conversation")
    max_tokens: Optional[int] = Field(300, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Temperature for response generation")

class LLMToTTSResponse(BaseModel):
    """Response model for the LLM to TTS workflow."""
    response_id: str = Field(..., description="Unique identifier for this response")
    llm_text: str = Field(..., description="Text generated by the LLM")
    audio_url: Optional[str] = Field(None, description="URL to access the generated audio")
    session_id: str = Field(..., description="Session identifier for tracking")
    emotion: Optional[str] = Field(None, description="Emotion detected or processed")
    model_emotion: Optional[str] = Field(None, description="Emotion the 3D model should display")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class MultiModalEmotionInput(BaseModel):
    """Input model for multi-modal emotion detection with semantic, tonal, and facial emotions."""
    user_speech: str = Field(..., description="The transcribed user speech")
    semantic_emotion: Optional[str] = Field(None, description="Emotion detected from the semantic content of speech")
    tonal_emotion: Optional[str] = Field(None, description="Emotion detected from the tone/prosody of speech")
    facial_emotion: Optional[str] = Field(None, description="Emotion detected from facial expressions")
    fused_emotion: Dict[str, float] = Field(..., description="Combined emotion prediction of all modalities")
    session_id: str = Field(..., description="Session identifier for tracking the conversation")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the input was collected")
    
    class Config:
        schema_extra = {
            "example": {
                "user_speech": "I'm feeling really anxious about my presentation tomorrow.",
                "semantic_emotion": "anxiety",
                "tonal_emotion": "fear",
                "facial_emotion": "worry",
                "fused_emotion": {
                    "sad": 0.721,
                    "neutral": 0.224,
                    "angry": 0.014,
                    "fearful": 0.014,
                    "disgust": 0.014,
                    "happy": 0.014
                },
                "session_id": "session_1234567890",
                "timestamp": "2023-04-01T12:00:00"
            }
        }