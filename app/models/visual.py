from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

class FacialLandmarks(BaseModel):
    """Model for facial landmarks data."""
    eye_positions: List[Tuple[float, float]] = Field(..., description="Eye positions (x,y coordinates)")
    mouth_position: List[Tuple[float, float]] = Field(..., description="Mouth corner positions")
    eyebrow_positions: List[Tuple[float, float]] = Field(..., description="Eyebrow positions")
    nose_position: Tuple[float, float] = Field(..., description="Nose position")
    face_contour: List[Tuple[float, float]] = Field(..., description="Face contour points")

class FacialFeatures(BaseModel):
    """Model for extracted facial features."""
    landmarks: FacialLandmarks = Field(..., description="Facial landmarks")
    eye_openness: float = Field(..., description="Measure of eye openness")
    mouth_openness: float = Field(..., description="Measure of mouth openness")
    eyebrow_raise: float = Field(..., description="Measure of eyebrow raise")
    head_pose: Dict[str, float] = Field(..., description="Head pose angles (pitch, yaw, roll)")
    
class FacialEmotionPrediction(BaseModel):
    """Model for emotion predictions from facial expressions."""
    emotion: str = Field(..., description="Detected primary emotion")
    confidence: float = Field(..., description="Confidence score for the prediction")
    secondary_emotions: Dict[str, float] = Field(default_factory=dict, description="Secondary emotions with confidence scores")

class VisualProcessingResult(BaseModel):
    """Complete result of visual/facial processing."""
    id: str = Field(..., description="Unique identifier for the processing result")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the processing was completed")
    features: FacialFeatures = Field(..., description="Extracted facial features")
    emotion_prediction: FacialEmotionPrediction = Field(..., description="Emotion prediction")
    face_detected: bool = Field(..., description="Whether a face was detected")
    face_quality: float = Field(..., description="Quality score for the detected face")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "visual_1234567890",
                "timestamp": "2023-04-01T12:00:00",
                "features": {
                    "landmarks": {
                        "eye_positions": [[0.3, 0.4], [0.7, 0.4]],
                        "mouth_position": [[0.4, 0.7], [0.6, 0.7]],
                        "eyebrow_positions": [[0.3, 0.35], [0.7, 0.35]],
                        "nose_position": [0.5, 0.5],
                        "face_contour": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]
                    },
                    "eye_openness": 0.8,
                    "mouth_openness": 0.2,
                    "eyebrow_raise": 0.6,
                    "head_pose": {
                        "pitch": 5.0,
                        "yaw": -2.0,
                        "roll": 1.0
                    }
                },
                "emotion_prediction": {
                    "emotion": "concern",
                    "confidence": 0.78,
                    "secondary_emotions": {
                        "fear": 0.45,
                        "sadness": 0.30
                    }
                },
                "face_detected": True,
                "face_quality": 0.92
            }
        } 