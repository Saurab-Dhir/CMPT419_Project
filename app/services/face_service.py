import logging
import io
from typing import BinaryIO, Dict, Any, Optional
from app.services.visual_service import VisualService

# Configure logging
logger = logging.getLogger(__name__)

class FaceService:
    """Service for facial feature detection and emotion classification.
    This is a wrapper around VisualService for backward compatibility."""
    
    def __init__(self):
        """Initialize the face service."""
        self.visual_service = VisualService()
        logger.info("FaceService initialized")
    
    async def process_image(self, image_file: BinaryIO, file_extension: str = "jpg") -> Dict[str, Any]:
        """
        Process an image file to detect faces and emotions.
        
        Args:
            image_file: The image file stream
            file_extension: The file extension (e.g., jpg, png)
            
        Returns:
            Dictionary with emotion and face detection data
        """
        # Use the visual service to process the image
        result = await self.visual_service.process_image(image_file, file_extension)
        
        # Convert to simplified format
        return {
            "face_detected": result.face_detected,
            "emotion": result.emotion_prediction.emotion if result.emotion_prediction else "neutral",
            "confidence": result.emotion_prediction.confidence if result.emotion_prediction else 0.0,
            "quality": result.face_quality
        }
    
    async def detect_emotion(self, image_file: BinaryIO) -> Dict[str, Any]:
        """
        Shorthand method to just get emotion from an image.
        
        Args:
            image_file: The image file stream
            
        Returns:
            Dictionary with emotion data
        """
        result = await self.process_image(image_file)
        return {
            "emotion": result["emotion"],
            "confidence": result["confidence"]
        }

# Create a singleton instance
face_service = FaceService() 