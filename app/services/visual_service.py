import uuid
import logging
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, BinaryIO
import numpy as np
import cv2

from app.models.visual import (
    FacialLandmarks,
    FacialFeatures,
    FacialEmotionPrediction,
    VisualProcessingResult
)
from app.services.deepface_service import DeepFaceService
from app.utils.logging import response_logger

# Configure logging
logger = logging.getLogger(__name__)

class VisualService:
    """Service for processing visual data and extracting facial features and emotions."""
    
    def __init__(self):
        """Initialize the visual service with required dependencies."""
        self.deepface_service = DeepFaceService()
        self._initialize_cache()
        logger.info("VisualService initialized")
    
    def _initialize_cache(self):
        """Initialize the cache for storing processed results."""
        self.cache = {}
        self.max_cache_size = 100
        self.cache_expiration_seconds = 300  # 5 minutes
    
    async def process_image(self, image_file: BinaryIO, file_extension: str) -> VisualProcessingResult:
        """
        Process an image file to detect faces, extract features, and predict emotions.
        
        Args:
            image_file: The image file stream
            file_extension: The file extension (e.g., jpg, png)
            
        Returns:
            VisualProcessingResult with extracted features and analysis
        """
        # Create a unique ID for this processing result
        processing_id = f"visual_{uuid.uuid4().hex[:10]}"
        
        try:
            # Convert image file to numpy array
            image_bytes = image_file.read()
            image = self._load_image(image_bytes)
            
            # Check if this image is in cache
            cache_key = self._generate_cache_key(image_bytes)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Cache hit for image processing")
                return cached_result
            
            # Process the image
            logger.info(f"Processing image with id {processing_id}")
            result = self._process_image_internal(image, processing_id)
            
            # Cache the result
            self._add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            # Return a result indicating failure
            return self._create_error_result(processing_id, str(e))
    
    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Load image from bytes into numpy array.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Image as numpy array
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image data")
        
        return image
    
    def _process_image_internal(self, image: np.ndarray, processing_id: str) -> VisualProcessingResult:
        """
        Internal method to process an image and extract features.
        
        Args:
            image: Image as numpy array
            processing_id: Unique ID for this processing
            
        Returns:
            VisualProcessingResult with extracted features and analysis
        """
        try:
            # Use DeepFace service to detect face
            face_detection = self.deepface_service.detect_face(image)
            
            if not face_detection.get("detected", False):
                logger.warning("No face detected in the image")
                return self._create_no_face_result(processing_id)
            
            # Extract face quality
            face_quality = face_detection.get("quality", 0.0)
            
            # Analyze face to get emotion predictions
            face_analysis = self.deepface_service.analyze_face(image, face_detection)
            
            # Check if face_analysis is a dictionary
            if not isinstance(face_analysis, dict):
                logger.warning(f"Unexpected analyze_face result type: {type(face_analysis)}")
                face_analysis = {
                    "emotion": "neutral",
                    "emotion_confidence": 0.0,
                    "secondary_emotions": {}
                }
            
            # Extract facial landmarks and calculate metrics
            landmarks = self.deepface_service.extract_landmarks(image, face_detection)
            metrics = self.deepface_service.calculate_metrics(landmarks)
            
            # Create facial features object
            features = FacialFeatures(
                landmarks=landmarks,
                eye_openness=metrics.get("eye_openness", 0.0),
                mouth_openness=metrics.get("mouth_openness", 0.0),
                eyebrow_raise=metrics.get("eyebrow_raise", 0.0),
                head_pose=metrics.get("head_pose", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
            )
            
            # Extract emotion data
            primary_emotion = face_analysis.get("emotion", "neutral")
            emotion_confidence = face_analysis.get("emotion_confidence", 0.0)
            secondary_emotions = face_analysis.get("secondary_emotions", {})
            
            # Create emotion prediction object
            emotion_prediction = FacialEmotionPrediction(
                emotion=primary_emotion,
                confidence=emotion_confidence,
                secondary_emotions=secondary_emotions
            )
            
            # Create and return the complete processing result
            return VisualProcessingResult(
                id=processing_id,
                timestamp=datetime.now(),
                features=features,
                emotion_prediction=emotion_prediction,
                face_detected=True,
                face_quality=face_quality
            )
        except Exception as e:
            logger.error(f"Error in _process_image_internal: {str(e)}")
            return self._create_no_face_result(processing_id)
    
    def _create_no_face_result(self, processing_id: str) -> VisualProcessingResult:
        """
        Create a result object for when no face is detected.
        
        Args:
            processing_id: Unique ID for this processing
            
        Returns:
            VisualProcessingResult with default values
        """
        # Create empty landmarks
        landmarks = FacialLandmarks(
            eye_positions=[(0.0, 0.0), (0.0, 0.0)],
            mouth_position=[(0.0, 0.0), (0.0, 0.0)],
            eyebrow_positions=[(0.0, 0.0), (0.0, 0.0)],
            nose_position=(0.0, 0.0),
            face_contour=[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        )
        
        # Create default features
        features = FacialFeatures(
            landmarks=landmarks,
            eye_openness=0.0,
            mouth_openness=0.0,
            eyebrow_raise=0.0,
            head_pose={"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        )
        
        # Create default emotion prediction
        emotion_prediction = FacialEmotionPrediction(
            emotion="unknown",
            confidence=0.0,
            secondary_emotions={}
        )
        
        # Create and return the complete processing result
        return VisualProcessingResult(
            id=processing_id,
            timestamp=datetime.now(),
            features=features,
            emotion_prediction=emotion_prediction,
            face_detected=False,
            face_quality=0.0
        )
    
    def _create_error_result(self, processing_id: str, error_message: str) -> VisualProcessingResult:
        """
        Create a result object for when an error occurs during processing.
        
        Args:
            processing_id: Unique ID for this processing
            error_message: Description of the error
            
        Returns:
            VisualProcessingResult with error information
        """
        logger.error(f"Error creating visual processing result: {error_message}")
        return self._create_no_face_result(processing_id)
    
    def _generate_cache_key(self, image_bytes: bytes) -> str:
        """
        Generate a cache key for an image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            String key for cache
        """
        import hashlib
        return hashlib.md5(image_bytes).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[VisualProcessingResult]:
        """
        Get a result from cache if it exists and is not expired.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached result or None if not found
        """
        if cache_key not in self.cache:
            return None
        
        cached_item = self.cache[cache_key]
        
        # Check if the cache entry has expired
        cache_age = (datetime.now() - cached_item["timestamp"]).total_seconds()
        if cache_age > self.cache_expiration_seconds:
            # Remove the expired item
            del self.cache[cache_key]
            return None
        
        return cached_item["result"]
    
    def _add_to_cache(self, cache_key: str, result: VisualProcessingResult):
        """
        Add a result to the cache.
        
        Args:
            cache_key: Cache key
            result: Result to cache
        """
        # Check if we need to remove old entries
        if len(self.cache) >= self.max_cache_size:
            # Remove the oldest entry
            oldest_key = min(self.cache.items(), key=lambda x: x[1]["timestamp"])[0]
            del self.cache[oldest_key]
        
        # Add the new entry
        self.cache[cache_key] = {
            "timestamp": datetime.now(),
            "result": result
        }

# Create a singleton instance
visual_service = VisualService() 