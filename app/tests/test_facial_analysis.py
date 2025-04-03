import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.services.deepface_service import DeepFaceService

# Fix the path to the test image
TEST_FACE_IMAGE_PATH = os.path.join("test_files", "test_face.jpg")

class TestFacialAnalysis:
    """Tests for the facial analysis functionality."""
    
    @pytest.fixture
    def setup_service(self):
        """Set up the DeepFaceService for testing."""
        with patch("app.services.deepface_service.cv2") as mock_cv2:
            # Mock cascade classifier
            mock_cv2.CascadeClassifier.return_value = MagicMock()
            mock_cv2.data.haarcascades = "mock/path/"
            
            # Initialize service
            service = DeepFaceService()
            
            # Mock face detection result for testing
            service.detect_face = MagicMock(return_value={
                "face_detected": True,
                "face_quality": 0.9,
                "face_box": [10, 10, 50, 50]  # x, y, w, h
            })
            
            # Return service and mocked cv2
            yield service, mock_cv2
    
    def test_analyze_face_with_deepface(self, setup_service):
        """Test facial analysis with DeepFace available."""
        service, mock_cv2 = setup_service
        
        # Set up DeepFace
        service.has_deepface = True
        service.deepface = MagicMock()
        
        # Mock DeepFace analyze result
        mock_analysis = [{
            "emotion": {
                "angry": 0.01,
                "disgust": 0.0,
                "fear": 0.05,
                "happy": 0.75,
                "sad": 0.09,
                "surprise": 0.1,
                "neutral": 0.0
            },
            "dominant_emotion": "happy",
            "age": 28,
            "gender": "Woman",
            "race": {
                "asian": 0.05,
                "indian": 0.03,
                "black": 0.02,
                "white": 0.85,
                "middle eastern": 0.02,
                "latino hispanic": 0.03
            }
        }]
        service.deepface.analyze = MagicMock(return_value=mock_analysis)
        
        # Mock cv2.imread to return a fake image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        
        # Call analyze_face with mock face detection result
        result = service.analyze_face(
            TEST_FACE_IMAGE_PATH, 
            {"face_detected": True, "face_box": [10, 10, 50, 50]}
        )
        
        # Check result
        assert result["emotion"] == "happy"
        assert result["emotion_confidence"] == 0.75
        assert isinstance(result["secondary_emotions"], dict)
        assert len(result["secondary_emotions"]) > 0
        assert "surprise" in result["secondary_emotions"]
        assert result["age"] == 28
        assert result["gender"] == "female"  # Should be normalized to lowercase
        
        # Verify method calls
        mock_cv2.imread.assert_called_once_with(TEST_FACE_IMAGE_PATH)
        service.deepface.analyze.assert_called_once()
    
    def test_analyze_face_with_opencv_fallback(self, setup_service):
        """Test facial analysis with OpenCV fallback when DeepFace is not available."""
        service, mock_cv2 = setup_service
        
        # Set up without DeepFace
        service.has_deepface = False
        
        # Mock cv2.imread to return a fake image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        
        # Mock OpenCV emotion classifier
        mock_emotion_net = MagicMock()
        service.emotion_net = mock_emotion_net
        
        # Mock prediction result
        mock_emotion_net.predict.return_value = (np.array([0.1, 0.05, 0.7, 0.05, 0.1]), None)
        service.emotion_labels = ["angry", "sad", "happy", "surprise", "neutral"]
        
        # Call analyze_face with mock face detection result
        result = service.analyze_face(
            TEST_FACE_IMAGE_PATH, 
            {"face_detected": True, "face_box": [10, 10, 50, 50]}
        )
        
        # Check result
        assert result["emotion"] == "happy"
        assert result["emotion_confidence"] > 0.6
        assert isinstance(result["secondary_emotions"], dict)
        assert len(result["secondary_emotions"]) > 0
        assert "neutral" in result["secondary_emotions"]
        
        # Basic metadata should still be there
        assert "age" in result
        assert "gender" in result
        
        # Verify method calls
        mock_cv2.imread.assert_called_once_with(TEST_FACE_IMAGE_PATH)
    
    def test_analyze_face_no_face_detected(self, setup_service):
        """Test facial analysis when no face is detected."""
        service, mock_cv2 = setup_service
        
        # Call analyze_face with no face detection result
        result = service.analyze_face(
            TEST_FACE_IMAGE_PATH, 
            {"face_detected": False}
        )
        
        # Check result
        assert result["emotion"] == "neutral"
        assert result["emotion_confidence"] == 0.0
        assert isinstance(result["secondary_emotions"], dict)
        assert len(result["secondary_emotions"]) == 0
        assert result["age"] == 0
        assert result["gender"] == "unknown"
    
    def test_analyze_face_error_handling(self, setup_service):
        """Test facial analysis error handling."""
        service, mock_cv2 = setup_service
        
        # Set up DeepFace with error behavior
        service.has_deepface = True
        service.deepface = MagicMock()
        service.deepface.analyze = MagicMock(side_effect=Exception("Test error"))
        
        # Mock cv2.imread to return a fake image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        
        # Call analyze_face
        result = service.analyze_face(
            TEST_FACE_IMAGE_PATH, 
            {"face_detected": True, "face_box": [10, 10, 50, 50]}
        )
        
        # Check default/fallback values are used
        assert result["emotion"] == "neutral"
        assert result["emotion_confidence"] > 0
        assert isinstance(result["secondary_emotions"], dict)
        assert "error" in result
        assert "Test error" in result["error"]
    
    def test_analyze_face_image_preprocessing(self, setup_service):
        """Test that face image is properly cropped before analysis."""
        service, mock_cv2 = setup_service
        
        # Set up DeepFace
        service.has_deepface = True
        service.deepface = MagicMock()
        service.deepface.analyze = MagicMock(return_value=[{
            "emotion": {"happy": 0.75, "neutral": 0.25},
            "dominant_emotion": "happy",
            "age": 28,
            "gender": "Woman"
        }])
        
        # Mock cv2.imread to return a fake image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        
        # Call analyze_face with face box
        result = service.analyze_face(
            TEST_FACE_IMAGE_PATH, 
            {"face_detected": True, "face_box": [10, 10, 50, 50]}
        )
        
        # Verify image preprocessing occurred
        mock_cv2.rectangle.assert_called()  # Should draw rectangle for debugging
        
        # Check result is as expected
        assert result["emotion"] == "happy"
    
    def test_analyze_face_normalization(self, setup_service):
        """Test that emotion and confidence values are normalized."""
        service, mock_cv2 = setup_service
        
        # Set up DeepFace with more complex emotion mix
        service.has_deepface = True
        service.deepface = MagicMock()
        
        # Mock DeepFace analyze with unusual emotion names and confidence values
        mock_analysis = [{
            "emotion": {
                "angry": 0.01,
                "disgust": 0.3,  # Second highest
                "fear": 0.05,
                "happiness": 0.45,  # Highest but different naming than expected
                "sadness": 0.09,
                "surpri_se": 0.1,  # Unusual naming
                "neutral": 0.0
            },
            "dominant_emotion": "happiness",  # Different naming
            "age": 28,
            "gender": "MAN"  # Unusual casing
        }]
        service.deepface.analyze = MagicMock(return_value=mock_analysis)
        
        # Mock cv2.imread to return a fake image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        
        # Call analyze_face
        result = service.analyze_face(
            TEST_FACE_IMAGE_PATH, 
            {"face_detected": True, "face_box": [10, 10, 50, 50]}
        )
        
        # Check normalization
        assert result["emotion"] == "happy"  # Should normalize to standard emotion name
        assert result["emotion_confidence"] == 0.45
        assert result["gender"] == "male"  # Should be normalized to lowercase
        assert "disgust" in result["secondary_emotions"]
        assert result["secondary_emotions"]["disgust"] == 0.3 