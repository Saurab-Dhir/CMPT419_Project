import io
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from app.main import app
from app.services.visual_service import VisualService
from app.models.visual import VisualProcessingResult

client = TestClient(app)

def create_test_image():
    """Create a test image with a simple face-like shape for testing."""
    # Create a 200x200 white image
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    # Draw a simple face-like shape
    # Face outline (circle)
    cv2.circle(img, (100, 100), 80, (200, 200, 200), -1)
    
    # Eyes (circles)
    cv2.circle(img, (70, 80), 15, (0, 0, 0), -1)
    cv2.circle(img, (130, 80), 15, (0, 0, 0), -1)
    
    # Mouth (ellipse)
    cv2.ellipse(img, (100, 130), (30, 15), 0, 0, 180, (0, 0, 0), -1)
    
    # Convert to bytes
    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(buffer.tobytes())
    img_bytes.name = 'test_face.jpg'
    
    return img_bytes

@pytest.mark.asyncio
async def test_full_visual_workflow():
    """
    Test the entire visual processing workflow from API to service.
    This test simulates an end-to-end flow but uses mocks for the DeepFace service.
    """
    # Create a test image
    test_image = create_test_image()
    
    # Create mock for the DeepFaceService methods
    with patch('app.services.deepface_service.DeepFaceService') as mock_deepface_class:
        # Create a mock instance
        mock_deepface = MagicMock()
        mock_deepface_class.return_value = mock_deepface
        
        # Configure the mocks
        mock_deepface.detect_face.return_value = {
            "detected": True,
            "quality": 0.85,
            "box": [50, 50, 100, 100]
        }
        
        mock_deepface.analyze_face.return_value = {
            "emotion": {
                "dominant_emotion": "happy",
                "emotion": {
                    "happy": 0.9,
                    "sad": 0.05,
                    "neutral": 0.03,
                    "angry": 0.02
                }
            }
        }
        
        # Mock the landmarks extraction
        mock_deepface.extract_landmarks.return_value = MagicMock()
        
        # Mock the metrics calculation
        mock_deepface.calculate_metrics.return_value = {
            "eye_openness": 0.7,
            "mouth_openness": 0.3,
            "eyebrow_raise": 0.5,
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        }
        
        # Reset the visual_service to use our mocked DeepFaceService
        from app.services.visual_service import visual_service
        visual_service.deepface_service = mock_deepface
        
        # Call the API endpoint
        response = client.post(
            "/api/v1/visual/process",
            files={"image": ("test_face.jpg", test_image, "image/jpeg")}
        )
        
        # Check the response status
        assert response.status_code == 200
        
        # Parse the response
        data = response.json()
        
        # Check the response structure and content
        assert "id" in data
        assert data["id"].startswith("visual_")
        assert "features" in data
        assert "emotion_prediction" in data
        assert "face_detected" in data
        assert "face_quality" in data
        
        # Check face detection status
        assert data["face_detected"] is True
        assert data["face_quality"] == 0.85
        
        # Check emotion prediction
        assert data["emotion_prediction"]["emotion"] == "happy"
        assert data["emotion_prediction"]["confidence"] == 0.9
        assert "sad" in data["emotion_prediction"]["secondary_emotions"]
        
        # Check features
        assert data["features"]["eye_openness"] == 0.7
        assert data["features"]["mouth_openness"] == 0.3
        assert data["features"]["eyebrow_raise"] == 0.5

@pytest.mark.asyncio
async def test_visual_workflow_no_face():
    """
    Test the visual processing workflow when no face is detected.
    """
    # Create a test image (plain white image with no face)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, buffer = cv2.imencode('.jpg', img)
    test_image = io.BytesIO(buffer.tobytes())
    test_image.name = 'no_face.jpg'
    
    # Create mock for the DeepFaceService methods
    with patch('app.services.deepface_service.DeepFaceService') as mock_deepface_class:
        # Create a mock instance
        mock_deepface = MagicMock()
        mock_deepface_class.return_value = mock_deepface
        
        # Configure the mock to return no face detected
        mock_deepface.detect_face.return_value = {
            "detected": False,
            "quality": 0.0
        }
        
        # Reset the visual_service to use our mocked DeepFaceService
        from app.services.visual_service import visual_service
        visual_service.deepface_service = mock_deepface
        
        # Call the API endpoint
        response = client.post(
            "/api/v1/visual/process",
            files={"image": ("no_face.jpg", test_image, "image/jpeg")}
        )
        
        # Check the response status
        assert response.status_code == 200
        
        # Parse the response
        data = response.json()
        
        # Check the response structure and content
        assert "id" in data
        assert data["id"].startswith("visual_")
        assert "features" in data
        assert "emotion_prediction" in data
        assert "face_detected" in data
        assert "face_quality" in data
        
        # Check face detection status
        assert data["face_detected"] is False
        assert data["face_quality"] == 0.0
        
        # Check default emotion values
        assert data["emotion_prediction"]["emotion"] == "unknown"
        assert data["emotion_prediction"]["confidence"] == 0.0
        assert data["emotion_prediction"]["secondary_emotions"] == {} 