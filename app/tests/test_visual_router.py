import io
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from app.main import app
from app.models.visual import (
    FacialLandmarks,
    FacialFeatures,
    FacialEmotionPrediction,
    VisualProcessingResult
)

client = TestClient(app)

def create_mock_result():
    """Create a mock VisualProcessingResult for testing."""
    # Create landmarks
    landmarks = FacialLandmarks(
        eye_positions=[(0.3, 0.4), (0.7, 0.4)],
        mouth_position=[(0.4, 0.7), (0.6, 0.7)],
        eyebrow_positions=[(0.3, 0.35), (0.7, 0.35)],
        nose_position=(0.5, 0.5),
        face_contour=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
    )
    
    # Create features
    features = FacialFeatures(
        landmarks=landmarks,
        eye_openness=0.8,
        mouth_openness=0.2,
        eyebrow_raise=0.6,
        head_pose={"pitch": 5.0, "yaw": -2.0, "roll": 1.0}
    )
    
    # Create emotion prediction
    emotion_prediction = FacialEmotionPrediction(
        emotion="happy",
        confidence=0.85,
        secondary_emotions={"sad": 0.05, "angry": 0.03, "neutral": 0.07}
    )
    
    # Create and return the complete result
    return VisualProcessingResult(
        id="visual_test123456",
        timestamp=datetime.now(),
        features=features,
        emotion_prediction=emotion_prediction,
        face_detected=True,
        face_quality=0.9
    )

def test_visual_status():
    """Test that the visual status endpoint returns the expected response."""
    response = client.get("/api/v1/visual/status")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"
    assert "services" in response.json()
    assert "cache" in response.json()

@patch("app.services.visual_service.visual_service.process_image")
def test_process_image_success(mock_process_image):
    """Test that the image processing endpoint successfully processes a valid image."""
    # Create a mock result
    mock_result = create_mock_result()
    mock_process_image.return_value = mock_result
    
    # Create a test file
    test_file = io.BytesIO(b"test image content")
    response = client.post(
        "/api/v1/visual/process",
        files={"image": ("test.jpg", test_file, "image/jpeg")}
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    
    # Verify the response structure and content
    assert "id" in data
    assert data["id"] == "visual_test123456"
    assert "features" in data
    assert "emotion_prediction" in data
    assert data["emotion_prediction"]["emotion"] == "happy"
    assert data["face_detected"] is True
    assert data["face_quality"] == 0.9

@patch("app.services.visual_service.visual_service.process_image")
def test_process_image_no_face(mock_process_image):
    """Test that the image processing endpoint handles images with no face correctly."""
    # Create a mock result with no face
    mock_result = create_mock_result()
    mock_result.face_detected = False
    mock_result.face_quality = 0.0
    mock_result.emotion_prediction.emotion = "unknown"
    mock_result.emotion_prediction.confidence = 0.0
    mock_result.emotion_prediction.secondary_emotions = {}
    
    mock_process_image.return_value = mock_result
    
    # Create a test file
    test_file = io.BytesIO(b"test image content")
    response = client.post(
        "/api/v1/visual/process",
        files={"image": ("test.jpg", test_file, "image/jpeg")}
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    
    # Verify the response structure and content
    assert "id" in data
    assert "features" in data
    assert "emotion_prediction" in data
    assert data["emotion_prediction"]["emotion"] == "unknown"
    assert data["face_detected"] is False
    assert data["face_quality"] == 0.0

def test_process_image_invalid_file_type():
    """Test that the image processing endpoint rejects invalid file types."""
    # Create a test file with incorrect content type
    test_file = io.BytesIO(b"test content")
    response = client.post(
        "/api/v1/visual/process",
        files={"image": ("test.txt", test_file, "text/plain")}
    )
    
    # Check the response
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

@patch("app.services.visual_service.visual_service.process_image")
def test_process_image_service_error(mock_process_image):
    """Test that the image processing endpoint handles service errors correctly."""
    # Configure the mock to raise an exception
    mock_process_image.side_effect = Exception("Test service error")
    
    # Create a test file
    test_file = io.BytesIO(b"test image content")
    response = client.post(
        "/api/v1/visual/process",
        files={"image": ("test.jpg", test_file, "image/jpeg")}
    )
    
    # Check the response
    assert response.status_code == 500
    assert "Error processing image" in response.json()["detail"] 