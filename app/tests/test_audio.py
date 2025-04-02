import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
import cv2
import numpy as np
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = TestClient(app)

def test_audio_status():
    """Test that the audio status endpoint returns the expected response."""
    response = client.get("/api/v1/audio/status")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"
    assert "services" in response.json()
    assert isinstance(response.json()["services"], dict)

def test_process_audio_invalid_file_type():
    """Test that the audio processing endpoint rejects invalid file types."""
    # Create a test file with incorrect content type
    test_file = io.BytesIO(b"test content")
    response = client.post(
        "/api/v1/audio/process",
        files={"audio": ("test.txt", test_file, "text/plain")},
        data={"duration": 5.0}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_process_audio_success():
    """Test that the audio processing endpoint successfully processes a valid audio file."""
    # Create a mock audio file
    mock_audio = io.BytesIO(b"mock audio data")
    response = client.post(
        "/api/v1/audio/process",
        files={"audio": ("test.wav", mock_audio, "audio/wav")},
        data={"duration": 5.0}
    )
    
    assert response.status_code == 200
    
    # Check that the response has the expected structure
    data = response.json()
    assert "id" in data
    assert "timestamp" in data
    assert "duration" in data
    assert "features" in data
    assert "transcription" in data
    assert "emotion_prediction" in data
    
    # Check that the duration matches what we sent
    assert data["duration"] == 5.0 