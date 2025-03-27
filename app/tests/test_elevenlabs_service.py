import pytest
from unittest.mock import patch, MagicMock
import os
import requests
from app.services.elevenlabs_service import ElevenLabsService

# Test data
TEST_TEXT = "This is a test response for speech synthesis."
TEST_VOICE_ID = "test_voice_123"
TEST_RESPONSE_ID = "test_response_123"

@pytest.fixture
def elevenlabs_service():
    """Create an ElevenLabsService instance for testing."""
    return ElevenLabsService()

@pytest.fixture
def mock_response():
    """Create a mock response for requests.post."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"\xFF\xFB\x90\x44\x00\x00\x00\x00"  # Minimal mock MP3 data
    return mock_resp

@pytest.mark.asyncio
async def test_synthesize_speech_success(elevenlabs_service, mock_response):
    """Test successful speech synthesis."""
    with patch("app.services.elevenlabs_service.requests.post", return_value=mock_response):
        static_path, full_path = await elevenlabs_service.synthesize_speech(
            text=TEST_TEXT,
            voice_id=TEST_VOICE_ID,
            response_id=TEST_RESPONSE_ID
        )
        
        # Check response format
        assert isinstance(static_path, str)
        assert isinstance(full_path, str)
        assert static_path.startswith("/audio/")
        assert TEST_RESPONSE_ID in static_path
        assert os.path.exists(full_path)
        
        # Cleanup test file
        if os.path.exists(full_path):
            os.remove(full_path)

@pytest.mark.asyncio
async def test_synthesize_speech_failure(elevenlabs_service):
    """Test fallback to mock audio on API failure."""
    with patch("app.services.elevenlabs_service.requests.post", side_effect=requests.exceptions.RequestException("API error")):
        static_path, full_path = await elevenlabs_service.synthesize_speech(
            text=TEST_TEXT,
            voice_id=TEST_VOICE_ID,
            response_id=TEST_RESPONSE_ID
        )
        
        # Check response format
        assert isinstance(static_path, str)
        assert isinstance(full_path, str)
        assert static_path.startswith("/audio/")
        assert "mock" in static_path
        assert os.path.exists(full_path)
        
        # Cleanup test file
        if os.path.exists(full_path):
            os.remove(full_path)

@pytest.mark.asyncio
async def test_synthesize_speech_no_api_key(elevenlabs_service):
    """Test fallback to mock audio when no API key is provided."""
    # Temporarily set API key to None
    original_api_key = elevenlabs_service.api_key
    elevenlabs_service.api_key = None
    
    try:
        static_path, full_path = await elevenlabs_service.synthesize_speech(
            text=TEST_TEXT,
            voice_id=TEST_VOICE_ID,
            response_id=TEST_RESPONSE_ID
        )
        
        # Check response format
        assert isinstance(static_path, str)
        assert isinstance(full_path, str)
        assert static_path.startswith("/audio/")
        assert "mock" in static_path
        assert os.path.exists(full_path)
        
        # Cleanup test file
        if os.path.exists(full_path):
            os.remove(full_path)
    finally:
        # Restore original API key
        elevenlabs_service.api_key = original_api_key

@pytest.mark.asyncio
async def test_synthesize_speech_auto_response_id(elevenlabs_service, mock_response):
    """Test auto-generation of response_id when none is provided."""
    with patch("app.services.elevenlabs_service.requests.post", return_value=mock_response):
        static_path, full_path = await elevenlabs_service.synthesize_speech(
            text=TEST_TEXT,
            voice_id=TEST_VOICE_ID
        )
        
        # Check response format
        assert isinstance(static_path, str)
        assert isinstance(full_path, str)
        assert static_path.startswith("/audio/")
        assert "tts_" in static_path  # Should have auto-generated ID
        assert os.path.exists(full_path)
        
        # Cleanup test file
        if os.path.exists(full_path):
            os.remove(full_path) 