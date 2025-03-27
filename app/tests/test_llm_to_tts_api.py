import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app.main import app
from app.models.response import LLMInput

# Create test client
client = TestClient(app)

# Sample test data
TEST_LLM_INPUT = {
    "text": "I'm feeling anxious about my upcoming presentation.",
    "emotion": "anxiety",
    "session_id": "test_session_123",
    "max_tokens": 300,
    "temperature": 0.7
}

@pytest.fixture
def mock_llm_service():
    """Create a mock for the LLM service."""
    with patch("app.routers.llm_to_tts.llm_service.process_llm_input") as mock:
        mock.return_value = AsyncMock(return_value=("This is a mock LLM response.", "mock_response_123"))
        yield mock

@pytest.fixture
def mock_elevenlabs_service():
    """Create a mock for the ElevenLabs service."""
    with patch("app.routers.llm_to_tts.elevenlabs_service.synthesize_speech") as mock:
        mock.return_value = AsyncMock(return_value=("/audio/mock_audio.mp3", "/full/path/to/audio.mp3"))
        yield mock

def test_process_llm_to_tts(mock_llm_service, mock_elevenlabs_service):
    """Test the synchronous LLM-to-TTS workflow endpoint."""
    response = client.post("/api/v1/llm-to-tts/process", json=TEST_LLM_INPUT)
    
    # Check response status and structure
    assert response.status_code == 200
    data = response.json()
    assert "response_id" in data
    assert "llm_text" in data
    assert "audio_url" in data
    assert "session_id" in data
    assert data["session_id"] == TEST_LLM_INPUT["session_id"]
    
    # Verify service calls
    mock_llm_service.assert_called_once()
    mock_elevenlabs_service.assert_called_once()

def test_process_llm_to_tts_async(mock_llm_service, mock_elevenlabs_service):
    """Test the asynchronous LLM-to-TTS workflow endpoint."""
    response = client.post("/api/v1/llm-to-tts/process-async", json=TEST_LLM_INPUT)
    
    # Check response status and structure
    assert response.status_code == 200
    data = response.json()
    assert "response_id" in data
    assert "llm_text" in data
    assert "session_id" in data
    assert data["session_id"] == TEST_LLM_INPUT["session_id"]
    
    # For async, audio_url should be None initially
    assert data["audio_url"] is None
    
    # Verify LLM service was called
    mock_llm_service.assert_called_once()
    
    # ElevenLabs service will be called in background, so not immediately verifiable

def test_workflow_status():
    """Test the workflow status endpoint."""
    response = client.get("/api/v1/llm-to-tts/status")
    
    # Check response status and structure
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "workflows" in data
    assert isinstance(data["workflows"], list)
    assert len(data["workflows"]) == 2  # Should have sync and async workflows

def test_invalid_input():
    """Test the endpoint with invalid input."""
    # Missing required fields
    invalid_input = {
        "text": "Test message",
        # Missing session_id which is required
    }
    
    response = client.post("/api/v1/llm-to-tts/process", json=invalid_input)
    assert response.status_code == 422  # Validation error 