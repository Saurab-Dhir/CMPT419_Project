import pytest
from unittest.mock import patch, MagicMock
from app.services.llm_service import LLMService
from app.models.response import LLMInput
import requests

# Test data
TEST_TEXT = "I'm feeling really stressed about my upcoming presentation."
TEST_EMOTION = "anxiety"
TEST_SESSION_ID = "test_session_123"

@pytest.fixture
def llm_service():
    """Create a LLMService instance for testing."""
    return LLMService()

@pytest.fixture
def mock_response():
    """Create a mock response for requests.post."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "I understand how stressful presentations can be. Remember that you've prepared well, and it's normal to feel nervous. Would it help to practice once more?"
                        }
                    ]
                }
            }
        ]
    }
    return mock_resp

@pytest.fixture
def llm_input():
    """Create a test LLMInput object."""
    return LLMInput(
        text=TEST_TEXT,
        emotion=TEST_EMOTION,
        session_id=TEST_SESSION_ID,
        max_tokens=300,
        temperature=0.7
    )

@pytest.mark.asyncio
async def test_process_llm_input_success(llm_service, llm_input, mock_response):
    """Test successful LLM response generation."""
    with patch("app.services.llm_service.requests.post", return_value=mock_response):
        response_text, response_id = await llm_service.process_llm_input(llm_input)
        
        # Check response format
        assert isinstance(response_text, str)
        assert isinstance(response_id, str)
        assert response_id.startswith("gemini_")
        
        # Check response content matches mock response
        expected_text = "I understand how stressful presentations can be. Remember that you've prepared well, and it's normal to feel nervous. Would it help to practice once more?"
        assert response_text == expected_text

@pytest.mark.asyncio
async def test_process_llm_input_failure(llm_service, llm_input):
    """Test fallback to mock response on API failure."""
    with patch("app.services.llm_service.requests.post", side_effect=requests.exceptions.RequestException("API error")):
        response_text, response_id = await llm_service.process_llm_input(llm_input)
        
        # Check response format
        assert isinstance(response_text, str)
        assert isinstance(response_id, str)
        assert response_id.startswith("gemini_")
        
        # Check response is a mock response related to anxiety
        assert "anxiety" in response_text.lower()

@pytest.mark.asyncio
async def test_process_llm_input_no_api_key(llm_service, llm_input):
    """Test fallback to mock response when no API key is provided."""
    # Temporarily set API key to None
    original_api_key = llm_service.api_key
    llm_service.api_key = None
    
    try:
        response_text, response_id = await llm_service.process_llm_input(llm_input)
        
        # Check response format
        assert isinstance(response_text, str)
        assert isinstance(response_id, str)
        assert response_id.startswith("gemini_")
        
        # Check for mock response content
        assert "anxiety" in response_text.lower()
    finally:
        # Restore original API key
        llm_service.api_key = original_api_key

@pytest.mark.asyncio
async def test_process_llm_input_empty_response(llm_service, llm_input):
    """Test fallback to mock when API returns empty response."""
    # Create a mock with empty candidates
    empty_mock = MagicMock()
    empty_mock.status_code = 200
    empty_mock.json.return_value = {"candidates": []}
    
    with patch("app.services.llm_service.requests.post", return_value=empty_mock):
        response_text, response_id = await llm_service.process_llm_input(llm_input)
        
        # Check response format
        assert isinstance(response_text, str)
        assert isinstance(response_id, str)
        assert response_id.startswith("gemini_")
        
        # Should be a mock response
        assert "anxiety" in response_text.lower() 