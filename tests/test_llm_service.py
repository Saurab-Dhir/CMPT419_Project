import asyncio
import sys
import os

# Add the current directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.llm_service import llm_service
from app.models.response import LLMInput

async def test_llm_service():
    """Test the LLM service by generating a response."""
    print("=== LLM Service Test ===")
    print(f"API Key present: {bool(llm_service.api_key)}")
    print(f"Using model: {llm_service.model}")
    
    # Test with a sample input
    test_prompt = "I'm feeling really stressed about my upcoming exam."
    
    print("\n=== Testing with direct prompt ===")
    print(f"Input prompt: {test_prompt}")
    response_text, response_id = await llm_service.generate_response(test_prompt, emotion="anxiety")
    print(f"Response ID: {response_id}")
    print(f"Response text: {response_text}")
    
    # Test with a structured input
    print("\n=== Testing with structured input ===")
    llm_input = LLMInput(
        text="I'm worried about my job interview tomorrow.",
        emotion="anxiety",
        temperature=0.7,
        max_tokens=300,
        session_id="test_session"
    )
    print(f"Input: {llm_input}")
    response_text, response_id = await llm_service.process_llm_input(llm_input)
    print(f"Response ID: {response_id}")
    print(f"Response text: {response_text}")
    
if __name__ == "__main__":
    asyncio.run(test_llm_service()) 