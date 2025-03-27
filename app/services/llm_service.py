import os
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
import requests
from app.core.config import settings
from app.utils.logging import response_logger
from app.models.response import LLMInput

class LLMService:
    """Service for generating responses using Google's Gemini models."""
    
    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.base_url = "https://generativelanguage.googleapis.com/v1/models"
        print(f"🔑 LLM API Key present: {bool(self.api_key)}")
        print(f"🤖 Using LLM model: {self.model}")
        
    async def generate_response(
        self, 
        prompt: str, 
        emotion: str = None, 
        temperature: float = 0.7,
        max_tokens: int = 300
    ) -> Tuple[str, str]:
        """
        Generate an empathetic response using Gemini.
        
        Args:
            prompt: The user's text to respond to
            emotion: Optional detected emotion to address (not required)
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Tuple of (generated response text, response_id)
        """
        # Create a unique ID for this response
        response_id = f"gemini_{uuid.uuid4().hex[:10]}"
        
        if not self.api_key:
            print("❌ No LLM API key provided, please set LLM_API_KEY in .env file")
            # Return mock response if no API key is provided
            response_text = self._get_mock_response(emotion)
            
            # Log the mock response
            response_logger.log_response(
                response_id=response_id,
                emotion=emotion or "unknown",
                user_text=prompt,
                response_text=response_text,
                metadata={"is_mock": True, "reason": "No API key"}
            )
            
            return response_text, response_id
        
        try:
            # Prepare the full prompt with empathetic instructions
            full_prompt = self._prepare_empathetic_prompt(prompt, emotion)
            
            # Construct the API endpoint URL with the model name and API key
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
            print(f"🌐 Sending request to Gemini API: {self.model}")
            
            # Prepare the request data
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": full_prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            # Make API request
            response = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for successful response
            response.raise_for_status()
            result = response.json()
            
            # Debug the response
            print(f"✅ Gemini API response received, status: {response.status_code}")
            
            # Extract the generated text from the response
            if "candidates" in result and len(result["candidates"]) > 0:
                generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
                response_text = generated_text.strip()
                print(f"✅ Generated text (first 50 chars): {response_text[:50]}...")
            else:
                print("❌ No candidates in Gemini response, response data:")
                print(json.dumps(result, indent=2)[:200] + "...")
                # Fallback to mock if no valid response
                response_text = self._get_mock_response(emotion)
                print("⚠️ Using mock response as fallback")
            
            # Log the response
            response_logger.log_response(
                response_id=response_id,
                emotion=emotion or "unknown",
                user_text=prompt,
                response_text=response_text,
                metadata={
                    "model": self.model, 
                    "temperature": temperature,
                    "is_mock": "candidates" not in result or len(result["candidates"]) == 0
                }
            )
            
            return response_text, response_id
            
        except Exception as e:
            print(f"❌ Error in LLM service: {str(e)}")
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response content: {e.response.text[:200]}")
            
            # Fallback to mock response on error
            response_text = self._get_mock_response(emotion)
            print("⚠️ Using mock response due to error")
            
            # Log the fallback response
            response_logger.log_response(
                response_id=response_id,
                emotion=emotion or "unknown",
                user_text=prompt,
                response_text=response_text,
                metadata={"error": str(e), "is_fallback": True}
            )
            
            return response_text, response_id
    
    async def process_llm_input(self, llm_input: LLMInput) -> Tuple[str, str]:
        """
        Process a structured LLM input and generate a response using Gemini.
        
        Args:
            llm_input: A structured input containing text, emotion, and other parameters
            
        Returns:
            Tuple of (generated response text, response_id)
        """
        # Create a unique ID for this response
        response_id = f"gemini_{uuid.uuid4().hex[:10]}"
        
        if not self.api_key:
            print("❌ No LLM API key provided, please set LLM_API_KEY in .env file")
            # Return mock response if no API key is provided
            response_text = self._get_mock_response(llm_input.emotion)
            
            # Log the mock response
            response_logger.log_response(
                response_id=response_id,
                emotion=llm_input.emotion or "unknown",
                user_text=llm_input.text,
                response_text=response_text,
                metadata={"is_mock": True, "reason": "No API key"}
            )
            
            return response_text, response_id
        
        try:
            # Prepare the full prompt with empathetic instructions
            full_prompt = self._prepare_empathetic_prompt(llm_input.text, llm_input.emotion)
            
            # Construct the API endpoint URL with the model name and API key
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
            print(f"🌐 Sending request to Gemini API: {self.model}")
            
            # Prepare the request data
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": full_prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": llm_input.temperature,
                    "maxOutputTokens": llm_input.max_tokens,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            # Make API request
            response = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for successful response
            response.raise_for_status()
            result = response.json()
            
            # Debug the response
            print(f"✅ Gemini API response received, status: {response.status_code}")
            
            # Extract the generated text from the response
            if "candidates" in result and len(result["candidates"]) > 0:
                generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
                response_text = generated_text.strip()
                print(f"✅ Generated text (first 50 chars): {response_text[:50]}...")
            else:
                print("❌ No candidates in Gemini response, response data:")
                print(json.dumps(result, indent=2)[:200] + "...")
                # Fallback to mock if no valid response
                response_text = self._get_mock_response(llm_input.emotion)
                print("⚠️ Using mock response as fallback")
            
            # Log the response
            response_logger.log_response(
                response_id=response_id,
                emotion=llm_input.emotion or "unknown",
                user_text=llm_input.text,
                response_text=response_text,
                metadata={
                    "model": self.model, 
                    "temperature": llm_input.temperature,
                    "session_id": llm_input.session_id,
                    "is_mock": "candidates" not in result or len(result["candidates"]) == 0
                }
            )
            
            return response_text, response_id
            
        except Exception as e:
            print(f"❌ Error in LLM service: {str(e)}")
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response content: {e.response.text[:200]}")
            
            # Fallback to mock response on error
            response_text = self._get_mock_response(llm_input.emotion)
            print("⚠️ Using mock response due to error")
            
            # Log the fallback response
            response_logger.log_response(
                response_id=response_id,
                emotion=llm_input.emotion or "unknown",
                user_text=llm_input.text,
                response_text=response_text,
                metadata={"error": str(e), "is_fallback": True}
            )
            
            return response_text, response_id
    
    def _prepare_empathetic_prompt(self, prompt: str, emotion: str = None) -> str:
        """Prepare a complete prompt with empathetic instructions."""
        emotion_guidance = f"The user is experiencing {emotion}. Be particularly sensitive to this emotion in your response." if emotion else "Respond with general empathy."
        
        return f"""You are an empathetic AI assistant designed to provide supportive and helpful responses.

Your task is to respond to the following message with genuine empathy and understanding.

User's message: "{prompt}"

{emotion_guidance}

Keep your response concise (2-3 sentences), conversational, and genuinely supportive. Be warm and understanding without being judgmental. Don't include phrases like "I understand" or "I'm sorry to hear that" - instead, show empathy through your specific response to their situation.
"""
    
    def _get_mock_response(self, emotion: str = None) -> str:
        """Return a mock response for testing or when API is unavailable."""
        print("⚠️ Using mock response - this should only happen when API is unavailable")
        responses = {
            "anxiety": "I notice you're feeling anxious. That's completely understandable - we all feel anxious at times. What specific aspect feels most overwhelming right now?",
            "sadness": "I can hear that you're feeling sad. It's okay to feel this way, and giving yourself space to experience these emotions is important. What might help you feel a bit more supported right now?",
            "anger": "I can sense your frustration. It's valid to feel angry when things don't go as expected. Would it help to talk about what triggered these feelings?",
            "fear": "Feeling afraid is a natural response when facing uncertainty. Your concerns are valid. What small step might help you feel a bit more secure in this situation?",
            "joy": "It's wonderful to hear you're feeling happy! These positive moments are worth savoring. What aspect of this experience feels most meaningful to you?",
            "default": "I'm here to listen and support you. Your feelings are valid, and I appreciate you sharing them. How can I best support you right now?"
        }
        
        if not emotion:
            return responses["default"]
        return responses.get(emotion.lower(), responses["default"])

# Create a singleton instance
llm_service = LLMService() 