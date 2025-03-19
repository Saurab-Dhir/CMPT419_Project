import os
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
import requests
from app.core.config import settings
from app.utils.logging import response_logger

class LLMService:
    """Service for generating responses using Google's Gemini models."""
    
    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.base_url = "https://generativelanguage.googleapis.com/v1/models"
        
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
            # Return mock response if no API key is provided
            response_text = self._get_mock_response(emotion)
            
            # Log the mock response
            response_logger.log_response(
                response_id=response_id,
                emotion=emotion or "unknown",
                user_text=prompt,
                response_text=response_text
            )
            
            return response_text, response_id
        
        try:
            # Prepare the full prompt with empathetic instructions
            full_prompt = self._prepare_empathetic_prompt(prompt, emotion)
            
            # Construct the API endpoint URL with the model name and API key
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
            
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
            
            # Extract the generated text from the response
            if "candidates" in result and len(result["candidates"]) > 0:
                generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
                response_text = generated_text.strip()
            else:
                # Fallback to mock if no valid response
                response_text = self._get_mock_response(emotion)
            
            # Log the response
            response_logger.log_response(
                response_id=response_id,
                emotion=emotion or "unknown",
                user_text=prompt,
                response_text=response_text,
                metadata={"model": self.model, "temperature": temperature}
            )
            
            return response_text, response_id
            
        except Exception as e:
            print(f"Error in LLM service: {str(e)}")
            # Fallback to mock response on error
            response_text = self._get_mock_response(emotion)
            
            # Log the fallback response
            response_logger.log_response(
                response_id=response_id,
                emotion=emotion or "unknown",
                user_text=prompt,
                response_text=response_text,
                metadata={"error": str(e), "is_fallback": True}
            )
            
            return response_text, response_id
    
    def _prepare_empathetic_prompt(self, prompt: str, emotion: str = None) -> str:
        """Prepare a complete prompt with empathetic instructions."""
        emotion_text = f" who is experiencing {emotion}" if emotion else ""
        
        return f"""You are an empathetic AI assistant designed to provide supportive and helpful responses.

Reply to the above text as an empathetic person. Do this like you're part of the conversation. Don't add any extra information or words just the response.

User's message: "{prompt}"

Keep your response concise (2-3 sentences) and conversational. Be warm, understanding, and genuinely supportive without being judgmental.
"""
    
    def _prepare_prompt(self, prompt: str, emotion: str) -> str:
        """Legacy prompt preparation method maintained for compatibility."""
        return self._prepare_empathetic_prompt(prompt, emotion)
    
    def _get_mock_response(self, emotion: str = None) -> str:
        """Return a mock response for testing or when API is unavailable."""
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