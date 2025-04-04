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
        self.__context = ""
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
    
    async def process_multimodal_input(self, multimodal_input) -> Tuple[str, str]:
        """
        Process a multimodal input with multiple emotion sources and generate a response.
        
        Args:
            multimodal_input: A MultiModalEmotionInput with speech transcription and emotions
            
        Returns:
            Tuple of (generated response text, response_id)
        """
        # Create a unique ID for this response
        response_id = f"gemini_mm_{uuid.uuid4().hex[:10]}"
        
        if not self.api_key:
            print("❌ No LLM API key provided, please set LLM_API_KEY in .env file")
            # Return mock response
            response_text = self._get_mock_multimodal_response(multimodal_input)
            return response_text, response_id
        
        try:
            # Prepare the full prompt with multimodal input
            full_prompt = self._prepare_multimodal_prompt(multimodal_input)
            
            # Construct the API endpoint URL with the model name and API key
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
            print(f"🌐 Sending multimodal request to Gemini API: {self.model}")
            
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
                    "temperature": 0.7,
                    "maxOutputTokens": 300,
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
            print(f"✅ Gemini API multimodal response received, status: {response.status_code}")
            
            # Extract the generated text from the response
            if "candidates" in result and len(result["candidates"]) > 0:
                generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
                response_text = generated_text.strip()
                print(f"✅ Generated text (first 50 chars): {response_text[:50]}...")
            else:
                print("❌ No candidates in Gemini multimodal response")
                # Fallback to mock
                response_text = self._get_mock_multimodal_response(multimodal_input)
            
            # Log the response
            response_logger.log_response(
                response_id=response_id,
                emotion=multimodal_input.semantic_emotion or "unknown",
                user_text=multimodal_input.user_speech,
                response_text=response_text,
                metadata={
                    "model": self.model,
                    "session_id": multimodal_input.session_id,
                    "semantic_emotion": multimodal_input.semantic_emotion,
                    "tonal_emotion": multimodal_input.tonal_emotion,
                    "facial_emotion": multimodal_input.facial_emotion
                }
            )

            # Update context
            self.__context += f"\nUSER: {multimodal_input.user_speech}"
            self.__context += f"\nYOU: {response_text}"
            print("===== CONVERSATION CONTEXT =====")
            print(self.__context)
            print("\n===================================")
            
            return response_text, response_id
            
        except Exception as e:
            print(f"❌ Error in LLM multimodal service: {str(e)}")
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response content: {e.response.text[:200]}")
            
            # Fallback to mock response on error
            response_text = self._get_mock_multimodal_response(multimodal_input)
            
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

    def _prepare_multimodal_prompt(self, multimodal_input) -> str:
        """
        Prepare a prompt for Gemini that includes multimodal emotional information.
        
        Args:
            multimodal_input: A MultiModalEmotionInput with speech and emotions
            
        Returns:
            Formatted prompt string
        """
        # Extract the user's speech and emotions
        user_speech = multimodal_input.user_speech
        fused_emotion = multimodal_input.fused_emotion
        
        # Construct a detailed prompt that includes emotional information
        prompt=f"""Reply to the user as a friend. Consider the following when generating a response:

USER INPUT:
{{
"user_speech": "{user_speech}",
"fused_emotion": {fused_emotion}
}}

USER INPUT DESCRIPTION:
1. The user_speech is a transcrfipt of what they said.
2. The fused_emotion is a combined prediction of emotions based on predictions on individual cues of tone, facial expression, and semantics.

CONTEXT:
{self.__context}

INSTRUCTIONS:
Generate a thoughtful, empathetic response that acknowledges these emotional signals. Account for the context of the conversation and try to maintain the catchball in the conversation (maybe by responding with a followup question or some other way that is natural in casual conversations).
Be natural and keep your response concise (in most cases under 50 words), human-like, and conversational.
Do not explicitly mention that you're aware of their emotions unless it feels natural to do so."""
        
        return prompt

    def _get_mock_multimodal_response(self, multimodal_input) -> str:
        """Generate a mock response for multimodal inputs when API is unavailable."""
        # Extract the user's speech and primary emotion
        user_speech = multimodal_input.user_speech
        primary_emotion = multimodal_input.semantic_emotion or multimodal_input.tonal_emotion or multimodal_input.facial_emotion or "neutral"
        
        # Mock responses based on emotion
        responses = {
            "happy": [
                "That's wonderful to hear! What's been the highlight of your day so far?",
                "I'm glad things are going well. Would you like to tell me more about it?"
            ],
            "sad": [
                "I understand that's difficult. Would you like to talk more about what's happening?",
                "I'm here for you. Take all the time you need to process those feelings."
            ],
            "angry": [
                "That sounds frustrating. What do you think might help in this situation?",
                "I understand why that would be upsetting. Would talking it through help?"
            ],
            "fear": [
                "It's okay to feel nervous about that. What's your biggest concern right now?",
                "I can see why that would cause anxiety. Is there something specific that worries you the most?"
            ],
            "surprise": [
                "Wow, that's unexpected! How are you processing this new information?",
                "That must have been quite a shock. How are you feeling about it now?"
            ],
            "disgust": [
                "That sounds really unpleasant. How are you handling the situation?",
                "I can understand why you'd feel that way. What would make things better?"
            ],
            "neutral": [
                "I see. What else has been on your mind lately?",
                "Thanks for sharing that. Is there anything specific you'd like to discuss?"
            ]
        }
        
        # Default to neutral if emotion not found
        emotion_category = primary_emotion.lower()
        if emotion_category not in responses:
            emotion_category = "neutral"
        
        # Select a random response for the emotion
        import random
        return random.choice(responses[emotion_category])


# Create a singleton instance
llm_service = LLMService() 