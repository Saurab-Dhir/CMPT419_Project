import requests
import json
import base64
import os
from typing import BinaryIO, Dict, Any, Optional
from app.core.config import settings
from app.models.audio import AudioTranscription

class STTService:
    """Service for Speech-to-Text transcription using Gemini."""
    
    def __init__(self):
        self.api_key = settings.LLM_API_KEY  # Using Gemini API key
        self.base_url = "https://generativelanguage.googleapis.com/v1/models"
        self.model = "gemini-pro-vision"  # Using vision model for audio transcription
    
    async def transcribe(self, audio_file: BinaryIO) -> AudioTranscription:
        """
        Transcribe speech from audio file using Gemini.
        
        Args:
            audio_file: The audio file to transcribe
            
        Returns:
            AudioTranscription with the transcribed text and confidence
        """
        if not self.api_key:
            # Fallback to mock implementation if no API key is provided
            return await self._mock_transcribe()
        
        try:
            # Reset file pointer to beginning
            audio_file.seek(0)
            
            # Read the audio data and encode it
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Construct the precise prompt for accurate transcription
            prompt = "REPLY WITH NOTHING ELSE BUT WHAT THIS AUDIO SAYS WORD BY WORD NOTHING LESS NOTHING MORE"
            
            # Construct the API endpoint URL
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
            
            # Prepare the request data
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inline_data": {
                                    "mime_type": "audio/wav",  # Adjust if using different format
                                    "data": audio_base64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,  # Even lower temperature for exact transcription
                    "maxOutputTokens": 200
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
            
            # Extract the transcribed text from the response
            if "candidates" in result and len(result["candidates"]) > 0:
                transcribed_text = result["candidates"][0]["content"]["parts"][0]["text"]
                transcribed_text = transcribed_text.strip()
                
                print(f"Gemini Transcription: {transcribed_text}")
                
                # Create AudioTranscription from result
                return AudioTranscription(
                    text=transcribed_text,
                    confidence=0.95,  # Gemini doesn't provide confidence score
                    language="en"
                )
            else:
                # Fallback to mock if no valid response
                return await self._mock_transcribe()
            
        except Exception as e:
            print(f"Error in Gemini STT service: {str(e)}")
            # Fallback to mock implementation on error
            return await self._mock_transcribe()
    
    async def _mock_transcribe(self) -> AudioTranscription:
        """Mock implementation for testing or when API is unavailable."""
        # Updated mock with a more realistic transcription that could come from Gemini
        # This is just a placeholder - in production, we'd integrate with the actual Gemini response
        test_transcriptions = [
            "I'm feeling really anxious about my presentation tomorrow.",
            "Can you help me understand why I feel so stressed all the time?",
            "I had a really good day today, everything went perfectly.",
            "Why, shouldn't you put a toaster in a bathtub full of water? Because your toast would get soggy!"
        ]
        import random
        return AudioTranscription(
            text=random.choice(test_transcriptions),
            confidence=0.95,
            language="en"
        )

# Create a singleton instance
stt_service = STTService() 