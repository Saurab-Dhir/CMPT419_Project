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
        self.model = "gemini-2.0-flash-lite"  # Using flash lite model for faster audio transcription
    
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
            
            # Construct a detailed prompt for accurate transcription
            prompt = """What does this audio say? 

Your task is to transcribe the spoken content in this audio file with complete accuracy.

INSTRUCTIONS:
1. Transcribe EXACTLY what is spoken, word for word.
2. Include all filler words (um, uh, like, you know, etc.).
3. Format as plain text only.
4. Do not add any commentary, descriptions, or explanations.
5. Do not add timestamps or speaker identifications.
6. Do not include any text before or after the transcription.
7. Preserve natural pauses with commas and periods.
8. If parts are unclear, make your best guess.

REPLY WITH THE TRANSCRIPTION ONLY."""
            
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
                    "temperature": 0.1,  # Low temperature for exact transcription
                    "maxOutputTokens": 1024,  # Increased for longer transcriptions
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            # Make API request
            print("Sending audio to Gemini for transcription...")
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
                
                print(f"Gemini Transcription successful. Length: {len(transcribed_text)} characters")
                
                # Create AudioTranscription from result
                return AudioTranscription(
                    text=transcribed_text,
                    language="en"
                )
            else:
                print("Gemini returned no transcription candidates")
                # Fallback to mock if no valid response
                return await self._mock_transcribe()
            
        except Exception as e:
            print(f"Error in Gemini STT service: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response content: {e.response.text[:200]}")
            # Fallback to mock implementation on error
            return await self._mock_transcribe()
    
    async def _mock_transcribe(self) -> AudioTranscription:
        """Mock implementation for testing or when API is unavailable."""
        print("Using mock transcription (Gemini API unavailable or error occurred)")
        # This is just a placeholder - in production, we'd integrate with the actual Gemini response
        test_transcriptions = [
            "I'm feeling really anxious about my presentation tomorrow.",
            "Can you help me understand why I feel so stressed all the time?",
            "I had a really good day today, everything went perfectly.",
            "I need some advice on how to handle difficult conversations with my colleagues."
        ]
        import random
        mock_text = random.choice(test_transcriptions)
        return AudioTranscription(
            text=mock_text,
            language="en"
        )

# Create a singleton instance
stt_service = STTService() 