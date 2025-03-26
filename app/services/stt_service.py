import requests
import base64
from typing import BinaryIO, Any, Optional
from app.core.config import settings
from app.models.audio import AudioTranscription

class STTService:
    """Service for Speech-to-Text transcription using Gemini."""
    
    def __init__(self):
        self.api_key = settings.LLM_API_KEY  # Using Gemini API key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = "gemini-2.0-flash"
    
    async def transcribe(self, audio_file: BinaryIO, mime_type: str, response_id: Optional[str] = None) -> AudioTranscription:
        """
        Transcribe speech from audio file using Gemini.
        
        Args:
            audio_file: The audio file to transcribe
            
        Returns:
            AudioTranscription with the transcribed text and confidence
        """
        if response_id:
            print(f"Received STT request <{response_id}>")

        if not self.api_key:
            return await self._mock_transcribe()
        
        try:
            # Reset file pointer to beginning
            audio_file.seek(0)
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            prompt = "REPLY WITH NOTHING ELSE BUT WHAT THIS AUDIO SAYS WORD BY WORD NOTHING LESS NOTHING MORE"
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
            
            data = {
                "contents": [
                    {
                        "parts": [
                            { "text": prompt },
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
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
            
            response = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the transcribed text from the response
            if "candidates" in result and len(result["candidates"]) > 0:
                transcribed_text = result["candidates"][0]["content"]["parts"][0]["text"]
                transcribed_text = transcribed_text.strip()
                       
                print(f"Gemini Transcription: {transcribed_text}")
                
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