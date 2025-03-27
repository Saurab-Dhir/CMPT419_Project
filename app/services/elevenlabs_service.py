import os
import requests
import uuid
from typing import Optional, Dict, Any, Tuple
from app.core.config import settings
from app.utils.logging import response_logger

class ElevenLabsService:
    """Service for Text-to-Speech synthesis using ElevenLabs API."""
    
    def __init__(self):
        self.api_key = settings.ELEVENLABS_API_KEY
        self.base_url = "https://api.elevenlabs.io/v1/text-to-speech"
        self.headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        self.output_dir = "static/audio"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def synthesize_speech(self, text: str, voice_id: Optional[str] = None, response_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Synthesize text to speech using ElevenLabs API.
        
        Args:
            text: The text to convert to speech
            voice_id: The voice ID to use (defaults to settings.TTS_VOICE)
            response_id: Optional ID to associate with this synthesis
            
        Returns:
            Tuple of (static URL path, full file path)
        """
        if not response_id:
            response_id = f"tts_{uuid.uuid4().hex[:10]}"
            
        if not self.api_key:
            # Return mock audio path if no API key is provided
            print("❌ No ElevenLabs API key provided, using mock audio")
            return None, None
        
        try:
            # Get the voice ID to use (default or specified)
            voice_id = voice_id or settings.TTS_VOICE or "21m00Tcm4TlvDq8ikWAM"  # Default to ElevenLabs "Rachel" voice
            
            # Prepare the request data, Note to future self: We need to add more information here to make each 
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            # Make API request to ElevenLabs
            endpoint_url = f"{self.base_url}/{voice_id}"
            
            response = requests.post(
                endpoint_url,
                json=data,
                headers=self.headers
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Generate a unique filename
            timestamp = uuid.uuid4().hex[:6]
            audio_filename = f"{response_id}_{timestamp}.mp3"
            
            # Save to static directory (for API access)
            static_file_path = f"{self.output_dir}/{audio_filename}"
            
            with open(static_file_path, "wb") as f:
                f.write(response.content)
            
            # Also save to output directory for logging
            output_path = response_logger.save_audio_file(
                audio_data=response.content,
                response_id=response_id,
                tts_source="elevenlabs"
            )
            
            # Return the URL path for API responses
            return f"/audio/{audio_filename}", output_path
            
        except Exception as e:
            print(f"❌ Error in ElevenLabs service: {str(e)}")
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response content: {e.response.text[:200]}")
            # Return mock audio path on error
            static_path, full_path = self._get_mock_audio_path(response_id)
            return static_path, full_path
    
    def _get_mock_audio_path(self, response_id: str = "mock") -> Tuple[str, str]:
        """Return mock audio paths for testing."""
        # Create a unique filename for this response
        audio_filename = f"{response_id}_mock.mp3"
        static_file_path = f"{self.output_dir}/{audio_filename}"
        
        # Copy the mock audio file to a unique filename if it doesn't exist
        if not os.path.exists(static_file_path) and os.path.exists(f"{self.output_dir}/mock_tts.mp3"):
            with open(f"{self.output_dir}/mock_tts.mp3", "rb") as src:
                with open(static_file_path, "wb") as dst:
                    dst.write(src.read())
        else:
            # Create a simple MP3 file with minimal data
            mock_audio = b"\xFF\xFB\x90\x44\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            with open(static_file_path, "wb") as f:
                f.write(mock_audio)
        
        return f"/audio/{audio_filename}", static_file_path

# Create singleton instance
elevenlabs_service = ElevenLabsService() 