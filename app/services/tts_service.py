import os
import requests
import uuid
import json
from typing import Optional, Dict, Any, BinaryIO, Tuple
from app.core.config import settings
from app.utils.logging import response_logger

class TTSService:
    """Service for Text-to-Speech synthesis using ElevenLabs API."""
    
    def __init__(self):
        self.api_key = settings.ELEVENLABS_API_KEY
        self.base_url = "https://api.elevenlabs.io/v1/text-to-speech"
        self.headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        self.output_dir = "static/audio"
        
        # Debug output for API key
        print(f"🔑 ElevenLabs API Key present: {bool(self.api_key)}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a mock audio file for testing
        self._create_mock_audio_file()
    
    async def synthesize(self, text: str, voice: Optional[str] = None, response_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Synthesize text to speech using ElevenLabs API.
        
        Args:
            text: The text to convert to speech
            voice: The voice to use (defaults to settings.TTS_VOICE)
            response_id: Optional ID to associate with this synthesis
            
        Returns:
            Tuple of (static URL path, full file path)
        """
        if not response_id:
            response_id = f"tts_{uuid.uuid4().hex[:10]}"
            
        print(f"🎙️ TTS Request: '{text[:30]}...' with API key: {bool(self.api_key)}")
            
        if not self.api_key:
            # Return mock audio path if no API key is provided
            print("❌ No ElevenLabs API key provided, using mock audio")
            static_path, full_path = self._get_mock_audio_path(response_id)
            return static_path, full_path
        
        try:
            # Get the voice ID to use (default or specified)
            voice_id = voice or settings.TTS_VOICE or "21m00Tcm4TlvDq8ikWAM"  # Default to ElevenLabs "Rachel" voice
            
            # Prepare the request data
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
            print(f"🌍 Making request to ElevenLabs: {endpoint_url}")
            
            response = requests.post(
                endpoint_url,
                json=data,
                headers=self.headers
            )
            
            # Check for successful response
            response.raise_for_status()
            print(f"✅ ElevenLabs API response received: {response.status_code}")
            
            # Generate a unique filename
            timestamp = uuid.uuid4().hex[:6]
            audio_filename = f"{response_id}_{timestamp}.mp3"
            
            # Save to both static directory (for API access) and output directory (for review)
            static_file_path = f"{self.output_dir}/{audio_filename}"
            
            with open(static_file_path, "wb") as f:
                f.write(response.content)
            
            # Also save to output directory for review
            output_path = response_logger.save_audio_file(
                audio_data=response.content,
                response_id=response_id,
                tts_source="elevenlabs"
            )
            
            # Return the URL path for API responses
            return f"/audio/{audio_filename}", output_path
            
        except Exception as e:
            print(f"❌ Error in TTS service: {str(e)}")
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
        
        return f"/audio/{audio_filename}", static_file_path
    
    def _create_mock_audio_file(self):
        """Create a mock audio file for testing if it doesn't exist."""
        mock_file_path = f"{self.output_dir}/mock_tts.mp3"
        if not os.path.exists(mock_file_path):
            # Create a simple MP3 file with minimal data
            # This is just a placeholder and won't play actual audio
            mock_audio = b"\xFF\xFB\x90\x44\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            with open(mock_file_path, "wb") as f:
                f.write(mock_audio)
                
            # Also save to output directory
            response_logger.save_audio_file(
                audio_data=mock_audio,
                response_id="mock",
                tts_source="mock"
            )

# Create a singleton instance
tts_service = TTSService() 