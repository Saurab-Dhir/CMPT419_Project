import requests
import base64
from typing import BinaryIO, Any, Optional
from app.core.config import settings
from app.models.audio import AudioTranscription

class STTService:
    def __init__(self):
        self.api_key = settings.LLM_API_KEY  # Using Gemini API key
        self.base_url = "https://generativelanguage.googleapis.com/v1/models"
        self.model = "gemini-2.0-flash-lite"  # Using flash lite model for faster audio transcription
    
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
            
            prompt = "REPLY WITH NOTHING ELSE BUT WHAT THIS AUDIO SAYS WORD BY WORD NOTHING LESS NOTHING MORE, BE ACCURATE AND EXACT TO THE LETTER, DO NOT ADD ANYTHING ELSE, DO NOT SAY ANYTHING ELSE, JUST REPLY WITH THE TEXT OF THE AUDIO"
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
                "generation_config": {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048
                }
            }
            
            response = requests.post(
                url, 
                json=request_data,
                timeout=60  # Longer timeout for audio processing
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract text from response
            if "candidates" in result and len(result["candidates"]) > 0:
                transcribed_text = result["candidates"][0]["content"]["parts"][0]["text"]
                transcribed_text = transcribed_text.strip()
                       
                print(f"Gemini Transcription: {transcribed_text}")
                
                return AudioTranscription(
                    text=transcribed_text,
                    language="en"
                )
            else:
                return ""
                
        except Exception as e:
            print(f"❌ Error in HTTP transcription: {str(e)}")
            return ""
    
    def _process_response(self, raw_response: str) -> Dict[str, Any]:
        """Process raw response text to extract JSON data"""
        if not raw_response:
            return {"transcription": "", "emotion": "neutral"}
            
        try:
            # Clean up markdown formatting if present
            if raw_response.startswith("```json") and raw_response.endswith("```"):
                # Extract JSON from markdown code block
                json_str = raw_response.replace("```json", "").replace("```", "").strip()
            elif raw_response.startswith("{") and raw_response.endswith("}"):
                json_str = raw_response
            else:
                # Try to extract JSON using regex
                json_match = re.search(r'\{.*?\}', raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = raw_response
            
            # Fix common JSON formatting issues
            # 1. Fix line breaks in values that break JSON parsing
            json_str = re.sub(r'"\s*,\s*"', '", "', json_str)
            json_str = re.sub(r'"\s*}\s*,\s*"', '"}, "', json_str)
                    
            parsed_data = json.loads(json_str)
            
            # Ensure we have required fields
            if "transcription" not in parsed_data:
                parsed_data["transcription"] = ""
                
            # Handle both "emotions" and "emotion" fields
            if "emotion" not in parsed_data:
                if "emotions" in parsed_data and "primary" in parsed_data["emotions"]:
                    parsed_data["emotion"] = parsed_data["emotions"]["primary"]
                else:
                    parsed_data["emotion"] = "neutral"
            
            # Simplify response to just transcription and emotion
            return {
                "transcription": parsed_data.get("transcription", ""),
                "emotion": parsed_data.get("emotion", "neutral")
            }
            
        except Exception as e:
            print(f"❌ Error parsing JSON response: {str(e)}")
            print(f"Raw response: {raw_response}")
            
            # Try to extract transcription using regex
            transcription_match = re.search(r'"transcription"\s*:\s*"([^"]*)"', raw_response)
            transcription = transcription_match.group(1) if transcription_match else ""
                
            # Extract emotion if possible
            emotion_match = re.search(r'"primary"\s*:\s*"([^"]*)"', raw_response)
            emotion1 = emotion_match.group(1) if emotion_match else "neutral"
            
            # Try another pattern for emotion
            emotion_match2 = re.search(r'"emotion"\s*:\s*"([^"]*)"', raw_response)
            emotion2 = emotion_match2.group(1) if emotion_match2 else emotion1
            
            return {
                "transcription": transcription,
                "emotion": emotion2
            }

# Create a singleton instance
stt_service = STTService() 