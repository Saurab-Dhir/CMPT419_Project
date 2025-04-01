import base64
import json
import logging
import re
import time
import os
import copy
from typing import Optional, Dict, Any, Union
import io
import asyncio
import requests

# Define global variable for availability
GENAI_AVAILABLE = False

# Try to import Google GenAI library, but gracefully handle if it's not available
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
    print("✅ Google GenAI library imported successfully")
except ImportError as e:
    print(f"⚠️ Google GenAI library not available: {str(e)}. Using fallback HTTP requests.")

class STTService:
    def __init__(self):
        self.api_key = "AIzaSyBgmTHFDl8IPSboKymLcZA4mXdl0USs9Fk"
        if not self.api_key:
            print("⚠️ LLM_API_KEY not set. Audio transcription will not work.")
            self.client = None
        else:
            if GENAI_AVAILABLE:
                # Initialize the Google Gen AI client
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    print("✅ Google Gen AI client initialized")
                except Exception as e:
                    print(f"⚠️ Failed to initialize Google GenAI client: {str(e)}")
                    self.client = None
                    # We can't modify global variables from here
                    # Let's just handle this with a local fallback
            else:
                self.client = None
        
        # Use the latest recommended model
        self.model_name = "gemini-2.0-flash-lite"
        print(f"🎯 Using model: {self.model_name} for audio transcription")
        
    async def transcribe_audio(self, audio_file: Any) -> Dict[str, Any]:
        # Check if we have an API key
        if not self.api_key:
            print("⚠️ No LLM API Key available.")
            return {"transcription": "", "emotion": "neutral"}
            
        # Get audio data and format
        try:
            # Handle different types of audio file inputs:
            # 1. Bytes object
            # 2. BytesIO object
            # 3. File-like objects (including SpooledTemporaryFile)
            
            if isinstance(audio_file, bytes):
                audio_data = audio_file
                content_type = "audio/wav"  # Default if not specified
            elif hasattr(audio_file, 'getvalue'):
                # For BytesIO objects
                audio_data = audio_file.getvalue()
                content_type = getattr(audio_file, 'content_type', "audio/wav")
            elif hasattr(audio_file, 'read'):
                # For file-like objects including SpooledTemporaryFile
                # Save the current position
                try:
                    curr_pos = audio_file.tell()
                    # Go to the beginning of the file
                    audio_file.seek(0)
                    # Read the file
                    audio_data = audio_file.read()
                    # Restore the position
                    audio_file.seek(curr_pos)
                except Exception as e:
                    # If seeking fails, just read the data
                    try:
                        audio_data = audio_file.read()
                    except:
                        audio_file.file.seek(0)
                        audio_data = audio_file.file.read()
                
                # Try to get content type
                content_type = getattr(audio_file, 'content_type', None)
                if not content_type:
                    # Check if it has a filename attribute to guess content type
                    if hasattr(audio_file, 'filename'):
                        if audio_file.filename.endswith('.wav'):
                            content_type = 'audio/wav'
                        elif audio_file.filename.endswith('.mp3'):
                            content_type = 'audio/mp3'
                        elif audio_file.filename.endswith('.webm'):
                            content_type = 'audio/webm'
                        else:
                            content_type = 'audio/wav'  # Default
                    else:
                        content_type = 'audio/wav'  # Default
            else:
                # Unknown type
                return {"transcription": "", "emotion": "neutral", 
                        "error": f"Unsupported audio file type: {type(audio_file)}"}
                
            print(f"🎤 Processing audio: {len(audio_data)/1024:.2f}KB, type: {content_type}")
            
            # Check if audio is empty
            if not audio_data or len(audio_data) < 100:
                print("⚠️ Audio data is empty or too small")
                return {"transcription": "", "emotion": "neutral"}
                
            # Create a unique session ID for this request to avoid cached responses
            session_id = f"{time.time()}_{hash(audio_data) % 10000}"
            
            # Standardize content type
            if "webm" in content_type.lower():
                content_type = "audio/webm"
            elif "mp3" in content_type.lower():
                content_type = "audio/mp3"
            else:
                content_type = "audio/wav"
                
            # Convert to base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Check audio size and truncate if needed (Gemini has ~10MB limit)
            max_size = 9 * 1024 * 1024  # 9MB to leave room for rest of request
            if len(audio_data) > max_size:
                print(f"⚠️ Audio too large ({len(audio_data)/1024/1024:.2f}MB), truncating")
                truncated_size = int(max_size)
                audio_data = audio_data[:truncated_size]
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
            # Create prompt
            prompt = f"""
            You are an AI assistant that transcribes audio and detects emotions.
            
            Session ID: {session_id}
            
            1. Listen to the audio file.
            2. Transcribe the content accurately - exactly what is said, word for word.
            3. Identify the emotions expressed in the speech.
            
            The provided audio is unique and should be processed as a new recording.
            
            Return only a JSON object with the following structure:
            {{
              "transcription": "the transcribed text",
              "emotion": "the main emotion (happy, sad, angry, neutral, etc.)"
            }}
            """
            
            # Start timing the request
            start_time = time.time()
            raw_response = ""
            
            # Use Google GenAI SDK if available, otherwise fall back to HTTP requests
            if GENAI_AVAILABLE and self.client:
                try:
                    # Prepare the parts for the API
                    text_part = types.Part.from_text(prompt)
                    audio_part = types.Part.from_data(
                        mime_type=content_type,
                        data=audio_data
                    )
                    
                    # Create content
                    content = types.Content(
                        role="user",
                        parts=[text_part, audio_part]
                    )
                    
                    print(f"📤 Sending request to Gemini model: {self.model_name}")
                    print(f"📤 Content type: {content_type}, Audio size: {len(audio_data)/1024:.2f}KB")
                    
                    # Use asyncio to run the blocking API call in a thread pool
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=content,
                        generation_config=types.GenerateContentConfig(
                            temperature=0.2,
                            top_p=0.8,
                            top_k=40,
                            max_output_tokens=2048,
                        )
                    )
                    
                    # Extract the response text
                    if not response:
                        print("❌ Empty response from Gemini")
                        return {"transcription": "", "emotion": "neutral"}
                    
                    # Get the raw text response
                    raw_response = response.text
                    
                except Exception as e:
                    print(f"❌ Error using Google GenAI SDK: {str(e)}")
                    # Fall back to HTTP request method
                    raw_response = await self._transcribe_with_http(prompt, content_type, audio_base64)
            else:
                # Fall back to HTTP requests
                raw_response = await self._transcribe_with_http(prompt, content_type, audio_base64)
            
            processing_time = time.time() - start_time
            print(f"⏱️ Gemini processing time: {processing_time:.2f} seconds")
            print(f"📥 Response received from Gemini: {raw_response[:100]}...")
            
            # Process the raw response to extract JSON
            return self._process_response(raw_response)
            
        except Exception as e:
            error_msg = f"STT service error: {str(e)}"
            print(f"❌ {error_msg}")
            return {"transcription": "", "emotion": "neutral", "error": error_msg}
    
    async def _transcribe_with_http(self, prompt: str, content_type: str, audio_base64: str) -> str:
        """Fallback method using direct HTTP requests when Google GenAI SDK is not available"""
        try:
            # Construct the API endpoint URL
            url = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:generateContent?key={self.api_key}"
            
            # Prepare request data
            request_data = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": content_type,
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
            
            # Log request (without the audio data)
            log_data = copy.deepcopy(request_data)
            log_data["contents"][0]["parts"][1]["inline_data"]["data"] = f"[{len(audio_base64)} bytes]"
            print(f"📤 Sending HTTP request to Gemini: {json.dumps(log_data, indent=2)}")
            
            # Send request
            response = requests.post(
                url, 
                json=request_data,
                timeout=60  # Longer timeout for audio processing
            )
            
            # Check if request was successful
            if response.status_code != 200:
                error_msg = f"Gemini API error: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return ""
            
            # Parse response
            result = response.json()
            
            # Extract text from response
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
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