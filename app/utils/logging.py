import os
import json
import datetime
from pathlib import Path

class ResponseLogger:
    """Utility to log and save responses from various API calls."""
    
    def __init__(self):
        self.logs_dir = "logs"
        self.transcriptions_dir = os.path.join(self.logs_dir, "transcriptions")
        self.responses_dir = os.path.join(self.logs_dir, "responses")
        self.audio_output_dir = "output/audio"
        
        # Create directories if they don't exist
        for directory in [self.logs_dir, self.transcriptions_dir, self.responses_dir, self.audio_output_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def log_transcription(self, audio_id: str, text: str, metadata: dict = None):
        """Log a transcription result to a file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{audio_id}_transcription.json"
        filepath = os.path.join(self.transcriptions_dir, filename)
        
        data = {
            "audio_id": audio_id,
            "timestamp": timestamp,
            "text": text
        }
        
        if metadata:
            data["metadata"] = metadata
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n===== TRANSCRIPTION [{audio_id}] =====")
        print(f"Text: {text}")
        print("====================================\n")
        
        return filepath
    
    def log_response(self, response_id: str, emotion: str, user_text: str, response_text: str, metadata: dict = None):
        """Log a Gemini response to a file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{response_id}_response.json"
        filepath = os.path.join(self.responses_dir, filename)
        
        data = {
            "response_id": response_id,
            "timestamp": timestamp,
            "emotion": emotion,
            "user_text": user_text,
            "response_text": response_text
        }
        
        if metadata:
            data["metadata"] = metadata
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n===== GEMINI RESPONSE [{response_id}] =====")
        print(f"User: {user_text}")
        print(f"Emotion: {emotion}")
        print(f"Response: {response_text}")
        print("========================================\n")
        
        return filepath
    
    def save_audio_file(self, audio_data: bytes, response_id: str, tts_source: str):
        """Save TTS audio to the output directory."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{response_id}_{tts_source}.mp3"
        filepath = os.path.join(self.audio_output_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        
        print(f"\n===== TTS AUDIO SAVED [{response_id}] =====")
        print(f"File: {filepath}")
        print("===================================\n")
        
        return filepath

# Create a singleton instance
response_logger = ResponseLogger() 