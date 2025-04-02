import uuid
from datetime import datetime
from typing import Optional, Dict, Any, BinaryIO
import json
import os
from pathlib import Path
from app.models.audio import AudioFeatures, AudioEmotionPrediction, AudioTranscription, AudioProcessingResult
from app.services.stt_service import stt_service
from app.services.tone_service import tone_service
from app.utils.logging import response_logger

class AudioService:
    """Service for processing audio data and extracting features and emotions."""
    
    def __init__(self):
        # Create debug directory for saving audio files
        self.debug_dir = Path("audio_debug")
        self.debug_dir.mkdir(exist_ok=True)
        print(f"✅ Audio debug directory created at: {self.debug_dir.absolute()}")
    
    async def process_audio(self, audio_file: BinaryIO, duration: float) -> AudioProcessingResult:
        """
        Process an audio file to extract features, transcribe speech, and predict emotions.
        
        Args:
            audio_file: The audio file stream
            duration: Duration of the audio in seconds
            
        Returns:
            AudioProcessingResult with extracted features and analysis
        """
        # Create a unique ID for this processing result
        processing_id = f"audio_{uuid.uuid4().hex[:10]}"
        
        # Log that we received audio for classification
        print("Audio was received by classifier")
        
        # Save a copy of the audio file for debugging
        audio_file.seek(0)
        audio_bytes = audio_file.read()
        
        # Determine content type and appropriate extension
        content_type = getattr(audio_file, 'content_type', None)
        if not content_type:
            # Try to infer from filename if available
            if hasattr(audio_file, 'filename'):
                filename = getattr(audio_file, 'filename', '')
                if filename.endswith('.wav'):
                    content_type = 'audio/wav'
                    ext = '.wav'
                elif filename.endswith('.mp3'):
                    content_type = 'audio/mp3'
                    ext = '.mp3'
                elif filename.endswith('.webm'):
                    content_type = 'audio/webm'
                    ext = '.webm'
                else:
                    content_type = 'audio/wav'
                    ext = '.wav'
            else:
                content_type = 'audio/wav'
                ext = '.wav'
        else:
            # Get extension from content type
            if 'webm' in content_type:
                ext = '.webm'
            elif 'mp3' in content_type:
                ext = '.mp3'
            else:
                ext = '.wav'
        
        # Save the file with timestamp and processing ID
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        debug_filename = f"{timestamp}_{processing_id}{ext}"
        debug_path = self.debug_dir / debug_filename
        
        with open(debug_path, 'wb') as f:
            f.write(audio_bytes)
            
        print(f"🔍 DEBUG: Audio saved to {debug_path}")
        
        # Reset file pointer for further processing
        audio_file.seek(0)
        
        # Transcribe speech using our STT service
        transcription_result = await stt_service.transcribe_audio(audio_file)
        
        # Create a transcription object from the result
        transcription = AudioTranscription(
            text=transcription_result.get("transcription", ""),
            language="en",
            raw_response=str(transcription_result)
        )
        
        # Log the transcription
        response_logger.log_transcription(
            audio_id=processing_id,
            text=transcription.text,
            metadata={
                "language": transcription.language, 
                "duration": duration,
                "debug_file_path": str(debug_path)
            }
        )
        
        # Create simple feature placeholder (to maintain structure)
        features = AudioFeatures(
            volume=0.0,
            pitch=0.0,
            speaking_rate=0.0,
            pauses=0,
            tonal_variation=0.0
        )
        
        # Extract emotion from the simplified Gemini response
        semantic_emotion = transcription_result.get("emotion", "neutral")
        print(f"Semantic emotion from Gemini: {semantic_emotion}")
        
        # Use tone service as fallback or for verification
        audio_file.seek(0)
        audio_bytes = audio_file.read()
        
        # Get the tone-based emotion prediction
        tone_emotion_prediction = tone_service.predict_emotion(audio_bytes)
        
        # Use the Gemini emotion if available, otherwise use tone prediction
        if semantic_emotion and semantic_emotion != "neutral":
            emotion_prediction = AudioEmotionPrediction(
                emotion=semantic_emotion,
                confidence=0.9,  # We trust Gemini's emotion detection
                secondary_emotions={
                    "angry": 0.1,
                    "disgust": 0.1,
                    "fearful": 0.1,
                    "happy": 0.1,
                    "sad": 0.1
                }
            )
        else:
            # Use tone service result
            emotion_prediction = tone_emotion_prediction
            
        # If the transcription is empty but we have an emotion, add a placeholder
        if not transcription.text.strip() and semantic_emotion and semantic_emotion != "neutral":
            print("Transcription is empty, using placeholder")
            transcription.text = "(No speech detected)"
        
        # Return the complete processing result with debug info
        result = AudioProcessingResult(
            id=processing_id,
            timestamp=datetime.now(),
            duration=duration,
            features=features,
            transcription=transcription,
            emotion_prediction=emotion_prediction
        )
        
        # Add debug file path to the metadata
        result.metadata = {
            "debug_file_path": str(debug_path)
        }
        
        return result

# Create a singleton instance
audio_service = AudioService() 