import uuid
from datetime import datetime
from typing import Optional, Dict, Any, BinaryIO
from app.models.audio import AudioFeatures, AudioEmotionPrediction, AudioTranscription, AudioProcessingResult
from app.services.stt_service import stt_service
from app.services.tone_service import tone_service
from app.utils.logging import response_logger

class AudioService:
    """Service for processing audio data and extracting features and emotions."""
    
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
        
        # Transcribe speech using our STT service
        transcription = await stt_service.transcribe(audio_file)
        
        # Log the transcription
        response_logger.log_transcription(
            audio_id=processing_id,
            text=transcription.text,
            metadata={"language": transcription.language, "duration": duration}
        )
        
        # Create simple feature placeholder (to maintain structure)
        features = AudioFeatures(
            volume=0.0,
            pitch=0.0,
            speaking_rate=0.0,
            pauses=0,
            tonal_variation=0.0
        )
        
        # Extract tone
        audio_file.seek(0)
        audio_bytes = audio_file.read()
        emotion_prediction = tone_service.predict_emotion(audio_bytes)

        
        # Return the complete processing result
        return AudioProcessingResult(
            id=processing_id,
            timestamp=datetime.now(),
            duration=duration,
            features=features,
            transcription=transcription,
            emotion_prediction=emotion_prediction
        )

# Create a singleton instance
audio_service = AudioService() 