import uuid
from typing import Optional, Dict, Any, BinaryIO, Tuple
import json
from datetime import datetime
import asyncio

# from app.models.audio import AudioProcessingResult
# from app.models.visual import VisualProcessingResult
from app.models.response import MultiModalEmotionInput, LLMToTTSResponse
from app.services.audio_service import audio_service
from app.services.visual_service import VisualService
from app.services.stt_service import stt_service
from app.services.llm_service import llm_service
from app.services.elevenlabs_service import elevenlabs_service
from multimodal_classification.multimodal_classification_model import Evidence, MultiModalClassifier
# from app.utils.logging import response_logger

class MultiModalService:
    """Service for processing multimodal inputs (audio + video) and generating responses."""
    
    def __init__(self):
        """Initialize the multimodal service."""
        self.visual_service = VisualService()
        print("🔄 MultiModalService initialized")
    
    async def process_multimodal_input(
        self, 
        audio_file: BinaryIO, 
        video_file: BinaryIO, 
        audio_duration: float,
        session_id: str,
        file_extension: str = "jpg"
    ) -> LLMToTTSResponse:
        """
        Process audio and video input to generate a response.
        
        Args:
            audio_file: The audio file stream
            video_file: The video file (or image) stream
            audio_duration: Duration of the audio in seconds
            session_id: Session identifier for tracking
            file_extension: The file extension for the video/image
            
        Returns:
            LLMToTTSResponse with generated text and audio URL
        """
        # Create a unique ID for this processing
        processing_id = f"mm_{uuid.uuid4().hex[:10]}"
        print(f"🎬 Starting multimodal processing {processing_id}")
        
        try:
            # Process audio to get transcription and tonal emotion
            audio_result = await audio_service.process_audio(audio_file, audio_duration)
            
            # Extract speech transcription
            transcription = audio_result.transcription.text
            if not transcription or transcription.strip() == "":
                transcription = "(No speech detected)"
            
            # The emotion is now directly available from audio_result
            semantic_emotion = audio_result.emotion_prediction.emotion
            
            # Extract tonal emotion (same as semantic in the new implementation)
            tonal_emotion = semantic_emotion
            
            # Process video to get facial emotion
            visual_result = await self.visual_service.process_image(video_file, file_extension)
            
            # Extract facial emotion
            facial_emotion = "neutral"
            if visual_result.emotion_prediction:
                facial_emotion = visual_result.emotion_prediction.emotion
            
            print(f"🔍 Detected emotions - Semantic: {semantic_emotion}, Tonal: {tonal_emotion}, Facial: {facial_emotion}")
            
            # Check if we should proceed with empty transcription
            if transcription == "(No speech detected)":
                print("⚠️ Empty transcription detected, proceeding with minimal input")
            
            # ======= START OF INSERTION
            tone = Evidence(
                emotion=tonal_emotion, 
                confidence=audio_result.emotion_prediction.confidence, 
                reliability=0.8)
            face = Evidence(
                emotion=facial_emotion, 
                confidence=visual_result.emotion_prediction.confidence, 
                reliability=0.9)
            semantics = Evidence(
                emotion=semantic_emotion, 
                confidence=0.8, 
                reliability=0.9)
            
            multimodal_model = MultiModalClassifier()
            combined_prediction = multimodal_model.predict(tone, face, semantics)
            print(f"\n===== MULTIMODAL LATE FUSION MODEL [{processing_id}] =====")
            print("FUSED PREDICTIONS:")
            multimodal_model.print_mass_function(combined_prediction, "tone, facial expression, semantics")
            print("========================================\n")
            # ======= END OF INSERTION

            
            # Create multimodal input for LLM
            multimodal_input = MultiModalEmotionInput(
                user_speech=transcription,
                semantic_emotion=semantic_emotion,
                tonal_emotion=tonal_emotion,
                facial_emotion=facial_emotion,
                fused_emotion=combined_prediction,
                session_id=session_id
            )

            
            # Generate response using LLM
            response_text, response_id = await llm_service.process_multimodal_input(multimodal_input)

            
            # Synthesize speech using ElevenLabs
            audio_url, full_path = await elevenlabs_service.synthesize_speech(
                text=response_text, 
                response_id=response_id
            )
            
            # Create and return the response
            return LLMToTTSResponse(
                response_id=response_id,
                llm_text=response_text,
                audio_url=audio_url,
                session_id=session_id,
                emotion=semantic_emotion,  # Use semantic emotion as primary
                metadata={
                    "processing_id": processing_id,
                    "semantic_emotion": semantic_emotion,
                    "tonal_emotion": tonal_emotion,
                    "facial_emotion": facial_emotion,
                    "audio_duration": audio_duration
                }
            )
            
        except Exception as e:
            print(f"❌ Error in multimodal processing: {str(e)}")
            
            # Create a simple error response
            return LLMToTTSResponse(
                response_id=f"error_{uuid.uuid4().hex[:6]}",
                llm_text="I'm sorry, but I couldn't process your input correctly. Could you try again?",
                audio_url=None,
                session_id=session_id,
                emotion="neutral",
                metadata={"error": str(e)}
            )

# Create a singleton instance
multimodal_service = MultiModalService() 