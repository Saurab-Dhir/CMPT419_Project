import torch
import numpy as np
import tempfile
import os
import traceback
from typing import Union, Dict, Any
from app.models.audio import AudioEmotionPrediction
import subprocess
import io
import wave
from pathlib import Path

class ToneService:
    def __init__(self, model_path="tone_classification/saved_models/tone_classifier.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.extractor = None
        self.label_classes = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
        
        # Create directory for temporary audio conversion
        self.temp_dir = Path("audio_temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        try:
            if os.path.exists(model_path):
                from tone_classification.data_loader import Wav2Vec2FeatureExtractor
                from tone_classification.tone_classification_model import ToneClassifierModel
                
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

                self.label_classes = checkpoint['label_encoder']
                output_dim = len(self.label_classes)

                self.model = ToneClassifierModel(input_dim=768, output_dim=len(self.label_classes)).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()

                # Wav2Vec2 embedding extractor
                self.extractor = Wav2Vec2FeatureExtractor(
                    model_name="facebook/wav2vec2-base",
                    device=self.device
                )
                print("✅ Tone classification model loaded successfully")
            else:
                print(f"⚠️ Tone classification model not found at {model_path}, using neutral response")
        except Exception as e:
            print(f"❌ Error loading tone classification model: {str(e)}")
            print("⚠️ Using neutral response classification")

    def predict_emotion(self, audio_data: bytes) -> AudioEmotionPrediction:
        """
        Predict emotion from audio data.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            AudioEmotionPrediction with emotion and confidence
        """
        # Check if model is available
        if self.model is None or self.extractor is None:
            print("⚠️ Tone model not available, using fallback")
            return self._predict_emotion_fallback(audio_data)
        
        try:
            # For debugging, save a copy of the original audio
            timestamp = os.path.join(self.temp_dir, f"original_{int(torch.rand(1)[0]*10000)}.bin")
            with open(timestamp, "wb") as f:
                f.write(audio_data)
            print(f"📢 Saved original audio for debugging to {timestamp}")
            
            # If audio might be WebM, try to convert it to WAV format
            converted_data = audio_data
            if b'webm' in audio_data[:100]:
                print("🔄 WebM format detected, attempting conversion to WAV")
                converted_data = self._convert_webm_to_wav(audio_data)
                if converted_data is None:
                    print("❌ WebM conversion failed, using fallback")
                    return self._predict_emotion_fallback(audio_data)
                else:
                    print(f"✅ Successfully converted WebM to WAV: {len(converted_data)/1024:.2f}KB")
                    # Save converted audio for debugging
                    conv_path = os.path.join(self.temp_dir, f"converted_{int(torch.rand(1)[0]*10000)}.wav")
                    with open(conv_path, "wb") as f:
                        f.write(converted_data)
                    print(f"📢 Saved converted audio for debugging to {conv_path}")
            
            # Create a temporary file for the audio
            fd, tmp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            # Write the audio data to the temporary file
            with open(tmp_path, 'wb') as f:
                f.write(converted_data)
            
            try:
                # Extract embedding
                emb = self.extractor.extract_embedding(tmp_path)
                
                # Since model.predict doesn't exist, let's implement it directly here
                with torch.no_grad():
                    # Ensure model is in eval mode
                    self.model.eval()
                    
                    # Convert embedding to tensor
                    if isinstance(emb, np.ndarray):
                        emb_tensor = torch.FloatTensor(emb).to(self.device)
                    else:
                        emb_tensor = emb.to(self.device)
                    
                    # Make sure it has the right shape (add batch dimension if needed)
                    if len(emb_tensor.shape) == 1:
                        emb_tensor = emb_tensor.unsqueeze(0)
                    
                    # Forward pass through the model
                    outputs = self.model(emb_tensor)
                    
                    # Get prediction
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    predicted_idx = np.argmax(probs)
                    predicted_label = self.label_classes[predicted_idx]
                    confidence = float(probs[predicted_idx])
                    
                    pred = {
                        'label': predicted_idx,
                        'emotion': predicted_label,
                        'prob': confidence
                    }
                
                # Get the emotion with highest probability
                emotion_idx = pred['label'] if isinstance(pred['label'], int) else 0
                emotion = self.label_classes[emotion_idx]
                confidence = float(pred['prob'])
                
                # Create secondary emotions dict
                secondary_emotions = {}
                for i, label in enumerate(self.label_classes):
                    if i != emotion_idx:
                        # If we have probabilities for all emotions, use them
                        if isinstance(probs, np.ndarray) and len(probs) == len(self.label_classes):
                            secondary_emotions[label] = float(probs[i])
                        else:
                            secondary_emotions[label] = 0.1  # Low confidence for all other emotions
                
                print(f"🎭 Detected emotion from tone: {emotion} (confidence: {confidence:.2f})")
                
                # Create emotion prediction object
                return AudioEmotionPrediction(
                    emotion=emotion,
                    confidence=confidence,
                    secondary_emotions=secondary_emotions
                )
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
        except Exception as e:
            print(f"❌ Error in tone classification: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return self._predict_emotion_fallback(audio_data)
    
    def _convert_webm_to_wav(self, webm_data: bytes) -> Union[bytes, None]:
        """
        Attempt to convert WebM audio to WAV format using FFmpeg if available,
        or using a pure Python approach if FFmpeg is not available.
        
        Args:
            webm_data: WebM audio data as bytes
            
        Returns:
            WAV audio data as bytes, or None if conversion fails
        """
        # First try FFmpeg if it's available
        try:
            # Create temporary files
            webm_fd, webm_path = tempfile.mkstemp(suffix='.webm')
            wav_fd, wav_path = tempfile.mkstemp(suffix='.wav')
            os.close(webm_fd)
            os.close(wav_fd)
            
            # Write WebM data to file
            with open(webm_path, 'wb') as f:
                f.write(webm_data)
            
            # Try to convert using FFmpeg
            try:
                result = subprocess.run(
                    ['ffmpeg', '-i', webm_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_path],
                    capture_output=True,
                    check=True,
                    timeout=10
                )
                
                # Read the converted WAV file
                with open(wav_path, 'rb') as f:
                    wav_data = f.read()
                
                print("✅ FFmpeg conversion successful")
                return wav_data
            
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                print(f"⚠️ FFmpeg conversion failed: {str(e)}")
                # Fall back to creating a silent WAV file
                return self._create_silent_wav()
            
            finally:
                # Clean up temporary files
                try:
                    os.unlink(webm_path)
                    os.unlink(wav_path)
                except:
                    pass
        
        except Exception as e:
            print(f"❌ WebM conversion error: {str(e)}")
            return self._create_silent_wav()
    
    def _create_silent_wav(self, duration_seconds=1.0, sample_rate=16000):
        """
        Create a silent WAV file with the specified duration and sample rate.
        Used as a fallback when conversion fails.
        """
        try:
            # Create a BytesIO buffer for the WAV file
            buffer = io.BytesIO()
            
            # Set up the WAV file parameters
            n_channels = 1
            sample_width = 2  # 16-bit
            n_frames = int(duration_seconds * sample_rate)
            
            # Create the WAV file
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(n_channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.setnframes(n_frames)
                
                # Write silence (zeros)
                wav_file.writeframes(b'\x00' * (n_frames * sample_width * n_channels))
            
            # Get the WAV data
            buffer.seek(0)
            wav_data = buffer.read()
            
            print(f"✅ Created silent WAV file ({len(wav_data)/1024:.2f}KB)")
            return wav_data
            
        except Exception as e:
            print(f"❌ Error creating silent WAV: {str(e)}")
            return None
            
    def _predict_emotion_fallback(self, audio_data: bytes) -> AudioEmotionPrediction:
        """Return neutral emotion when the model is not available."""
        print("Using neutral emotion as fallback")
        
        # Generate minimal secondary emotions
        secondary = {}
        for e in self.label_classes:
            if e != "neutral":
                secondary[e] = 0.1  # Low confidence for all emotions
                
        return AudioEmotionPrediction(
            emotion="neutral",
            confidence=0.9,  # High confidence for neutral
            secondary_emotions=secondary
        )

tone_service = ToneService()