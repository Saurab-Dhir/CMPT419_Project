import torch
import numpy as np
import tempfile
from typing import Union
from app.models.audio import AudioEmotionPrediction
from tone_classification.data_loader import Wav2Vec2FeatureExtractor
from tone_classification.tone_classification_model import ToneClassifierModel


class ToneService:
    def __init__(self, model_path="tone_classification/saved_models/tone_classifier.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def predict_emotion(self, audio_bytes: bytes) -> AudioEmotionPrediction:
        # Write audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Extract embedding
        emb = self.extractor.extract_embedding(tmp_path)
        emb_tensor = torch.from_numpy(emb).float().unsqueeze(0).to(self.device)


        # Get model prediction
        with torch.no_grad():
            logits = self.model(emb_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        emotion = self.label_classes[top_idx]
        confidence = float(probs[top_idx])
        secondary = {self.label_classes[i]: float(p) for i, p in enumerate(probs) if i != top_idx}

        return AudioEmotionPrediction(
            emotion=emotion,
            confidence=confidence,
            secondary_emotions=secondary
        )

tone_service = ToneService()