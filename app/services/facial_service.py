from deepface import DeepFace

class FacialService:
    """Service for facial emotion detection using DeepFace."""

    async def detect_emotion(self, image_path: str) -> str:
        try:
            result = DeepFace.analyze(img_path=image_path, actions=['emotion'])
            return result[0]['dominant_emotion']
        except Exception as e:
            print(f"❌ Emotion detection error: {e}")
            return "unknown"


facial_service = FacialService()
