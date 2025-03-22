from fastapi import APIRouter, UploadFile, File, HTTPException
from tempfile import NamedTemporaryFile
from app.models.visual import FacialEmotionPrediction
from app.services.facial_service import facial_service
import shutil

router = APIRouter()

@router.post("/detect", response_model=FacialEmotionPrediction)
async def detect_facial_emotion(image: UploadFile = File(...)):
    """
    Detect emotion from an uploaded face image.
    """
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(image.file, tmp)
            tmp_path = tmp.name

        emotion = await facial_service.detect_emotion(tmp_path)
        return FacialEmotionPrediction(
            emotion=emotion,
            confidence=0.9,
            secondary_emotions={}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Facial emotion detection failed: {e}")
