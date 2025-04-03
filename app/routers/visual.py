import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional

from app.services.visual_service import visual_service
from app.models.visual import VisualProcessingResult

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/visual",
    tags=["visual"],
    responses={404: {"description": "Not found"}},
)

# Supported image types and max file size
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@router.post("/process", response_model=VisualProcessingResult)
async def process_image(
    image: UploadFile = File(...),
):
    """
    Process an uploaded image to extract facial features and emotions.
    
    Parameters:
    - image: Image file to analyze
    
    Returns:
    - VisualProcessingResult: Analysis results including facial features and emotions
    """
    logger.info(f"Received image processing request: {image.filename}")
    
    # Validate file type
    if image.content_type not in SUPPORTED_IMAGE_TYPES:
        logger.warning(f"Unsupported file type: {image.content_type}")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {image.content_type}. Supported types are: {', '.join(SUPPORTED_IMAGE_TYPES)}"
        )
    
    # Get file extension
    file_extension = image.filename.split(".")[-1].lower() if "." in image.filename else "jpg"
    
    try:
        # Process the image
        result = await visual_service.process_image(image.file, file_extension)
        
        # Log the result
        logger.info(f"Image processed successfully: {result.id}, face detected: {result.face_detected}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.get("/status")
async def status() -> Dict[str, Any]:
    """
    Check the status of the visual processing service and its dependencies.
    
    Returns:
    - Dictionary with status information
    """
    logger.info("Checking visual service status")
    
    # Create status report
    status_report = {
        "status": "operational",
        "services": {
            "visual_service": "operational",
            "deepface": "operational" if hasattr(visual_service.deepface_service, "has_deepface") and 
                                        visual_service.deepface_service.has_deepface else "limited",
            "facial_landmarks": "operational"
        },
        "cache": {
            "size": len(visual_service.cache),
            "max_size": visual_service.max_cache_size,
            "expiration_seconds": visual_service.cache_expiration_seconds
        }
    }
    
    return status_report 