import io
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.services.visual_service import VisualService
from app.models.visual import (
    FacialLandmarks,
    FacialFeatures,
    FacialEmotionPrediction,
    VisualProcessingResult
)

@pytest.fixture
def visual_service():
    """Create a VisualService with mocked dependencies for testing."""
    with patch('app.services.visual_service.DeepFaceService') as mock_deepface_service_class:
        # Create a mock instance of DeepFaceService
        mock_deepface_service = MagicMock()
        mock_deepface_service_class.return_value = mock_deepface_service
        
        # Configure the mock to return successful face detection
        mock_deepface_service.detect_face.return_value = {
            "detected": True,
            "quality": 0.9,
            "box": [100, 100, 200, 200]
        }
        
        # Configure the mock for face analysis
        mock_deepface_service.analyze_face.return_value = {
            "emotion": {
                "dominant_emotion": "happy",
                "emotion": {
                    "happy": 0.85,
                    "sad": 0.05,
                    "angry": 0.03,
                    "neutral": 0.07
                }
            },
            "age": 30,
            "gender": "Male"
        }
        
        # Create mock landmarks
        mock_landmarks = FacialLandmarks(
            eye_positions=[(0.3, 0.4), (0.7, 0.4)],
            mouth_position=[(0.4, 0.7), (0.6, 0.7)],
            eyebrow_positions=[(0.3, 0.35), (0.7, 0.35)],
            nose_position=(0.5, 0.5),
            face_contour=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
        )
        mock_deepface_service.extract_landmarks.return_value = mock_landmarks
        
        # Create mock metrics
        mock_metrics = {
            "eye_openness": 0.8,
            "mouth_openness": 0.2,
            "eyebrow_raise": 0.6,
            "head_pose": {
                "pitch": 5.0,
                "yaw": -2.0,
                "roll": 1.0
            }
        }
        mock_deepface_service.calculate_metrics.return_value = mock_metrics
        
        # Create the service
        service = VisualService()
        
        # Replace the DeepFaceService instance with our mock
        service.deepface_service = mock_deepface_service
        
        yield service

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a simple 100x100 black image with a white square in the middle
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = 255
    
    # Convert to bytes
    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(buffer)
    img_bytes.name = 'test.jpg'
    
    return img_bytes

@pytest.mark.asyncio
async def test_visual_service_initialization():
    """Test that the visual service initializes properly."""
    with patch('app.services.visual_service.DeepFaceService'):
        service = VisualService()
        assert service is not None
        assert service.cache == {}
        assert service.max_cache_size == 100
        assert service.cache_expiration_seconds == 300

@pytest.mark.asyncio
async def test_process_image_with_face(visual_service):
    """Test that process_image returns correct result when face is detected."""
    # Create a mock image file
    mock_image = io.BytesIO(b"mock image data")
    
    # Mock the _load_image and _generate_cache_key methods
    with patch.object(visual_service, '_load_image', return_value=np.zeros((100, 100, 3), dtype=np.uint8)) as mock_load:
        with patch.object(visual_service, '_generate_cache_key', return_value='test_key') as mock_key:
            
            # Process the image
            result = await visual_service.process_image(mock_image, 'jpg')
            
            # Check that the methods were called
            mock_load.assert_called_once()
            mock_key.assert_called_once()
            
            # Check the result
            assert isinstance(result, VisualProcessingResult)
            assert result.face_detected is True
            assert result.face_quality == 0.9
            assert result.emotion_prediction.emotion == "happy"
            assert result.emotion_prediction.confidence == 0.85
            assert "sad" in result.emotion_prediction.secondary_emotions
            assert result.features.eye_openness == 0.8
            assert result.features.mouth_openness == 0.2
            assert result.features.eyebrow_raise == 0.6

@pytest.mark.asyncio
async def test_process_image_no_face(visual_service):
    """Test that process_image returns appropriate result when no face is detected."""
    # Configure the mock to return no face detection
    visual_service.deepface_service.detect_face.return_value = {
        "detected": False,
        "quality": 0.0
    }
    
    # Create a mock image file
    mock_image = io.BytesIO(b"mock image data")
    
    # Mock the _load_image and _generate_cache_key methods
    with patch.object(visual_service, '_load_image', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        with patch.object(visual_service, '_generate_cache_key', return_value='test_key'):
            
            # Process the image
            result = await visual_service.process_image(mock_image, 'jpg')
            
            # Check the result
            assert isinstance(result, VisualProcessingResult)
            assert result.face_detected is False
            assert result.face_quality == 0.0
            assert result.emotion_prediction.emotion == "unknown"
            assert result.emotion_prediction.confidence == 0.0
            assert result.emotion_prediction.secondary_emotions == {}

@pytest.mark.asyncio
async def test_process_image_with_error(visual_service):
    """Test that process_image handles errors gracefully."""
    # Configure the mock to raise an exception
    visual_service.deepface_service.detect_face.side_effect = Exception("Test error")
    
    # Create a mock image file
    mock_image = io.BytesIO(b"mock image data")
    
    # Mock the _load_image and _generate_cache_key methods
    with patch.object(visual_service, '_load_image', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        with patch.object(visual_service, '_generate_cache_key', return_value='test_key'):
            
            # Process the image
            result = await visual_service.process_image(mock_image, 'jpg')
            
            # Check the result
            assert isinstance(result, VisualProcessingResult)
            assert result.face_detected is False
            assert result.face_quality == 0.0
            assert result.emotion_prediction.emotion == "unknown"

@pytest.mark.asyncio
async def test_cache_functionality(visual_service):
    """Test that the caching mechanism works correctly."""
    # Create a mock image file
    mock_image = io.BytesIO(b"mock image data")
    cache_key = "test_cache_key"
    
    # Mock the _load_image and _generate_cache_key methods
    with patch.object(visual_service, '_load_image', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        with patch.object(visual_service, '_generate_cache_key', return_value=cache_key):
            
            # Process the image first time
            result1 = await visual_service.process_image(mock_image, 'jpg')
            
            # Check that detect_face was called
            visual_service.deepface_service.detect_face.assert_called_once()
            
            # Reset the mock
            visual_service.deepface_service.detect_face.reset_mock()
            
            # Process the image second time (should use cache)
            result2 = await visual_service.process_image(mock_image, 'jpg')
            
            # Check that detect_face was not called again
            visual_service.deepface_service.detect_face.assert_not_called()
            
            # Check the results are the same
            assert result1.id == result2.id

@pytest.mark.asyncio
async def test_cache_expiration(visual_service):
    """Test that the cache expiration works correctly."""
    # Create a mock image file
    mock_image = io.BytesIO(b"mock image data")
    cache_key = "test_cache_key"
    
    # Mock the _load_image and _generate_cache_key methods
    with patch.object(visual_service, '_load_image', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        with patch.object(visual_service, '_generate_cache_key', return_value=cache_key):
            
            # Set a very short cache expiration time for the test
            visual_service.cache_expiration_seconds = -1  # Make it expire immediately
            
            # Process the image first time
            result1 = await visual_service.process_image(mock_image, 'jpg')
            
            # Reset the mock
            visual_service.deepface_service.detect_face.reset_mock()
            
            # Process the image second time (should NOT use cache due to expiration)
            result2 = await visual_service.process_image(mock_image, 'jpg')
            
            # Check that detect_face was called again
            visual_service.deepface_service.detect_face.assert_called_once()
            
            # Check the results have different IDs
            assert result1.id != result2.id

def test_load_image(visual_service):
    """Test the _load_image method."""
    # Create a small test image
    img_data = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img_data)
    img_bytes = bytes(buffer)
    
    # Test loading the image
    with patch('cv2.imdecode', return_value=img_data) as mock_decode:
        result = visual_service._load_image(img_bytes)
        mock_decode.assert_called_once()
        assert np.array_equal(result, img_data)

def test_load_image_failure(visual_service):
    """Test _load_image when decoding fails."""
    # Create some invalid image data
    invalid_bytes = b"not an image"
    
    # Test loading with cv2.imdecode returning None
    with patch('cv2.imdecode', return_value=None):
        with pytest.raises(ValueError, match="Failed to decode image data"):
            visual_service._load_image(invalid_bytes)

def test_generate_cache_key(visual_service):
    """Test _generate_cache_key method."""
    # Test with a known input
    test_bytes = b"test data"
    expected_hash = "eb733a00c0c9d336e65691a37ab54293"  # MD5 hash of "test data"
    
    # Generate the cache key
    cache_key = visual_service._generate_cache_key(test_bytes)
    
    # Check the result
    assert cache_key == expected_hash 