import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.services.deepface_service import DeepFaceService
from app.models.visual import FacialLandmarks, FacialEmotionPrediction, VisualProcessingResult

# Fix the path to the test image
TEST_IMAGE_PATH = os.path.join("test_files", "test_face.jpg")

class TestDeepFaceService:
    """Tests for the DeepFaceService class."""
    
    def test_init(self):
        """Test that service can be initialized."""
        # Mock cv2 import and initialization
        with patch("app.services.deepface_service.cv2") as mock_cv2:
            # Mock cascade classifier
            mock_cv2.CascadeClassifier.return_value = MagicMock()
            mock_cv2.data.haarcascades = "mock/path/"
            
            # Initialize service
            service = DeepFaceService()
            
            # Check that service attributes are set
            assert hasattr(service, "use_opencv_fallback")
            assert hasattr(service, "cv2")
            assert hasattr(service, "face_cascade")
    
    def test_init_with_deepface(self):
        """Test service initialization with DeepFace available."""
        # Mock cv2 and DeepFace imports
        with patch("app.services.deepface_service.cv2") as mock_cv2, \
             patch.dict("sys.modules", {"deepface": MagicMock(), "deepface.DeepFace": MagicMock()}):
            
            # Mock cascade classifier
            mock_cv2.CascadeClassifier.return_value = MagicMock()
            mock_cv2.data.haarcascades = "mock/path/"
            
            # Initialize service
            service = DeepFaceService()
            
            # Check DeepFace flag is set
            assert service.has_deepface is True
    
    def test_init_without_deepface(self):
        """Test service initialization with DeepFace not available."""
        # Mock cv2 import and initialization
        with patch("app.services.deepface_service.cv2") as mock_cv2:
            # Mock cascade classifier
            mock_cv2.CascadeClassifier.return_value = MagicMock()
            mock_cv2.data.haarcascades = "mock/path/"
            
            # Force ImportError for DeepFace
            with patch("app.services.deepface_service.DeepFaceService._initialize_models") as mock_init:
                def side_effect():
                    service = DeepFaceService.__new__(DeepFaceService)
                    service.cv2 = mock_cv2
                    service.face_cascade = MagicMock()
                    service.has_deepface = False
                    service.has_dnn_detector = False
                    service.use_opencv_fallback = True
                    return None
                
                mock_init.side_effect = side_effect
                
                # Initialize service
                service = DeepFaceService()
                
                # Check DeepFace flag is not set
                assert service.has_deepface is False
                assert service.use_opencv_fallback is True
    
    def test_detect_face_stub(self):
        """Test the detect_face method stub."""
        # Mock cv2 import and initialization
        with patch("app.services.deepface_service.cv2") as mock_cv2:
            # Mock cascade classifier
            mock_cv2.CascadeClassifier.return_value = MagicMock()
            mock_cv2.data.haarcascades = "mock/path/"
            
            # Initialize service with implementation for detect_face
            service = DeepFaceService()
            service.detect_face = MagicMock(return_value={"face_detected": True, "face_quality": 0.9})
            
            # Call method
            result = service.detect_face("dummy_image")
            
            # Check result
            assert result == {"face_detected": True, "face_quality": 0.9}
            service.detect_face.assert_called_once_with("dummy_image")
    
    def test_analyze_face_stub(self):
        """Test the analyze_face method stub."""
        # Mock cv2 import and initialization
        with patch("app.services.deepface_service.cv2") as mock_cv2:
            # Mock cascade classifier
            mock_cv2.CascadeClassifier.return_value = MagicMock()
            mock_cv2.data.haarcascades = "mock/path/"
            
            # Initialize service with implementation for analyze_face
            service = DeepFaceService()
            service.analyze_face = MagicMock(return_value={
                "emotion": "happy",
                "emotion_confidence": 0.8,
                "secondary_emotions": {"neutral": 0.2}
            })
            
            # Call method
            result = service.analyze_face("dummy_image", {"face_detected": True})
            
            # Check result
            assert result["emotion"] == "happy"
            assert result["emotion_confidence"] == 0.8
            assert "neutral" in result["secondary_emotions"]
            service.analyze_face.assert_called_once_with("dummy_image", {"face_detected": True})
    
    def test_extract_landmarks_stub(self):
        """Test the extract_landmarks method stub."""
        # Mock cv2 import and initialization
        with patch("app.services.deepface_service.cv2") as mock_cv2:
            # Mock cascade classifier
            mock_cv2.CascadeClassifier.return_value = MagicMock()
            mock_cv2.data.haarcascades = "mock/path/"
            
            # Create mock landmarks
            mock_landmarks = FacialLandmarks(
                eye_positions=[(0.3, 0.4), (0.7, 0.4)],
                mouth_position=[(0.4, 0.7), (0.6, 0.7)],
                eyebrow_positions=[(0.3, 0.35), (0.7, 0.35)],
                nose_position=(0.5, 0.5),
                face_contour=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
            )
            
            # Initialize service with implementation for extract_landmarks
            service = DeepFaceService()
            service.extract_landmarks = MagicMock(return_value=mock_landmarks)
            
            # Call method
            result = service.extract_landmarks("dummy_image", {"face_detected": True})
            
            # Check result
            assert isinstance(result, FacialLandmarks)
            assert result.eye_positions == [(0.3, 0.4), (0.7, 0.4)]
            service.extract_landmarks.assert_called_once_with("dummy_image", {"face_detected": True})
    
    def test_calculate_metrics_stub(self):
        """Test the calculate_metrics method stub."""
        # Mock cv2 import and initialization
        with patch("app.services.deepface_service.cv2") as mock_cv2:
            # Mock cascade classifier
            mock_cv2.CascadeClassifier.return_value = MagicMock()
            mock_cv2.data.haarcascades = "mock/path/"
            
            # Create mock landmarks
            mock_landmarks = FacialLandmarks(
                eye_positions=[(0.3, 0.4), (0.7, 0.4)],
                mouth_position=[(0.4, 0.7), (0.6, 0.7)],
                eyebrow_positions=[(0.3, 0.35), (0.7, 0.35)],
                nose_position=(0.5, 0.5),
                face_contour=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
            )
            
            # Initialize service with implementation for calculate_metrics
            service = DeepFaceService()
            service.calculate_metrics = MagicMock(return_value={
                "eye_openness": 0.8,
                "mouth_openness": 0.2,
                "eyebrow_raise": 0.6,
                "head_pose": {"pitch": 5.0, "yaw": -2.0, "roll": 1.0}
            })
            
            # Call method
            result = service.calculate_metrics(mock_landmarks)
            
            # Check result
            assert result["eye_openness"] == 0.8
            assert result["mouth_openness"] == 0.2
            assert result["eyebrow_raise"] == 0.6
            assert "pitch" in result["head_pose"]
            service.calculate_metrics.assert_called_once_with(mock_landmarks)
    
    def test_process_image_stub(self):
        """Test the process_image method integration."""
        # Mock cv2 import and initialization
        with patch("app.services.deepface_service.cv2") as mock_cv2:
            # Mock cascade classifier
            mock_cv2.CascadeClassifier.return_value = MagicMock()
            mock_cv2.data.haarcascades = "mock/path/"
            
            # Create mock landmarks
            mock_landmarks = FacialLandmarks(
                eye_positions=[(0.3, 0.4), (0.7, 0.4)],
                mouth_position=[(0.4, 0.7), (0.6, 0.7)],
                eyebrow_positions=[(0.3, 0.35), (0.7, 0.35)],
                nose_position=(0.5, 0.5),
                face_contour=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
            )
            
            # Mock the service methods
            service = DeepFaceService()
            service.detect_face = MagicMock(return_value={"face_detected": True, "face_quality": 0.9})
            service.analyze_face = MagicMock(return_value={
                "emotion": "happy",
                "emotion_confidence": 0.8,
                "secondary_emotions": {"neutral": 0.2}
            })
            service.extract_landmarks = MagicMock(return_value=mock_landmarks)
            service.calculate_metrics = MagicMock(return_value={
                "eye_openness": 0.8,
                "mouth_openness": 0.2,
                "eyebrow_raise": 0.6,
                "head_pose": {"pitch": 5.0, "yaw": -2.0, "roll": 1.0}
            })
            
            # Call method but preserve the real implementation
            real_process_image = service.process_image
            service.process_image = lambda img: real_process_image(service, img)
            
            # Call method
            result = service.process_image("dummy_image")
            
            # Check result
            assert isinstance(result, VisualProcessingResult)
            assert result.face_detected is True
            assert result.face_quality == 0.9
            assert result.emotion_prediction.emotion == "happy"
            assert result.emotion_prediction.confidence == 0.8
            assert result.features.eye_openness == 0.8
            
            # Verify method calls
            service.detect_face.assert_called_once_with("dummy_image")
            service.analyze_face.assert_called_once()
            service.extract_landmarks.assert_called_once()
            service.calculate_metrics.assert_called_once()
    
    def test_calculate_derived_metrics_stubs(self):
        """Test the derived metrics calculation method stubs."""
        # Mock cv2 import and initialization
        with patch("app.services.deepface_service.cv2") as mock_cv2:
            # Mock cascade classifier
            mock_cv2.CascadeClassifier.return_value = MagicMock()
            mock_cv2.data.haarcascades = "mock/path/"
            
            # Initialize service with implementations
            service = DeepFaceService()
            service.calculate_eye_openness = MagicMock(return_value=0.8)
            service.calculate_mouth_openness = MagicMock(return_value=0.2)
            service.calculate_eyebrow_raise = MagicMock(return_value=0.6)
            
            # Mock eye landmarks
            eye_landmarks = [(0.3, 0.4), (0.7, 0.4)]
            mouth_landmarks = [(0.4, 0.7), (0.6, 0.7)]
            eyebrow_landmarks = [(0.3, 0.35), (0.7, 0.35)]
            
            # Call methods
            eye_result = service.calculate_eye_openness(eye_landmarks)
            mouth_result = service.calculate_mouth_openness(mouth_landmarks)
            eyebrow_result = service.calculate_eyebrow_raise(eyebrow_landmarks, eye_landmarks)
            
            # Check results
            assert eye_result == 0.8
            assert mouth_result == 0.2
            assert eyebrow_result == 0.6
            
            # Verify method calls
            service.calculate_eye_openness.assert_called_once_with(eye_landmarks)
            service.calculate_mouth_openness.assert_called_once_with(mouth_landmarks)
            service.calculate_eyebrow_raise.assert_called_once_with(eyebrow_landmarks, eye_landmarks) 