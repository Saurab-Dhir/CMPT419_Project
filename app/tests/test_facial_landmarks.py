import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.services.deepface_service import DeepFaceService
from app.models.visual import FacialLandmarks

# Fix the path to the test image
TEST_FACE_IMAGE_PATH = os.path.join("test_files", "test_face.jpg")

class TestFacialLandmarks:
    """Tests for the facial landmarks extraction functionality."""
    
    @pytest.fixture
    def setup_service(self):
        """Set up the DeepFaceService for testing."""
        with patch("app.services.deepface_service.cv2") as mock_cv2:
            # Mock cascade classifier
            mock_cv2.CascadeClassifier.return_value = MagicMock()
            mock_cv2.data.haarcascades = "mock/path/"
            
            # Initialize service
            service = DeepFaceService()
            
            # Mock face detection result for testing
            service.detect_face = MagicMock(return_value={
                "face_detected": True,
                "face_quality": 0.9,
                "face_box": [10, 10, 50, 50]  # x, y, w, h
            })
            
            # Return service and mocked cv2
            yield service, mock_cv2
    
    def test_extract_landmarks_with_dlib(self, setup_service):
        """Test landmark extraction using dlib if available."""
        service, mock_cv2 = setup_service
        
        # Set up dlib mock
        with patch.dict("sys.modules", {"dlib": MagicMock()}):
            # Create mock dlib objects
            mock_dlib = MagicMock()
            mock_detector = MagicMock()
            mock_predictor = MagicMock()
            
            # Mock face detection
            mock_rect = MagicMock()
            mock_detector.return_value = [mock_rect]
            
            # Mock landmark prediction
            mock_shape = MagicMock()
            mock_predictor.return_value = mock_shape
            
            # Mock the shape points
            mock_points = []
            # Create 68 mock points with x,y coordinates
            for i in range(68):
                point = MagicMock()
                point.x = 10 + i % 10
                point.y = 20 + i % 10
                mock_points.append(point)
            
            # Set up shape iteration
            mock_shape.__iter__.return_value = mock_points
            mock_shape.__len__.return_value = 68
            
            # Assign mocks to service
            service.dlib = mock_dlib
            service.dlib.get_frontal_face_detector.return_value = mock_detector
            service.dlib.shape_predictor.return_value = mock_predictor
            service.has_dlib = True
            
            # Mock cv2.imread to return a fake image
            mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_cv2.imread.return_value = mock_image
            
            # Call extract_landmarks
            result = service.extract_landmarks(
                TEST_FACE_IMAGE_PATH, 
                {"face_detected": True, "face_box": [10, 10, 50, 50]}
            )
            
            # Check result
            assert isinstance(result, FacialLandmarks)
            assert len(result.eye_positions) > 0
            assert len(result.mouth_position) > 0
            assert len(result.eyebrow_positions) > 0
            assert result.nose_position is not None
            assert len(result.face_contour) > 0
            
            # Verify method calls
            mock_cv2.imread.assert_called_once_with(TEST_FACE_IMAGE_PATH)
            mock_detector.assert_called()
            mock_predictor.assert_called()
    
    def test_extract_landmarks_with_opencv(self, setup_service):
        """Test landmark extraction using OpenCV fallback."""
        service, mock_cv2 = setup_service
        
        # Set up service without dlib
        service.has_dlib = False
        
        # Mock cv2.imread to return a fake image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        
        # Mock OpenCV face detector result
        mock_faces = np.array([[10, 10, 50, 50]])
        service.face_cascade.detectMultiScale.return_value = mock_faces
        
        # Mock OpenCV facial landmark detection
        face_landmarks_detector = MagicMock()
        service.face_landmarks_detector = face_landmarks_detector
        
        # Mock landmark detection result
        mock_landmarks = np.array([
            # Eye landmarks (right, left)
            [15, 15], [25, 15], [35, 15], [45, 15],
            # Eyebrow landmarks
            [15, 10], [25, 10], [35, 10], [45, 10],
            # Nose landmark
            [30, 25],
            # Mouth landmarks
            [20, 35], [30, 40], [40, 35],
            # Face contour landmarks
            [10, 10], [20, 5], [30, 5], [40, 5], [50, 10],
            [50, 20], [50, 30], [40, 40], [30, 45], [20, 40], [10, 30], [10, 20]
        ])
        face_landmarks_detector.detect.return_value = mock_landmarks
        
        # Call extract_landmarks
        result = service.extract_landmarks(
            TEST_FACE_IMAGE_PATH, 
            {"face_detected": True, "face_box": [10, 10, 50, 50]}
        )
        
        # Check result
        assert isinstance(result, FacialLandmarks)
        assert len(result.eye_positions) > 0
        assert len(result.mouth_position) > 0
        assert len(result.eyebrow_positions) > 0
        assert result.nose_position is not None
        assert len(result.face_contour) > 0
        
        # Verify method calls
        mock_cv2.imread.assert_called_once_with(TEST_FACE_IMAGE_PATH)
        face_landmarks_detector.detect.assert_called()
    
    def test_extract_landmarks_no_face(self, setup_service):
        """Test landmark extraction when no face is detected."""
        service, mock_cv2 = setup_service
        
        # Call extract_landmarks with no face detection result
        result = service.extract_landmarks(
            TEST_FACE_IMAGE_PATH, 
            {"face_detected": False}
        )
        
        # Check result has default values
        assert isinstance(result, FacialLandmarks)
        assert len(result.eye_positions) == 2  # Should have default eye positions
        assert len(result.mouth_position) == 2  # Should have default mouth positions
        assert len(result.eyebrow_positions) == 2  # Should have default eyebrow positions
        assert result.nose_position == (0.5, 0.5)  # Should have default nose position
        assert len(result.face_contour) == 4  # Should have default face contour
    
    def test_extract_landmarks_error_handling(self, setup_service):
        """Test landmark extraction error handling."""
        service, mock_cv2 = setup_service
        
        # Set up service with error behavior
        service.has_dlib = True
        service.dlib = MagicMock()
        service.dlib.get_frontal_face_detector.side_effect = Exception("Test error")
        
        # Mock cv2.imread to return a fake image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        
        # Call extract_landmarks
        result = service.extract_landmarks(
            TEST_FACE_IMAGE_PATH, 
            {"face_detected": True, "face_box": [10, 10, 50, 50]}
        )
        
        # Check default values are used
        assert isinstance(result, FacialLandmarks)
        assert len(result.eye_positions) == 2
        assert len(result.mouth_position) == 2
        assert len(result.eyebrow_positions) == 2
        assert result.nose_position == (0.5, 0.5)
        assert len(result.face_contour) == 4
    
    def test_landmark_normalization(self, setup_service):
        """Test that landmark coordinates are normalized correctly."""
        service, mock_cv2 = setup_service
        
        # Set up service without dlib
        service.has_dlib = False
        
        # Mock image dimensions
        mock_image = np.zeros((200, 100, 3), dtype=np.uint8)  # Height=200, Width=100
        mock_cv2.imread.return_value = mock_image
        
        # Mock OpenCV face detector result
        mock_faces = np.array([[10, 10, 50, 50]])
        service.face_cascade.detectMultiScale.return_value = mock_faces
        
        # Mock OpenCV facial landmark detection
        face_landmarks_detector = MagicMock()
        service.face_landmarks_detector = face_landmarks_detector
        
        # Mock landmark detection result with integer coordinates
        mock_landmarks = np.array([
            # Eye landmarks (right, left) - raw pixel coords
            [25, 30], [75, 30],
            # Eyebrow landmarks - raw pixel coords
            [25, 20], [75, 20],
            # Nose landmark - raw pixel coords
            [50, 50],
            # Mouth landmarks - raw pixel coords
            [35, 70], [65, 70],
            # Face contour landmarks - raw pixel coords
            [10, 10], [90, 10], [90, 90], [10, 90]
        ])
        face_landmarks_detector.detect.return_value = mock_landmarks
        
        # Call extract_landmarks
        result = service.extract_landmarks(
            TEST_FACE_IMAGE_PATH, 
            {"face_detected": True, "face_box": [10, 10, 50, 50]}
        )
        
        # Check normalization (all values should be between 0 and 1)
        assert isinstance(result, FacialLandmarks)
        
        # Check eye positions are normalized
        for pos in result.eye_positions:
            assert 0 <= pos[0] <= 1, f"X coordinate {pos[0]} not normalized"
            assert 0 <= pos[1] <= 1, f"Y coordinate {pos[1]} not normalized"
            
        # Check normalization of some specific points
        # The first eye should be at 25% width, 15% height in normalized coordinates
        assert abs(result.eye_positions[0][0] - 0.25) < 0.1
        assert abs(result.eye_positions[0][1] - 0.15) < 0.1
        
        # The nose should be at 50% width, 25% height in normalized coordinates
        assert abs(result.nose_position[0] - 0.5) < 0.1
        assert abs(result.nose_position[1] - 0.25) < 0.1 