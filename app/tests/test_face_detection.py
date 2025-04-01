import os
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

from app.services.deepface_service import DeepFaceService

# Fix the path to the test images
TEST_FACE_IMAGE_PATH = os.path.join("test_files", "test_face.jpg")
TEST_NO_FACE_IMAGE_PATH = os.path.join("test_files", "test_no_face.jpg")

class TestFaceDetection:
    """Tests for the face detection functionality."""
    
    @pytest.fixture
    def setup_service(self):
        """Set up the DeepFaceService for testing."""
        with patch("cv2.CascadeClassifier") as mock_cascade:
            with patch("cv2.dnn") as mock_dnn:
                with patch("cv2.imread") as mock_imread:
                    # Create a mock for cv2
                    mock_cv2 = MagicMock()
                    mock_cv2.CascadeClassifier = mock_cascade
                    mock_cv2.dnn = mock_dnn
                    mock_cv2.imread = mock_imread
                    mock_cv2.data.haarcascades = "mock/path/"
                    
                    # Initialize service
                    service = DeepFaceService()
                    
                    # Override the cv2 attribute in the service
                    service.cv2 = mock_cv2
                    service.face_cascade = mock_cascade.return_value
                    
                    # Return service and mocked cv2
                    yield service, mock_cv2
    
    def test_detect_face_with_path(self, setup_service):
        """Test face detection with image path input."""
        service, mock_cv2 = setup_service
        
        # Mock cv2.imread to return a fake image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        
        # Mock face detection with cascade classifier
        mock_faces = np.array([[10, 10, 50, 50]])  # x, y, w, h format
        service.face_cascade.detectMultiScale.return_value = mock_faces
        
        # Call detect_face with path
        result = service.detect_face(TEST_FACE_IMAGE_PATH)
        
        # Check result
        assert result["face_detected"] is True
        assert "face_quality" in result
        assert "face_box" in result
        assert len(result["face_box"]) == 4  # x, y, w, h
        
        # Verify method calls
        mock_cv2.imread.assert_called_once_with(TEST_FACE_IMAGE_PATH)
        service.face_cascade.detectMultiScale.assert_called_once()
    
    def test_detect_face_with_array(self, setup_service):
        """Test face detection with numpy array input."""
        service, mock_cv2 = setup_service
        
        # Create a fake image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock face detection with cascade classifier
        mock_faces = np.array([[10, 10, 50, 50]])  # x, y, w, h format
        service.face_cascade.detectMultiScale.return_value = mock_faces
        
        # Call detect_face with array
        result = service.detect_face(image)
        
        # Check result
        assert result["face_detected"] is True
        assert "face_quality" in result
        assert "face_box" in result
        assert len(result["face_box"]) == 4  # x, y, w, h
        
        # Verify method calls
        mock_cv2.imread.assert_not_called()
        service.face_cascade.detectMultiScale.assert_called_once()
    
    def test_detect_face_no_face(self, setup_service):
        """Test face detection with image containing no face."""
        service, mock_cv2 = setup_service
        
        # Mock cv2.imread to return a fake image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        
        # Mock face detection with cascade classifier (empty result)
        service.face_cascade.detectMultiScale.return_value = np.array([])
        
        # Call detect_face
        result = service.detect_face(TEST_NO_FACE_IMAGE_PATH)
        
        # Check result
        assert result["face_detected"] is False
        assert "face_quality" in result
        assert result["face_quality"] == 0.0
        assert "face_box" not in result
        
        # Verify method calls
        mock_cv2.imread.assert_called_once_with(TEST_NO_FACE_IMAGE_PATH)
        service.face_cascade.detectMultiScale.assert_called_once()
    
    def test_detect_face_multiple_faces(self, setup_service):
        """Test face detection with multiple faces in image."""
        service, mock_cv2 = setup_service
        
        # Mock cv2.imread to return a fake image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        
        # Mock face detection with cascade classifier (multiple faces)
        mock_faces = np.array([
            [10, 10, 50, 50],  # face 1: x, y, w, h
            [60, 10, 30, 30]   # face 2: x, y, w, h
        ])
        service.face_cascade.detectMultiScale.return_value = mock_faces
        
        # Call detect_face
        result = service.detect_face(TEST_FACE_IMAGE_PATH)
        
        # Check result
        assert result["face_detected"] is True
        assert "face_quality" in result
        assert "face_box" in result
        assert len(result["face_box"]) == 4  # x, y, w, h
        assert "all_faces" in result
        assert len(result["all_faces"]) == 2
        
        # Verify largest face is selected (by area)
        assert result["face_box"][2] * result["face_box"][3] >= result["all_faces"][1][2] * result["all_faces"][1][3]
        
        # Verify method calls
        mock_cv2.imread.assert_called_once_with(TEST_FACE_IMAGE_PATH)
        service.face_cascade.detectMultiScale.assert_called_once()
    
    def test_detect_face_with_dnn(self, setup_service):
        """Test face detection with DNN model if available."""
        service, mock_cv2 = setup_service
        
        # Set up DNN detector
        service.has_dnn_detector = True
        service.face_net = MagicMock()
        
        # Mock DNN processing
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        mock_cv2.dnn.blobFromImage.return_value = np.zeros((1, 3, 300, 300))
        
        # Mock detection results
        mock_detections = np.zeros((1, 1, 1, 7))  # Format for DNN detections
        mock_detections[0, 0, 0, 2] = 0.9  # confidence score
        mock_detections[0, 0, 0, 3:7] = [0.1, 0.1, 0.6, 0.6]  # normalized coordinates
        service.face_net.forward.return_value = mock_detections
        
        # Call detect_face
        result = service.detect_face(TEST_FACE_IMAGE_PATH)
        
        # Check result
        assert result["face_detected"] is True
        assert result["face_quality"] > 0.8  # high confidence
        assert "face_box" in result
        assert len(result["face_box"]) == 4  # x, y, w, h
        
        # Verify method calls
        mock_cv2.imread.assert_called_once_with(TEST_FACE_IMAGE_PATH)
        mock_cv2.dnn.blobFromImage.assert_called_once()
        service.face_net.setInput.assert_called_once()
        service.face_net.forward.assert_called_once()
    
    def test_detect_face_invalid_input(self, setup_service):
        """Test face detection with invalid input."""
        service, mock_cv2 = setup_service
        
        # Mock cv2.imread to return None for invalid path
        mock_cv2.imread.return_value = None
        
        # Call detect_face with invalid path
        result = service.detect_face("invalid_path.jpg")
        
        # Check result
        assert result["face_detected"] is False
        assert "face_quality" in result
        assert result["face_quality"] == 0.0
        assert "error" in result
        assert "Invalid image" in result["error"]
        
        # Verify method calls
        mock_cv2.imread.assert_called_once_with("invalid_path.jpg")
    
    def test_detect_face_with_deepface(self, setup_service):
        """Test face detection with DeepFace if available."""
        service, mock_cv2 = setup_service
        
        # Set up DeepFace
        service.has_deepface = True
        service.deepface = MagicMock()
        
        # Mock deepface.extract_faces
        mock_faces_result = [{"confidence": 0.95, "facial_area": {"x": 10, "y": 10, "w": 50, "h": 50}}]
        service.deepface.extract_faces = MagicMock(return_value=mock_faces_result)
        
        # Mock cv2.imread to return a fake image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image
        
        # Call detect_face
        result = service.detect_face(TEST_FACE_IMAGE_PATH)
        
        # Check result
        assert result["face_detected"] is True
        assert result["face_quality"] > 0.9
        assert "face_box" in result
        assert len(result["face_box"]) == 4  # x, y, w, h
        
        # Verify method calls
        mock_cv2.imread.assert_called_once_with(TEST_FACE_IMAGE_PATH)
        service.deepface.extract_faces.assert_called_once() 