import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.services.deepface_service import DeepFaceService
from app.models.visual import FacialLandmarks

class TestMetricsCalculation:
    """Tests for the facial metrics calculation functionality."""
    
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
                    
                    # Return service
                    yield service
    
    @pytest.fixture
    def sample_landmarks(self):
        """Create sample facial landmarks for testing."""
        return FacialLandmarks(
            eye_positions=[
                (0.3, 0.4), (0.32, 0.38), (0.34, 0.37), (0.36, 0.38),  # Left eye
                (0.64, 0.38), (0.66, 0.37), (0.68, 0.38), (0.7, 0.4)   # Right eye
            ],
            mouth_position=[
                (0.4, 0.7), (0.45, 0.72), (0.5, 0.73), (0.55, 0.72), (0.6, 0.7),  # Upper lip
                (0.6, 0.71), (0.55, 0.74), (0.5, 0.75), (0.45, 0.74), (0.4, 0.71)  # Lower lip
            ],
            eyebrow_positions=[
                (0.3, 0.35), (0.34, 0.33), (0.38, 0.32),  # Left eyebrow
                (0.62, 0.32), (0.66, 0.33), (0.7, 0.35)   # Right eyebrow
            ],
            nose_position=(0.5, 0.5),
            face_contour=[
                (0.3, 0.3), (0.4, 0.2), (0.5, 0.18), (0.6, 0.2), (0.7, 0.3),
                (0.75, 0.4), (0.8, 0.5), (0.75, 0.6), (0.7, 0.7),
                (0.6, 0.8), (0.5, 0.82), (0.4, 0.8), (0.3, 0.7),
                (0.25, 0.6), (0.2, 0.5), (0.25, 0.4)
            ]
        )
    
    def test_calculate_eye_openness(self, setup_service, sample_landmarks):
        """Test calculation of eye openness."""
        service = setup_service
        
        # Extract eye landmarks
        left_eye = sample_landmarks.eye_positions[:4]
        right_eye = sample_landmarks.eye_positions[4:]
        
        # Calculate eye openness for each eye
        left_openness = service.calculate_eye_openness(left_eye)
        right_openness = service.calculate_eye_openness(right_eye)
        
        # Check results
        assert 0 <= left_openness <= 1
        assert 0 <= right_openness <= 1
        
        # Test with eyes closed (smaller vertical distance)
        closed_eye = [(0.3, 0.39), (0.32, 0.38), (0.34, 0.38), (0.36, 0.39)]
        closed_openness = service.calculate_eye_openness(closed_eye)
        
        # Check that open eyes have higher openness value than closed eyes
        assert left_openness > closed_openness
    
    def test_calculate_mouth_openness(self, setup_service, sample_landmarks):
        """Test calculation of mouth openness."""
        service = setup_service
        
        # Calculate mouth openness
        mouth_openness = service.calculate_mouth_openness(sample_landmarks.mouth_position)
        
        # Check result
        assert 0 <= mouth_openness <= 1
        
        # Test with mouth closed (smaller vertical distance)
        closed_mouth = [
            (0.4, 0.7), (0.45, 0.7), (0.5, 0.7), (0.55, 0.7), (0.6, 0.7),
            (0.6, 0.71), (0.55, 0.71), (0.5, 0.71), (0.45, 0.71), (0.4, 0.71)
        ]
        closed_openness = service.calculate_mouth_openness(closed_mouth)
        
        # Check that open mouth has higher openness value than closed mouth
        assert mouth_openness > closed_openness
    
    def test_calculate_eyebrow_raise(self, setup_service, sample_landmarks):
        """Test calculation of eyebrow raise."""
        service = setup_service
        
        # Calculate eyebrow raise
        eyebrow_raise = service.calculate_eyebrow_raise(
            sample_landmarks.eyebrow_positions,
            sample_landmarks.eye_positions
        )
        
        # Check result
        assert 0 <= eyebrow_raise <= 1
        
        # Test with lowered eyebrows (closer to eyes)
        lowered_eyebrows = [
            (0.3, 0.38), (0.34, 0.37), (0.38, 0.36),  # Left eyebrow
            (0.62, 0.36), (0.66, 0.37), (0.7, 0.38)   # Right eyebrow
        ]
        lowered_raise = service.calculate_eyebrow_raise(
            lowered_eyebrows,
            sample_landmarks.eye_positions
        )
        
        # Check that raised eyebrows have higher value than lowered
        assert eyebrow_raise > lowered_raise
    
    def test_calculate_metrics(self, setup_service, sample_landmarks):
        """Test the calculate_metrics method."""
        service = setup_service
        
        # Mock individual metrics calculations
        service.calculate_eye_openness = MagicMock(return_value=0.8)
        service.calculate_mouth_openness = MagicMock(return_value=0.2)
        service.calculate_eyebrow_raise = MagicMock(return_value=0.6)
        
        # Calculate metrics
        metrics = service.calculate_metrics(sample_landmarks)
        
        # Check result structure
        assert "eye_openness" in metrics
        assert "mouth_openness" in metrics
        assert "eyebrow_raise" in metrics
        assert "head_pose" in metrics
        
        # Check values
        assert metrics["eye_openness"] == 0.8
        assert metrics["mouth_openness"] == 0.2
        assert metrics["eyebrow_raise"] == 0.6
        assert "pitch" in metrics["head_pose"]
        assert "yaw" in metrics["head_pose"]
        assert "roll" in metrics["head_pose"]
        
        # Verify method calls
        service.calculate_eye_openness.assert_called()
        service.calculate_mouth_openness.assert_called_with(sample_landmarks.mouth_position)
        service.calculate_eyebrow_raise.assert_called_with(
            sample_landmarks.eyebrow_positions,
            sample_landmarks.eye_positions
        )
    
    def test_metrics_with_invalid_landmarks(self, setup_service):
        """Test metrics calculation with invalid landmarks."""
        service = setup_service
        
        # Create invalid landmarks (empty lists)
        invalid_landmarks = FacialLandmarks(
            eye_positions=[],
            mouth_position=[],
            eyebrow_positions=[],
            nose_position=(0.5, 0.5),
            face_contour=[]
        )
        
        # Calculate metrics
        metrics = service.calculate_metrics(invalid_landmarks)
        
        # Check default values are used
        assert metrics["eye_openness"] == 0.5  # Default value
        assert metrics["mouth_openness"] == 0.0  # Default value
        assert metrics["eyebrow_raise"] == 0.0  # Default value
        assert metrics["head_pose"]["pitch"] == 0.0  # Default value 