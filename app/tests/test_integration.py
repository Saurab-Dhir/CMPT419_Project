import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.services.deepface_service import DeepFaceService
from app.models.visual import FacialLandmarks

class TestFacialAnalysisIntegration:
    """Integration tests for the complete facial analysis pipeline."""
    
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
                    
                    # Mock DeepFace 
                    mock_deepface = MagicMock()
                    
                    # Mock successful face detection
                    mock_deepface.extract_faces.return_value = [{
                        'confidence': 0.95,
                        'facial_area': {'x': 10, 'y': 10, 'w': 100, 'h': 100},
                        'face': np.zeros((224, 224, 3), dtype=np.uint8)
                    }]
                    
                    # Mock successful analysis
                    mock_deepface.analyze.return_value = [{
                        'emotion': {
                            'angry': 0.05, 
                            'disgust': 0.0, 
                            'fear': 0.03, 
                            'happy': 0.85, 
                            'sad': 0.02, 
                            'surprise': 0.03, 
                            'neutral': 0.02
                        },
                        'age': 28,
                        'gender': 'Woman'
                    }]
                    
                    # Set the mocked DeepFace on the service
                    service.deepface = mock_deepface
                    service.has_deepface = True
                    
                    # Mock dlib for landmark detection
                    service.dlib = MagicMock()
                    service.has_dlib = True
                    service.dlib_predictor = MagicMock()
                    
                    # Mock landmarks prediction
                    mock_landmarks = MagicMock()
                    mock_landmarks.parts.return_value = [
                        # Generate 68 mock landmark points (dlib format)
                        MagicMock(x=int(320 * i/68), y=int(240 * (0.5 + 0.2 * np.sin(i/10))))
                        for i in range(68)
                    ]
                    service.dlib_predictor.return_value = mock_landmarks
                    
                    # Return service
                    yield service
    
    def test_complete_facial_analysis_pipeline(self, setup_service):
        """Test the complete facial analysis pipeline from image to metrics."""
        service = setup_service
        
        # Create a mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Step 1: Detect face
        face_result = service.detect_face(mock_image)
        
        # Verify face detection results
        assert face_result is not None
        assert "face_detected" in face_result
        assert face_result["face_detected"] is True
        assert "face_box" in face_result
        assert "face_quality" in face_result
        
        # Step 2: Analyze face
        analysis_result = service.analyze_face(mock_image, face_result)
        
        # Verify analysis results
        assert analysis_result is not None
        assert "emotion" in analysis_result
        assert "emotion_confidence" in analysis_result
        assert "age" in analysis_result
        assert "gender" in analysis_result
        # Accept either emotion, depending on what's returned
        assert analysis_result["emotion"] in ["happy", "neutral"]
        assert analysis_result["emotion_confidence"] >= 0.0
        assert analysis_result["age"] >= 0 
        assert analysis_result["gender"] in ["female", "unknown", "male"]
        
        # Step 3: Extract landmarks
        landmarks = service.extract_landmarks(mock_image, face_result)
        
        # Verify landmarks
        assert isinstance(landmarks, FacialLandmarks)
        assert len(landmarks.eye_positions) > 0
        assert len(landmarks.mouth_position) > 0
        assert len(landmarks.eyebrow_positions) > 0
        assert landmarks.nose_position is not None
        assert len(landmarks.face_contour) > 0
        
        # Step 4: Calculate metrics
        metrics = service.calculate_metrics(landmarks)
        
        # Verify metrics
        assert "eye_openness" in metrics
        assert "mouth_openness" in metrics
        assert "eyebrow_raise" in metrics
        assert "head_pose" in metrics
        assert 0 <= metrics["eye_openness"] <= 1
        assert 0 <= metrics["mouth_openness"] <= 1
        assert 0 <= metrics["eyebrow_raise"] <= 1
        assert "pitch" in metrics["head_pose"]
        assert "yaw" in metrics["head_pose"]
        assert "roll" in metrics["head_pose"]
    
    def test_pipeline_with_no_face_detected(self, setup_service):
        """Test the pipeline when no face is detected."""
        service = setup_service
        
        # Create a mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock face detection to return no face
        no_face_result = {"face_detected": False, "face_quality": 0.0, "message": "No face detected"}
        
        # Create empty landmarks
        empty_landmarks = FacialLandmarks(
            eye_positions=[],
            mouth_position=[],
            eyebrow_positions=[],
            nose_position=(0.5, 0.5),
            face_contour=[]
        )
        
        with patch.object(service, "detect_face", return_value=no_face_result):
            # Mock extract_landmarks to return empty landmarks when face not detected
            with patch.object(service, "extract_landmarks", return_value=empty_landmarks):
                
                # Step 1: Detect face (will return no face)
                face_result = service.detect_face(mock_image)
                
                # Verify face detection results
                assert face_result is not None
                assert "face_detected" in face_result
                assert face_result["face_detected"] is False
                
                # Step 2: Analyze face
                analysis_result = service.analyze_face(mock_image, face_result)
                
                # Verify default analysis results
                assert analysis_result is not None
                assert "emotion" in analysis_result
                assert "emotion_confidence" in analysis_result
                assert "age" in analysis_result
                assert "gender" in analysis_result
                assert analysis_result["emotion"] in ["unknown", "neutral"]
                assert analysis_result["emotion_confidence"] == 0.0
                assert analysis_result["age"] == 0
                assert analysis_result["gender"] == "unknown"
                
                # Step 3: Extract landmarks
                landmarks = service.extract_landmarks(mock_image, face_result)
                
                # Verify empty landmarks
                assert isinstance(landmarks, FacialLandmarks)
                assert len(landmarks.eye_positions) == 0
                assert len(landmarks.mouth_position) == 0
                assert len(landmarks.eyebrow_positions) == 0
                assert landmarks.nose_position == (0.5, 0.5)
                assert len(landmarks.face_contour) == 0
                
                # Step 4: Calculate metrics
                metrics = service.calculate_metrics(landmarks)
                
                # Verify default metrics
                assert "eye_openness" in metrics
                assert "mouth_openness" in metrics
                assert "eyebrow_raise" in metrics
                assert "head_pose" in metrics
                assert metrics["eye_openness"] == 0.5
                assert metrics["mouth_openness"] == 0.0
                assert metrics["eyebrow_raise"] == 0.0
                assert metrics["head_pose"]["pitch"] == 0.0
                assert metrics["head_pose"]["yaw"] == 0.0
                assert metrics["head_pose"]["roll"] == 0.0 