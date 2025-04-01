import os
import logging
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

# Import models
from app.models.visual import (
    FacialLandmarks,
    FacialFeatures,
    FacialEmotionPrediction,
    VisualProcessingResult
)

# Configure logging
logger = logging.getLogger(__name__)

class DeepFaceService:
    """Service for facial detection and analysis using DeepFace and OpenCV."""

    # Standard emotion labels for normalization
    STANDARD_EMOTIONS = {
        "angry": ["anger", "angry"],
        "disgust": ["disgust", "disgusted"],
        "fear": ["fear", "feared", "fearful", "afraid"],
        "happy": ["happiness", "happy", "joy", "joyful"],
        "sad": ["sad", "sadness", "unhappy"],
        "surprise": ["surprise", "surprised", "surprising", "amazed", "shock", "shocked", "surpri_se"],
        "neutral": ["neutral", "none"]
    }
    
    # Gender normalization
    GENDER_MAPPING = {
        "woman": "female",
        "man": "male",
        "female": "female",
        "male": "male",
        "f": "female",
        "m": "male"
    }

    def __init__(self, use_opencv_fallback: bool = True):
        """
        Initialize the DeepFace service.
        
        Args:
            use_opencv_fallback: Whether to use OpenCV as fallback if DeepFace is not available
        """
        self.use_opencv_fallback = use_opencv_fallback
        self._initialize_models()
        logger.info("DeepFaceService initialized")
    
    def _initialize_models(self):
        """Load and initialize required models."""
        try:
            # Import DeepFace lazily to avoid import errors if not installed
            import cv2
            self.cv2 = cv2
            
            try:
                from deepface import DeepFace
                self.deepface = DeepFace
                self.has_deepface = True
                logger.info("DeepFace models initialized successfully")
            except ImportError:
                self.has_deepface = False
                logger.warning("DeepFace not available, using OpenCV fallback")
                
            # Initialize OpenCV face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Try to load DNN face detector if available
            try:
                model_file = os.path.join(cv2.data.haarcascades, "..", "face_detector", "opencv_face_detector_uint8.pb")
                config_file = os.path.join(cv2.data.haarcascades, "..", "face_detector", "opencv_face_detector.pbtxt")
                
                if os.path.exists(model_file) and os.path.exists(config_file):
                    self.face_net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                    self.has_dnn_detector = True
                    logger.info("DNN face detector loaded successfully")
                else:
                    self.has_dnn_detector = False
                    logger.warning("DNN face detector model files not found")
            except Exception as e:
                self.has_dnn_detector = False
                logger.warning(f"Failed to load DNN face detector: {str(e)}")
            
            # Initialize emotion classifier (simple fallback)
            self.emotion_labels = ["angry", "sad", "happy", "surprise", "neutral"]
            
            # Try to load pre-trained emotion classifier if available
            try:
                emotion_model_path = os.path.join(cv2.data.haarcascades, "..", "emotion_ferplus", "emotion_ferplus.onnx")
                if os.path.exists(emotion_model_path):
                    self.emotion_net = cv2.dnn.readNetFromONNX(emotion_model_path)
                    logger.info("Emotion classifier loaded successfully")
                else:
                    logger.warning("Emotion classifier model file not found")
            except Exception as e:
                logger.warning(f"Failed to load emotion classifier: {str(e)}")
            
            # Try to load dlib for facial landmarks if available
            try:
                import dlib
                self.dlib = dlib
                
                # Initialize face detector and shape predictor
                self.face_detector = dlib.get_frontal_face_detector()
                
                # Try to locate the shape predictor model file
                predictor_path = os.path.join(cv2.data.haarcascades, "..", "shape_predictor_68_face_landmarks.dat")
                if not os.path.exists(predictor_path):
                    # Alternative paths to check
                    alt_paths = [
                        "shape_predictor_68_face_landmarks.dat",
                        os.path.join("models", "shape_predictor_68_face_landmarks.dat")
                    ]
                    for path in alt_paths:
                        if os.path.exists(path):
                            predictor_path = path
                            break
                
                if os.path.exists(predictor_path):
                    self.shape_predictor = dlib.shape_predictor(predictor_path)
                    self.has_dlib = True
                    logger.info("dlib facial landmark detector loaded successfully")
                else:
                    self.has_dlib = False
                    logger.warning("dlib shape predictor model file not found")
            except ImportError:
                self.has_dlib = False
                logger.warning("dlib not available, using OpenCV fallback for facial landmarks")
            except Exception as e:
                self.has_dlib = False
                logger.warning(f"Failed to initialize dlib: {str(e)}")
            
            # Initialize OpenCV facial landmark detector as fallback
            try:
                # Try to initialize face mesh detector from MediaPipe if available
                import mediapipe as mp
                self.mp = mp
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5
                )
                self.has_mediapipe = True
                logger.info("MediaPipe face mesh detector loaded successfully")
            except ImportError:
                self.has_mediapipe = False
                logger.warning("MediaPipe not available for facial landmarks")
                
        except ImportError as e:
            logger.error(f"Failed to initialize required dependencies: {str(e)}")
            raise
    
    def _normalize_emotion(self, emotion: str) -> str:
        """
        Normalize emotion names to standard format.
        
        Args:
            emotion: Input emotion name
            
        Returns:
            Normalized emotion name
        """
        emotion = emotion.lower().strip()
        
        # Try direct match first
        if emotion in self.STANDARD_EMOTIONS:
            return emotion
        
        # Try to find in variants
        for standard, variants in self.STANDARD_EMOTIONS.items():
            if emotion in variants:
                return standard
        
        # Default to neutral if no match
        return "neutral"
    
    def _normalize_gender(self, gender: str) -> str:
        """
        Normalize gender to standard format.
        
        Args:
            gender: Input gender string
            
        Returns:
            Normalized gender string (male, female, or unknown)
        """
        if not gender:
            return "unknown"
            
        gender = gender.lower().strip()
        return self.GENDER_MAPPING.get(gender, "unknown")
    
    def detect_face(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect face in an image.
        
        Args:
            image: Input image path or numpy array
            
        Returns:
            Dictionary with face detection results including:
                - detected: Whether a face was detected
                - quality: Confidence score (0.0-1.0)
                - box: Coordinates of the detected face (x, y, w, h)
                - all_faces: List of all detected faces (if multiple)
                - error: Error message (if any)
        """
        logger.info("Detecting faces in image")
        
        try:
            # Convert image path to numpy array if needed
            if isinstance(image, str):
                logger.debug(f"Loading image from path: {image}")
                img = self.cv2.imread(image)
                if img is None:
                    logger.error(f"Invalid image path or corrupted image: {image}")
                    return {
                        "detected": False,
                        "quality": 0.0,
                        "error": f"Invalid image path or corrupted image: {image}"
                    }
            else:
                img = image
            
            # Get image dimensions
            img_height, img_width = img.shape[:2]
            
            # Try DeepFace first if available
            if self.has_deepface:
                try:
                    logger.debug("Attempting face detection with DeepFace")
                    faces = self.deepface.extract_faces(
                        img, 
                        detector_backend='opencv',
                        enforce_detection=False,
                        align=True
                    )
                    
                    if faces and len(faces) > 0:
                        # Get the face with highest confidence
                        best_face = max(faces, key=lambda x: x.get('confidence', 0))
                        
                        # Extract face coordinates
                        facial_area = best_face.get('facial_area', {})
                        face_box = [
                            facial_area.get('x', 0),
                            facial_area.get('y', 0),
                            facial_area.get('w', 0),
                            facial_area.get('h', 0)
                        ]
                        
                        # If multiple faces detected, include all of them
                        all_faces = []
                        if len(faces) > 1:
                            for face in faces:
                                area = face.get('facial_area', {})
                                all_faces.append([
                                    area.get('x', 0),
                                    area.get('y', 0),
                                    area.get('w', 0),
                                    area.get('h', 0)
                                ])
                        
                        result = {
                            "detected": True,
                            "quality": best_face.get('confidence', 0.8),
                            "box": face_box
                        }
                        
                        if all_faces:
                            result["all_faces"] = all_faces
                        
                        logger.info(f"DeepFace detected {len(faces)} face(s)")
                        return result
                
                except Exception as e:
                    logger.warning(f"DeepFace face detection failed: {str(e)}, falling back to OpenCV")
            
            # Try DNN-based detector if available
            if self.has_dnn_detector:
                try:
                    logger.debug("Attempting face detection with DNN")
                    # Prepare input blob for the network
                    blob = self.cv2.dnn.blobFromImage(
                        img, 1.0, (300, 300), [104, 117, 123], False, False
                    )
                    
                    # Set input and perform inference
                    self.face_net.setInput(blob)
                    detections = self.face_net.forward()
                    
                    # Process detections
                    faces = []
                    confidences = []
                    
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        
                        # Filter out weak detections
                        if confidence > 0.5:
                            # Get normalized coordinates
                            x1 = int(detections[0, 0, i, 3] * img_width)
                            y1 = int(detections[0, 0, i, 4] * img_height)
                            x2 = int(detections[0, 0, i, 5] * img_width)
                            y2 = int(detections[0, 0, i, 6] * img_height)
                            
                            # Convert to x, y, w, h format
                            x = max(0, x1)
                            y = max(0, y1)
                            w = min(img_width - x, x2 - x1)
                            h = min(img_height - y, y2 - y1)
                            
                            faces.append([x, y, w, h])
                            confidences.append(confidence)
                    
                    if len(faces) > 0:
                        # Find face with highest confidence
                        best_idx = np.argmax(confidences)
                        
                        result = {
                            "detected": True,
                            "quality": float(confidences[best_idx]),
                            "box": faces[best_idx]
                        }
                        
                        # Include all faces if multiple detected
                        if len(faces) > 1:
                            result["all_faces"] = faces
                        
                        logger.info(f"DNN detected {len(faces)} face(s)")
                        return result
                
                except Exception as e:
                    logger.warning(f"DNN face detection failed: {str(e)}, falling back to Haar Cascade")
            
            # Fall back to Haar Cascade
            logger.debug("Attempting face detection with Haar Cascade")
            gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
            
            # Log image properties and shape
            logger.info(f"Image shape: {img.shape}, type: {img.dtype}")
            logger.info(f"Attempting Haar Cascade detection with image of size {img.shape[:2]}")
            
            # Preprocess the grayscale image to improve contrast
            try:
                # Apply histogram equalization to improve contrast
                equalized = self.cv2.equalizeHist(gray)
                
                # Try with the equalized image first
                faces = self.face_cascade.detectMultiScale(
                    equalized, 
                    scaleFactor=1.1, 
                    minNeighbors=3,
                    minSize=(20, 20)
                )
                
                logger.info(f"Haar Cascade with equalized image: found {len(faces)} faces")
                
                # If no faces found, try original grayscale with more lenient parameters
                if len(faces) == 0:
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.05,  # More gradual scaling
                        minNeighbors=2,    # Even more lenient
                        minSize=(20, 20)
                    )
                    logger.info(f"Haar Cascade with lenient parameters: found {len(faces)} faces")
            except Exception as e:
                logger.warning(f"Image preprocessing failed: {str(e)}, falling back to basic detection")
                # Fall back to original detection
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=3,
                    minSize=(20, 20)
                )
                logger.info(f"Haar Cascade detection result: found {len(faces)} faces")
            
            if len(faces) > 0:
                # If multiple faces, select the largest one (by area)
                if len(faces) > 1:
                    areas = [w * h for (x, y, w, h) in faces]
                    largest_idx = np.argmax(areas)
                    face_box = faces[largest_idx]
                else:
                    face_box = faces[0]
                
                # Estimate quality based on face size relative to image
                face_area = face_box[2] * face_box[3]
                image_area = img_width * img_height
                quality = min(0.9, face_area / image_area * 10)  # Normalize and cap at 0.9
                
                result = {
                    "detected": True,
                    "quality": float(quality),
                    "box": face_box.tolist() if isinstance(face_box, np.ndarray) else face_box
                }
                
                # Include all faces if multiple detected
                if len(faces) > 1:
                    result["all_faces"] = faces.tolist() if isinstance(faces, np.ndarray) else faces
                
                logger.info(f"Haar Cascade detected {len(faces)} face(s)")
                return result
            else:
                logger.info("No faces detected in the image")
                return {
                    "detected": False,
                    "quality": 0.0
                }
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return {
                "detected": False,
                "quality": 0.0,
                "error": f"Face detection failed: {str(e)}"
            }
    
    def analyze_face(self, image: Union[str, np.ndarray], detected_face: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze facial attributes including emotion, age, gender.
        
        Args:
            image: Input image path or numpy array
            detected_face: Previously detected face information (optional)
            
        Returns:
            Dictionary with facial analysis results including:
                - emotion: Primary emotion detected
                - emotion_confidence: Confidence in primary emotion
                - secondary_emotions: Dictionary of other emotions with confidences
                - age: Estimated age
                - gender: Detected gender
                - error: Error message (if any)
        """
        logger.info("Analyzing facial attributes")
        
        # Default return values
        default_result = {
            "emotion": "neutral",
            "emotion_confidence": 0.0,
            "secondary_emotions": {},
            "age": 0,
            "gender": "unknown"
        }
        
        # Check if face was detected
        if detected_face is None or not detected_face.get("detected", False):
            logger.info("No face detected, returning default values")
            return default_result
        
        try:
            # Load image if path provided
            if isinstance(image, str):
                logger.debug(f"Loading image from path: {image}")
                img = self.cv2.imread(image)
                if img is None:
                    logger.error(f"Invalid image path or corrupted image: {image}")
                    return {**default_result, "error": f"Invalid image path or corrupted image: {image}"}
            else:
                img = image
            
            # Extract face region if face box is provided
            face_box = detected_face.get("box")
            if face_box and len(face_box) == 4:
                x, y, w, h = face_box
                # Ensure coordinates are within image bounds
                x, y = max(0, x), max(0, y)
                w = min(img.shape[1] - x, w)
                h = min(img.shape[0] - y, h)
                
                # Extract face ROI with padding to improve accuracy
                padding_percent = 0.2  # Add 20% padding around the face
                pad_x = int(w * padding_percent)
                pad_y = int(h * padding_percent)
                
                # Calculate padded coordinates while ensuring they're within image bounds
                padded_x = max(0, x - pad_x)
                padded_y = max(0, y - pad_y)
                padded_w = min(img.shape[1] - padded_x, w + 2 * pad_x)
                padded_h = min(img.shape[0] - padded_y, h + 2 * pad_y)
                
                # Extract the padded face region
                face_img = img[padded_y:padded_y + padded_h, padded_x:padded_x + padded_w]
                
                # Check if face extraction succeeded
                if face_img.size == 0:
                    logger.warning("Failed to extract valid face region, using full image")
                    face_img = img
                
                logger.debug("Face region extracted for analysis")
            else:
                face_img = img
                logger.debug("Using full image for analysis (no face box provided)")
            
            # Try DeepFace first if available
            if self.has_deepface:
                try:
                    logger.debug("Analyzing face with DeepFace directly")
                    
                    # Use DeepFace directly for emotion analysis
                    analysis = self.deepface.analyze(
                        img_path=face_img,  # Pass the face image directly
                        actions=['emotion'],  # Only analyze emotions for speed
                        enforce_detection=False,  # Skip detection since we already have the face
                        detector_backend='skip'  # Skip detection since we already extracted the face
                    )
                    
                    if not analysis:
                        logger.warning("DeepFace returned empty analysis results")
                        raise ValueError("Empty analysis results")
                        
                    # Get the result (will be a list of one item or a single dict)
                    result = analysis[0] if isinstance(analysis, list) else analysis
                    
                    # Extract emotions
                    emotions = result.get("emotion", {})
                    if not emotions:
                        logger.warning("No emotion data in DeepFace result")
                        raise ValueError("No emotion data")
                    
                    # Find the dominant emotion (highest confidence)
                    emotion_items = sorted(
                        emotions.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    
                    # Get primary emotion
                    primary_emotion = self._normalize_emotion(emotion_items[0][0])
                    primary_confidence = emotion_items[0][1]
                    
                    # Get secondary emotions
                    secondary_emotions = {}
                    for emotion_name, confidence in emotion_items[1:]:
                        if confidence > 0.05:  # Only include emotions with confidence above 5%
                            secondary_emotions[self._normalize_emotion(emotion_name)] = confidence
                    
                    # Log the emotion detection results
                    logger.info(f"DeepFace emotion detection: {primary_emotion} ({primary_confidence:.2f})")
                    logger.debug(f"Secondary emotions: {secondary_emotions}")
                    
                    # Return the emotion analysis results
                    analysis_result = {
                        "emotion": primary_emotion,
                        "emotion_confidence": primary_confidence,
                        "secondary_emotions": secondary_emotions,
                        "age": 0,  # We're only analyzing emotions
                        "gender": "unknown"  # We're only analyzing emotions
                    }
                    
                    return analysis_result
                    
                except Exception as e:
                    logger.warning(f"Direct DeepFace emotion analysis failed: {str(e)}, trying conventional approach")
                    # Continue with conventional approach below
            
            # If we reached here, try the original approach with DeepFace if available
            if self.has_deepface:
                try:
                    logger.debug("Analyzing face with conventional DeepFace approach")
                    analysis = self.deepface.analyze(
                        face_img,
                        actions=['emotion', 'age', 'gender'],
                        enforce_detection=False,
                        detector_backend='skip'  # Skip detection as we already have the face
                    )
                    
                    # Rest of the original DeepFace analysis code...
                    if analysis and len(analysis) > 0:
                        result = analysis[0]
                        
                        # Extract emotions and sort by confidence
                        emotions = result.get("emotion", {})
                        emotion_items = sorted(
                            emotions.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        
                        # Get primary and secondary emotions
                        primary_emotion = "neutral"
                        primary_confidence = 0.0
                        secondary_emotions = {}
                        
                        # Handle the case when dominant_emotion is a dict or not string
                        try:
                            if "dominant_emotion" in result:
                                if isinstance(result["dominant_emotion"], str):
                                    primary_emotion = self._normalize_emotion(result["dominant_emotion"])
                                else:
                                    # If dominant_emotion is not a string, use the highest confidence emotion
                                    if emotion_items:
                                        primary_emotion = self._normalize_emotion(emotion_items[0][0])
                            elif emotion_items:
                                primary_emotion = self._normalize_emotion(emotion_items[0][0])
                        except Exception as e:
                            logger.warning(f"Error processing dominant emotion: {str(e)}")
                            if emotion_items:
                                primary_emotion = self._normalize_emotion(emotion_items[0][0])
                        
                        # Process emotion results
                        for emotion_name, confidence in emotion_items:
                            try:
                                norm_name = self._normalize_emotion(emotion_name)
                                if norm_name == primary_emotion:
                                    primary_confidence = confidence
                                else:
                                    # Only include emotions with some confidence
                                    if confidence > 0.01:
                                        secondary_emotions[norm_name] = confidence
                            except Exception as e:
                                logger.warning(f"Error normalizing emotion {emotion_name}: {str(e)}")
                                continue
                        
                        # Normalize gender
                        gender = "unknown"
                        if "gender" in result:
                            try:
                                if isinstance(result["gender"], str):
                                    gender = self._normalize_gender(result["gender"])
                            except Exception as e:
                                logger.warning(f"Error normalizing gender: {str(e)}")
                        
                        # Age can be used directly
                        age = result.get("age", 0)
                        
                        analysis_result = {
                            "emotion": primary_emotion,
                            "emotion_confidence": primary_confidence,
                            "secondary_emotions": secondary_emotions,
                            "age": age,
                            "gender": gender
                        }
                        
                        logger.info(f"DeepFace analysis complete: {primary_emotion} ({primary_confidence:.2f})")
                        return analysis_result
                
                except Exception as e:
                    logger.warning(f"Conventional DeepFace analysis failed: {str(e)}, using fallback")
                    # Continue to fallback method
                    default_result["error"] = f"DeepFace analysis failed: {str(e)}"
            
            # OpenCV fallback for emotion classification
            try:
                if hasattr(self, 'emotion_net'):
                    logger.debug("Analyzing face with OpenCV emotion classifier")
                    
                    # Preprocess image for emotion classification
                    face_gray = self.cv2.cvtColor(face_img, self.cv2.COLOR_BGR2GRAY)
                    face_resized = self.cv2.resize(face_gray, (64, 64))
                    face_normalized = face_resized / 255.0
                    face_blob = self.cv2.dnn.blobFromImage(face_normalized)
                    
                    # Run inference
                    self.emotion_net.setInput(face_blob)
                    emotion_preds = self.emotion_net.forward()[0]
                    
                    # Process emotion results
                    emotion_idx = np.argmax(emotion_preds)
                    primary_emotion = self.emotion_labels[emotion_idx]
                    primary_confidence = float(emotion_preds[emotion_idx])
                    
                    # Get secondary emotions
                    secondary_emotions = {}
                    for i, conf in enumerate(emotion_preds):
                        if i != emotion_idx and conf > 0.1:
                            secondary_emotions[self.emotion_labels[i]] = float(conf)
                    
                    # Combine with default values
                    default_result.update({
                        "emotion": primary_emotion,
                        "emotion_confidence": primary_confidence,
                        "secondary_emotions": secondary_emotions
                    })
                    
                    logger.info(f"OpenCV emotion analysis complete: {primary_emotion} ({primary_confidence:.2f})")
                else:
                    logger.warning("OpenCV emotion classifier not available, using default values")
                    # Add a minimal set of emotions as fallback
                    default_result.update({
                        "emotion": "neutral",
                        "emotion_confidence": 0.6,
                        "secondary_emotions": {"happy": 0.2, "sad": 0.1}
                    })
            except Exception as e:
                logger.warning(f"OpenCV emotion analysis failed: {str(e)}")
                # Add a minimal set of emotions as fallback
                default_result.update({
                    "emotion": "neutral",
                    "emotion_confidence": 0.5,
                    "secondary_emotions": {"happy": 0.3}
                })
            
            return default_result
            
        except Exception as e:
            logger.error(f"Face analysis failed: {str(e)}")
            return {**default_result, "error": f"Face analysis failed: {str(e)}"}
    
    def extract_landmarks(self, image: Union[str, np.ndarray], detected_face: Optional[Dict] = None) -> FacialLandmarks:
        """
        Extract facial landmarks from an image.
        
        Args:
            image: Input image path or numpy array
            detected_face: Previously detected face (optional)
            
        Returns:
            FacialLandmarks object with extracted points
        """
        logger.info("Extracting facial landmarks")
        
        # Default landmarks (fallback values)
        default_landmarks = FacialLandmarks(
            eye_positions=[(0.3, 0.4), (0.7, 0.4)],
            mouth_position=[(0.4, 0.7), (0.6, 0.7)],
            eyebrow_positions=[(0.3, 0.35), (0.7, 0.35)],
            nose_position=(0.5, 0.5),
            face_contour=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
        )
        
        # Check if face was detected
        if detected_face is None or not detected_face.get("detected", False):
            logger.info("No face detected, returning default landmarks")
            return default_landmarks
        
        try:
            # Load image if path provided
            if isinstance(image, str):
                logger.debug(f"Loading image from path: {image}")
                img = self.cv2.imread(image)
                if img is None:
                    logger.error(f"Invalid image path or corrupted image: {image}")
                    return default_landmarks
            else:
                img = image
            
            # Get image dimensions
            img_height, img_width = img.shape[:2]
            
            # Extract face box
            face_box = detected_face.get("box", [0, 0, img_width, img_height])
            x, y, w, h = face_box
            
            # Ensure coordinates are within image bounds
            x, y = max(0, x), max(0, y)
            w = min(img_width - x, w)
            h = min(img_height - y, h)
            
            # Try dlib for landmark detection if available
            if self.has_dlib:
                try:
                    logger.debug("Extracting landmarks with dlib")
                    
                    # Convert BGR to RGB for dlib
                    rgb_image = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
                    
                    # Convert face box to dlib rectangle
                    rect = self.dlib.rectangle(x, y, x + w, y + h)
                    
                    # Detect facial landmarks
                    shape = self.shape_predictor(rgb_image, rect)
                    
                    # Convert dlib shape to points
                    points = [(p.x, p.y) for p in shape.parts()]
                    
                    # Normalize coordinates
                    normalized_points = [(p[0] / img_width, p[1] / img_height) for p in points]
                    
                    # Extract specific landmark groups based on dlib's 68-point model
                    # References: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
                    
                    # Eye landmarks (36-47 in dlib's model)
                    left_eye = normalized_points[36:42]  # Left eye
                    right_eye = normalized_points[42:48]  # Right eye
                    eye_positions = left_eye + right_eye
                    
                    # Mouth landmarks (48-67 in dlib's model)
                    mouth_position = normalized_points[48:68]
                    
                    # Eyebrow landmarks (17-26 in dlib's model)
                    eyebrow_positions = normalized_points[17:27]
                    
                    # Nose landmark (30 in dlib's model)
                    nose_position = normalized_points[30]
                    
                    # Face contour landmarks (0-16 in dlib's model)
                    face_contour = normalized_points[0:17]
                    
                    # Create and return landmarks object
                    return FacialLandmarks(
                        eye_positions=eye_positions,
                        mouth_position=mouth_position,
                        eyebrow_positions=eyebrow_positions,
                        nose_position=nose_position,
                        face_contour=face_contour
                    )
                    
                except Exception as e:
                    logger.warning(f"dlib landmark detection failed: {str(e)}, using fallback")
            
            # Try MediaPipe for landmark detection if available
            if hasattr(self, 'has_mediapipe') and self.has_mediapipe:
                try:
                    logger.debug("Extracting landmarks with MediaPipe")
                    
                    # Convert BGR to RGB for MediaPipe
                    rgb_image = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
                    
                    # Process the image with MediaPipe
                    results = self.face_mesh.process(rgb_image)
                    
                    if results.multi_face_landmarks:
                        # Get the first face
                        face_landmarks = results.multi_face_landmarks[0]
                        
                        # Extract landmark coordinates and normalize
                        points = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                        
                        # MediaPipe returns normalized coordinates already, no need to normalize
                        # Based on MediaPipe's 468 points model: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
                        
                        # Extract specific landmark groups from MediaPipe's 468-point model
                        # These indices may need adjustment based on your exact requirements
                        
                        # Eye landmarks (approximate mapping)
                        left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
                        right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
                        
                        left_eye = [points[i] for i in left_eye_indices]
                        right_eye = [points[i] for i in right_eye_indices]
                        eye_positions = left_eye + right_eye
                        
                        # Mouth landmarks (approximate mapping)
                        mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
                        mouth_position = [points[i] for i in mouth_indices]
                        
                        # Eyebrow landmarks (approximate mapping)
                        left_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53]
                        right_eyebrow_indices = [336, 296, 334, 293, 300, 285, 295, 282, 283]
                        
                        eyebrow_positions = [points[i] for i in left_eyebrow_indices + right_eyebrow_indices]
                        
                        # Nose landmark
                        nose_position = points[1]  # Tip of nose
                        
                        # Face contour landmarks
                        contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162]
                        
                        face_contour = [points[i] for i in contour_indices]
                        
                        # Create and return landmarks object
                        return FacialLandmarks(
                            eye_positions=eye_positions,
                            mouth_position=mouth_position,
                            eyebrow_positions=eyebrow_positions,
                            nose_position=nose_position,
                            face_contour=face_contour
                        )
                    
                except Exception as e:
                    logger.warning(f"MediaPipe landmark detection failed: {str(e)}, using OpenCV fallback")
            
            # OpenCV-based fallback for facial landmarks
            try:
                logger.debug("Extracting landmarks with OpenCV")
                
                # Extract face region
                face_roi = img[y:y+h, x:x+w]
                
                # Convert to grayscale for better feature detection
                gray_roi = self.cv2.cvtColor(face_roi, self.cv2.COLOR_BGR2GRAY)
                
                # Use Haar cascades for eye detection
                eye_cascade = self.cv2.CascadeClassifier(self.cv2.data.haarcascades + 'haarcascade_eye.xml')
                eyes = eye_cascade.detectMultiScale(gray_roi, 1.1, 4)
                
                # Simple approach for features
                eye_positions = []
                if len(eyes) >= 2:
                    # Use detected eyes
                    for (ex, ey, ew, eh) in eyes[:2]:
                        # Add center of eye, normalized to image coordinates
                        center_x = (x + ex + ew/2) / img_width
                        center_y = (y + ey + eh/2) / img_height
                        eye_positions.append((center_x, center_y))
                else:
                    # Fallback to approximate eye positions
                    left_eye_x = x + w * 0.3
                    right_eye_x = x + w * 0.7
                    eyes_y = y + h * 0.4
                    
                    eye_positions = [
                        (left_eye_x / img_width, eyes_y / img_height),
                        (right_eye_x / img_width, eyes_y / img_height)
                    ]
                
                # Approximate other facial features based on face dimensions
                # Eyebrows slightly above eyes
                eyebrow_positions = [
                    (eye_positions[0][0], eye_positions[0][1] - 0.05),
                    (eye_positions[1][0], eye_positions[1][1] - 0.05)
                ]
                
                # Nose at center of face
                nose_x = (x + w/2) / img_width
                nose_y = (y + h*0.55) / img_height
                nose_position = (nose_x, nose_y)
                
                # Mouth below nose
                mouth_left_x = (x + w*0.4) / img_width
                mouth_right_x = (x + w*0.6) / img_width
                mouth_y = (y + h*0.7) / img_height
                
                mouth_position = [
                    (mouth_left_x, mouth_y),
                    (mouth_right_x, mouth_y)
                ]
                
                # Face contour based on detected face
                face_contour = [
                    (x / img_width, y / img_height),  # Top-left
                    ((x + w) / img_width, y / img_height),  # Top-right
                    ((x + w) / img_width, (y + h) / img_height),  # Bottom-right
                    (x / img_width, (y + h) / img_height)  # Bottom-left
                ]
                
                return FacialLandmarks(
                    eye_positions=eye_positions,
                    mouth_position=mouth_position,
                    eyebrow_positions=eyebrow_positions,
                    nose_position=nose_position,
                    face_contour=face_contour
                )
                
            except Exception as e:
                logger.warning(f"OpenCV landmark detection failed: {str(e)}, using default landmarks")
            
            # Return default landmarks as last resort
            logger.info("Using default landmarks")
            return default_landmarks
            
        except Exception as e:
            logger.error(f"Facial landmark extraction failed: {str(e)}")
            return default_landmarks
    
    def calculate_eye_openness(self, eye_points):
        """
        Calculate the openness of an eye based on its landmarks.
        
        Args:
            eye_points (list): List of (x, y) tuples representing eye landmarks.
            
        Returns:
            float: Value between 0 and 1 representing eye openness.
        """
        try:
            if not eye_points or len(eye_points) < 4:
                return 0.5  # Default value
            
            # Get vertical points (top and bottom of eye)
            sorted_by_y = sorted(eye_points, key=lambda p: p[1])
            top_point = sorted_by_y[0]
            bottom_point = sorted_by_y[-1]
            
            # Get horizontal points (left and right of eye)
            sorted_by_x = sorted(eye_points, key=lambda p: p[0])
            left_point = sorted_by_x[0]
            right_point = sorted_by_x[-1]
            
            # Calculate eye dimensions
            eye_height = bottom_point[1] - top_point[1]
            eye_width = right_point[0] - left_point[0]
            
            # Avoid division by zero
            if eye_width == 0:
                return 0.5
            
            # Calculate aspect ratio: height/width
            aspect_ratio = eye_height / eye_width
            
            # Normalize to 0-1 range (typical values range from 0.1 to 0.5)
            # A value close to 0 means closed, close to 1 means fully open
            normalized_openness = min(1.0, max(0.0, aspect_ratio * 5.0))
            
            return normalized_openness
            
        except Exception as e:
            self.logger.error(f"Error calculating eye openness: {str(e)}")
            return 0.5  # Default value
    
    def calculate_mouth_openness(self, mouth_points):
        """
        Calculate the openness of the mouth based on its landmarks.
        
        Args:
            mouth_points (list): List of (x, y) tuples representing mouth landmarks.
            
        Returns:
            float: Value between 0 and 1 representing mouth openness.
        """
        try:
            if not mouth_points or len(mouth_points) < 4:
                return 0.0  # Default value - mouth closed
            
            # Get vertical points for mouth
            sorted_by_y = sorted(mouth_points, key=lambda p: p[1])
            top_point = sorted_by_y[0]
            bottom_point = sorted_by_y[-1]
            
            # Get horizontal points for mouth
            sorted_by_x = sorted(mouth_points, key=lambda p: p[0])
            left_point = sorted_by_x[0]
            right_point = sorted_by_x[-1]
            
            # Calculate mouth dimensions
            mouth_height = bottom_point[1] - top_point[1]
            mouth_width = right_point[0] - left_point[0]
            
            # Avoid division by zero
            if mouth_width == 0:
                return 0.0
            
            # Calculate aspect ratio: height/width
            aspect_ratio = mouth_height / mouth_width
            
            # Normalize to 0-1 range (typical values range from 0.05 to 0.8)
            # A value close to 0 means closed, close to 1 means fully open
            normalized_openness = min(1.0, max(0.0, aspect_ratio * 2.5))
            
            return normalized_openness
            
        except Exception as e:
            self.logger.error(f"Error calculating mouth openness: {str(e)}")
            return 0.0  # Default value - mouth closed
    
    def calculate_eyebrow_raise(self, eyebrow_points, eye_points):
        """
        Calculate how raised the eyebrows are compared to the eyes.
        
        Args:
            eyebrow_points (list): List of (x, y) tuples representing eyebrow landmarks.
            eye_points (list): List of (x, y) tuples representing eye landmarks.
            
        Returns:
            float: Value between 0 and 1 representing eyebrow raise.
        """
        try:
            if not eyebrow_points or not eye_points or len(eyebrow_points) < 2 or len(eye_points) < 2:
                return 0.0  # Default value
            
            # Get average y-coordinate for eyebrows and eyes
            avg_eyebrow_y = sum(p[1] for p in eyebrow_points) / len(eyebrow_points)
            avg_eye_y = sum(p[1] for p in eye_points) / len(eye_points)
            
            # Calculate vertical distance between eyebrows and eyes
            vertical_distance = avg_eye_y - avg_eyebrow_y
            
            # Normalize to 0-1 range (typical values range from 0.02 to 0.2)
            # A value close to 0 means not raised, close to 1 means highly raised
            normalized_raise = min(1.0, max(0.0, vertical_distance * 10.0))
            
            return normalized_raise
            
        except Exception as e:
            self.logger.error(f"Error calculating eyebrow raise: {str(e)}")
            return 0.0  # Default value
    
    def _estimate_head_pose(self, landmarks):
        """
        Estimate the head pose (pitch, yaw, roll) from facial landmarks.
        
        Args:
            landmarks (FacialLandmarks): The facial landmarks object.
            
        Returns:
            dict: Dictionary containing pitch, yaw, and roll estimates.
        """
        try:
            if not landmarks.face_contour or len(landmarks.face_contour) < 10:
                return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
            
            # Simplified head pose estimation based on facial landmarks
            # For a more accurate estimation, a 3D model would be needed
            
            # Estimate roll (head tilt) based on eye positions
            if len(landmarks.eye_positions) >= 8:
                left_eye_center = (
                    sum(p[0] for p in landmarks.eye_positions[:4]) / 4,
                    sum(p[1] for p in landmarks.eye_positions[:4]) / 4
                )
                right_eye_center = (
                    sum(p[0] for p in landmarks.eye_positions[4:]) / 4,
                    sum(p[1] for p in landmarks.eye_positions[4:]) / 4
                )
                
                # Calculate angle of eye line relative to horizontal
                dx = right_eye_center[0] - left_eye_center[0]
                dy = right_eye_center[1] - left_eye_center[1]
                
                if dx == 0:
                    roll = 0.0
                else:
                    roll = np.arctan(dy / dx) * 180 / np.pi
            else:
                roll = 0.0
            
            # Estimate yaw (left-right head rotation) based on face contour asymmetry
            if len(landmarks.face_contour) >= 16:
                left_side = landmarks.face_contour[:8]
                right_side = landmarks.face_contour[8:]
                
                left_width = max(p[0] for p in left_side) - min(p[0] for p in left_side)
                right_width = max(p[0] for p in right_side) - min(p[0] for p in right_side)
                
                if left_width + right_width > 0:
                    yaw = ((right_width - left_width) / (right_width + left_width)) * 90
                else:
                    yaw = 0.0
            else:
                yaw = 0.0
            
            # Estimate pitch (up-down head rotation) based on face height ratio
            if landmarks.nose_position and len(landmarks.face_contour) >= 2:
                face_top = min(p[1] for p in landmarks.face_contour)
                face_bottom = max(p[1] for p in landmarks.face_contour)
                nose_y = landmarks.nose_position[1]
                
                face_height = face_bottom - face_top
                if face_height > 0:
                    # Normalize nose position relative to face height
                    nose_relative_pos = (nose_y - face_top) / face_height
                    # Convert to pitch angle (-45 to 45 degrees)
                    pitch = (nose_relative_pos - 0.5) * 90
                else:
                    pitch = 0.0
            else:
                pitch = 0.0
            
            return {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating head pose: {str(e)}")
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    
    def calculate_metrics(self, landmarks):
        """
        Calculate derived metrics from facial landmarks.
        
        Args:
            landmarks (FacialLandmarks): The facial landmarks object.
            
        Returns:
            dict: Dictionary containing the derived metrics.
        """
        metrics = {}
        
        # Calculate eye openness (average of both eyes)
        if landmarks.eye_positions and len(landmarks.eye_positions) >= 8:
            left_eye = landmarks.eye_positions[:4]
            right_eye = landmarks.eye_positions[4:8]
            
            left_openness = self.calculate_eye_openness(left_eye)
            right_openness = self.calculate_eye_openness(right_eye)
            
            metrics["eye_openness"] = (left_openness + right_openness) / 2
        else:
            metrics["eye_openness"] = 0.5  # Default value
        
        # Calculate mouth openness
        metrics["mouth_openness"] = self.calculate_mouth_openness(landmarks.mouth_position)
        
        # Calculate eyebrow raise
        metrics["eyebrow_raise"] = self.calculate_eyebrow_raise(
            landmarks.eyebrow_positions, 
            landmarks.eye_positions
        )
        
        # Estimate head pose
        metrics["head_pose"] = self._estimate_head_pose(landmarks)
        
        return metrics
    
    def process_image(self, image: Union[str, np.ndarray]) -> VisualProcessingResult:
        """
        Process an image to extract facial features and emotions.
        
        Args:
            image: Input image path or numpy array
            
        Returns:
            VisualProcessingResult object with all processing results
        """
        logger.info("Processing image for facial analysis")
        
        try:
            # Detect face
            face_detection = self.detect_face(image)
            
            if not face_detection.get("detected", False):
                # Return early with minimal result if no face detected
                return VisualProcessingResult(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    face_detected=False,
                    face_quality=0.0,
                    features=None,  # Will be populated with defaults or None
                    emotion_prediction=None  # Will be populated with defaults or None
                )
            
            # Analyze facial attributes
            analysis_result = self.analyze_face(image, face_detection)
            
            # Extract landmarks
            landmarks = self.extract_landmarks(image, face_detection)
            
            # Calculate metrics
            metrics = self.calculate_metrics(landmarks)
            
            # Create complete result
            result = VisualProcessingResult(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                face_detected=True,
                face_quality=face_detection.get("quality", 0.0),
                features=FacialFeatures(
                    landmarks=landmarks,
                    eye_openness=metrics.get("eye_openness", 0.0),
                    mouth_openness=metrics.get("mouth_openness", 0.0),
                    eyebrow_raise=metrics.get("eyebrow_raise", 0.0),
                    head_pose=metrics.get("head_pose", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
                ),
                emotion_prediction=FacialEmotionPrediction(
                    emotion=analysis_result.get("emotion", "neutral"),
                    confidence=analysis_result.get("emotion_confidence", 0.0),
                    secondary_emotions=analysis_result.get("secondary_emotions", {})
                )
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in image processing: {str(e)}")
            raise 