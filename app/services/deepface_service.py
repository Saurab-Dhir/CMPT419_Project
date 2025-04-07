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
                # First check in our models directory
                model_paths = [
                    os.path.join("models", "opencv_face_detector_uint8.pb"),
                    os.path.join(cv2.data.haarcascades, "..", "face_detector", "opencv_face_detector_uint8.pb")
                ]
                config_paths = [
                    os.path.join("models", "opencv_face_detector.pbtxt"),
                    os.path.join(cv2.data.haarcascades, "..", "face_detector", "opencv_face_detector.pbtxt")
                ]
                
                model_file = None
                config_file = None
                
                # Find first available model and config files
                for path in model_paths:
                    if os.path.exists(path):
                        model_file = path
                        break
                
                for path in config_paths:
                    if os.path.exists(path):
                        config_file = path
                        break
                
                if model_file and config_file:
                    logger.info(f"Loading DNN face detector from {model_file} and {config_file}")
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
                emotion_model_paths = [
                    os.path.join("models", "emotion_ferplus.onnx"),
                    os.path.join(cv2.data.haarcascades, "..", "emotion_ferplus", "emotion_ferplus.onnx")
                ]
                
                emotion_model_path = None
                for path in emotion_model_paths:
                    if os.path.exists(path):
                        emotion_model_path = path
                        break
                
                if emotion_model_path:
                    logger.info(f"Loading emotion classifier from {emotion_model_path}")
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
                predictor_paths = [
                    os.path.join("models", "shape_predictor_68_face_landmarks.dat"),
                    os.path.join(cv2.data.haarcascades, "..", "shape_predictor_68_face_landmarks.dat"),
                    "shape_predictor_68_face_landmarks.dat"
                ]
                
                predictor_path = None
                for path in predictor_paths:
                    if os.path.exists(path):
                        predictor_path = path
                        break
                
                if predictor_path:
                    logger.info(f"Loading shape predictor from {predictor_path}")
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
        if not emotion:
            return "neutral"
            
        emotion = emotion.lower().strip()
        
        # Try direct match first
        if emotion in self.STANDARD_EMOTIONS:
            return emotion
        
        # Try to find in variants
        for standard, variants in self.STANDARD_EMOTIONS.items():
            if emotion in variants:
                return standard
                
        # Try partial matching for better fuzzy matching
        for standard, variants in self.STANDARD_EMOTIONS.items():
            for variant in variants:
                if variant in emotion or emotion in variant:
                    print(f"🔍 Fuzzy matched emotion '{emotion}' to standard '{standard}'")
                    return standard
        
        # Map any additional DeepFace-specific emotions
        deepface_mapping = {
            "frustration": "angry",
            "confusion": "surprise",
            "disappointment": "sad",
            "excitement": "happy",
            "anticipation": "surprise",
            "distress": "sad",
        }
        
        if emotion in deepface_mapping:
            mapped = deepface_mapping[emotion]
            print(f"🔍 Mapped DeepFace emotion '{emotion}' to '{mapped}'")
            return mapped
        
        # Default to neutral if no match
        print(f"⚠️ Unknown emotion '{emotion}', defaulting to neutral")
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
        Analyze face to extract attributes like emotion, age, gender.
        
        Args:
            image: Input image path or numpy array
            detected_face: Dictionary with face detection results (optional)
            
        Returns:
            Dictionary with facial attributes
        """
        logger.info("Analyzing face in image")
        print("🔎 Analyzing face for emotions...")
        
        # Enable debug visualization
        DEBUG_VISUALIZATION = True
        
        try:
            # Convert image path to numpy array if needed
            if isinstance(image, str):
                img = self.cv2.imread(image)
                if img is None:
                    logger.error(f"Invalid image path or corrupted image: {image}")
                    return {"emotion": "neutral", "emotion_confidence": 0.0}
            else:
                img = image
            
            # Detect face if not provided
            if detected_face is None or not detected_face.get("detected", False):
                detected_face = self.detect_face(img)
            
            # Check if face was detected
            if not detected_face.get("detected", False):
                logger.warning("No face detected, cannot analyze attributes")
                return {"emotion": "neutral", "emotion_confidence": 0.0}
            
            # Extract face region
            face_box = detected_face.get("box", [0, 0, img.shape[1], img.shape[0]])
            x, y, w, h = face_box
            face_img = img[y:y+h, x:x+w]
            
            # Ensure face image is not empty
            if face_img.size == 0:
                logger.warning("Empty face region")
                return {"emotion": "neutral", "emotion_confidence": 0.0}
                
            # Save face image for debugging
            if DEBUG_VISUALIZATION:
                import os
                debug_dir = os.path.join("debug", "faces")
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                debug_file = os.path.join(debug_dir, f"face_{timestamp}.jpg")
                self.cv2.imwrite(debug_file, face_img)
                print(f"🔎 Saved face image for debugging: {debug_file}")
            
            # Try DeepFace first if available
            if self.has_deepface:
                try:
                    print("🔎 Using DeepFace for emotion analysis")
                    # Use multiple detection backends for better results
                    backends = ['opencv', 'retinaface', 'mtcnn', 'ssd']
                    
                    # Try different backends until one works
                    face_analysis = None
                    used_backend = None
                    
                    for backend in backends:
                        try:
                            print(f"🔎 Trying DeepFace with {backend} backend")
                            face_analysis = self.deepface.analyze(
                                face_img, 
                                actions=['emotion'],
                                detector_backend=backend,
                                enforce_detection=False,
                                silent=True
                            )
                            if face_analysis and len(face_analysis) > 0:
                                used_backend = backend
                                print(f"🔎 Successfully detected with {backend} backend")
                                break
                        except Exception as backend_e:
                            print(f"🔎 {backend} backend failed: {str(backend_e)}")
                            continue
                    
                    if face_analysis and len(face_analysis) > 0:
                        analysis = face_analysis[0]
                        print(f"🔎 DeepFace analysis with {used_backend} backend raw result: {analysis}")
                        
                        # Extract emotion
                        emotions = analysis.get('emotion', {})
                        if emotions:
                            # Find emotion with highest confidence
                            emotions_list = [(e, c) for e, c in emotions.items()]
                            emotions_list.sort(key=lambda x: x[1], reverse=True)
                            
                            # Print all emotions with their confidence for debugging
                            print("🔎 All detected emotions:")
                            for emotion, confidence in emotions_list:
                                print(f"  - {emotion}: {confidence:.2f}%")
                            
                            # Get top emotions
                            primary_emotion, primary_confidence = emotions_list[0]
                            
                            # Adjust emotion detection threshold - if neutral is barely winning, use the second emotion
                            if primary_emotion == "neutral" and len(emotions_list) > 1:
                                second_emotion, second_confidence = emotions_list[1]
                                # If the second emotion is close in confidence (within 15%), use it instead
                                if primary_confidence - second_confidence < 15:
                                    print(f"🔎 Overriding neutral with close second emotion: {second_emotion} ({second_confidence:.2f}%)")
                                    primary_emotion = second_emotion
                                    primary_confidence = second_confidence
                            
                            primary_emotion = self._normalize_emotion(primary_emotion)
                            
                            # Get secondary emotions (excluding primary)
                            secondary_emotions = {
                                self._normalize_emotion(e): float(c) / 100.0 
                                for e, c in emotions_list[1:] 
                                if c > 1.0  # Only include emotions with >1% confidence
                            }
                            
                            # Normalize confidence to 0-1 range
                            primary_confidence = float(primary_confidence) / 100.0
                            
                            print(f"🔎 DeepFace detected emotion: {primary_emotion} ({primary_confidence:.2f})")
                            print(f"🔎 Secondary emotions: {secondary_emotions}")
                            
                            # Save annotated face with detected emotion for debugging
                            if DEBUG_VISUALIZATION:
                                import os
                                debug_dir = os.path.join("debug", "emotions")
                                os.makedirs(debug_dir, exist_ok=True)
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                
                                # Create a copy for annotation
                                annotated_face = face_img.copy()
                                
                                # Draw emotion text
                                text = f"{primary_emotion} ({primary_confidence:.2f})"
                                cv_font = self.cv2.FONT_HERSHEY_SIMPLEX
                                text_size = self.cv2.getTextSize(text, cv_font, 0.7, 2)[0]
                                text_x = (annotated_face.shape[1] - text_size[0]) // 2
                                
                                # Draw colored background based on emotion
                                color_map = {
                                    "happy": (0, 255, 0),    # Green
                                    "sad": (255, 0, 0),      # Blue
                                    "angry": (0, 0, 255),    # Red
                                    "fear": (255, 0, 255),   # Purple
                                    "surprise": (0, 255, 255), # Yellow
                                    "disgust": (0, 128, 128), # Brown
                                    "neutral": (128, 128, 128) # Gray
                                }
                                
                                color = color_map.get(primary_emotion, (255, 255, 255))
                                self.cv2.putText(annotated_face, text, (text_x, 30), cv_font, 0.7, color, 2)
                                
                                # Add second emotion if available
                                if secondary_emotions:
                                    second_emotion, second_conf = list(secondary_emotions.items())[0]
                                    second_text = f"{second_emotion} ({second_conf:.2f})"
                                    self.cv2.putText(annotated_face, second_text, (text_x, 60), cv_font, 0.5, color_map.get(second_emotion, (200, 200, 200)), 1)
                                
                                debug_file = os.path.join(debug_dir, f"emotion_{primary_emotion}_{timestamp}.jpg")
                                self.cv2.imwrite(debug_file, annotated_face)
                                print(f"🔎 Saved annotated emotion image: {debug_file}")
                            
                            return {
                                "emotion": primary_emotion,
                                "emotion_confidence": primary_confidence,
                                "secondary_emotions": secondary_emotions
                            }
                
                except Exception as e:
                    logger.warning(f"DeepFace analysis failed: {str(e)}")
                    print(f"⚠️ DeepFace analysis error: {str(e)}")
            
            # Fallback to OpenCV-based detection
            print("🔎 Falling back to OpenCV-based emotion detection")
            try:
                # Resize to expected size
                resized_face = self.cv2.resize(face_img, (48, 48))
                gray_face = self.cv2.cvtColor(resized_face, self.cv2.COLOR_BGR2GRAY)
                
                # Create a blob from the image
                blob = self.cv2.dnn.blobFromImage(
                    gray_face, 1.0, (48, 48), (0, 0, 0), 
                    swapRB=False, crop=False
                )
                
                # Check if we have the emotion network
                if hasattr(self, 'emotion_net'):
                    self.emotion_net.setInput(blob)
                    predictions = self.emotion_net.forward()
                    print(f"🔎 OpenCV emotion raw predictions: {predictions}")
                    
                    # Get the emotion with highest confidence
                    emotion_idx = np.argmax(predictions[0])
                    confidence = float(predictions[0][emotion_idx])
                    
                    # Map to emotion label
                    emotion_labels = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]
                    emotion = "neutral"
                    if emotion_idx < len(emotion_labels):
                        emotion = emotion_labels[emotion_idx]
                    
                    # Create secondary emotions
                    secondary_emotions = {}
                    for i, conf in enumerate(predictions[0]):
                        if i != emotion_idx and i < len(emotion_labels) and conf > 0.1:
                            secondary_emotions[emotion_labels[i]] = float(conf)
                    
                    print(f"🔎 OpenCV detected emotion: {emotion} ({confidence:.2f})")
                    print(f"🔎 Secondary emotions: {secondary_emotions}")
                    
                    return {
                        "emotion": self._normalize_emotion(emotion),
                        "emotion_confidence": confidence,
                        "secondary_emotions": secondary_emotions
                    }
                else:
                    # Very basic emotion detection based on facial features
                    landmarks = self.extract_landmarks(img, detected_face)
                    metrics = self.calculate_metrics(landmarks)
                    
                    eye_openness = metrics.get("eye_openness", 0.5)
                    mouth_openness = metrics.get("mouth_openness", 0.5)
                    eyebrow_raise = metrics.get("eyebrow_raise", 0.5)
                    
                    print(f"🔎 Facial metrics - Eye: {eye_openness:.2f}, Mouth: {mouth_openness:.2f}, Eyebrow: {eyebrow_raise:.2f}")
                    
                    # Very simplified rules for emotion detection
                    if mouth_openness > 0.7:
                        emotion = "surprise" if eyebrow_raise > 0.6 else "happy"
                    elif eyebrow_raise > 0.7:
                        emotion = "surprise"
                    elif eye_openness < 0.3:
                        emotion = "sad"
                    else:
                        emotion = "neutral"
                    
                    print(f"🔎 Basic feature-based emotion: {emotion} (confidence: 0.6)")
                    
                    # Create simple secondary emotions
                    secondary_emotions = {
                        "happy": 0.2,
                        "neutral": 0.2
                    }
                    if emotion != "happy" and emotion != "neutral":
                        secondary_emotions = {
                            "neutral": 0.4
                        }
                    
                    return {
                        "emotion": emotion,
                        "emotion_confidence": 0.6,  # Low confidence for this basic method
                        "secondary_emotions": secondary_emotions
                    }
                
            except Exception as e:
                logger.warning(f"OpenCV emotion detection failed: {str(e)}")
                print(f"⚠️ OpenCV emotion detection error: {str(e)}")
            
            # Default to neutral if all methods fail
            print("🔎 All emotion detection methods failed, defaulting to neutral")
            return {
                "emotion": "neutral",
                "emotion_confidence": 0.99,
                "secondary_emotions": {"happy": 0.01}
            }
            
        except Exception as e:
            logger.error(f"Error analyzing face: {str(e)}")
            print(f"⚠️ Face analysis error: {str(e)}")
            return {
                "emotion": "neutral",
                "emotion_confidence": 0.99,
                "secondary_emotions": {}
            }
    
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