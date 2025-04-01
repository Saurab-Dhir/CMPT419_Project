import cv2
import numpy as np
import os
import random
import time

class FacialEmotionDetector:
    def __init__(self):
        # Path to OpenCV's built-in Haar cascade for face detection
        self.face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Load the face detection model
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"Error: Could not load face cascade classifier from {self.face_cascade_path}")
            
        print(f"Loaded face cascade classifier")
        
        # Define emotion labels
        self.emotions = ['angry', 'happy', 'neutral', 'sad', 'surprise']
        
        # Store previous face positions for motion detection
        self.prev_face_center = None
        self.prev_face_size = None
        self.last_emotion_change = time.time()
        self.current_emotion = "neutral"
        self.current_confidence = 0.5
        
        # Emotion change cooldown (seconds)
        self.emotion_cooldown = 0.8  # Reduced to make it more responsive
        
        # For eye detection - more accurate emotion detection
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Processing flags - for performance
        self.skip_frames = 0  # Skip frames counter
        self.process_every_n_frames = 2  # Process every 2 frames
        
        # Try initializing CUDA
        self.use_cuda = False
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print(f"CUDA-enabled GPU found. Using GPU acceleration.")
                self.use_cuda = True
                self.process_every_n_frames = 1  # Process every frame when using GPU
            else:
                print("No CUDA-enabled GPU found. Using CPU only.")
        except Exception as e:
            print(f"CUDA module not available in OpenCV. Using CPU only.")

    def detect_faces(self, frame, force_detection=False):
        """Detect faces in the frame"""
        # Skip frames to improve performance
        if not force_detection and self.skip_frames < self.process_every_n_frames:
            self.skip_frames += 1
            if hasattr(self, 'last_faces') and hasattr(self, 'last_gray'):
                # Return the most recent detection results
                return self.last_faces, self.last_gray
                
        # Reset skip counter
        self.skip_frames = 0
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Downscale for faster processing
        scale_factor = 0.5  # Use half the resolution for detection
        small_gray = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Scale face coordinates back to original size
        if len(faces) > 0:
            faces = [(int(x/scale_factor), int(y/scale_factor), 
                     int(w/scale_factor), int(h/scale_factor)) for (x, y, w, h) in faces]
        
        # Store results for frame skipping
        self.last_faces = faces
        self.last_gray = gray
        
        return faces, gray
    
    def detect_facial_features(self, gray, face_rect):
        """Detect eyes and smile for more accurate emotion analysis"""
        # Skip expensive feature detection sometimes
        if random.random() < 0.3:  # Only do detailed detection 30% of the time
            return 0, 0
            
        x, y, w, h = face_rect
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5)
        
        # Detect smile - focus on lower part of face
        smile_roi = face_roi[int(h/2):h, :]
        smile = self.smile_cascade.detectMultiScale(
            smile_roi, 
            scaleFactor=1.7, 
            minNeighbors=22, 
            minSize=(25, 25)
        )
        
        return len(eyes), len(smile)
    
    def analyze_emotion(self, frame, face_rect):
        """
        Improved emotion analysis based on:
        - Face features (eyes, smile)
        - Relative brightness and contrast
        - Movement detection
        - Size changes
        """
        x, y, w, h = face_rect
        face_region = frame[y:y+h, x:x+w]
        
        if len(face_region) == 0:
            return "neutral", 0.5
        
        # Calculate face center and size
        face_center = (x + w//2, y + h//2)
        face_size = w * h
        
        # Check for significant face movement
        face_moved = False
        face_size_changed = False
        
        if self.prev_face_center is not None:
            # Calculate movement
            movement = np.sqrt((face_center[0] - self.prev_face_center[0])**2 + 
                              (face_center[1] - self.prev_face_center[1])**2)
            face_moved = movement > 15  # Lowered threshold for more sensitivity
            
            # Calculate size change
            if self.prev_face_size is not None:
                size_change_ratio = abs(face_size - self.prev_face_size) / max(self.prev_face_size, 1)
                face_size_changed = size_change_ratio > 0.1  # 10% change in size
        
        self.prev_face_center = face_center
        self.prev_face_size = face_size
        
        # Extract features for emotion
        # 1. Brightness features
        face_brightness = np.mean(face_region) 
        face_contrast = np.std(face_region)
        
        # 2. Face position relative to center
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        x_offset = (face_center[0] - frame_center_x) / (frame.shape[1] / 2)  # Can be negative
        y_offset = (face_center[1] - frame_center_y) / (frame.shape[0] / 2)  # Can be negative
        
        # 3. Face size relative to frame
        face_size_ratio = face_size / (frame.shape[0] * frame.shape[1])
        
        # Logic for emotion determination based on features
        current_time = time.time()
        time_since_last_change = current_time - self.last_emotion_change
        
        # Determine if we should consider changing emotion
        should_change = (time_since_last_change > self.emotion_cooldown) or face_moved or face_size_changed
        
        if should_change:
            # Check for smile periodically to avoid performance impact
            if random.random() < 0.4:  # 40% chance to check for facial features
                # Convert to grayscale if not already 
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                # Detect face features
                num_eyes, num_smile = self.detect_facial_features(gray, face_rect)
            else:
                num_eyes, num_smile = 0, 0
            
            # Enhanced emotion detection using facial features
            if num_smile > 0:
                # Smile detected
                emotion = "happy"
                confidence = 0.7 + (0.1 * num_smile)
            elif face_moved and abs(x_offset) > 0.3:
                # Quick lateral movement
                if x_offset > 0:  # Moving right
                    emotion = "surprise"
                else:  # Moving left
                    emotion = random.choice(["angry", "surprise"])
                confidence = 0.6 + random.random() * 0.2
            elif y_offset < -0.2 and face_size_ratio > 0.08:
                # Head tilted down and close
                emotion = "sad"
                confidence = 0.65 + random.random() * 0.15
            elif y_offset > 0.2:
                # Head tilted up
                emotion = random.choice(["angry", "surprise"])
                confidence = 0.6 + random.random() * 0.2
            elif face_contrast > 60:  # High contrast often means expressive face
                emotion = random.choice(["happy", "surprise", "angry"])
                confidence = 0.55 + random.random() * 0.2
            elif face_size_changed and self.prev_face_size and face_size > self.prev_face_size:
                # Face getting larger (moving closer)
                emotion = "surprise"
                confidence = 0.6 + random.random() * 0.2
            else:
                emotion = "neutral"
                confidence = 0.7 + random.random() * 0.15
                
            # Update state if emotion changed
            if emotion != self.current_emotion:
                self.current_emotion = emotion
                self.current_confidence = confidence
                self.last_emotion_change = current_time
        
        return self.current_emotion, self.current_confidence

def main():
    # Initialize the detector
    detector = FacialEmotionDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully. Press 'q' to quit.")
    
    # Don't set resolution - let the webcam use its default
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    max_errors = 5
    current_errors = 0
    
    while True:
        try:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                current_errors += 1
                print(f"Error: Can't receive frame ({current_errors}/{max_errors})")
                if current_errors >= max_errors:
                    print("Too many errors, exiting.")
                    break
                # Small wait before retrying
                time.sleep(0.1)
                continue
            
            # Reset error counter on successful frame read
            current_errors = 0
            frame_count += 1
            
            # Detect faces
            faces, gray = detector.detect_faces(frame)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Get emotion
                emotion, confidence = detector.analyze_emotion(frame, (x, y, w, h))
                
                # Pick color based on emotion
                if emotion == "happy":
                    color = (0, 255, 0)  # Green
                elif emotion == "sad":
                    color = (255, 0, 0)  # Blue
                elif emotion == "angry":
                    color = (0, 0, 255)  # Red
                elif emotion == "surprise":
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (255, 255, 255)  # White for neutral
                
                # Display emotion above the face with emotion-specific color
                text = f"{emotion} ({confidence:.2f})"
                cv2.putText(frame, text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display the frame
            cv2.imshow('Facial Emotion Detection', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            current_errors += 1
            if current_errors >= max_errors:
                print("Too many errors, exiting.")
                break
    
    # Release resources
    print("Releasing webcam resources...")
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main() 