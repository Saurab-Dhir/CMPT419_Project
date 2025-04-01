# Facial Analysis System

A comprehensive facial analysis system that provides face detection, emotion recognition, age/gender estimation, facial landmark extraction, and derived metrics calculation for human-computer interaction applications.

## Features

- **Face Detection**: Detect faces in images using multiple methods (DeepFace, DNN, Haar Cascades)
- **Facial Analysis**: Extract emotions, age, and gender from detected faces
- **Facial Landmarks**: Extract detailed facial landmarks (eyes, mouth, eyebrows, nose, face contour)
- **Derived Metrics**: Calculate metrics like eye openness, mouth openness, eyebrow raise, and head pose
- **Fallback Mechanisms**: Graceful degradation when primary libraries are unavailable
- **Comprehensive Testing**: Complete test coverage for all components

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- DeepFace (optional, provides enhanced analysis)
- dlib (optional, provides enhanced landmark detection)
- MediaPipe (optional, provides alternative landmark detection)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/facial-analysis.git
cd facial-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install optional dependencies for enhanced features:
```bash
pip install deepface dlib mediapipe
```

## Usage

### As a Python Library

```python
from app.services.deepface_service import DeepFaceService

# Initialize the service
service = DeepFaceService()

# Process an image
image = cv2.imread("path/to/image.jpg")

# Step 1: Detect face
face_result = service.detect_face(image)

# Step 2: Analyze face
analysis = service.analyze_face(image, face_result)

# Step 3: Extract landmarks
landmarks = service.extract_landmarks(image, face_result)

# Step 4: Calculate metrics
metrics = service.calculate_metrics(landmarks)

# Access results
print(f"Emotion: {analysis['emotion']}")
print(f"Age: {analysis['age']}")
print(f"Gender: {analysis['gender']}")
print(f"Eye openness: {metrics['eye_openness']}")
```

### Demo Application

The system includes a demo application that can process images or video:

#### Process a single image:
```bash
python -m app.main --image path/to/image.jpg
```

#### Process a video file:
```bash
python -m app.main --video path/to/video.mp4
```

#### Use webcam for real-time analysis:
```bash
python -m app.main --video 0
```

## Project Structure

```
app/
├── models/
│   └── visual.py         # Data models for facial analysis
├── services/
│   └── deepface_service.py # Core service for facial analysis
├── tests/
│   ├── test_face_detection.py     # Tests for face detection
│   ├── test_facial_analysis.py    # Tests for emotion/age/gender
│   ├── test_facial_landmarks.py   # Tests for landmark extraction
│   ├── test_metrics_calculation.py # Tests for derived metrics
│   └── test_integration.py        # Integration tests
└── main.py               # Demo application
```

## Testing

Run the test suite to verify the functionality:

```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeepFace for facial analysis
- dlib for facial landmark detection
- OpenCV for computer vision utilities
- MediaPipe for alternative landmark detection
