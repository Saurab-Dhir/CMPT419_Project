# EmotiVoice: Multimodal Emotion Recognition System

A comprehensive multimodal emotion analysis system that processes webcam video and microphone input to detect emotions from facial expressions, speech tone, and language content. The system generates empathetic responses based on the detected emotions.

## Features

- **Multimodal Emotion Detection**: Detects emotions from three sources:
  - **Facial Emotion**: Analyzes facial expressions using DeepFace
  - **Tonal Emotion**: Analyzes voice tone using audio classification
  - **Semantic Emotion**: Extracts emotion from speech content using Gemini AI
  
- **Real-time Processing**: Processes webcam video and microphone input in real-time
- **Empathetic AI Responses**: Generates contextually appropriate responses based on detected emotions
- **Text-to-Speech**: Converts AI responses to natural-sounding speech using ElevenLabs
- **WebSocket Communication**: Enables real-time bidirectional communication

## Complete System Workflow

1. **User Input Collection**
   - Webcam captures video frames at regular intervals
   - Microphone records audio in chunks and buffers it
   - WebSocket sends both to the server when sufficient data is collected

2. **Video Processing (Facial Emotion)**
   - DeepFace service detects faces in the video frame
   - Facial landmarks and features are extracted
   - Emotion classification is performed on the face (happy, sad, angry, etc.)
   - Results are stored with confidence scores

3. **Audio Processing (Dual Analysis)**
   - **Speech-to-Text Transcription**:
     - Audio is converted to base64 format
     - Sent to Gemini API for transcription
     - Transcript is analyzed for semantic emotional content
   - **Tone Analysis**:
     - Audio features are extracted using specialized models
     - Voice tone is classified into emotional categories
     - Confidence scores are calculated for each emotion

4. **Emotion Integration**
   - All three emotion sources (facial, tonal, semantic) are combined in a `MultiModalEmotionInput` object
   - Each emotion type is preserved independently (not averaged or weighted)

5. **Response Generation**
   - The LLM service receives the transcription and all emotion data
   - A carefully crafted prompt is sent to Gemini API
   - The prompt includes instructions to consider all three emotion types
   - An empathetic response is generated based on context

6. **Text-to-Speech Conversion**
   - The generated text response is sent to ElevenLabs API
   - Natural-sounding speech is synthesized
   - Audio file is saved and URL is generated

7. **Response Delivery**
   - Complete response package is sent back via WebSocket:
     - Original transcription
     - All detected emotions (facial, tonal, semantic)
     - Generated text response
     - Audio URL for playback

## Requirements

- Python 3.7+
- FastAPI
- WebSockets
- OpenCV
- NumPy
- TensorFlow/PyTorch (for tone analysis)
- DeepFace (for facial emotion detection)
- Google Gemini API key
- ElevenLabs API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Saurab-Dhir/CMPT419_Project
cd CMPT419_Project
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

4. Set up environment variables by creating a `.env` file:
```
ELEVENLABS_API_KEY=your_elevenlabs_api_key
LLM_API_KEY=your_gemini_api_key
```

5. Install DeepFace for enhanced facial analysis:
```bash
pip install deepface
```

## Running the Application

1. Start the server:
```bash
python python -m uvicorn app.main:app --reload
```

2. Open your browser and navigate to:
```
http://localhost:8000/3d-conversation
```

3. Allow access to your webcam and microphone when prompted

4. Use the application:
   - Click "Start Recording" to begin capturing audio and video
   - Speak naturally while facing the camera
   - Click "Stop Recording" when finished
   - View the analysis results and AI response
   - Listen to the synthesized speech response

## Project Structure

```
app/
├── models/             # Data models for the application
│   ├── audio.py        # Audio processing models
│   ├── response.py     # Response models including MultiModalEmotionInput
│   └── visual.py       # Visual processing models
├── routers/            # API route definitions
│   ├── realtime.py     # WebSocket handler for real-time processing
├── services/           # Core service implementations
│   ├── audio_service.py       # Audio processing service
│   ├── deepface_service.py    # Facial analysis service
│   ├── elevenlabs_service.py  # Text-to-speech service
│   ├── llm_service.py         # LLM service using Gemini API
│   ├── multimodal_service.py  # Combined multimodal processing
│   ├── stt_service.py         # Speech-to-text service
│   ├── tone_service.py        # Tone analysis service
│   └── visual_service.py      # Visual processing service
├── utils/              # Utility functions
├── main.py             # FastAPI application setup
static/
├── webcam.html         # Frontend interface for webcam interaction
├── styles.css          # Styling for the interface
└── js/                 # JavaScript for the frontend
```

## Troubleshooting

- **No emotions detected**: Ensure good lighting and clear audio
- **Facial emotion always neutral**: Try different expressions or check lighting
- **No audio transcription**: Check microphone permissions and audio levels
- **API key errors**: Verify your API keys in the `.env` file

## License

This project is proprietary and protected by copyright. All rights reserved. Unauthorized copying, modification, distribution, or use of this software is strictly prohibited unless permission granted from author.

## Acknowledgments

- DeepFace for facial analysis
- Google Gemini for LLM and STT capabilities
- ElevenLabs for high-quality text-to-speech
- FastAPI and WebSockets for the real-time server