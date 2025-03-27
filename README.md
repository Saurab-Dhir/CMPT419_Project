# Mirror Mirror on the Wall: Empathetic Self-talk Coach

## Project Overview
An AI-powered empathetic self-talk coach application that processes audio and visual inputs to understand emotional states and provide supportive, empathetic responses.

## Features
- Audio processing pipeline for speech capture and analysis
- Visual processing for facial expression analysis
- Emotion classification from combined inputs
- LLM-generated empathetic responses
- Text-to-Speech for audio feedback

## Workflow Architecture
Our application processes two types of input:

### 1. Audio Input Processing
Audio input goes through two parallel processes:
- **Speech-to-Text (STT)**: Using Google's Gemini API, we convert spoken language into text that can be analyzed and responded to.
- **Tone Analysis**: The audio is analyzed to detect emotional patterns in the user's voice (pitch, volume, speaking rate, pauses, etc.).

### 2. Video Input Processing
- **Facial Expression Analysis**: Using Meta DeepFace technology, we analyze facial expressions to detect emotions.
- **Feature Extraction**: We track facial landmarks, eye openness, mouth position, and head pose to enhance emotion detection.

### 3. Multi-modal Emotion Classification
- Results from both audio and visual analyses are combined to produce a more accurate emotional assessment.
- This hybrid approach allows for better understanding of the user's emotional state than either method alone.

### 4. Response Generation
- Emotional assessment and transcribed text are sent to Gemini LLM.
- The LLM generates an empathetic, supportive response tailored to the user's emotional state.

### 5. Text-to-Speech Conversion
- The text response is converted to speech using ElevenLabs' advanced TTS service.
- The spoken response is delivered to the user, completing the interaction loop.

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker and Docker Compose (optional, for containerized setup)

### Local Development Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/CMPT419_Project.git
   cd CMPT419_Project
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the development server:
   ```
   uvicorn app.main:app --reload
   ```

5. Configure API Keys:
   Create a `.env` file in the root directory with the following variables:
   ```
   LLM_API_KEY=your-gemini-api-key
   ELEVENLABS_API_KEY=your-elevenlabs-api-key
   ```

   You'll need to obtain API keys from:
   - [Google AI Studio](https://ai.google.dev/) for Gemini
   - [ElevenLabs](https://elevenlabs.io/) for text-to-speech

### Docker Development Setup
1. Build and start the containers:
   ```
   docker-compose up --build
   ```

## Testing the Audio Processing Workflow

We've implemented a basic audio processing workflow that:
1. Takes audio input
2. Sends it to Gemini for speech-to-text transcription
3. Logs it for emotion classification (classifier currently in development)

To test this workflow:

1. Make sure the server is running:
   ```
   uvicorn app.main:app --reload
   ```

2. Run the test script:
   ```
   python test_audio_workflow.py
   ```
   This will use a sample audio file from the test_files directory and send it to the API.

3. Alternatively, test via the Swagger UI:
   - Open http://localhost:8000/docs
   - Navigate to the `/api/v1/audio/process` endpoint
   - Upload an audio file and set the duration
   - Click "Execute"

## API Documentation
Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Running Tests
Run tests using pytest:
```
pytest app/tests/
```

## Project Structure
```
app/
├── core/           # Application core modules and configuration
├── models/         # Pydantic models for request/response schemas
│   ├── audio.py    # Models for audio processing
│   ├── visual.py   # Models for visual/facial processing
│   └── response.py # Models for LLM responses and combined data
├── routers/        # API route handlers
│   ├── audio.py    # Audio processing endpoints
│   ├── llm_to_tts.py # Combined LLM and TTS workflow
│   ├── response.py # Response generation endpoints
│   └── tts.py      # Text-to-speech endpoints
├── services/       # Business logic implementation
│   ├── audio_service.py # Audio processing service
│   ├── elevenlabs_service.py # TTS service using ElevenLabs
│   ├── llm_service.py # LLM service using Gemini
│   └── stt_service.py # Speech-to-text service
└── utils/          # Utility functions and logging
```

## License
[MIT](LICENSE)
