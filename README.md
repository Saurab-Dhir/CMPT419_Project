# Mirror Mirror on the Wall: Empathetic Self-talk Coach

## Project Overview
An AI-powered empathetic self-talk coach application that processes audio and visual inputs to understand emotional states and provide supportive, empathetic responses.

## Features
- Audio processing pipeline for speech capture and analysis
- Visual processing for facial expression analysis
- Emotion classification from combined inputs
- LLM-generated empathetic responses
- Text-to-Speech for audio feedback

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
├── core/           # Application core modules
├── models/         # Pydantic models for request/response schemas
├── routers/        # API route handlers
├── services/       # Business logic implementation
├── tests/          # Test cases
└── utils/          # Utility functions
```

## License
[MIT](LICENSE)
