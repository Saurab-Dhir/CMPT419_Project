import asyncio
import os
import sys
from pathlib import Path
import traceback

# Add the current directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("dotenv not installed, using environment variables as is")

from app.services.stt_service import stt_service

async def test_stt_service():
    # Check if API key is available
    if not os.getenv("LLM_API_KEY"):
        print("⚠️ LLM_API_KEY not set in environment variables. Please set it in .env file.")
        return
    
    print("🔍 Testing STT service with a basic audio file...")
    
    # Try to find a test audio file
    test_files = [
        "test_audio.wav",  # Try a direct test file
        "static/test_audio.wav",  # Check in static dir
        "tests/test_audio.wav",  # Check in tests dir
    ]
    
    test_file_path = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_file_path = file_path
            break
    
    if not test_file_path:
        print("Creating a simple test audio file (silent)...")
        # Create a simple WAV file with minimal content for testing
        with open("test_audio.wav", "wb") as f:
            # Simple WAV header for a 1-second silent file
            f.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x11+\x00\x00"\x56\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
        test_file_path = "test_audio.wav"
    
    print(f"📂 Using test audio file: {test_file_path}")
    
    # Open the audio file for testing
    with open(test_file_path, "rb") as f:
        audio_data = f.read()
    
    try:
        # Call the STT service
        print("📞 Calling transcribe_audio method...")
        result = await stt_service.transcribe_audio(audio_data)
        
        # Display the result
        print("\n📊 Result:")
        print(f"Transcription: {result.get('transcription', '')}")
        print(f"Emotions: {result.get('emotions', {})}")
        print(f"Error (if any): {result.get('error', 'None')}")
        
        return result
    except Exception as e:
        print(f"❌ Error during STT service test: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting STT service test...")
    result = asyncio.run(test_stt_service())
    print("\n✅ Test completed.") 