import asyncio
import os
import json
from app.services.tts_service import tts_service

async def test_elevenlabs_tts():
    """Test ElevenLabs TTS integration directly"""
    print("\n🔊 TESTING ELEVENLABS TEXT-TO-SPEECH:")
    
    # Use a simple test text
    test_text = "Hello, this is a test of ElevenLabs text to speech synthesis. I hope it works well!"
    
    print(f"Input Text: '{test_text}'")
    print("Sending request to ElevenLabs...")
    
    # Call the TTS service directly
    audio_url, output_path = await tts_service.synthesize(
        text=test_text,
        response_id="elevenlabs_test"
    )
    
    print(f"Audio URL: {audio_url}")
    print(f"Output File: {output_path}")
    
    # Check if the file exists and has actual content
    if output_path and os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Output file size: {file_size} bytes")
        
        if file_size > 1000:  # If file is reasonably sized (not just a mock)
            print("✅ SUCCESS: ElevenLabs TTS generated a real audio file")
        else:
            print("⚠️ WARNING: File exists but is very small, might be a mock")
    else:
        print("❌ ERROR: Output file was not created")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_elevenlabs_tts()) 