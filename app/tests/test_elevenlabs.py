import os
import requests
import json

def test_elevenlabs_tts():
    """Test ElevenLabs TTS integration directly"""
    print("\n🔊 TESTING ELEVENLABS TEXT-TO-SPEECH:")
    
    # Prompt for the API key if not provided
    api_key = input("Enter your ElevenLabs API key: ")
    if not api_key:
        print("No API key provided. Exiting...")
        return
    
    # ElevenLabs API endpoint
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default to "Rachel" voice
    base_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    # Test text
    test_text = "Hello, this is a test of ElevenLabs text to speech synthesis. I hope it works well!"
    
    # Headers
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Data
    data = {
        "text": test_text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    print(f"Input Text: '{test_text}'")
    print(f"Making request to ElevenLabs: {base_url}")
    
    try:
        # Make the API request
        response = requests.post(
            base_url,
            json=data,
            headers=headers
        )
        
        # Check response
        response.raise_for_status()
        print(f"✅ ElevenLabs API response received: {response.status_code}")
        
        # Save the audio file
        output_dir = "output/audio"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = f"{output_dir}/elevenlabs_test.mp3"
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        # Check file size
        file_size = os.path.getsize(output_path)
        print(f"Audio file saved to: {output_path}")
        print(f"Output file size: {file_size} bytes")
        
        if file_size > 1000:
            print("✅ SUCCESS: ElevenLabs TTS generated a real audio file")
        else:
            print("⚠️ WARNING: File exists but is very small, might be a mock")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error in request: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text[:200]}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_elevenlabs_tts() 