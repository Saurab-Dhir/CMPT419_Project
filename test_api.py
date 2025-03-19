import requests
import json
import os
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"

def print_separator():
    """Print a separator line."""
    print("\n" + "="*80 + "\n")

def test_health():
    """Test the health check endpoint"""
    print("\n🔍 TESTING HEALTH CHECK ENDPOINT:")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print_separator()

def test_audio_process():
    """Test the audio processing endpoint"""
    print("\n🎤 TESTING AUDIO PROCESSING ENDPOINT:")
    audio_file_path = "test_files/test_1.wav"
    
    print(f"Using audio file: {audio_file_path}")
    
    # Open the audio file in binary mode
    with open(audio_file_path, "rb") as audio_file:
        # Prepare the form data
        files = {"audio": (audio_file_path, audio_file, "audio/wav")}
        data = {"duration": 5.0}
        
        # Make the request
        print("Sending request to process audio...")
        response = requests.post(
            f"{BASE_URL}/api/v1/audio/process",
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\n📝 TRANSCRIPTION RESULT:")
            print(f"Text: '{result['transcription']['text']}'")
            print(f"Confidence: {result['transcription']['confidence']}")
            
            print("\n😊 EMOTION PREDICTION:")
            print(f"Primary Emotion: {result['emotion_prediction']['emotion']}")
            print(f"Confidence: {result['emotion_prediction']['confidence']}")
            print("Secondary Emotions:")
            for emotion, confidence in result['emotion_prediction']['secondary_emotions'].items():
                print(f"  - {emotion}: {confidence}")
                
            # Save the full response to a file for reference
            with open("logs/last_audio_response.json", "w") as f:
                json.dump(result, f, indent=2)
                print("\nFull response saved to logs/last_audio_response.json")
        else:
            print("Error:", response.text)
    
    print_separator()

def test_tts():
    """Test the TTS endpoint"""
    print("\n🔊 TESTING TEXT-TO-SPEECH ENDPOINT:")
    data = {
        "text": "Hello, this is a test of text to speech synthesis. I hope it works well!",
        "voice": None
    }
    
    print(f"Input Text: '{data['text']}'")
    print("Sending request to synthesize speech...")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/tts/synthesize",
        json=data
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Audio URL: {result['audio_url']}")
        if 'output_path' in result and result['output_path']:
            print(f"Output File: {result['output_path']}")
            
        # Save the response to a file
        with open("logs/last_tts_response.json", "w") as f:
            json.dump(result, f, indent=2)
            print("Full response saved to logs/last_tts_response.json")
    else:
        print("Error:", response.text)
    
    print_separator()

def test_response_generation():
    """Test the response generation endpoint"""
    print("\n💬 TESTING RESPONSE GENERATION ENDPOINT:")
    data = {
        "text": "I feel really anxious about my upcoming presentation next week.",
        "emotion": "anxiety",
        "session_id": "test_session_123",
        "generate_audio": True
    }
    
    print(f"User Input: '{data['text']}'")
    print(f"Emotion: {data['emotion']}")
    print("Sending request to generate response...")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/response/generate",
        json=data
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        
        print("\n🤖 GEMINI RESPONSE:")
        print(f"Response: '{result['response']['text']}'")
        print(f"Emotion Addressed: {result['response']['emotion_addressed']}")
        print(f"Response Type: {result['response']['response_type']}")
        
        if result['audio_url']:
            print(f"\n🔊 TTS AUDIO: {result['audio_url']}")
            
        # Save the response to a file
        with open("logs/last_gemini_response.json", "w") as f:
            json.dump(result, f, indent=2)
            print("Full response saved to logs/last_gemini_response.json")
    else:
        print("Error:", response.text)
    
    print_separator()

def test_audio_pipeline():
    """Test the complete audio pipeline endpoint"""
    print("\n🔄 TESTING COMPLETE AUDIO PIPELINE:")
    audio_file_path = "test_files/test_1.wav"
    
    print(f"Using audio file: {audio_file_path}")
    
    # Open the audio file in binary mode
    with open(audio_file_path, "rb") as audio_file:
        # Prepare the form data
        files = {"audio": (audio_file_path, audio_file, "audio/wav")}
        data = {
            "session_id": "test_pipeline_session",
            "generate_audio": "true"
        }
        
        # Make the request
        print("Sending request to process audio through complete pipeline...")
        response = requests.post(
            f"{BASE_URL}/api/v1/response/audio-pipeline",
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            
            print("\n🎙️ COMPLETE PIPELINE RESULTS:")
            print(f"Response ID: {result['id']}")
            print(f"Gemini Response: '{result['response']['text']}'")
            
            if result['audio_url']:
                print(f"Response Audio: {result['audio_url']}")
                
            # Save the full response to a file for reference
            with open("logs/last_pipeline_response.json", "w") as f:
                json.dump(result, f, indent=2)
                print("\nFull response saved to logs/last_pipeline_response.json")
        else:
            print("Error:", response.text)
    
    print_separator()

def setup():
    """Setup test environment."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    print("🔧 Test setup complete. Log directory created.")

if __name__ == "__main__":
    print("🚀 STARTING API TESTS...\n")
    setup()
    test_health()
    test_audio_process()
    test_tts()
    test_response_generation()
    test_audio_pipeline()  # Test the new complete pipeline
    print("✅ ALL TESTS COMPLETED!")
    print("\nCheck the 'logs' directory for detailed outputs")
    print("Check the 'output/audio' directory for generated TTS files") 