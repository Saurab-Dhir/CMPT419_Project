import requests
import json
import os
import time

def test_audio_workflow():
    """Test the audio processing workflow with a sample audio file."""
    
    # Define the API endpoint
    url = "http://localhost:8000/api/v1/audio/process"
    
    # Path to a sample audio file for testing
    # Using the existing test file in the repository
    test_audio_file = "test_files/test_1.wav"
    
    if not os.path.exists(test_audio_file):
        print(f"Error: Test file {test_audio_file} not found.")
        print("Please ensure the test_files directory contains test_1.wav")
        return
    
    # Prepare the form data
    files = {
        'audio': (os.path.basename(test_audio_file), open(test_audio_file, 'rb'), 'audio/wav')
    }
    data = {
        'duration': 5.0  # Duration in seconds (adjust to match your file)
    }
    
    print(f"Sending audio file {test_audio_file} to {url}")
    print("This will test the following workflow:")
    print("1. Audio file -> Gemini API for transcription")
    print("2. Audio file -> Emotion classifier placeholder (in development)")
    start_time = time.time()
    
    # Make the request
    try:
        response = requests.post(url, files=files, data=data)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            elapsed_time = time.time() - start_time
            
            print(f"\nSuccess! Processing completed in {elapsed_time:.2f} seconds")
            print("\nTranscription Result from Gemini:")
            print(f"ID: {result['id']}")
            print(f"Text: {result['transcription']['text']}")
            
            print("\nEmotion Placeholder (Classifier in development):")
            print(f"Primary emotion: {result['emotion_prediction']['emotion']}")
            
        else:
            print(f"Error: Status code {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error testing audio workflow: {str(e)}")
    
    finally:
        # Close the file
        files['audio'][1].close()

if __name__ == "__main__":
    test_audio_workflow() 