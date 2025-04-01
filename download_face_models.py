"""
Download OpenCV DNN face detection model files.
"""
import os
import urllib.request

def download_opencv_dnn_models():
    # Create models directory if it doesn't exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created {models_dir} directory")
    
    # Model URLs
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb"
    config_url = "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/opencv_face_detector.pbtxt"
    
    # Download paths
    model_path = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
    config_path = os.path.join(models_dir, "opencv_face_detector.pbtxt")
    
    # Download model file
    if not os.path.exists(model_path):
        try:
            print(f"Downloading face detection model to {model_path}...")
            urllib.request.urlretrieve(model_url, model_path)
            print("Model file downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model file: {str(e)}")
    else:
        print(f"Model file already exists at {model_path}")
    
    # Download config file
    if not os.path.exists(config_path):
        try:
            print(f"Downloading face detection config to {config_path}...")
            urllib.request.urlretrieve(config_url, config_path)
            print("Config file downloaded successfully!")
        except Exception as e:
            print(f"Error downloading config file: {str(e)}")
    else:
        print(f"Config file already exists at {config_path}")
    
    print("\nYou can now run the dependency test with: python test_facial_detection_deps.py")

if __name__ == "__main__":
    download_opencv_dnn_models() 