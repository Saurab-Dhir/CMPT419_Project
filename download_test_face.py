"""
Download a sample face image for testing facial detection dependencies.
"""
import os
import urllib.request

def download_test_face():
    # Create test_files directory if it doesn't exist
    if not os.path.exists("test_files"):
        os.makedirs("test_files")
        print("Created test_files directory")
    
    # Define the URL for a sample face image
    image_url = "https://github.com/opencv/opencv/raw/master/samples/data/lena.jpg"
    output_path = "test_files/test_face.jpg"
    
    # Download the image if it doesn't exist
    if not os.path.exists(output_path):
        try:
            print(f"Downloading test face image to {output_path}...")
            urllib.request.urlretrieve(image_url, output_path)
            print("Download successful!")
        except Exception as e:
            print(f"Error downloading test image: {str(e)}")
            print("Please download a face image manually and place it at test_files/test_face.jpg")
    else:
        print(f"Test image already exists at {output_path}")
    
    print("\nYou can now run the dependency test with: python test_facial_detection_deps.py")

if __name__ == "__main__":
    download_test_face() 