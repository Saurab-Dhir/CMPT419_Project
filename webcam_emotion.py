import os
# Enable GPU by removing the line that disables it
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable GPU

import cv2
import numpy as np
from deepface import DeepFace
import time

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened. Running emotion analysis. Press 'q' to quit.")
    
    try: # Wrap the main loop in a try...finally to ensure release
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break
                
            # Create a copy of frame for display
            display_frame = frame.copy()
            
            try:
                # Analyze emotion using DeepFace (only emotion action)
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                # Check if a face was detected and analysis was successful
                if isinstance(result, list) and len(result) > 0:
                    # Get the dominant emotion and confidence for the first detected face
                    emotion = result[0]['dominant_emotion']
                    confidence = result[0]['emotion'][emotion]
                    
                    # Print the dominant emotion to the console
                    print(f"Detected Emotion: {emotion} ({confidence:.2f})           \r", end='')
                    
                    # Get face region and draw rectangle
                    region = result[0]['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    
                    # Choose color based on emotion
                    if emotion == "happy":
                        color = (0, 255, 0)  # Green
                    elif emotion == "sad":
                        color = (255, 0, 0)  # Blue
                    elif emotion == "angry":
                        color = (0, 0, 255)  # Red
                    elif emotion == "surprise":
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (255, 255, 255)  # White for neutral
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Display emotion text
                    text = f"{emotion} ({confidence:.2f})"
                    cv2.putText(display_frame, text, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                else:
                    # Print "No face detected" if no face found
                    print("Detected Emotion: No face detected                \r", end='')
                    # Display "No face detected" on frame
                    cv2.putText(display_frame, "No face detected", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                # Print a simple error message to the console
                print(f"Analysis Error: {e}                            \r", end='')
                # Display error on frame
                cv2.putText(display_frame, "Analysis Error", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Emotion Detection', display_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting.")
    finally:
        # Release resources
        print("\nReleasing webcam...")
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

if __name__ == "__main__":
    main() 