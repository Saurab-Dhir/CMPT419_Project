from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from app.routers.api import api_router
from app.core.config import settings
import cv2
import numpy as np
import argparse
import logging
from datetime import datetime
from app.services.deepface_service import DeepFaceService
from app.models.visual import FacialLandmarks
from app.routers.realtime import websocket_endpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Empathetic Self-talk Coach API",
    description="API for the 'Mirror mirror on the wall!' project",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Add WebSocket endpoint directly to the app as well for easier access
app.add_websocket_route("/ws/{client_id}", websocket_endpoint)

# Create static directory for audio files if it doesn't exist
os.makedirs("static/audio", exist_ok=True)

# Mount the static directory
app.mount("/audio", StaticFiles(directory="static/audio"), name="audio")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/animations", StaticFiles(directory="animations"), name="animations")

@app.get("/")
async def root():
    """Redirect to webcam interface."""
    return {"message": "Welcome to the Empathetic Self-talk Coach API", "documentation": "/docs"}

@app.get("/webcam")
async def webcam_interface():
    """Serve the webcam interface."""
    from fastapi.responses import FileResponse
    return FileResponse("static/webcam.html")

@app.get("/3d-conversation")
async def conversation_interface():
    """Serve the 3D conversation interface with integrated webcam."""
    from fastapi.responses import FileResponse
    return FileResponse("static/3d-conversation.html")

@app.get("/websocket-test")
async def websocket_test():
    """Serve the WebSocket test interface."""
    from fastapi.responses import FileResponse
    return FileResponse("static/websocket-test.html")

@app.get("/3d-emotions")
async def animation_interface():
    """Serve the 3D animation viewer interface."""
    from fastapi.responses import FileResponse
    return FileResponse("static/3d_animations.html")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "ok", "message": "API is running"}

def draw_landmarks(image, landmarks):
    """
    Draw facial landmarks on the image.
    
    Args:
        image: The image to draw on
        landmarks: FacialLandmarks object
    
    Returns:
        Image with landmarks drawn
    """
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    # Draw eyes
    for eye_point in landmarks.eye_positions:
        x, y = int(eye_point[0] * w), int(eye_point[1] * h)
        cv2.circle(vis_image, (x, y), 2, (0, 255, 255), -1)
    
    # Draw mouth
    for mouth_point in landmarks.mouth_position:
        x, y = int(mouth_point[0] * w), int(mouth_point[1] * h)
        cv2.circle(vis_image, (x, y), 2, (0, 0, 255), -1)
    
    # Draw eyebrows
    for eyebrow_point in landmarks.eyebrow_positions:
        x, y = int(eyebrow_point[0] * w), int(eyebrow_point[1] * h)
        cv2.circle(vis_image, (x, y), 2, (255, 0, 0), -1)
    
    # Draw nose
    nose_x, nose_y = int(landmarks.nose_position[0] * w), int(landmarks.nose_position[1] * h)
    cv2.circle(vis_image, (nose_x, nose_y), 3, (0, 255, 0), -1)
    
    # Draw face contour
    for i, point in enumerate(landmarks.face_contour):
        x, y = int(point[0] * w), int(point[1] * h)
        cv2.circle(vis_image, (x, y), 2, (255, 255, 0), -1)
        if i > 0:
            prev_x, prev_y = int(landmarks.face_contour[i-1][0] * w), int(landmarks.face_contour[i-1][1] * h)
            cv2.line(vis_image, (prev_x, prev_y), (x, y), (255, 255, 0), 1)
    
    return vis_image

def draw_analysis_results(image, face_detection, analysis, metrics=None):
    """
    Draw analysis results on the image.
    
    Args:
        image: The image to draw on
        face_detection: Face detection results
        analysis: Face analysis results
        metrics: Optional metrics from face landmarks
    
    Returns:
        Image with results drawn
    """
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    # Draw face bounding box if detected
    if face_detection.get("detected", False):
        x, y, width, height = face_detection.get("box", (0, 0, 0, 0))
        cv2.rectangle(vis_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    # Create background for text
    overlay = vis_image.copy()
    cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
    vis_image = cv2.addWeighted(overlay, 0.6, vis_image, 0.4, 0)
    
    # Add analysis text
    y_offset = 30
    cv2.putText(vis_image, f"Emotion: {analysis.get('emotion', 'unknown')}", (15, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 25
    
    cv2.putText(vis_image, f"Confidence: {analysis.get('emotion_confidence', 0):.2f}", (15, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 25
    
    cv2.putText(vis_image, f"Age: {analysis.get('age', 0)}", (15, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 25
    
    cv2.putText(vis_image, f"Gender: {analysis.get('gender', 'unknown')}", (15, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 25
    
    # Add metrics if available
    if metrics:
        cv2.putText(vis_image, f"Eye openness: {metrics.get('eye_openness', 0):.2f}", (15, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        cv2.putText(vis_image, f"Mouth openness: {metrics.get('mouth_openness', 0):.2f}", (15, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        cv2.putText(vis_image, f"Eyebrow raise: {metrics.get('eyebrow_raise', 0):.2f}", (15, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return vis_image

def process_image(image_path, service):
    """
    Process a single image file.
    
    Args:
        image_path: Path to the image file
        service: DeepFaceService instance
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return
    
    # Run facial analysis pipeline
    face_detection = service.detect_face(image)
    analysis = service.analyze_face(image, face_detection)
    landmarks = service.extract_landmarks(image, face_detection)
    metrics = service.calculate_metrics(landmarks)
    
    # Visualize results
    vis_image = draw_analysis_results(image, face_detection, analysis, metrics)
    vis_image = draw_landmarks(vis_image, landmarks)
    
    # Create output filename
    filename = os.path.basename(image_path)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{timestamp}_{filename}")
    
    # Save result
    cv2.imwrite(output_path, vis_image)
    logger.info(f"Processed image saved to: {output_path}")
    
    # Display results
    print("\n===== Facial Analysis Results =====")
    print(f"Face detected: {face_detection.get('detected', False)}")
    if face_detection.get("detected", False):
        print(f"Face quality: {face_detection.get('quality', 0):.2f}")
        print(f"Emotion: {analysis.get('emotion', 'unknown')}")
        print(f"Confidence: {analysis.get('emotion_confidence', 0):.2f}")
        print(f"Age: {analysis.get('age', 0)}")
        print(f"Gender: {analysis.get('gender', 'unknown')}")
        print("\nDerived Metrics:")
        print(f"Eye openness: {metrics.get('eye_openness', 0):.2f}")
        print(f"Mouth openness: {metrics.get('mouth_openness', 0):.2f}")
        print(f"Eyebrow raise: {metrics.get('eyebrow_raise', 0):.2f}")
        head_pose = metrics.get("head_pose", {})
        print(f"Head pose (pitch, yaw, roll): {head_pose.get('pitch', 0):.1f}°, {head_pose.get('yaw', 0):.1f}°, {head_pose.get('roll', 0):.1f}°")
    else:
        print(f"Message: {face_detection.get('message', 'Unknown error')}")

def process_video(video_path, service):
    """
    Process a video file or webcam.
    
    Args:
        video_path: Path to the video file or 0 for webcam
        service: DeepFaceService instance
    """
    # Open video source
    if video_path == "0":
        cap = cv2.VideoCapture(0)
        logger.info("Using webcam as video source")
    else:
        cap = cv2.VideoCapture(video_path)
        logger.info(f"Processing video: {video_path}")
    
    # Check if video opened successfully
    if not cap.isOpened():
        logger.error("Error opening video source")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer if not using webcam
    output_video = None
    if video_path != "0":
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(video_path)
        output_path = os.path.join(output_dir, f"{timestamp}_{filename}")
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Output video will be saved to: {output_path}")
    
    # Process video frames
    frame_count = 0
    processing_every_n_frames = 5  # Process every 5 frames to improve performance
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process only every n frames
        if frame_count % processing_every_n_frames == 0:
            # Run facial analysis pipeline
            face_detection = service.detect_face(frame)
            analysis = service.analyze_face(frame, face_detection)
            landmarks = service.extract_landmarks(frame, face_detection)
            metrics = service.calculate_metrics(landmarks)
            
            # Visualize results
            vis_frame = draw_analysis_results(frame, face_detection, analysis, metrics)
            
            if face_detection.get("detected", False):
                vis_frame = draw_landmarks(vis_frame, landmarks)
        else:
            # For skipped frames, use the frame without processing
            vis_frame = frame
        
        # Add frame number
        cv2.putText(vis_frame, f"Frame: {frame_count}", (15, height - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the resulting frame
        cv2.imshow('Facial Analysis', vis_frame)
        
        # Write to output video if not using webcam
        if output_video is not None:
            output_video.write(vis_frame)
        
        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if output_video is not None:
        output_video.release()
    cv2.destroyAllWindows()
    logger.info("Video processing complete")

def main():
    """Main function to run the facial analysis demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Facial Analysis Demo')
    parser.add_argument('--image', type=str, help='Path to input image file')
    parser.add_argument('--video', type=str, help='Path to input video file, or "0" for webcam')
    
    args = parser.parse_args()
    
    # Check if an input source is provided
    if not args.image and not args.video:
        parser.error("Please provide either an image file with --image or a video file with --video")
    
    # Initialize the facial analysis service
    logger.info("Initializing DeepFaceService...")
    service = DeepFaceService()
    
    # Process input
    if args.image:
        process_image(args.image, service)
    elif args.video:
        process_video(args.video, service)

if __name__ == "__main__":
    main() 