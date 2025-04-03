#!/usr/bin/env python3
"""
Script to test the visual processing capabilities of the application.
This script allows manual testing of facial detection, emotion recognition,
and feature extraction on image files.
"""

import argparse
import os
import sys
import time
import json
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import application components
try:
    from app.services.visual_service import visual_service
    from app.services.deepface_service import DeepFaceService
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Current sys.path:", sys.path)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image():
    """Create a simple test image with a face-like shape for testing."""
    # Create a 200x200 white image
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    # Draw a simple face-like shape
    # Face outline (circle)
    cv2.circle(img, (100, 100), 80, (200, 200, 200), -1)
    
    # Eyes (circles)
    cv2.circle(img, (70, 80), 15, (0, 0, 0), -1)
    cv2.circle(img, (130, 80), 15, (0, 0, 0), -1)
    
    # Mouth (ellipse)
    cv2.ellipse(img, (100, 130), (30, 15), 0, 0, 180, (0, 0, 0), -1)
    
    return img

def visualize_results(image, result):
    """
    Visualize the facial detection and analysis results on the image.
    
    Args:
        image: Original image as numpy array
        result: Analysis result from the visual service
    """
    # Create a copy of the image for drawing
    img_copy = image.copy()
    
    # Add text for the emotion
    emotion = result["emotion_prediction"]["emotion"]
    confidence = result["emotion_prediction"]["confidence"]
    
    # Draw text with emotion and confidence
    text = f"{emotion}: {confidence:.2f}"
    cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 255), 2)
    
    # If there are facial landmarks, draw them
    if result["face_detected"]:
        try:
            # Extract landmarks
            landmarks = result["features"]["landmarks"]
            
            # Draw eyes
            for eye_pos in landmarks["eye_positions"]:
                cv2.circle(img_copy, (int(eye_pos[0] * image.shape[1]), int(eye_pos[1] * image.shape[0])), 
                          5, (0, 255, 0), -1)
            
            # Draw mouth
            for mouth_pos in landmarks["mouth_position"]:
                cv2.circle(img_copy, (int(mouth_pos[0] * image.shape[1]), int(mouth_pos[1] * image.shape[0])), 
                          5, (0, 0, 255), -1)
            
            # Draw eyebrows
            for eyebrow_pos in landmarks["eyebrow_positions"]:
                cv2.circle(img_copy, (int(eyebrow_pos[0] * image.shape[1]), int(eyebrow_pos[1] * image.shape[0])), 
                          5, (255, 0, 0), -1)
            
            # Draw nose
            nose_pos = landmarks["nose_position"]
            cv2.circle(img_copy, (int(nose_pos[0] * image.shape[1]), int(nose_pos[1] * image.shape[0])), 
                      5, (255, 255, 0), -1)
            
            # Draw face contour
            face_contour = landmarks["face_contour"]
            points = []
            for pos in face_contour:
                points.append([int(pos[0] * image.shape[1]), int(pos[1] * image.shape[0])])
            
            points = np.array(points, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(img_copy, [points], True, (0, 255, 255), 2)
            
            # Add metrics text
            eye_openness = result["features"]["eye_openness"]
            mouth_openness = result["features"]["mouth_openness"]
            eyebrow_raise = result["features"]["eyebrow_raise"]
            
            cv2.putText(img_copy, f"Eye: {eye_openness:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img_copy, f"Mouth: {mouth_openness:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img_copy, f"Eyebrow: {eyebrow_raise:.2f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        except Exception as e:
            logger.error(f"Error visualizing landmarks: {str(e)}")
            cv2.putText(img_copy, "Error visualizing landmarks", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # If no face detected, add text
        cv2.putText(img_copy, "No face detected", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return img_copy

async def process_image(image_path, output_dir=None, show_result=True, save_result=False):
    """
    Process an image file and display/save the results.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save results (optional)
        show_result: Whether to display the result
        save_result: Whether to save the result
    """
    try:
        # Check if the image path exists
        if not os.path.exists(image_path):
            if image_path.lower() == "test":
                # Generate a test image
                logger.info("Generating test image")
                img = create_test_image()
                path_base = "test_image"
            else:
                logger.error(f"Image file not found: {image_path}")
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error(f"Absolute path attempted: {os.path.abspath(image_path)}")
                return
        else:
            # Read the image
            logger.info(f"Reading image from: {image_path}")
            img = cv2.imread(image_path)
            
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                logger.error(f"File exists but could not be loaded - Check format and permissions")
                # Try to get file info
                try:
                    file_size = os.path.getsize(image_path)
                    logger.error(f"File size: {file_size} bytes")
                except Exception as e:
                    logger.error(f"Error getting file info: {str(e)}")
                return
                
            path_base = os.path.splitext(os.path.basename(image_path))[0]
        
        # Start timing
        start_time = time.time()
        
        # Open the image file
        with open(image_path, "rb") if image_path.lower() != "test" else open(f"test_face_{int(time.time())}.jpg", "wb+") as f:
            # If using the test image, save it first
            if image_path.lower() == "test":
                cv2.imwrite(f.name, img)
                f.seek(0)
                path_base = os.path.splitext(os.path.basename(f.name))[0]
            
            # Get the file extension
            file_extension = os.path.splitext(f.name)[1][1:].lower()
            
            # Process the image
            logger.info("Processing image with visual service")
            result = await visual_service.process_image(f, file_extension)
            
            # Convert to dictionary
            result_dict = result.dict()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            # Add processing time to result
            result_dict["processing_time"] = processing_time
            
            # Visualize results
            if show_result or save_result:
                # Create visualization
                viz_img = visualize_results(img, result_dict)
                
                # Show the result
                if show_result:
                    plt.figure(figsize=(12, 8))
                    plt.subplot(1, 2, 1)
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title("Original Image")
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
                    plt.title("Analysis Result")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                
                # Save the results
                if save_result:
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save visualization image
                        out_img_path = os.path.join(output_dir, f"{path_base}_analysis.jpg")
                        cv2.imwrite(out_img_path, viz_img)
                        logger.info(f"Saved visualization to: {out_img_path}")
                        
                        # Save result JSON
                        out_json_path = os.path.join(output_dir, f"{path_base}_result.json")
                        with open(out_json_path, 'w') as json_file:
                            json.dump(result_dict, json_file, indent=2, default=str)
                        logger.info(f"Saved result data to: {out_json_path}")
                    else:
                        logger.warning("No output directory specified for saving results")
            
            # Print emotion summary
            emotion = result_dict["emotion_prediction"]["emotion"]
            confidence = result_dict["emotion_prediction"]["confidence"]
            face_detected = result_dict["face_detected"]
            
            logger.info(f"Face detected: {face_detected}")
            if face_detected:
                logger.info(f"Primary emotion: {emotion} (confidence: {confidence:.2f})")
                logger.info(f"Secondary emotions: {result_dict['emotion_prediction']['secondary_emotions']}")
                logger.info(f"Eye openness: {result_dict['features']['eye_openness']:.2f}")
                logger.info(f"Mouth openness: {result_dict['features']['mouth_openness']:.2f}")
                logger.info(f"Eyebrow raise: {result_dict['features']['eyebrow_raise']:.2f}")
            
            return result_dict
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test visual processing on images")
    parser.add_argument("image_path", help="Path to the image file or 'test' for a generated test image")
    parser.add_argument("--output", "-o", help="Directory to save results")
    parser.add_argument("--no-display", action="store_true", help="Do not display results")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--batch", action="store_true", help="Process all images in the directory")
    
    args = parser.parse_args()
    
    # Import asyncio here to avoid issues in Jupyter notebooks
    import asyncio
    
    if args.batch and os.path.isdir(args.image_path):
        # Process all images in the directory
        directory = args.image_path
        image_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            logger.error(f"No image files found in directory: {directory}")
            return
        
        logger.info(f"Processing {len(image_files)} images in batch mode")
        
        # Process each image
        for image_file in image_files:
            logger.info(f"Processing: {os.path.basename(image_file)}")
            asyncio.run(process_image(
                image_file, 
                output_dir=args.output, 
                show_result=not args.no_display,
                save_result=args.save
            ))
    else:
        # Process a single image
        asyncio.run(process_image(
            args.image_path, 
            output_dir=args.output, 
            show_result=not args.no_display,
            save_result=args.save
        ))

if __name__ == "__main__":
    main() 