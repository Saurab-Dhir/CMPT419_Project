import asyncio
import base64
import io
import json
import time
import uuid
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, BinaryIO
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.models.response import LLMToTTSResponse, MultiModalEmotionInput
from app.services.audio_service import audio_service
from app.services.visual_service import visual_service
from app.services.llm_service import llm_service
from app.services.stt_service import stt_service
from app.services.elevenlabs_service import elevenlabs_service
from app.services.multimodal_service import multimodal_service
from multimodal_classification.multimodal_classification_model import Evidence, MultiModalClassifier

router = APIRouter()

# Store active connections
active_connections: Dict[str, WebSocket] = {}
# Store audio buffers for each connection
audio_buffers: Dict[str, List[Any]] = {}
# Store the most recent frame for each connection
recent_frames: Dict[str, Optional[BinaryIO]] = {}

# Debug directory for saving audio from webcam
debug_dir = Path("webcam_audio_debug")
debug_dir.mkdir(exist_ok=True)
print(f"✅ Webcam audio debug directory created at: {debug_dir.absolute()}")

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time processing of webcam and microphone input.
    
    Messages should be JSON with the following format:
    {
        "type": "audio"|"video"|"session_start"|"session_end",
        "data": "base64-encoded-data", // for audio/video
        "session_id": "unique-session-id", // required for all messages
        "timestamp": 1234567890, // optional
        "duration": 1.5 // audio duration in seconds (for audio)
    }
    """
    try:
        await websocket.accept()
        active_connections[client_id] = websocket
        audio_buffers[client_id] = []
        
        print(f"WebSocket connection established for client {client_id}")
        
        session_id = None
        
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                print(f"Client {client_id} disconnected")
                break
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                session_id = message.get("session_id")
                
                if not session_id:
                    try:
                        await websocket.send_json({
                            "error": "session_id is required",
                            "status": "error"
                        })
                    except Exception as e:
                        print(f"Error sending error message: {str(e)}")
                    continue
                
                if message_type == "session_start":
                    # Initialize or reset session data
                    audio_buffers[client_id] = []
                    recent_frames[client_id] = None
                    try:
                        await websocket.send_json({
                            "status": "success",
                            "message": "Session started",
                            "session_id": session_id
                        })
                    except Exception as e:
                        print(f"Error acknowledging session start: {str(e)}")
                        if websocket.client_state == WebSocketState.DISCONNECTED:
                            break
                
                elif message_type == "session_end":
                    # Clean up session data
                    if client_id in audio_buffers:
                        audio_buffers[client_id] = []
                    if client_id in recent_frames:
                        del recent_frames[client_id]
                    try:
                        await websocket.send_json({
                            "status": "success",
                            "message": "Session ended",
                            "session_id": session_id
                        })
                    except Exception as e:
                        print(f"Error acknowledging session end: {str(e)}")
                        if websocket.client_state == WebSocketState.DISCONNECTED:
                            break
                
                elif message_type == "audio":
                    # Process audio chunk
                    audio_base64 = message.get("data")
                    if not audio_base64:
                        continue
                    
                    # Get the content type if provided
                    content_type = message.get("content_type")
                    
                    # Check if this is the final audio chunk
                    is_final = message.get("is_final", False)
                    
                    # Decode audio data
                    try:
                        audio_data = base64.b64decode(audio_base64)
                        
                        # Create a buffer with content type
                        audio_buffer = io.BytesIO(audio_data)
                        if content_type:
                            audio_buffer.content_type = content_type
                            
                        # Add to audio buffers
                        audio_buffers[client_id].append(audio_buffer)
                        
                        # Log audio data size
                        audio_size_kb = len(audio_data) / 1024
                        print(f"📊 Received audio chunk: {audio_size_kb:.2f}KB, type: {content_type}")
                            
                        # If this is the final chunk or we've accumulated enough audio
                        audio_duration = message.get("duration", 0)
                        if is_final or len(audio_buffers[client_id]) >= 10 or audio_duration >= 3.0:
                            print(f"🔊 Processing audio: {len(audio_buffers[client_id])} chunks with duration {audio_duration}s")
                            # Process if we have a video frame available
                            if client_id in recent_frames and recent_frames[client_id] is not None:
                                # Create a background task to process the data
                                # This prevents blocking the WebSocket receive loop
                                asyncio.create_task(
                                    process_multimodal_data_safely(websocket, client_id, session_id, audio_duration)
                                )
                                
                                # Acknowledge receipt of audio
                                try:
                                    await websocket.send_json({
                                        "status": "processing",
                                        "message": "Processing audio and video...",
                                        "session_id": session_id
                                    })
                                except Exception as e:
                                    print(f"Error acknowledging audio processing: {str(e)}")
                                    if websocket.client_state == WebSocketState.DISCONNECTED:
                                        break
                            else:
                                try:
                                    await websocket.send_json({
                                        "error": "No video frame available for processing",
                                        "status": "error",
                                        "session_id": session_id
                                    })
                                except Exception as e:
                                    print(f"Error sending error message: {str(e)}")
                                    if websocket.client_state == WebSocketState.DISCONNECTED:
                                        break
                        else:
                            try:
                                await websocket.send_json({
                                    "status": "buffering",
                                    "message": f"Audio received ({len(audio_buffers[client_id])} chunks) and buffering",
                                    "session_id": session_id
                                })
                            except Exception as e:
                                print(f"Error acknowledging audio buffer: {str(e)}")
                                if websocket.client_state == WebSocketState.DISCONNECTED:
                                    break
                    except Exception as e:
                        print(f"Error processing audio: {str(e)}")
                        try:
                            await websocket.send_json({
                                "error": f"Audio processing error: {str(e)}",
                                "status": "error",
                                "session_id": session_id
                            })
                        except Exception as send_err:
                            print(f"Error sending error message: {str(send_err)}")
                            if websocket.client_state == WebSocketState.DISCONNECTED:
                                break
                
                elif message_type == "video":
                    # Process video frame
                    video_base64 = message.get("data")
                    if not video_base64:
                        continue
                    
                    # Decode video data
                    video_data = base64.b64decode(video_base64)
                    video_io = io.BytesIO(video_data)
                    
                    # Store the frame as BytesIO object instead of numpy array
                    recent_frames[client_id] = video_io
                    
                    try:
                        await websocket.send_json({
                            "status": "success",
                            "message": "Video frame received",
                            "session_id": session_id
                        })
                    except Exception as e:
                        print(f"Error acknowledging video frame: {str(e)}")
                        if websocket.client_state == WebSocketState.DISCONNECTED:
                            break
                
                else:
                    try:
                        await websocket.send_json({
                            "error": f"Unknown message type: {message_type}",
                            "status": "error"
                        })
                    except Exception as e:
                        print(f"Error sending error message: {str(e)}")
                        if websocket.client_state == WebSocketState.DISCONNECTED:
                            break
            
            except json.JSONDecodeError:
                try:
                    await websocket.send_json({
                        "error": "Invalid JSON",
                        "status": "error"
                    })
                except Exception as e:
                    print(f"Error sending error message: {str(e)}")
                    if websocket.client_state == WebSocketState.DISCONNECTED:
                        break
            except Exception as e:
                try:
                    await websocket.send_json({
                        "error": str(e),
                        "status": "error"
                    })
                except Exception as send_err:
                    print(f"Error sending error message: {str(send_err)}")
                    if websocket.client_state == WebSocketState.DISCONNECTED:
                        break
    
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
    except Exception as e:
        print(f"Unexpected error in WebSocket endpoint: {str(e)}")
        traceback.print_exc()
    finally:
        # Clean up when client disconnects
        if client_id in active_connections:
            del active_connections[client_id]
        if client_id in audio_buffers:
            del audio_buffers[client_id]
        if client_id in recent_frames:
            del recent_frames[client_id]
        print(f"Cleaned up resources for client {client_id}")

async def process_multimodal_data_safely(websocket: WebSocket, client_id: str, session_id: str, audio_duration: float):
    """Wrapper around process_multimodal_data with additional error handling"""
    try:
        await process_multimodal_data(websocket, client_id, session_id, audio_duration)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected during processing for client {client_id}")
    except Exception as e:
        print(f"Error in process_multimodal_data: {str(e)}")
        traceback.print_exc()
        # Try to notify the client about the error
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.send_json({
                    "error": f"Processing error: {str(e)}",
                    "status": "error",
                    "session_id": session_id
                })
        except Exception as send_err:
            print(f"Could not send error to client: {str(send_err)}")

async def process_multimodal_data(websocket: WebSocket, client_id: str, session_id: str, audio_duration: float):
    """Process accumulated audio and the most recent video frame."""
    try:
        # Add a timeout to the WebSocket operations
        timeout = 60.0  # 60 seconds timeout
        start_time = time.time()
        
        # Better check for WebSocket connection status
        try:
            # Light ping to check connection
            await websocket.send_json({
                "status": "processing",
                "message": "Starting audio and video processing...",
                "session_id": session_id
            })
        except Exception as e:
            print(f"⚠️ WebSocket connection issue: {str(e)}")
            return
            
        # Check for empty buffer
        if not audio_buffers[client_id]:
            print("❌ No audio data in buffer")
            try:
                await websocket.send_json({
                    "error": "No audio data received",
                    "status": "error",
                    "session_id": session_id
                })
            except Exception:
                pass
            return
            
        try:    
            # Combine audio chunks with detailed logging
            print(f"🔊 Processing {len(audio_buffers[client_id])} audio chunks for {client_id}")
            
            # First attempt - try to combine preserving the content type of the first chunk
            content_type = None
            if hasattr(audio_buffers[client_id][0], 'content_type'):
                content_type = audio_buffers[client_id][0].content_type
                print(f"📊 Using content type from first chunk: {content_type}")
            
            # Extract raw bytes from all buffers
            audio_bytes = b''.join([buffer.getvalue() for buffer in audio_buffers[client_id]])
            print(f"📊 Combined audio size: {len(audio_bytes)/1024:.2f}KB")
            
            # Save a copy of the combined audio to disk for debugging
            try:
                # Determine file extension based on content type
                ext = ".webm"  # Default extension
                if content_type:
                    if "wav" in content_type:
                        ext = ".wav"
                    elif "mp3" in content_type:
                        ext = ".mp3"
                
                # Generate a unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_filename = f"webcam_{timestamp}_{session_id[:8]}{ext}"
                debug_path = debug_dir / debug_filename
                
                # Save the audio to disk
                with open(debug_path, "wb") as f:
                    f.write(audio_bytes)
                
                print(f"🔍 DEBUG: Webcam audio saved to {debug_path}")
            except Exception as e:
                print(f"⚠️ Error saving debug audio: {str(e)}")
            
            # Create a new buffer with the combined data
            audio_io = io.BytesIO(audio_bytes)
            if content_type:
                audio_io.content_type = content_type
                print(f"📊 Combined audio: {len(audio_bytes)/1024:.2f}KB, type: {content_type}")
            else:
                content_type = "audio/webm"  # Default to webm if no content type
                audio_io.content_type = content_type
                print(f"📊 Combined audio: {len(audio_bytes)/1024:.2f}KB, using default type: {content_type}")
                
            audio_io.seek(0)
            
            # Reset the audio buffer after copying the data
            audio_buffers[client_id] = []
        except Exception as e:
            print(f"❌ Error combining audio: {str(e)}")
            await websocket.send_json({
                "error": f"Error combining audio: {str(e)}",
                "status": "error",
                "session_id": session_id
            })
            return
        
        # Initialize response
        response = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "transcription": "",
            "emotions": {
                "facial": None,
                "speech": None,
                "semantic": None
            },
            "status": "processing"
        }
        
        # Get the most recent frame
        if client_id not in recent_frames or recent_frames[client_id] is None:
            print("❌ No video frame available")
            await websocket.send_json({
                "error": "No video frame available",
                "status": "error",
                "session_id": session_id
            })
            return
            
        try:
            video_io = recent_frames[client_id]
            video_data = video_io.getvalue()
            # Create a fresh BytesIO to avoid any position issues
            video_io = io.BytesIO(video_data)
            video_io.seek(0)
            
            print(f"📊 Video frame size: {len(video_data)/1024:.2f}KB")
        except Exception as e:
            print(f"❌ Error preparing video: {str(e)}")
            await websocket.send_json({
                "error": f"Error preparing video: {str(e)}",
                "status": "error",
                "session_id": session_id
            })
            return
        
        # Tell the client we're processing
        await websocket.send_json({
            "status": "processing",
            "message": "Processing audio and video...",
            "session_id": session_id
        })
        
        # Process audio (make a copy to avoid reusing the same buffer)
        audio_copy = io.BytesIO(audio_bytes)
        audio_copy.seek(0)
        if content_type:
            audio_copy.content_type = content_type
            
        try:
            audio_result = await audio_service.process_audio(audio_copy, audio_duration)
            
            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout:
                await websocket.send_json({
                    "error": "Processing timeout",
                    "status": "error",
                    "session_id": session_id
                })
                return
        except Exception as e:
            print(f"❌ Error processing audio: {str(e)}")
            await websocket.send_json({
                "error": f"Audio processing error: {str(e)}",
                "status": "error",
                "session_id": session_id
            })
            return
        
        # If we have a transcription service, transcribe the audio
        transcription = ""
        transcription_result = None
        if stt_service is not None:
            try:
                # Create another copy for STT to avoid buffer reuse issues
                stt_audio = io.BytesIO(audio_bytes)
                stt_audio.seek(0)
                if content_type:
                    stt_audio.content_type = content_type
                
                # Use the updated transcribe_audio method
                transcription_result = await stt_service.transcribe_audio(stt_audio)
                
                # Extract the transcription and emotions
                transcription = transcription_result.get("transcription", "")
                emotion_data = transcription_result.get("emotions", {})
                emotion = emotion_data.get("primary", "neutral")
                confidence = emotion_data.get("confidence", 0.5)
                
                print(f"🎙️ Transcription: '{transcription}'")
                print(f"😀 Emotion from speech: {emotion} ({confidence:.2f})")
                
                # If we have a transcription, include it in the response
                if transcription:
                    response["transcription"] = transcription
                    response["emotions"]["speech"] = {"emotion": emotion, "confidence": confidence}
                    
                # Check if we've exceeded the timeout
                if time.time() - start_time > timeout:
                    await websocket.send_json({
                        "error": "Processing timeout",
                        "status": "error",
                        "session_id": session_id
                    })
                    return
            except Exception as e:
                print(f"❌ Error in transcription: {str(e)}")
                response["error"] = f"Transcription error: {str(e)}"
        
        # Process the video (create a copy)
        try:
            # Make a copy of the video buffer
            video_copy = io.BytesIO(video_io.getvalue())
            video_copy.seek(0)
            
            # Process video
            visual_result = await visual_service.process_image(video_copy, "jpg")
            
            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout:
                await websocket.send_json({
                    "error": "Processing timeout",
                    "status": "error",
                    "session_id": session_id
                })
                return
        except Exception as e:
            print(f"❌ Error processing video: {str(e)}")
            await websocket.send_json({
                "error": f"Video processing error: {str(e)}",
                "status": "error",
                "session_id": session_id
            })
            return
        
        # Extract semantic emotion from Gemini
        semantic_emotion = "neutral"
        if hasattr(audio_result, 'transcription') and hasattr(audio_result.transcription, 'raw_response'):
            try:
                raw_response = audio_result.transcription.raw_response
                if raw_response:
                    print(f"DEBUG - Raw response: {raw_response[:100]}")
                    
                    # Clean up the raw response if it's a string
                    if isinstance(raw_response, str):
                        # Remove any 'single quotes' around the JSON string that might be there
                        # from the string representation of the dictionary
                        if raw_response.startswith("'") and raw_response.endswith("'"):
                            raw_response = raw_response[1:-1]
                        
                        # Handle markdown code blocks
                        if raw_response.startswith("```json") and raw_response.endswith("```"):
                            raw_response = raw_response[7:-3].strip()
                        elif raw_response.startswith("```") and raw_response.endswith("```"):
                            raw_response = raw_response[3:-3].strip()
                        
                        # Handle string representation of dictionaries
                        if raw_response.startswith("{'") or raw_response.startswith('{"'):
                            try:
                                # First try to parse as JSON
                                json_response = json.loads(raw_response)
                                semantic_emotion = json_response.get("emotion", "neutral")
                                print(f"✅ Successfully parsed emotion as JSON: {semantic_emotion}")
                            except json.JSONDecodeError:
                                try:
                                    # If that fails, try to evaluate as Python dict (safer than eval)
                                    import ast
                                    dict_response = ast.literal_eval(raw_response)
                                    semantic_emotion = dict_response.get("emotion", "neutral")
                                    print(f"✅ Successfully parsed emotion using ast: {semantic_emotion}")
                                except (SyntaxError, ValueError) as e:
                                    # If all parsing fails, try regex
                                    import re
                                    emotion_match = re.search(r'"emotion":\s*"([^"]+)"', raw_response)
                                    if emotion_match:
                                        semantic_emotion = emotion_match.group(1)
                                        print(f"✅ Successfully extracted emotion using regex: {semantic_emotion}")
                                    else:
                                        print(f"❌ Could not extract emotion using any method from: {raw_response[:50]}...")
                    # Handle direct dictionary/object
                    elif isinstance(raw_response, dict):
                        semantic_emotion = raw_response.get("emotion", "neutral")
                        print(f"✅ Got emotion directly from dict: {semantic_emotion}")
                    else:
                        print(f"⚠️ Unexpected raw_response type: {type(raw_response)}")
            except Exception as e:
                print(f"❌ Error accessing semantic emotion: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
        else:
            # Try to get emotion directly from audio_result if the new simplified format is used
            if hasattr(audio_result, 'emotion_prediction') and audio_result.emotion_prediction:
                semantic_emotion = audio_result.emotion_prediction.emotion
                print(f"✅ Got semantic emotion directly from emotion_prediction: {semantic_emotion}")
        
        # Extract tonal emotion
        tonal_emotion = "neutral"
        if hasattr(audio_result, 'emotion_prediction') and audio_result.emotion_prediction:
            tonal_emotion = audio_result.emotion_prediction.emotion
        
        # Extract facial emotion and confidence
        facial_emotion = "neutral"
        facial_confidence = 0
        if visual_result and hasattr(visual_result, 'emotion_prediction') and visual_result.emotion_prediction:
            facial_emotion = visual_result.emotion_prediction.emotion
            facial_confidence = getattr(visual_result.emotion_prediction, 'confidence', 0)
            print(f"🧑 Facial emotion detected: {facial_emotion} (confidence: {facial_confidence:.2f})")
        
        # Check if we have detected a face
        face_detected = getattr(visual_result, 'face_detected', False)
        if not face_detected:
            await websocket.send_json({
                "status": "warning",
                "message": "No face detected in video frame",
                "session_id": session_id
            })
        
        # Log all emotions for debugging
        print(f"🔍 EMOTION SUMMARY - Semantic: {semantic_emotion}, Tonal: {tonal_emotion}, Facial: {facial_emotion}")

        TONE_CLASSIFIER_RELIABILITY = 0.55
        FACE_CLASSIFIER_RELIABILITY = 0.87
        SEMANTIC_CLASSIFIER_RELIABILITY = 0.85
        tone = Evidence(
            emotion=tonal_emotion, 
            confidence=audio_result.emotion_prediction.confidence, 
            reliability=TONE_CLASSIFIER_RELIABILITY)
        face = Evidence(
            emotion=facial_emotion, 
            confidence=visual_result.emotion_prediction.confidence, 
            reliability=FACE_CLASSIFIER_RELIABILITY)
        semantics = Evidence(
            emotion=semantic_emotion, 
            confidence=0.8, 
            reliability=SEMANTIC_CLASSIFIER_RELIABILITY)
        
        multimodal_model = MultiModalClassifier()
        combined_prediction = multimodal_model.predict(tone, face, semantics)
        print(f"\n===== MULTIMODAL LATE FUSION MODEL [{session_id}] =====")
        print("FUSED PREDICTIONS:")
        multimodal_model.print_mass_function(combined_prediction, "tone, facial expression, semantics")
        print("========================================\n")
        
        # Create multimodal input
        multimodal_input = MultiModalEmotionInput(
            user_speech=transcription,
            semantic_emotion=semantic_emotion,
            tonal_emotion=tonal_emotion,
            facial_emotion=facial_emotion,
            fused_emotion=combined_prediction,
            session_id=session_id
        )
        
        # Generate response
        try:
            response_text, response_id, model_emotion = await llm_service.process_multimodal_input(multimodal_input)
            
            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout:
                await websocket.send_json({
                    "error": "Processing timeout",
                    "status": "error",
                    "session_id": session_id
                })
                return
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            await websocket.send_json({
                "error": f"Response generation error: {str(e)}",
                "status": "error",
                "session_id": session_id
            })
            return
        
        # Synthesize speech
        audio_url = None
        try:
            audio_url, _ = await elevenlabs_service.synthesize_speech(
                text=response_text,
                response_id=response_id
            )
        except Exception as e:
            print(f"❌ Error synthesizing speech: {str(e)}")
            # Continue even if speech synthesis fails
        
        # Build a more comprehensive response with debugging info
        response = {
            "status": "complete",
            "transcription": transcription,
            "semantic_emotion": semantic_emotion,
            "tonal_emotion": tonal_emotion,
            "facial_emotion": facial_emotion,
            "response_text": response_text,
            "audio_url": audio_url,
            "session_id": session_id,
            "response_id": response_id,
            "face_detected": face_detected,
            "audio_duration": audio_duration,
            "debug": {
                "empty_transcription": not transcription or transcription == "(No speech detected)",
                "processing_time": time.time() - start_time
            }
        }
        
        # Send the result back to the client
        await websocket.send_json(response)
        
        # Send an updated emotion analysis event for the 3D animation viewer with the model_emotion
        await websocket.send_json({
            "type": "emotion_analysis",
            "emotion": model_emotion,  # Use the model's emotion response instead of facial_emotion
            "confidence": 1.0,
            "timestamp": time.time(),
            "session_id": session_id,
            "metadata": {
                "semantic_emotion": semantic_emotion,
                "tonal_emotion": tonal_emotion,
                "facial_emotion": facial_emotion,
                "model_emotion": model_emotion
            }
        })
        
        # Log the emotion we're sending
        print(f"🎭 Sending emotion to 3D model: {model_emotion} (from LLM response)")
        print(f"🎭 Detected emotions - Semantic: {semantic_emotion}, Tonal: {tonal_emotion}, Facial: {facial_emotion}")
        
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"❌ Error in processing multimodal data: {error_msg}")
        print(f"Stack trace: {stack_trace}")
        
        await websocket.send_json({
            "error": error_msg,
            "status": "error",
            "session_id": session_id,
            "detail": stack_trace[:500] if stack_trace else None
        }) 