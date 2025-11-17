# main.py
import io
import cv2
import time
import numpy as np
import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from realtime_classifier import RealTimePoseClassifier

# Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("backend")

app = FastAPI(title="Scelet-Based Actions Backend")

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Loading model...")
classifier = RealTimePoseClassifier(model_path="models/ntu_npy", min_buffer_size=8)
logger.info("Model loaded successfully!")

@app.websocket("/ws/predict")
async def predict_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to /ws/predict")

    fps_window = []
    frame_counter = 0
    try:
        while True:
            # receive binary frame (frontend sends image/jpeg arraybuffer)
            frame_bytes = await websocket.receive_bytes()
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Received frame could not be decoded")
                continue

            frame_counter += 1
            start_time = time.time()

            # process frame -> get normalized NTU skeleton (25,3), ntu_pixels (25), mp_pixels (33)
            ntu_skeleton, ntu_pixels, mp_pixels, pose_landmarks = classifier.process_single_frame(frame)

            prediction_text = "no_pose"
            conf = 0.0
            top3 = None

            if ntu_skeleton is not None:
                # run model pipeline that uses buffer inside classifier
                try:
                    pred_idx, conf_val, top3 = classifier.predict_from_skeleton(ntu_skeleton)
                    conf = float(conf_val or 0.0)
                    if pred_idx is not None and pred_idx < len(classifier.action_names):
                        prediction_text = classifier.action_names[pred_idx]
                    elif pred_idx is not None:
                        prediction_text = f"class_{pred_idx}"
                    else:
                        prediction_text = "unknown"
                except Exception as e:
                    logger.error(f"Prediction exception: {e}")
                    prediction_text = "error"
            else:
                logger.info("No skeleton detected in frame")

            # compute FPS
            elapsed = time.time() - start_time
            fps = 1.0 / (elapsed + 1e-8)
            fps_window.append(fps)
            if len(fps_window) > 10:
                fps_window.pop(0)
            avg_fps = sum(fps_window) / len(fps_window)

            # Build message
            msg = {
                "type": "prediction",
                "prediction": prediction_text,
                "confidence": round(float(conf), 3),
                "fps": round(avg_fps, 2),
                # NTU-centered skeleton (25) and MediaPipe skeleton (33)
                "skeleton": ntu_pixels if ntu_pixels is not None else [None]*25,
                "mp_skeleton": mp_pixels if mp_pixels is not None else [None]*33,
                "top3": top3
            }

            # send if connected
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_json(msg)

            # small sleep to avoid busy-loop; frontend sends at its own rate
            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        logger.warning("Client disconnected (WebSocketDisconnect)")
    except Exception as e:
        logger.exception(f"Error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
        except:
            pass
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()
        logger.info("WebSocket handler finished")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")