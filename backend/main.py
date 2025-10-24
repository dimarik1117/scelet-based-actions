from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import json
import traceback
import numpy as np
import cv2
import os

# Импортируем только RealTimePoseClassifier (функцию predict по base64 реализуем здесь)
from ml_realtime_inference import RealTimePoseClassifier

app = FastAPI(title="Skeleton-based Action Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Инициализация модели...")
try:
    global_classifier = RealTimePoseClassifier()
    print("Модель загружена!")
except Exception as e:
    print("Ошибка при инициализации модели:", e)
    raise

class ImageRequest(BaseModel):
    image: str  # base64 строки

@app.get("/")
def home():
    return {"message": "Skeleton-based Action Recognition API is running"}

# POST для теста через Postman (использует глобальный classifier)
@app.post("/predict-image")
async def predict_image(request: ImageRequest):
    try:
        result = predict_single_image_base64_from_classifier(request.image, global_classifier)
        return result
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def predict_single_image_base64_from_classifier(base64_str: str, classifier: RealTimePoseClassifier):
    """
    Предсказание для одного изображения (base64). Используем уже загруженный classifier.
    """
    try:
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",", 1)[1]

        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Не удалось декодировать изображение"}

        results = classifier.converter.process_frame(image)
        if not results or not getattr(results, "pose_landmarks", None):
            return {"action": None, "confidence": 0.0, "message": "Поза не найдена"}

        ntu_skeleton = classifier.converter.mediapipe_to_ntu_skeleton(results.pose_landmarks)

        for _ in range(10):
            classifier.pose_buffer.append(ntu_skeleton)

        features = classifier.extract_realtime_features(ntu_skeleton)
        if features is None or len(features) == 0:
            return {"action": None, "confidence": 0.0, "message": "Недостаточно данных"}

        pred, conf = classifier.predict_action(features)
        if pred is None:
            return {"action": None, "confidence": 0.0, "message": "Не удалось классифицировать"}

        action_name = classifier.action_names[pred] if pred < len(classifier.action_names) else str(pred)
        return {"action": action_name, "confidence": float(conf)}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    print("Ожидание WebSocket соединения...")
    await websocket.accept()
    print("WebSocket клиент подключен")
    classifier = global_classifier
    try:
        while True:
            # Получаем бинарные данные (JPEG кадр)
            frame_bytes = await websocket.receive_bytes()
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid frame data"
                }))
                continue

            results = classifier.converter.process_frame(frame)
            if not results or not getattr(results, "pose_landmarks", None):
                classifier.pose_buffer.clear()
                payload = {
                    "type": "prediction",
                    "prediction": None,
                    "confidence": 0.0
                }
                print("Отправка предсказания:", payload)
                await websocket.send_text(json.dumps(payload))
                continue

            ntu_skeleton = classifier.converter.mediapipe_to_ntu_skeleton(results.pose_landmarks)
            features = classifier.extract_realtime_features(ntu_skeleton)
            if features is None:
                payload = {
                    "type": "prediction",
                    "prediction": None,
                    "confidence": 0.0
                }
                print("Отправка предсказания:", payload)
                await websocket.send_text(json.dumps(payload))
                continue

            pred, conf = classifier.predict_action(features)
            pred, conf = classifier.smooth_prediction(pred, conf)
            if pred is not None:
                action_name = classifier.action_names[pred] if pred < len(classifier.action_names) else str(pred)
                payload = {
                    "type": "prediction",
                    "prediction": action_name,
                    "confidence": float(conf)
                }
            else:
                payload = {
                    "type": "prediction",
                    "prediction": None,
                    "confidence": 0.0
                }

            print("Отправка предсказания:", payload)
            await websocket.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        print("WebSocket клиент отключен")
    except Exception as e:
        print("Ошибка в WebSocket:", e)
        print(traceback.format_exc())
        try:
            await websocket.close()
        except Exception:
            pass