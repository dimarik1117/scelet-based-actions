from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import io
from PIL import Image
import traceback

# Импорт твоей модели
from ml_realtime_inference import RealTimePoseClassifier

# Создаём приложение
app = FastAPI(title="Skeleton-based Action Recognition API")

# Разрешаем фронтенду обращаться к API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # потом можно указать конкретно localhost:5173, если нужно
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализируем модель
print("Инициализация модели...")
classifier = RealTimePoseClassifier()
print("Модель загружена!")


# Модель входных данных
class ImageRequest(BaseModel):
    image: str  # base64-строка


@app.post("/predict")
async def predict(request: ImageRequest):
    try:
        # Декодируем base64 -> изображение
        image_data = base64.b64decode(request.image.split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Пропускаем через MediaPipe и модель
        results = classifier.converter.process_frame(frame)

        if not results.pose_landmarks:
            return {"action": None, "confidence": 0.0}

        # Конвертация позы и предсказание
        ntu_skeleton = classifier.converter.mediapipe_to_ntu_skeleton(results.pose_landmarks)
        features = classifier.extract_realtime_features(ntu_skeleton)

        if features is None:
            return {"action": None, "confidence": 0.0}

        pred, conf = classifier.predict_action(features)
        pred, conf = classifier.smooth_prediction(pred, conf)

        if pred is not None:
            action_name = classifier.action_names[pred]
            return {"action": action_name, "confidence": float(conf)}

        return {"action": None, "confidence": 0.0}

    except Exception as e:
        print("Ошибка при обработке кадра:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"message": "Skeleton-based Action Recognition API is running"}

from fastapi import WebSocket, WebSocketDisconnect
import json

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")

    try:
        while True:
            # Получаем бинарные данные (JPEG-кадр)
            frame_bytes = await websocket.receive_bytes()

            # Преобразуем в OpenCV-матрицу
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Обработка кадра моделью
            results = classifier.converter.process_frame(frame)

            if not results.pose_landmarks:
                await websocket.send_text(json.dumps({
                    "type": "prediction",
                    "prediction": None,
                    "confidence": 0.0
                }))
                continue

            ntu_skeleton = classifier.converter.mediapipe_to_ntu_skeleton(results.pose_landmarks)
            features = classifier.extract_realtime_features(ntu_skeleton)

            if features is None:
                await websocket.send_text(json.dumps({
                    "type": "prediction",
                    "prediction": None,
                    "confidence": 0.0
                }))
                continue

            pred, conf = classifier.predict_action(features)
            pred, conf = classifier.smooth_prediction(pred, conf)

            if pred is not None:
                action_name = classifier.action_names[pred]
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

            await websocket.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print("Ошибка в WebSocket:", e)
        await websocket.close()
