from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64, io, json, traceback
import numpy as np
import cv2
from PIL import Image

# --- Импорт модели ---
from ml_realtime_inference import RealTimePoseClassifier, predict_single_image_base64

# --- Создание FastAPI приложения ---
app = FastAPI(title="Skeleton-based Action Recognition API")

# --- Настройка CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Глобальная инициализация модели (для HTTP) ---
print("Инициализация модели...")
global_classifier = RealTimePoseClassifier()
print("Модель загружена!")


# --- Pydantic модель для POST-запросов ---
class ImageRequest(BaseModel):
    image: str  # base64 строка


@app.get("/")
def home():
    return {"message": "Skeleton-based Action Recognition API is running"}

# Предсказание одиночного изображения (Postman)

@app.post("/predict-image")
async def predict_image(request: ImageRequest):
    """
    Endpoint для теста одиночного изображения (base64)
    """
    try:
        result = predict_single_image_base64(request.image)
        return result
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket для стриминга видео с фронтенда

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    """
    Принимает поток кадров (JPEG bytes) с фронтенда.
    Для каждого клиента создается СВОЙ экземпляр модели,
    чтобы буфер поз не сбрасывался между кадрами.
    """
    print("Ожидание WebSocket соединения...")
    await websocket.accept()
    print("✅ WebSocket клиент подключен")

    # Создаем отдельный экземпляр классификатора под клиента
    classifier = RealTimePoseClassifier()

    try:
        while True:
            # Получаем кадр (байты изображения)
            frame_bytes = await websocket.receive_bytes()
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid frame data"
                }))
                continue

            # Обработка через Mediapipe
            results = classifier.converter.process_frame(frame)
            if not results.pose_landmarks:
                await websocket.send_text(json.dumps({
                    "type": "prediction",
                    "prediction": None,
                    "confidence": 0.0
                }))
                continue

            # Конвертация позы
            ntu_skeleton = classifier.converter.mediapipe_to_ntu_skeleton(results.pose_landmarks)

            # Извлечение признаков из буфера поз
            features = classifier.extract_realtime_features(ntu_skeleton)
            if features is None:
                # Просто продолжаем накапливать
                continue

            # Предсказание
            pred, conf = classifier.predict_action(features)
            pred, conf = classifier.smooth_prediction(pred, conf)

            # Отправка ответа на фронт
            if pred is not None:
                action_name = classifier.action_names[pred]
                payload = {
                    "type": "prediction",
                    "prediction": action_name,
                    "confidence": float(conf)
                }
                print(f"Отправляю предсказание: {action_name} ({conf:.2f})")
            else:
                payload = {
                    "type": "prediction",
                    "prediction": None,
                    "confidence": 0.0
                }

            await websocket.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        print("WebSocket клиент отключен")
    except Exception as e:
        print("Ошибка в WebSocket:", e)
        print(traceback.format_exc())
        await websocket.close()
