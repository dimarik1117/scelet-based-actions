import asyncio
import websockets
import json
import cv2
import numpy as np

async def test_websocket():
    uri = "ws://127.0.0.1:8000/ws/predict"
    async with websockets.connect(uri, max_size=None) as websocket:
        print("Connected to WebSocket")

        # Загружаем тестовое изображение (например, frame.jpg)
        # или просто создаём чёрный кадр
        frame = cv2.imread("clap2.jpg")
        if frame is None:
            frame = (255 * np.ones((480, 640, 3), dtype=np.uint8))

        # Кодируем в JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        await websocket.send(buffer.tobytes())
        print("📤 Sent frame")

        # Ожидаем ответ от сервера
        response = await websocket.recv()
        data = json.loads(response)
        print("📥 Received response:", data)

asyncio.run(test_websocket())
