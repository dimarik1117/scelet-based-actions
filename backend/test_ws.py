import asyncio
import websockets
import json
import cv2
import numpy as np

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws/predict"
    async with websockets.connect(uri) as websocket:
        print("Подключено к WebSocket")

        # Загружаем любой тестовый кадр
        frame = cv2.imread("water.jpg")
        _, buffer = cv2.imencode(".jpg", frame)
        await websocket.send(buffer.tobytes())
        print("Отправлен кадр")

        # Получаем ответ
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print("Ответ от backend:", data)

asyncio.run(test_ws())
