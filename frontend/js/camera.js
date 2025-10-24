import { WebSocketClient } from "./ws_client.js";

class CameraManager {
  constructor() {
    this.videoElement = document.getElementById('camera-video');
    this.canvasElement = document.getElementById('camera-canvas');
    this.startButton = document.getElementById('btn-start-camera');
    this.stopButton = document.getElementById('btn-stop-camera');
    this.stream = null;
    this.isCameraActive = false;

    this.wsClient = new WebSocketClient((data) => this.onPredictionUpdate(data));
    this.wsClient.connect();

    this.initEventListeners();
  }

  initEventListeners() {
    this.startButton.addEventListener('click', () => this.startCamera());
    this.stopButton.addEventListener('click', () => this.stopCamera());
  }

  async startCamera() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });

      this.videoElement.srcObject = this.stream;
      this.isCameraActive = true;
      this.updateUI(true);

      // Подождём пока WebSocket подключится перед отправкой кадров
      await this.wsClient.waitForConnection();

      this.processVideo();
      console.log('Камера успешно запущена');
    } catch (error) {
      console.error('Ошибка при запуске камеры:', error);
      this.handleCameraError(error);
    }
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.videoElement.srcObject = null;
    this.isCameraActive = false;
    this.updateUI(false);

    const ctx = this.canvasElement.getContext('2d');
    ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    console.log('Камера остановлена');
  }

  updateUI(isCameraRunning) {
    if (isCameraRunning) {
      this.startButton.disabled = true;
      this.stopButton.disabled = false;
      this.startButton.textContent = 'Камера запущена';
    } else {
      this.startButton.disabled = false;
      this.stopButton.disabled = true;
      this.startButton.textContent = 'Запустить камеру';
    }
  }

  processVideo() {
  if (!this.isCameraActive) return;
  const ctx = this.canvasElement.getContext('2d');

  const waitForVideoReady = () => new Promise((resolve) => {
    if (this.videoElement.videoWidth > 0 && this.videoElement.videoHeight > 0) {
      resolve();
    } else {
      this.videoElement.addEventListener('loadeddata', () => resolve(), { once: true });
    }
  });

  const drawFrame = async () => {
    if (!this.isCameraActive) return;
    ctx.drawImage(this.videoElement, 0, 0, this.canvasElement.width, this.canvasElement.height);
    await this.wsClient.sendFrame(this.canvasElement);
    setTimeout(() => requestAnimationFrame(drawFrame), 500);
  };

  waitForVideoReady().then(() => {
    this.canvasElement.width = this.videoElement.videoWidth;
    this.canvasElement.height = this.videoElement.videoHeight;
    console.log("Видео готово, начинаем передачу кадров...");
    drawFrame();
  });
}

  // Обновляем UI по предсказаниям
  onPredictionUpdate(data) {
    const ctx = this.canvasElement.getContext('2d');
    // небольшой резерв: если canvas пуст — ничего не рисуем
    if (!ctx) return;

    ctx.font = '20px Arial';
    ctx.fillStyle = 'red';

    const text = data && data.prediction
      ? `Действие: ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`
      : 'Нет действия';

    // Стираем место сверху, чтобы не накладывалось
    ctx.clearRect(0, 0, 400, 40);
    ctx.fillText(text, 10, 30);

    // Обновим элементы в правой панели (если они есть)
    const resultEl = document.getElementById('prediction-result');
    const fillEl = document.getElementById('prediction-fill');
    const confEl = document.getElementById('prediction-confidence');
    if (resultEl) resultEl.textContent = data && data.prediction ? data.prediction : 'Не определено';
    if (confEl) confEl.textContent = data && typeof data.confidence === 'number' ? `${(data.confidence*100).toFixed(1)}%` : '-';
    if (fillEl && data && typeof data.confidence === 'number') {
      const pct = Math.min(100, Math.max(0, Math.round(data.confidence * 100)));
      fillEl.style.width = `${pct}%`;
    }
  }

  handleCameraError(error) {
    let errorMessage = 'Неизвестная ошибка камеры';
    if (error && error.name) {
      switch (error.name) {
        case 'NotAllowedError':
          errorMessage = 'Доступ к камере запрещен.';
          break;
        case 'NotFoundError':
          errorMessage = 'Камера не найдена.';
          break;
        case 'NotSupportedError':
          errorMessage = 'Ваш браузер не поддерживает доступ к камере.';
          break;
        case 'NotReadableError':
          errorMessage = 'Камера используется другим приложением.';
          break;
      }
    }
    alert(`Ошибка: ${errorMessage}`);
    this.updateUI(false);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new CameraManager();
});