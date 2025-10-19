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
                    width: { ideal: 1920 },
                    height: { ideal: 1080 },
                    facingMode: 'user'
                }
            });

            this.videoElement.srcObject = this.stream;
            this.isCameraActive = true;
            this.updateUI(true);

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
        this.canvasElement.width = this.videoElement.videoWidth;
        this.canvasElement.height = this.videoElement.videoHeight;

        const drawFrame = async () => {
            if (!this.isCameraActive) return;

            ctx.drawImage(this.videoElement, 0, 0, this.canvasElement.width, this.canvasElement.height);

            // === Отправляем кадр на backend ===
            await this.wsClient.sendFrame(this.canvasElement);

            // Можно отправлять не каждый кадр, чтобы снизить нагрузку
            await new Promise(r => setTimeout(r, 300)); // каждые 300 мс

            requestAnimationFrame(drawFrame);
        };

        this.videoElement.addEventListener('loadedmetadata', () => {
            drawFrame();
        });
    }

    // Обновляем UI по предсказаниям
    onPredictionUpdate(data) {
        const ctx = this.canvasElement.getContext('2d');
        ctx.font = '20px Arial';
        ctx.fillStyle = 'red';
        ctx.fillText(
            data.prediction
                ? `Действие: ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`
                : 'Нет действия',
            10,
            30
        );
    }

    handleCameraError(error) {
        let errorMessage = 'Неизвестная ошибка камеры';

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

        alert(`Ошибка: ${errorMessage}`);
        this.updateUI(false);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new CameraManager();
});