class CameraManager {
    constructor() {
        this.videoElement = document.getElementById('camera-video');
        this.canvasElement = document.getElementById('camera-canvas');
        this.startButton = document.getElementById('btn-start-camera');
        this.stopButton = document.getElementById('btn-stop-camera');
        this.stream = null;
        this.isCameraActive = false;

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
            this.stream.getTracks().forEach(track => {
                track.stop();
            });
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

        const drawFrame = () => {
            if (!this.isCameraActive) return;

            ctx.drawImage(this.videoElement, 0, 0, this.canvasElement.width, this.canvasElement.height);
            
            // Здесь позже будет вызов ML модели для распознавания поз
            // this.detectPose(ctx);

            requestAnimationFrame(drawFrame);
        };

        this.videoElement.addEventListener('loadedmetadata', () => {
            drawFrame();
        });
    }

    // Метод для распознавания поз (будет заполнен позже)
    detectPose(ctx) {
        // Здесь будет логика работы с ML моделью
        // Пока просто рисуем тестовый текст
        ctx.font = '20px Arial';
        ctx.fillStyle = 'red';
        ctx.fillText('Распознавание поз...', 10, 30);
    }

    handleCameraError(error) {
        let errorMessage = 'Неизвестная ошибка камеры';
        
        switch(error.name) {
            case 'NotAllowedError':
                errorMessage = 'Доступ к камере запрещен. Разрешите доступ в настройках браузера.';
                break;
            case 'NotFoundError':
                errorMessage = 'Камера не найдена. Убедитесь, что камера подключена.';
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