// camera.js
import { WebSocketClient } from "./ws_client.js";

class CameraManager {
  constructor() {
    this.videoElement = document.getElementById('camera-video');
    this.canvasElement = document.getElementById('camera-canvas');
    this.overlayCanvas = document.getElementById('canvas-overlay');
    this.startButton = document.getElementById('btn-start-camera');
    this.stopButton = document.getElementById('btn-stop-camera');
    this.stream = null;
    this.isCameraActive = false;

    // callback will receive backend data
    this.wsClient = new WebSocketClient((data) => this.onPredictionUpdate(data));
    this.wsClient.connect();

    this.initEventListeners();
    // skeleton styling
    this.jointRadius = 5;
    this.jointColor = "lime";
    this.boneColor = "cyan";
  }

  initEventListeners() {
    this.startButton.addEventListener('click', () => this.startCamera());
    this.stopButton.addEventListener('click', () => this.stopCamera());
  }

  async startCamera() {
    try {
      // wait for WS before starting camera to avoid frames getting queued
      console.log("Запуск камеры и подключение к WebSocket...");
      await this.wsClient.waitForConnection(5000);
      console.log("WS подключён, запускаем камеру");

      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
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

    // Очистка всех canvas
    const clearCanvas = (id) => {
      const el = document.getElementById(id);
      if (!el) return;
      const ctx = el.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, el.width, el.height);
    };

    clearCanvas('camera-canvas');
    clearCanvas('canvas-overlay');
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
    const ctx = this.canvasElement.getContext('2d', { willReadFrequently: true });

    const waitForVideoReady = () => new Promise((resolve) => {
      if (this.videoElement.videoWidth > 0 && this.videoElement.videoHeight > 0) resolve();
      else this.videoElement.addEventListener('loadeddata', () => resolve(), { once: true });
    });

    const drawFrame = async () => {
      if (!this.isCameraActive) return;

      // mirror the video when drawing to canvas so user sees selfie view
      ctx.save();
      ctx.translate(this.canvasElement.width, 0);
      ctx.scale(-1, 1);

      // draw current video frame to canvas
      ctx.drawImage(this.videoElement, 0, 0, this.canvasElement.width, this.canvasElement.height);
      ctx.restore();

      // send frame to backend
      await this.wsClient.sendFrame(this.canvasElement);

      // schedule next frame (control rate with setTimeout)
      setTimeout(() => requestAnimationFrame(drawFrame), 150); // ~7 fps — adjust if you want faster
    };

    waitForVideoReady().then(() => {
      // match canvas sizes to video
      this.canvasElement.width = this.videoElement.videoWidth;
      this.canvasElement.height = this.videoElement.videoHeight;
      this.overlayCanvas.width = this.videoElement.videoWidth;
      this.overlayCanvas.height = this.videoElement.videoHeight;
      console.log("🎞 Видео готово, начинаем передачу кадров...");
      drawFrame();
    });
  }

  // draw skeleton received from backend
  drawSkeletonOnCanvas(skeleton, mpSkeleton=null) {
    const ctx = this.overlayCanvas.getContext('2d');
    ctx.clearRect(0,0,this.overlayCanvas.width,this.overlayCanvas.height);
    if (!ctx) return;

    ctx.save();
    try {
      // draw small translucent background for text
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fillRect(0, 0, 420, 36);

      // If mpSkeleton provided (33 points) prefer to draw it — it matches MediaPipe demo
      if (mpSkeleton && Array.isArray(mpSkeleton) && mpSkeleton.length >= 33) {
        // Draw an expanded set of MP connections including requested facial links and both-hand links
        const mpPairs = [
          // arms: shoulder (11/12) -> elbow -> wrist
          [11,13],[13,15],[12,14],[14,16],
          [11,12], // shoulders
          // torso -> hips (approx)
          [11,23],[12,24],[23,24],
          // legs (approx)
          [23,25],[25,27],[24,26],[26,28],
          // head/neck chain
          [0,1],[1,2],[2,3],
          // Additional face connections previously requested:
          [8,6],[6,5],[5,4],      // 8-6-5-4
          [4,0],[0,1],           // 4-0-1
          [1,2],[2,3],[3,7],     // 1-2-3-7
          [9,10],                // 9-10
          // Hand connections (right hand set earlier)
          [16,18],[16,20],[16,22],[18,20],
          // Mirror for left hand (added now)
          [15,17],[15,19],[15,21],[17,19]
        ];

        ctx.lineWidth = 2;
        ctx.strokeStyle = this.boneColor;
        for (let [a,b] of mpPairs) {
          const p1 = mpSkeleton[a];
          const p2 = mpSkeleton[b];
          if (!p1 || !p2) continue;
          ctx.beginPath();
          ctx.moveTo(p1[0], p1[1]);
          ctx.lineTo(p2[0], p2[1]);
          ctx.stroke();
        }
        // draw joints
        for (let i=0;i<mpSkeleton.length;i++){
          const p = mpSkeleton[i];
          if (!p) continue;
          ctx.beginPath();
          ctx.fillStyle = this.jointColor;
          ctx.arc(p[0], p[1], this.jointRadius, 0, 2*Math.PI);
          ctx.fill();
          ctx.fillStyle = "white";
          ctx.font = "12px Arial";
          ctx.fillText(String(i), p[0]+6, p[1]-6);
        }
        return;
      }

      // fallback: draw NTU skeleton (25) as before
      if (!skeleton || !Array.isArray(skeleton)) return;
      const bonePairs = [
        [0,1],[1,20],[20,2],[2,3],
        [2,4],[4,5],[5,6],[6,7],[6,21],[6,22],
        [2,8],[8,9],[9,10],[10,11],[10,23],[10,24],
        [0,12],[12,13],[13,14],[14,15],
        [0,16],[16,17],[17,18],[18,19]
      ];
      // draw bones
      ctx.lineWidth = 2;
      ctx.strokeStyle = this.boneColor;
      for (let [a,b] of bonePairs) {
        const p1 = skeleton[a];
        const p2 = skeleton[b];
        if (!p1 || !p2) continue;
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
      }
      // draw joints
      for (let i=0;i<skeleton.length;i++){
        const p = skeleton[i];
        if (!p) continue;
        ctx.beginPath();
        ctx.fillStyle = this.jointColor;
        ctx.arc(p[0], p[1], this.jointRadius, 0, 2*Math.PI);
        ctx.fill();
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.fillText(String(i), p[0]+6, p[1]-6);
      }
    } finally {
      ctx.restore();
    }
  }

  // Called whenever ws_client gets new prediction object
  onPredictionUpdate(data) {
    if (!data) return;
    // Prefer drawing mp_skeleton if available (matches MediaPipe local demo)
    const mpSk = data.mp_skeleton || null;
    const ntuSk = data.skeleton || null;
    this.drawSkeletonOnCanvas(ntuSk, mpSk);

    // draw prediction text and fill UI elements
    const ctx = this.overlayCanvas.getContext('2d');
    if (ctx) {
      ctx.font = '18px Arial';
      ctx.fillStyle = 'white';
      const pred = data.prediction || '—';
      const conf = (typeof data.confidence === 'number') ? `${(data.confidence*100).toFixed(1)}%` : '-';
      const fps = data.fps || '-';
      // clear top-left area and draw
      ctx.clearRect(0, 0, 420, 36);
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(0, 0, 420, 36);
      ctx.fillStyle = '#00FF00';
      ctx.fillText(`Action: ${pred} (${conf})  FPS:${fps}`, 8, 22);
    }

    // update side panel if exists
    const resultEl = document.getElementById('prediction-result');
    const fillEl = document.getElementById('prediction-fill');
    const confEl = document.getElementById('prediction-confidence');
    if (resultEl) resultEl.textContent = data.prediction || 'Не определено';
    if (confEl) confEl.textContent = (typeof data.confidence === 'number') ? `${(data.confidence*100).toFixed(1)}%` : '-';
    if (fillEl && typeof data.confidence === 'number') {
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