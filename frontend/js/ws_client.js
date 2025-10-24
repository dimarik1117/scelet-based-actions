console.log("ws_client.js загружен");

export class WebSocketClient {
  constructor(onPredictionUpdate) {
    this.ws = null;
    this.onPredictionUpdate = onPredictionUpdate;
    this.isConnected = false;
    this._connectPromiseResolve = null;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) return;
    console.log("Попытка подключения к WebSocket...");
    this.ws = new WebSocket("ws://127.0.0.1:8000/ws/predict");
    this.ws.binaryType = "arraybuffer";

    // промис для ожидания подключения
    this._connectPromise = new Promise((resolve) => {
      this._connectPromiseResolve = resolve;
    });

    this.ws.onopen = () => {
      this.isConnected = true;
      console.log("WebSocket подключен");
      if (this._connectPromiseResolve) {
        this._connectPromiseResolve();
        this._connectPromiseResolve = null;
      }
    };

    this.ws.onclose = (event) => {
      this.isConnected = false;
      console.warn("WebSocket отключен:", event.code, event.reason);
      // Попробуем переподключиться через 1s
      setTimeout(() => this.connect(), 1000);
    };

    this.ws.onerror = (error) => {
      console.error("Ошибка WebSocket:", error);
    };

    this.ws.onmessage = (event) => {
      try {
        // ожидаем текстовый JSON
        const text = typeof event.data === "string" ? event.data : new TextDecoder().decode(event.data);
        const data = JSON.parse(text);
        // console.log("Получено сообщение от backend:", data);
        if (data && data.type === "prediction") {
          this.onPredictionUpdate(data);
        } else if (data && data.type === "error") {
          console.warn("Backend error message:", data);
        }
      } catch (err) {
        console.error("Ошибка парсинга WS-сообщения:", err);
      }
    };
  }

  // ожидание установки соединения (используется фронтом)
  waitForConnection(timeoutMs = 5000) {
    if (this.isConnected) return Promise.resolve();
    if (this._connectPromise) {
      return Promise.race([
        this._connectPromise,
        new Promise((_, reject) => setTimeout(() => reject(new Error("WS connection timeout")), timeoutMs))
      ]);
    }
    // если connect ещё не вызывался — вызываем и ждём
    this.connect();
    return this.waitForConnection(timeoutMs);
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
  }

  async sendFrame(canvas) {
    console.log("Отправка кадра через WS...");
    if (!this.isConnected || !canvas || !this.ws) return;
    return new Promise((resolve) => {
      canvas.toBlob(async (blob) => {
        if (!blob) return resolve();
        try {
          // уменьшаем качество до 0.4 чтобы снизить нагрузку
          // blob уже в формате image/png/jpeg в зависимости от toBlob вызова — но toBlob не принимает качество для png
          // поэтому используем canvas.toBlob с типом и качеством выше (в CameraManager мы вызываем sendFrame с canvas)
          const arrayBuffer = await blob.arrayBuffer();
          this.ws.send(arrayBuffer);
        } catch (err) {
          console.error("Ошибка при отправке кадра:", err);
        } finally {
          resolve();
        }
      }, "image/jpeg", 0.4); // качество 0.4 — компромисс скорость/качество
    });
  }
}