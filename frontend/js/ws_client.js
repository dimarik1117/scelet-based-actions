console.log("📦 ws_client.js загружен");

export class WebSocketClient {
    constructor(onPredictionUpdate) {
        this.ws = null;
        this.onPredictionUpdate = onPredictionUpdate;
        this.isConnected = false;
    }

    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) return;

        console.log("🔌 Попытка подключения к WebSocket...");
        this.ws = new WebSocket("ws://127.0.0.1:8000/ws/predict");
        this.ws.binaryType = "arraybuffer";

        this.ws.onopen = () => {
            this.isConnected = true;
            console.log("✅ WebSocket подключен");
        };

        this.ws.onclose = (event) => {
            this.isConnected = false;
            console.warn(`❌ WebSocket отключен:`, event.code, event.reason);
        };

        this.ws.onerror = (error) => {
            console.error("⚠️ Ошибка WebSocket:", error);
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log("📩 Получено сообщение:", data);
                if (data.type === "prediction") {
                    this.onPredictionUpdate(data);
                }
            } catch (err) {
                console.error("Ошибка парсинга WS-сообщения:", err);
            }
        };
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    async sendFrame(canvas) {
        if (!this.isConnected || !canvas) return;

        return new Promise((resolve) => {
            canvas.toBlob(async (blob) => {
                if (!blob) return resolve();
                const buffer = await blob.arrayBuffer();
                this.ws.send(buffer);
                resolve();
            }, "image/jpeg", 0.7);
        });
    }
}
