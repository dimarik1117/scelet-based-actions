import numpy as np
import joblib
import tensorflow as tf
from feature_extractor import NTUFeatureExtractor

class ModelPredictor:
    def __init__(self,
                 model_path="models/ntu_npy/svm_model.pkl",
                 scaler_path="models/ntu_npy/feature_scaler.pkl",
                 actions_path="models/ntu_npy/action_names.pkl"):

        if model_path.endswith(".tflite"):
            self.model_type = "tflite"
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model_type = "sklearn"
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)

        self.actions = joblib.load(actions_path)
        self.extractor = NTUFeatureExtractor()

    def predict(self, skeleton):
        """ skeleton — np.array формы (25, 3) """
        try:
            features = self.extractor.extract_features(skeleton)
            features = np.array(features).reshape(1, -1)

            if self.model_type == "sklearn":
                features = self.scaler.transform(features)
                probs = self.model.predict_proba(features)[0]
            else:
                self.interpreter.set_tensor(self.input_details[0]['index'], features.astype(np.float32))
                self.interpreter.invoke()
                probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            pred_idx = int(np.argmax(probs))
            confidence = float(np.max(probs))
            action = self.actions[pred_idx]

            return action, confidence

        except Exception as e:
            print(f"Prediction error: {e}")
            return "error", 0.0