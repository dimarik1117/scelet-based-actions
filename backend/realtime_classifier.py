# realtime_classifier.py
import os
import pickle
import cv2
import numpy as np
from collections import deque
import logging
from converter import MediaPipeToNTUConverter
from feature_extractor import NTUProcessedFeatureExtractor

logger = logging.getLogger("realtime_classifier")
logging.getLogger("realtime_classifier").setLevel(logging.INFO)

class RealTimePoseClassifier:
    def __init__(self, model_path="models/ntu_npy", min_buffer_size=20, max_buffer=60, use_scaler=False):
        logger.info("Initializing RealTimePoseClassifier...")
        model_file = os.path.join(model_path, "svm_model.pkl")
        scaler_file = os.path.join(model_path, "feature_scaler.pkl")
        names_file = os.path.join(model_path, "action_names.pkl")

        try:
            with open(model_file, "rb") as f:
                self.svm_model = pickle.load(f)
            # scaler may or may not be present; load if exists but we'll not force using it
            if os.path.exists(scaler_file):
                try:
                    with open(scaler_file, "rb") as f:
                        self.scaler = pickle.load(f)
                except Exception:
                    self.scaler = None
            else:
                self.scaler = None

            with open(names_file, "rb") as f:
                self.action_names = pickle.load(f)
        except Exception as e:
            logger.exception("Error loading model artifacts")
            raise

        # converter and extractor
        self.converter = MediaPipeToNTUConverter()
        self.feature_extractor = NTUProcessedFeatureExtractor()

        # buffers for temporal features
        self.pose_buffer = deque(maxlen=max_buffer)  # keeps recent NTU skeletons (25,3)
        self.prediction_buffer = deque(maxlen=5)

        # expected feature length from scaler (if available)
        try:
            self.expected_feature_length = int(self.scaler.mean_.shape[0]) if self.scaler is not None else None
        except Exception:
            self.expected_feature_length = None

        self.min_buffer_size = max(3, int(min_buffer_size))
        self.use_scaler = bool(use_scaler)  # by default False for realtime amplitude preservation
        logger.info(f"Expected feature length: {self.expected_feature_length}")
        logger.info(f"min_buffer_size={self.min_buffer_size}, buffer_max={self.pose_buffer.maxlen}, use_scaler={self.use_scaler}")

    # frame -> ntu + pixel joints(API preserved)
    def process_single_frame(self, frame):
        """
        Input: BGR frame (numpy)
        Returns:
            ntu_skeleton (25,3) normalized (centered/scaled) or None,
            pixel_joints: list of 25 (x,y) pixel coords (or (None,None) if missing),
            debug_info string
        """
        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.converter.pose.process(rgb)

            if not results.pose_landmarks:
                return None, None, "mediapipe_no_landmarks"

            ntu = self.converter.mediapipe_to_ntu_skeleton(results.pose_landmarks)
            normalized = self.converter.normalize_skeleton(ntu)

            mapping = self.converter.create_correct_joint_mapping()
            pixel_joints = []
            lm = results.pose_landmarks.landmark
            for j in range(25):
                ref = mapping.get(j)
                if callable(ref):
                    try:
                        if j == 0:
                            lx = (lm[23].x + lm[24].x) / 2.0
                            ly = (lm[23].y + lm[24].y) / 2.0
                            pixel_joints.append((lx * w, ly * h))
                        elif j == 1:
                            lx = (lm[23].x + lm[24].x + lm[11].x + lm[12].x) / 4.0
                            ly = (lm[23].y + lm[24].y + lm[11].y + lm[12].y) / 4.0
                            pixel_joints.append((lx * w, ly * h))
                        else:
                            lx = lm[0].x
                            ly = lm[0].y
                            pixel_joints.append((lx * w, ly * h))
                    except Exception:
                        pixel_joints.append((None, None))
                elif isinstance(ref, int):
                    if ref < len(lm):
                        l = lm[ref]
                        try:
                            pixel_joints.append((l.x * w, l.y * h))
                        except Exception:
                            pixel_joints.append((None, None))
                    else:
                        pixel_joints.append((None, None))
                else:
                    pixel_joints.append((None, None))

            return normalized.astype(np.float32), pixel_joints, "ok"

        except Exception as e:
            logger.exception("process_single_frame_error")
            return None, None, f"process_single_frame_error: {e}"

    # create sequence & features
    def create_simple_sequence(self):
        """
        Build (300,150) sequence with last frames placed at end (person2 zeros)
        """
        if len(self.pose_buffer) == 0:
            return None
        seq = np.zeros((300, 150), dtype=np.float32)
        current_sequence = np.array(self.pose_buffer)  # (frames,25,3)
        frames_to_use = current_sequence[-20:] if len(current_sequence) > 20 else current_sequence
        start_idx = 300 - len(frames_to_use)
        for i, frame in enumerate(frames_to_use):
            try:
                f = np.array(frame).reshape(25, 3)
            except Exception:
                f = np.zeros((25, 3), dtype=np.float32)
            p1 = f.flatten()
            p2 = np.zeros(75, dtype=np.float32)
            seq[start_idx + i] = np.concatenate([p1, p2])
        return seq

    def extract_simple_features(self, ntu_skeleton):
        """
        Append skeleton to buffer and extract features using training extractor.
        If extractor fails, use fallback minimal features.
        """
        if ntu_skeleton is None:
            return None

        try:
            ntu_skeleton = np.array(ntu_skeleton, dtype=np.float32).reshape(25, 3)
        except Exception:
            logger.warning("Invalid ntu_skeleton shape when extracting features")
            return None

        # append
        self.pose_buffer.append(ntu_skeleton)
        logger.info(f"Skeleton detected with variance: {np.var(ntu_skeleton):.6f}")
        logger.info(f"Buffer length: {len(self.pose_buffer)}")
        if len(self.pose_buffer) >= 2:
            last = np.array(self.pose_buffer[-1]).reshape(25,3)
            prev = np.array(self.pose_buffer[-2]).reshape(25,3)
            dif = np.linalg.norm(last - prev, axis=1)
            logger.info(f"Δ между последним буфером и новым: {np.mean(dif):.4f}")

        if len(self.pose_buffer) < self.min_buffer_size:
            logger.info(f"Waiting for buffer to fill: {len(self.pose_buffer)}/{self.min_buffer_size}")
            return None

        sequence = self.create_simple_sequence()
        if sequence is None:
            return None

        # Try extracting features using extractor used in training
        try:
            if hasattr(self.feature_extractor, "extract_features_from_sequence"):
                features = self.feature_extractor.extract_features_from_sequence(sequence)
            elif hasattr(self.feature_extractor, "extract_from_sequence"):
                features = self.feature_extractor.extract_from_sequence(sequence)
            else:
                if hasattr(self.feature_extractor, "extract_features"):
                    features = self.feature_extractor.extract_features(sequence)
                else:
                    raise AttributeError("No valid feature extraction method on extractor")
            if features is None or (hasattr(features, "__len__") and len(features) == 0):
                raise ValueError("Feature extractor returned empty features")
        except Exception as e:
            logger.warning(f"Feature extractor failed or missing methods: {e}. Using fallback minimal features.")
            features = self._fallback_minimal_features(sequence)

        try:
            features = np.array(features, dtype=np.float32).flatten()
        except Exception:
            features = np.zeros(self.expected_feature_length or 300, dtype=np.float32)

        # pad/trim to expected length if scaler expected it (keeps compatibility)
        if self.expected_feature_length:
            if len(features) < self.expected_feature_length:
                padded = np.zeros(self.expected_feature_length, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            else:
                features = features[:self.expected_feature_length]

        # safety: NaN/Inf -> 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"Raw features length: {len(features)} min/max: ({np.min(features):.4f},{np.max(features):.4f}) mean:{np.mean(features):.4f}")

        return features

    def _fallback_minimal_features(self, sequence):
        try:
            reshaped = sequence.reshape(300, 2, 25, 3)
            main = reshaped[:, 0, :, :]
        except Exception:
            try:
                main = np.array(sequence[-1, :75]).reshape(25, 3)
                main = np.expand_dims(main, axis=0)
            except:
                return np.zeros(100, dtype=np.float32)

        frames = []
        if main.shape[0] >= 3:
            frames_idx = [0, main.shape[0]//2, -1]
            for idx in frames_idx:
                frames.append(main[idx])
        else:
            for i in range(min(3, main.shape[0])):
                frames.append(main[i])

        features = []
        for f in frames:
            features.extend(f.flatten().tolist())
            bone_pairs = [(0,1), (1,20), (20,2), (2,3), (2,4)]
            for a,b in bone_pairs:
                if a < f.shape[0] and b < f.shape[0]:
                    vec = f[b] - f[a]
                    features.append(np.linalg.norm(vec))
                else:
                    features.append(0.0)

        if len(features) < 300:
            features.extend([0.0] * (300 - len(features)))
        else:
            features = features[:300]
        return np.array(features, dtype=np.float32)

    # prediction
    def predict_from_skeleton(self, ntu_skeleton):
        """
        Given a single ntu_skeleton, produce (pred_idx, confidence, top3)
        Uses buffer-based features prepared by extract_simple_features
        """
        features = self.extract_simple_features(ntu_skeleton)
        if features is None:
            return None, 0.0, None

        try:
            X = np.array(features).reshape(1, -1)
            # scaling: use scaler only if explicitly enabled and available
            if self.use_scaler and hasattr(self, "scaler") and self.scaler is not None:
                try:
                    X_scaled = self.scaler.transform(X)
                except Exception:
                    logger.warning("Scaler transform failed, using raw features")
                    X_scaled = X
            else:
                X_scaled = X

            try:
                logger.info(f"Feature stats after scaling - min:{np.min(X_scaled):.4f} max:{np.max(X_scaled):.4f} mean:{np.mean(X_scaled):.4f}")
            except:
                pass

            probs = self.svm_model.predict_proba(X_scaled)[0]
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))

            top3_idx = np.argsort(probs)[-3:][::-1]
            top3 = [(self.action_names[i] if i < len(self.action_names) else f"class_{i}", float(probs[i])) for i in top3_idx]

            logger.info(f"Top-3 predictions: {top3}")
            logger.info(f"Predicted: {self.action_names[pred] if pred < len(self.action_names) else f'class_{pred}'} (idx={pred}) conf={conf:.3f}")

            return pred, conf, top3
        except Exception as e:
            logger.exception(f"Prediction error: {e}")
            return None, 0.0, None