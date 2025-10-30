# realtime_classifier.py (with run_realtime_demo)
import os
import pickle
import cv2
import numpy as np
from collections import deque
import logging
import mediapipe as mp
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
        self.use_scaler = bool(use_scaler)
        logger.info(f"Expected feature length: {self.expected_feature_length}")
        logger.info(f"min_buffer_size={self.min_buffer_size}, buffer_max={self.pose_buffer.maxlen}, use_scaler={self.use_scaler}")

        # MediaPipe drawing helpers
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def process_single_frame(self, frame):
        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.converter.pose.process(rgb)

            if not results.pose_landmarks:
                return None, None, None

            ntu = self.converter.mediapipe_to_ntu_skeleton(results.pose_landmarks.landmark)
            normalized = self.converter.normalize_skeleton(ntu)

            # Pixel joints computed for potential overlay (not used for centered NTU draw)
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
                        elif j == 20:
                            shoulder_x = (lm[11].x + lm[12].x) / 2.0
                            shoulder_y = (lm[11].y + lm[12].y) / 2.0
                            hip_x = (lm[23].x + lm[24].x) / 2.0
                            hip_y = (lm[23].y + lm[24].y) / 2.0
                            lx = shoulder_x * 0.3 + hip_x * 0.7
                            ly = shoulder_y * 0.3 + hip_y * 0.7
                            pixel_joints.append((lx * w, ly * h))
                        elif j == 2:
                            lx = (lm[11].x + lm[12].x) / 2.0
                            ly = (lm[11].y + lm[12].y) / 2.0
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

            return normalized.astype(np.float32), pixel_joints, results.pose_landmarks

        except Exception as e:
            logger.exception("process_single_frame_error")
            return None, None, None

    def create_simple_sequence(self):
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
        if ntu_skeleton is None:
            return None
        try:
            ntu_skeleton = np.array(ntu_skeleton, dtype=np.float32).reshape(25, 3)
        except Exception:
            logger.warning("Invalid ntu_skeleton shape when extracting features")
            return None
        self.pose_buffer.append(ntu_skeleton)
        logger.info(f"Skeleton detected with variance: {np.var(ntu_skeleton):.6f}")
        logger.info(f"Buffer length: {len(self.pose_buffer)}")
        if len(self.pose_buffer) < self.min_buffer_size:
            logger.info(f"Waiting for buffer to fill: {len(self.pose_buffer)}/{self.min_buffer_size}")
            return None
        sequence = self.create_simple_sequence()
        if sequence is None:
            return None
        try:
            if hasattr(self.feature_extractor, "extract_features_from_sequence"):
                features = self.feature_extractor.extract_features_from_sequence(sequence)
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
        if self.expected_feature_length:
            if len(features) < self.expected_feature_length:
                padded = np.zeros(self.expected_feature_length, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            else:
                features = features[:self.expected_feature_length]
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

    def predict_from_skeleton(self, ntu_skeleton):
        features = self.extract_simple_features(ntu_skeleton)
        if features is None:
            return None, 0.0, None
        try:
            X = np.array(features).reshape(1, -1)
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

    # Visualization helpers (centered NTU skeleton like old pipeline)
    def visualize_centered_ntu(self, frame, normalized_skeleton):
        """Draw NTU skeleton centered in the middle of the frame (old behavior)."""
        if normalized_skeleton is None:
            return frame
        h, w = frame.shape[:2]
        display = normalized_skeleton.copy().astype(np.float32)
        # compute spine length (in normalized units) to scale for display
        try:
            spine_len = np.linalg.norm(display[2] - display[0])
            if spine_len == 0:
                spine_len = 1.0
        except:
            spine_len = 1.0
        # scale to occupy a good portion of the frame (similar to old code)
        scale_factor = min(h, w) * 0.8 / (spine_len + 1e-9)
        display *= scale_factor
        # center to image center
        display[:, 0] += w // 2
        display[:, 1] += h // 2
        # draw joints and bones
        bone_pairs = self.feature_extractor.bone_pairs
        for i, joint in enumerate(display):
            x, y, _ = joint
            if np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        for a, b in bone_pairs:
            if a < len(display) and b < len(display):
                p1 = display[a]
                p2 = display[b]
                if np.all(np.isfinite(p1)) and np.all(np.isfinite(p2)):
                    x1, y1, _ = p1
                    x2, y2, _ = p2
                    if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 3)
        return frame

    def run_realtime_demo(self, camera_index=0):
        """Run webcam demo drawing MediaPipe landmarks + centered NTU skeleton (old visual)."""
        print("Starting realtime demo (press q to quit, r to reset buffer)...")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        last_time = cv2.getTickCount()
        fps = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            norm, pixel_joints, pose_landmarks = self.process_single_frame(frame)
            # Draw MediaPipe landmarks (full 33 points) if available
            if pose_landmarks is not None:
                try:
                    self.mp_drawing.draw_landmarks(frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS, mp.solutions.drawing_styles.get_default_pose_landmarks_style().landmark_drawing_spec, mp.solutions.drawing_styles.get_default_pose_landmarks_style().connection_drawing_spec)
                except Exception:
                    pass
            # Draw centered NTU skeleton (old-style visualization)
            frame = self.visualize_centered_ntu(frame, norm)
            # show FPS
            current = cv2.getTickCount()
            dt = (current - last_time) / cv2.getTickFrequency()
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)
            last_time = current
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow("Real-time Pose Classification (MediaPipe + NTU)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.pose_buffer.clear()
                self.prediction_buffer.clear()
                print("Buffer reset")
        cap.release()
        cv2.destroyAllWindows()
        print("Demo ended")

if __name__ == "__main__":
    try:
        classifier = RealTimePoseClassifier(min_buffer_size=3)
        classifier.run_realtime_demo()
    except Exception as e:
        print(f"Error starting demo: {e}")
