import time
from collections import deque
import os
import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
from collections import deque
from mediapipe_to_ntu import MediaPipeToNTUConverter
from feature_extractor import NTUProcessedFeatureExtractor

class RealTimePoseClassifier:
    def __init__(self, model_path='models/ntu_npy'):
        # Load trained model and artifacts with proper error handling
        print("Loading trained model...")
        
        try:
            # First, check if the model path exists
            if not os.path.exists(model_path):
                print(f"Model path '{model_path}' does not exist!")
                print("Available directories:")
                for item in os.listdir('.'):
                    if os.path.isdir(item):
                        print(f"  - {item}")
                raise FileNotFoundError(f"Model path '{model_path}' not found")
            
            print(f"Model path exists: {model_path}")
            print(f"Files in model directory: {os.listdir(model_path)}")
            
            # Load model files one by one with error handling
            model_file = os.path.join(model_path, 'svm_model.pkl')
            scaler_file = os.path.join(model_path, 'feature_scaler.pkl')
            names_file = os.path.join(model_path, 'action_names.pkl')
            
            print("Loading SVM model...")
            with open(model_file, 'rb') as f:
                self.svm_model = pickle.load(f)
            print("SVM model loaded")
            
            print("Loading feature scaler...")
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Feature scaler loaded")
            
            print("Loading action names...")
            with open(names_file, 'rb') as f:
                self.action_names = pickle.load(f)
            print("Action names loaded")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Initialize converters
        print("Initializing MediaPipe converter...")
        self.converter = MediaPipeToNTUConverter()
        
        print("Initializing feature extractor...")
        self.feature_extractor = NTUProcessedFeatureExtractor()

        # Buffer for temporal features
        self.pose_buffer = deque(maxlen=30)
        self.prediction_buffer = deque(maxlen=10)
        self.fps_history = deque(maxlen=30)

        print(f"Model loaded successfully!")
        print(f"Action classes: {len(self.action_names)}")
        print(f"Sample actions: {self.action_names[:5]}...") 

    def extract_realtime_features(self, ntu_skeleton):
        """
        Extract features compatible with our trained model for real-time inference
        """
        # Add to pose buffer
        self.pose_buffer.append(ntu_skeleton)

        if len(self.pose_buffer) < 5:  # Wait for enough frames
            return None

        # Convert buffer to sequence format
        sequence = np.array(self.pose_buffer)  # (frames, 25, 3)

        # Since we need (300, 150) format for our feature extractor,
        # we'll create a mock sequence that matches the expected format
        mock_sequence = self.create_mock_sequence(sequence)

        # Extract features using our trained feature extractor
        features = self.feature_extractor.extract_features_from_sequence(mock_sequence)

        return features

    def create_mock_sequence(self, real_sequence):
        """
        Create a mock sequence in the format expected by our feature extractor
        (300, 150) where 150 = 2 persons × 25 joints × 3 coordinates
        """
        # Create a 300-frame sequence with our real data
        mock_sequence = np.zeros((300, 150))

        # Fill the beginning with our real data
        num_real_frames = len(real_sequence)
        if num_real_frames > 0:
            # Reshape real sequence to match expected format
            # We only have one person, so we'll put it in the first person slot
            for i in range(min(num_real_frames, 300)):
                # First person data (75 features: 25 joints × 3 coordinates)
                person1_data = real_sequence[i].flatten()
                mock_sequence[i, :75] = person1_data[:75]  # Ensure correct length

                # Second person remains zeros (we only track one person)
                # mock_sequence[i, 75:] = 0  # Already zeros

        return mock_sequence

    def predict_action(self, features):
        """Make prediction using SVM model"""
        if features is None or len(features) == 0:
            return None, 0.0

        # Scale features
        features_scaled = self.scaler.transform([features])

        # Predict
        prediction = self.svm_model.predict(features_scaled)[0]
        probabilities = self.svm_model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)

        return prediction, confidence

    def smooth_prediction(self, current_pred, current_confidence):
        """Apply temporal smoothing to predictions"""
        if current_pred is None:
            if len(self.prediction_buffer) > 0:
                # Return most common prediction from buffer
                predictions = [p[0] for p in self.prediction_buffer]
                most_common = max(set(predictions), key=predictions.count)
                avg_confidence = np.mean([p[1] for p in self.prediction_buffer if p[0] == most_common])
                return most_common, avg_confidence
            else:
                return None, 0.0

        # Add current prediction to buffer
        self.prediction_buffer.append((current_pred, current_confidence))

        if len(self.prediction_buffer) < 3:  # Need minimum buffer size
            return current_pred, current_confidence

        # Get most common prediction from buffer
        predictions = [p[0] for p in self.prediction_buffer]
        most_common = max(set(predictions), key=predictions.count)

        # Calculate average confidence for the most common prediction
        common_confidences = [p[1] for p in self.prediction_buffer if p[0] == most_common]
        avg_confidence = np.mean(common_confidences)

        return most_common, avg_confidence

    def draw_skeleton(self, image, landmarks):
        """Draw MediaPipe skeleton on image"""
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )

    def draw_prediction(self, image, prediction, confidence, fps):
        """Draw prediction and info on image"""
        if prediction is not None and confidence > 0.3:  # Confidence threshold
            action_name = self.action_names[prediction]

            # Draw prediction box
            cv2.rectangle(image, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (400, 100), (255, 255, 255), 2)

            # Draw prediction text
            cv2.putText(image, f"Action: {action_name}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f"FPS: {fps:.1f}", (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(image, "No pose detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def run_realtime_demo(self):
        """Main real-time inference loop with better error handling"""
        print("🎥 Initializing webcam...")
        
        # Test webcam access
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot access webcam. Trying alternative camera indices...")
            
            # Try different camera indices
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"✅ Found webcam at index {i}")
                    break
                cap.release()
            else:
                print("No webcam found. Please check your camera connection.")
                return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Test camera read
        ret, test_frame = cap.read()
        if not ret:
            print(" Cannot read from webcam")
            cap.release()
            return
        else:
            print("Webcam is working")
        
        print("Starting real-time pose classification...")
        print("Press 'q' to quit, 'r' to reset buffer")

        last_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Calculate FPS
            current_time = time.time()
            frame_count += 1
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                self.fps_history.append(fps)
                frame_count = 0
                last_time = current_time
                avg_fps = np.mean(self.fps_history) if self.fps_history else 0
            else:
                avg_fps = np.mean(self.fps_history) if self.fps_history else 0

            try:
                # Process frame with MediaPipe
                results = self.converter.process_frame(frame)

                if results.pose_landmarks:
                    # Draw skeleton
                    self.draw_skeleton(frame, results.pose_landmarks)

                    # Convert to NTU format
                    ntu_skeleton = self.converter.mediapipe_to_ntu_skeleton(results.pose_landmarks)

                    # Extract features
                    features = self.extract_realtime_features(ntu_skeleton)

                    # Make prediction
                    if features is not None:
                        current_pred, current_conf = self.predict_action(features)
                        final_pred, final_conf = self.smooth_prediction(current_pred, current_conf)
                        
                        if final_pred is not None:
                            action_name = self.action_names[final_pred]
                            print(f"Prediction: {action_name} (confidence: {final_conf:.2f}, FPS: {avg_fps:.1f})")
                    else:
                        final_pred, final_conf = None, 0.0
                else:
                    final_pred, final_conf = None, 0.0
                    # Clear buffer when no pose is detected
                    self.pose_buffer.clear()

                # Draw prediction
                self.draw_prediction(frame, final_pred, final_conf, avg_fps)

            except Exception as e:
                print(f"Error in processing frame: {e}")
                final_pred, final_conf = None, 0.0

            # Display frame
            cv2.imshow('Real-time Pose Classification', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset buffers
                self.pose_buffer.clear()
                self.prediction_buffer.clear()
                print("Buffers reset")

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time demo ended")

# Simple test function
def test_inference():
    """Test the inference pipeline without webcam"""
    classifier = RealTimePoseClassifier()
    print("Inference pipeline ready!")
    return classifier