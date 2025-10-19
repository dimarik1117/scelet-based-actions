import cv2
import mediapipe as mp
from ml_realtime_inference import RealTimePoseClassifier  # замени на реальный импорт

# Загружаем фото
image = cv2.imread("2chel.jpg")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if results.pose_landmarks:
    print("Pose detected, running classification...")
    classifier = RealTimePoseClassifier()
    ntu_skeleton = classifier.converter.mediapipe_to_ntu_skeleton(results.pose_landmarks)
    features = classifier.extract_realtime_features(ntu_skeleton)
    if features is not None:
        pred, conf = classifier.predict_action(features)
        print(f"Predicted action: {classifier.action_names[pred]} (confidence={conf:.2f})")
    else:
        print("Not enough frames to extract features")
else:
    print("No pose detected in this image")
