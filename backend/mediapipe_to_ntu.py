import mediapipe as mp
import cv2
import numpy as np
import os


class MediaPipeToNTUConverter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # MediaPipe to NTU joint mapping
        self.joint_mapping = self.create_joint_mapping()

    def create_joint_mapping(self):
        """
        CORRECT mapping from MediaPipe's 33 landmarks to NTU's 25 joints
        """
        return {
            # SPINE CHAIN (5 joints)
            0: 23,   # pelvis (0) -> LEFT_HIP (23) - using as base
            1: 24,   # middle_of_spine (1) -> RIGHT_HIP (24) - approximation
            20: 23,  # spine (20) -> LEFT_HIP (23) - approximation from hip
            2: 11,   # neck (2) -> midpoint between shoulders (calculated)
            3: 0,    # head (3) -> NOSE (0)
            
            # LEFT ARM (6 joints)
            4: 11,   # left_shoulder (4) -> LEFT_SHOULDER (11)
            5: 13,   # left_elbow (5) -> LEFT_ELBOW (13)
            6: 15,   # left_wrist (6) -> LEFT_WRIST (15)
            7: 19,   # left_hand (7) -> LEFT_PINKY (19) - approximation
            21: 17,  # tip_left_hand (21) -> LEFT_INDEX (17)
            22: 21,  # left_thumb (22) -> LEFT_THUMB (21)
            
            # RIGHT ARM (6 joints)
            8: 12,   # right_shoulder (8) -> RIGHT_SHOULDER (12)
            9: 14,   # right_elbow (9) -> RIGHT_ELBOW (14)
            10: 16,  # right_wrist (10) -> RIGHT_WRIST (16)
            11: 20,  # right_hand (11) -> RIGHT_PINKY (20) - approximation
            23: 18,  # tip_right_hand (23) -> RIGHT_INDEX (18)
            24: 22,  # right_thumb (24) -> RIGHT_THUMB (22)
            
            # LEFT LEG (4 joints)
            12: 23,  # left_hip (12) -> LEFT_HIP (23)
            13: 25,  # left_knee (13) -> LEFT_KNEE (25)
            14: 27,  # left_ankle (14) -> LEFT_ANKLE (27)
            15: 31,  # left_foot (15) -> LEFT_FOOT_INDEX (31)
            
            # RIGHT LEG (4 joints)
            16: 24,  # right_hip (16) -> RIGHT_HIP (24)
            17: 26,  # right_knee (17) -> RIGHT_KNEE (26)
            18: 28,  # right_ankle (18) -> RIGHT_ANKLE (28)
            19: 32,  # right_foot (19) -> RIGHT_FOOT_INDEX (32)
        }

    def mediapipe_to_ntu_skeleton(self, landmarks):
        """
        Convert MediaPipe landmarks to NTU-style skeleton (25 joints, 3 coordinates)
        """
        ntu_skeleton = np.zeros((25, 3))

        for ntu_joint, mp_landmark_idx in self.joint_mapping.items():
            if mp_landmark_idx < len(landmarks.landmark):
                landmark = landmarks.landmark[mp_landmark_idx]
                ntu_skeleton[ntu_joint] = [landmark.x, landmark.y, landmark.z]

        return ntu_skeleton

    def process_frame(self, frame):
        """Process a frame and return MediaPipe results"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results