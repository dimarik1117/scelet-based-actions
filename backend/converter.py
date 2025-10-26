import mediapipe as mp
import cv2
import numpy as np

class MediaPipeToNTUConverter:
    def __init__(self):
        # Настройки MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        # фиксированный масштаб (устанавливается при первом валидном кадре)
        self._initial_scale = None

    def create_correct_joint_mapping(self):
        """
        CORRECT mapping from MediaPipe's 33 landmarks to NTU's 25 joints
        Based on anatomical correspondence
        """
        return {
            # SPINE CHAIN (4 joints)
            0: self.calculate_pelvis,      # pelvis - calculated from hips
            1: self.calculate_middle_spine, # middle_of_spine - calculated
            20: self.calculate_spine,      # spine - calculated  
            2: self.calculate_neck,   # neck - using midpoint between shoulders (MediaPipe 11,12)
            3: 0,    # head - NOSE (0)
            
            # LEFT ARM (6 joints)
            4: 11,   # left_shoulder - LEFT_SHOULDER (11)
            5: 13,   # left_elbow - LEFT_ELBOW (13)
            6: 15,   # left_wrist - LEFT_WRIST (15)
            7: 19,   # left_hand - LEFT_PINKY (19) - closest approximation
            21: 17,  # tip_left_hand - LEFT_INDEX (17)
            22: 21,  # left_thumb - LEFT_THUMB (21)
            
            # RIGHT ARM (6 joints)
            8: 12,   # right_shoulder - RIGHT_SHOULDER (12)
            9: 14,   # right_elbow - RIGHT_ELBOW (14)
            10: 16,  # right_wrist - RIGHT_WRIST (16)
            11: 20,  # right_hand - RIGHT_PINKY (20) - closest approximation
            23: 18,  # tip_right_hand - RIGHT_INDEX (18)
            24: 22,  # right_thumb - RIGHT_THUMB (22)
            
            # LEFT LEG (4 joints)
            12: 23,  # left_hip - LEFT_HIP (23)
            13: 25,  # left_knee - LEFT_KNEE (25)
            14: 27,  # left_ankle - LEFT_ANKLE (27)
            15: 31,  # left_foot - LEFT_FOOT_INDEX (31)
            
            # RIGHT LEG (4 joints)
            16: 24,  # right_hip - RIGHT_HIP (24)
            17: 26,  # right_knee - RIGHT_KNEE (26)
            18: 28,  # right_ankle - RIGHT_ANKLE (28)
            19: 32,  # right_foot - RIGHT_FOOT_INDEX (32)
        }

    def calculate_pelvis(self, landmarks):
        left_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
        right_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
        return (left_hip + right_hip) / 2

    def calculate_middle_spine(self, landmarks):
        left_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
        right_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
        left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
        right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
        hip_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        return (hip_center + shoulder_center) / 2

    def calculate_spine(self, landmarks):
        left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
        right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
        left_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
        right_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        spine = shoulder_center * 0.3 + hip_center * 0.7
        return spine

    def calculate_neck(self, landmarks):
        left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
        right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
        return (left_shoulder + right_shoulder) / 2

    def mediapipe_to_ntu_skeleton(self, landmarks):
        ntu_skeleton = np.zeros((25, 3))
        mapping = self.create_correct_joint_mapping()

        for ntu_joint, mp_reference in mapping.items():
            if callable(mp_reference):
                ntu_skeleton[ntu_joint] = mp_reference(landmarks.landmark)
            elif mp_reference < len(landmarks.landmark):
                landmark = landmarks.landmark[mp_reference]
                ntu_skeleton[ntu_joint] = [landmark.x, landmark.y, landmark.z]

        return ntu_skeleton

    def normalize_skeleton(self, skeleton):
        pelvis = skeleton[0].copy()
        centered_skeleton = skeleton - pelvis

        # try to compute spine length for scale
        try:
            spine_base = centered_skeleton[0]
            spine_top = centered_skeleton[2]
            spine_length = np.linalg.norm(spine_top - spine_base)
        except Exception:
            spine_length = None

        # Если масштаб ещё не зафиксирован — фиксируем его по первому валидному кадру
        if self._initial_scale is None:
            if spine_length is not None and spine_length > 1e-4:
                self._initial_scale = spine_length
            else:
                # fallback — расстояние между плечами и тазом
                try:
                    left_shoulder = skeleton[11]
                    right_shoulder = skeleton[12]
                    left_hip = skeleton[23]
                    right_hip = skeleton[24]
                    alt_len = np.linalg.norm((left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
                    self._initial_scale = alt_len if alt_len > 1e-4 else 1.0
                except Exception:
                    self._initial_scale = 1.0

        if self._initial_scale is None or self._initial_scale <= 1e-6:
            scaled_skeleton = centered_skeleton
        else:
            scaled_skeleton = centered_skeleton / (self._initial_scale + 1e-9)

        return scaled_skeleton

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None, None
        ntu_skeleton = self.mediapipe_to_ntu_skeleton(results.pose_landmarks)
        normalized_skeleton = self.normalize_skeleton(ntu_skeleton)
        return normalized_skeleton, results.pose_landmarks