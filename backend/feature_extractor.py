from scipy.spatial import distance
import numpy as np
import os


class NTUProcessedFeatureExtractor:
    def __init__(self):
        # CORRECT NTU RGB+D 25 Joints
        self.joint_names = [
            "pelvis",       # 1(base of spine)
            "middle_of_spine",      # 2 
            "neck", "head", # 3, 4
            "left_shoulder", "left_elbow", "left_wrist",  # 5, 6, 7
            "left_hand", "right_shoulder", # 8, 9 
            "right_elbow", "right_wrist",  # 10, 11
            "right_hand", "left_hip", # 12, 13
             "left_knee", "left_ankle", "left_foot",  # 14, 15, 16
            "right_hip", "right_knee", "right_ankle", "right_foot",  # 17, 18, 19, 20
            "spine", "tip_left_hand" # 21, 22
            "left_thumb", "tip_right_hand" #23, 24
            "right_thumb" #25
        ]
        
        # CORRECT BONE PAIRS for NTU RGB+D
        self.bone_pairs = [
            # Spine chain (4 bones)
            (1, 0),    # pelvis → middle_of_spine
            (20, 1),    # middle_of_spine → spine  
            (2, 20),    # spine → neck
            (3, 2),    # neck → head
            
            # Left arm chain (5 bones)
            (4, 2),    # neck → left_shoulder (connects to neck, not spine_2)
            (5, 4),    # left_shoulder → left_elbow
            (6, 5),    # left_elbow → left_wrist
            (7, 6),   # left_wrist → left_hand
            (21, 6), # left_wrist -> tip_left_hand
            (22, 6),   # left_wrist → left_thumb
            
            # Right arm chain (5 bones)
            (8, 2),    # neck → right_shoulder
            (9, 8),   # right_shoulder → right_elbow
            (10, 9),  # right_elbow → right_wrist
            (11, 10),  # right_wrist → right_hand
            (23, 10), # right_wrist -> tip_right_hand
            (24, 10),  # right_wrist → right_thumb
            
            # Left leg chain (4 bones)
            (12, 0),   # pelvis → left_hip
            (13, 12),  # left_hip → left_knee
            (14, 13),  # left_knee → left_ankle
            (15, 14),  # left_ankle → left_foot
            
            # Right leg chain (4 bones)
            (16, 0),   # pelvis → right_hip  
            (17, 16),  # right_hip → right_knee
            (18, 17),  # right_knee → right_ankle
            (19, 18),  # right_ankle → right_foot
        ]

        '''
            self.bone_pairs = [
            # Spine chain (4 bones)
            (2, 1),    # pelvis → middle_of_spine
            (21, 2),   # middle_of_spine → spine  
            (3, 21),   # spine → neck
            (4, 3),    # neck → head
            
            # Left arm chain (6 bones)
            (5, 3),    # neck → left_shoulder
            (6, 5),    # left_shoulder → left_elbow
            (7, 6),    # left_elbow → left_wrist
            (8, 7),    # left_wrist → left_hand
            (22, 7),   # left_wrist → tip_left_hand
            (23, 7),   # left_wrist → left_thumb
            
            # Right arm chain (6 bones)
            (9, 3),    # neck → right_shoulder
            (10, 9),   # right_shoulder → right_elbow
            (11, 10),  # right_elbow → right_wrist
            (12, 11),  # right_wrist → right_hand
            (24, 11),  # right_wrist → tip_right_hand
            (25, 11),  # right_wrist → right_thumb
            
            # Left leg chain (4 bones)
            (13, 1),   # pelvis → left_hip
            (14, 13),  # left_hip → left_knee
            (15, 14),  # left_knee → left_ankle
            (16, 15),  # left_ankle → left_foot
            
            # Right leg chain (4 bones)
            (17, 1),   # pelvis → right_hip  
            (18, 17),  # right_hip → right_knee
            (19, 18),  # right_knee → right_ankle
            (20, 19),  # right_ankle → right_foot
        ]
        
        '''
    def extract_features_from_sequence(self, sequence):
        """
        Extract features from preprocessed NTU sequence
        sequence shape: (300, 150) -> will be reshaped to (300, 2, 25, 3)
        """
        # Reshape sequence
        reshaped_sequence = sequence.reshape(300, 2, 25, 3)

        # Use only the first person (main subject)
        main_person_sequence = reshaped_sequence[:, 0, :, :]  # (300, 25, 3)

        # Remove zero-padded frames
        valid_frames = self.get_valid_frames(main_person_sequence)

        if len(valid_frames) == 0:
            return np.array([])

        features = []

        # Extract spatial features from each valid frame
        spatial_features = []
        for frame in valid_frames:
            frame_feat = self.extract_spatial_features(frame)
            spatial_features.append(frame_feat)

        spatial_features = np.array(spatial_features)

        # Extract temporal features
        temporal_features = self.extract_temporal_features(spatial_features)
        features.extend(temporal_features)

        # Statistical features across sequence
        statistical_features = self.extract_statistical_features(spatial_features)
        features.extend(statistical_features)

        return np.array(features)

    def get_valid_frames(self, sequence):
        """Extract non-zero padded frames"""
        valid_frames = []
        for frame in sequence:
            # Check if frame has meaningful data (not all zeros)
            if np.any(frame != 0):
                valid_frames.append(frame)
        return np.array(valid_frames)

    def extract_spatial_features(self, skeleton_frame):
        """Extract spatial features from a single frame"""
        features = []

        # 1. Flatten joint coordinates
        features.extend(skeleton_frame.flatten())

        # 2. Bone vectors between connected joints - WITH SAFETY CHECKS
        for j1, j2 in self.bone_pairs:
            if (j1 < len(skeleton_frame) and j2 < len(skeleton_frame) and
                np.any(skeleton_frame[j1] != 0) and np.any(skeleton_frame[j2] != 0)):
                
                bone_vector = skeleton_frame[j2] - skeleton_frame[j1]
                features.extend(bone_vector)
                # Bone length
                bone_length = np.linalg.norm(bone_vector)
                features.append(bone_length)
            else:
                # Pad with zeros for missing bones
                features.extend([0, 0, 0, 0])

        # 3. Key joint distances from spine base (joint 0) 
        key_joints = [3, 7, 11, 15, 19]  # Head, left_hand, right_hand, left_foot, right_foot
        spine_base = skeleton_frame[0]
        for joint_idx in key_joints:
            if (joint_idx < len(skeleton_frame) and 
                np.any(skeleton_frame[joint_idx] != 0) and 
                np.any(spine_base != 0)):
                
                distance_vec = skeleton_frame[joint_idx] - spine_base
                features.extend(distance_vec)
                features.append(np.linalg.norm(distance_vec))
            else:
                features.extend([0, 0, 0, 0])

        # 4. Joint angles for major limbs 
        angles = self.calculate_joint_angles(skeleton_frame)
        features.extend(angles)

        return np.array(features)

    def calculate_joint_angles(self, skeleton):
        """Calculate joint angles for major limbs """
        angles = []

        # Left elbow angle (shoulder-elbow-wrist)
        # Joints: left_shoulder(4), left_elbow(5), left_wrist(6)
        if (len(skeleton) > 6 and 
            np.any(skeleton[4] != 0) and np.any(skeleton[5] != 0) and np.any(skeleton[6] != 0)):
            left_shoulder = skeleton[4]    
            left_elbow = skeleton[5]        
            left_wrist = skeleton[6]       
            angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            angles.append(angle)
        else:
            angles.append(0.0)

        # Right elbow angle  
        # Joints: right_shoulder(8), right_elbow(9), right_wrist(10)
        if (len(skeleton) > 10 and 
            np.any(skeleton[8] != 0) and np.any(skeleton[9] != 0) and np.any(skeleton[10] != 0)):
            right_shoulder = skeleton[8]  
            right_elbow = skeleton[9]      
            right_wrist = skeleton[10]     
            angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            angles.append(angle)
        else:
            angles.append(0.0)

        # Left knee angle
        # Joints: left_hip(12), left_knee(13), left_ankle(14)
        if (len(skeleton) > 14 and 
            np.any(skeleton[12] != 0) and np.any(skeleton[13] != 0) and np.any(skeleton[14] != 0)):
            left_hip = skeleton[12]        
            left_knee = skeleton[13]       
            left_ankle = skeleton[14]     
            angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            angles.append(angle)
        else:
            angles.append(0.0)

        # Right knee angle
        # Joints: right_hip(16), right_knee(17), right_ankle(18)
        if (len(skeleton) > 18 and 
            np.any(skeleton[16] != 0) and np.any(skeleton[17] != 0) and np.any(skeleton[18] != 0)):
            right_hip = skeleton[16]      
            right_knee = skeleton[17]      
            right_ankle = skeleton[18]     
            angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            angles.append(angle)
        else:
            angles.append(0.0)

        return angles

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        vec1 = point1 - point2
        vec2 = point3 - point2

        if (np.linalg.norm(vec1) > 0.001 and np.linalg.norm(vec2) > 0.001 and
            np.any(vec1 != 0) and np.any(vec2 != 0)):
            
            cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cosine_angle = np.clip(cosine_angle, -1, 1)
            return np.arccos(cosine_angle)
        return 0.0

    def extract_temporal_features(self, frame_features):
        """Extract temporal features across frames"""
        features = []

        if len(frame_features) > 1:
            # Velocities (frame-to-frame differences)
            velocities = np.diff(frame_features, axis=0)

            # Mean velocity
            if len(velocities) > 0:
                mean_velocity = np.mean(velocities, axis=0)
                features.extend(mean_velocity)

                # Velocity statistics
                velocity_magnitudes = np.linalg.norm(velocities, axis=1)
                features.append(np.mean(velocity_magnitudes))
                features.append(np.std(velocity_magnitudes))
                features.append(np.max(velocity_magnitudes))

        return features

    def extract_statistical_features(self, frame_features):
        """Extract statistical features across sequence"""
        features = []

        if len(frame_features) > 0:
            # Mean positions
            mean_positions = np.mean(frame_features, axis=0)
            features.extend(mean_positions)

            # Standard deviation
            std_positions = np.std(frame_features, axis=0)
            features.extend(std_positions)

            # Range (max-min)
            range_positions = np.ptp(frame_features, axis=0)
            features.extend(range_positions)

            # Motion intensity (overall variance)
            overall_variance = np.mean(std_positions)
            features.append(overall_variance)

        return features