import numpy as np

class NTUProcessedFeatureExtractor:
    def __init__(self):
        # Bone pairs consistent with NTU indexes
        self.bone_pairs = [
            (0, 1), (1, 20), (20, 2), (2, 3),
            (2, 4), (4, 5), (5, 6), (6, 7), (6, 21), (6, 22),
            (2, 8), (8, 9), (9, 10), (10, 11), (10, 23), (10, 24),
            (0, 12), (12, 13), (13, 14), (14, 15),
            (0, 16), (16, 17), (17, 18), (18, 19)
        ]

        self.hand_joints = [7, 21, 22, 11, 23, 24]
        self.arm_joints = [4, 5, 6, 8, 9, 10]
        self.leg_joints = [12, 13, 14, 15, 16, 17, 18, 19]

        self.extraction_stats = {
            'total_sequences': 0,
            'successful_extractions': 0,
            'failed_min_frames': 0,
            'failed_empty_features': 0
        }

    def extract_features_from_sequence(self, sequence):
        self.extraction_stats['total_sequences'] += 1

        try:
            reshaped = sequence.reshape(300, 2, 25, 3)
        except Exception:
            return np.zeros(100, dtype=np.float32)

        main_person_seq = reshaped[:, 0, :, :]  # (300, 25, 3)

        valid_frames = self.get_valid_frames(main_person_seq)

        if len(valid_frames) < 5:
            self.extraction_stats['failed_min_frames'] += 1
            return self.extract_minimal_features(valid_frames)

        features = []
        spatial = self.extract_robust_spatial_features(valid_frames)
        features.extend(spatial)
        temporal = self.extract_robust_temporal_features(valid_frames)
        features.extend(temporal)
        statistical = self.extract_robust_statistical_features(valid_frames)
        features.extend(statistical)
        action = self.extract_robust_action_features(valid_frames)
        features.extend(action)

        try:
            delta = (valid_frames[-1] - valid_frames[0]).flatten().tolist()
            features.extend(delta)
        except Exception:
            pass

        features_array = np.array(features, dtype=np.float32)
        if len(features_array) == 0 or np.all(features_array == 0):
            self.extraction_stats['failed_empty_features'] += 1
            return self.extract_minimal_features(valid_frames)

        self.extraction_stats['successful_extractions'] += 1
        return features_array

    def extract_minimal_features(self, valid_frames):
        features = []
        if len(valid_frames) == 0:
            return np.zeros(100, dtype=np.float32)

        for i in range(min(3, len(valid_frames))):
            frame = valid_frames[i]
            features.extend(frame.flatten().tolist())
            for j1, j2 in self.bone_pairs[:5]:
                if (j1 < len(frame) and j2 < len(frame)):
                    bone_vector = frame[j2] - frame[j1]
                    features.extend(bone_vector[:2])
                    features.append(np.linalg.norm(bone_vector))

        if len(features) < 100:
            features.extend([0.0] * (100 - len(features)))
        else:
            features = features[:100]

        return np.array(features, dtype=np.float32)

    def extract_robust_spatial_features(self, valid_frames):
        features = []
        frame_indices = [0, len(valid_frames)//2, -1]
        used_frames = 0
        for idx in frame_indices:
            if idx < len(valid_frames):
                frame_features = self.extract_spatial_features(valid_frames[idx])
                features.extend(frame_features)
                used_frames += 1
        if used_frames == 1 and len(features) > 0:
            features.extend(features)
        return features

    def extract_spatial_features(self, skeleton_frame):
        features = []
        try:
            features.extend(skeleton_frame.flatten().tolist())
        except Exception:
            features.extend([0.0] * (25*3))

        for j1, j2 in self.bone_pairs:
            if (j1 < len(skeleton_frame) and j2 < len(skeleton_frame) and
                np.any(skeleton_frame[j1] != 0) and np.any(skeleton_frame[j2] != 0)):
                bone_vector = skeleton_frame[j2] - skeleton_frame[j1]
                features.extend(bone_vector.tolist())
                bone_length = np.linalg.norm(bone_vector)
                try:
                    spine_length = np.linalg.norm(skeleton_frame[2] - skeleton_frame[1])
                    if spine_length > 1e-2:
                        features.append(bone_length / spine_length)
                    else:
                        features.append(bone_length)
                except Exception:
                    features.append(bone_length)
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        key_joints = [3, 7, 11, 15, 19]
        spine_base = skeleton_frame[0]
        for joint_idx in key_joints:
            if joint_idx < len(skeleton_frame) and np.any(skeleton_frame[joint_idx] != 0) and np.any(spine_base != 0):
                distance_vec = skeleton_frame[joint_idx] - spine_base
                features.extend(distance_vec.tolist())
                features.append(np.linalg.norm(distance_vec))
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        angles = self.calculate_joint_angles(skeleton_frame)
        features.extend(angles)

        return features

    def extract_robust_temporal_features(self, valid_frames):
        features = []
        try:
            if len(valid_frames) < 2:
                return [0.0] * 14

            velocities = np.diff(valid_frames, axis=0)  # (T-1, J, 3)
            accelerations = np.diff(velocities, axis=0) if velocities.shape[0] > 1 else np.zeros((0,) + velocities.shape[1:])

            def group_stats(joint_list):
                if velocities.shape[0] == 0:
                    return [0.0, 0.0]
                group_vel = velocities[:, joint_list, :]
                norms = np.linalg.norm(group_vel, axis=2)
                flat = norms.flatten()
                meanv = float(np.nan_to_num(np.mean(flat), nan=0.0))
                stdv = float(np.nan_to_num(np.std(flat), nan=0.0))
                return [meanv, stdv]

            hv_mean, hv_std = group_stats(self.hand_joints)
            av_mean, av_std = group_stats(self.arm_joints)
            lv_mean, lv_std = group_stats(self.leg_joints)
            features.extend([hv_mean, hv_std, av_mean, av_std, lv_mean, lv_std])

            if accelerations.shape[0] > 0:
                acc_norms = np.linalg.norm(accelerations, axis=2).flatten()
                acc_mean = float(np.nan_to_num(np.mean(acc_norms), nan=0.0))
                acc_std = float(np.nan_to_num(np.std(acc_norms), nan=0.0))
            else:
                acc_mean = 0.0
                acc_std = 0.0
            features.extend([acc_mean, acc_std])

            try:
                if len(velocities) > 1:
                    motion_smoothness = float(np.nan_to_num(np.mean(np.linalg.norm(np.diff(velocities, axis=0), axis=2)), nan=0.0))
                else:
                    motion_smoothness = 0.0
            except:
                motion_smoothness = 0.0
            features.append(motion_smoothness)

            periodicity = self.calculate_simple_periodicity(valid_frames)
            features.extend(periodicity)

            if len(features) < 14:
                features.extend([0.0] * (14 - len(features)))

        except Exception:
            features = [0.0] * 14

        return features

    def calculate_simple_periodicity(self, valid_frames):
        if len(valid_frames) < 3:
            return [0.0, 0.0]
        try:
            torso_joints = [0, 1, 2, 20]
            torso_positions = valid_frames[:, torso_joints, :]
            torso_motion = np.linalg.norm(torso_positions, axis=2).mean(axis=1)
            if len(torso_motion) > 1:
                motion_variance = np.var(torso_motion)
                if len(torso_motion) > 2:
                    autocorr = np.corrcoef(torso_motion[:-1], torso_motion[1:])[0,1]
                    if np.isnan(autocorr):
                        autocorr = 0.0
                else:
                    autocorr = 0.0
            else:
                motion_variance = 0.0
                autocorr = 0.0
            return [autocorr, motion_variance]
        except:
            return [0.0, 0.0]

    def extract_robust_statistical_features(self, valid_frames):
        features = []
        if len(valid_frames) == 0:
            return [0.0] * 30
        try:
            major_joints = [0, 2, 4, 5, 6, 8, 9, 10, 12, 13, 16, 17]
            for joint_idx in major_joints:
                if joint_idx < valid_frames.shape[1]:
                    joint_data = valid_frames[:, joint_idx, :]
                    if np.any(joint_data != 0):
                        features.extend(np.mean(joint_data, axis=0).tolist())
                        features.extend(np.std(joint_data, axis=0).tolist())
                    else:
                        features.extend([0.0] * 6)
            if len(valid_frames) > 1:
                try:
                    major_joint_data = valid_frames[:, major_joints, :]
                    frame_centers = np.mean(major_joint_data, axis=1)
                    center_movement = np.diff(frame_centers, axis=0)
                    if len(center_movement) > 0:
                        movement_magnitudes = np.linalg.norm(center_movement, axis=1)
                        features.append(np.mean(movement_magnitudes))
                    else:
                        features.append(0.0)
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
        except Exception:
            features.extend([0.0] * (30 - len(features)))
        return features

    def extract_robust_action_features(self, valid_frames):
        features = []
        try:
            hand_features = self.extract_simple_hand_features(valid_frames)
            features.extend(hand_features)
            motion_ratio = self.extract_simple_motion_ratios(valid_frames)
            features.extend(motion_ratio)
            symmetry = self.extract_simple_symmetry(valid_frames)
            features.extend(symmetry)
        except:
            features.extend([0.0] * 15)
        return features

    def extract_simple_hand_features(self, valid_frames):
        if len(valid_frames) < 2:
            return [0.0] * 3
        features = []
        try:
            first_frame = valid_frames[0]
            last_frame = valid_frames[-1]
            hand_distances = []
            for frame in [first_frame, last_frame]:
                if (len(frame) > 11 and np.any(frame[7] != 0) and np.any(frame[11] != 0)):
                    distance = np.linalg.norm(frame[7] - frame[11])
                    hand_distances.append(distance)
            if hand_distances:
                features.append(np.mean(hand_distances))
                features.append(np.min(hand_distances))
            else:
                features.extend([0.0, 0.0])
            if len(valid_frames) > 1:
                hand_positions = valid_frames[:, [7, 11], :]
                hand_motion = np.mean(np.linalg.norm(np.diff(hand_positions, axis=0), axis=2))
                features.append(hand_motion)
            else:
                features.append(0.0)
        except:
            features.extend([0.0] * (3 - len(features)))
        return features

    def extract_simple_motion_ratios(self, valid_frames):
        if len(valid_frames) < 2:
            return [0.0, 0.0]
        try:
            upper_body_joints = [2, 4, 5, 6, 8, 9, 10]
            lower_body_joints = [0, 12, 13, 16, 17]
            velocities = np.diff(valid_frames, axis=0)
            if len(velocities) > 0:
                upper_vel = velocities[:, upper_body_joints, :]
                lower_vel = velocities[:, lower_body_joints, :]
                upper_motion = np.mean(np.linalg.norm(upper_vel, axis=2)) if upper_vel.size else 0.0
                lower_motion = np.mean(np.linalg.norm(lower_vel, axis=2)) if lower_vel.size else 0.0
                if lower_motion > 0.001:
                    motion_ratio = upper_motion / lower_motion
                else:
                    motion_ratio = upper_motion * 10
            else:
                upper_motion = 0.0
                motion_ratio = 0.0
            return [motion_ratio, upper_motion]
        except:
            return [0.0, 0.0]

    def extract_simple_symmetry(self, valid_frames):
        if len(valid_frames) < 2:
            return [0.0, 0.0]
        try:
            left_joints = [4, 5, 6, 12, 13]
            right_joints = [8, 9, 10, 16, 17]
            left_positions = valid_frames[:, left_joints, :].mean(axis=1)
            right_positions = valid_frames[:, right_joints, :].mean(axis=1)
            position_differences = np.linalg.norm(left_positions - right_positions, axis=1)
            symmetry_score = 1.0 - np.mean(position_differences) / 2.0
            symmetry_score = np.clip(symmetry_score, 0.0, 1.0)
            return [symmetry_score, np.mean(position_differences)]
        except:
            return [0.0, 0.0]

    def get_valid_frames(self, sequence):
        frame_has_data = np.any(sequence != 0, axis=(1,2))
        valid_indices = np.where(frame_has_data)[0]
        if len(valid_indices) == 0:
            return sequence[:5]
        return sequence[valid_indices]

    def calculate_joint_angles(self, skeleton):
        angles = []
        try:
            if (len(skeleton) > 6 and np.any(skeleton[4] != 0) and np.any(skeleton[5] != 0) and np.any(skeleton[6] != 0)):
                left_shoulder = skeleton[4]
                left_elbow = skeleton[5]
                left_wrist = skeleton[6]
                angles.append(self.calculate_angle(left_shoulder, left_elbow, left_wrist))
            else:
                angles.append(0.0)

            if (len(skeleton) > 10 and np.any(skeleton[8] != 0) and np.any(skeleton[9] != 0) and np.any(skeleton[10] != 0)):
                right_shoulder = skeleton[8]
                right_elbow = skeleton[9]
                right_wrist = skeleton[10]
                angles.append(self.calculate_angle(right_shoulder, right_elbow, right_wrist))
            else:
                angles.append(0.0)

            if (len(skeleton) > 2 and np.any(skeleton[0] != 0) and np.any(skeleton[20] != 0) and np.any(skeleton[2] != 0)):
                angles.append(self.calculate_angle(skeleton[0], skeleton[20], skeleton[2]))
            else:
                angles.append(0.0)

        except:
            angles = angles + [0.0] * (3 - len(angles))
        return angles

    def calculate_angle(self, point1, point2, point3):
        vec1 = point1 - point2
        vec2 = point3 - point2
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 1e-6 and norm2 > 1e-6:
            cosine_angle = np.dot(vec1, vec2) / (norm1 * norm2)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            return float(np.arccos(cosine_angle))
        return 0.0
