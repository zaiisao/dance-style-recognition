import numpy as np
from scipy.spatial import ConvexHull


class LMAExtractor:
    def __init__(self, window_size=55, fps=60):
        """
        Laban Movement Analysis Feature Extractor.
        Faithfully implements the 55-feature vector described in Turab et al. (2025),
        incorporating specific lag-based Space metrics and threshold-based Initiation.
        """
        self.window_size = window_size
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

        # Standard SMPL 24-joint topology
        self.IDX = {
            "PELVIS": 0,
            "L_HIP": 1,
            "R_HIP": 2,
            "SPINE1": 3,
            "L_KNEE": 4,
            "R_KNEE": 5,
            "SPINE2": 6,
            "L_ANKLE": 7,
            "R_ANKLE": 8,
            "SPINE3": 9,
            "L_FOOT": 10,
            "R_FOOT": 11,
            "NECK": 12,
            "L_COLLAR": 13,
            "R_COLLAR": 14,
            "HEAD": 15,
            "L_SHOULDER": 16,
            "R_SHOULDER": 17,
            "L_ELBOW": 18,
            "R_ELBOW": 19,
            "L_WRIST": 20,
            "R_WRIST": 21,
            "L_HAND": 22,
            "R_HAND": 23,
        }

        # [cite_start]The 6 Key Joints identified from SHAP plots and Effort descriptions [cite: 183-191]
        self.KEY_JOINTS = ["HEAD", "PELVIS", "L_WRIST", "R_WRIST", "L_ANKLE", "R_ANKLE"]

        # Weights for Global Sums (extremities get higher weight)
        self.weights = {k: 1.0 for k in self.KEY_JOINTS}
        for k in ["L_WRIST", "R_WRIST", "L_ANKLE", "R_ANKLE"]:
            self.weights[k] = 1.5

    def _impute_missing_data(self, joint_seq):
        """Linearly interpolates missing frames to ensure continuity."""
        n_frames = len(joint_seq)
        valid_indices = [i for i, x in enumerate(joint_seq) if len(x) > 0]

        if not valid_indices:
            return np.zeros((n_frames, 24, 3))

        full_seq = np.zeros((n_frames, 24, 3))

        # Fill known values
        for i in valid_indices:
            full_seq[i] = joint_seq[i]

        # Interpolate gaps
        for j in range(24):
            for c in range(3):
                vals = full_seq[valid_indices, j, c]
                full_seq[:, j, c] = np.interp(range(n_frames), valid_indices, vals)
        return full_seq

    def _normalize_pose_to_floor(self, joints, floor_models):
        """
        Converts Camera Space -> Floor-Relative Height.
        [cite_start]Crucial for 'Floor Aware Body Modeling'[cite: 83].
        """
        normalized = np.copy(joints)
        n_frames = len(joints)

        for i in range(n_frames):
            z_vals = joints[i, :, 2].reshape(-1, 1)
            try:
                floor_y = floor_models[i].predict(z_vals)
            except Exception:
                # Fallback: assume floor is 1 meter below root if model fails
                floor_y = joints[i, :, 1] + 1.0

            # Y-down coordinate system assumption (common in OpenCV/SMPL)
            normalized[i, :, 1] = floor_y - joints[i, :, 1]

        return normalized

    def extract_all_features(self, all_joints, all_volumes, all_floor_models):
        """
        Extracts the 55 LMA features with corrected Equation 1 & 2 logic.
        """
        # 1. Preprocessing
        cleaned_joints = self._impute_missing_data(all_joints)
        norm_joints = self._normalize_pose_to_floor(cleaned_joints, all_floor_models)

        # 2. Derivatives (Full Sequence)
        vel = np.gradient(norm_joints, self.dt, axis=0)
        acc = np.gradient(vel, self.dt, axis=0)
        jerk = np.gradient(acc, self.dt, axis=0)  # Required for 'Flow'

        n_frames = len(all_joints)
        w_main = self.window_size

        # --- PRE-CALCULATE INITIATION THRESHOLDS (Equation 1 Correction) ---
        # [cite_start]"Threshold calculated using standard-deviation of the entire sequence" [cite: 98]
        # We calculate the raw initiation metric for the whole video first.
        w_init = 5  # Short window for initiation
        init_thresholds = {}
        raw_init_values = {}

        for name in self.KEY_JOINTS:
            idx = self.IDX[name]
            raw_vals = []
            for t in range(n_frames - w_init):
                # Raw metric: ||P(t+w) - P(t)|| / dt
                delta = norm_joints[t + w_init, idx] - norm_joints[t, idx]
                val = np.linalg.norm(delta) / (w_init * self.dt)
                raw_vals.append(val)

            # Convert to array and compute std dev for threshold
            raw_vals = np.array(raw_vals)
            if len(raw_vals) > 0:
                sigma = np.std(raw_vals)
                # If sigma is tiny (static), use a small epsilon to avoid noise triggers
                init_thresholds[name] = max(sigma, 1e-3)
                # Pad to match n_frames length
                padded = np.zeros(n_frames)
                padded[: len(raw_vals)] = raw_vals
                raw_init_values[name] = padded
            else:
                init_thresholds[name] = 1.0
                raw_init_values[name] = np.zeros(n_frames)

        # Initialize Dictionary
        feats = {k: np.zeros(n_frames) for k in ["weight", "time", "flow", "space", "body_volume"]}

        # 3. Frame-by-Frame Extraction
        for t in range(n_frames):
            # Causal Window: [t-w+1 : t+1]
            start = max(0, t - w_main + 1)
            end = t + 1
            curr_pose = norm_joints[t]

            # ---------------------------------------------------------
            # COMPONENT 1: RAW KINEMATICS (6 Features)
            #
            # ---------------------------------------------------------
            for name in self.KEY_JOINTS:
                idx = self.IDX[name]
                v_mag = np.mean(np.linalg.norm(vel[start:end, idx, :], axis=1))
                self._add_feat(feats, f"{name}_vel", v_mag, t)

            # ---------------------------------------------------------
            # COMPONENT 2: EFFORT (28 Features)
            # ---------------------------------------------------------
            sums = {"Weight": 0, "Time": 0, "Flow": 0, "Space": 0}

            for name in self.KEY_JOINTS:
                idx = self.IDX[name]
                wt = self.weights[name]

                # A. Weight (Kinetic Energy) [Eq 4]
                v_sq = np.sum(vel[start:end, idx, :] ** 2, axis=1)
                ke = np.mean(0.5 * v_sq)
                self._add_feat(feats, f"{name}_KE", ke, t)
                sums["Weight"] += ke * wt

                # B. Time (Acceleration) [Eq 5]
                a_mag = np.mean(np.linalg.norm(acc[start:end, idx, :], axis=1))
                self._add_feat(feats, f"{name}_Accel", a_mag, t)
                sums["Time"] += a_mag * wt

                # C. Flow (Jerkiness)
                j_mag = np.mean(np.linalg.norm(jerk[start:end, idx, :], axis=1))
                self._add_feat(feats, f"{name}_Jerk", j_mag, t)
                sums["Flow"] += j_mag * wt

                # D. Space (Lagged Directness) [Eq 2 Correction]
                # [cite_start]"Sum of distances... ||P(t) - P(t-w)||... for a short time-window" [cite: 104-105]
                # We interpret 'w' as a lag parameter distinct from window T.
                # If w > 1, this measures volumetric "reach" rather than arc length.
                w_lag = 5  # Lag of ~0.08s (5 frames @ 60fps)

                traj = norm_joints[start:end, idx, :]  # Shape (Window_Len, 3)
                win_len = len(traj)

                numerator = 0.0
                if win_len > w_lag:
                    # Sum ||P(i) - P(i-w)||
                    for i in range(w_lag, win_len):
                        dist_lag = np.linalg.norm(traj[i] - traj[i - w_lag])
                        numerator += dist_lag
                else:
                    # Fallback for very first frames
                    numerator = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))

                # Denominator: ||P(T) - P(t1)|| (Displacement of the whole window)
                disp = np.linalg.norm(traj[-1] - traj[0])

                # Compute Space Metric
                space_val = numerator / (disp + 1e-6)
                if space_val < 1.0:
                    space_val = 1.0

                self._add_feat(feats, f"{name}_Directness", space_val, t)
                sums["Space"] += space_val * wt

            self._add_feat(feats, "Effort_Weight_Global", sums["Weight"], t)
            feats["weight"][t] = sums["Weight"]

            self._add_feat(feats, "Effort_Time_Global", sums["Time"], t)
            feats["time"][t] = sums["Time"]

            self._add_feat(feats, "Effort_Flow_Global", sums["Flow"], t)
            feats["flow"][t] = sums["Flow"]

            self._add_feat(feats, "Effort_Space_Global", sums["Space"], t)
            feats["space"][t] = sums["Space"]

            # ---------------------------------------------------------
            # COMPONENT 3: SPACE (8 Features)
            # ---------------------------------------------------------
            def dist(k1, k2):
                return np.linalg.norm(curr_pose[self.IDX[k1]] - curr_pose[self.IDX[k2]])

            self._add_feat(feats, "Dispersion_Head", dist("HEAD", "SPINE2"), t)
            self._add_feat(feats, "Dispersion_R_Wrist", dist("R_WRIST", "SPINE2"), t)
            self._add_feat(feats, "Dispersion_L_Wrist", dist("L_WRIST", "SPINE2"), t)
            self._add_feat(feats, "Dispersion_R_Ankle", dist("R_ANKLE", "PELVIS"), t)
            self._add_feat(feats, "Dispersion_L_Ankle", dist("L_ANKLE", "PELVIS"), t)

            root_traj = norm_joints[start:end, self.IDX["PELVIS"], :]
            total_path = np.sum(np.linalg.norm(np.diff(root_traj, axis=0), axis=1))
            total_disp = np.linalg.norm(root_traj[-1] - root_traj[0])
            curvature = total_path / (total_disp + 1e-6)

            self._add_feat(feats, "Traj_Path_Length", total_path, t)
            self._add_feat(feats, "Traj_Displacement", total_disp, t)
            self._add_feat(feats, "Traj_Curvature", curvature, t)

            # ---------------------------------------------------------
            # COMPONENT 4: SHAPE (1 Feature)
            # ---------------------------------------------------------
            feats["body_volume"][t] = all_volumes[t]

            # ---------------------------------------------------------
            # COMPONENT 5: BODY (12 Features)
            # ---------------------------------------------------------
            # A. Distances
            self._add_feat(feats, "Dist_Hand_Shoulder_L", dist("L_WRIST", "L_SHOULDER"), t)
            self._add_feat(feats, "Dist_Hand_Shoulder_R", dist("R_WRIST", "R_SHOULDER"), t)
            self._add_feat(feats, "Dist_Ankle_Knee_L", dist("L_ANKLE", "L_KNEE"), t)
            self._add_feat(feats, "Dist_Ankle_Knee_R", dist("R_ANKLE", "R_KNEE"), t)
            self._add_feat(feats, "Dist_Hands", dist("L_WRIST", "R_WRIST"), t)
            self._add_feat(feats, "Dist_Feet", dist("L_ANKLE", "R_ANKLE"), t)

            # B. Initiation (Eq 1 Correction)
            # [cite_start]"Initiation(t) ... > epsilon" [cite: 96]
            # Implements the detection event logic rather than continuous value.
            for name in self.KEY_JOINTS:
                raw_val = raw_init_values[name][t]
                threshold = init_thresholds[name]

                # Boolean Thresholding (Detection Event)
                if raw_val > threshold:
                    init_feat = 1.0
                else:
                    init_feat = 0.0

                self._add_feat(feats, f"Initiation_{name}", init_feat, t)

        return feats

    def _add_feat(self, feat_dict, key, val, t):
        if key not in feat_dict:
            feat_dict[key] = np.zeros(len(feat_dict["weight"]))
        feat_dict[key][t] = val