import numpy as np

class LMAExtractor:
    def __init__(self, fps=30.0, window_size=5):
        """
        Implements Turab et al.'s Feature Extraction.
        window_size: 'w' from Eq 5 (Sliding Window for smoothing derivatives).
        """
        self.fps = fps
        self.window = window_size
        
        # SMPL Indices (Standard mapping)
        self.IDX = {
            'pelvis': 0, 'head': 15,
            'l_hand': 20, 'r_hand': 21, 
            'l_foot': 7, 'r_foot': 8
        }
        self.effectors = [self.IDX['l_hand'], self.IDX['r_hand'], self.IDX['l_foot'], self.IDX['r_foot']]

    def _clean_input_data(self, joints_list, volumes_list):
        """
        Converts messy list of joints into a clean (T, 24, 3) numpy array.
        Handles None, [], and shape mismatches.
        """
        clean_joints = []
        clean_volumes = []
        
        # Standard Shape: (24 joints, 3 coords)
        target_shape = (24, 3)
        
        for j, v in zip(joints_list, volumes_list):
            # 1. Handle Volumes
            if v is None or np.isnan(v):
                clean_volumes.append(0.0)
            else:
                clean_volumes.append(float(v))
                
            # 2. Handle Joints
            if j is None or len(j) == 0:
                # Missing frame -> Fill with Zeros (or previous frame if you prefer)
                clean_joints.append(np.zeros(target_shape))
            else:
                j_np = np.array(j)
                # Fix dimensions: (1, 24, 3) -> (24, 3)
                if j_np.ndim == 3: 
                    j_np = j_np[0]
                
                if j_np.shape == target_shape:
                    clean_joints.append(j_np)
                else:
                    # Shape mismatch fallback
                    clean_joints.append(np.zeros(target_shape))
                    
        return np.array(clean_joints), np.array(clean_volumes)

    def _get_window_derivative(self, data):
        """
        Calculates derivative using sliding window (Paper Eq. 5).
        Returns zeros for the first/last 'w' frames (padding).
        """
        T = len(data)
        deriv = np.zeros_like(data)
        w = self.window
        
        # Vectorized implementation of Eq 5: (x[t+w] - x[t-w]) / 2w
        # This acts as a Low-Pass Filter, naturally killing high-freq jitter.
        # We only compute where the window fits (w to T-w)
        if T > 2 * w:
            for t in range(w, T - w):
                deriv[t] = (data[t + w] - data[t - w]) / (2 * w)
            
        return deriv

    def extract_all_features(self, all_joints, all_volumes, floor_params=None):
        """
        Main extraction pipeline.
        all_joints: List of arrays or (T, 24, 3)
        all_volumes: List of floats or (T,)
        """
        # 1. SANITIZE DATA (The Fix for ValueError)
        joints_seq, volumes_seq = self._clean_input_data(all_joints, all_volumes)
        
        features = {}
        
        # --- 2. EFFORT (Dynamics) ---
        # Calculate Velocity (1st Derivative)
        vel = self._get_window_derivative(joints_seq) # (T, 24, 3)
        speed = np.linalg.norm(vel, axis=2) # Magnitude
        
        # Calculate Acceleration (2nd Derivative)
        acc = self._get_window_derivative(vel) # (T, 24, 3)
        jerk = self._get_window_derivative(acc)
        acc_mag = np.linalg.norm(acc, axis=2)
        jerk_mag = np.linalg.norm(jerk, axis=2)
        
        # Feature: WEIGHT (Strong vs Light) -> Kinetic Energy
        # "Strong" movements have high impact (high speed/energy)
        features['weight'] = np.mean(speed[:, self.effectors]**2, axis=1)
        
        # Feature: TIME (Sudden vs Sustained) -> Acceleration Magnitude
        # "Sudden" movements have high acceleration spikes
        features['time'] = np.mean(acc_mag[:, self.effectors], axis=1)

        # Feature: FLOW (Free vs Bound) -> Mean of Jerk
        features['flow'] = np.mean(jerk_mag[:, self.effectors], axis=1)

        # --- 3. SHAPE (Volume) ---
        # Feature: SHAPE FLOW (Grow vs Shrink)
        # Derivative of volume: + = Growing, - = Shrinking
        features['shape'] = self._get_window_derivative(volumes_seq)

        # --- 4. SPACE (Direct vs Indirect) ---
        # "Direct" = Straight line paths. "Indirect" = Wavy paths.
        w = self.window
        space_score = np.ones(len(joints_seq))
        
        if len(joints_seq) > 2 * w:
            for t in range(w, len(joints_seq) - w):
                # Path for Right Hand
                hand_pos = joints_seq[t-w : t+w, self.IDX['r_hand']]
                
                # Actual Path Length (sum of step distances)
                steps = np.diff(hand_pos, axis=0)
                path_len = np.sum(np.linalg.norm(steps, axis=1))
                
                # Linear Displacement (Start to End)
                displacement = np.linalg.norm(hand_pos[-1] - hand_pos[0])
                
                # Ratio: 1.0 = Direct, >1.0 = Indirect (Wavy)
                if displacement > 0.01:
                    space_score[t] = path_len / displacement
                else:
                    space_score[t] = 1.0 # Stationary is Direct
                
        features['space'] = space_score

        # Return a dictionary of 1D arrays
        # (Do NOT return np.array(features) because 'features' is a dict)
        return features