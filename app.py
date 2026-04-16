import os
import time
import numpy as np
import pandas as pd
from collections import Counter
from stable_baselines3 import PPO
from src.fabricator import EEGFabricator

class AppInterfaceManager:
    def __init__(self, model_root="./models/", results_csv="./results/loso_results.csv", threshold=0.90, channels=4, sfreq=250):
        self.fabricator = EEGFabricator()
        self.labels = {0: "Low", 1: "Medium", 2: "High"}
        self.ensemble = []
        self.weights = [] 
        
        # Buffer Configuration
        self.sfreq = sfreq
        self.window_size = int(sfreq * 2) 
        self.channels = channels
        self.buffer = np.zeros((self.channels, self.window_size))
        self.points_collected = 0

        # 1. Load Performance Data for Weighting
        print(f"🔍 Filtering models with Accuracy >= {threshold}...")
        try:
            df = pd.read_csv(results_csv)
            champions_df = df[df['Accuracy'] >= threshold]
            # Create a map of Subject ID -> Accuracy for weighting
            champion_map = dict(zip(champions_df['Subject'].astype(str), champions_df['Accuracy']))
        except Exception as e:
            print(f"⚠️ Could not read CSV ({e}). Falling back to uniform weights.")
            champion_map = None

        # 2. Dynamically load models
        for root, dirs, files in os.walk(model_root):
            if "best_model.zip" in files:
                sid = os.path.basename(root)
                
                if champion_map is None or sid in champion_map:
                    model_path = os.path.join(root, "best_model.zip")
                    print(f"📦 Loading Champion: Subject {sid}")
                    self.ensemble.append(PPO.load(model_path))
                    
                    # Weighting: Performance squared gives experts exponentially more power
                    acc = champion_map.get(sid, 1.0) if champion_map else 1.0
                    self.weights.append(acc ** 2) 

        if not self.ensemble:
            print("❌ WARNING: No models met the threshold! Ensemble is empty.")

    def update_buffer(self, new_samples):
        """Slides the window to incorporate fresh EEG data."""
        n = new_samples.shape[1]
        self.buffer = np.roll(self.buffer, -n, axis=1)
        self.buffer[:, -n:] = new_samples
        self.points_collected += n

    def _calibrate_confidence(self, weighted_votes_dict, temperature=0.6):
        """
        Applies Temperature Scaling to the ensemble votes.
        Temperature < 1.0 'sharpens' the consensus.
        """
        # Convert dict to array [Low, Med, High]
        logits = np.array([weighted_votes_dict[i] for i in range(3)], dtype=np.float32)
        
        # Apply scaling
        scaled_logits = logits / temperature
        
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return np.max(probs), probs

    def predict(self, rejection_threshold=0.45, temperature=0.6):
        if self.points_collected < self.window_size:
            return {"status": "buffering", "progress": f"{self.points_collected}/{self.window_size}"}
        
        obs = self.fabricator.extract_features(self.buffer)
        
        # 1. Collect Weighted Votes
        weighted_votes = {0: 0.0, 1: 0.0, 2: 0.0}
        raw_counts = {0: 0, 1: 0, 2: 0}
        
        for model, weight in zip(self.ensemble, self.weights):
            action = int(model.predict(obs, deterministic=True)[0])
            weighted_votes[action] += weight
            raw_counts[action] += 1
        
        # 2. Apply Calibration (Temperature Scaling)
        # This replaces the old simple ratio confidence
        calibrated_conf, all_probs = self._calibrate_confidence(weighted_votes, temperature)
        
        # 3. Determine Winner
        final_action = np.argmax(all_probs)

        # 4. Apply Rejection Logic
        # Now uses the calibrated confidence for a more stable decision
        is_uncertain = calibrated_conf < rejection_threshold
        display_label = "Uncertain" if is_uncertain else self.labels[final_action]

        return {
            "status": "success",
            "is_uncertain": is_uncertain,
            "workload_level": -1 if is_uncertain else final_action,
            "label": display_label,
            "confidence": calibrated_conf,
            "vote_map": {self.labels[k]: v for k, v in raw_counts.items()},
            "probs": all_probs.tolist(),
            "timestamp": time.time()
        }



# import os
# import time
# import numpy as np
# import pandas as pd
# from collections import Counter
# from stable_baselines3 import PPO
# from src.fabricator import EEGFabricator

# class AppInterfaceManager:
#     def __init__(self, model_root="./models/", results_csv="./results/loso_results.csv", threshold=0.90, channels=4, sfreq=250):
#         self.fabricator = EEGFabricator()
#         self.labels = {0: "Low", 1: "Medium", 2: "High"}
#         self.ensemble = []
#         self.weights = [] # Store weights for each model
        
#         # --- ADD THIS SECTION ---
#         self.sfreq = sfreq
#         self.window_size = int(sfreq * 2) # 2-second window
#         self.channels = channels
        
#         # Create the actual matrix (e.g., 4 channels x 500 samples)
#         self.buffer = np.zeros((self.channels, self.window_size))
#         self.points_collected = 0
#         # ------------------------

#         # 1. Load Performance Data and Filter
#         print(f"🔍 Filtering models with Accuracy >= {threshold}...")
#         try:
#             df = pd.read_csv(results_csv)
#             # Filter champions
#             champions_df = df[df['Accuracy'] >= threshold]
#             champion_map = dict(zip(champions_df['Subject'].astype(str), champions_df['Accuracy']))
#         except Exception as e:
#             print(f"⚠️ Could not read CSV ({e}). Falling back to uniform weights.")
#             champion_map = None

#         # 2. Dynamically load models with Weights
#         for root, dirs, files in os.walk(model_root):
#             if "best_model.zip" in files:
#                 sid = os.path.basename(root)
                
#                 if champion_map is None or sid in champion_map:
#                     model_path = os.path.join(root, "best_model.zip")
#                     print(f"📦 Loading Champion: Subject {sid}")
#                     self.ensemble.append(PPO.load(model_path))
                    
#                     # Calculate weight: Performance squared favors experts heavily
#                     # If CSV failed, default weight is 1.0
#                     acc = champion_map.get(sid, 1.0) if champion_map else 1.0
#                     self.weights.append(acc ** 2) 

#     def update_buffer(self, new_samples):
#         """
#         Receives new EEG samples from hardware and slides the 
#         internal buffer to keep the most recent window.
#         """
#         # n is the number of new points (e.g., 100)
#         n = new_samples.shape[1]
#         # Roll the existing data to the left to make room
#         self.buffer = np.roll(self.buffer, -n, axis=1)
#         # Insert the fresh data at the end (the right side)
#         self.buffer[:, -n:] = new_samples
#         # Keep track of how much data we've collected so we don't 
#         # predict on an empty buffer
#         self.points_collected += n

#     def predict(self, rejection_threshold=0.45):
#         if self.points_collected < self.window_size:
#             return {"status": "buffering", "progress": f"{self.points_collected}/{self.window_size}"}
        
#         obs = self.fabricator.extract_features(self.buffer)
        
#         # 1. Collect Weighted Votes
#         weighted_votes = {0: 0.0, 1: 0.0, 2: 0.0}
#         raw_counts = {0: 0, 1: 0, 2: 0}
        
#         for model, weight in zip(self.ensemble, self.weights):
#             action = int(model.predict(obs, deterministic=True)[0])
#             weighted_votes[action] += weight
#             raw_counts[action] += 1
        
#         # 2. Determine Winner and Confidence
#         final_action = max(weighted_votes, key=weighted_votes.get)
#         total_weight = sum(weighted_votes.values())
#         confidence_val = weighted_votes[final_action] / total_weight

#         # 3. Apply Rejection Logic
#         # If confidence is too low, we mark it as "Uncertain"
#         is_uncertain = confidence_val < rejection_threshold
#         display_label = "Uncertain" if is_uncertain else self.labels[final_action]

#         return {
#             "status": "success",
#             "is_uncertain": is_uncertain,
#             "workload_level": -1 if is_uncertain else final_action,
#             "label": display_label,
#             "confidence": confidence_val,
#             "vote_map": {self.labels[k]: v for k, v in raw_counts.items()},
#             "timestamp": time.time()
#         }

# import os
# import time
# import pandas as pd
# import numpy as np
# from collections import Counter
# from stable_baselines3 import PPO
# from src.fabricator import EEGFabricator

# class AppInterfaceManager:
#     # Added 'threshold' and 'results_csv' arguments here
#     def __init__(self, model_root="./models/", results_csv="./results/loso_results.csv", threshold=0.98, channels=4, sfreq=250):
#         self.fabricator = EEGFabricator()
#         self.labels = {0: "Low", 1: "Medium", 2: "High"}
#         self.ensemble = []
        
#         # 1. Filter Champions by Accuracy Threshold
#         print(f"🔍 Filtering models with Accuracy >= {threshold}...")
#         try:
#             df = pd.read_csv(results_csv)
#             # Only get IDs where accuracy is high
#             champion_ids = df[df['Accuracy'] >= threshold]['Subject'].astype(str).tolist()
#         except Exception as e:
#             print(f"⚠️ Could not read CSV ({e}). Loading all models instead.")
#             champion_ids = None

#         # 2. Dynamically load models
#         for root, dirs, files in os.walk(model_root):
#             if "best_model.zip" in files:
#                 subject_id = os.path.basename(root)
                
#                 # Only load if it's in our champion list (or if list is None)
#                 if champion_ids is None or subject_id in champion_ids:
#                     model_path = os.path.join(root, "best_model.zip")
#                     print(f"📦 Loading Champion: Subject {subject_id}")
#                     self.ensemble.append(PPO.load(model_path))

#         if not self.ensemble:
#             raise FileNotFoundError(f"❌ No models found meeting threshold {threshold} in {model_root}")

#         # 3. Buffer Setup
#         self.sfreq = sfreq
#         self.window_size = int(sfreq * 2) 
#         self.buffer = np.zeros((channels, self.window_size))
#         self.points_collected = 0

#     def update_buffer(self, new_samples):
#         n = new_samples.shape[1]
#         self.buffer = np.roll(self.buffer, -n, axis=1)
#         self.buffer[:, -n:] = new_samples
#         self.points_collected += n
        
#     def predict(self):
#             if self.points_collected < self.window_size:
#                 return {"status": "buffering", "progress": f"{self.points_collected}/{self.window_size}"}

#             obs = self.fabricator.extract_features(self.buffer)
            
#             # Collect individual votes
#             votes = [int(model.predict(obs, deterministic=True)[0]) for model in self.ensemble]
            
#             # Create Consensus Map
#             vote_counts = Counter(votes)
            
#             # Ensure all labels exist in the map even if 0 votes
#             vote_map = {self.labels[i]: vote_counts.get(i, 0) for i in range(3)}
            
#             final_action, count = vote_counts.most_common(1)[0]
#             confidence_val = count / len(self.ensemble)

#             return {
#                 "status": "success",
#                 "workload_level": final_action,
#                 "label": self.labels[final_action],
#                 "confidence": confidence_val,
#                 "vote_map": vote_map, # New data field
#                 "timestamp": time.time()
#             }

    # def predict(self):
    #     if self.points_collected < self.window_size:
    #         return {"status": "buffering", "progress": f"{self.points_collected}/{self.window_size}"}

    #     obs = self.fabricator.extract_features(self.buffer)
        
    #     # Get votes from all models
    #     votes = [int(model.predict(obs, deterministic=True)[0]) for model in self.ensemble]
        
    #     vote_counts = Counter(votes)
    #     final_action, count = vote_counts.most_common(1)[0]
        
    #     # Convert confidence to float for the test loop math
    #     confidence_val = count / len(self.ensemble)

    #     return {
    #         "status": "success",
    #         "workload_level": final_action, # Renamed to match your test loop
    #         "label": self.labels[final_action],
    #         "confidence": confidence_val, # Return as float for easier comparison
    #         "timestamp": time.time()
    #     }