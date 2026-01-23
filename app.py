import os
import time
import pandas as pd
import numpy as np
from collections import Counter
from stable_baselines3 import PPO
from src.fabricator import EEGFabricator

class AppInterfaceManager:
    # Added 'threshold' and 'results_csv' arguments here
    def __init__(self, model_root="./models/", results_csv="./results/loso_results.csv", threshold=0.98, channels=4, sfreq=250):
        self.fabricator = EEGFabricator()
        self.labels = {0: "Low", 1: "Medium", 2: "High"}
        self.ensemble = []
        
        # 1. Filter Champions by Accuracy Threshold
        print(f"🔍 Filtering models with Accuracy >= {threshold}...")
        try:
            df = pd.read_csv(results_csv)
            # Only get IDs where accuracy is high
            champion_ids = df[df['Accuracy'] >= threshold]['Subject'].astype(str).tolist()
        except Exception as e:
            print(f"⚠️ Could not read CSV ({e}). Loading all models instead.")
            champion_ids = None

        # 2. Dynamically load models
        for root, dirs, files in os.walk(model_root):
            if "best_model.zip" in files:
                subject_id = os.path.basename(root)
                
                # Only load if it's in our champion list (or if list is None)
                if champion_ids is None or subject_id in champion_ids:
                    model_path = os.path.join(root, "best_model.zip")
                    print(f"📦 Loading Champion: Subject {subject_id}")
                    self.ensemble.append(PPO.load(model_path))

        if not self.ensemble:
            raise FileNotFoundError(f"❌ No models found meeting threshold {threshold} in {model_root}")

        # 3. Buffer Setup
        self.sfreq = sfreq
        self.window_size = int(sfreq * 2) 
        self.buffer = np.zeros((channels, self.window_size))
        self.points_collected = 0

    def update_buffer(self, new_samples):
        n = new_samples.shape[1]
        self.buffer = np.roll(self.buffer, -n, axis=1)
        self.buffer[:, -n:] = new_samples
        self.points_collected += n
        
    def predict(self):
            if self.points_collected < self.window_size:
                return {"status": "buffering", "progress": f"{self.points_collected}/{self.window_size}"}

            obs = self.fabricator.extract_features(self.buffer)
            
            # Collect individual votes
            votes = [int(model.predict(obs, deterministic=True)[0]) for model in self.ensemble]
            
            # Create Consensus Map
            vote_counts = Counter(votes)
            
            # Ensure all labels exist in the map even if 0 votes
            vote_map = {self.labels[i]: vote_counts.get(i, 0) for i in range(3)}
            
            final_action, count = vote_counts.most_common(1)[0]
            confidence_val = count / len(self.ensemble)

            return {
                "status": "success",
                "workload_level": final_action,
                "label": self.labels[final_action],
                "confidence": confidence_val,
                "vote_map": vote_map, # New data field
                "timestamp": time.time()
            }

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