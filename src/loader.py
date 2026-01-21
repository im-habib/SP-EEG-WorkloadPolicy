import os
import numpy as np
import pandas as pd

class CLDriveLoader:
    def __init__(self, eeg_root="./CLDrive/EEG", label_root="./CLDrive/Labels"):
        self.eeg_root = eeg_root
        self.label_root = label_root

    def load_subject(self, subject_id):
        # Ensure subject_id is a string and remove any path components
        sid = str(subject_id).split('/')[-1]
        sub_path = os.path.join(self.eeg_root, sid)
        label_file = os.path.join(self.label_root, f"{sid}.csv")
        
        # Debugging aid:
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"❌ Label file missing: {label_file}. Check if the ID matches the EEG folder name.")

        # Load Labels
        labels_df = pd.read_csv(label_file)
        y = np.digitize(labels_df.iloc[:, 0].values, bins=[4, 7]) - 1
        
        all_signals = []
        for level in range(1, 10):
            data_path = os.path.join(sub_path, f"eeg_data_level_{level}.csv")
            base_path = os.path.join(sub_path, f"eeg_baseline_level_{level}.csv")
            
            if os.path.exists(data_path):
                df = pd.read_csv(data_path).iloc[:, :4]
                df = df.interpolate(method='linear').fillna(0)
                signal = np.nan_to_num(df.values, nan=0.0)
                
                if os.path.exists(base_path):
                    b_df = pd.read_csv(base_path).iloc[:, :4].interpolate().fillna(0)
                    baseline = np.nan_to_num(b_df.values, nan=0.0)
                    signal = signal - np.mean(baseline, axis=0)
                
                all_signals.append(signal.T)
        
        if not all_signals:
            raise ValueError(f"❌ No EEG data files found in {sub_path}")
            
        return np.concatenate(all_signals, axis=1), y