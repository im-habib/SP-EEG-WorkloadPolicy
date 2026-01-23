import numpy as np
from scipy.signal import welch, detrend

class EEGFabricator:
    def __init__(self, fs=125):
        self.fs = fs
        # Note: At 125Hz, we must keep bands below 62.5Hz (Nyquist Limit)
        self.bands = {
            'theta': (4, 8), 
            'alpha': (8, 13), 
            'beta': (13, 30), 
            'gamma': (30, 45)
        }

    def extract_features(self, window):
        """
        Input window shape: (channels, samples)
        Example: (14, 125) for 14 channels at 1s window
        """
        # 1. Basic Cleaning
        window = np.nan_to_num(window)
        window = detrend(window, axis=-1)
        
        de_vector = []
        for channel_data in window:
            # 2. PSD via Welch - Match nperseg to signal length (e.g., 125)
            # This fixes the UserWarning and provides the best 1Hz resolution
            sig_len = len(channel_data)
            f, psd = welch(channel_data, self.fs, nperseg=sig_len)
            
            for _, (low, high) in self.bands.items():
                idx = np.logical_and(f >= low, f <= high)
                
                # 3. Band Power (Integration)
                if np.any(idx):
                    # Using np.trapezoid as per latest Scipy/Numpy standards
                    band_pow = np.trapezoid(psd[idx], f[idx])
                else:
                    band_pow = 1e-6 
                
                # 4. Differential Entropy (DE) proxy
                # DE = 0.5 * log(2πeσ²)
                de = 0.5 * np.log(2 * np.pi * np.e * max(band_pow, 1e-6))
                de_vector.append(de)
        
        features = np.array(de_vector, dtype=np.float32)

        # 5. ROBUST SCALING
        # This makes the features 'distribution-aware' for the PPO agent
        median = np.median(features)
        q75, q25 = np.percentile(features, [75, 25])
        iqr = (q75 - q25) + 1e-6
        
        robust_features = (features - median) / iqr
        
        # 6. CLIPPING for PPO Stability
        # Bound features to [-5, 5] to prevent policy entropy collapse
        return np.clip(robust_features, -5.0, 5.0)

# import numpy as np
# from scipy.signal import welch, detrend

# class EEGFabricator:
#     def __init__(self, fs=256):
#         self.fs = fs
#         self.bands = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}

#     def extract_features(self, window):
#         # 1. Critical Clean: Replace any residual NaNs with 0 before math
#         window = np.nan_to_num(window)
        
#         # 2. Detrend (axis -1 is the time axis)
#         window = detrend(window, axis=-1)
        
#         de_vector = []
#         for channel_data in window:
#             f, psd = welch(channel_data, self.fs, nperseg=self.fs)
#             for _, (low, high) in self.bands.items():
#                 idx = np.logical_and(f >= low, f <= high)
#                 # Use np.trapezoid for NumPy 2.0+
#                 band_pow = np.trapezoid(psd[idx], f[idx])
#                 # DE Formula with epsilon to prevent log(0)
#                 de = 0.5 * np.log(2 * np.pi * np.e * (max(band_pow, 1e-6)))
#                 de_vector.append(de)
#         return np.array(de_vector, dtype=np.float32)