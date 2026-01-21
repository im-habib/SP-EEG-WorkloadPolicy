import numpy as np
from scipy.signal import welch, detrend

class EEGFabricator:
    def __init__(self, fs=256):
        self.fs = fs
        self.bands = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}

    def extract_features(self, window):
        # 1. Critical Clean: Replace any residual NaNs with 0 before math
        window = np.nan_to_num(window)
        
        # 2. Detrend (axis -1 is the time axis)
        window = detrend(window, axis=-1)
        
        de_vector = []
        for channel_data in window:
            f, psd = welch(channel_data, self.fs, nperseg=self.fs)
            for _, (low, high) in self.bands.items():
                idx = np.logical_and(f >= low, f <= high)
                # Use np.trapezoid for NumPy 2.0+
                band_pow = np.trapezoid(psd[idx], f[idx])
                # DE Formula with epsilon to prevent log(0)
                de = 0.5 * np.log(2 * np.pi * np.e * (max(band_pow, 1e-6)))
                de_vector.append(de)
        return np.array(de_vector, dtype=np.float32)