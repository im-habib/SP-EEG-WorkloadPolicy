import numpy as np

class LiveStreamBuffer:
    def __init__(self, channels=4, sfreq=250, window_sec=2):
        self.sfreq = sfreq
        self.window_size = int(sfreq * window_sec) # 500 points
        self.channels = channels
        # Initialize an empty buffer (zeros)
        self.buffer = np.zeros((channels, self.window_size))
        self.points_collected = 0

    def add_data(self, new_samples):
        """
        new_samples: a small chunk of data from the device (e.g., 5-10 samples)
        shape: (channels, n_samples)
        """
        n = new_samples.shape[1]
        # Shift old data to the left, add new data to the right
        self.buffer = np.roll(self.buffer, -n, axis=1)
        self.buffer[:, -n:] = new_samples
        
        self.points_collected += n

    def get_last_2_seconds(self):
        """This is the 'obs' provider for your agent.predict()"""
        if self.points_collected < self.window_size:
            return None # Not enough data yet to make a prediction
        return self.buffer