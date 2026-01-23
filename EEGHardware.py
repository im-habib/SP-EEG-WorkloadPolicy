import time
import numpy as np

class EEGHardware:
    def __init__(self, channels=4, sfreq=250, chunk_size=100):
        self.channels = channels
        self.sfreq = sfreq
        self.chunk_size = chunk_size
        self.current_state = "low"

    def set_workload(self, level):
        self.current_state = level

    def pull_latest_data(self):
        # Simulate continuous time
        t = np.linspace(time.time(), time.time() + (self.chunk_size/self.sfreq), self.chunk_size)
        
        # Determine frequency based on state
        f = 10.0 if self.current_state == "low" else 25.0
        amp = 15.0 if self.current_state == "low" else 50.0
        
        # Generate 4 channels of signal
        data = np.array([amp * np.sin(2 * np.pi * f * t) for _ in range(self.channels)])
        # Add realistic noise
        data += np.random.normal(0, 2.0, data.shape)
        
        time.sleep(self.chunk_size / self.sfreq)
        return data
if __name__ == "__main__":
    hw = EEGHardware()
    print(f"🚀 Streaming {hw.chunk_size} samples of high-noise data...")
    for _ in range(5):
        data = hw.pull_latest_data()
        print("\n--- New Data Request (Garbage Included) ---")
        print(data[:, :5]) # Print first 5 samples of each channel
        print(f"Shape: {data.shape} | Max Voltage: {np.max(data):.2f} uV")