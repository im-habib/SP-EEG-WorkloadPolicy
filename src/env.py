import numpy as np
import gymnasium as gym
from gymnasium import spaces

class WorkloadEnv(gym.Env):
    def __init__(self, signals, labels, fabricator, window_sec=2):
        super().__init__()
        self.signals = signals
        self.labels = labels
        self.fab = fabricator
        self.win_len = window_sec * fabricator.fs
        
        self.observation_space = spaces.Box(low=-30, high=30, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.ptr = 0
        self.prev_action = None

    def step(self, action):
        # Determine index for labels (labels are usually lower frequency than EEG)
        label_idx = min(self.ptr // (10 * self.fab.fs), len(self.labels)-1)
        target = self.labels[label_idx]
        
        # Reward logic
        reward = 1.0 if action == target else -1.0
        # Stability Factor: Penalty for changing labels too fast
        if self.prev_action is not None and action != self.prev_action:
            reward -= 0.3 
            
        self.prev_action = action
        self.ptr += self.win_len
        done = (self.ptr + self.win_len) >= self.signals.shape[1]
        
        obs = self.fab.extract_features(self.signals[:, self.ptr:self.ptr+self.win_len])
        return obs, reward, done, False, {"truth": target}

    def reset(self, seed=None, options=None):
        self.ptr = 0
        self.prev_action = None
        return self.fab.extract_features(self.signals[:, :self.win_len]), {}