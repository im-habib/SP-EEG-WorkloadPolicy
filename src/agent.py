import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class WorkloadAgent:
    def __init__(self, env, log_dir="./logs/"):
        os.makedirs(log_dir, exist_ok=True)
        self.env = Monitor(env, log_dir)
        # n_steps should be 2048 (default) for PPO
        self.model = PPO("MlpPolicy", self.env, verbose=1, tensorboard_log=log_dir)

    def train(self, steps=20480):
        print(f"🚀 Training for {steps} steps (Progress Bar Active)...")
        self.model.learn(total_timesteps=steps, progress_bar=True)

    def predict(self, obs):
        return self.model.predict(obs, deterministic=True)