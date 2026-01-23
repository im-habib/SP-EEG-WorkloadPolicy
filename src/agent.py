import os
import torch as th
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement

class StatusUpdateCallback(BaseCallback):
    def __init__(self, shared_status, shared_step, subject_id, total_steps):
        super().__init__()
        self.shared_status = shared_status
        self.shared_step = shared_step
        self.subject_id = subject_id
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        self.shared_step.value += 1
        
        # Update shared status every 512 steps for the table
        if self.n_calls % 512 == 0:
            mean_rew = 0.0
            if len(self.model.ep_info_buffer) > 0:
                mean_rew = np.mean([info['r'] for info in self.model.ep_info_buffer])
            
            # Send current step count to the dictionary
            self.shared_status[self.subject_id] = {
                "reward": f"{mean_rew:.1f}", 
                "status": "⏳ TRAINING",
                "current_step": self.n_calls,
                "total_steps": self.total_steps
            }
        return True

class HParamCallback(BaseCallback):
    def __init__(self, subject_id):
        super().__init__()
        self.subject_id = subject_id

    def _on_training_start(self) -> None:
        hparam_dict = {
            "subject_id": self.subject_id,
            "learning_rate": self.model.learning_rate,
            "n_steps": self.model.n_steps,
        }
        metric_dict = {
            "rollout/ep_rew_mean": 0,
            "eval/mean_reward": 0,
            "train/loss": 0,
        }
        self.logger.record("hparams", HParam(hparam_dict, metric_dict))

    def _on_step(self) -> bool:
        return True

class WorkloadAgent:
    def __init__(self, env, subject_id, log_dir="./logs/", shared_status=None, shared_step=None):
        self.subject_id = subject_id
        self.shared_status = shared_status
        self.shared_step = shared_step 
        self.save_dir = f"./models/{subject_id}/"
        os.makedirs(self.save_dir, exist_ok=True)
        
        monitored_env = Monitor(env)
        self.venv = DummyVecEnv([lambda: monitored_env])
        self.env = VecNormalize(self.venv, norm_obs=True, norm_reward=True)

        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=512, 
            policy_kwargs={
                "net_arch": [64, 64], 
                "activation_fn": th.nn.Tanh
            }
        )

    def train(self, steps=50000):
        callbacks = [HParamCallback(self.subject_id)]
        
        if self.shared_status is not None:
            callbacks.append(StatusUpdateCallback(
                self.shared_status, 
                self.shared_step, 
                self.subject_id, 
                total_steps=steps
            ))

        # 1. INSTANT CEILING: Stop immediately if we hit peak performance.
        # Based on your 100% Acc subjects, a reward of 18000-20000 is usually "Perfection".
        # This is the "mic drop" moment.
        instant_break = StopTrainingOnRewardThreshold(
            reward_threshold=18000, 
            verbose=1
        )

        # 2. PLATEAU DETECTION: Stop if the model gets "stuck".
        # We lower patience to 3 evals (3 * 1024 steps) to be more aggressive.
        plateau_break = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=3, 
            min_evals=5, 
            verbose=1
        )
        
        # 3. COMBINED EVALUATION CALLBACK
        eval_callback = EvalCallback(
            self.env, 
            callback_on_new_best=instant_break, # Checks for perfection
            callback_after_eval=plateau_break,  # Checks for plateau
            best_model_save_path=self.save_dir,
            log_path=self.save_dir, 
            eval_freq=1024, 
            deterministic=True
        )
        callbacks.append(eval_callback)

        # Start learning
        self.model.learn(
            total_timesteps=steps,
            callback=callbacks,
            tb_log_name=self.subject_id 
        )
    def predict(self, obs):
        norm_obs = self.env.normalize_obs(obs)
        return self.model.predict(norm_obs, deterministic=True)
    
# import os
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# class WorkloadAgent:
#     def __init__(self, env, log_dir="./logs/", save_dir="./models/best/"):
#         self.save_dir = save_dir
#         self.log_dir = log_dir
        
#         # 1. Wrap the environment for SB3 requirements
#         # We use DummyVecEnv because VecNormalize requires a vectorized environment
#         self.venv = DummyVecEnv([lambda: Monitor(env)])
        
#         # 2. Add the Normalization Wrapper
#         # norm_obs=True: Scales features
#         # clip_obs=10.: Prevents massive spikes from breaking the gradients
#         self.env = VecNormalize(self.venv, norm_obs=True, norm_reward=True, clip_obs=10.)
        
#         self.hyperparams = {
#             "learning_rate": 3e-4,
#             "n_steps": 2048, # Increased for better stability with larger datasets
#             "batch_size": 128,
#             "n_epochs": 10,
#             "gamma": 0.99,
#             "ent_coef": 0.05, # Increased entropy to encourage exploration of all states
#         }

#         self.model = PPO(
#             "MlpPolicy", 
#             self.env, 
#             verbose=0, 
#             tensorboard_log=self.log_dir, 
#             **self.hyperparams
#         )

#     def train(self, steps=50000, run_name="default_run", reward_threshold=1.5):
#         stop_callback = StopTrainingOnRewardThreshold(
#             reward_threshold=reward_threshold, 
#             verbose=1
#         )

#         eval_callback = EvalCallback(
#             self.env, 
#             best_model_save_path=self.save_dir,
#             log_path=self.save_dir, 
#             eval_freq=5000,
#             deterministic=True, 
#             callback_on_new_best=stop_callback
#         )

#         self.model.learn(
#             total_timesteps=steps, 
#             callback=eval_callback, 
#             tb_log_name=run_name,
#             progress_bar=True
#         )
        
#         # IMPORTANT: Save the normalization statistics (mean/var)
#         # Without this, you can't run the model in your 'test.py' later!
#         stats_path = os.path.join(self.save_dir, "vec_normalize.pkl")
#         self.env.save(stats_path)

#     def predict(self, obs):
#         # When predicting, we must use the env's normalization
#         # but NOT update the moving average
#         norm_obs = self.env.normalize_obs(obs)
#         return self.model.predict(norm_obs, deterministic=True)

# import os
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# class WorkloadAgent:
#     def __init__(self, env, log_dir="./logs/", save_dir="./models/best/"):
#         self.save_dir = save_dir
#         self.log_dir = log_dir
#         self.env = Monitor(env)
        
#         self.hyperparams = {
#             "learning_rate": 3e-4,
#             "n_steps": 1024,
#             "batch_size": 64,
#             "n_epochs": 10,
#             "gamma": 0.99,
#             "ent_coef": 0.01,
#         }

#         self.model = PPO(
#             "MlpPolicy", 
#             self.env, 
#             verbose=0, 
#             tensorboard_log=self.log_dir, 
#             **self.hyperparams
#         )

#     def train(self, steps=20480, run_name="default_run", reward_threshold=0.95):
#         # 1. Early Stopping Logic
#         stop_callback = StopTrainingOnRewardThreshold(
#             reward_threshold=reward_threshold, 
#             verbose=1
#         )

#         # 2. Evaluation and Best Model Checkpoint
#         eval_callback = EvalCallback(
#             self.env, 
#             best_model_save_path=self.save_dir,
#             log_path=self.save_dir, 
#             eval_freq=2000,
#             deterministic=True, 
#             callback_on_new_best=stop_callback
#         )

#         # 3. Train with TensorBoard Run Name
#         self.model.learn(
#             total_timesteps=steps, 
#             callback=eval_callback, 
#             tb_log_name=run_name,
#             progress_bar=True
#         )

#     def predict(self, obs):
#         return self.model.predict(obs, deterministic=True)