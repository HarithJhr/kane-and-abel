import gym
import numpy as np
import os
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
import torch as th
from torch import nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ========================================================
# Initialize training variables and properties
# ========================================================
VERSION = "5m_timesteps"
TRAINING_TIMESTEPS = 5000000
model_folder = "RL/model_files"
LOG_DIR = "RL/training_logs"
RENDERMODE = "rgb_array"
train = False # Set to True to train the model, False to test the model
render_device = "cuda" if th.cuda.is_available() else "cpu" # Set to CUDA if available, else CPU
if train is False:
    RENDERMODE = "human"



# Custom Joypad Space to remove seed
class CustomJoypadSpace(JoypadSpace):
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        return super().reset()

# Custom Feature Extractor for Vision using CNN
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# Custom Skip Frame Wrapper
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip: int):
        super().__init__(env)
        self._skip = skip

    def step(self, action: np.ndarray) -> tuple:
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, terminated, truncated, info

# Custom Reward and Done Wrapper
class CustomReward(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomReward, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        
    def reset(self, **kwargs):
        """Resets attributes"""
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Overrides the step method to change the reward and done conditions"""
        state, reward, terminated, truncated, info = self.env.step(action)
        reward += max(0, info['x_pos'] - self.max_x)
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info["flag_get"]:
            reward += 500
            terminated = True
            print("GOAL")
        if info["life"] < 2:
            reward -= 500
            terminated = True
        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x_pos"]
        return state, reward / 10., terminated, truncated, info

# Environment pre-processing
def make_env(level: str = "", render_mode: str = "rgb_array"):
    """Create a Super Mario Bros environment with all necessary wrappers and prepares it for vectorized environments"""
    def _init():
        env = gym_super_mario_bros.make(f'SuperMarioBros{level}-v1',render_mode=render_mode , apply_api_compatibility=True)
        env = CustomJoypadSpace(env, SIMPLE_MOVEMENT)
        env = SkipFrame(env, skip=4)
        env = CustomReward(env)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=84)
        return env
    return _init

class Logger(BaseCallback):
    def __init__(self, check_freq: int, reward_log_freq: int, save_path: str, log_path: str, verbose: int = 1):
        super(Logger, self).__init__(verbose)
        self.check_freq = check_freq  # Frequency to save models
        self.reward_log_freq = reward_log_freq  # Frequency to log rewards
        self.save_path = save_path  # Directory to save models
        self.log_path = log_path  # Path to save reward logs

    def _init_callback(self) -> None:
        """Ensure the save path and log path exist."""
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:
        """Called at each step during training."""
        # Save the model periodically
        if self.model.num_timesteps % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"model_timestep_{self.model.num_timesteps}")
            self.model.save(model_path)

            # Print additional information
            if self.verbose > 0:
                print(f"Model saved at step {self.model.num_timesteps} to {model_path}")

        # Log cumulative rewards every episode end
        for info in self.locals["infos"]:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_steps = info["episode"]["l"]
                global_steps = self.model.num_timesteps
                
                # Log the rewards to rewards.log
                with open(self.log_path, "a") as log_file:
                    log_file.write(f"Timestep: {global_steps}, Reward: {ep_reward}, Episode_Steps: {ep_steps}\n")
                    
                # Print additional information
                if self.verbose > 0:
                    print(f"Highest reward saved at timestep {self.model.num_timesteps} to {self.log_path}")
        return True

if __name__ == "__main__":
    if train is True:
        # Use SubprocVecEnv for parallel environments
        # ONLY FOR TRAINING
        num_envs = 2  # Adjust based on CPU/GPU/RAM capabilities
        env = SubprocVecEnv([make_env(render_mode=RENDERMODE) for _ in range(num_envs)])
    else:
        # Use vectorized environment for evaluation
        env = DummyVecEnv([make_env(render_mode=RENDERMODE)])
        
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)

    # Monitor the environment
    env = VecMonitor(env)

    # Reset the environment for testing
    env.reset()

    if train:
        # Define the policy - using custom CNN
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )
        
        # Train the model
        model = PPO("CnnPolicy", 
                    env, 
                    tensorboard_log=LOG_DIR,
                    verbose=1, 
                    device=render_device, 
                    learning_rate=0.0002, 
                    ent_coef=0.01, 
                    policy_kwargs=policy_kwargs, 
                    gae_lambda=1.0, 
                    gamma=0.9
                )
        
        callback = Logger(
            check_freq=10000,  # Save model every 100,000 steps
            reward_log_freq=10000,  # Log rewards every 10,000 steps
            save_path= model_folder + "/checkpoints",  # Directory to save models
            log_path= model_folder + "/rewards.log"
        )

        # Set the environment
        model.set_env(env)

        # Train the model
        model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=callback)

        # Save the trained model
        model.save(model_folder + "/final_model_" + VERSION)
    else:
        # Load the saved model
        try:
            model = PPO.load(model_folder + "/final_model_" + VERSION, device=render_device)
            print(f"==============================\nLoaded model from {model_folder}\n==============================\n")
        except FileNotFoundError:
            print(f"Model file {model_folder} not found. Ensure the file path is correct.")
            exit()
            
        obs = env.reset()
        done = False
        total_reward = 0
        info = 0

        # Test the model
        print("==============================\nTesting the model...\n==============================\n")
        while not done:
            # Ensure the observation array has valid memory layout
            obs = obs.copy()  # Fix negative strides

            # Use the model to predict the next action
            action, _ = model.predict(obs)
            
            # Take a step in the environment
            obs, reward, done, info = env.step(action)

            # Accumulate rewards
            total_reward += reward
            

        print(f"\n==============================\nTotal reward achieved: {total_reward}\n==============================\n")
        env.close()
