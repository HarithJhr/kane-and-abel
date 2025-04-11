from vizdoom import * 
import time
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ===================
# =  CONFIGURATION  =
# ===================

maps = ["basic", "defend_the_center", "defend_the_line"] # List of maps to train on
MAP = maps[1] # Selected map
VERSION = 'model_final' # Version of the model
CHECKPOINT_DIR = f"RL/{VERSION}/checkpoints" # Directory to save model checkpoints
LOG_DIR = f"RL/logs/{VERSION}" # Directory to save the logs
REWARDS_DIR = f"RL/{VERSION}/rewards.log" # File to save the reward logs
TIMESTEPS = 1000000 # Number of timesteps to train the model
TRAIN = False # False to skip training | True to train models

# ===================
# ===================
# ===================


# Create Vizdoom OpenAI Gym Environment
class VizDoomGym(Env): 
    # Function that is called when we start the env
    def __init__(self, map_name="basic", render=False): 
        super().__init__()

        # Setup the game 
        self.game = vizdoom.DoomGame()
        self.game.load_config(os.path.join(scenarios_path, f"{map_name}.cfg"))
        self.game.set_available_game_variables([vizdoom.GameVariable.AMMO2, vizdoom.GameVariable.HEALTH, vizdoom.GameVariable.KILLCOUNT])
        
        # Initialize the attributes
        self.prev_ammo = None
        self.prev_health = None
        self.prev_killcount = None
        
        
        # Render frame logic
        if render == False: 
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        
        # Start the game 
        self.game.init()
        
        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8) 
        self.action_space = Discrete(3)
    
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes a step in the environment.
        This replaces the step method in Gymnasium
        """
        # Specify action and take step 
        actions = np.identity(3)
        reward = self.game.make_action(actions[action], 4) 

        if self.game.get_state():

            # Get the game variables
            ammo = self.game.get_state().game_variables[0]
            health = self.game.get_state().game_variables[1]
            killcount = self.game.get_state().game_variables[2]
            
            # Reward logic
            if self.prev_ammo is None:
                # Initialize starting values
                self.prev_ammo = ammo
                self.prev_health = health
                self.prev_killcount = killcount
            else:
                if ammo < self.prev_ammo and killcount == self.prev_killcount:
                    reward -= 3 # Heavily penalize shooting without killing
                if killcount > self.prev_killcount:
                        reward += 10 # Heavily incentivize killing
            
            # Update previous values
            self.prev_ammo = ammo
            self.prev_killcount = killcount
            
            # Process the state
            state = self.game.get_state().screen_buffer
            state = self.preprocess_state(state)

            # Update info dict
            info = {
                "ammo": ammo,
                "health": health,
                "killcount": killcount
            }
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        # Format info dict
        info = {"info":info}

        # Check if the episode is done
        done = self.game.is_episode_finished()

        # Hardcode truncated to False since vizdoom doesn't return truncated
        truncated = False
        
        return state, reward, done, truncated, info 
    
    
    def reset(self, seed=None) -> tuple[np.ndarray, dict]: 
        """
        Resets the environment
        """
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.preprocess_state(state), {}
    
    def preprocess_state(self, observation) -> np.ndarray:
        """
        Preprocesses the image for training
        """
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state
    
    def close(self) -> None: 
        """
        Closes the environment
        """
        self.game.close()

class CustomFeatureExtractor(BaseFeaturesExtractor): 
    """
    Custom CNN feature extractor for the VizDoom environment.
    This follows the source code for the CNNPOLICY used for the PPO model in Stable Baselines 3.
    We create this to ease potential future modifications to the model architecture.
    Based on current testing, the model is able to learn with this default architecture.
    """
    def __init__(self, observation_space, features_dim=256):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
    
        # Define the CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Get the number of features after the CNN
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        # Define the linear layer
        self.linear = nn.Linear(n_flatten, features_dim)
        
    def forward(self, observations: torch.Tensor) -> nn.Sequential:
        """
        Forward pass of the feature extractor
        """
        return self.linear(self.cnn(observations))

# Create training logger
class Logger(BaseCallback):
    def __init__(self, check_freq, save_path, rewards_path,verbose=1):
        super(Logger, self).__init__(verbose)

        # Initialize the attributes
        self.check_freq = check_freq
        self.save_path = save_path
        self.rewards_path = rewards_path
        self.current_episode_reward = 0

    def _init_callback(self):
        # Ensures directory exists
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.rewards_path is not None:
            os.makedirs(os.path.dirname(self.rewards_path), exist_ok=True)
        
    def _on_step(self) -> bool:
        # Save the model every check_freq timesteps
        if self.model.num_timesteps % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"model_timestep_{self.model.num_timesteps}")
            self.model.save(model_path)
            
            # Print the model path if verbose
            if self.verbose > 0:
                print(f"Saving model at timestep {self.model.num_timesteps} to {model_path}")
                
        # Calculate the episode reward
        reward = self.locals["rewards"]
        self.current_episode_reward += reward[0]
        
        # Check if the episode is done
        episode_done = self.locals["dones"][0]

        # Log episode rewards
        if episode_done:

            # Log rewards to rewards.log
            with open(self.rewards_path, "a") as f:
                f.write(f"Timestep: {self.num_timesteps}, Reward: {self.current_episode_reward}\n")
            
            # Print the rewards if verbose
            if self.verbose > 0:
                print(f"Saved rewards at {self.num_timesteps} to {self.rewards_path}")
            
            # Reset the current episode reward
            self.current_episode_reward = 0

        return True


# Define policy kwargs to enable custom feature extractor is used
policy_kwargs = dict(
    features_extractor_class=CustomFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

# Create the callback object
callback = Logger(check_freq=50000, save_path=CHECKPOINT_DIR, rewards_path=REWARDS_DIR)


if __name__ == "__main__":
    if TRAIN:
        # Non rendered environment
        env = VizDoomGym(map_name=MAP)
        
        # Check the environment
        print("\n=========  Checking environment...  =========\n")
        check_env(env) # Nothing will be printed if the environment is valid
        print("\n=========  Environment checked!  =========\n")
        
        # Create the model
        model = PPO('CnnPolicy',
                    env,
                    tensorboard_log=LOG_DIR,
                    verbose=1,
                    ent_coef=0.02,
                    learning_rate=0.0002,
                    clip_range=0.1,
                    n_steps=4096
                )
        
        # Train and save the model
        print("\n====================\n  Training agent...  \n====================\n")
        model.learn(total_timesteps=TIMESTEPS, callback=callback)
        model.save(f"RL/{VERSION}/final_model")
        env.close()
        print("\n====================\n  Training done!  \n====================\n")

    else:
        # Load the model
        model = PPO.load(f"RL/{VERSION}/final_model")
        
        # Rendered environment
        env = VizDoomGym(map_name=MAP, render=True)
        
        # Check the environment
        print("\n=========  Checking environment...  =========\n")
        check_env(env) # Nothing will be printed if the environment is valid
        print("\n=========  Environment checked!  =========\n")
        
        print("\n====================\n  Testing agent...  \n====================\n")
        
        # Test the model for a specified number of episodes
        for episode in range(10):

            # Reset the environment
            obs, _ = env.reset()

            # Initialize variables
            done = False
            total_reward = 0
            killcount = 0

            # Run the episode
            while not done: 
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                time.sleep(0.05) # Slow down the game for better visualization
                total_reward += reward
                if info["info"] != 0:
                    killcount = info["info"]["killcount"]
            print(f"Total Reward for episode {episode + 1} is {total_reward}\nKillcount: {killcount}")
            
            # Pause before next episode
            time.sleep(0.5)

        print("\n====================\n  Done!  \n====================\n")
        
