import os
import random
import cv2
import numpy as np
import vizdoom
import gymnasium
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


# ===================
# =  CONFIGURATION  =
# ===================

VERSION = "final_model" # Version of the model
LOG_DIR = f"Hybrid/logs/{VERSION}" # Directory to save logs
TIMESTEPS = 200000 # Total training timesteps
TRAIN = False # False to skip training | True to train models


# Define allowed actions for each FSM state.
allowed_actions_dict = {
    "SEARCH": [
        [0, 0, 0, 1, 0, 0, 0],   # move forward
        [0, 0, 0, 0, 0, 1, 0],   # turn left
        [0, 0, 0, 0, 0, 0, 1]   # turn right
    ],
    "AIM": [
        [0, 0, 0, 0, 0, 1, 0],  # Turn left
        [0, 0, 0, 0, 0, 0, 1],   # Turn right
        [0, 0, 1, 0, 0, 0, 0]   # Shoot
    ]
}

class TrainRLAgent(gymnasium.Env):
    """
    Training environment for the RL agent.
    Reward structure will differ based on the type of FSM-state PPO model
    currently undergoing training.
    """

    def __init__(self, allowed_actions, render=False):
        super(TrainRLAgent, self).__init__()
        self.allowed_actions = allowed_actions

        # Set up the DoomGame with deadly_corridor config
        self.game = vizdoom.DoomGame()
        self.game.load_config(os.path.join(vizdoom.scenarios_path, "deadly_corridor.cfg"))
        self.game.set_doom_skill(1)
        self.game.set_available_game_variables([vizdoom.GameVariable.HEALTH, vizdoom.GameVariable.KILLCOUNT, vizdoom.GameVariable.AMMO2])
        self.killcount = 0
        self.ammo = 52

        # Render frame logic
        if render == False: 
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        
        # Initialize the game
        self.game.init()
        
        # Define a simple observation space
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8) 
        
        # The action space is discrete: an index into allowed_actions
        self.action_space = Discrete(len(allowed_actions))

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Map the discrete action to the actual game action
        game_action = self.allowed_actions[action]
        reward = self.game.make_action(game_action)
        done = self.game.is_episode_finished()

        # Set truncated value as vizdoom doesn't support the trancated flag
        truncated = False

        if self.game.get_state():
            # Preprocess the state
            state = self.game.get_state().screen_buffer
            state = self.preprocess_state(state)
            
            # Get the game variables
            killcount = self.game.get_state().game_variables[1]
            ammo = self.game.get_state().game_variables[2]
            
            # Reward structure
            if ammo < self.ammo and killcount == self.killcount:
                reward -= 0.5   
            if killcount > self.killcount:
                reward += 10
                self.killcount = killcount   
            if killcount > 1 or ammo == 0:
                done = True
        else:
            state = np.zeros(self.observation_space.shape)
        
        # Penalize the agent for taking too long to finish the episode
        reward -= 0.01
        
        return state, reward, done, truncated, {}

    def preprocess_state(self, observation) -> np.ndarray:
        """
        Preprocess the observation to a format that the model can understand.
        We resize and apply greyscale to the image.
        """
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))

        return state
    
    def reset(self, seed=None) -> tuple[np.array, dict]:
        self.game.new_episode()
        state = self.game.get_state().screen_buffer

        return self.preprocess_state(state), {}

    def close(self) -> None:
        """
        Close the environment.
        """
        self.game.close()


def train_model(state_name, allowed_actions, total_timesteps) -> PPO:
    """
    Train a PPO model for a given FSM state.
    """
    env = TrainRLAgent(allowed_actions)
    
    # Check the environment
    print("\n=========  Checking environment...  =========\n")
    check_env(env)
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
    model.learn(total_timesteps=total_timesteps)
    model.save(f"Hybrid/models/{VERSION}/{state_name.lower()}")
    env.close()
    print("\n====================\n  Training done!  \n====================\n")
    
    return model


class HybridAgent:
    """
    Hybrid agent that combines FSM and RL.
    """
    def __init__(self):
        # Initialize ViZDoom for deadly_corridor
        self.game = vizdoom.DoomGame()
        config_path = os.path.join(vizdoom.scenarios_path, "deadly_corridor.cfg")
        self.game.load_config(config_path)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_available_game_variables([vizdoom.GameVariable.HEALTH, vizdoom.GameVariable.KILLCOUNT])
        self.game.init()
        self.game.set_doom_skill(1)

        self.fsm_state = "SEARCH"  # Initial high-level state

        # Load pretrained PPO models
        self.models = {
            "SEARCH": PPO.load(f"Hybrid/models/{VERSION}/search.zip"),
            "AIM": PPO.load(f"Hybrid/models/{VERSION}/aim.zip")
        }

    def detect_enemies(self):
        state = self.game.get_state()
        if not state or not hasattr(state, "labels"):
            return []
        
        # Convert the screen buffer to an OpenCV image for visualization
        img = state.screen_buffer
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        enemy_labels = []
        # Loop through all labels returned by ViZDoom
        for label in state.labels:
            # Check for the enemy object. Adjust the name if your scenario uses a different enemy label
            if label.object_name in ["Zombieman", "ShotgunGuy", "ChaingunGuy"] and label.width > 15:  
                enemy_labels.append(label)
                # Draw a rectangle using the label's coordinates
                # The label object typically provides x, y, width, and height
                x, y, w, h = label.x, label.y, label.width, label.height
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label.object_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Enemy Labels", img)
        cv2.waitKey(1)
        
        return enemy_labels

    def update_state(self):
        """Update FSM state based on enemy detection."""
        contours = self.detect_enemies()
        enemy_detected = len(contours) > 0

        # Basic transitions:
        if self.fsm_state == "SEARCH":
            if enemy_detected:
                self.fsm_state = "AIM"
        elif self.fsm_state == "AIM":
            if not enemy_detected:
                self.fsm_state = "SEARCH"

    def get_observation(self):
        state = self.game.get_state()
        if state:
            img = state.screen_buffer
            gray = cv2.cvtColor(np.moveaxis(img, 0, -1), cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
            state = np.reshape(resize, (100,160,1))
            return state
        else:
            return np.zeros((100,160,1), dtype=np.uint8)

    def run(self, episodes=5):
        for episode in range(episodes):
            self.game.new_episode()
            print(f"Starting episode {episode+1}")
            while not self.game.is_episode_finished():
                self.update_state()
                obs = self.get_observation()
                # PPO models expect observations with a batch dimension
                obs_batch = obs[None, ...]

                # Select the model based on the current FSM state
                model = self.models[self.fsm_state]
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

                # Map the discrete action back to the actual game action
                allowed_actions = allowed_actions_dict[self.fsm_state]
                # In case the model outputs an index not in range, default to the first action
                game_action = allowed_actions[action] if action < len(allowed_actions) else allowed_actions[0]

                self.game.make_action(game_action)
            print(f"Episode {episode+1} finished with reward: {self.game.get_total_reward()}")
        self.game.close()


if __name__ == "__main__":

    if TRAIN:
        # Train the PPO models for each FSM state
        model_search = train_model("SEARCH", allowed_actions_dict["SEARCH"], TIMESTEPS)
        model_aim = train_model("AIM", allowed_actions_dict["AIM"], TIMESTEPS)

    # Initialize and run the hybrid agent
    agent = HybridAgent()
    agent.run(3)
