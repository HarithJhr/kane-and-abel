import unittest
import numpy as np
import gym
from gym.spaces import Box
import torch as th
from rl_main import CustomJoypadSpace, CustomCNN, SkipFrame, CustomReward, make_env, SIMPLE_MOVEMENT

# Dummy Environments for Testing

class DummyEnvSkip(gym.Env):
    """A simple dummy environment for testing wrappers."""
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, **kwargs):
        return np.zeros((84,84,3), dtype=np.uint8)
    
    def step(self, action):
        # Always return a constant observation, reward, and info dict.
        obs = np.zeros((84,84,3), dtype=np.uint8)
        reward = 1
        terminated = False
        truncated = False
        info = {"x_pos": 10, "score": 0, "flag_get": False, "life": 3}
        return obs, reward, terminated, truncated, info

class DummyEnvFlag(gym.Env):
    """Dummy environment simulating reaching a goal."""
    def reset(self, **kwargs):
        return np.zeros((84,84,3), dtype=np.uint8)
    def step(self, action):
        obs = np.zeros((84,84,3), dtype=np.uint8)
        reward = 10  # base reward
        terminated = False
        truncated = False
        # Simulate that the flag was reached.
        info = {"x_pos": 100, "score": 200, "flag_get": True, "life": 3}
        return obs, reward, terminated, truncated, info

class DummyEnvLife(gym.Env):
    """Dummy environment simulating low life scenario."""
    def reset(self, **kwargs):
        return np.zeros((84,84,3), dtype=np.uint8)
    def step(self, action):
        obs = np.zeros((84,84,3), dtype=np.uint8)
        reward = 5
        terminated = False
        truncated = False
        # Simulate low life (life < 2)
        info = {"x_pos": 50, "score": 100, "flag_get": False, "life": 1}
        return obs, reward, terminated, truncated, info

class DummyEnvJoypad(gym.Env):
    def reset(self, **kwargs):
        return kwargs

# Unit Tests

class TestCustomCNN(unittest.TestCase):
    # Test the CustomCNN model
    def test_output_shape(self):
        # Create a dummy observation space with shape (channels, height, width)
        obs_shape = (3, 84, 84)
        observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        features_dim = 512
        model = CustomCNN(observation_space, 512)

        # Create a dummy observation tensor with shape (batch_size, channels, height, width)
        dummy_obs = th.zeros((1,) + obs_shape, dtype=th.float32)
        output = model(dummy_obs)

        # Expect output shape (1, features_dim)
        self.assertEqual(output.shape, (1, features_dim))

class TestSkipFrame(unittest.TestCase):
    # Test the SkipFrame wrapper
    def test_skip_frame_aggregation(self):
        dummy_env = DummyEnvSkip()
        skip = 4
        wrapped_env = SkipFrame(dummy_env, skip)

        # Calling step should sum rewards over 4 steps (1 for each step)
        obs, total_reward, terminated, truncated, info = wrapped_env.step(0)

        self.assertEqual(total_reward, 4)

class TestCustomReward(unittest.TestCase):
    # Test the CustomReward wrapper for the "flag_get" flag
    def test_custom_reward_flag_get(self):
        dummy_env = DummyEnvFlag()
        wrapped_env = CustomReward(dummy_env)

        # Ensure initial values are reset properly
        wrapped_env.max_x = 0
        wrapped_env.current_x = 0
        state, modified_reward, terminated, truncated, info = wrapped_env.step(0)

        self.assertAlmostEqual(modified_reward, 61.0)
        self.assertTrue(terminated)

    # Test the CustomReward wrapper for the "life" flag
    def test_custom_reward_low_life(self):
        dummy_env = DummyEnvLife()
        wrapped_env = CustomReward(dummy_env)

        state, modified_reward, terminated, truncated, info = wrapped_env.step(0)

        self.assertAlmostEqual(modified_reward, -44.5)
        self.assertTrue(terminated)

class TestMakeEnv(unittest.TestCase):
    def test_make_env_wrappers(self):
        # make_env returns an initializer function
        env_init = make_env(render_mode="rgb_array")
        env = env_init()

        # Check if the environment is an instance of gym.Env (or wrapped)
        self.assertTrue(isinstance(env, gym.Env))

        # Reset and check observation shape; after GrayScaleObservation and ResizeObservation
        # The observation shape should be (84, 84, 1) or similar.
        obs, _ = env.reset()
        
        # Verify the observation
        self.assertEquals(obs.shape, (84, 84, 1))

class TestCustomJoypadSpace(unittest.TestCase):
    def test_reset_without_seed(self):
        # Test if the seed is removed from the reset kwargs        
        env = DummyEnvJoypad()
        joypad_env = CustomJoypadSpace(env, SIMPLE_MOVEMENT)
        
        reset_kwargs = {"seed": 42, "other": "value"}
        result = joypad_env.reset(**reset_kwargs)
        
        self.assertNotIn("seed", result)
        self.assertIsInstance(result, dict)

if __name__ == '__main__':
    unittest.main()
