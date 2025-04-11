import os
import unittest
import numpy as np
import cv2
import torch as th
import torch
from gymnasium.spaces import Box
from unittest.mock import MagicMock
from rl_main import VizDoomGym, CustomFeatureExtractor, Logger

# Dummy Classes to Simulate VizDoom

class DummyState:
    """A dummy state object with a screen_buffer and game_variables attribute."""
    def __init__(self, screen_buffer, game_variables=None):
        self.screen_buffer = screen_buffer
        # Provide default game variables if none are provided (e.g., ammo, health, killcount)
        self.game_variables = game_variables if game_variables is not None else [15, 100, 0]


class DummyGame:
    """A dummy game to simulate vizdoom.DoomGame behavior for testing."""
    def __init__(self):
        # Create a dummy screen buffer with shape (channels, height, width)
        self.state = DummyState(np.random.randint(0, 256, (3, 240, 320), dtype=np.uint8))
        self.episode_finished = False

    def new_episode(self):
        self.episode_finished = False
        self.state = DummyState(np.random.randint(0, 256, (3, 240, 320), dtype=np.uint8))

    def get_state(self):
        return self.state

    def make_action(self, action, frames):
        # Return a constant reward for testing purposes.
        return 1

    def is_episode_finished(self):
        return self.episode_finished


# Unit Tests

class TestVizDoomGym(unittest.TestCase):
    def setUp(self):
        # Create an instance of VizDoomGym with render disabled
        self.env = VizDoomGym(map_name="basic", render=False)

        # Override the internal game with DummyGame
        self.env.game = DummyGame()

    def test_preprocess_state(self):
        """Test that preprocess_state converts an input image to the expected shape."""
        # Create a dummy observation with shape (3, 240, 320)
        dummy_obs = np.random.randint(0, 256, (3, 240, 320), dtype=np.uint8)
        processed = self.env.preprocess_state(dummy_obs)

        self.assertEqual(processed.shape, (100, 160, 1))

    def test_reset(self):
        """Test that reset returns a properly preprocessed observation and an info dict."""
        # Create a dummy screen buffer
        dummy_screen = np.random.randint(0, 256, (3, 240, 320), dtype=np.uint8)

        # Override get_state to return our dummy state
        self.env.game.get_state = lambda: DummyState(dummy_screen)
        obs, info = self.env.reset()

        self.assertEqual(obs.shape, (100, 160, 1))
        self.assertIsInstance(info, dict)

    def test_step(self):
        """Test the step function returns observation, reward, done, truncated, and info correctly."""
        # Create a dummy screen buffer
        dummy_screen = np.random.randint(0, 256, (3, 240, 320), dtype=np.uint8)

        # Override get_state to return a dummy state with game_variables
        self.env.game.get_state = lambda: DummyState(dummy_screen, game_variables=[15, 100, 0])

        # Initialize previous game variables for reward logic
        self.env.prev_ammo = 15
        self.env.prev_killcount = 0
        obs, reward, done, truncated, info = self.env.step(0)

        self.assertEqual(obs.shape, (100, 160, 1))
        self.assertTrue(isinstance(reward, (int, float)))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)



class TestCustomFeatureExtractor(unittest.TestCase):
    def test_forward(self):
        """Test that the custom CNN feature extractor outputs the correct shape."""
        # Create a dummy observation space with shape (1, 100, 160)
        obs_space = Box(low=0, high=255, shape=(1, 100, 160), dtype=np.uint8)
        extractor = CustomFeatureExtractor(obs_space, features_dim=256)

        # Create a dummy input tensor of shape (batch_size, channels, height, width)
        dummy_input = th.zeros((1, 1, 100, 160), dtype=th.float32)
        output = extractor(dummy_input)

        self.assertEqual(output.shape, (1, 256))


if __name__ == '__main__':
    unittest.main()
