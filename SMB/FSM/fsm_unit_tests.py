import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from fsm_main import FiniteStateMachine, MarioEnvironment

class TestFiniteStateMachine(unittest.TestCase):
    def setUp(self):
        # Verify that the FiniteStateMachine class is initialized correctly
        self.fsm = FiniteStateMachine()

    def test_initial_state(self):
        # Verify the initial state is "IDLE"
        self.assertEqual(self.fsm.current_state, "IDLE")

    def test_state_transition_enemy(self):
        # From "IDLE", trigger "ENEMY" should transition to "JUMP-RIGHT"
        self.fsm.state_transition("ENEMY")
        self.assertEqual(self.fsm.current_state, "JUMP-RIGHT")

    def test_state_transition_default_unknown_trigger(self):
        # When an unknown trigger is provided, default transition should be used
        self.fsm.current_state = "JUMP-RIGHT"
        self.fsm.state_transition("UNKNOWN")
        self.assertEqual(self.fsm.current_state, "MOVE-RIGHT")

    def test_get_action(self):
        # Verify that get_action returns the correct index based on the current state
        self.fsm.current_state = "IDLE"
        self.assertEqual(self.fsm.get_action(), self.fsm.action_space.index(['NOOP']))
        self.fsm.current_state = "MOVE-RIGHT"
        self.assertEqual(self.fsm.get_action(), self.fsm.action_space.index(['right']))
        self.fsm.current_state = "JUMP-RIGHT"
        self.assertEqual(self.fsm.get_action(), self.fsm.action_space.index(['right', 'A', 'B']))

    def test_update_state_enemy(self):
        # Verify that the FSM transitions to "JUMP-RIGHT" when an enemy is detected
        dummy_state = np.zeros((240, 256, 3), dtype=np.uint8)
        self.fsm.current_state = "IDLE"
        self.fsm.update_state(dummy_state, unittest_mode=True, unittest_obs_return_val="ENEMY", unittest_pipe_return_val=False)
        self.assertEqual(self.fsm.current_state, "JUMP-RIGHT")
        self.assertEqual(self.fsm.action, "ENEMY")

    def test_update_state_pipe(self):
        # Verify that the FSM transitions to "JUMP-RIGHT" when a pipe is detected
        dummy_state = np.zeros((240, 256, 3), dtype=np.uint8)
        self.fsm.current_state = "IDLE"
        self.fsm.update_state(dummy_state, unittest_mode=True, unittest_obs_return_val="PIPE", unittest_pipe_return_val=False)
        self.assertEqual(self.fsm.current_state, "JUMP-RIGHT")
        self.assertEqual(self.fsm.action, "PIPE")

    def test_update_state_valley(self):
        # Verify that the FSM transitions to "JUMP-RIGHT" when a valley is detected
        dummy_state = np.zeros((240, 256, 3), dtype=np.uint8)
        self.fsm.current_state = "IDLE"
        self.fsm.update_state(dummy_state, unittest_mode=True, unittest_obs_return_val="NONE", unittest_pipe_return_val=True)
        self.assertEqual(self.fsm.current_state, "JUMP-RIGHT")
        self.assertEqual(self.fsm.action, "VALLEY")

class TestMarioEnvironment(unittest.TestCase):
    @patch('gym_super_mario_bros.make')
    def test_initialize_environment(self, mock_make):
        # Create a dummy environment to be returned by gym_super_mario_bros.make
        dummy_env = MagicMock()

        # Simulate reset() returning a tuple (frame, info)
        dummy_env.reset.return_value = (np.zeros((240, 256, 3), dtype=np.uint8), None)
        mock_make.return_value = dummy_env

        # Initialize MarioEnvironment
        mario_env = MarioEnvironment()

        # Verify that the environment is wrapped with JoypadSpace
        from nes_py.wrappers import JoypadSpace
        self.assertIsInstance(mario_env.env, JoypadSpace)

        # Verify that reset() was called and current_frame is set correctly
        self.assertTrue(isinstance(mario_env.current_frame, np.ndarray))

if __name__ == "__main__":
    unittest.main()
