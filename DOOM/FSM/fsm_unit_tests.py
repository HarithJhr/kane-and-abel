import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from fsm_main import DoomFSM
import cv2


class TestDoomFSMTransition(unittest.TestCase):
    def setUp(self):
        # Create a DoomFSM instance and replace its game with a dummy to avoid real game calls
        self.fsm = DoomFSM("basic")
        self.fsm.game = MagicMock()

    def test_transition_search_to_aim(self):
        # In SEARCH state, if detect_enemies returns a non-empty list, state should change to AIM
        self.fsm.state = "SEARCH"
        self.fsm.enemy_centered = False

        # Override detect_enemies to simulate enemy detection
        self.fsm.detect_enemies = lambda: [1] # Non-empty list implies enemy detected
        self.fsm.transition()

        self.assertEqual(self.fsm.state, "AIM")
    
    def test_transition_aim_to_search(self):
        # In AIM state, if no enemies are detected, state should revert to SEARCH
        self.fsm.state = "AIM"
        self.fsm.detect_enemies = lambda: [] # No enemy detected
        self.fsm.enemy_centered = False
        self.fsm.transition()

        self.assertEqual(self.fsm.state, "SEARCH")
    
    def test_transition_aim_to_attack(self):
        # In AIM state with enemy detected and centered, state should change to ATTACK
        self.fsm.state = "AIM"
        self.fsm.detect_enemies = lambda: [1] # Non-empty list implies enemy detected
        self.fsm.enemy_centered = True
        self.fsm.transition()

        self.assertEqual(self.fsm.state, "ATTACK")
    
    def test_transition_attack_to_aim(self):
        # In ATTACK state, if enemy is no longer centered, state should revert to AIM
        self.fsm.state = "ATTACK"
        self.fsm.enemy_centered = False

        # Override get_screen_state to simulate an image
        dummy_image = np.zeros((3, 240, 320), dtype=np.uint8)
        self.fsm.get_screen_state = lambda: cv2.cvtColor(np.transpose(dummy_image, (1, 2, 0)), cv2.COLOR_RGB2BGR)

        self.fsm.transition()

        self.assertEqual(self.fsm.state, "AIM")

class TestDoomFSMAimAttack(unittest.TestCase):
    def setUp(self):
        self.fsm = DoomFSM("basic")
        self.fsm.game = MagicMock() # Use a dummy game object

    @patch.object(DoomFSM, 'nearest_enemy')
    def test_aim_enemy_centered(self, mock_nearest_enemy):
        # Simulate a centered enemy
        # For an image width of 800, center_x would be 400
        mock_nearest_enemy.return_value = (400, 375, 50, 400)
        result = self.fsm.aim()

        # When enemy is centered, aim returns an empty string and sets enemy_centered True.
        self.assertEqual(result, "")
        self.assertTrue(self.fsm.enemy_centered)
    
    @patch.object(DoomFSM, 'nearest_enemy')
    def test_aim_enemy_left(self, mock_nearest_enemy):
        # Simulate an enemy to the left of center
        mock_nearest_enemy.return_value = (390, 375, 50, 400) # Difference = 10 > tolerance of 2
        result = self.fsm.aim()

        self.assertEqual(result, "left")
        self.assertFalse(self.fsm.enemy_centered)
    
    @patch.object(DoomFSM, 'nearest_enemy')
    def test_aim_enemy_right(self, mock_nearest_enemy):
        # Simulate an enemy to the right of center
        mock_nearest_enemy.return_value = (410, 375, 50, 400) # Difference = 10 > tolerance of 2
        result = self.fsm.aim()

        self.assertEqual(result, "right")
        self.assertFalse(self.fsm.enemy_centered)
    
    @patch.object(DoomFSM, 'nearest_enemy')
    def test_attack_no_enemy(self, mock_nearest_enemy):
        # Simulate no enemy being detected
        mock_nearest_enemy.return_value = None
        result = self.fsm.attack()

        self.assertEqual(result, "")
        self.assertFalse(self.fsm.enemy_centered)
    
    @patch.object(DoomFSM, 'nearest_enemy')
    def test_attack_enemy_not_centered(self, mock_nearest_enemy):
        # Simulate enemy detected but not centered (difference > tolerance of 10)
        mock_nearest_enemy.return_value = (380, 375, 50, 400) # Difference = 20
        result = self.fsm.attack()

        self.assertEqual(result, "")
        self.assertFalse(self.fsm.enemy_centered)
    
    @patch.object(DoomFSM, 'nearest_enemy')
    def test_attack_enemy_centered(self, mock_nearest_enemy):
        # Simulate enemy detected and centered (difference <= tolerance)

        mock_nearest_enemy.return_value = (395, 375, 50, 400) # Difference = 5
        result = self.fsm.attack()
        
        self.assertEqual(result, "attack")

if __name__ == '__main__':
    unittest.main()
