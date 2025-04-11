import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import cv2
from detector_functions import detect_high_obstacles, detect_valley
import numpy as np


class FiniteStateMachine:
    def __init__(self, action_space: list = None):
        # Initialize the FSM with a predefined movement scheme and state transitions
        if action_space is None:
            action_space = [
                ["NOOP"],
                ["right"],
                ["right", "A", "B"]
            ]
        self.action_space = action_space
        self.action = "none"

        # Initial state
        self.current_state = "IDLE"

        # Define the state transition table
        self.transitions = {
            "IDLE": {
                "ENEMY": "JUMP-RIGHT",
                "PIPE": "JUMP-RIGHT",
                "VALLEY": "JUMP-RIGHT",
                "DEFAULT": "MOVE-RIGHT"
            },
            "MOVE-RIGHT": {
                "ENEMY": "JUMP-RIGHT",
                "PIPE": "JUMP-RIGHT",
                "VALLEY": "JUMP-RIGHT",
                "DEFAULT": "MOVE-RIGHT"
            },
            "JUMP-RIGHT": {
                "ENEMY": "JUMP-RIGHT",
                "PIPE": "JUMP-RIGHT",
                "VALLEY": "JUMP-RIGHT",
                "DEFAULT": "MOVE-RIGHT"
            }
        }

    def state_transition(self, trigger: str) -> None:
        # Transition to the next state based on the current state and trigger
        if trigger in self.transitions[self.current_state]:
            self.current_state = self.transitions[self.current_state][trigger]
        else:
            self.current_state = self.transitions[self.current_state]["DEFAULT"]

    def get_action(self) -> int:
        # Map the current state to an action
        state_to_action = {
            "IDLE": self.action_space.index(["NOOP"]),
            "MOVE-RIGHT": self.action_space.index(["right"]),
            "JUMP-RIGHT": self.action_space.index(["right", "A", "B"])
        }
        return state_to_action.get(self.current_state, self.action_space.index(["NOOP"]))

    def update_state(self, state: np.ndarray, unittest_mode: bool = False, unittest_obs_return_val: str = "null", unittest_pipe_return_val: bool = False) -> None:
        # Update the FSM state based on the current environment
        if detect_high_obstacles(state, unittest_mode = unittest_mode, unittest_return_val=unittest_obs_return_val) == "ENEMY":
            self.state_transition("ENEMY")
            self.action = "ENEMY"
        elif detect_high_obstacles(state, unittest_mode = unittest_mode, unittest_return_val=unittest_obs_return_val) == "PIPE":
            self.state_transition("PIPE")
            self.action = "PIPE"
        elif detect_valley(state, unittest_mode = unittest_mode, unittest_return_val=unittest_pipe_return_val):
            self.state_transition("VALLEY")
            self.action = "VALLEY"
        else:
            self.state_transition("DEFAULT")
            self.action = "DEFAULT"


class MarioEnvironment:
    def __init__(self, env_name: str = "SuperMarioBros-1-1-v1"):
        # Initialize the Mario environment and the FSM
        self.env_name = env_name
        self.fsm = FiniteStateMachine()  # Initialize FSM
        self.env = self.initialize_environment()
        self.current_frame, _ = self.env.reset()  # Reset returns a tuple
        self.done = False

    def initialize_environment(self) -> JoypadSpace:
        # Create and configure the Mario environment
        env = gym_super_mario_bros.make(
            self.env_name,
            render_mode="human",  # Enable visual rendering
            apply_api_compatibility=True
        )
        return JoypadSpace(env, self.fsm.action_space)

    def draw_visuals(self, state: np.ndarray):
        # Display agent vision in a single window
        # Copy the state to draw debugging rectangles
        copied_state = state.copy()
        
        # Draw rectangles to visualize detection regions
        cv2.rectangle(copied_state, (130, 160), (190, 210), (0, 0, 255), 2)  # Enemy/Pipe detection region in RED
        cv2.rectangle(copied_state, (125, 210), (130, 240), (0, 255, 0), 2)  # Valley detection region in GREEN
        cv2.rectangle(copied_state, (140, 145), (145, 155), (0, 255, 255), 2)  # Brick gap detection region in YELLOW
        
        # Add text to display the current state
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(copied_state, f"State: {self.fsm.current_state}", (10, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(copied_state, f"Action: {self.fsm.action}", (10, 60), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert the frame to grayscale and apply thresholding
        gray_frame = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        _, thresholded_frame = cv2.threshold(gray_frame, 10, 255, cv2.THRESH_BINARY)

        # Convert single-channel grayscale frame to BGR for consistency in display
        thresholded_bgr = cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR)

        # Resize both frames to the same dimensions for concatenation
        copied_state_resized = cv2.resize(copied_state, (300, 300))
        thresholded_resized = cv2.resize(thresholded_bgr, (300, 300))

        # Combine the frames horizontally
        combined_frame = cv2.hconcat([copied_state_resized, thresholded_resized])

        # Display the combined frame in a single window
        cv2.imshow("Detection Regions                                                                  FSM Model's Vision", combined_frame)

    def step(self) -> None:
        # Execute a single step in the Mario environment
        self.fsm.update_state(self.current_frame)  # Update FSM based on the environment
        action = self.fsm.get_action()  # Get the action from the FSM
        self.current_frame, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        self.draw_visuals(self.current_frame)

    def run(self) -> None:
        # Main loop to execute the Mario environment with FSM logic until the game ends
        while not self.done:
            self.step()
            self.env.render()
        self.env.close()


if __name__ == "__main__":
    # Create and run the Mario environment
    mario_env = MarioEnvironment()
    mario_env.run()
