from FSM.fsm_main import *
import os

"""
This script is used to evaluate the FSM model on all the maps of the Super Mario Bros game.
The evaluation is done by running the model on each map and calculating the percentage of the map completed by using the agent's x position.
The results are saved in a CSV file named data_fsm.csv in the FSM subfolder.
"""

# Variable to store the number of extra pixels in the environment ie the number of pixels that are not part of the map but are displayed
extra_pixels = 423

# Dictionary of the length of each map based on the level's image size
map_length = {
    "1-1": 3584 - extra_pixels,
    "1-2": 3072 - extra_pixels,
    "1-3": 3584 - extra_pixels,
    "1-4": 3328 - extra_pixels,
    "2-1": 3584 - extra_pixels,
    "2-2": 3072 - extra_pixels,
    "2-3": 4608 - extra_pixels,
    "2-4": 3328 - extra_pixels,
    "3-1": 3584 - extra_pixels,
    "3-2": 4352 - extra_pixels,
    "3-3": 3584 - extra_pixels,
    "3-4": 3328 - extra_pixels,
    "4-1": 3840 - extra_pixels,
    "4-2": 2584 - extra_pixels,
    "4-3": 3328 - extra_pixels,
    "4-4": 3840 - extra_pixels,
    "5-1": 3584 - extra_pixels,
    "5-2": 3584 - extra_pixels,
    "5-3": 3584 - extra_pixels,
    "5-4": 3328 - extra_pixels,
    "6-1": 4096 - extra_pixels,
    "6-2": 3840 - extra_pixels,
    "6-3": 3840 - extra_pixels,
    "6-4": 3328 - extra_pixels,
    "7-1": 3328 - extra_pixels,
    "7-2": 3072 - extra_pixels,
    "7-3": 4608 - extra_pixels,
    "7-4": 4352 - extra_pixels,
    "8-1": 6400 - extra_pixels,
    "8-2": 3840 - extra_pixels,
    "8-3": 4608 - extra_pixels,
    "8-4": 5120 - extra_pixels
}

# Create an evaluation class which inherits from the MarioEnvironment class
class EvalEnv(MarioEnvironment):
    def __init__(self, env_name: str = "SuperMarioBros-1-1-v1"):
        super(EvalEnv, self).__init__(env_name)
        
        # added attribute to store the info dictionary
        self.info = {}
        
    def step(self) -> dict:
        # Execute a single step in the Mario environment
        self.fsm.update_state(self.current_frame)  # Update FSM based on the environment
        action = self.fsm.get_action()  # Get the action from the FSM
        self.current_frame, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        self.draw_visuals(self.current_frame)
        
        # added a return statement to get the info
        return info

    def run(self):
        # Main loop to execute the Mario environment with FSM logic until the game ends
        while not self.done:
            info = self.step()
            self.env.render()
            
            # added a line to save the info
            self.info = info
            
        self.env.close()

if __name__ == "__main__":
    # world
    for w in range(8):
        # level
        for l in range(4):
            print(f"==============================Testing the model on map {w+1}-{l+1}...")
            mario_env = EvalEnv(f"SuperMarioBros-{w+1}-{l+1}-v1")
            mario_env.run()
            distance = mario_env.info["x_pos"]
            level_completed = round(distance / map_length[f"{w+1}-{l+1}"] * 100, 2)
            
            if not os.path.exists("FSM/data_fsm.csv"):
                with open("FSM/data_fsm.csv", "w") as f:
                    f.write("level,level_completed\n")
            with open("FSM/data_fsm.csv", "a") as f:
                f.write(f"{w+1}-{l+1},{level_completed}\n")
    
    print(f"==============================\nEvaluation Finished!\n==============================")
    