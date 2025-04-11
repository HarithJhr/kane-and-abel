from fsm_main import *
import numpy as np

"""
This script is used to evaluate the model on the three maps: basic, defend_the_center, and defend_the_line.
The model will be evaluated for 10 episodes on each map and the average killcount, max killcount, average reward,
max reward, and standard deviation of the killcount will be recorded in a CSV file.
"""

class EvalEnv(DoomFSM):
    def __init__(self, map_name):
        super().__init__(map_name)

    def run(self, episodes=10):
        """
        Overriding the run method to evaluate the model.
        Source code is mostly the same with the exception of data recording
        """

        print(f"\n================\nEvaluating {self.map_name}\n================\n")

        # Initialize the variables
        total_killcount = 0
        max_killcount = 0
        total_reward = 0
        max_reward = 0
        killcount_arr = []

        for episode in range(episodes):
            print(f"==============================Testing the model on map {self.map_name} @ episode {episode+1}...")

            self.game.new_episode()
            episode_killcount = 0
            while not self.game.is_episode_finished():
                self.detect_enemies()  # Perform enemy detection
                self.transition()
                self.action()

                # Get the killcount before the episode ends
                if self.game.get_state():
                    episode_killcount = self.game.get_state().game_variables[1]

            # Get the game variables
            episode_reward = self.game.get_total_reward()

            # Update the killcount and reward
            total_killcount += episode_killcount
            total_reward += episode_reward
            max_killcount = max(max_killcount, episode_killcount)
            max_reward = max(max_reward, episode_reward)
            killcount_arr.append(episode_killcount)

        # record data
        avg_killcount = round(total_killcount / episodes, 2)
        avg_reward = round(total_reward / episodes, 2)
        standard_deviation = round(np.std(killcount_arr), 2)

        # Save the data
        if not os.path.exists("FSM/data_fsm.csv"):
            with open("FSM/data_fsm.csv", "w") as f:
                f.write("map,avg_killcount,max_killcount,avg_reward,max_reward,standard_deviation\n")
        with open("FSM/data_fsm.csv", "a") as f:
            f.write(f"{self.map_name},{avg_killcount},{max_killcount},{avg_reward},{max_reward},{standard_deviation}\n")
        
        self.game.close()

if __name__ == "__main__":
    maps = ["basic", "defend_the_center", "defend_the_line"]

    # Evaluate the model for all maps
    for map_name in maps:
        doom = EvalEnv(map_name)
        doom.run(10)