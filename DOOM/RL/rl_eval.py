from rl_main import *
import numpy as np

"""
This script is used to evaluate the model on the three maps: basic, defend_the_center, and defend_the_line.
The model will be evaluated for 10 episodes on each map and the average killcount, max killcount, average reward,
max reward, and standard deviation of the killcount will be recorded in a CSV file.
"""

if __name__ == "__main__":
    maps = ["basic", "defend_the_center", "defend_the_line"]

    # Load the model
    model = PPO.load(f"RL/{VERSION}/final_model")

    for map_name in maps:
        print(f"\n================\nEvaluating {map_name}\n================\n")
        
        # Rendered environment
        env = VizDoomGym(map_name=map_name,render=True)
        
        # Initialize the variables
        total_killcount = 0
        max_killcount = 0
        total_reward = 0
        max_reward = 0
        killcount_arr = []

        # Test the model for 10 episodes
        for episode in range(10): 
            print(f"==============================Testing the model on map {map_name} @ episode {episode+1}...")

            obs, _ = env.reset()
            done = False

            total_ep_reward = 0
            total_ep_killcount = 0

            # Run the episode
            while not done: 
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                total_ep_reward += reward

                # Record the killcount before the episode ends
                if info["info"] != 0:
                    total_ep_killcount = info["info"]["killcount"]
            
            # Update the total killcount and reward
            total_killcount += total_ep_killcount
            total_reward += total_ep_reward
            max_killcount = max(max_killcount, total_ep_killcount)
            max_reward = max(max_reward, total_ep_reward)
            killcount_arr.append(total_ep_killcount)

        # record data
        avg_killcount = round(total_killcount / 10, 2)
        avg_reward = round(total_reward / 10, 2)
        standard_deviation = round(np.std(killcount_arr), 2)

        # Save the data
        if not os.path.exists("RL/data_rl.csv"):
            with open("RL/data_rl.csv", "w") as f:
                f.write("map,avg_killcount,max_killcount,avg_reward,max_reward,standard_deviation\n")
        with open("RL/data_rl.csv", "a") as f:
            f.write(f"{map_name},{avg_killcount},{max_killcount},{avg_reward},{max_reward},{standard_deviation}\n")
        