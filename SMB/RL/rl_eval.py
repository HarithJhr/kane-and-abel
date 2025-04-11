from RL.rl_main import *

"""
This script is used to evaluate the PPO model on all the maps of the Super Mario Bros game.
The evaluation is done by running the model on each map 10 times and calculating the average percentage of the map completed 
by using the agent's x position. This script also stores the furthest distance reached by the agent in each map.
The results are saved in a CSV file named data_rl.csv in the RL subfolder.
"""


VERSION = "CUDA5m_V2"
model_path = "F:/zsmb_" + VERSION

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

if __name__ == "__main__":
    # Load the saved model
    try:
        model = PPO.load(model_path + "/final_model_" + VERSION, device="cuda")
        print(f"==============================Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"==============================Model file {model_path} not found. Ensure the file path is correct.")
        exit()

    # world
    for w in range(8):
        # level
        for l in range(4):
            # Create the environment
            env = DummyVecEnv([make_env(f"-{w+1}-{l+1}")])
            env = VecFrameStack(env, n_stack=4)
            env = VecMonitor(env)
            obs = env.reset()

            # Initialize variables for storing
            total_reward = 0
            info = 0
            total_distance = 0
            furthest_distance = 0

            # Run the model for 10 episodes - to get the average performance due to its stochastic nature
            for i in range(10):
                print(f"==============================Testing the model on map {w+1}-{l+1} @ episode {i+1}...")
                done = False

                while not done:
                    # Ensure the observation array has valid memory layout
                    obs = obs.copy()  # Fix negative strides

                    # Use the model to predict the next action
                    action, _ = model.predict(obs)
                    
                    # Take a step in the environment
                    obs, reward, done, info = env.step(action)

                    # Accumulate rewards
                    total_reward += reward
                
                # Save total distance and furthest distance
                total_distance += info[0]["x_pos"]
                furthest_distance = max(info[0]["x_pos"], furthest_distance)
                env.reset()
                
            # Save total distance and furthest distance
            avg_distance = total_distance / 10
            level_completed = round(avg_distance / map_length[f"{w+1}-{l+1}"] * 100, 2)
            furthest = round(furthest_distance / map_length[f"{w+1}-{l+1}"] * 100, 2)
            
            # Save the results to a CSV file
            if not os.path.exists("RL/data_rl.csv"):
                with open("RL/data_rl.csv", "w") as f:
                    f.write("level,level_completed,furthest\n")
            with open("RL/data_rl.csv", "a") as f:
                f.write(f"{w+1}-{l+1},{level_completed},{furthest}\n")
            
            env.close()


    print(f"==============================\nEvaluation Finished!\n==============================")
