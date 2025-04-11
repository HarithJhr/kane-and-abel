# 🤖 Kane and Abel: AIs that play Games
Solo university Capstone project of a  comparative assessment of finite state machines and reinforcement learning with PPO, and produced an experimental hybrid framework combining finite state machines and PPO. This project was created in conjunction with University of London's Final Project module as part of the BSc Computer Science degree.


## ✅ How to set up project workspace
1. Use "SMB" and "DOOM" folders separately. These folders must be the root folder for the workspace or else the code will not work.
2. If you use Anaconda, you can create a conda env using the environment.yml file found in the respective workspace folders.
3. If you don't use Anaconda, create separate environments for SMB and DOOM with the following Python versions:
   - SMB: python 3.8.x
   - DOOM: python 3.10.x

   Install pip and use pip to install dependencies using the requirements.txt file found in the respective workspace folders.


## 🚀 How To Run
1. Select a game (SMB or DOOM)
2. Select which agent (FSM, RL, HYBRID) you would like to run
3. In its respective folder, run the file suffixed with "_main" - be sure to run with the correct environment

eg if you want to run the RL agent for DOOM, run the file in this path "DOOM/RL/rl_main.py"

## 🛑 Important Notes
1. Any Python file suffixed with "_eval" is used to evaluate the agents. It is automated and you can run that file to generate a csv file of agent data
2. Any Python file suffixed with "_unit_tests" is used for unit tests to facilitate Test Driven Development
3. Important config variables are available at the top of every python file. If you want to train, set the TRAIN boolean to True. By default, TRAIN is 
False. This means that running the main files will immediately run the trained model.


## 📁 File Structure
```
SMB
├── FSM
│   ├── detector_functions.py
│   ├── fsm_eval.py
│   ├── fsm_main.py
│   └── fsm_unit_tests.py
├── RL
│   ├── model_files
│   │   └── final_model_5m_timesteps.zip
│   ├── rl_eval.py
│   ├── rl_main.py
│   ├── rl_unit_tests.py
│   └── training_logs
├── environment_smb.yml
└── requirements_smb.txt
```

```
DOOM
├── FSM
│   ├── fsm_eval.py
│   ├── fsm_main.py
│   └── fsm_unit_tests.py
├── Hybrid
│   ├── hybrid_main.py
│   └── training_files
│       └── final
│           ├── aim.zip
│           └── search.zip
├── RL
│   ├── rl_eval.py
│   ├── rl_main.py
│   ├── rl_unit_tests.py
│   └── model_final
│       └── final_model.zip
├── environment_doom.yml
└── requirements_doom.txt
```
