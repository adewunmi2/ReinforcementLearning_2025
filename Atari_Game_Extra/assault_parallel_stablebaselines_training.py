# WE HAVE TO DO THE PARALLEL TRAINING IN A PLAIN PYTHON SCRIPT
# SINCE JUPYTER NOTEBOOK DOESN'T SUPPORT THE NEEDED SUBPROCESSES

# remember to install tensorboard
# pip install tensorboard

#############################################
### TENSORBOARD USAGE, INSTRUCTIONS BELOW ###
#############################################

# AFTER THE TRAINING STARTS, open a new cmd -window and launch tensorboard, e.g.
# tensorboard --logdir=THE_PATH_TO_YOUR_LOG_FOLDER .... for example:
# tensorboard --logdir=C:\reinforcementlearning_spring2025\tensorboard\PPO_1

import gymnasium as gym
import torch
import ale_py

# you can try different algorithms for your agent, PPO is good with action games
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_atari_env

# Vector Frame Stack is usually beneficial in Atari games
# for example: environment is processed in stack of 4 frames, which allows
# the agent to see a few frames into the future as it learns
from stable_baselines3.common.vec_env import VecFrameStack

# we should save a checkpoint of our agent every 160k timesteps
from stable_baselines3.common.callbacks import CheckpointCallback

# this is the reason we can't use Jupyter Notebook here => Subprocesses
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

# some other helpful imports
import re
import os
import time

######################################################################
### FIND THE MOST RECENT CHECKPOINT OF MODEL TRAINING (if present) ###
######################################################################

current_dir = os.path.dirname(__file__)
os.chdir(current_dir)

# let's create a new identifier for our checkpoint, in this format:
# STARTINGTIME-ppo-gamename_stepamount_steps.zip
# 1743603419-ppo-riverraid_4160000_steps.zip
current_time = int(time.time())

# amount of simultaneous environments TRAINED TOGETHER
# more environments => faster training => more memory needed
number_of_envs = 32

# make a checkpoint callback

# NOTE: save_freq is MULTIPLIED by the number of envs!
# 32 * 5000 = 160k => means, we are going to save a checkpoint 
# every 160k timesteps

# name_prefix luckily takes care of most in this case
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./logs/",
    name_prefix=f"{current_time}-ppo-assault"
)

#####################################
### CONFIGURE AND START TRAINING  ###
#####################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env_name = 'AssaultNoFrameskip-v4'

    # create multiple connected envs at same time by using make_atari_env + SubprocVecEnv
    env = make_atari_env(env_name, n_envs=number_of_envs, vec_env_cls=SubprocVecEnv, seed=42)

    # use frame stacking for 4 frames, so the agent can see a few frames to future game states'
    env = VecFrameStack(env, n_stack=4)

    # conditional logic => do we continue from a previous check point
    # or do we start from scratch
    if os.path.exists("logs/"):
        print("CHECKPOINT FOUND!")

        # get all checkpoint files in the folder
        models = os.listdir("logs/")

        # ChatGPT generated originally (AND SURE LOOKS LIKE IT!!!!)'
        # function to extract numbers from the checkpoint file name
        def extract_numbers(filename):
            numbers = list(map(int, re.findall(r'\d+', filename)))
            return (numbers[0], numbers[1])
        
        # custom sorting
        sorted_files = sorted(models, key=extract_numbers)
        models = sorted_files

        print("Using model: ", models[-1])

        # LOAD MODEL FROM CHECKPOINT
        model = PPO.load(f"logs/{models[-1]}", env=env, learning_rate=0.0003, device=device,
                        n_steps = 256,
                        n_epochs = 4)
    else:
        print("No existing model/checkpoint found!")

        # create model from scratch
        # LOAD MODEL FROM CHECKPOINT
        model = PPO("CnnPolicy", 
                        env=env, 
                        verbose=1,
                        learning_rate=0.0003, 
                        tensorboard_log="./tensorboard/",
                        device=device,
                        n_steps = 256,
                        n_epochs = 4)


    # start training
    model.learn(total_timesteps=1_500_000, progress_bar=True, tb_log_name="PPO", callback=checkpoint_callback)

if __name__ == "__main__":
    main()
