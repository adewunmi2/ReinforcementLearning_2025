import os
import time
import re

import gymnasium as gym
import torch
import ale_py

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

def main():
    # Ensure working directory is correct
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard", exist_ok=True)

    # Timestamp for unique checkpoint/model names
    current_time = str(int(time.time()))

    # Define the environment name and number of parallel environments
    env_name = 'AssaultNoFrameskip-v4'
    number_of_envs = 8  # Lowered from 32 for compatibility with many machines

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create parallel Atari environment with frame stacking
    env = make_atari_env(env_name, n_envs=number_of_envs, vec_env_cls=SubprocVecEnv, seed=42)
    env = VecFrameStack(env, n_stack=4)

    # Define a checkpoint callback (every 10k steps * 8 envs = 80k total steps between saves)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # per environment
        save_path="logs",
        name_prefix=f"{current_time}-ppo-assault"
    )

    # Initialize the PPO model with TensorBoard logging enabled
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        learning_rate=0.0003,
        device=device,
        n_steps=256,
        n_epochs=4,
    )

    # Start training (change total_timesteps as needed)
    model.learn(total_timesteps=500000, callback=checkpoint_callback, tb_log_name="PPO_1")

    # Save final model
    model.save(f"logs/{current_time}-ppo-assault-final.zip")

if __name__ == "__main__":
    main()
