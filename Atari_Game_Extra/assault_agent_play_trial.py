
# Here, making use of the saved model
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import RecordVideo
from IPython.display import clear_output
import matplotlib.pyplot as plt

# using another version the wrapper
class SurvivalAssaultWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.time_alive = 0  

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.time_alive += 1
        reward = 0.1  # Small survival reward per step
        return obs, reward, done, trunc, info
    
# Create environment with video recording
env = gym.make("AssaultNoFrameskip-v4", render_mode="human")  # Ensure RGB mode
env = SurvivalAssaultWrapper(env)

# Load the trained model
model = PPO.load("custom_assault_v4_ppo")

# Reset the environment
obs  = env.reset()

# play the game with the agent loaded above
try:
    obs, _ = env.reset()
    start_time = time.time()
    
    while time.time() - start_time < 90:
        action, _ = model.predict(obs)  # Choose action
        obs, reward, done, trunc, info = env.step(action)
        env.render()

        if done or trunc:
            obs, _ = env.reset()  # Restart if game over
except Exception as e:
    print(f"Error: {e}")
finally:
    env.close()  # Ensure the environment is properly closed

