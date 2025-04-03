# Prompt for this:
# After saving the model: "model.save("custom_riverraid_v5_ppo")"
# I want to have another python script (.py), where I load this model and allow the agent 
# to play the game, render mode = human

# the usual thing with GPT code -> fix the ale_py import 
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Custom Environment Class (to match the one used during training)
# class CustomassaultV5(gym.Env):
class CustomRiverraidV5(gym.Env):
    def __init__(self):
        # Set render_mode when creating the environment
        self.env = gym.make('ALE/Riverraid-v5', render_mode='human')  # Set render_mode here
        # self.env = gym.make('ALE/Assault-v5', render_mode='human')  # Set render_mode here
        self.time_alive = 0  # Track survival time ALE/Assault-v5

        # Make sure the action and observation spaces are the same as the original environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def reset(self, *args, **kwargs):
        self.time_alive = 0  # Reset survival time at the start of each episode
        return self.env.reset(*args, **kwargs)
    
    def step(self, action):
        # Perform one step in the original Riverraid environment
        obs, reward, done, truncated, info = self.env.step(action)

        # Increase survival time on each step if the agent is still alive
        if not done:
            self.time_alive += 1
        else:
            self.time_alive = 0  # Reset the timer if the agent dies

        # Custom reward: reward for surviving 10 seconds
        if self.time_alive >= 10:
            reward += 1  # Give a bonus reward for surviving 10 seconds

        return obs, reward, done, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()


# Create the custom Riverraid-v5 environment with render_mode set to "human"
env = CustomRiverraidV5()
# env = CustomAssaultV5()

# Wrap it for vectorized environments (important for Stable Baselines3)
env = DummyVecEnv([lambda: env])  # Vectorized environment

# Load the trained model
model = PPO.load("custom_riverraid_v5_ppo")
# model = PPO.load("custom_assault_v5_ppo")

# Reset the environment
obs  = env.reset()

# Play the game and render it in human mode
while True:
    # Get action from the model
    action, _states = model.predict(obs, deterministic=True)

    # Step through the environment
    obs, reward, done, _ = env.step(action)

    # Render the game (this step is now managed by the environment's internal render method)
    env.render()

    # Break if the game is done or truncated
    if done:
        print("Game Over!")
        break

# Close the environm
