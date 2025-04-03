import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# import the play utils from gymnasium
from gymnasium.utils import play

# create the mountain car environment
env = gym.make('Assault', render_mode="rgb_array", max_episode_steps=1000)

# for the button mapping:
# https://gymnasium.farama.org/v0.29.0/environments/atari/riverraid/

# map the keys
action_keys = {
    "a": 4, # move left
    "d": 3, # move right
    "l": 6,  # press fire to left
    "k": 5,
    "i": 2
}

# no operation in this environment is 0
no_operation_button = 0

# noop = no operation, which key means "do nothing"
play.play(env, 
          fps=24, 
          keys_to_action=action_keys, 
          wait_on_player=False,
          zoom=2,
          noop=no_operation_button)
