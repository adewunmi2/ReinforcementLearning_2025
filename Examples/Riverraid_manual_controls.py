# this example uses the gymnasium utils play -tools
# in order to try some basic manual control

# in order to run this code, it has to be run via the venv
# in the terminal:
# cd lecture7_atari_experiments1\riverraid
# python riverraid_manual_controls.py

# depending on your setup, you have to first navigate to the correct folder before running the file
# or just use a relative path to your Python script, for example:
# python lecture7_atari_experiments1\riverraid\riverraid_manual_controls.py

# IMPORTANT!
# because the Atari games have been moved into a separate module
# WE HAVE TO REGISTER THE ATARI GAMES INTO OUR GYMNASIUM
# so that they are recognized
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# import the play utils from gymnasium
from gymnasium.utils import play

# create the mountain car environment
env = gym.make('Riverraid', render_mode="rgb_array", max_episode_steps=1000)

# for the button mapping:
# https://gymnasium.farama.org/v0.29.0/environments/atari/riverraid/

# map the keys
action_keys = {
    "a": 4, # move left
    "d": 3, # move right
    "l": 1  # press fire
}

# no operation in this environment is 0
no_operation_button = 0

# noop = no operation, which key means "do nothing"
play.play(env, 
          fps=24, 
          keys_to_action=action_keys, 
          wait_on_player=False,
          zoom=3,
          noop=no_operation_button)
