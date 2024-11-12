# This file is modified from <https://github.com/cjy1992/gym-carla.git>:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gymnasium as gym
import gym_carla
import carla
from stable_baselines3 import SAC
from stable_baselines3 import DQN

def main():
  save_name = "SAC_dist"

  # Set gym-carla environment
  env = gym.make('carla-v1')

  model = SAC("MlpPolicy", env, device="cuda", buffer_size=500, verbose=1, tensorboard_log="./tensorboard_DQN/")
  
  model.learn(total_timesteps=1000)
  model.save(save_name)
  
  print("Done Training")

if __name__ == '__main__':
  main()
