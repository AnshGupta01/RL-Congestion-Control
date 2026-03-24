from stable_baselines3 import PPO
from env import CongestionEnv
import numpy as np

env = CongestionEnv()

model = PPO("MlpPolicy", env, verbose=1, n_steps=8, batch_size=8)

model.learn(total_timesteps=200)
np.save("history.npy", env.history)

# model.save("cc_rl_model")
model.save("cc_rl_model_2")