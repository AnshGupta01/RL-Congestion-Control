from stable_baselines3 import PPO
from env import CongestionEnv

import matplotlib.pyplot as plt


env = CongestionEnv()
model = PPO.load("cc_rl_model_2")

obs, _ = env.reset()

total_throughput = 0
total_delay = 0
steps = 10

for _ in range(steps):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)

    throughput = obs[0] * 50

    
    delay = env.get_delay()

    total_throughput += throughput
    total_delay += delay

    loss = getattr(env, "last_loss", 0)
    print(f"Throughput: {throughput:.2f}, Delay: {delay:.2f}, Loss: {loss}")

    

print("RL Avg Throughput:", total_throughput / steps)
print("RL Avg Delay:", total_delay / steps)
