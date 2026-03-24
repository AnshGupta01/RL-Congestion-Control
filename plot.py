import numpy as np
import matplotlib.pyplot as plt

data = np.load("history.npy")

throughput = data[:, 0]
delay = data[:, 1]
reward = data[:, 2]

# Throughput plot
plt.figure()
plt.plot(throughput)
plt.title("RL Throughput over Time")
plt.xlabel("Steps")
plt.ylabel("Mbps")
plt.savefig("plots/throughput.png")   # ✅ saves file
plt.close()

# Delay plot
plt.figure()
plt.plot(delay)
plt.title("Delay over Time")
plt.xlabel("Steps")
plt.ylabel("ms")
plt.savefig("plots/delay.png")        # ✅ saves file
plt.close()

# Reward plot
plt.figure()
plt.plot(reward)
plt.title("Reward over Time")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.savefig("plots/reward.png")       # ✅ saves file
plt.close()

print("Plots saved successfully!")
