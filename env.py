import gymnasium as gym
import numpy as np
import subprocess
import re

class CongestionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # state: throughput, delay
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=(2,), dtype = np.float32)
        
        # actions: decrease, same, increase window
        self.action_space = gym.spaces.Discrete(3)

        self.step_count = 0
        self.last_delay = 0
        
        self.window_sizes = [64, 128, 256, 512, 1024]  # KB
        self.idx = 2  # start at 256KB

    def step(self, action):
        # adjust window index
        self.step_count+=1
        if action == 0:
            self.idx = max(0, self.idx - 1)
        elif action == 2:
            self.idx = min(len(self.window_sizes)-1, self.idx + 1)

        window = self.window_sizes[self.idx]

        throughput = self.get_throughput(window)

        delay = self.get_delay()
        print(f"Delay: {delay:.2f} ms")

        # Normalize values
        norm_throughput = throughput / 50
        norm_delay = delay / 200

        norm_throughput = min(norm_throughput, 1.0)
        norm_delay = min(norm_delay, 1.0)

        # reward
        loss = getattr(self, "last_loss", 0)

        # reward = norm_throughput - norm_delay - 0.001 * loss
        reward = norm_throughput - 0.5 * norm_delay  

        # reward = 1.5 * norm_throughput - norm_delay     #high throughput bias
        # reward = norm_throughput - 0.5 * norm_delay - 0.001 * loss     #less delay penalty

        # Normalized state
        state = np.array([norm_throughput, norm_delay], dtype=np.float32)

        if not hasattr(self, "history"):
            self.history = []

        self.history.append((throughput, delay, reward))
        

        return state, reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 2
        self.step_count = 0
        self.last_delay = 0
        self.last_loss = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def get_throughput(self, window):
        cmd = [
            "iperf3",
            "-c", "10.0.0.1",
            "-t", "1",
            "-w", f"{window}K"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout

        # throughput
        throughput_match = re.search(r'(\d+\.?\d*)\s*Mbits/sec.*sender', output)

        # retransmissions (loss)
        loss_match = re.search(r'(\d+)\s+sender', output)

        throughput = float(throughput_match.group(1)) if throughput_match else 0
        loss = int(loss_match.group(1)) if loss_match else 0

        self.last_loss = loss  # IMPORTANT

        return throughput

    def get_delay(self):
        cmd = ["ping", "-c", "3", "10.0.0.1"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        output = result.stdout

        # extract avg RTT
        match = re.search(r'rtt .* = .*?/([\d\.]+)/', output)
        if match:
            return float(match.group(1))

        return 0
    


#iperf3 -c 10.0.0.1 -t 10    throughput
#ping -c 5 10.0.0.1     delay