import gymnasium as gym
import numpy as np
import subprocess
import re
import time

class PacerEnv(gym.Env):
    """
    PacerRL: Reinforcement Learning Environment for Congestion Control.
    Supports both Real-world (iperf3/ping) and Simulated network modes.
    """
    
    def __init__(self, target_ip="127.0.0.1", mode="sim", congestion_level=25, max_bitrate="500M"):
        super().__init__()
        self.target_ip = target_ip
        self.mode = mode
        self.congestion_level = congestion_level
        self.max_bitrate = max_bitrate
        
        # State: [norm_throughput, norm_delay, norm_loss, norm_cwnd, norm_delta_delay]
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        # Actions: 0 (Decr), 1 (Same), 2 (Incr)
        self.action_space = gym.spaces.Discrete(3)

        # CWND sizes (KB)
        self.window_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 2  # Start at 256KB
        self.step_count = 0
        self.last_delay = 10.0
        self.last_loss = 0
        self.last_action = 1
        self.history = []
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        action = int(action)
        
        # Update window index
        if action == 0: self.idx = max(0, self.idx - 1)
        elif action == 2: self.idx = min(len(self.window_sizes) - 1, self.idx + 1)
        
        current_window = self.window_sizes[self.idx]

        # Fetch Metrics
        throughput, delay, loss = self.fetch_metrics(current_window)
        
        delta_delay = delay - self.last_delay
        self.last_delay = delay

        # Normalization (Reset to 1000Mbps as we now throttle)
        norm_throughput = np.clip(throughput / 1000.0, 0, 1.0)
        norm_delay = np.clip((delay - 10) / 150.0, 0, 1.0)
        norm_loss = np.clip(loss / 5.0, 0, 1.0)
        norm_cwnd = (self.idx + 1) / len(self.window_sizes)
        norm_delta_delay = np.clip(delta_delay / 100.0, -1.0, 1.0)

        # Optimized PacerRL Reward
        throughput_score = 30.0 * np.log1p(norm_throughput)
        delay_penalty = 10.0 * norm_delay
        loss_penalty = 30.0 * norm_loss
        
        reward = throughput_score - delay_penalty - loss_penalty - 0.1 * abs(norm_delta_delay)
        
        if action != self.last_action: reward -= 0.05
        if throughput < 1.0: reward -= 2.0

        self.last_action = action
        state = np.array([norm_throughput, norm_delay, norm_loss, norm_cwnd, norm_delta_delay], dtype=np.float32)
        
        done = self.step_count >= 50
        return state, reward, done, False, {"tr": throughput, "de": delay, "lo": loss}

    def fetch_metrics(self, window):
        if self.mode == "sim":
            noise = (self.congestion_level / 100.0)
            max_cap = float(self.max_bitrate.replace('M', ''))
            throughput = min(max_cap, (window / 512.0) * (50 * (1 - noise)) + np.random.uniform(-5, 5))
            delay = 10 + (window / 256.0) * 5 + np.random.uniform(0, 5 + noise * 20)
            loss = 0 if window < 2048 else np.random.binomial(1, 0.05 + noise * 0.1)
            return max(0, throughput), delay, loss
        else:
            try:
                # Use bitrate flag -b to throttle the connection
                cmd = ["iperf3", "-c", self.target_ip, "-t", "1.0", "-w", f"{window}K", "-b", self.max_bitrate]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                tr_match = re.search(r'(\d+\.?\d*)\s*Mbits/sec.*sender', res.stdout)
                lo_match = re.search(r'(\d+)\s+sender', res.stdout)
                throughput = float(tr_match.group(1)) if tr_match else 0.0
                loss = int(lo_match.group(1)) if lo_match else 0
                
                p_cmd = ["ping", "-c", "1", "-W", "1", self.target_ip]
                res_p = subprocess.run(p_cmd, capture_output=True, text=True, timeout=2)
                d_match = re.search(r'time[=<]([\d\.]+)\s*ms', res_p.stdout)
                delay = float(d_match.group(1)) if d_match else 100.0
                return throughput, delay, loss
            except Exception as e:
                # Log error to terminal to help debugging
                print(f"⚠️ Real Network Error: {e}")
                return 0.0, 100.0, 0
