<p align="center">
  <img src="public/PacerRL-logo.png" width="90" style="border-radius: 16px;" />
</p>

<h1 align="center">PacerRL</h1>

<p align="center">
  <strong>Built for</strong><br/>
  <a href="#" target="_blank">
    <img src="https://img.icons8.com/color/48/artificial-intelligence.png" width="48" style="vertical-align: middle;" />
    <br/>
    <strong>Advanced Networking Research</strong>
  </a>
</p>

<p align="center">
  A modern, AI-driven congestion control framework designed to optimize TCP windowing for high-speed retail and enterprise networks.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Stable_Baselines3-PPO-FF6F00?logo=pytorch&logoColor=white" alt="SB3" />
  <img src="https://img.shields.io/badge/Gymnasium-v1.x-00A99D?logo=openai&logoColor=white" alt="Gymnasium" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/iperf3-Benchmark-000000?logo=linux&logoColor=white" alt="iperf3" />
</p>

<p align="center">
  <!-- Screenshot placeholder -->
  <img src="" alt="PacerRL Dashboard Screenshot" width="100%" />
</p>

---

## 🏗️ Project Structure

- **`src/env.py`**: The core `PacerEnv` gymnasium environment. Supports both **Real Network** (iperf3) and **Simulation** modes.
- **`dashboard.py`**: A premium Streamlit dashboard for real-time comparison between PacerRL and standard TCP (AIMD).
- **`train_pacer.py`**: A clean CLI script for training new PacerRL models.
- **`cc_ppo_optimized.zip`**: The latest trained PacerRL model.

## 📊 Core Metrics

PacerRL optimizes for the "Golden Triangle" of networking:
1.  **Throughput**: Maximize bits per second.
2.  **Delay (RTT)**: Minimize bufferbloat and lag.
3.  **Loss**: Maintain connection reliability.
4.  **Efficiency Score**: A "Power" metric calculated as `Throughput / Delay`.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following installed:
- Python 3.9+
- `iperf3` (for Real Network mode)
- `ping` (for latency measurement)

### 2. Installation
```bash
pip install stable-baselines3 gymnasium numpy pandas streamlit
```

### 3. Running the Dashboard
The easiest way to see PacerRL in action:
```bash
streamlit run dashboard.py
```

## 🛠️ Training PacerRL

### Simulation Mode (Fast)
Recommended for initial testing and demo preparation:
```bash
python3 train_pacer.py --steps 50000 --mode sim --name my_pacer_model
```

---
Developed with ❤️ for Advanced Congestion Control Research
