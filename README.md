# RL Congestion Control

A reinforcement learning prototype for congestion control using PPO (`stable-baselines3`) on a custom `gymnasium` environment.

The agent adjusts TCP window size and learns from:
- Throughput from `iperf3`
- Delay (RTT) from `ping`

## Project Structure

- `env.py` - Custom RL environment (`CongestionEnv`)
- `train.py` - PPO training script
- `test_model.py` - Loads a trained model and prints throughput/delay stats
- `plot.py` - Generates training metric plots from saved history
- `history.npy` - Saved `(throughput, delay, reward)` samples
- `plots/` - Output folder for generated plots

## How It Works

The environment action space is discrete:
- `0` -> decrease window size
- `1` -> keep same window size
- `2` -> increase window size

Window sizes (KB): `[64, 128, 256, 512, 1024]`

State (normalized):
- Throughput / 50
- Delay / 200

Reward:
- `reward = norm_throughput - 0.5 * norm_delay`

## Requirements

- Python 3.9+
- `iperf3` installed and accessible in PATH
- `ping` available
- Reachable test host at `10.0.0.1`

Python packages:
- `stable-baselines3`
- `gymnasium`
- `numpy`
- `matplotlib`

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install stable-baselines3 gymnasium numpy matplotlib
```

3. Ensure network tools are installed:

```bash
# Debian/Ubuntu
sudo apt update
sudo apt install -y iperf3 iputils-ping
```

## Network Testbed Prerequisite

This project expects an `iperf3` server running at `10.0.0.1`.

On the server machine:

```bash
iperf3 -s
```

From the training machine, verify connectivity:

```bash
ping -c 3 10.0.0.1
iperf3 -c 10.0.0.1 -t 1
```

## Training

Run:

```bash
python train.py
```

This will:
- Train PPO for `200` timesteps
- Save history to `history.npy`
- Save model to `cc_rl_model_2.zip`

## Evaluate Trained Model

Run:

```bash
python test_model.py
```

The script loads `cc_rl_model_2` and prints per-step:
- Throughput
- Delay
- Loss (if available from parser)

It also prints average throughput and delay.

## Plot Results

Run:

```bash
python plot.py
```

Generated files:
- `plots/throughput.png`
- `plots/delay.png`
- `plots/reward.png`

## Notes

- Throughput and delay are normalized with fixed constants (`50 Mbps`, `200 ms`).
- The current code uses a hardcoded endpoint (`10.0.0.1`) in both `iperf3` and `ping` calls.
- If command outputs differ by OS/version, regex parsing in `env.py` may need adjustment.

## Troubleshooting

- `FileNotFoundError: iperf3`:
  - Install `iperf3` and ensure it is in PATH.
- Throughput always `0`:
  - Check server IP, firewall rules, and that `iperf3 -s` is running.
- Delay always `0`:
  - Verify `ping` output format matches the regex in `get_delay()`.
- Model load failure in `test_model.py`:
  - Ensure training completed and `cc_rl_model_2.zip` exists.
