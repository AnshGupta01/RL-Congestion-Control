import argparse
from stable_baselines3 import PPO
from src.env import PacerEnv

def train(steps=50000, mode="sim", target_ip="127.0.0.1", model_name="pacer_rl_model"):
    """
    Train the PacerRL model using PPO.
    """
    print(f"🚀 Training PacerRL in {mode} mode for {steps} steps...")
    
    env = PacerEnv(target_ip=target_ip, mode=mode)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01
    )
    
    model.learn(total_timesteps=steps)
    model.save(model_name)
    print(f"\n✅ Training Complete. Model saved as '{model_name}.zip'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PacerRL Congestion Control Agent")
    parser.add_argument("--steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--mode", type=str, default="sim", choices=["sim", "real"], help="Training mode")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Target IP for real mode")
    parser.add_argument("--name", type=str, default="cc_ppo_optimized", help="Output model name")
    
    args = parser.parse_args()
    train(steps=args.steps, mode=args.mode, target_ip=args.ip, model_name=args.name)
