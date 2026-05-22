import argparse
import os
os.environ["MUJOCO_GL"] = "egl"
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import imageio
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Actor, self).__init__()
        self.hidden1 = nn.Linear(in_dim, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.mu_layer = nn.Linear(64, out_dim)
        self.log_std = nn.Parameter(torch.full((out_dim,), -0.5))

    def forward(self, state: torch.Tensor):
        x = torch.tanh(self.hidden1(state))
        x = torch.tanh(self.hidden2(x))
        mu = self.mu_layer(x)
        
        std = torch.exp(torch.clamp(self.log_std, -20.0, 2.0))
        dist = Normal(mu, std)
        return dist.mean

def normalize_obs(obs, mean, var):
    normalized = (obs - mean) / np.sqrt(var + 1e-8)
    return np.clip(normalized, -10.0, 10.0)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = Actor(17, 6).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    obs_rms_mean = checkpoint['obs_rms_mean']
    obs_rms_var = checkpoint['obs_rms_var']

    os.makedirs(args.output_dir, exist_ok=True)
    total = 0

    for seed in range(args.episodes):
        env = gym.make("Walker2d-v5", render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)
        obs = normalize_obs(obs, obs_rms_mean, obs_rms_var)
        done = False
        episode_reward = 0
        frames = []

        while not done:
            frame = env.render()
            frames.append(frame)

            obs_tensor = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                action = actor(obs_tensor).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            obs = next_obs
            obs = normalize_obs(obs, obs_rms_mean, obs_rms_var)

        out_path = os.path.join(args.output_dir, f"seed{seed}.mp4")
        with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
            for f in frames:
                video.append_data(f)

        print(f"seed: {seed}, eval reward: {episode_reward:.2f} → {out_path}")
        total += episode_reward
        env.close()

    print(f"\nAverage reward over {args.episodes} episodes: {total / args.episodes:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output_dir", type=str, default="./videos_task3")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    evaluate(args)