import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import imageio
import os
import argparse

class DQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
           nn.Linear(num_states, 128),
           nn.ReLU(),
           nn.Linear(128, 128),
           nn.ReLU(),
           nn.Linear(128, num_actions)
        )       

    def forward(self, x):
        return self.network(x)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]

    model = DQN(num_states, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    total = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = obs
        done = False
        total_reward = 0
        frames = []

        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_obs

        out_path = os.path.join(args.output_dir, f"task1_eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
            for f in frames:
                video.append_data(f)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")
        total += total_reward
        
    print(f"Average reward over {args.episodes} episodes: {total / args.episodes:.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation")
    args = parser.parse_args()
    evaluate(args)