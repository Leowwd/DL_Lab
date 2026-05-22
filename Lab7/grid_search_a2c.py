#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Search for A2C Pendulum — Empirical Study on Key Parameters

Sweeps over:
  - actor_lr: actor learning rate
  - critic_lr: critic learning rate
  - entropy_weight: entropy coefficient for exploration
  - reward_scale: reward scaling factor
  - update_freq: gradient accumulation frequency

For each combination, a full A2C training run is launched with W&B logging.
Results are saved to a structured directory and a summary table is printed.
"""

import os
import sys
import json
import random
import itertools
import datetime
import argparse

import numpy as np
import torch.nn as nn
import torch
import gymnasium as gym
import wandb

from a2c_pendulum import A2CAgent, seed_torch


def make_args_namespace(base_args: dict) -> argparse.Namespace:
    """Convert a dict to an argparse.Namespace (what A2CAgent expects)."""
    return argparse.Namespace(**base_args)

def run_single_experiment(config: dict, run_idx: int, total_runs: int) -> dict:
    """
    Run a single A2C training experiment with the given config.
    Returns a dict with the config and the best eval reward achieved.
    """
    actor_lr = config["actor_lr"]
    critic_lr = config["critic_lr"]
    ent = config["entropy_weight"]
    reward_scale = config["reward_scale"]
    update_freq = config["update_freq"]
    seed = config["seed"]

    tag = f"alr{actor_lr}_clr{critic_lr}_ent{ent}_rs{reward_scale}_uf{update_freq}_seed{seed}"
    save_dir = os.path.join(config["save_root"], tag)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  [{run_idx}/{total_runs}]  actor_lr={actor_lr}  critic_lr={critic_lr}  "
          f"entropy={ent}  reward_scale={reward_scale}  update_freq={update_freq}  seed={seed}")
    print(f"  Save dir: {save_dir}")
    print(f"{'='*80}\n")

    # Build the full args namespace expected by A2CAgent
    args = make_args_namespace({
        "discount_factor": config["discount_factor"],
        "num_episodes": config["num_episodes"],
        "entropy_weight": ent,
        "seed": seed,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "max_grad_norm": config["max_grad_norm"],
        "update_freq": update_freq,
        "reward_scale": reward_scale,
        "save_dir": save_dir,
    })

    # W&B init (each run gets its own wandb run)
    wandb.init(
        project=config["wandb_project"],
        name=f"grid_{tag}",
        group="grid_search_a2c",
        config=vars(args),
        reinit=True,
    )

    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)

    # Create env & agent
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    agent = A2CAgent(env, args)

    original_train = agent.train

    def patched_train():
        """Train with save paths redirected to save_dir."""
        agent.is_test = False
        step_count = 0
        best_eval_reward = -float('inf')

        agent.actor_optimizer.zero_grad()
        agent.critic_optimizer.zero_grad()

        state, _ = agent.env.reset(seed=agent.seed)
        from tqdm import tqdm
        for ep in tqdm(range(1, agent.num_episodes)):
            if ep > 1:
                state, _ = agent.env.reset()
            score = 0
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done = agent.step(action)

                actor_loss, critic_loss = agent.update_model()

                step_count += 1
                if step_count % agent.update_freq == 0 or done:
                    nn.utils.clip_grad_norm_(agent.actor.parameters(), agent.max_grad_norm)
                    nn.utils.clip_grad_norm_(agent.critic.parameters(), agent.max_grad_norm)
                    agent.actor_optimizer.step()
                    agent.critic_optimizer.step()
                    agent.actor_optimizer.zero_grad()
                    agent.critic_optimizer.zero_grad()

                state = next_state
                score += reward

                wandb.log({
                    "charts/global_step": step_count,
                    "losses/actor_loss": actor_loss,
                    "losses/critic_loss": critic_loss,
                })

                if done:
                    print(f"Episode {ep}: Total Reward = {score:.2f}")
                    wandb.log({
                        "charts/global_step": step_count,
                        "charts/episodic_return": score,
                        "episode": ep,
                    })

            # Periodic evaluation
            if ep % 10 == 0:
                eval_env = gym.make("Pendulum-v1")
                eval_reward = agent.evaluate(eval_env, n_episodes=20)
                eval_env.close()
                print(f"  [Eval] Episode {ep}, Avg Reward: {eval_reward:.2f}")
                wandb.log({
                    "charts/global_step": step_count,
                    "charts/eval_reward": eval_reward,
                })

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    torch.save({
                        'actor_state_dict': agent.actor.state_dict(),
                        'critic_state_dict': agent.critic.state_dict(),
                    }, os.path.join(save_dir, "a2c_pendulum_best.pt"))
                    print(f"  [Save] New best model: {eval_reward:.2f}")

        # Save final model
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
        }, os.path.join(save_dir, "a2c_pendulum_final.pt"))

        return best_eval_reward

    best_eval_reward = patched_train()

    best_model_path = os.path.join(save_dir, "a2c_pendulum_best.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=agent.device)
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])

    eval_env = gym.make("Pendulum-v1")
    agent.is_test = True
    eval_rewards = []
    for eval_seed in range(20):
        state, _ = eval_env.reset(seed=eval_seed)
        done = False
        ep_reward = 0
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward
        eval_rewards.append(ep_reward)
    eval_env.close()
    agent.is_test = False

    final_eval_reward = float(np.mean(eval_rewards))
    print(f"  [Final Eval] seed 0-19 avg reward: {final_eval_reward:.2f}")

    wandb.log({
        "grid/best_eval_reward": best_eval_reward,
        "grid/final_eval_reward": final_eval_reward,
    })
    wandb.finish()

    result = {
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "entropy_weight": ent,
        "reward_scale": reward_scale,
        "update_freq": update_freq,
        "seed": seed,
        "best_eval_reward": float(best_eval_reward),
        "final_eval_reward": final_eval_reward,
        "save_dir": save_dir,
    }
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_summary_table(results: list):
    """Print a nicely formatted summary table sorted by final eval reward."""
    results_sorted = sorted(results, key=lambda r: r["final_eval_reward"], reverse=True)

    print(f"\n{'='*120}")
    print(f"  A2C GRID SEARCH RESULTS -- sorted by final eval reward (seed 0-19 avg, descending)")
    print(f"{'='*120}")
    print(f"  {'Rank':<6} {'ActorLR':<10} {'CriticLR':<10} {'Entropy':<10} {'RwdScale':<10} "
          f"{'UpdFreq':<10} {'Seed':<6} {'Final Eval':<14} {'Train Best':<14}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*14} {'-'*14}")

    for i, r in enumerate(results_sorted, 1):
        marker = " ★" if i == 1 else ""
        print(f"  {i:<6} {r['actor_lr']:<10} {r['critic_lr']:<10} {r['entropy_weight']:<10} "
              f"{r['reward_scale']:<10} {r['update_freq']:<10} {r['seed']:<6} "
              f"{r['final_eval_reward']:<14.2f} {r['best_eval_reward']:<14.2f}{marker}")

    print(f"{'='*120}")
    best = results_sorted[0]
    print(f"\n  BEST CONFIG: actor_lr={best['actor_lr']}, critic_lr={best['critic_lr']}, "
          f"entropy={best['entropy_weight']}, reward_scale={best['reward_scale']}, "
          f"update_freq={best['update_freq']}")
    print(f"     Final Eval Reward (seed 0-19 avg): {best['final_eval_reward']:.2f}")
    print(f"     Model saved at:   {best['save_dir']}/a2c_pendulum_best.pt\n")


def main():
    parser = argparse.ArgumentParser(description="Grid Search for A2C Pendulum")

    parser.add_argument(
        "--actor-lrs", type=float, nargs="+",
        default=[3e-4],
        help="List of actor learning rates to search"
    )
    parser.add_argument(
        "--critic-lrs", type=float, nargs="+",
        default=[1e-3],
        help="List of critic learning rates to search"
    )
    parser.add_argument(
        "--entropy-weights", type=float, nargs="+",
        default=[0.1],
        help="List of entropy coefficient values to search"
    )
    parser.add_argument(
        "--reward-scales", type=float, nargs="+",
        default=[0.2],
        help="List of reward scaling factors to search"
    )
    parser.add_argument(
        "--update-freqs", type=int, nargs="+",
        default=[48],
        help="List of gradient accumulation frequencies to search"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[2, 3, 41, 42],
        help="List of random seeds"
    )

    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=4000)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument(
        "--save-root", type=str,
        default="grid_search_a2c_results",
        help="Root directory for all grid search outputs"
    )
    parser.add_argument(
        "--wandb-project", type=str,
        default="DLP-Lab7-A2C-Pendulum-GridSearch",
        help="W&B project name for grid search"
    )

    args = parser.parse_args()

    # Generate all combinations
    combos = list(itertools.product(
        args.actor_lrs, args.critic_lrs, args.entropy_weights,
        args.reward_scales, args.update_freqs, args.seeds
    ))
    total_runs = len(combos)

    print(f"\nA2C Pendulum Grid Search")
    print(f"   Actor LRs:        {args.actor_lrs}")
    print(f"   Critic LRs:       {args.critic_lrs}")
    print(f"   Entropy Weights:  {args.entropy_weights}")
    print(f"   Reward Scales:    {args.reward_scales}")
    print(f"   Update Freqs:     {args.update_freqs}")
    print(f"   Seeds:            {args.seeds}")
    print(f"   Total runs:       {total_runs}")
    print(f"   Episodes/run:     {args.num_episodes}\n")

    os.makedirs(args.save_root, exist_ok=True)

    base_config = {
        "discount_factor": args.discount_factor,
        "num_episodes": args.num_episodes,
        "max_grad_norm": args.max_grad_norm,
        "save_root": args.save_root,
        "wandb_project": args.wandb_project,
    }

    results = []
    for idx, (alr, clr, ent, rs, uf, seed) in enumerate(combos, 1):
        config = {
            **base_config,
            "actor_lr": alr,
            "critic_lr": clr,
            "entropy_weight": ent,
            "reward_scale": rs,
            "update_freq": uf,
            "seed": seed,
        }
        result = run_single_experiment(config, idx, total_runs)
        results.append(result)

        # Save intermediate results in case of crash
        with open(os.path.join(args.save_root, "all_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    print_summary_table(results)

    # Save final summary
    with open(os.path.join(args.save_root, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Full results saved to: {args.save_root}/all_results.json")


if __name__ == "__main__":
    main()
