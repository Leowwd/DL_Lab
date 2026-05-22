#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Search for PPO Pendulum — Empirical Study on Key Parameters

Sweeps over:
  - epsilon (clipping parameter): controls how far the new policy can deviate
  - entropy_weight (entropy coefficient): encourages exploration

For each combination, a full PPO training run is launched with W&B logging.
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
import torch
import gymnasium as gym
import wandb
from ppo_pendulum import PPOAgent, seed_torch


def make_args_namespace(base_args: dict) -> argparse.Namespace:
    """Convert a dict to an argparse.Namespace (what PPOAgent expects)."""
    return argparse.Namespace(**base_args)


def run_single_experiment(config: dict, run_idx: int, total_runs: int) -> dict:
    """
    Run a single PPO training experiment with the given config.
    Returns a dict with the config and the best eval reward achieved.
    """
    eps = config["epsilon"]
    ent = config["entropy_weight"]
    seed = config["seed"]

    tag = f"eps{eps}_ent{ent}_seed{seed}"
    save_dir = os.path.join(config["save_root"], tag)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  [{run_idx}/{total_runs}]  epsilon={eps}  entropy_weight={ent}  seed={seed}")
    print(f"  Save dir: {save_dir}")
    print(f"{'='*70}\n")

    # Build the full args namespace expected by PPOAgent
    args = make_args_namespace({
        "discount_factor": config["discount_factor"],
        "tau": config["tau"],
        "batch_size": config["batch_size"],
        "epsilon": eps,
        "num_episodes": config["num_episodes"],
        "rollout_len": config["rollout_len"],
        "entropy_weight": ent,
        "seed": seed,
        "update_epoch": config["update_epoch"],
        "actor_lr": config["actor_lr"],
        "critic_lr": config["critic_lr"],
        "max_grad_norm": config["max_grad_norm"],
        "value_coef": config["value_coef"],
        "save_dir": save_dir,
    })

    # W&B init (each run gets its own wandb run)
    wandb.init(
        project=config["wandb_project"],
        name=f"grid_{tag}",
        group="grid_search",
        config=vars(args),
        reinit=True,
    )

    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)

    # Create env & agent
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    agent = PPOAgent(env, args)

    # Train — returns best_eval_reward thanks to our modification
    best_eval_reward = agent.train()

    # Load the best model and evaluate with seeds 0-19 (matching spec)
    best_model_path = os.path.join(save_dir, "ppo_pendulum_best.pt")
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

    # Save result metadata
    result = {
        "epsilon": eps,
        "entropy_weight": ent,
        "seed": seed,
        "best_eval_reward": float(best_eval_reward),
        "final_eval_reward": final_eval_reward,
        "save_dir": save_dir,
    }
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_summary_table(results: list):
    """Print a nicely formatted summary table sorted by final eval reward (spec-compliant)."""
    results_sorted = sorted(results, key=lambda r: r["final_eval_reward"], reverse=True)

    print(f"\n{'='*100}")
    print(f"  GRID SEARCH RESULTS -- sorted by final eval reward (seed 0-19 avg, descending)")
    print(f"{'='*100}")
    print(f"  {'Rank':<6} {'Epsilon':<10} {'Entropy':<10} {'Seed':<6} {'Final Eval (spec)':<20} {'Train Best':<14} {'Save Dir'}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*6} {'-'*20} {'-'*14} {'-'*30}")

    for i, r in enumerate(results_sorted, 1):
        marker = " *" if i == 1 else ""
        print(f"  {i:<6} {r['epsilon']:<10} {r['entropy_weight']:<10} {r['seed']:<6} "
              f"{r['final_eval_reward']:<20.2f} {r['best_eval_reward']:<14.2f} {r['save_dir']}{marker}")

    print(f"{'='*100}")
    best = results_sorted[0]
    print(f"\n  BEST CONFIG: epsilon={best['epsilon']}, entropy_weight={best['entropy_weight']}")
    print(f"     Final Eval Reward (seed 0-19 avg): {best['final_eval_reward']:.2f}")
    print(f"     Model saved at:   {best['save_dir']}/ppo_pendulum_best.pt\n")


def main():
    parser = argparse.ArgumentParser(description="Grid Search for PPO Pendulum")

    parser.add_argument(
        "--epsilons", type=float, nargs="+",
        default=[0.3],
        help="List of clipping parameter values to search"
    )
    parser.add_argument(
        "--entropy-weights", type=float, nargs="+",
        default=[0.0],
        help="List of entropy coefficient values to search"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[1, 2, 3, 42],
        help="List of random seeds (run each config with multiple seeds for robustness)"
    )

    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--update-epoch", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--value-coef", type=float, default=0.5)

    parser.add_argument(
        "--save-root", type=str,
        default="grid_search_results",
        help="Root directory for all grid search outputs"
    )
    parser.add_argument(
        "--wandb-project", type=str,
        default="DLP-Lab7-PPO-Pendulum-GridSearch",
        help="W&B project name for grid search"
    )

    args = parser.parse_args()

    # Generate all combinations
    combos = list(itertools.product(args.epsilons, args.entropy_weights, args.seeds))
    total_runs = len(combos)

    print(f"\nPPO Pendulum Grid Search")
    print(f"   Epsilons:        {args.epsilons}")
    print(f"   Entropy Weights: {args.entropy_weights}")
    print(f"   Seeds:           {args.seeds}")
    print(f"   Total runs:      {total_runs}")
    print(f"   Save root:       {args.save_root}")
    print(f"   Episodes/run:    {args.num_episodes}\n")

    os.makedirs(args.save_root, exist_ok=True)

    # Base config (shared across all runs)
    base_config = {
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "discount_factor": args.discount_factor,
        "num_episodes": args.num_episodes,
        "tau": args.tau,
        "batch_size": args.batch_size,
        "rollout_len": args.rollout_len,
        "update_epoch": args.update_epoch,
        "max_grad_norm": args.max_grad_norm,
        "value_coef": args.value_coef,
        "save_root": args.save_root,
        "wandb_project": args.wandb_project,
    }

    results = []
    for idx, (eps, ent, seed) in enumerate(combos, 1):
        config = {**base_config, "epsilon": eps, "entropy_weight": ent, "seed": seed}
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
