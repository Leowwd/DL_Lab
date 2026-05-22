#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import os
import random
from collections import deque
from typing import Deque, List, Tuple

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

def orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)):
    """Orthogonal initialization for linear layers."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 2,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.hidden1 = nn.Linear(in_dim, 128)
        orthogonal_init(self.hidden1, gain=np.sqrt(2))
        
        self.hidden2 = nn.Linear(128, 128)
        orthogonal_init(self.hidden2, gain=np.sqrt(2))
        
        self.mu_layer = nn.Linear(128, out_dim)
        orthogonal_init(self.mu_layer, gain=0.01)
        
        # Learnable log_std parameter (state-independent)
        self.log_std = nn.Parameter(torch.full((out_dim,), -0.5))
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = torch.tanh(self.hidden1(state))
        x = torch.tanh(self.hidden2(x))
        mu = 2.0 * torch.tanh(self.mu_layer(x))
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        action = dist.sample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.hidden1 = nn.Linear(in_dim, 128)
        orthogonal_init(self.hidden1, gain=np.sqrt(2))
        
        self.hidden2 = nn.Linear(128, 128)
        orthogonal_init(self.hidden2, gain=np.sqrt(2))
        
        self.out = nn.Linear(128, 1)
        orthogonal_init(self.out, gain=1.0)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = torch.tanh(self.hidden1(state))
        x = torch.tanh(self.hidden2(x))
        value = self.out(x)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    values = values + [next_value]
    gae = 0
    gae_returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns.insert(0, gae + values[step])
    #############################
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.max_grad_norm = getattr(args, 'max_grad_norm', 0.5)
        self.value_coef = getattr(args, 'value_coef', 0.5)
        self.save_dir = getattr(args, 'save_dir', '.')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action).sum(dim=-1))

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        
        terminated_arr = np.reshape(terminated, (1, -1)).astype(np.float32)

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1.0 - terminated_arr).to(self.device))

        done = np.reshape(terminated or truncated, (1, -1))
        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions).detach()
        returns = torch.cat(returns).detach().squeeze(-1)
        values = torch.cat(self.values).detach().squeeze(-1)
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        actor_losses, critic_losses = [], []
        total_clip_fracs = []
        total_approx_kls = []
        total_entropies = []

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action).sum(dim=-1)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            ############TODO#############
            # Clipped surrogate objective
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            
            # Entropy bonus
            entropy = dist.entropy().sum(dim=-1).mean()
            
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_weight * entropy
            #############################

            # critic_loss
            ############TODO#############
            # Value function loss with optional clipping
            value_pred = self.critic(state).squeeze(-1)
            # Clipped value loss
            value_pred_clipped = old_value + torch.clamp(
                value_pred - old_value, -self.epsilon, self.epsilon
            )
            value_loss_unclipped = (value_pred - return_).pow(2)
            value_loss_clipped = (value_pred_clipped - return_).pow(2)
            critic_loss = self.value_coef * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            #############################
            
            # Diagnostics
            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > self.epsilon).float().mean().item()
                approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                total_clip_fracs.append(clip_frac)
                total_approx_kls.append(approx_kl)
                total_entropies.append(entropy.item())
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
        avg_clip_frac = sum(total_clip_fracs) / len(total_clip_fracs) if total_clip_fracs else 0
        avg_approx_kl = sum(total_approx_kls) / len(total_approx_kls) if total_approx_kls else 0
        avg_entropy = sum(total_entropies) / len(total_entropies) if total_entropies else 0

        return actor_loss, critic_loss, avg_clip_frac, avg_approx_kl, avg_entropy

    def evaluate(self, env, n_episodes=20, seed_start=0):
        """Evaluate the agent over n_episodes with deterministic policy."""
        self.is_test = True
        rewards = []
        for seed in range(seed_start, seed_start + n_episodes):
            state, _ = env.reset(seed=seed)
            done = False
            episode_reward = 0
            while not done:
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)
        self.is_test = False
        return np.mean(rewards)

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        best_eval_reward = -float('inf')
        
        for ep in tqdm(range(1, self.num_episodes)):
            score = 0
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    print(f"\nEpisode {episode_count}: Total Reward = {score:.2f}")
                    
                    wandb.log({
                        "charts/global_step": self.total_step,
                        "charts/episodic_return": score,
                    })
                    score = 0

            actor_loss, critic_loss, clip_frac, approx_kl, entropy = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            # W&B logging per update
            wandb.log({
                "charts/global_step": self.total_step,
                "losses/actor_loss": actor_loss,
                "losses/critic_loss": critic_loss,
                "losses/clip_fraction": clip_frac,
                "losses/approx_kl": approx_kl,
                "losses/entropy": entropy,
            })
            
            # Periodic evaluation
            if ep % 10 == 0:
                eval_env = gym.make("Pendulum-v1")
                eval_reward = self.evaluate(eval_env, n_episodes=20)
                eval_env.close()
                print(f"  [Eval] Update {ep}, Avg Reward: {eval_reward:.2f}")
                wandb.log({
                    "charts/global_step": self.total_step,
                    "charts/eval_reward": eval_reward,
                })
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    torch.save({
                        'actor_state_dict': self.actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                    }, os.path.join(self.save_dir, "ppo_pendulum_best.pt"))
                    print(f"  [Save] New best model: {eval_reward:.2f}")

        # Save final model
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, os.path.join(self.save_dir, "ppo_pendulum_final.pt"))
        
        return best_eval_reward
        
        # termination
        self.env.close()

    def test(self, video_folder: str):
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = tmp_env
 
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--entropy-weight", type=float, default=0.003) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=384)
    parser.add_argument("--update-epoch", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--value-coef", type=float, default=0.7)
    parser.add_argument("--save-dir", type=str, default=".", help="Directory to save model checkpoints")
    args = parser.parse_args()
 
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-PPO-Pendulum", name=args.wandb_run_name, save_code=True,
               config=vars(args))
    
    agent = PPOAgent(env, args)
    agent.train()
