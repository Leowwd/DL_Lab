#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
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
from typing import Tuple

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

def orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)):
    """Orthogonal initialization for linear layers."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
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
        log_std = torch.clamp(self.log_std, -20, 2.0)
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
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.max_grad_norm = getattr(args, 'max_grad_norm', 0.5)
        self.update_freq = getattr(args, 'update_freq', 32)  # gradient accumulation frequency
        self.reward_scale = getattr(args, 'reward_scale', 0.1)  # reward scaling factor
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action
        clipped_action = selected_action.clamp(-2.0, 2.0)

        if not self.is_test:
            log_prob = dist.log_prob(clipped_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return clipped_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, float(terminated)])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        state, log_prob, next_state, reward, done = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        
        ############TODO#############
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        
        scaled_reward = reward * self.reward_scale
        
        # Compute current value and next value
        curr_value = self.critic(state)
        next_value = self.critic(next_state_t).detach()
        
        # TD target: r_scaled + \gamma * V(s') * mask
        td_target = scaled_reward + self.gamma * next_value * mask
        
        # Critic loss: (TD target - V(s))^2
        value_loss = F.mse_loss(curr_value, td_target.detach())
        #############################

        # Accumulate critic gradient
        value_loss.backward()

        # advantage = Q_t - V(s_t)
        ############TODO#############
        # Recompute advantage (detached) for actor update
        advantage = (td_target - curr_value).detach()
        
        # Compute entropy bonus: H = 0.5 * (1 + log(2*\pi * sigma^2)) per dimension
        _, dist = self.actor(state)
        entropy = dist.entropy().sum(dim=-1)
        
        # Actor loss: -E[log \pi(a|s) * A(s,a)] - c_entropy * H
        policy_loss = -(log_prob * advantage + self.entropy_weight * entropy)
        #############################

        # Accumulate actor gradient (no step yet!)
        policy_loss.backward()

        return policy_loss.item(), value_loss.item()

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
        """Train the agent."""
        self.is_test = False
        step_count = 0
        best_eval_reward = -float('inf')
        
        # Zero gradients once at the beginning for gradient accumulation
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        state, _ = self.env.reset(seed=self.seed)
        for ep in tqdm(range(1, self.num_episodes)):
            actor_losses, critic_losses, scores = [], [], []
            if ep > 1:
                state, _ = self.env.reset()
            score = 0
            done = False
            while not done:
                # self.env.render()
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                step_count += 1
                if step_count % self.update_freq == 0 or done:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                state = next_state
                score += reward
                # W&B logging
                wandb.log({
                    "charts/global_step": step_count,
                    "losses/actor_loss": actor_loss,
                    "losses/critic_loss": critic_loss,
                    }) 
                # if episode ends
                if done:
                    scores.append(score)
                    print(f"Episode {ep}: Total Reward = {score:.2f}")
                    # W&B logging
                    wandb.log({
                        "charts/global_step": step_count,
                        "charts/episodic_return": score,
                        "episode": ep,
                        })  
            
            # Periodic evaluation
            if ep % 10 == 0:
                eval_env = gym.make("Pendulum-v1")
                eval_reward = self.evaluate(eval_env, n_episodes=20)
                eval_env.close()
                print(f"  [Eval] Episode {ep}, Avg Reward: {eval_reward:.2f}")
                wandb.log({
                    "charts/global_step": step_count,
                    "charts/eval_reward": eval_reward,
                })
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    torch.save({
                        'actor_state_dict': self.actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                    }, "a2c_pendulum_best.pt")
                    print(f"  [Save] New best model: {eval_reward:.2f}")
        
        # Save final model
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, "a2c_pendulum_final.pt")

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
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--entropy-weight", type=float, default=0.0025)
    parser.add_argument("--update-freq", type=int, default=16) 
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True,
               config=vars(args))
    
    agent = A2CAgent(env, args)
    agent.train()