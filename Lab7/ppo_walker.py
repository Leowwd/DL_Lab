#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
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


class RunningMeanStd:
    """Running mean and standard deviation tracker for observation normalization."""
    def __init__(self, shape=(), epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x):
        """Update running statistics with a batch of observations."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch moments using parallel algorithm."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m_2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count


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
        
        self.hidden1 = nn.Linear(in_dim, 64)
        orthogonal_init(self.hidden1, gain=np.sqrt(2))
        
        self.hidden2 = nn.Linear(64, 64)
        orthogonal_init(self.hidden2, gain=np.sqrt(2))
        
        self.mu_layer = nn.Linear(64, out_dim)
        orthogonal_init(self.mu_layer, gain=0.01)
        
        # Learnable log_std parameter (state-independent)
        self.log_std = nn.Parameter(torch.full((out_dim,), -0.5))
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = torch.tanh(self.hidden1(state))
        x = torch.tanh(self.hidden2(x))
        mu = self.mu_layer(x)
        
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        action = dist.rsample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.hidden1 = nn.Linear(in_dim, 64)
        orthogonal_init(self.hidden1, gain=np.sqrt(2))
        
        self.hidden2 = nn.Linear(64, 64)
        orthogonal_init(self.hidden2, gain=np.sqrt(2))
        
        self.out = nn.Linear(64, 1)
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
        self.total_timesteps = args.total_timesteps
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.max_grad_norm = getattr(args, 'max_grad_norm', 0.5)
        self.value_coef = getattr(args, 'value_coef', 0.5)
        self.n_envs = args.n_envs
        self.lr = args.lr
        self.anneal_lr = getattr(args, 'anneal_lr', True)
        self.save_dir = getattr(args, 'save_dir', '.')
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.single_observation_space.shape[0]
        self.action_dim = env.single_action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # Single optimizer for both actor and critic (common in PPO MuJoCo)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr, eps=1e-5
        )

        # Observation and reward normalization
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.ret_rms = RunningMeanStd(shape=())
        self.returns_for_norm = np.zeros(self.n_envs)

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def normalize_obs(self, obs, update=True):
        """Normalize observations using running mean/std."""
        if update:
            self.obs_rms.update(obs)
        normalized = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(normalized, -10.0, 10.0)
    
    def normalize_reward(self, rewards):
        """Normalize rewards using running return std."""
        self.returns_for_norm = self.returns_for_norm * self.gamma + rewards
        self.ret_rms.update(self.returns_for_norm.reshape(-1))
        return rewards / np.sqrt(self.ret_rms.var + 1e-8)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            return selected_action.cpu().detach().numpy(), log_prob.cpu().detach(), value.cpu().detach()

        return selected_action.cpu().detach().numpy()

    def update_model(self, 
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     old_values: torch.Tensor,
                     returns: torch.Tensor,
                     advantages: torch.Tensor,
                     ) -> Tuple[float, float, float, float, float]:
        """Update the model by gradient descent."""
        
        batch_size = states.size(0)
        actor_losses, critic_losses = [], []
        total_clip_fracs = []
        total_approx_kls = []
        total_entropies = []

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(self.update_epoch):
            # Shuffle indices for each epoch
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, self.batch_size):
                end = start + self.batch_size
                if end > batch_size:
                    break
                mb_indices = indices[start:end]
                
                state = states[mb_indices].to(self.device)
                action = actions[mb_indices].to(self.device)
                old_log_prob = old_log_probs[mb_indices].to(self.device)
                old_value = old_values[mb_indices].to(self.device)
                return_ = returns[mb_indices].to(self.device)
                adv = advantages[mb_indices].to(self.device)
                
                # Forward pass
                _, dist = self.actor(state)
                log_prob = dist.log_prob(action).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                value_pred = self.critic(state).squeeze(-1)
                
                # Ratio
                ratio = (log_prob - old_log_prob).exp()
                
                # Clipped surrogate objective
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss with clipping
                value_pred_clipped = old_value + torch.clamp(
                    value_pred - old_value, -self.epsilon, self.epsilon
                )
                value_loss_unclipped = (value_pred - return_).pow(2)
                value_loss_clipped = (value_pred_clipped - return_).pow(2)
                critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_weight * entropy
                
                # Diagnostics
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.epsilon).float().mean().item()
                    approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                    total_clip_fracs.append(clip_frac)
                    total_approx_kls.append(approx_kl)
                    total_entropies.append(entropy.item())
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0
        avg_clip_frac = sum(total_clip_fracs) / len(total_clip_fracs) if total_clip_fracs else 0
        avg_approx_kl = sum(total_approx_kls) / len(total_approx_kls) if total_approx_kls else 0
        avg_entropy = sum(total_entropies) / len(total_entropies) if total_entropies else 0

        return avg_actor_loss, avg_critic_loss, avg_clip_frac, avg_approx_kl, avg_entropy

    def _make_checkpoint(self, path):
        """Save a checkpoint with all necessary state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'obs_rms_mean': self.obs_rms.mean,
            'obs_rms_var': self.obs_rms.var,
            'obs_rms_count': self.obs_rms.count,
        }, path)

    def train(self):
        """Train the PPO agent with vectorized environments."""
        self.is_test = False
        
        num_updates = self.total_timesteps // (self.rollout_len * self.n_envs)
        best_eval_reward = -float('inf')
        
        milestones = [1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000]
        milestone_best_reward = {}   # milestone_step -> best eval reward seen so far in that bracket
        milestone_idx = 0            # which bracket we are currently in
        
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Reset all envs
        obs, _ = self.env.reset(seed=self.seed)
        obs = self.normalize_obs(obs)
        
        for update in tqdm(range(1, num_updates + 1)):
            # Learning rate annealing
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lr_now = frac * self.lr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr_now
            
            # Storage for rollout
            mb_states = []
            mb_actions = []
            mb_log_probs = []
            mb_rewards = []
            mb_values = []
            mb_dones = []
            
            # Collect rollout
            for step in range(self.rollout_len):
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                with torch.no_grad():
                    action, dist = self.actor(obs_tensor)
                    value = self.critic(obs_tensor).squeeze(-1)
                    log_prob = dist.log_prob(action).sum(dim=-1)
                
                action_np = action.cpu().numpy()
                
                # Step environment
                next_obs, rewards, terminated, truncated, infos = self.env.step(action_np)
                dones = np.logical_or(terminated, truncated)
                
                # Normalize reward
                normalized_rewards = self.normalize_reward(rewards)
                
                # Store
                mb_states.append(torch.FloatTensor(obs))
                mb_actions.append(action.cpu())
                mb_log_probs.append(log_prob.cpu())
                mb_rewards.append(torch.FloatTensor(normalized_rewards))
                mb_values.append(value.cpu())
                mb_dones.append(torch.FloatTensor(terminated.astype(np.float32)))
                
                # Reset return normalizer for done envs
                self.returns_for_norm[dones] = 0.0
                
                # Normalize next obs
                obs = self.normalize_obs(next_obs)
                
                self.total_step += self.n_envs
                
                # Log episode returns from vectorized env
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info is not None and "episode" in info:
                            ep_return = info["episode"]["r"]
                            ep_length = info["episode"]["l"]
                            wandb.log({
                                "charts/global_step": self.total_step,
                                "charts/episodic_return": ep_return,
                                "charts/episode_length": ep_length,
                            })
                            print(f"\n  Step {self.total_step}: Episode Return = {ep_return:.2f}")
            
            # Compute GAE
            with torch.no_grad():
                last_value = self.critic(torch.FloatTensor(obs).to(self.device)).squeeze(-1).cpu()
            
            # Stack tensors
            mb_states = torch.stack(mb_states)        # (T, n_envs, obs_dim)
            mb_actions = torch.stack(mb_actions)      # (T, n_envs, act_dim)
            mb_log_probs = torch.stack(mb_log_probs)  # (T, n_envs)
            mb_rewards = torch.stack(mb_rewards)      # (T, n_envs)
            mb_values = torch.stack(mb_values)        # (T, n_envs)
            mb_dones = torch.stack(mb_dones)          # (T, n_envs)
            
            # GAE computation
            mb_advantages = torch.zeros_like(mb_rewards)
            last_gae = 0
            for t in reversed(range(self.rollout_len)):
                if t == self.rollout_len - 1:
                    next_non_terminal = 1.0 - mb_dones[t]
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - mb_dones[t]
                    next_values = mb_values[t + 1]
                
                delta = mb_rewards[t] + self.gamma * next_values * next_non_terminal - mb_values[t]
                mb_advantages[t] = last_gae = delta + self.gamma * self.tau * next_non_terminal * last_gae
            
            mb_returns = mb_advantages + mb_values
            
            # Flatten batch
            batch_states = mb_states.reshape(-1, self.obs_dim)
            batch_actions = mb_actions.reshape(-1, self.action_dim)
            batch_log_probs = mb_log_probs.reshape(-1)
            batch_values = mb_values.reshape(-1)
            batch_returns = mb_returns.reshape(-1)
            batch_advantages = mb_advantages.reshape(-1)
            
            # Update
            actor_loss, critic_loss, clip_frac, approx_kl, entropy = self.update_model(
                batch_states, batch_actions, batch_log_probs, batch_values,
                batch_returns, batch_advantages
            )
            
            # W&B logging per update
            wandb.log({
                "charts/global_step": self.total_step,
                "losses/actor_loss": actor_loss,
                "losses/critic_loss": critic_loss,
                "losses/clip_fraction": clip_frac,
                "losses/approx_kl": approx_kl,
                "losses/entropy": entropy,
                "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            })
            
            print(f"\nUpdate {update}/{num_updates}, Step {self.total_step}, "
                  f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
                  f"Clip Frac: {clip_frac:.3f}, Approx KL: {approx_kl:.4f}")
            
            # Periodic evaluation
            if update % 5 == 0:
                eval_reward = self.evaluate_walker(n_episodes=10)
                print(f"  [Eval] Update {update}, Step {self.total_step}, Avg Reward: {eval_reward:.2f}")
                wandb.log({
                    "charts/global_step": self.total_step,
                    "charts/eval_reward": eval_reward,
                })
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self._make_checkpoint(os.path.join(save_dir, "ppo_walker_best.pt"))
                    print(f"  [Save] New global best model: {eval_reward:.2f}")
                
                current_milestone = None
                for ms in milestones:
                    if self.total_step <= ms:
                        current_milestone = ms
                        break
                if current_milestone is None:
                    current_milestone = "over_3M"
                
                # Save best within this bracket
                if current_milestone not in milestone_best_reward or eval_reward > milestone_best_reward[current_milestone]:
                    milestone_best_reward[current_milestone] = eval_reward
                    ms_label = f"{current_milestone // 1_000_000}M" if isinstance(current_milestone, int) and current_milestone >= 1_000_000 else str(current_milestone)
                    if isinstance(current_milestone, int):
                        if current_milestone % 1_000_000 == 0:
                            ms_label = f"{current_milestone // 1_000_000}M"
                        else:
                            ms_label = f"{current_milestone / 1_000_000:.1f}M"
                    else:
                        ms_label = "over_3M"
                    ckpt_path = os.path.join(save_dir, f"ppo_walker_within_{ms_label}.pt")
                    self._make_checkpoint(ckpt_path)
                    print(f"  [Milestone] Best within {ms_label}: {eval_reward:.2f} (step {self.total_step})")
        
        # Save final model
        self._make_checkpoint(os.path.join(save_dir, "ppo_walker_final.pt"))
        
        # ── Print milestone summary ──
        print(f"\n{'='*70}")
        print(f"  MILESTONE CHECKPOINT SUMMARY")
        print(f"{'='*70}")
        for ms in milestones:
            if ms % 1_000_000 == 0:
                label = f"{ms // 1_000_000}M"
            else:
                label = f"{ms / 1_000_000:.1f}M"
            reward = milestone_best_reward.get(ms, None)
            if reward is not None:
                print(f"  Within {label:>5s} steps: best eval = {reward:.2f}  -> ppo_walker_within_{label}.pt")
            else:
                print(f"  Within {label:>5s} steps: (not reached)")
        over_reward = milestone_best_reward.get("over_3M", None)
        if over_reward is not None:
            print(f"  Over   3M   steps: best eval = {over_reward:.2f}  -> ppo_walker_within_over_3M.pt")
        print(f"  Global best:       {best_eval_reward:.2f}  -> ppo_walker_best.pt")
        print(f"{'='*70}\n")
        
        # termination
        self.env.close()

    def evaluate_walker(self, n_episodes=10):
        """Evaluate the agent on a single Walker2d env with deterministic policy."""
        self.is_test = True
        eval_env = gym.make("Walker2d-v5")
        rewards = []
        for seed in range(n_episodes):
            state, _ = eval_env.reset(seed=seed)
            state = self.normalize_obs(state.reshape(1, -1), update=False).flatten()
            done = False
            episode_reward = 0
            while not done:
                action = self.select_action(state.reshape(1, -1))
                action = action.flatten()
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = self.normalize_obs(next_state.reshape(1, -1), update=False).flatten()
            rewards.append(episode_reward)
        eval_env.close()
        self.is_test = False
        return np.mean(rewards)

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

def make_env(env_id, seed, idx):
    """Create a thunk for vectorized env creation."""
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-run")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=0.0)  # MuJoCo doesn't need entropy push
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--update-epoch", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--anneal-lr", type=bool, default=True)
    parser.add_argument("--save-dir", type=str, default="walker_checkpoints",
                        help="Directory to save milestone checkpoints")
    args = parser.parse_args()
 
    # environment — use SyncVectorEnv with 8 parallel envs
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    envs = gym.vector.SyncVectorEnv(
        [make_env("Walker2d-v5", seed, i) for i in range(args.n_envs)]
    )
    
    wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True,
               config=vars(args))
    
    agent = PPOAgent(envs, args)
    agent.train()
