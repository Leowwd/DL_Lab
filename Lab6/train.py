"""
train.py — Trainer for Conditional DDPM

Features:
  - Exponential Moving Average (EMA) of model weights
  - Classifier-Free Guidance training (random condition dropout)
  - Gradient clipping
  - Periodic evaluation and checkpointing
"""

import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ICLEVRDataset, get_test_conditions
from model import ConditionalUNet
from ddpm import DDPMScheduler


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update shadow parameters"""
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def get_model(self):
        """Return the EMA shadow model."""
        return self.shadow


class Trainer:
    """Training loop for the Conditional DDPM.

    Attributes:
        model: ConditionalUNet
        ema: EMA wrapper
        scheduler: DDPMScheduler
        optimizer: AdamW
        dataloader: training data
        device: cuda/cpu
    """

    def __init__(self, config):
        """
        Args:
            config: argparse namespace or dict-like with training hyperparameters
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Dataset & DataLoader
        self.dataset = ICLEVRDataset(
            img_dir=config.img_dir,
            train_json_path=config.train_json,
            objects_json_path=config.objects_json,
            img_size=config.img_size,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Model
        self.model = ConditionalUNet(
            in_channels=3,
            out_channels=3,
            base_channels=config.base_channels,
            channel_mults=config.channel_mults,
            num_classes=config.num_classes,
            time_dim=config.time_dim,
            dropout=config.dropout,
        ).to(self.device)

        # EMA
        self.ema = EMA(self.model, decay=config.ema_decay)

        # Scheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule=config.schedule,
            device=self.device,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs,
        )

        # Loss
        self.criterion = nn.MSELoss()

        # Tracking
        self.best_acc = 0.0
        self.start_epoch = 0

        # Load checkpoint if resuming
        if config.resume and os.path.exists(config.resume):
            self._load_checkpoint(config.resume)

    def _load_checkpoint(self, path):
        """Resume training from a checkpoint."""
        print(f"[Trainer] Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.ema.shadow.load_state_dict(ckpt['ema_model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_acc = ckpt.get('best_acc', 0.0)
        print(f"[Trainer] Resumed at epoch {self.start_epoch}, best_acc={self.best_acc:.4f}")

    def _save_checkpoint(self, epoch, acc, filename='best_model.pth'):
        """Save model checkpoint."""
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'ema_model': self.ema.shadow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_acc': acc,
        }
        torch.save(ckpt, os.path.join(self.config.save_dir, filename))
        print(f"[Trainer] Saved checkpoint: {filename} (acc={acc:.4f})")

    def train(self):
        """Main training loop."""
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Load test conditions for periodic evaluation
        test_conditions = get_test_conditions(
            self.config.test_json, self.config.objects_json,
        ).to(self.device)

        print(f"[Trainer] Training on {len(self.dataset)} images, "
              f"batch_size={self.config.batch_size}, epochs={self.config.epochs}")
        print(f"[Trainer] Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")

            for images, conditions in pbar:
                images = images.to(self.device)
                conditions = conditions.to(self.device)
                B = images.shape[0]

                # Sample random timesteps
                t = self.scheduler.sample_timesteps(B)

                # Sample noise and create noisy images
                noise = torch.randn_like(images)
                x_t, _ = self.scheduler.add_noise(images, t, noise)

                # Classifier-free guidance: randomly drop condition with probability p_uncond
                if self.config.p_uncond > 0:
                    mask = torch.rand(B, device=self.device) < self.config.p_uncond
                    conditions[mask] = 0.0  # zero out condition for unconditional training

                # Predict noise
                noise_pred = self.model(x_t, t, conditions)

                # Compute loss
                loss = self.criterion(noise_pred, noise)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                # Update EMA
                self.ema.update(self.model)

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            self.lr_scheduler.step()
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"[Epoch {epoch+1}] avg_loss={avg_loss:.4f}, lr={self.lr_scheduler.get_last_lr()[0]:.6f}")

            # Periodic evaluation
            if (epoch + 1) % self.config.eval_every == 0 or (epoch + 1) == self.config.epochs:
                acc = self._evaluate(test_conditions)
                print(f"[Epoch {epoch+1}] test.json accuracy: {acc:.4f}")

                if acc > self.best_acc:
                    self.best_acc = acc
                    self._save_checkpoint(epoch, acc, 'best_model.pth')

            # Save latest checkpoint every N epochs
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch, self.best_acc, f'checkpoint_epoch{epoch+1}.pth')

        # Save final model
        self._save_checkpoint(self.config.epochs - 1, self.best_acc, 'final_model.pth')
        print(f"[Trainer] Training complete. Best accuracy: {self.best_acc:.4f}")

    @torch.no_grad()
    def _evaluate(self, conditions):
        """Generate images from test conditions and compute accuracy.

        Args:
            conditions: (N, 24) one-hot label tensor

        Returns:
            accuracy: float
        """
        # Cache evaluator to avoid reloading 44MB checkpoint every call
        if not hasattr(self, '_evaluator'):
            from evaluator import evaluation_model
            self._evaluator = evaluation_model()

        ema_model = self.ema.get_model()
        ema_model.eval()

        B = conditions.shape[0]
        shape = (B, 3, self.config.img_size, self.config.img_size)

        # Generate images using EMA model
        images = self.scheduler.sample(
            ema_model, conditions, shape, self.device,
            cfg_scale=self.config.cfg_scale,
        )

        # Clamp to [-1, 1] (already normalized)
        images = images.clamp(-1, 1)

        acc = self._evaluator.eval(images, conditions)
        return acc

