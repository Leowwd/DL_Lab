"""
ddpm.py — DDPM Noise Scheduler

Implements:
  - Linear beta schedule
  - Forward diffusion: q(x_t | x_0)
  - Reverse denoising step: p(x_{t-1} | x_t)
  - Full sampling loop with Classifier-Free Guidance (CFG)
"""

import math
import torch


class DDPMScheduler:
    """Denoising Diffusion Probabilistic Model scheduler.

    Manages the noise schedule and provides methods for adding noise
    (forward process) and denoising (reverse process) with optional
    classifier-free guidance.
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule='linear', device='cuda'):
        """
        Args:
            num_timesteps: total diffusion steps T
            beta_start: \beta_1
            beta_end: \beta_T
            schedule: 'linear', 'quad', or 'cos'
            device: torch device
        """
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.device = device

        if schedule == 'linear':
            self.beta = self._linear_schedule(beta_start, beta_end, num_timesteps, device)
        elif schedule == 'quad':
            self.beta = self._quad_schedule(beta_start, beta_end, num_timesteps, device)
        elif schedule == 'cos':
            self.beta = self._cosine_schedule(num_timesteps, device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Pre-compute useful quantities
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_recip_alpha = 1.0 / self.sqrt_alpha

        alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bar[:-1]])
        self.posterior_variance = self.beta * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)

    def _linear_schedule(self, beta_start, beta_end, num_timesteps, device):
        return torch.linspace(beta_start, beta_end, num_timesteps, device=device)

    def _quad_schedule(self, beta_start, beta_end, num_timesteps, device):
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, device=device) ** 2

    def _cosine_schedule(self, num_timesteps, device, s=0.008):
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, device=device)
        f_t = torch.cos(((x / num_timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar = f_t / f_t[0]
        beta = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clip(beta, 0.0, 0.999)

    def add_noise(self, x0, t, noise=None):
        """Forward diffusion

        Args:
            x0: (B, C, H, W) clean images
            t: (B,) timestep indices
            noise: optional pre-sampled noise

        Returns:
            x_t: noisy images
            noise: the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]

        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    def sample_timesteps(self, batch_size):
        """Sample random timesteps uniformly from [0, T).

        Returns:
            t: (batch_size,) integer timesteps on self.device
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

    @torch.no_grad()
    def denoise_step(self, model, x_t, t, condition, cfg_scale=3.0):
        """Single reverse step with classifier-free guidance.

        Computes: x_{t-1} from x_t

        Args:
            model: noise prediction network
            x_t: (B, C, H, W) current noisy image
            t: (B,) current timestep (same value for all in batch)
            condition: (B, num_classes) condition vector
            cfg_scale: classifier-free guidance scale (0 = unconditional)

        Returns:
            x_{t-1}: denoised one step
        """
        B = x_t.shape[0]

        # Predict noise with condition
        noise_pred_cond = model(x_t, t, condition)

        if cfg_scale > 0:
            # Predict noise without condition (unconditional)
            uncond = torch.zeros_like(condition)
            noise_pred_uncond = model(x_t, t, uncond)

            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond

        # Compute x_{t-1} using DDPM formula
        t_idx = t[0].item()  # same t for all samples
        alpha_t = self.alpha[t_idx]
        alpha_bar_t = self.alpha_bar[t_idx]
        beta_t = self.beta[t_idx]

        coeff = beta_t / self.sqrt_one_minus_alpha_bar[t_idx]
        mean = self.sqrt_recip_alpha[t_idx] * (x_t - coeff * noise_pred)

        if t_idx > 0:
            variance = self.posterior_variance[t_idx]
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(variance) * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def sample(self, model, condition, shape, device, cfg_scale=3.0,
               return_intermediates=False, intermediate_steps=None):
        """Full reverse sampling loop: x_T → x_0

        Args:
            model: noise prediction network (in eval mode)
            condition: (B, num_classes) condition vectors
            shape: (B, C, H, W) output shape
            device: torch device
            cfg_scale: CFG guidance scale
            return_intermediates: if True, return intermediate x_t snapshots
            intermediate_steps: list of timesteps to capture (for denoising viz)

        Returns:
            x_0: (B, C, H, W) generated images in [-1, 1]
            intermediates: list of tensors (only if return_intermediates=True)
        """
        model.eval()
        B = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)
        condition = condition.to(device)

        intermediates = []
        if intermediate_steps is None:
            # Default: ~10 evenly spaced steps including T-1 (noise) and 0 (clean)
            intermediate_steps = set(
                [self.num_timesteps - 1]
                + list(range(0, self.num_timesteps, self.num_timesteps // 10))
            )
        else:
            intermediate_steps = set(intermediate_steps)

        # Capture the initial pure noise as the first frame
        if return_intermediates:
            intermediates.append(x.clone().cpu())

        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            x = self.denoise_step(model, x, t, condition, cfg_scale)

            if return_intermediates and t_idx in intermediate_steps:
                intermediates.append(x.clone().cpu())

        if return_intermediates:
            return x, intermediates
        return x
