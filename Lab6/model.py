"""
model.py — Conditional UNet for DDPM

Architecture:
  - Sinusoidal time embeddings
  - Multi-label condition embedding (24-dim one-hot → cond_dim)
  - UNet encoder-decoder with skip connections
  - Self-attention at lower resolutions
  - ResBlocks with time + condition injection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep t."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (B,) integer timesteps
        Returns:
            (B, dim) embedding
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ConditionEmbedding(nn.Module):
    """Embed multi-label condition (24-dim one-hot) to a dense vector."""

    def __init__(self, num_classes, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, condition):
        """
        Args:
            condition: (B, num_classes) multi-hot vector
        Returns:
            (B, cond_dim) embedding
        """
        return self.net(condition)


class ResBlock(nn.Module):
    """Residual block with time + condition injection via adaptive scaling."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        # Project combined time+condition embedding to out_ch for scale+shift
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch * 2),
        )
        # Residual shortcut
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        """
        Args:
            x: (B, C, H, W)
            emb: (B, emb_dim) combined time + condition embedding
        """
        h = self.conv1(x)

        # Adaptive scale + shift from embedding
        scale_shift = self.emb_proj(emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention with group norm."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class DownBlock(nn.Module):
    """Encoder block: ResBlock + optional Attention + 2× downsampling."""

    def __init__(self, in_ch, out_ch, emb_dim, has_attn=False, dropout=0.1):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, emb_dim, dropout)
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, emb):
        h = self.res(x, emb)
        h = self.attn(h)
        skip = h
        h = self.downsample(h)
        return h, skip


class UpBlock(nn.Module):
    """Decoder block: 2× upsample + concat skip + ResBlock + optional Attention."""

    def __init__(self, in_ch, out_ch, emb_dim, has_attn=False, dropout=0.1):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch + out_ch, out_ch, emb_dim, dropout)  # concat skip
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()

    def forward(self, x, skip, emb):
        h = self.upsample(x)
        # Handle size mismatches due to odd dimensions
        if h.shape != skip.shape:
            h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, skip], dim=1)
        h = self.res(h, emb)
        h = self.attn(h)
        return h

class ConditionalUNet(nn.Module):
    """UNet conditioned on timestep t and multi-label class condition.

    Channel progression: 64 → 128 → 256 → 512
    Attention at 16×16 and 8×8 resolutions.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_classes=24,
        time_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.time_dim = time_dim

        # Embeddings
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_embed = ConditionEmbedding(num_classes, time_dim)

        # Combined embedding dimension
        emb_dim = time_dim

        # Initial convolution
        ch = base_channels * channel_mults[0]  # 64
        self.init_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Encoder (down)
        channels = [ch]
        self.downs = nn.ModuleList()
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            has_attn = (i >= 2)  # attention at 16×16 and 8×8
            self.downs.append(DownBlock(ch, out_ch, emb_dim, has_attn, dropout))
            ch = out_ch
            channels.append(ch)

        # Bottleneck
        self.mid_res1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_res2 = ResBlock(ch, ch, emb_dim, dropout)

        # Decoder (up)
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            has_attn = (i >= 2)
            self.ups.append(UpBlock(ch, out_ch, emb_dim, has_attn, dropout))
            ch = out_ch

        # Final output
        self.final = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, t, condition):
        """
        Args:
            x: (B, 3, H, W) noisy image
            t: (B,) integer timesteps
            condition: (B, 24) multi-hot label vector
        Returns:
            (B, 3, H, W) predicted noise ε
        """
        # Compute embeddings
        t_emb = self.time_embed(t)         # (B, time_dim)
        c_emb = self.cond_embed(condition)  # (B, time_dim)
        emb = t_emb + c_emb                # additive fusion

        # Initial conv
        h = self.init_conv(x)

        # Encoder
        skips = []
        for down in self.downs:
            h, skip = down(h, emb)
            skips.append(skip)

        # Bottleneck
        h = self.mid_res1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, emb)

        # Decoder
        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, emb)

        return self.final(h)
