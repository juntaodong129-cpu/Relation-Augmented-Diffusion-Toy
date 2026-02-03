from __future__ import annotations
import math
import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding.
    t: (B,) int64 or float
    return: (B,dim)
    """
    half = dim // 2
    device = t.device
    t = t.float()
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device).float() / half
    )  # (half,)
    args = t[:, None] * freqs[None, :]  # (B,half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B,2*half)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=device)], dim=1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        # add time embedding
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class TinyUNet(nn.Module):
    """
    Very small conditional UNet-ish model for 64x64 masks.
    Predicts epsilon (noise).
    """
    def __init__(self, in_ch: int, cond_ch: int, base_ch: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(in_ch + cond_ch, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, time_dim)
        self.downsample = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)  # 64->32

        self.mid = ResBlock(base_ch, time_dim)

        self.upsample = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)  # 32->64
        self.up1 = ResBlock(base_ch, time_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B,1,H,W)
        cond: (B,cond_ch,H,W)
        t: (B,) int64
        """
        t_emb = timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        h = torch.cat([x_t, cond], dim=1)
        h = self.in_conv(h)

        h = self.down1(h, t_emb)
        h = self.downsample(h)

        h = self.mid(h, t_emb)

        h = self.upsample(h)
        h = self.up1(h, t_emb)

        out = self.out_conv(self.act(self.out_norm(h)))
        return out
