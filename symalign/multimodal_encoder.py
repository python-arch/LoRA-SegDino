from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SmallMaskEncoder


class SmallImageEncoder(nn.Module):
    """
    Lightweight CNN that maps a masked RGB image region to an embedding.
    Input: (B,3,H,W) float in [0,1] (or normalized; consistency matters more than scale).
    Output: (B,D) normalized embedding.
    """

    def __init__(self, embed_dim: int = 64, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = self.head(z)
        return F.normalize(z, dim=-1)


class FusionMLP(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, z_mask: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_mask, z_img], dim=-1)
        return F.normalize(self.proj(z), dim=-1)


class FusionAttention(nn.Module):
    """
    Computes a soft weight over (mask,image) embeddings then projects to embed_dim.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2),
        )
        self.proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, z_mask: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([z_mask, z_img], dim=-1)
        w = torch.softmax(self.attn(concat), dim=-1)  # (B,2)
        zm = w[:, 0:1] * z_mask
        zi = w[:, 1:2] * z_img
        fused = torch.cat([zm, zi], dim=-1)
        return F.normalize(self.proj(fused), dim=-1)


@dataclass(frozen=True)
class MultiModalConfig:
    embed_dim: int = 64
    mask_width: int = 32
    image_width: int = 32
    fusion: str = "mlp"  # mlp|attn


class MultiModalSymbolicEncoder(nn.Module):
    """
    Produces (z_fused, z_mask, z_img).
    The mask encoder is typically loaded from a pretrained `E_theta` checkpoint and frozen.
    """

    def __init__(self, mask_encoder: SmallMaskEncoder, cfg: MultiModalConfig) -> None:
        super().__init__()
        self.mask_encoder = mask_encoder
        self.image_encoder = SmallImageEncoder(embed_dim=cfg.embed_dim, width=cfg.image_width)
        if cfg.fusion == "attn":
            self.fusion = FusionAttention(embed_dim=cfg.embed_dim)
        else:
            self.fusion = FusionMLP(embed_dim=cfg.embed_dim)
        self.cfg = cfg

    def forward(self, image: torch.Tensor, mask_pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        image: (B,3,H,W)
        mask_pair: (B,2,H,W) where channels are [mask01, boundary_band01] or [prob, boundary]
        """
        z_mask = self.mask_encoder(mask_pair)
        mask_ch = mask_pair[:, 0:1].clamp(0.0, 1.0)
        masked_image = image * mask_ch
        z_img = self.image_encoder(masked_image)
        z_fused = self.fusion(z_mask, z_img)
        return z_fused, z_mask, z_img
