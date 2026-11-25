"""Fuse multi-layer transformer features into a 3-D spatial feature map."""

from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from ncntm.models.common.layers import PointwiseLinear, group_norm


class FeatureProjector3D(nn.Module):
    """Fuse multi-layer DiT features into a 3-D spatial feature map."""

    def __init__(
        self,
        hidden_dim: int,
        num_feature_layers: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        total_in = hidden_dim * num_feature_layers
        self.proj = nn.Sequential(
            PointwiseLinear(total_in, out_dim * 2),
            group_norm(out_dim * 2),
            nn.GELU(),
            PointwiseLinear(out_dim * 2, out_dim),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        d: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        spatial = [
            rearrange(f, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
            for f in features
        ]
        fused = torch.cat(spatial, dim=1)
        return self.proj(fused)
