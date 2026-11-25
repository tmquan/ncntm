"""Small primitive layers shared by the backbone wrappers."""

import torch
import torch.nn as nn
from einops import rearrange


def group_norm(ch: int) -> nn.GroupNorm:
    """GroupNorm with the largest power-of-two number of groups that divides ``ch``."""
    num_groups = max(g for g in (1, 2, 4, 8, 16, 32) if ch % g == 0)
    return nn.GroupNorm(num_groups, ch)


class PointwiseLinear(nn.Module):
    """Drop-in replacement for Conv{2,3}d(k=1) using nn.Linear.

    Avoids non-contiguous gradient strides that cause DDP warnings.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = rearrange(x, "b c ... -> b ... c").to(self.linear.weight.dtype)
        return rearrange(self.linear(x_in), "b ... c -> b c ...")
