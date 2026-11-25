"""Volumetric building blocks of VoxelDiT2-3D.

3-D counterparts of ``pixdit_core/modules.py`` (NVlabs/PixelDiT). The
transformer pieces -- RMSNorm, SwiGLU feed-forward, QK-normed rotary
attention, adaLN-modulated block, timestep embedder -- keep the upstream
parameter names, so the released 2-D checkpoints load into them unchanged
(see :mod:`ncntm.models.voxeldit2.checkpoint`). What is new is the
geometry: cubic voxel patches and a three-axis rotary embedding.

Token order is ``(z, y, x)`` over the patch grid. Inside a patch the input
features are channel-major ``(c, kz, ky, kx)`` (the 3-D analogue of
``F.unfold``) and the output features channel-minor ``(kz, ky, kx, c)``
(the analogue of the upstream ``PatchFinalLayer`` view).
"""

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

Size3 = Tuple[int, int, int]


def as_size3(value) -> Size3:
    """``16`` -> ``(16, 16, 16)``; a length-3 sequence is returned as a tuple."""
    if isinstance(value, int):
        return (value, value, value)
    out = tuple(int(v) for v in value)
    if len(out) != 3:
        raise ValueError(f"expected an int or three ints (z, y, x), got {value!r}.")
    return out  # type: ignore[return-value]


def patchify_3d(x: torch.Tensor, patch: Size3) -> torch.Tensor:
    """``[B, C, D, H, W]`` -> ``[B, N, C * pz * py * px]``, features ``(c, kz, ky, kx)``."""
    b, c, d, h, w = x.shape
    pz, py, px = patch
    gz, gy, gx = d // pz, h // py, w // px
    x = x.reshape(b, c, gz, pz, gy, py, gx, px)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b, gz * gy * gx, c * pz * py * px)


def unpatchify_3d(
    tokens: torch.Tensor, patch: Size3, grid: Size3, channels: int,
) -> torch.Tensor:
    """``[B, N, pz * py * px * C]`` (features ``(kz, ky, kx, c)``) -> ``[B, C, D, H, W]``."""
    b = tokens.shape[0]
    pz, py, px = patch
    gz, gy, gx = grid
    x = tokens.reshape(b, gz, gy, gx, pz, py, px, channels)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.reshape(b, channels, gz * pz, gy * py, gx * px)


def rope_axes(head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Frequency and axis (0 = z, 1 = y, 2 = x) of every rotary pair of a head.

    The 2-D layout of PixelDiT2 gives pair ``k`` the frequency
    ``f[k // 2]`` on x (even ``k``) or y (odd ``k``). Here every third pair
    (``k % 3 == 2``) is handed to z with the frequency it already had, so x
    and y keep two thirds of their pretrained channels unchanged and each of
    the three axes spans the whole frequency range.
    """
    if head_dim % 4:
        raise ValueError(f"head_dim must be divisible by 4 for the rotary layout, got {head_dim}.")
    base = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 4)[: head_dim // 4].float() / head_dim))
    k = torch.arange(head_dim // 2)
    freq = base[k // 2]
    axis = torch.where(k % 2 == 0, 2, 1)
    axis = torch.where(k % 3 == 2, torch.zeros_like(axis), axis)
    return freq, axis


def precompute_freqs_cis_3d(
    head_dim: int, grid: Sequence[int], scale: float = 16.0,
) -> torch.Tensor:
    """Complex rotations ``[gz * gy * gx, head_dim // 2]`` for a ``(gz, gy, gx)`` token grid.

    Positions run from 0 to ``scale`` along every axis whatever the grid
    size, as in the upstream ``precompute_freqs_cis_2d``.
    """
    gz, gy, gx = (int(g) for g in grid)
    freq, axis = rope_axes(head_dim)
    z = torch.linspace(0, scale, gz)
    y = torch.linspace(0, scale, gy)
    x = torch.linspace(0, scale, gx)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    pos = torch.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], dim=1)
    angles = pos[:, axis] * freq[None, :]
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``[B, N, H, Dh]`` queries and keys by ``freqs_cis`` ``[N, Dh / 2]``."""
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepConditioner(nn.Module):
    """Sinusoidal embedding of a unit time ``t`` in ``[0, 1]`` followed by an MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class ClassEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    """SwiGLU with the upstream ``2/3`` hidden-width rule."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, hidden_multiple: int = 0) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_multiple = int(hidden_multiple)
        if hidden_multiple > 0:
            hidden_dim = ((hidden_dim + hidden_multiple - 1) // hidden_multiple) * hidden_multiple
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        return self.w2(self.ffn_dropout(hidden))


class RotaryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rotary_emb(self.q_norm(q), self.k_norm(k), freqs_cis=pos)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        x = scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class ModulatedDiTBlock(nn.Module):
    """Pre-norm attention + SwiGLU block with per-token adaLN (shift, scale, gate)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_hidden_multiple: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RotaryAttention(
            hidden_size, num_heads=num_heads, qkv_bias=False,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(
            hidden_size, int(hidden_size * mlp_ratio),
            drop=proj_drop, hidden_multiple=mlp_hidden_multiple,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(apply_adaln(self.norm1(x), shift_msa, scale_msa), pos)
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PatchTokenEmbedder(nn.Module):
    """Linear embedding of a flattened voxel patch."""

    def __init__(self, in_features: int, embed_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PatchFinalLayer3D(nn.Module):
    """adaLN-modulated linear map from a token to its ``pz * py * px * C`` voxels."""

    def __init__(self, hidden_size: int, patch: Size3, out_channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch) * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        if shift.dim() == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        return self.linear(apply_adaln(self.norm(x), shift, scale))


__all__ = [
    "ClassEmbedder",
    "FeedForward",
    "ModulatedDiTBlock",
    "PatchFinalLayer3D",
    "PatchTokenEmbedder",
    "RMSNorm",
    "RotaryAttention",
    "Size3",
    "TimestepConditioner",
    "apply_adaln",
    "apply_rotary_emb",
    "as_size3",
    "patchify_3d",
    "precompute_freqs_cis_3d",
    "rope_axes",
    "unpatchify_3d",
]
