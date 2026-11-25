"""Representation grounding for VoxelDiT2-3D.

PixelDiT2 grounds every patch token in the features of a frozen DINOv3 ViT:
the encoder sees the current (noisy) input, a small timestep-modulated
transformer (``dino_proj``) maps its patch tokens to the DiT width, and the
result joins the timestep and class embeddings in the per-token adaLN
condition.

DINOv3 is a 2-D image encoder. The volumetric version runs it on sections of
the volume along each of ``axes`` (``z``, ``y``, ``x``): within every patch
of that axis, ``ceil(p / slice_stride)`` evenly spaced sections are encoded
and their tokens averaged, which lands them on the 3-D patch grid. The
per-axis grids are then averaged (tri-planar grounding), so the condition
does not depend on which axis the crop was cut along. ``dino_proj`` keeps
the upstream parameter names and runs with the 3-D rotary embedding.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncntm.models.voxeldit2.layers import (
    ModulatedDiTBlock,
    Size3,
    TimestepConditioner,
    precompute_freqs_cis_3d,
)


@dataclass(frozen=True)
class EncoderSpec:
    timm_name: str
    feature_dim: int
    patch_size: int


GROUNDING_ENCODERS: Dict[str, EncoderSpec] = {
    "dinov3_vits16": EncoderSpec("vit_small_patch16_dinov3_qkvb.lvd1689m", 384, 16),
    "dinov3_vitb16": EncoderSpec("vit_base_patch16_dinov3_qkvb.lvd1689m", 768, 16),
    "dinov3_vitl16": EncoderSpec("vit_large_patch16_dinov3_qkvb.lvd1689m", 1024, 16),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Tensor dim of each axis in ``[B, C, D, H, W]``.
_AXIS_DIM = {"z": 2, "y": 3, "x": 4}


def build_encoder(name: str, pretrained: bool = True) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "VoxelDiT2-3D grounding needs timm (pip install timm), or set "
            "model.grounding_encoder: null to train without grounding."
        ) from exc
    if name not in GROUNDING_ENCODERS:
        raise ValueError(
            f"Unknown grounding encoder {name!r}. Available: {sorted(GROUNDING_ENCODERS)}"
        )
    return timm.create_model(
        GROUNDING_ENCODERS[name].timm_name,
        pretrained=pretrained,
        dynamic_img_size=True,
    )


class TimestepProjection(nn.Module):
    """Encoder tokens -> DiT width through timestep-modulated DiT blocks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_size: int,
        num_blocks: int = 1,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"TimestepProjection hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}"
            )
        self.head_dim = hidden_size // num_heads
        self.input_proj = nn.Linear(in_dim, hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(hidden_size)
        self.blocks = nn.ModuleList(
            [ModulatedDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_size, out_dim, bias=True)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        self._pos_cache: Dict[Size3, torch.Tensor] = {}

    def fetch_pos(self, grid: Size3, device) -> torch.Tensor:
        if grid not in self._pos_cache:
            self._pos_cache[grid] = precompute_freqs_cis_3d(self.head_dim, grid)
        return self._pos_cache[grid].to(device)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor, grid: Size3) -> torch.Tensor:
        x = self.input_proj(tokens)
        c = self.t_embedder(t).unsqueeze(1).to(dtype=x.dtype)
        pos = self.fetch_pos(grid, x.device)
        for block in self.blocks:
            x = block(x, c, pos)
        return self.output_proj(x)


class GroundingConditioner3D(nn.Module):
    """Tri-planar frozen-DINOv3 grounding on a 3-D patch grid.

    Args:
        encoder_name: Key of :data:`GROUNDING_ENCODERS`.
        hidden_size: DiT width.
        patch: DiT voxel patch ``(pz, py, px)``.
        proj_num_blocks / proj_num_heads: ``dino_proj`` depth and heads.
        axes: Section axes, any non-empty subset of ``("z", "y", "x")``.
        slice_stride: Section spacing (voxels) inside a patch along the
            section axis; ``1`` encodes every section.
        chunk: Sections per encoder call (bounds activation memory).
        pretrained_encoder: Load the DINOv3 weights through timm.
    """

    def __init__(
        self,
        encoder_name: str,
        hidden_size: int,
        patch: Size3,
        proj_num_blocks: int = 1,
        proj_num_heads: int = 16,
        axes: Sequence[str] = ("z", "y", "x"),
        slice_stride: int = 4,
        chunk: int = 64,
        pretrained_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.spec = GROUNDING_ENCODERS[encoder_name]
        self.encoder_name = encoder_name
        self.patch = tuple(int(p) for p in patch)
        self.axes = tuple(str(a).lower() for a in axes)
        unknown = [a for a in self.axes if a not in _AXIS_DIM]
        if not self.axes or unknown:
            raise ValueError(f"grounding axes must be a non-empty subset of z, y, x; got {axes!r}.")
        self.slice_stride = max(1, int(slice_stride))
        self.chunk = max(1, int(chunk))
        self.encoder = build_encoder(encoder_name, pretrained=pretrained_encoder)
        self.encoder.requires_grad_(False)
        self.dino_proj = TimestepProjection(
            in_dim=self.spec.feature_dim,
            out_dim=hidden_size,
            hidden_size=hidden_size,
            num_blocks=proj_num_blocks,
            num_heads=proj_num_heads,
        )
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _section_offsets(self, p: int, device) -> torch.Tensor:
        stride = min(self.slice_stride, p)
        return torch.arange(stride // 2, p, stride, device=device)

    def _encode_sections(self, images: torch.Tensor, grid_uv: Tuple[int, int]) -> torch.Tensor:
        """``[S, 3, U, V]`` in ``[-1, 1]`` -> encoder tokens ``[S, gu * gv, F]``."""
        ps = self.spec.patch_size
        size = (grid_uv[0] * ps, grid_uv[1] * ps)
        enc_param = next(self.encoder.parameters())
        out = []
        for start in range(0, images.shape[0], self.chunk):
            x = images[start:start + self.chunk].to(enc_param.dtype)
            x = (x + 1.0) * 0.5
            x = (x - self.image_mean.to(x.dtype)) / self.image_std.to(x.dtype)
            if tuple(x.shape[-2:]) != size:
                x = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
            features = self.encoder.forward_features(x)
            prefix = int(getattr(self.encoder, "num_prefix_tokens", 0))
            out.append(features[:, prefix:, :])
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Size3]:
        """``[B, C, D, H, W]`` in ``[-1, 1]`` -> tokens ``[B, gz * gy * gx, F]`` and the grid."""
        b, c = x.shape[:2]
        grid = tuple(int(s) // p for s, p in zip(x.shape[-3:], self.patch))
        if c == 1:
            x = x.expand(-1, 3, -1, -1, -1)
        elif c != 3:
            raise ValueError(f"grounding expects 1 or 3 input channels, got {c}.")
        fused = None
        for axis in self.axes:
            dim = _AXIS_DIM[axis]
            a = dim - 2
            n_a, p_a = grid[a], self.patch[a]
            offsets = self._section_offsets(p_a, x.device)
            k = int(offsets.numel())
            idx = (torch.arange(n_a, device=x.device)[:, None] * p_a + offsets[None, :]).reshape(-1)
            sections = x.index_select(dim, idx).movedim(dim, 1)          # [B, S, 3, U, V]
            sections = sections.reshape(b * n_a * k, 3, *sections.shape[-2:])
            grid_uv = tuple(g for i, g in enumerate(grid) if i != a)
            tokens = self._encode_sections(sections, grid_uv)            # [B*S, gu*gv, F]
            tokens = tokens.float().reshape(b, n_a, k, *grid_uv, -1).mean(dim=2)
            tokens = tokens.movedim(1, 1 + a)                            # [B, gz, gy, gx, F]
            fused = tokens if fused is None else fused + tokens
        fused = fused / len(self.axes)
        return fused.reshape(b, -1, fused.shape[-1]), grid

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tokens, grid = self.encode(x)
        return self.dino_proj(tokens.to(dtype=self.dino_proj.input_proj.weight.dtype), t, grid)


def build_grounding(
    encoder_name: Optional[str], **kwargs,
) -> Optional[GroundingConditioner3D]:
    """``None`` / ``"none"`` disables grounding."""
    if encoder_name is None or str(encoder_name).lower() in ("", "none", "null"):
        return None
    return GroundingConditioner3D(str(encoder_name), **kwargs)


__all__ = [
    "GROUNDING_ENCODERS",
    "GroundingConditioner3D",
    "TimestepProjection",
    "build_encoder",
    "build_grounding",
]
