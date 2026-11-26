"""VoxelDiT2-3D: a representation-grounded pixel diffusion transformer on voxels.

Volumetric version of ``PixelDiT2`` (``pixdit_core/pixeldit2_c2i.py``,
NVlabs/PixelDiT). There is no VAE: the volume is cut into ``pz x py x px``
voxel patches, each patch is one token, and the final adaLN layer maps every
token back to its voxels. The network is an x-predictor: given ``x_t`` and
``t`` it returns the clean volume (the caller forms the velocity).

Per token the adaLN condition is ``SiLU(grounding + t_emb + y_emb)`` (see
:mod:`ncntm.models.voxeldit2.grounding`). From block
``in_context_start`` on, ``in_context_len`` copies of the class embedding are
prepended to the sequence; they carry an identity rotation.
"""

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ncntm.models.voxeldit2.grounding import build_grounding
from ncntm.models.voxeldit2.layers import (
    ClassEmbedder,
    ModulatedDiTBlock,
    PatchFinalLayer3D,
    PatchTokenEmbedder,
    Size3,
    TimestepConditioner,
    as_size3,
    patchify_3d,
    precompute_freqs_cis_3d,
    unpatchify_3d,
)


class VoxelDiT2_3D(nn.Module):
    """Pixel-space DiT over ``[B, C, D, H, W]`` volumes.

    Args:
        in_channels: Volume channels (1 for EM). The output has as many.
        hidden_size / num_heads / depth: Transformer geometry.
        patch_size: Voxel patch, an int or ``(pz, py, px)``.
        num_classes: Learned class embeddings; index ``num_classes`` is the
            null class. EM uses one class.
        in_context_len / in_context_start: In-context class tokens.
        attn_dropout / proj_dropout: Applied to the middle half of the blocks.
        mlp_hidden_multiple: Round the SwiGLU width up to this multiple.
        grounding_encoder: DINOv3 key, or ``None`` for no grounding.
        grounding_proj_blocks / grounding_proj_heads: ``dino_proj`` geometry.
        grounding_axes / grounding_slice_stride / grounding_chunk /
        grounding_pretrained: See
            :class:`~ncntm.models.voxeldit2.grounding.GroundingConditioner3D`.
        single_silu_cond: Skip the SiLU on the summed condition.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_size: int = 1280,
        num_heads: int = 16,
        depth: int = 32,
        patch_size: Union[int, Sequence[int]] = 16,
        num_classes: int = 1,
        in_context_len: int = 32,
        in_context_start: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_hidden_multiple: int = 0,
        grounding_encoder: Optional[str] = "dinov3_vits16",
        grounding_proj_blocks: int = 1,
        grounding_proj_heads: Optional[int] = None,
        grounding_axes: Sequence[str] = ("z", "y", "x"),
        grounding_slice_stride: int = 4,
        grounding_chunk: int = 64,
        grounding_pretrained: bool = True,
        single_silu_cond: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.depth = int(depth)
        self.patch_size: Size3 = as_size3(patch_size)
        self.num_classes = int(num_classes)
        self.in_context_len = int(in_context_len)
        self.in_context_start = int(in_context_start)
        self.single_silu_cond = bool(single_silu_cond)
        self.gradient_checkpointing = False

        if not 0 <= self.in_context_start < self.depth:
            raise ValueError(
                f"in_context_start must be in [0, {self.depth - 1}], got {self.in_context_start}"
            )

        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.y_embedder = ClassEmbedder(self.num_classes + 1, self.hidden_size)
        self.s_embedder = PatchTokenEmbedder(
            self.in_channels * math.prod(self.patch_size), self.hidden_size, bias=True,
        )
        self.in_context_posemb = (
            nn.Parameter(torch.zeros(1, self.in_context_len, self.hidden_size))
            if self.in_context_len > 0 else None
        )
        quarter = self.depth // 4
        self.patch_blocks = nn.ModuleList([
            ModulatedDiTBlock(
                self.hidden_size,
                self.num_heads,
                attn_drop=attn_dropout if quarter <= i < quarter * 3 else 0.0,
                proj_drop=proj_dropout if quarter <= i < quarter * 3 else 0.0,
                mlp_hidden_multiple=mlp_hidden_multiple,
            )
            for i in range(self.depth)
        ])
        self.final_layer = PatchFinalLayer3D(self.hidden_size, self.patch_size, self.out_channels)

        self.initialize_weights()

        self.dino_conditioner = build_grounding(
            grounding_encoder,
            hidden_size=self.hidden_size,
            patch=self.patch_size,
            proj_num_blocks=grounding_proj_blocks,
            proj_num_heads=self.num_heads if grounding_proj_heads is None else grounding_proj_heads,
            axes=grounding_axes,
            slice_stride=grounding_slice_stride,
            chunk=grounding_chunk,
            pretrained_encoder=grounding_pretrained,
        )
        self._pos_cache: Dict[Tuple[Size3, int], torch.Tensor] = {}

    def initialize_weights(self) -> None:
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.in_context_posemb is not None:
            nn.init.normal_(self.in_context_posemb, std=0.02)
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def token_grid(self, spatial: Sequence[int]) -> Size3:
        """``(D, H, W)`` -> ``(gz, gy, gx)``; raises when a side is not a patch multiple."""
        if any(int(s) % p for s, p in zip(spatial, self.patch_size)):
            raise ValueError(
                f"Input size {tuple(spatial)} must be divisible by patch_size={self.patch_size}."
            )
        return tuple(int(s) // p for s, p in zip(spatial, self.patch_size))  # type: ignore[return-value]

    def fetch_pos(self, grid: Size3, device, num_context_tokens: int = 0) -> torch.Tensor:
        key = (grid, num_context_tokens)
        if key not in self._pos_cache:
            pos = precompute_freqs_cis_3d(self.head_dim, grid)
            if num_context_tokens > 0:
                identity = torch.ones(num_context_tokens, pos.shape[-1], dtype=pos.dtype)
                pos = torch.cat([identity, pos], dim=0)
            self._pos_cache[key] = pos
        return self._pos_cache[key].to(device)

    def _class_embedding(self, y: Optional[torch.Tensor], batch: int, device) -> torch.Tensor:
        if y is None:
            y = torch.zeros(batch, dtype=torch.long, device=device)
        return self.y_embedder(y.to(device=device, dtype=torch.long)).view(batch, 1, self.hidden_size)

    def _condition(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_emb: torch.Tensor,
        y_emb: torch.Tensor,
        num_patches: int,
        skip_grounding: bool,
        skip_grounding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        null = t_emb.unsqueeze(1) + y_emb
        if not self.single_silu_cond:
            null = F.silu(null)
        null = null.expand(-1, num_patches, -1)
        if skip_grounding or self.dino_conditioner is None:
            return null
        c = self.dino_conditioner(x, t) + t_emb.unsqueeze(1) + y_emb
        if not self.single_silu_cond:
            c = F.silu(c)
        if skip_grounding_mask is not None:
            c = torch.where(skip_grounding_mask.view(-1, 1, 1), null, c)
        return c

    def _run_block(self, block: nn.Module, s, c, pos):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(block, s, c, pos, use_reentrant=False)
        return block(s, c, pos)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        skip_grounding: bool = False,
        skip_grounding_mask: Optional[torch.Tensor] = None,
        feature_layers: Optional[Sequence[int]] = None,
        cond_extra: Optional[torch.Tensor] = None,
    ):
        """Clean-volume prediction ``[B, C, D, H, W]``.

        ``cond_extra`` ``[B, hidden]`` is added to the class condition (the
        projected prompt text); ``None`` leaves the condition unchanged.

        With ``feature_layers`` the return is ``(x_pred, {layer: tokens})``
        where ``tokens`` is the ``[B, N, hidden]`` output of that block with
        the in-context tokens removed.
        """
        batch = x.shape[0]
        grid = self.token_grid(x.shape[-3:])
        num_patches = math.prod(grid)
        t = t.reshape(-1).to(device=x.device)
        if t.numel() == 1 and batch > 1:
            t = t.expand(batch)

        t_emb = self.t_embedder(t)
        y_emb = self._class_embedding(y, batch, x.device)
        if cond_extra is not None:
            y_emb = y_emb + cond_extra.to(y_emb.dtype).view(batch, 1, self.hidden_size)
        c = self._condition(x, t, t_emb, y_emb, num_patches, skip_grounding, skip_grounding_mask)

        s = self.s_embedder(patchify_3d(x, self.patch_size).to(self.s_embedder.proj.weight.dtype))
        c = c.to(s.dtype)

        use_context = self.in_context_len > 0
        pos = self.fetch_pos(grid, x.device)
        if use_context:
            pos_ctx = self.fetch_pos(grid, x.device, self.in_context_len)
            c_ctx = torch.cat(
                [c.mean(dim=1, keepdim=True).expand(-1, self.in_context_len, -1), c], dim=1,
            )

        wanted = set(int(i) for i in feature_layers) if feature_layers is not None else set()
        features: Dict[int, torch.Tensor] = {}
        inserted = False
        for i, block in enumerate(self.patch_blocks):
            if use_context and not inserted and i == self.in_context_start:
                ctx = y_emb.to(s.dtype).expand(-1, self.in_context_len, -1)
                s = torch.cat([ctx + self.in_context_posemb.to(s.dtype), s], dim=1)
                inserted = True
            s = self._run_block(block, s, c_ctx if inserted else c, pos_ctx if inserted else pos)
            if i in wanted:
                features[i] = s[:, self.in_context_len:] if inserted else s
        if inserted:
            s = s[:, self.in_context_len:]

        out = self.final_layer(s, c)
        x_pred = unpatchify_3d(out, self.patch_size, grid, self.out_channels)
        if feature_layers is None:
            return x_pred
        return x_pred, features


__all__ = ["VoxelDiT2_3D"]
