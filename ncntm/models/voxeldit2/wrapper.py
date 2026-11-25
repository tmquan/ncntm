"""VoxelDiT2-3D backbone wrapper for volumetric EM.

Drives :class:`~ncntm.models.voxeldit2.network.VoxelDiT2_3D` directly on
voxels -- no VAE, no latent grid -- and exposes the interface the joint
Lightning module uses: ``forward`` returns the unified
``[B, head_channels, D, H, W]`` head, ``rectified_flow_loss`` trains the
pixel-space flow and ``few_step_reconstruct`` integrates it.

Head (pixel space)::

    blocks {feature_layers} --FeatureProjector3D--> [B, F, gz, gy, gx]
        --linear unpatchify--> [B, pixel_channels, D, H, W] --+
    DiT x-prediction (final adaLN layer)  [B, 1, D, H, W] ----+--> TaskHead3D
    input skip stem (full resolution)     [B, skip, D, H, W] -+

Flow convention (shared with :mod:`ncntm.losses.field`): ``t = 0`` is the
clean volume ``x0``, ``t = 1`` the degraded input ``x1``, ``x_t = (1 - t) x0
+ t x1`` and ``v = x1 - x0``. PixelDiT2 is an x-predictor; the velocity is
``(x_t - x_pred) / max(t, t_eps)`` and the loss is the velocity MSE, as in the
upstream ``GroundedFlowTrainer``. The task forward runs the DiT on the
degraded volume at ``feature_timestep`` (``1``: the degraded endpoint).

The backbone is ``self.dit``, so the module's ``model.dit.*`` optimiser
group, the ``freeze_dit_backbone`` schedule and the
``ckpt_path_skip_prefixes`` warm-start filter apply to it unchanged. The
frozen DINOv3 encoder (``dit.dino_conditioner.encoder``) never trains.

References:
    - PixelDiT2: Representation-Grounded Pixel Diffusion Transformers (NeurIPS 2026)
    - https://github.com/NVlabs/PixelDiT
"""

import logging
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from ncntm.losses import HEAD_CHANNELS
from ncntm.losses.field import euler_integrate, interpolate_rectified_flow
from ncntm.models.common import (
    FeatureProjector3D,
    PointwiseLinear,
    TaskHead3D,
    group_norm,
    resolve_freeze_dit_backbone,
)
from ncntm.models.voxeldit2.checkpoint import download_checkpoint, load_pretrained_2d
from ncntm.models.voxeldit2.network import VoxelDiT2_3D
from ncntm.models.voxeldit2.variants import _VARIANT_CONFIGS

logger = logging.getLogger(__name__)

_DEFAULT = object()
_CKPT_TARGETS = {"dit", "head"}


class VoxelDiT2Wrapper3D(nn.Module):
    """VoxelDiT2-3D as a volumetric EM reconstruction / segmentation backbone.

    Args:
        in_channels: Input channels (1 for EM).
        head_channels: Unified head width (affinities + sem + raw).
        feature_size: Width of the fused multi-layer feature map and of the
            task-head refinement blocks.
        variant: Key of :data:`~ncntm.models.voxeldit2.variants._VARIANT_CONFIGS`.
        dtype: DiT weight dtype (``bf16`` / ``fp16`` / ``fp32``).
        pretrained: Inflate the variant's released 2-D checkpoint (or
            ``checkpoint``) into the 3-D DiT. Variants without released
            weights train from scratch.
        checkpoint: A local file or a filename in the variant's HF repo;
            overrides the variant default.
        freeze_dit_backbone: ``bool`` or warm-up epochs ``N``.
        gradient_checkpointing: ``bool`` or a subset of ``{"dit", "head"}``.
        feature_layers: DiT blocks read by the task head (default: four
            evenly spaced blocks).
        cache_dir / hf_token: Checkpoint download location and token.
        dropout: Channel dropout before the task head's last 1x1 conv.
        dit_dropout: PixelDiT2 projection dropout on the middle half of
            the blocks.
        pixel_channels: Channels of the unpatchified feature volume.
        highres_skip / highres_skip_channels: Full-resolution conv stem on
            the input, concatenated before the task head.
        feature_timestep: Flow time of the task forward.
        flow_t_range: Uniform range of the training flow time.
        flow_t_eps: Lower clamp of ``t`` in the velocity denominator.
        grounding_encoder: Override the variant's DINOv3 key; ``None``
            disables grounding.
        grounding_axes / grounding_slice_stride / grounding_chunk /
        grounding_pretrained: See
            :class:`~ncntm.models.voxeldit2.grounding.GroundingConditioner3D`.
        dit_overrides: Extra :class:`VoxelDiT2_3D` kwargs on top of the
            variant (e.g. ``depth``, ``patch_size``).
    """

    _variant_configs = _VARIANT_CONFIGS

    def __init__(
        self,
        in_channels: int = 1,
        head_channels: int = HEAD_CHANNELS,
        feature_size: int = 64,
        variant: str = "H/16",
        dtype: str = "bf16",
        pretrained: bool = True,
        checkpoint: Optional[str] = None,
        freeze_dit_backbone: Union[bool, int] = False,
        gradient_checkpointing: Union[bool, Sequence[str]] = False,
        feature_layers: Optional[List[int]] = None,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        dropout: float = 0.0,
        dit_dropout: float = 0.0,
        pixel_channels: int = 16,
        highres_skip: bool = True,
        highres_skip_channels: int = 8,
        feature_timestep: float = 1.0,
        flow_t_range: Tuple[float, float] = (0.02, 0.98),
        flow_t_eps: float = 0.05,
        grounding_encoder: Any = _DEFAULT,
        grounding_axes: Sequence[str] = ("z", "y", "x"),
        grounding_slice_stride: int = 4,
        grounding_chunk: int = 64,
        grounding_pretrained: bool = True,
        dit_overrides: Optional[dict] = None,
    ) -> None:
        super().__init__()
        key = str(variant).upper()
        if key not in self._variant_configs:
            raise ValueError(f"Unknown variant {variant!r}. Choose from: {list(self._variant_configs)}")
        self.variant = key
        self.cfg = self._variant_configs[key]
        self.in_channels = int(in_channels)
        self.head_channels = int(head_channels)
        self.feature_size = int(feature_size)
        self._dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self._feature_timestep = float(feature_timestep)
        lo, hi = (float(v) for v in flow_t_range)
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"flow_t_range must satisfy 0 <= lo < hi <= 1, got {flow_t_range}.")
        self._flow_t_range = (lo, hi)
        self._flow_t_eps = float(flow_t_eps)

        initial_frozen, thaw_epoch = resolve_freeze_dit_backbone(freeze_dit_backbone)
        self._freeze_dit_backbone = initial_frozen
        self._dit_thaw_epoch: Optional[int] = thaw_epoch

        if gradient_checkpointing is True:
            targets = set(_CKPT_TARGETS)
        elif not gradient_checkpointing:
            targets = set()
        else:
            targets = {str(t).strip().lower() for t in gradient_checkpointing}
            unknown = targets - _CKPT_TARGETS
            if unknown:
                raise ValueError(
                    f"Unknown gradient_checkpointing targets {sorted(unknown)}; "
                    f"choose from {sorted(_CKPT_TARGETS)}."
                )
        self._ckpt_targets = targets

        net_kwargs = self.cfg.network_kwargs()
        net_kwargs.update(
            in_channels=self.in_channels,
            proj_dropout=float(dit_dropout),
            grounding_axes=tuple(grounding_axes),
            grounding_slice_stride=int(grounding_slice_stride),
            grounding_chunk=int(grounding_chunk),
            grounding_pretrained=bool(grounding_pretrained),
        )
        if grounding_encoder is not _DEFAULT:
            net_kwargs["grounding_encoder"] = grounding_encoder
        net_kwargs.update(dict(dit_overrides or {}))
        self.dit = VoxelDiT2_3D(**net_kwargs)

        self._backbone_loaded = False
        if pretrained:
            self._load_pretrained(checkpoint, cache_dir, hf_token)
        self.dit.to(self._dtype)

        depth = self.dit.depth
        if feature_layers is not None:
            layers = sorted({int(i) for i in feature_layers})
            if not layers or layers[0] < 0 or layers[-1] >= depth:
                raise ValueError(f"feature_layers must lie in [0, {depth - 1}], got {feature_layers}.")
        else:
            layers = sorted({depth // 4, depth // 2, 3 * depth // 4, depth - 1})
        self._feature_layers = layers

        patch = self.dit.patch_size
        self.pixel_channels = int(pixel_channels)
        self.feature_projector = FeatureProjector3D(
            hidden_dim=self.dit.hidden_size,
            num_feature_layers=len(layers),
            out_dim=self.feature_size,
        )
        self.to_pixels = PointwiseLinear(self.feature_size, math.prod(patch) * self.pixel_channels)
        if highres_skip:
            c = int(highres_skip_channels)
            self.skip_stem: Optional[nn.Module] = nn.Sequential(
                nn.Conv3d(self.in_channels, c, kernel_size=3, padding=1),
                group_norm(c),
                nn.GELU(),
                nn.Conv3d(c, c, kernel_size=3, padding=1),
                group_norm(c),
                nn.GELU(),
            )
            skip_ch = c
        else:
            self.skip_stem = None
            skip_ch = 0
        self.head = TaskHead3D(
            in_channels=self.pixel_channels + self.dit.out_channels + skip_ch,
            out_channels=self.head_channels,
            refine_channels=self.feature_size,
            dropout=dropout,
        )

        if initial_frozen:
            self.freeze_dit_backbone()
        else:
            self.unfreeze_dit_backbone(_log=False)
        self.dit.gradient_checkpointing = "dit" in self._ckpt_targets

        logger.info(
            "%s initialised: variant=%s, patch=%s, feature_layers=%s, grounding=%s, "
            "backbone_loaded=%s, frozen_dit=%s, dit_thaw_epoch=%s, grad_ckpt=%s, "
            "params=%s (trainable=%s)",
            type(self).__name__, self.variant, patch, layers,
            getattr(self.dit.dino_conditioner, "encoder_name", None),
            self._backbone_loaded, self._freeze_dit_backbone, self._dit_thaw_epoch,
            sorted(self._ckpt_targets),
            f"{self.get_num_parameters(trainable_only=False):,}",
            f"{self.get_num_parameters(trainable_only=True):,}",
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _load_pretrained(
        self, checkpoint: Optional[str], cache_dir: Optional[str], hf_token: Optional[str],
    ) -> None:
        filename = checkpoint or self.cfg.hf_filename
        if not filename:
            logger.warning(
                "VoxelDiT2-3D variant %s has no released weights -- training from random init.",
                self.variant,
            )
            return
        try:
            path = download_checkpoint(
                self.cfg.hf_repo_id, filename, revision=self.cfg.hf_revision,
                cache_dir=cache_dir, token=hf_token,
            )
            load_pretrained_2d(self.dit, path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PixelDiT2 checkpoint %s could not be loaded (%s) -- training from random init.",
                filename, exc,
            )
            return
        self._backbone_loaded = True

    # ------------------------------------------------------------------
    # Pixel space
    # ------------------------------------------------------------------

    @property
    def vae_input_pm1(self) -> bool:
        """The DiT works on ``[-1, 1]`` voxels, so the ``raw`` target is scaled likewise.

        The name is the one the Lightning modules read.
        """
        return True

    @staticmethod
    def _as_volume(x: torch.Tensor) -> torch.Tensor:
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()
        return x.unsqueeze(1) if x.dim() == 4 else x

    def _to_model_space(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """``[0, 1]`` volume -> ``[-1, 1]`` in the DiT dtype, padded to whole patches."""
        x = self._as_volume(x)
        spatial = tuple(int(s) for s in x.shape[-3:])
        pz, py, px = self.dit.patch_size
        pad = ((px - spatial[2] % px) % px, (py - spatial[1] % py) % py, (pz - spatial[0] % pz) % pz)
        x = x.to(self._dit_dtype()) * 2.0 - 1.0
        if any(pad):
            x = F.pad(x, (0, pad[0], 0, pad[1], 0, pad[2]), mode="replicate")
        return x, spatial  # type: ignore[return-value]

    def _dit_dtype(self) -> torch.dtype:
        return self.dit.s_embedder.proj.weight.dtype

    @staticmethod
    def _crop(x: torch.Tensor, spatial: Sequence[int]) -> torch.Tensor:
        d, h, w = spatial
        if tuple(x.shape[-3:]) == (d, h, w):
            return x
        return x[..., :d, :h, :w]

    def _timesteps(self, value: Union[float, torch.Tensor], batch: int, device) -> torch.Tensor:
        if torch.is_tensor(value):
            t = value.to(device=device, dtype=torch.float32).reshape(-1)
            return t.expand(batch) if t.numel() == 1 else t
        return torch.full((batch,), float(value), device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Task forward
    # ------------------------------------------------------------------

    def _head_input(
        self, x_model: torch.Tensor, x_pred: torch.Tensor, *feats: torch.Tensor,
    ) -> torch.Tensor:
        grid = self.dit.token_grid(x_model.shape[-3:])
        head_dtype = next(self.head.parameters()).dtype
        fused = self.feature_projector([f.to(head_dtype) for f in feats], *grid)
        pz, py, px = self.dit.patch_size
        pix = rearrange(
            self.to_pixels(fused),
            "b (kz ky kx c) gz gy gx -> b c (gz kz) (gy ky) (gx kx)",
            kz=pz, ky=py, kx=px,
        )
        parts = [pix, x_pred.to(head_dtype)]
        if self.skip_stem is not None:
            parts.append(self.skip_stem(x_model.to(head_dtype)))
        return torch.cat(parts, dim=1)

    def _forward_model_space(self, x_model: torch.Tensor, spatial: Sequence[int]) -> torch.Tensor:
        t = self._timesteps(self._feature_timestep, x_model.shape[0], x_model.device)
        x_pred, feats = self.dit(x_model, t, feature_layers=self._feature_layers)
        feat_list = [feats[i] for i in self._feature_layers]
        if "head" in self._ckpt_targets and self.training and torch.is_grad_enabled():
            # One checkpoint per stage: the head runs at full voxel resolution,
            # where recomputing all of it at once holds every stage's activations.
            out = torch.utils.checkpoint.checkpoint(
                self._head_input, x_model, x_pred, *feat_list, use_reentrant=False,
            )
            for stage in self.head.block:
                out = torch.utils.checkpoint.checkpoint(stage, out, use_reentrant=False)
        else:
            out = self.head(self._head_input(x_model, x_pred, *feat_list))
        return self._crop(out, spatial)

    def forward(self, x: torch.Tensor, prompts: Any = None) -> torch.Tensor:
        """``[B, C, D, H, W]`` in ``[0, 1]`` -> ``[B, head_channels, D, H, W]`` (raw logits / values).

        ``prompts`` is accepted for interface parity and ignored.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_model, spatial = self._to_model_space(x)
            return self._forward_model_space(x_model, spatial)

    # ------------------------------------------------------------------
    # Rectified flow
    # ------------------------------------------------------------------

    def predict_velocity(
        self, x_t: torch.Tensor, t_unit: Union[float, torch.Tensor], prompts: Any = None,
    ) -> torch.Tensor:
        """Velocity ``(x_t - x_pred) / max(t, t_eps)`` of a model-space state ``x_t``."""
        t = self._timesteps(t_unit, x_t.shape[0], x_t.device)
        x_pred = self.dit(x_t.to(self._dit_dtype()), t)
        scale = t.clamp_min(self._flow_t_eps).view(-1, 1, 1, 1, 1)
        return (x_t.float() - x_pred.float()) / scale

    def _match_clean_grid(self, degraded: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        clean = self._as_volume(clean)
        if clean.shape[-3:] == degraded.shape[-3:]:
            return clean
        return F.interpolate(
            clean.float(), size=degraded.shape[-3:], mode="trilinear", align_corners=False,
        ).to(dtype=degraded.dtype)

    def rectified_flow_loss(
        self, degraded: torch.Tensor, clean: torch.Tensor, prompts: Any = None,
    ) -> torch.Tensor:
        """Velocity MSE on the straight path from the clean volume (``t = 0``) to the degraded one.

        Both are dataset-range ``[0, 1]`` volumes; a coarser clean target is
        trilinearly resampled onto the degraded grid. ``t`` is uniform in
        ``flow_t_range``, one draw per sample.
        """
        with torch.autocast(device_type=degraded.device.type, enabled=False):
            degraded = self._as_volume(degraded)
            x1, spatial = self._to_model_space(degraded)
            x0, _ = self._to_model_space(self._match_clean_grid(degraded, clean))
            x0, x1 = x0.detach().float(), x1.detach().float()
            lo, hi = self._flow_t_range
            t = torch.empty(x0.shape[0], device=x0.device).uniform_(lo, hi)
            x_t, velocity = interpolate_rectified_flow(x0, x1, t)
            pred = self.predict_velocity(x_t, t)
            return F.mse_loss(self._crop(pred, spatial), self._crop(velocity, spatial))

    @torch.no_grad()
    def few_step_reconstruct(
        self, images: torch.Tensor, prompts: Any = None, steps: int = 4,
    ) -> torch.Tensor:
        """Task head on the volume Euler-integrated from the degraded input toward clean."""
        with torch.autocast(device_type=images.device.type, enabled=False):
            x1, spatial = self._to_model_space(images)
            refined = euler_integrate(
                x1.float(), lambda state, t: self.predict_velocity(state, t), steps,
            )
            return self._forward_model_space(refined.to(x1.dtype), spatial)

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _freeze_grounding_encoder(self) -> None:
        conditioner = self.dit.dino_conditioner
        if conditioner is not None:
            conditioner.encoder.requires_grad_(False)
            conditioner.encoder.eval()

    def freeze_dit_backbone(self) -> None:
        self.dit.requires_grad_(False)
        self._freeze_dit_backbone = True
        logger.info("VoxelDiT2-3D backbone frozen (%s trainable params).",
                    f"{self.get_num_parameters(True):,}")

    def unfreeze_dit_backbone(self, _log: bool = True) -> None:
        """Thaw the DiT; the DINOv3 grounding encoder stays frozen."""
        self.dit.requires_grad_(True)
        self.dit.train(self.training)
        self._freeze_grounding_encoder()
        self._freeze_dit_backbone = False
        if _log:
            logger.info("VoxelDiT2-3D backbone thawed (%s trainable params).",
                        f"{self.get_num_parameters(True):,}")

    @property
    def dit_thaw_epoch(self) -> Optional[int]:
        return self._dit_thaw_epoch

    # ------------------------------------------------------------------
    # Gradient checkpointing / utilities
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self, targets: Sequence[str] = ("dit", "head")) -> None:
        self._ckpt_targets |= {str(t).lower() for t in targets} & _CKPT_TARGETS
        self.dit.gradient_checkpointing = "dit" in self._ckpt_targets

    def disable_gradient_checkpointing(self) -> None:
        self._ckpt_targets = set()
        self.dit.gradient_checkpointing = False

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_output_channels(self) -> int:
        return self.head_channels


__all__ = ["VoxelDiT2Wrapper3D"]
