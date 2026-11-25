"""Variant registry for VoxelDiT2-3D.

Transformer geometry follows the upstream ``PixelDiT2_models`` table
(``pixdit_core/pixeldit2_c2i.py``); the patch is a ``16^3`` voxel cube, so a
``256^3`` crop is a ``16^3 = 4096``-token sequence. Only ``H/16`` has
released weights (``nvidia/PixelDiT2-ImageNet``, 2-D, ImageNet 256 px); with
``pretrained: true`` they are inflated to 3-D by
:mod:`ncntm.models.voxeldit2.checkpoint`. The other variants train from
scratch.

Approximate DiT size (without the frozen DINOv3 ViT-S/16, 22M); the
per-block adaLN projection (``6 * hidden^2``) is a third of every block:

=======  ======  =====  =====  ======
variant  hidden  depth  heads  params
=======  ======  =====  =====  ======
B/16        768     12     12  ~0.14B
L/16       1024     24     16  ~0.47B
H/16       1280     32     16  ~0.96B
=======  ======  =====  =====  ======
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class _VariantConfig:
    hidden_size: int
    num_heads: int
    depth: int
    in_context_start: int
    patch_size: Tuple[int, int, int] = (16, 16, 16)
    in_context_len: int = 32
    mlp_hidden_multiple: int = 0
    grounding_encoder: Optional[str] = "dinov3_vits16"
    grounding_proj_blocks: int = 1
    grounding_proj_heads: Optional[int] = 16
    hf_repo_id: Optional[str] = None
    hf_filename: Optional[str] = None
    hf_revision: str = "main"
    extra: Dict[str, Any] = field(default_factory=dict)

    def network_kwargs(self) -> Dict[str, Any]:
        """Constructor kwargs of :class:`~ncntm.models.voxeldit2.network.VoxelDiT2_3D`."""
        out = asdict(self)
        for key in ("hf_repo_id", "hf_filename", "hf_revision", "extra"):
            out.pop(key)
        out.update(self.extra)
        return out


_VARIANT_CONFIGS: Dict[str, _VariantConfig] = {
    "B/16": _VariantConfig(hidden_size=768, num_heads=12, depth=12, in_context_start=4),
    "L/16": _VariantConfig(hidden_size=1024, num_heads=16, depth=24, in_context_start=8),
    "H/16": _VariantConfig(
        hidden_size=1280,
        num_heads=16,
        depth=32,
        in_context_start=8,
        hf_repo_id="nvidia/PixelDiT2-ImageNet",
        hf_filename="pixeldit2_h16_in256_ep600.ckpt",
    ),
}


__all__ = ["_VARIANT_CONFIGS", "_VariantConfig"]
