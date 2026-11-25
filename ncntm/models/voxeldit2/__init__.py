"""VoxelDiT2-3D: representation-grounded pixel diffusion transformer for EM volumes.

A volumetric port of PixelDiT2 (NVlabs/PixelDiT, commit ``f7d7c35``): the
DiT runs on ``16^3`` voxel patches of the EM volume itself, with tri-planar
DINOv3 grounding and a 3-D rotary embedding. See :class:`VoxelDiT2Wrapper3D`.

The files of this package are derived from PixelDiT and are distributed
under the NVIDIA Source Code License (NSCLv1, non-commercial use only); see
``LICENSE`` next to this file.

References:
    - PixelDiT2: Representation-Grounded Pixel Diffusion Transformers, arXiv 2609.24919
    - https://huggingface.co/nvidia/PixelDiT2-ImageNet
"""

from ncntm.models.voxeldit2.network import VoxelDiT2_3D
from ncntm.models.voxeldit2.variants import _VARIANT_CONFIGS
from ncntm.models.voxeldit2.wrapper import VoxelDiT2Wrapper3D

__all__ = ["VoxelDiT2Wrapper3D", "VoxelDiT2_3D", "_VARIANT_CONFIGS"]
