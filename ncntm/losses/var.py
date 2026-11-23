"""Residual next-scale reconstruction, coarse to fine, ending at 4 nm.

Visual autoregressive modeling (Tian et al., 2024, arXiv:2404.02905)
predicts an image as a residual pyramid. Each scale is the detail the
coarser scales have not explained, and the last scale is the original
resolution. Algorithms 1 and 2 of that paper quantize those residuals.
The EM target is continuous and the Wan VAE stays frozen, so the
quantizer here is the identity: average-pool to the scale, nearest-
upsample back to the 4 nm grid, and subtract.

The sum of the scale volumes is exactly the 4 nm input. The last volume
is the 4 nm residual. A crop measured at 4 nm is scored on every scale,
so the loss supervises that 4 nm reconstruction. A coarser crop keeps the
native-voxel L1 plus any scale it was measured finely enough to support.
Its 4 nm residual is still part of the raw head, and it is not matched to
a 4 nm target the crop does not have.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


def scale_factors(scales_nm: Sequence[float], fine_nm: float = 4.0) -> Tuple[int, ...]:
    """Integer pool factors for a coarse-to-fine schedule ending at ``fine_nm``.

    ``32, 16, 8, 4`` with ``fine_nm=4`` is ``(8, 4, 2, 1)``. Each coarser
    factor has to be divisible by the next so the grids nest, and the last
    factor is 1, the 4 nm grid itself.
    """
    fine_nm = float(fine_nm)
    if fine_nm <= 0:
        raise ValueError(f"fine_nm must be positive, got {fine_nm}.")
    scales = [float(s) for s in scales_nm]
    if not scales:
        raise ValueError("VAR needs at least the finest scale.")
    if abs(scales[-1] - fine_nm) > 1e-3:
        raise ValueError(
            f"The last VAR scale must be the fine grid ({fine_nm:g} nm); "
            f"got {scales[-1]:g} nm."
        )
    factors: List[int] = []
    for scale in scales:
        ratio = scale / fine_nm
        factor = int(round(ratio))
        if factor < 1 or abs(ratio - factor) > 1e-3:
            raise ValueError(
                f"VAR scale {scale:g} nm is not an integer multiple of {fine_nm:g} nm."
            )
        factors.append(factor)
    for prev, nxt in zip(factors, factors[1:]):
        if prev <= nxt or prev % nxt != 0:
            raise ValueError(
                "VAR scales must be strictly coarse-to-fine and nested; "
                f"got factors {factors}."
            )
    return tuple(factors)


def residual_components(x: torch.Tensor, factors: Sequence[int]) -> List[torch.Tensor]:
    """Split ``x`` into residual scale volumes that sum back to ``x``.

    ``x`` is ``[N, C, D, H, W]`` on the 4 nm grid. ``factors`` is coarse to
    fine and ends at 1. Each coarser scale is the average of the residual
    still unexplained, nearest-repeated onto the 4 nm grid and then
    subtracted. The last tensor is whatever remains, which is the 4 nm
    residual.
    """
    factors = tuple(int(f) for f in factors)
    if not factors or factors[-1] != 1:
        raise ValueError(
            f"VAR factors must end at 1 (the 4 nm grid); got {factors}."
        )
    full = tuple(int(s) for s in x.shape[-3:])
    f = x.float()
    parts: List[torch.Tensor] = []
    for factor in factors:
        if factor == 1:
            parts.append(f)
            break
        if any(size % factor != 0 for size in full):
            raise ValueError(
                f"VAR factor {factor} does not divide spatial size {full}."
            )
        coarse = F.avg_pool3d(f, kernel_size=factor, stride=factor)
        up = F.interpolate(coarse, size=full, mode="nearest")
        parts.append(up)
        f = f - up
    return parts


def scale_supported(
    native: Sequence[int],
    fine: Sequence[int],
    scale_nm: float,
    fine_nm: float,
) -> bool:
    """True when every axis was measured at least as finely as ``scale_nm``.

    Native voxel size is ``fine_nm * fine_voxels / native_voxels``. A 40 nm
    section does not support an 8 nm or 4 nm scale.
    """
    for n_vox, f_vox in zip(native, fine):
        n_vox = int(n_vox)
        f_vox = int(f_vox)
        if n_vox < 1:
            return False
        native_nm = float(fine_nm) * f_vox / n_vox
        if native_nm > float(scale_nm) + 1e-3:
            return False
    return True


__all__ = ["residual_components", "scale_factors", "scale_supported"]
