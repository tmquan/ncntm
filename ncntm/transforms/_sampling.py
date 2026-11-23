"""Small shared sampling helpers for the connectomics transform pipeline."""

import numpy as np


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Sample log-uniformly from ``[lo, hi]`` (scale-symmetric).

    Used by :mod:`~ncntm.transforms.degrade` (voxel-size factor).
    """
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
