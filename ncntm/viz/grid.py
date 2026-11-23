"""The voxel grid: the one place where array indices become physical coordinates.

Everything in ``ncntm.viz`` follows the conventions fixed here.

Axis order
    Arrays are stored (z, y, x): ``volume[k, j, i]``.  Coordinates are (x, y, z).
    The reversal happens once, in this module, and nowhere else.

Voxel placement (corner convention, as in Neuroglancer)
    Voxel ``[k, j, i]`` is the box  x in [i*dx, (i+1)*dx],  y in [j*dy, (j+1)*dy],
    z in [k*dz, (k+1)*dz].  Its centre is at ((i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz).
    The whole array fills the box [0, nx*dx] x [0, ny*dy] x [0, nz*dz].

Units
    Geometry is computed and exchanged in nanometres (float64).
    A scene (USD stage, renderer) uses 1 unit = 1 micrometre: divide by ``NM_PER_UNIT``.

Display spaces
    ``"physical"``  every voxel has its true size (dx, dy, dz).
    ``"index"``     every voxel is drawn as a cube of the in-plane pixel size g = min(dx, dy).
                    This is what you see when anisotropy is ignored.  x and y keep their
                    size; z shrinks (or grows) by the factor dz / g.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NM_PER_UNIT = 1000.0  # 1 scene unit = 1 micrometre = 1000 nm
SPACES = ("physical", "index")


def zyx_to_xyz(a):
    """Reverse the last axis: (z, y, x) -> (x, y, z), or back.  A reflection of the
    coordinate *labels* only; it does not mirror the tissue."""
    return np.asarray(a)[..., ::-1]


@dataclass(frozen=True)
class VoxelGrid:
    """Shape and voxel size of a (z, y, x) array.

    >>> g = VoxelGrid((100, 512, 512), (30.0, 6.0, 6.0))      # SNEMI3D
    >>> g.extent_xyz_nm                                        # a cube, although the array is flat
    array([3072., 3072., 3000.])
    >>> g.anisotropy
    5.0
    """

    shape_zyx: tuple
    voxel_size_zyx_nm: tuple

    def __post_init__(self):
        shape = tuple(int(n) for n in self.shape_zyx)
        size = tuple(float(s) for s in self.voxel_size_zyx_nm)
        if len(shape) != 3 or len(size) != 3:
            raise ValueError("shape_zyx and voxel_size_zyx_nm must have three entries (z, y, x)")
        if min(size) <= 0:
            raise ValueError(f"voxel size must be positive, got {size}")
        object.__setattr__(self, "shape_zyx", shape)
        object.__setattr__(self, "voxel_size_zyx_nm", size)

    # ---- sizes -----------------------------------------------------------------
    @property
    def shape_xyz(self) -> np.ndarray:
        return np.array(self.shape_zyx[::-1], dtype=np.int64)

    @property
    def voxel_size_xyz_nm(self) -> np.ndarray:
        return np.array(self.voxel_size_zyx_nm[::-1], dtype=np.float64)

    @property
    def in_plane_nm(self) -> float:
        """g = min(dx, dy): the edge of the cube a voxel is drawn as in index space."""
        dx, dy, _ = self.voxel_size_xyz_nm
        return float(min(dx, dy))

    @property
    def anisotropy(self) -> float:
        """dz / g.  1 for isotropic data, 5 for SNEMI3D, 10 for CREMI."""
        return float(self.voxel_size_xyz_nm[2] / self.in_plane_nm)

    @property
    def is_isotropic(self) -> bool:
        s = self.voxel_size_xyz_nm
        return bool(np.allclose(s, s[0]))

    def display_voxel_size_xyz_nm(self, space: str = "physical") -> np.ndarray:
        """Size of one voxel as drawn in ``space``."""
        if space == "physical":
            return self.voxel_size_xyz_nm
        if space == "index":
            return np.full(3, self.in_plane_nm)
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")

    def extent_nm(self, space: str = "physical") -> np.ndarray:
        """Edge lengths (x, y, z) of the whole array as drawn in ``space``."""
        return self.shape_xyz * self.display_voxel_size_xyz_nm(space)

    @property
    def extent_xyz_nm(self) -> np.ndarray:
        """True physical edge lengths (x, y, z) of the whole array."""
        return self.extent_nm("physical")

    # ---- index <-> nanometres ---------------------------------------------------
    def index_to_nm(self, ijk, space: str = "physical") -> np.ndarray:
        """Continuous index coordinates (i, j, k) = (x, y, z) order -> nanometres.
        Integer coordinates are voxel corners; add 0.5 for centres."""
        return np.asarray(ijk, dtype=np.float64) * self.display_voxel_size_xyz_nm(space)

    def nm_to_index(self, xyz_nm, space: str = "physical") -> np.ndarray:
        """Nanometres -> continuous index coordinates (i, j, k).  ``floor`` gives the voxel."""
        return np.asarray(xyz_nm, dtype=np.float64) / self.display_voxel_size_xyz_nm(space)

    def voxel_centers_nm(self, kji, space: str = "physical") -> np.ndarray:
        """Array indices [k, j, i] (as returned by ``np.argwhere``) -> centres (x, y, z) in nm."""
        ijk = zyx_to_xyz(np.asarray(kji, dtype=np.float64))
        return (ijk + 0.5) * self.display_voxel_size_xyz_nm(space)

    def nm_to_voxel(self, xyz_nm, space: str = "physical") -> np.ndarray:
        """Nanometres -> integer array indices [k, j, i] of the containing voxel (not clipped)."""
        ijk = np.floor(self.nm_to_index(xyz_nm, space)).astype(np.int64)
        return zyx_to_xyz(ijk)

    # ---- physical <-> display space ----------------------------------------------
    def space_scale(self, space: str = "physical") -> np.ndarray:
        """Per-axis factor A (x, y, z) that maps true physical nm to nm as drawn in ``space``.
        Points transform as p -> A * p; covariances as S -> diag(A) S diag(A)."""
        return self.display_voxel_size_xyz_nm(space) / self.voxel_size_xyz_nm

    def to_space(self, xyz_nm, space: str = "physical") -> np.ndarray:
        """Move points given in true physical nm into ``space`` (identity for "physical")."""
        return np.asarray(xyz_nm, dtype=np.float64) * self.space_scale(space)

    def covariance_to_space(self, cov_nm2, space: str = "physical") -> np.ndarray:
        """Move covariance matrices (..., 3, 3) given in true physical nm^2 into ``space``."""
        a = self.space_scale(space)
        return np.asarray(cov_nm2, dtype=np.float64) * a[:, None] * a[None, :]

    # ---- matrices ------------------------------------------------------------------
    def index_to_unit_matrix(self, space: str = "physical") -> np.ndarray:
        """4x4 matrix taking continuous index coordinates (i, j, k, 1) to scene units
        (micrometres).  Its diagonal IS the anisotropy; there is nothing else in it."""
        m = np.eye(4)
        m[0, 0], m[1, 1], m[2, 2] = self.display_voxel_size_xyz_nm(space) / NM_PER_UNIT
        return m

    # ---- deliberately wrong grids, for teaching --------------------------------------
    def as_cubes(self) -> "VoxelGrid":
        """The same array with the voxel size a careless program assumes: cubes of the
        in-plane pixel size.  Feed this to any converter to see what ignoring anisotropy does."""
        g = self.in_plane_nm
        return VoxelGrid(self.shape_zyx, (g, g, g))

    def strided(self, step: int, axis: int = 0) -> "VoxelGrid":
        """Grid of ``array[::step]`` along array axis ``axis`` (0 = z): fewer, thicker voxels."""
        shape = list(self.shape_zyx)
        size = list(self.voxel_size_zyx_nm)
        shape[axis] = len(range(0, shape[axis], step))
        size[axis] = size[axis] * step
        return VoxelGrid(tuple(shape), tuple(size))

    def describe(self) -> str:
        nz, ny, nx = self.shape_zyx
        dz, dy, dx = self.voxel_size_zyx_nm
        ex, ey, ez = self.extent_xyz_nm / 1000.0
        return (
            f"{nx} x {ny} x {nz} voxels (x, y, z) of {dx:g} x {dy:g} x {dz:g} nm "
            f"= {ex:.2f} x {ey:.2f} x {ez:.2f} um, anisotropy {self.anisotropy:g}:1"
        )
