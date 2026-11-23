"""Plain containers for the derived representations.

Every coordinate is in TRUE PHYSICAL NANOMETRES, (x, y, z) order, corner convention
(see ``grid.py``).  Moving a representation into "index" space, or into scene units,
is done by the code that draws it, never stored here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bricks:
    """instance, drawn voxel by voxel: one box per voxel."""

    kji: np.ndarray                    # (N, 3) int   array indices [k, j, i] of the voxels
    centers_xyz_nm: np.ndarray         # (N, 3) float voxel centres
    size_xyz_nm: np.ndarray            # (3,)   float voxel size (dx, dy, dz)
    instance_id: int = 0
    n_voxels_total: int = 0            # voxels of the instance (N may be the surface shell only)

    def __len__(self) -> int:
        return len(self.centers_xyz_nm)


@dataclass
class Mesh:
    """wiremesh / surface: triangles, counter-clockwise seen from outside."""

    vertices_xyz_nm: np.ndarray        # (V, 3) float
    faces: np.ndarray                  # (F, 3) int32
    instance_id: int = 0

    def edges(self) -> np.ndarray:
        """Unique undirected edges (E, 2): the wires of the wiremesh."""
        f = np.asarray(self.faces)
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        e.sort(axis=1)
        return np.unique(e, axis=0).astype(np.int32)

    def area_nm2(self) -> float:
        v = self.vertices_xyz_nm
        f = self.faces
        return float(0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum())

    def volume_nm3(self) -> float:
        """Signed enclosed volume; positive when the winding is outward."""
        v = self.vertices_xyz_nm
        f = self.faces
        return float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6.0)


@dataclass
class Skeleton:
    """skeleton: a tree (or forest) of nodes with radii."""

    vertices_xyz_nm: np.ndarray        # (N, 3) float
    edges: np.ndarray                  # (E, 2) int32 indices into vertices
    radii_nm: np.ndarray               # (N,)   float distance from node to the nearest boundary
    instance_id: int = 0

    def cable_length_nm(self) -> float:
        v = self.vertices_xyz_nm
        e = self.edges
        return float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).sum()) if len(e) else 0.0

    def degrees(self) -> np.ndarray:
        return np.bincount(np.asarray(self.edges).ravel(), minlength=len(self.vertices_xyz_nm))

    def n_branch_points(self) -> int:
        return int((self.degrees() > 2).sum())

    def n_end_points(self) -> int:
        return int((self.degrees() == 1).sum())

    def polylines(self) -> list:
        """Split the graph at branch and end points into chains of vertex indices,
        so each chain can be drawn as one curve."""
        n = len(self.vertices_xyz_nm)
        adj = [[] for _ in range(n)]
        for a, b in np.asarray(self.edges):
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
        deg = np.array([len(x) for x in adj])
        seen = set()
        lines = []
        starts = [i for i in range(n) if deg[i] != 2 and deg[i] > 0]
        for s in starts:
            for nb in adj[s]:
                if (s, nb) in seen:
                    continue
                line = [s, nb]
                seen.add((s, nb))
                seen.add((nb, s))
                prev, cur = s, nb
                while deg[cur] == 2:
                    nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                    seen.add((cur, nxt))
                    seen.add((nxt, cur))
                    line.append(nxt)
                    prev, cur = cur, nxt
                lines.append(np.array(line, dtype=np.int32))
        # closed loops with no branch or end point
        for a, b in np.asarray(self.edges):
            a, b = int(a), int(b)
            if (a, b) in seen:
                continue
            line = [a, b]
            seen.add((a, b))
            seen.add((b, a))
            prev, cur = a, b
            while cur != a:
                nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                seen.add((cur, nxt))
                seen.add((nxt, cur))
                line.append(nxt)
                prev, cur = cur, nxt
            lines.append(np.array(line, dtype=np.int32))
        return lines


@dataclass
class Gaussians:
    """gaussian: a mixture of 3D Gaussians, each a mean, a covariance and a weight."""

    means_xyz_nm: np.ndarray           # (K, 3)    float
    covariances_nm2: np.ndarray        # (K, 3, 3) float symmetric positive definite
    weights: np.ndarray                # (K,)      float fraction of the instance's voxels
    colors_rgb: np.ndarray = field(default=None)  # (K, 3) float in [0, 1], optional
    instance_id: int = 0

    def __len__(self) -> int:
        return len(self.means_xyz_nm)

    def principal_axes(self):
        """Eigen-decomposition of each covariance.

        Returns ``(rotations, sigmas_nm)``: rotations (K, 3, 3) are proper (det = +1) with
        the principal directions in their columns, sigmas (K, 3) the standard deviations
        along them, so  covariance = R diag(sigma^2) R^T."""
        w, v = np.linalg.eigh(np.asarray(self.covariances_nm2, dtype=np.float64))
        flip = np.linalg.det(v) < 0
        v[flip, :, 0] *= -1.0
        return v, np.sqrt(np.clip(w, 0.0, None))

    def elongation(self) -> np.ndarray:
        """Longest over shortest standard deviation of each Gaussian."""
        _, s = self.principal_axes()
        return s.max(axis=1) / np.maximum(s.min(axis=1), 1e-12)


def rotation_to_quaternion_wxyz(rot: np.ndarray) -> np.ndarray:
    """Proper rotation matrices (..., 3, 3) -> unit quaternions (..., 4) as (w, x, y, z)."""
    r = np.asarray(rot, dtype=np.float64)
    out = np.empty(r.shape[:-2] + (4,))
    m00, m11, m22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    tr = m00 + m11 + m22
    q = np.stack(
        [
            1.0 + tr,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        axis=-1,
    )
    best = q.argmax(axis=-1)
    for idx in np.ndindex(best.shape):
        m = r[idx]
        b = best[idx]
        if b == 0:
            v = np.array([1.0 + tr[idx], m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
        elif b == 1:
            v = np.array([m[2, 1] - m[1, 2], q[idx][1], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0]])
        elif b == 2:
            v = np.array([m[0, 2] - m[2, 0], m[0, 1] + m[1, 0], q[idx][2], m[1, 2] + m[2, 1]])
        else:
            v = np.array([m[1, 0] - m[0, 1], m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], q[idx][3]])
        v = v / np.linalg.norm(v)
        out[idx] = v if v[0] >= 0 else -v
    return out
