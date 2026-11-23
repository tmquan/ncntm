"""From one instance to the other representations: bricks, wiremesh, skeleton, gaussian.

Every converter takes ``(mask, grid)``:

    mask   bool array (z, y, x), True inside the instance  (``sample.mask(i)``)
    grid   VoxelGrid of that array                         (``sample.grid``)

and returns a container of ``types.py`` in TRUE PHYSICAL NANOMETRES, (x, y, z), corner
convention: voxel [k, j, i] is the box [i*dx, (i+1)*dx] x [j*dy, (j+1)*dy] x [k*dz, (k+1)*dz].

Where the voxel size enters
    The voxel size is read from ``grid`` and from nowhere else, through three of its
    members: ``voxel_size_xyz_nm`` (spacing for the mesher and the skeletoniser, the
    within-voxel term of a covariance, the volume of a voxel), ``voxel_centers_nm``
    (index -> nanometres) and ``space_scale`` (cube space <-> physical space).  Search
    this file for those three names to see every place.

Ignoring anisotropy
    There is no "wrong" switch.  To see what a program that ignores the voxel size
    produces, pass ``grid.as_cubes()`` instead of ``grid``:

        right = instance_to_mesh(mask, grid)
        wrong = instance_to_mesh(mask, grid.as_cubes())

Axis order inside this module
    The libraries (zmesh, kimimaro, edt, scikit-image) apply spacing[n] to ARRAY AXIS n
    and return coordinates in array-axis order.  ``mask.T`` is a zero-copy (x, y, z) view
    of the (z, y, x) array, so meshing or skeletonising ``mask.T`` with (dx, dy, dz)
    gives (x, y, z) nanometres directly, with the winding untouched.  (Reversing the
    columns of a (z, y, x) result instead is a mirror image and turns a mesh inside out.)
    The libraries put the CENTRE of voxel n at n * spacing; the corner convention puts
    it at (n + 0.5) * spacing.  That half voxel is added in ``_to_corner_frame``.
"""
from __future__ import annotations

import numpy as np

from .grid import VoxelGrid
from .types import Bricks, Gaussians, Mesh, Skeleton

# kimimaro TEASAR parameters.  Lengths are in nm.  ``const`` is the one that matters on
# anisotropic data: the path is a chain of voxels, each step costs the same whatever its
# physical length, and with a small const every flat end of an object grows a twig.
TEASAR_DEFAULTS = {
    "scale": 1.5,
    "const": 300.0,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_detection_threshold": 750,
    "soma_acceptance_threshold": 3500,
    "soma_invalidation_scale": 2,
    "soma_invalidation_const": 300,
}

# Fixed colours for the parts of a mixture (matplotlib "tab10"), RGB in [0, 1].
PALETTE_RGB = np.array(
    [
        [0.122, 0.467, 0.706],
        [1.000, 0.498, 0.055],
        [0.173, 0.627, 0.173],
        [0.839, 0.153, 0.157],
        [0.580, 0.404, 0.741],
        [0.549, 0.337, 0.294],
        [0.890, 0.467, 0.761],
        [0.498, 0.498, 0.498],
        [0.737, 0.741, 0.133],
        [0.090, 0.745, 0.812],
    ]
)


# ---- input checks and the two coordinate helpers every converter uses ---------------------
def _check_mask(mask, grid) -> np.ndarray:
    """Return ``mask`` as a bool array after checking that it fits ``grid`` and is not empty.
    (A wrong axis order cannot be detected; a wrong shape, type or an empty mask can.)"""
    if not isinstance(grid, VoxelGrid):
        raise TypeError(f"grid must be a VoxelGrid, got {type(grid).__name__}")
    mask = np.asarray(mask)
    if mask.dtype != bool:
        raise TypeError(f"mask must be a boolean array (for example labels == i), got dtype {mask.dtype}")
    if mask.ndim != 3 or tuple(mask.shape) != tuple(grid.shape_zyx):
        raise ValueError(f"mask has shape {mask.shape} but the grid describes a (z, y, x) array of shape {grid.shape_zyx}")
    if not mask.any():
        raise ValueError("mask is empty: there is no instance to convert")
    return mask


def _crop_to_mask(mask, margin: int = 0):
    """Tight bounding box of the mask, grown by ``margin`` voxels but never beyond the array.

    Returns ``(sub, start_kji)``: ``sub = mask[k0:k1, j0:j1, i0:i1]`` and its first index."""
    start, stop = [], []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        hit = np.flatnonzero(mask.any(axis=other))
        start.append(max(int(hit[0]) - margin, 0))
        stop.append(min(int(hit[-1]) + 1 + margin, mask.shape[axis]))
    sub = mask[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
    return sub, np.array(start, dtype=np.int64)


def _to_corner_frame(xyz_library, grid, first_voxel_kji) -> np.ndarray:
    """Library coordinates -> this package's coordinates, (x, y, z) nm, float64.

    ``xyz_library`` come from a library that was given a sub-array whose voxel [0, 0, 0]
    is voxel ``first_voxel_kji`` of the full array (negative when the sub-array was
    zero-padded), with (dx, dy, dz) as spacing.  The library puts that voxel's centre at
    0; in the corner convention it is at (first + 0.5) * voxel size."""
    return np.asarray(xyz_library, dtype=np.float64) + grid.voxel_centers_nm(first_voxel_kji)


# ---- instance -> bricks ---------------------------------------------------------------------
def instance_to_bricks(mask, grid, surface_only=True, max_bricks=None, *, instance_id=0) -> Bricks:
    """One box per voxel of the instance.

    surface_only  keep only voxels that have at least one of their 6 face neighbours
                  outside the mask, or that lie on the border of the array (the shell is
                  all a renderer can see of a solid object).
    max_bricks    if the result has more voxels than this, keep an evenly spaced subset
                  in array order (deterministic).  The picture then has holes;
                  ``n_voxels_total`` always counts the whole instance.

    Returns Bricks: ``kji`` array indices, ``centers_xyz_nm`` voxel centres
    ((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz), ``size_xyz_nm`` = (dx, dy, dz)."""
    mask = _check_mask(mask, grid)
    sub, start = _crop_to_mask(mask)
    keep = sub
    if surface_only:
        p = np.pad(sub, 1)                               # False outside the array
        inside = (
            p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1]
            & p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1]
            & p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:]
        )
        keep = sub & ~inside
    kji = np.argwhere(keep) + start
    if max_bricks is not None and len(kji) > int(max_bricks):
        if int(max_bricks) < 1:
            raise ValueError(f"max_bricks must be at least 1, got {max_bricks}")
        kji = kji[np.linspace(0, len(kji) - 1, int(max_bricks)).round().astype(np.int64)]
    return Bricks(
        kji=kji,
        centers_xyz_nm=grid.voxel_centers_nm(kji),
        size_xyz_nm=grid.voxel_size_xyz_nm,
        instance_id=int(instance_id),
        n_voxels_total=int(np.count_nonzero(sub)),
    )


# ---- instance -> mesh -------------------------------------------------------------------------
def instance_to_mesh(mask, grid, simplify_factor=None, max_error_nm=None, smooth_sigma_nm=None,
                     backend="zmesh", *, instance_id=0) -> Mesh:
    """Surface of the instance by marching cubes at the level 0.5 of the 0/1 mask.

    Where the surface sits.  Marching cubes samples the mask at voxel centres and puts a
    vertex half-way between an inside centre and an outside centre, which is exactly on
    the face the two voxels share.  So, unsmoothed and unsimplified, every vertex lies on
    a voxel face of the corner convention, and a solid block of voxels [i0:i1] has the
    bounding box [i0*dx, i1*dx] (same for y, z).  Edges and corners of the block are cut
    off at 45 degrees in index space; a solid box of nx * ny * nz voxels encloses
    (nx*ny*nz - (nx + ny + nz - 3)/2 - 5/6) voxel volumes.

    The mask is padded with background before meshing, so the surface is closed where
    the instance touches the border of the array.  Triangles are counter-clockwise seen
    from outside (the signed volume is checked to be positive).

    simplify_factor  reduce the triangle count by this factor, or by more where flat
                     terraces merge at no cost.  The section steps of anisotropic data
                     stay: good for a wireframe.
    max_error_nm     zmesh only: largest distance a vertex may move during simplification
                     (default: the largest voxel edge).
    smooth_sigma_nm  blur the mask with a Gaussian of this sigma before meshing.  The
                     sigma is in nm and converted per axis (sigma / voxel size), so the
                     blur is round in physical space.  Removes the staircase; the surface
                     no longer lies on voxel faces and thin parts shrink: a plate thinner
                     than 1.35 sigma disappears, whichever axis it is normal to.  Half the
                     largest voxel edge is a good sigma: the area of a 400 nm sphere on
                     30 x 6 x 6 nm voxels is 21 % too large without blur, 13 % with
                     sigma = 3 nm and under 1 % with sigma = 15 nm.  Needs backend="skimage".
    backend          "zmesh" (fast) or "skimage" (skimage.measure.marching_cubes).  Both
                     give the same surface.  With "skimage", simplification uses
                     zmesh.simplify_fqmr, which has no error bound.
    """
    mask = _check_mask(mask, grid)
    if backend not in ("zmesh", "skimage"):
        raise ValueError(f"backend must be 'zmesh' or 'skimage', got {backend!r}")
    if simplify_factor is not None and not simplify_factor >= 1:
        raise ValueError(f"simplify_factor must be >= 1, got {simplify_factor}")
    if smooth_sigma_nm is not None and not smooth_sigma_nm > 0:
        raise ValueError(f"smooth_sigma_nm must be positive, got {smooth_sigma_nm}")
    if smooth_sigma_nm is not None and backend == "zmesh":
        raise ValueError("smooth_sigma_nm needs backend='skimage' (zmesh meshes labels, not a blurred field)")
    if max_error_nm is not None and backend == "skimage":
        raise ValueError("max_error_nm needs backend='zmesh' (the skimage path has no error bound)")
    dx, dy, dz = (float(d) for d in grid.voxel_size_xyz_nm)
    simplify = simplify_factor is not None and simplify_factor > 1

    # zero padding on every side: 1 voxel closes the surface, a blur needs room for 3 sigma
    sub, start = _crop_to_mask(mask)
    pad = np.ones(3, dtype=np.int64)
    if smooth_sigma_nm is not None:
        sigma_voxels_zyx = float(smooth_sigma_nm) / np.array([dz, dy, dx])
        pad = np.ceil(3.0 * sigma_voxels_zyx).astype(np.int64) + 1
    padded = np.pad(sub, [(int(p), int(p)) for p in pad])
    first_voxel = start - pad                            # array index of padded[0, 0, 0]

    if backend == "zmesh":
        import zmesh

        mesher = zmesh.Mesher((dx, dy, dz))              # spacing for the (x, y, z) view below
        mesher.mesh(padded.view(np.uint8).T)             # .T: zero-copy (x, y, z) view
        if max_error_nm is None:
            max_error_nm = max(dx, dy, dz)
        got = mesher.get(1, normals=False, reduction_factor=int(simplify_factor) if simplify else 0,
                         max_error=float(max_error_nm), voxel_centered=False)
        mesher.clear()
        vertices, faces = got.vertices, got.faces        # (x, y, z) nm, outward winding
    else:
        from skimage.measure import marching_cubes

        field = padded.astype(np.float32)
        if smooth_sigma_nm is not None:
            from scipy import ndimage

            field = ndimage.gaussian_filter(field, sigma=sigma_voxels_zyx, mode="constant", cval=0.0)
            if field.max() <= 0.5:
                raise ValueError(f"smooth_sigma_nm={smooth_sigma_nm} blurs this instance away (nothing above 0.5)")
        zyx, faces, _, _ = marching_cubes(field, level=0.5, spacing=(dz, dy, dx))
        # skimage returns (z, y, x) columns and winds by the left-hand rule.  Reversing
        # the columns to (x, y, z) is a mirror image, which turns left-handed winding
        # into outward counter-clockwise winding: the faces must NOT be flipped as well.
        vertices = zyx[:, ::-1]
        if simplify:
            import zmesh

            small = zmesh.simplify_fqmr(
                zmesh.Mesh(np.ascontiguousarray(vertices, dtype=np.float32), np.ascontiguousarray(faces, dtype=np.uint32), None),
                target_count=max(4, int(len(faces) / simplify_factor)),
            )
            vertices, faces = small.vertices, small.faces

    mesh = Mesh(
        vertices_xyz_nm=_to_corner_frame(vertices, grid, first_voxel),
        faces=np.ascontiguousarray(faces, dtype=np.int32),
        instance_id=int(instance_id),
    )
    if len(mesh.faces) == 0:
        raise ValueError("marching cubes returned no triangles for this mask")
    if not mesh.volume_nm3() > 0:
        raise RuntimeError("mesh has a non-positive signed volume: the winding is inside out (axis order bug)")
    return mesh


# ---- instance -> skeleton ---------------------------------------------------------------------
def instance_to_skeleton(mask, grid, smooth_nm=None, *, dust_nm3=1.0e6, instance_id=0, **teasar) -> Skeleton:
    """Centre line of the instance with kimimaro (TEASAR), a tree per connected piece.

    vertices  centres of voxels of the instance, (x, y, z) nm: ((i + 0.5) * dx, ...).
              Where the instance leaves the array the skeleton ends in the middle of the
              cut cross-section (kimimaro ``fix_borders``).
    radii     nm, from the anisotropic Euclidean distance transform: the distance from
              the node's voxel centre to the centre of the nearest voxel outside the
              instance.  That is about half a voxel more than the distance to the surface
              (a rod 7 voxels of 6 nm wide gets 24 nm, its half width is 21 nm).  The
              border of the array does not count as outside, because the tissue goes on
              there (one exception, a kimimaro rule: an instance that fills the WHOLE
              array is treated as ending at the border).
    edges     pairs of vertex indices.

    smooth_nm   None, or the window (nm of arc length) of a moving average along each
                branch (``smooth_skeleton``).  The raw path is a chain of voxel centres:
                on 30 nm sections it zig-zags and over-states the cable length.  Four
                times the largest voxel edge is a good window.
    dust_nm3    connected pieces smaller than this physical volume are skipped
                (kimimaro's own threshold is a voxel count, which means a different size
                for every voxel size).  The volume is voxel count * dx * dy * dz of the
                ``grid`` given, so under ``grid.as_cubes()`` the same piece counts as
                smaller and may be skipped; dust_nm3=0 keeps every piece.
    **teasar    overrides of ``TEASAR_DEFAULTS`` (scale 1.5, const 300 nm, ...).  The
                invalidation radius around a traced path is scale * radius + const (nm).

    kimimaro runs with parallel=1 (no worker processes), so this is safe to call from a
    notebook kernel or from inside the Isaac Sim process.

    Raises ValueError for an empty mask and for one too small to have a centre line."""
    mask = _check_mask(mask, grid)
    unknown = set(teasar) - set(TEASAR_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown TEASAR parameter(s) {sorted(unknown)}; allowed: {sorted(TEASAR_DEFAULTS)}")
    import kimimaro

    dx, dy, dz = (float(d) for d in grid.voxel_size_xyz_nm)
    # one voxel of margin where the array allows it, so that the cropped block only
    # touches its own border where the instance touches the border of the array
    sub, start = _crop_to_mask(mask, margin=1)
    found = kimimaro.skeletonize(
        np.ascontiguousarray(sub).view(np.uint8).T,      # zero-copy (x, y, z) view, label 1
        teasar_params={**TEASAR_DEFAULTS, **teasar},
        anisotropy=(dx, dy, dz),                         # spacing for the (x, y, z) view
        dust_threshold=int(float(dust_nm3) / (dx * dy * dz)),
        fix_branching=True,
        fix_borders=True,
        parallel=1,
        progress=False,
    )
    if 1 not in found or len(found[1].vertices) < 2:
        n = int(np.count_nonzero(sub))
        raise ValueError(
            f"mask is too small to skeletonise: {n} voxels = {n * dx * dy * dz:.3g} nm^3 "
            f"with voxels of {dx:g} x {dy:g} x {dz:g} nm "
            f"(pieces under dust_nm3 = {dust_nm3:.3g} nm^3 are skipped)"
        )
    skeleton = Skeleton(
        vertices_xyz_nm=_to_corner_frame(found[1].vertices, grid, start),
        edges=np.ascontiguousarray(found[1].edges, dtype=np.int32).reshape(-1, 2),
        radii_nm=np.asarray(found[1].radii, dtype=np.float64).reshape(-1),
        instance_id=int(instance_id),
    )
    if not np.isfinite(skeleton.vertices_xyz_nm).all():
        raise RuntimeError("kimimaro returned non-finite vertices")
    if smooth_nm is not None:
        skeleton = smooth_skeleton(skeleton, smooth_nm)
    return skeleton


def smooth_skeleton(skeleton, window_nm) -> Skeleton:
    """Moving average of the skeleton curve over ``window_nm`` nanometres of arc length.

    Each chain between two branch or end points is treated as a polyline p(s), s = arc
    length in nm, and every vertex is replaced by the mean of p over [s - h, s + h] with
    h = window_nm / 2.  Near the ends of a chain h shrinks to the distance to the nearer
    end, so branch points and end points do not move and the graph (edges, radii) is
    unchanged.  A straight chain is left exactly as it is; a zig-zag gets shorter.
    Returns a new Skeleton."""
    if not window_nm > 0:
        raise ValueError(f"window_nm must be positive, got {window_nm}")
    v = np.asarray(skeleton.vertices_xyz_nm, dtype=np.float64)
    out = v.copy()
    for chain in skeleton.polylines():
        if len(chain) < 3:
            continue
        p = v[chain]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(step)])                 # arc length at each vertex
        # integral of p from 0 to s[n] (trapezoid rule is exact for a polyline)
        whole = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (p[:-1] + p[1:]) * step[:, None], axis=0)])

        def integral(t):
            """Integral of p from arc length 0 to t, for an array of t."""
            n = np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(step) - 1)   # segment of t
            u = t - s[n]                                                              # nm into it
            slope = (p[n + 1] - p[n]) / np.maximum(step[n], 1e-12)[:, None]
            return whole[n] + u[:, None] * (p[n] + 0.5 * u[:, None] * slope)

        half = np.minimum(0.5 * float(window_nm), np.minimum(s, s[-1] - s))
        inner = half > 0
        mean = (integral(s[inner] + half[inner]) - integral(s[inner] - half[inner])) / (2.0 * half[inner])[:, None]
        out[chain[inner]] = mean
    return Skeleton(out, skeleton.edges.copy(), skeleton.radii_nm.copy(), skeleton.instance_id)


# ---- instance -> gaussians ----------------------------------------------------------------------
def _nearest_centre(points, centres, chunk=1 << 16) -> np.ndarray:
    """Index of the nearest centre for every point: (N, 3), (K, 3) -> (N,) int64."""
    out = np.empty(len(points), dtype=np.int64)
    c2 = (centres ** 2).sum(axis=1)
    for a in range(0, len(points), chunk):
        # |p - c|^2 = |p|^2 - 2 p.c + |c|^2 ; |p|^2 does not change which c is nearest
        out[a:a + chunk] = (c2 - 2.0 * (points[a:a + chunk] @ centres.T)).argmin(axis=1)
    return out


def _kmeans_parts(points, k, seed, max_points, n_iterations=30) -> np.ndarray:
    """Split ``points`` (N, 3) into at most k compact parts by k-means.  Returns the part
    index of every point, (N,) int64 in 0..K-1 with every part non-empty.

    Distances are plain Euclidean distances between the coordinates given, so the UNIT of
    ``points`` decides what "compact" means: pass nanometres, not indices.
    Centres are fitted on at most ``max_points`` randomly chosen points (k-means++ start,
    then Lloyd iterations), after which every point joins its nearest centre.
    The same seed gives the same parts."""
    rng = np.random.default_rng(seed)
    points = points - points.mean(axis=0)                # better conditioned, same distances
    fit = points if len(points) <= max_points else points[rng.choice(len(points), int(max_points), replace=False)]
    k = min(int(k), len(fit))
    # k-means++: each new centre is a point drawn with probability ~ squared distance to the centres so far
    centres = np.empty((k, 3))
    centres[0] = fit[rng.integers(len(fit))]
    d2 = ((fit - centres[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        cum = np.cumsum(d2)
        centres[j] = fit[min(np.searchsorted(cum, rng.random() * cum[-1], side="right"), len(fit) - 1)]
        d2 = np.minimum(d2, ((fit - centres[j]) ** 2).sum(axis=1))
    # Lloyd: assign to the nearest centre, move each centre to the mean of its points
    for _ in range(n_iterations):
        part = _nearest_centre(fit, centres)
        n = np.bincount(part, minlength=k)
        mean = np.stack([np.bincount(part, weights=fit[:, a], minlength=k) for a in range(3)], axis=1)
        moved = np.where(n[:, None] > 0, mean / np.maximum(n, 1)[:, None], centres)
        if np.array_equal(moved, centres):
            break
        centres = moved
    part = _nearest_centre(points, centres)
    used = np.flatnonzero(np.bincount(part, minlength=k))            # drop empty parts
    return np.searchsorted(used, part)


def _fit_mixture(mask, grid, k, seed, max_points, part=None, instance_id=0):
    """Shared by all Gaussian functions.  Returns ``(Gaussians, part)``.

    Points are the voxel centres in the nanometres of ``grid``; ``part`` (the part index
    of every voxel, in ``np.argwhere(mask)`` order) is found by k-means on those points
    unless it is given.  Per part: weight = share of the voxels, mean = mean of the
    centres, covariance = covariance of the centres + diag(dx^2, dy^2, dz^2) / 12 (the
    spread of the tissue inside one voxel: a voxel is a box, not a point)."""
    mask = _check_mask(mask, grid)
    if int(k) < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    sub, start = _crop_to_mask(mask)
    points = grid.voxel_centers_nm(np.argwhere(sub) + start)         # (N, 3) (x, y, z) nm
    if part is None:
        if int(max_points) < 1:
            raise ValueError(f"max_points must be at least 1, got {max_points}")
        part = _kmeans_parts(points, k, seed, max_points)
    n_parts = int(part.max()) + 1
    n = np.bincount(part, minlength=n_parts).astype(np.float64)
    origin = points.mean(axis=0)                                     # sums of small numbers are more exact
    q = points - origin
    mean = np.stack([np.bincount(part, weights=q[:, a], minlength=n_parts) for a in range(3)], axis=1) / n[:, None]
    cov = np.empty((n_parts, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            second = np.bincount(part, weights=q[:, a] * q[:, b], minlength=n_parts) / n
            cov[:, a, b] = cov[:, b, a] = second - mean[:, a] * mean[:, b]
    cov += np.diag(grid.voxel_size_xyz_nm ** 2 / 12.0)               # within-voxel box term
    gaussians = Gaussians(
        means_xyz_nm=mean + origin,
        covariances_nm2=cov,
        weights=n / n.sum(),
        colors_rgb=PALETTE_RGB[np.arange(n_parts) % len(PALETTE_RGB)],
        instance_id=int(instance_id),
    )
    return gaussians, part


def instance_to_gaussians(mask, grid, k=64, seed=0, max_points=60000, *, instance_id=0) -> Gaussians:
    """The instance as a mixture of at most k 3D Gaussians.

    The voxel CENTRES IN PHYSICAL NANOMETRES are split into k compact parts by k-means
    (centres fitted on at most ``max_points`` voxels; the same seed gives the same
    result).  Each part becomes one Gaussian:

        weight      share of the instance's voxels in the part (weights sum to 1)
        mean        mean of the part's voxel centres, (x, y, z) nm
        covariance  covariance of the part's voxel centres
                    + diag(dx^2, dy^2, dz^2) / 12, nm^2
        colour      ``PALETTE_RGB[part % 10]``

    The second covariance term is the variance of a uniform box of one voxel.  With it a
    solid box of a * b * c voxels has exactly diag((a*dx)^2, (b*dy)^2, (c*dz)^2) / 12,
    and a part that lies in a single section still has a thickness."""
    return _fit_mixture(mask, grid, k, seed, max_points, instance_id=instance_id)[0]


def single_gaussian(mask, grid, *, instance_id=0) -> Gaussians:
    """The whole instance as ONE Gaussian: mean and covariance of all its voxel centres
    (plus the within-voxel term diag(dx^2, dy^2, dz^2) / 12), weight 1.

    For a solid ellipsoid with semi-axes a, b, c the covariance is
    R diag(a^2, b^2, c^2) / 5 R^T, so the semi-axes are sqrt(5) standard deviations."""
    mask = _check_mask(mask, grid)
    one_part = np.zeros(int(np.count_nonzero(mask)), dtype=np.int64)       # every voxel in part 0
    return _fit_mixture(mask, grid, k=1, seed=0, max_points=0, part=one_part, instance_id=instance_id)[0]


def voxel_gaussian(grid, kji, scale=1.0) -> Gaussians:
    """"A voxel is a small anisotropic Gaussian": one Gaussian per voxel index [k, j, i].

    mean        the voxel centre, (x, y, z) nm
    covariance  scale * diag(dx^2, dy^2, dz^2) / 12, nm^2
    weight      1 / number of voxels given

    scale = 1 (default) is the Gaussian with the same variance as the uniform voxel box:
    sigma = edge / sqrt(12) = 0.29 edge.  scale = 3 gives sigma = edge / 2, for which the
    1-sigma ellipsoid touches the six faces of the voxel.  All Gaussians of a grid have the
    same shape: elongated dz / dx along z."""
    if not isinstance(grid, VoxelGrid):
        raise TypeError(f"grid must be a VoxelGrid, got {type(grid).__name__}")
    kji = np.asarray(kji)
    if not np.issubdtype(kji.dtype, np.integer) or kji.shape[-1:] != (3,) or kji.ndim > 2:
        raise ValueError(f"kji must be integer array indices of shape (3,) or (N, 3), got {kji.dtype} {kji.shape}")
    kji = kji.reshape(-1, 3)
    if len(kji) == 0 or (kji < 0).any() or (kji >= np.array(grid.shape_zyx)).any():
        raise ValueError(f"kji must name at least one voxel inside the array of shape {grid.shape_zyx}")
    if not scale > 0:
        raise ValueError(f"scale must be positive, got {scale}")
    cov = float(scale) * np.diag(grid.voxel_size_xyz_nm ** 2 / 12.0)
    return Gaussians(
        means_xyz_nm=grid.voxel_centers_nm(kji),
        covariances_nm2=np.repeat(cov[None], len(kji), axis=0),
        weights=np.full(len(kji), 1.0 / len(kji)),
        colors_rgb=PALETTE_RGB[np.arange(len(kji)) % len(PALETTE_RGB)],
    )


def gaussians_wrong_ways(mask, grid, k=64, seed=0, max_points=60000) -> dict:
    """The correct mixture and four instructive wrong or half-wrong ones.

    A program that ignores anisotropy works with ``grid.as_cubes()``: it believes every
    voxel is a cube of edge g = min(dx, dy).  Its numbers are index-space numbers times g
    (lengths) or g^2 (covariances).  Two things can then go wrong, and the variants
    separate them:

        clustering   k-means in cube space finds other parts than k-means in nanometres,
                     because distances along z are counted dz / g times too short.
        covariance   a covariance S_cube measured in cube space must be brought to
                     physical space as  D S_cube D^T  with D = diag(dx, dy, dz) / g.
                     Leaving it as it is ("index covariance times g^2") is the classic
                     mistake: the Gaussian keeps the shape it has in index space.

    Returned dict (every value a ``Gaussians``; all MEANS are at their true physical
    positions, so the variants can be drawn on top of the true data):

        "correct"                    ``instance_to_gaussians(mask, grid, k, seed)``
        "index_fit_scaled"           parts AND covariances from ``grid.as_cubes()``,
                                     covariances left as they are: both mistakes
        "index_fit_transformed"      parts from ``grid.as_cubes()``, covariances properly
                                     transformed (D S D^T): only the clustering is wrong
        "same_clusters_scaled"       the parts of "correct", covariances measured in cube
                                     space and left as they are: only the covariance is wrong
        "same_clusters_transformed"  the parts of "correct", covariances measured in cube
                                     space and properly transformed: equal to "correct"

    For an isotropic grid all five are the same."""
    mask = _check_mask(mask, grid)
    cubes = grid.as_cubes()
    correct, part = _fit_mixture(mask, grid, k, seed, max_points)
    index_fit, _ = _fit_mixture(mask, cubes, k, seed, max_points)             # the careless program, start to end
    same_clusters, _ = _fit_mixture(mask, cubes, k, seed, max_points, part=part)
    return {
        "correct": correct,
        "index_fit_scaled": _cubes_to_physical(index_fit, grid, transform_covariance=False),
        "index_fit_transformed": _cubes_to_physical(index_fit, grid, transform_covariance=True),
        "same_clusters_scaled": _cubes_to_physical(same_clusters, grid, transform_covariance=False),
        "same_clusters_transformed": _cubes_to_physical(same_clusters, grid, transform_covariance=True),
    }


def _cubes_to_physical(gaussians, grid, transform_covariance) -> Gaussians:
    """Move Gaussians fitted with ``grid.as_cubes()`` to the true grid.

    d = (dx, dy, dz) / g is the per-axis stretch from cube space to physical space
    ((1, 1, 5) for SNEMI3D).  Means are always moved: mean * d.  Covariances are moved
    properly, cov[a, b] * d[a] * d[b]  (= D S D^T), only if ``transform_covariance``."""
    d = 1.0 / grid.space_scale("index")
    cov = gaussians.covariances_nm2
    if transform_covariance:
        cov = cov * d[:, None] * d[None, :]
    return Gaussians(gaussians.means_xyz_nm * d, cov.copy(), gaussians.weights.copy(),
                     gaussians.colors_rgb.copy(), gaussians.instance_id)


# ---- numbers for the anisotropy exhibit -----------------------------------------------------------
MEASURED = (
    "volume_nm3",
    "mesh_area_nm2",
    "mesh_volume_nm3",
    "cable_length_nm",
    "cable_length_smooth_nm",
    "mean_radius_nm",
    "n_branch_points",
)


def mean_radius_nm(skeleton) -> float:
    """Mean radius ALONG THE CABLE in nm: every edge counts with its length (nm) and with
    the mean radius of its two nodes.

    Why not the plain mean over the nodes: the nodes are one voxel apart, so their spacing
    depends on the direction.  On 30 x 6 x 6 nm voxels a stretch inside the section plane
    has five times more nodes per nanometre than one that crosses the sections, and a
    mean over nodes would count it five times."""
    v, e, r = skeleton.vertices_xyz_nm, skeleton.edges, skeleton.radii_nm
    length = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)         # nm, one per edge
    if length.sum() == 0:                                            # no edge to weight with
        return float(np.mean(r))
    return float((length * 0.5 * (r[e[:, 0]] + r[e[:, 1]])).sum() / length.sum())


def measure(mask, grid, smooth_nm=None) -> dict:
    """Numbers one would report about an instance, computed with the voxel size of ``grid``.

        n_voxels                voxels in the mask (does not depend on the grid)
        volume_nm3              n_voxels * dx * dy * dz
        mesh_area_nm2           area of the unsmoothed, unsimplified surface mesh.  On thick
                                sections that surface is a staircase and reads too much:
                                +21 % for a sphere of 400 nm radius on 30 x 6 x 6 nm voxels
        mesh_volume_nm3         volume enclosed by that mesh
        cable_length_nm         summed edge length of the raw skeleton (chain of voxel centres)
        cable_length_smooth_nm  the same after ``smooth_skeleton`` with window ``smooth_nm``
        smooth_nm               that window; default 4 x the largest voxel edge of ``grid``
        mean_radius_nm          ``mean_radius_nm(skeleton)``: mean radius along the cable
        n_branch_points         skeleton nodes with more than two neighbours
        n_end_points            skeleton nodes with one neighbour

    Pass ``grid.as_cubes()`` to get what a program that ignores anisotropy would report.

    The skeleton is made with dust_nm3=0: every piece of the mask is measured, however
    small.  A size threshold in nm^3 would depend on the voxel size it is computed with
    (an object of 2e6 nm^3 on 30 x 6 x 6 nm voxels is only 4e5 nm^3 under
    ``grid.as_cubes()``) and would drop an object from one column of ``compare_anisotropy``
    and keep it in the other."""
    mask = _check_mask(mask, grid)
    if smooth_nm is None:
        smooth_nm = 4.0 * float(grid.voxel_size_xyz_nm.max())
    n_voxels = int(np.count_nonzero(mask))
    mesh = instance_to_mesh(mask, grid)
    skeleton = instance_to_skeleton(mask, grid, dust_nm3=0.0)
    return {
        "n_voxels": n_voxels,
        "volume_nm3": n_voxels * float(np.prod(grid.voxel_size_xyz_nm)),
        "mesh_area_nm2": mesh.area_nm2(),
        "mesh_volume_nm3": mesh.volume_nm3(),
        "cable_length_nm": skeleton.cable_length_nm(),
        "cable_length_smooth_nm": smooth_skeleton(skeleton, smooth_nm).cable_length_nm(),
        "smooth_nm": float(smooth_nm),
        "mean_radius_nm": mean_radius_nm(skeleton),
        "n_branch_points": skeleton.n_branch_points(),
        "n_end_points": skeleton.n_end_points(),
    }


def compare_anisotropy(mask, grid, reference=None, smooth_nm=None) -> list:
    """The anisotropy exhibit as a table: one row (a plain dict) per quantity of ``MEASURED``.

        "quantity"           name, as in ``measure``
        "correct"            measure(mask, grid)
        "as_cubes"           measure(mask, grid.as_cubes()): anisotropy ignored
        "as_cubes_percent"   100 * (as_cubes - correct) / correct

    reference: the same object on a better (isotropic) grid, either a ``(mask, grid)``
    pair or the dict ``measure`` returned for it.  Then each row also has

        "reference"                  the value on the reference grid
        "correct_percent_of_ref"     100 * (correct - reference) / reference
        "as_cubes_percent_of_ref"    100 * (as_cubes - reference) / reference

    All three measurements use the same smoothing window ``smooth_nm`` (default: 4 x the
    largest voxel edge of ``grid``), so only the voxel size differs between the columns."""
    mask = _check_mask(mask, grid)
    if smooth_nm is None:
        smooth_nm = 4.0 * float(grid.voxel_size_xyz_nm.max())
    correct = measure(mask, grid, smooth_nm)
    as_cubes = measure(mask, grid.as_cubes(), smooth_nm)
    if reference is not None and not isinstance(reference, dict):
        reference = measure(reference[0], reference[1], smooth_nm)

    def percent(value, base):
        """Change of value against base in per cent; 0 if they are equal, inf if only base is 0."""
        if value == base:
            return 0.0
        return 100.0 * (value - base) / base if base else float("inf")

    rows = []
    for quantity in MEASURED:
        row = {
            "quantity": quantity,
            "correct": correct[quantity],
            "as_cubes": as_cubes[quantity],
            "as_cubes_percent": percent(as_cubes[quantity], correct[quantity]),
        }
        if reference is not None:
            row["reference"] = reference[quantity]
            row["correct_percent_of_ref"] = percent(correct[quantity], reference[quantity])
            row["as_cubes_percent_of_ref"] = percent(as_cubes[quantity], reference[quantity])
        rows.append(row)
    return rows
