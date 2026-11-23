"""Tests of ncntm.viz.convert: phantoms with known answers, then the real samples.

    python -m pytest tests/test_convert.py        or        python tests/test_convert.py

The box, rod, L, tilted-ellipsoid, oblique-rod and plate phantoms use the voxel size
(dz, dy, dx) = (30, 8, 6) nm: three different numbers, so a swapped axis or a swapped voxel
size changes the answer and a test fails.  The sphere and the ellipsoid in the xz plane use
the SNEMI3D voxel size (30, 6, 6) nm, because their numbers are quoted for that data.
"""
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ncntm.viz import convert as C  # noqa: E402
from ncntm.viz import samples as S  # noqa: E402
from ncntm.viz.grid import VoxelGrid  # noqa: E402
from ncntm.viz.types import Bricks, Gaussians, Mesh, Skeleton  # noqa: E402

VOXEL_ZYX = (30.0, 8.0, 6.0)
DX, DY, DZ = 6.0, 8.0, 30.0


# ---- helpers -------------------------------------------------------------------------------
def raises(error, fn, *args, **kwargs):
    """True if fn(*args, **kwargs) raises ``error`` (so the tests also run without pytest)."""
    try:
        fn(*args, **kwargs)
    except error:
        return True
    return False


@lru_cache(maxsize=None)
def sample(name):
    return S.load_sample(name)


def box_phantom():
    """A solid box of nz * ny * nx = 3 * 10 * 20 voxels: x [180, 300], y [80, 160], z [150, 240] nm."""
    grid = VoxelGrid((20, 40, 60), VOXEL_ZYX)
    mask = np.zeros(grid.shape_zyx, dtype=bool)
    mask[5:8, 10:20, 30:50] = True
    return mask, grid


BOX_LO, BOX_HI = np.array([180.0, 80.0, 150.0]), np.array([300.0, 160.0, 240.0])


def bevelled_box_volume_nm3(nx, ny, nz, dx, dy, dz):
    """Marching cubes cuts the 12 edges and 8 corners of a box of voxels at 45 degrees (index space)."""
    return dx * dy * dz * (nx * ny * nz - (nx + ny + nz - 3) / 2.0 - 5.0 / 6.0)


def bevelled_box_area_nm2(nx, ny, nz, dx, dy, dz):
    a, b, c = (nx - 1) * dx, (ny - 1) * dy, (nz - 1) * dz          # flat part of each face
    faces = 2.0 * (a * b + b * c + c * a)
    bevels = 4.0 * (a * np.hypot(dy, dz) / 2 + b * np.hypot(dx, dz) / 2 + c * np.hypot(dx, dy) / 2)
    corners = 8.0 * 0.5 * np.sqrt((dx * dy) ** 2 + (dy * dz) ** 2 + (dz * dx) ** 2) / 4
    return faces + bevels + corners


def edge_use_counts(faces):
    """How many triangles use each undirected edge; all 2 for a closed manifold surface."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0, return_counts=True)[1]


def rod_phantom(axis):
    """A straight rod along array axis ``axis`` that crosses the whole array (40 voxels).

    Its cross-section is chosen so that it is about square in nanometres, has ONE centre
    voxel line (odd voxel counts, no tie in the distance to the wall) and a different
    voxel count along its two axes, so that swapping any two voxel sizes changes the
    cable length or the radius.  Returns mask, grid, centre index and the voxel counts
    of the cross-section along the two other array axes."""
    cross = {0: (5, 7), 1: (3, 11), 2: (3, 11)}[axis]             # voxels along the other two array axes
    others = [a for a in range(3) if a != axis]
    shape = [0, 0, 0]
    shape[axis] = 40
    box = [slice(None)] * 3
    centre = {}
    for a, n in zip(others, cross):
        shape[a] = n + 6
        box[a] = slice(3, 3 + n)
        centre[a] = 3 + n // 2
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(box)] = True
    return mask, VoxelGrid(shape, VOXEL_ZYX), centre, dict(zip(others, cross))


def rod_expectation(mask, voxel_zyx, axis, cross):
    """(cable length, radius) in nm of the rod, from first principles."""
    cable = (mask.shape[axis] - 1) * voxel_zyx[axis]              # first voxel centre to last voxel centre
    # centre of the middle voxel to the centre of the first voxel outside, along each cross axis
    radius = min((n // 2 + 1) * voxel_zyx[a] for a, n in cross.items())
    return cable, radius


def l_phantom(peg=False):
    """An L in the xy plane: one arm runs to the +x face of the array, one to the +y face.
    With peg=True a short stub along +z sits on the elbow, which makes the object chiral."""
    grid = VoxelGrid((12, 40, 60), VOXEL_ZYX)
    mask = np.zeros(grid.shape_zyx, dtype=bool)
    mask[4:7, 5:16, 10:60] = True            # arm along x: z 120..210, y 40..128, x 60..360 nm
    mask[4:7, 5:40, 10:25] = True            # arm along y: z 120..210, y 40..320, x 60..150 nm
    if peg:
        mask[7:10, 5:16, 10:25] = True       # stub along z: z 210..300 over the elbow
    return mask, grid


def ellipsoid_phantom(shape_zyx=(60, 200, 200), voxel_zyx=(30.0, 6.0, 6.0), semi_nm=(500.0, 200.0, 150.0), tilt_deg=60.0):
    """Voxels whose centre is inside a solid ellipsoid in the middle of a 30 x 6 x 6 nm grid.
    Its long axis lies in the xz plane, ``tilt_deg`` from x towards z.
    Returns mask, grid, centre (nm), rotation (principal directions in columns), true covariance."""
    grid = VoxelGrid(shape_zyx, voxel_zyx)
    t = np.deg2rad(tilt_deg)
    rot = np.array([[np.cos(t), 0.0, -np.sin(t)], [0.0, 1.0, 0.0], [np.sin(t), 0.0, np.cos(t)]])
    semi = np.asarray(semi_nm)
    centre = grid.extent_xyz_nm / 2.0
    kji = np.indices(grid.shape_zyx).reshape(3, -1).T
    along_axes = (grid.voxel_centers_nm(kji) - centre) @ rot
    mask = (((along_axes / semi) ** 2).sum(axis=1) <= 1.0).reshape(grid.shape_zyx)
    return mask, grid, centre, rot, rot @ np.diag(semi ** 2 / 5.0) @ rot.T


def angle_deg(u, v):
    """Angle between two directions (sign does not matter)."""
    c = abs(float(np.dot(u, v))) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(min(1.0, c))))


def long_axis(gaussians, n=0):
    """Direction of the largest standard deviation of component n."""
    rotations, sigmas = gaussians.principal_axes()
    return rotations[n][:, np.argmax(sigmas[n])]


def total_moments(g):
    """Mean and covariance of the whole mixture (law of total covariance)."""
    mean = (g.weights[:, None] * g.means_xyz_nm).sum(axis=0)
    d = g.means_xyz_nm - mean
    cov = (g.weights[:, None, None] * (g.covariances_nm2 + d[:, :, None] * d[:, None, :])).sum(axis=0)
    return mean, cov


def voxels_of(points_xyz_nm, grid):
    """Array indices [k, j, i] of the voxels that contain the points, clipped to the array
    (a point exactly on the upper border of the array belongs to the last voxel)."""
    return np.clip(grid.nm_to_voxel(points_xyz_nm), 0, np.array(grid.shape_zyx) - 1)


def grow_by_one_voxel(mask):
    from scipy import ndimage

    return ndimage.binary_dilation(mask, structure=np.ones((3, 3, 3), dtype=bool))


# ---- mesh ------------------------------------------------------------------------------------
def test_box_mesh_lies_on_the_voxel_faces():
    mask, grid = box_phantom()
    for backend in ("zmesh", "skimage"):
        mesh = C.instance_to_mesh(mask, grid, backend=backend, instance_id=7)
        assert isinstance(mesh, Mesh) and mesh.instance_id == 7
        v, f = mesh.vertices_xyz_nm, mesh.faces
        assert v.dtype == np.float64 and f.dtype == np.int32 and v.shape[1] == 3 and f.shape[1] == 3
        # bounding box = the box of voxels [30:50] x [10:20] x [5:8] in the corner convention
        assert np.allclose(v.min(axis=0), BOX_LO, atol=1e-3), (backend, v.min(axis=0))
        assert np.allclose(v.max(axis=0), BOX_HI, atol=1e-3), (backend, v.max(axis=0))
        # every vertex sits on a voxel face: one coordinate is a whole number of voxels,
        # the other two are voxel centres (whole number + 0.5)
        index = v / grid.voxel_size_xyz_nm
        on_face = np.isclose(index, np.round(index), atol=1e-4)
        at_centre = np.isclose(index % 1.0, 0.5, atol=1e-4)
        assert (on_face.sum(axis=1) == 1).all() and (at_centre.sum(axis=1) == 2).all(), backend
        # closed, one piece, no hole: every edge joins two triangles, Euler characteristic 2
        assert (edge_use_counts(f) == 2).all(), backend
        assert len(v) - len(mesh.edges()) + len(f) == 2, backend


def test_box_mesh_volume_and_area_are_those_of_the_bevelled_box():
    mask, grid = box_phantom()
    volume = bevelled_box_volume_nm3(20, 10, 3, DX, DY, DZ)
    area = bevelled_box_area_nm2(20, 10, 3, DX, DY, DZ)
    assert volume == 841200.0                                      # 584.17 of the 600 voxel volumes
    for backend in ("zmesh", "skimage"):
        mesh = C.instance_to_mesh(mask, grid, backend=backend)
        assert mesh.volume_nm3() > 0                               # outward winding
        assert np.isclose(mesh.volume_nm3(), volume, rtol=1e-6), (backend, mesh.volume_nm3())
        assert np.isclose(mesh.area_nm2(), area, rtol=1e-6), (backend, mesh.area_nm2(), area)
        # the triangle normals point away from the centre of the box
        v, f = mesh.vertices_xyz_nm, mesh.faces
        normal = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        outward = v[f].mean(axis=1) - (BOX_LO + BOX_HI) / 2
        assert (np.einsum("ij,ij->i", normal, outward) > 0).all(), backend


def test_mesh_is_closed_where_the_instance_touches_the_array_border():
    grid = VoxelGrid((4, 5, 6), VOXEL_ZYX)
    mask = np.ones(grid.shape_zyx, dtype=bool)                     # fills the whole array
    for backend in ("zmesh", "skimage"):
        mesh = C.instance_to_mesh(mask, grid, backend=backend)
        v = mesh.vertices_xyz_nm
        assert np.allclose(v.min(axis=0), 0.0, atol=1e-3)
        assert np.allclose(v.max(axis=0), grid.extent_xyz_nm, atol=1e-3)       # [0, N * d]
        assert np.allclose(v.max(axis=0), [36.0, 40.0, 120.0], atol=1e-3)
        assert (edge_use_counts(mesh.faces) == 2).all()
        assert np.isclose(mesh.volume_nm3(), bevelled_box_volume_nm3(6, 5, 4, DX, DY, DZ), rtol=1e-6)


def test_a_wrong_voxel_size_moves_the_mesh():
    mask, grid = box_phantom()
    # ignoring anisotropy: the same triangles, every vertex scaled by g / (dx, dy, dz) = (1, 6/8, 6/30)
    assert np.allclose(grid.space_scale("index"), [1.0, 0.75, 0.2])
    right = C.instance_to_mesh(mask, grid)
    cubes = C.instance_to_mesh(mask, grid.as_cubes())
    assert np.array_equal(right.faces, cubes.faces)
    assert np.allclose(cubes.vertices_xyz_nm, right.vertices_xyz_nm * grid.space_scale("index"), atol=1e-3)
    assert np.allclose(cubes.vertices_xyz_nm.max(axis=0), [50 * 6.0, 20 * 6.0, 8 * 6.0], atol=1e-3)
    assert np.isclose(cubes.volume_nm3() / right.volume_nm3(), 6.0 ** 3 / (6.0 * 8.0 * 30.0), rtol=1e-6)
    # voxel size given in the wrong order: the bounding box is not the box of the phantom
    for wrong in ((6.0, 8.0, 30.0), (8.0, 6.0, 30.0), (30.0, 6.0, 8.0)):
        v = C.instance_to_mesh(mask, VoxelGrid(grid.shape_zyx, wrong)).vertices_xyz_nm
        assert not (np.allclose(v.min(axis=0), BOX_LO, atol=1e-3) and np.allclose(v.max(axis=0), BOX_HI, atol=1e-3)), wrong
    # array transposed (x and z swapped) with the right voxel size: also not the phantom
    v = C.instance_to_mesh(mask.T.copy(), VoxelGrid(mask.T.shape, VOXEL_ZYX)).vertices_xyz_nm
    assert not np.allclose(v.max(axis=0), BOX_HI, atol=1e-3)


def test_sphere_mesh_staircase_smoothing_and_simplification():
    grid = VoxelGrid((40, 160, 160), (30.0, 6.0, 6.0))
    centre = grid.extent_xyz_nm / 2.0
    kji = np.indices(grid.shape_zyx).reshape(3, -1).T
    mask = (np.linalg.norm(grid.voxel_centers_nm(kji) - centre, axis=1) <= 400.0).reshape(grid.shape_zyx)
    true_volume, true_area = 4.0 / 3.0 * np.pi * 400.0 ** 3, 4.0 * np.pi * 400.0 ** 2

    raw = C.instance_to_mesh(mask, grid)
    assert abs(raw.volume_nm3() / true_volume - 1) < 0.01
    assert 0.19 < raw.area_nm2() / true_area - 1 < 0.23           # the staircase of 30 nm sections: +21 %
    assert np.allclose(raw.vertices_xyz_nm.mean(axis=0), centre, atol=1.0)
    same = C.instance_to_mesh(mask, grid, backend="skimage")
    assert np.isclose(same.volume_nm3(), raw.volume_nm3(), rtol=1e-6)
    assert np.isclose(same.area_nm2(), raw.area_nm2(), rtol=1e-6)

    # a blur of half a section (15 nm, round in physical space) removes the staircase
    smooth = C.instance_to_mesh(mask, grid, smooth_sigma_nm=15.0, backend="skimage")
    assert abs(smooth.area_nm2() / true_area - 1) < 0.01          # measured +0.5 %
    assert abs(smooth.volume_nm3() / true_volume - 1) < 0.015
    # a blur of a tenth of a section (3 nm) is too little: 13 % of the 21 % are still there
    too_little = C.instance_to_mesh(mask, grid, smooth_sigma_nm=3.0, backend="skimage")
    assert 0.11 < too_little.area_nm2() / true_area - 1 < 0.15
    assert np.allclose(smooth.vertices_xyz_nm.mean(axis=0), centre, atol=1.0)
    assert (edge_use_counts(smooth.faces) == 2).all()

    # simplification: far fewer triangles, still closed, still outward, same place, steps remain
    for kwargs in (dict(simplify_factor=20), dict(simplify_factor=20, max_error_nm=6.0),
                   dict(simplify_factor=20, backend="skimage"),
                   dict(simplify_factor=20, backend="skimage", smooth_sigma_nm=15.0)):
        small = C.instance_to_mesh(mask, grid, **kwargs)
        assert len(small.faces) < len(raw.faces) / 8, kwargs
        assert small.volume_nm3() > 0 and abs(small.volume_nm3() / true_volume - 1) < 0.02, kwargs
        assert (edge_use_counts(small.faces) == 2).all(), kwargs
        lo, hi = small.vertices_xyz_nm.min(axis=0), small.vertices_xyz_nm.max(axis=0)
        assert np.allclose(lo, centre - 400.0, atol=31.0) and np.allclose(hi, centre + 400.0, atol=31.0), kwargs
    stepped = C.instance_to_mesh(mask, grid, simplify_factor=20)
    assert stepped.area_nm2() / true_area - 1 > 0.10              # simplification keeps the staircase
    # a tighter error bound stops the simplification earlier; the default bound is the largest voxel edge
    assert len(C.instance_to_mesh(mask, grid, simplify_factor=20, max_error_nm=6.0).faces) > len(stepped.faces)
    explicit = C.instance_to_mesh(mask, grid, simplify_factor=20, max_error_nm=30.0)
    assert np.array_equal(explicit.faces, stepped.faces) and np.array_equal(explicit.vertices_xyz_nm, stepped.vertices_xyz_nm)
    assert np.array_equal(C.instance_to_mesh(mask, grid, simplify_factor=1).faces, raw.faces)       # factor 1 = no simplification
    # the factor is a lower bound: on a box the flat faces merge at no cost
    box, box_grid = box_phantom()
    full = C.instance_to_mesh(box, box_grid)
    for backend in ("zmesh", "skimage"):
        halved = C.instance_to_mesh(box, box_grid, simplify_factor=2, backend=backend)
        assert len(halved.faces) <= len(full.faces) / 2, backend
        assert abs(halved.volume_nm3() / full.volume_nm3() - 1) < 0.02, backend
    assert len(C.instance_to_mesh(box, box_grid, simplify_factor=2).faces) < len(full.faces) / 5      # zmesh: 176 of 1156


def test_mesh_argument_errors():
    mask, grid = box_phantom()
    assert raises(ValueError, C.instance_to_mesh, mask, grid, backend="vtk")
    assert raises(ValueError, C.instance_to_mesh, mask, grid, smooth_sigma_nm=15.0)                 # zmesh cannot blur
    assert raises(ValueError, C.instance_to_mesh, mask, grid, smooth_sigma_nm=-1.0, backend="skimage")
    assert raises(ValueError, C.instance_to_mesh, mask, grid, max_error_nm=5.0, backend="skimage")
    assert raises(ValueError, C.instance_to_mesh, mask, grid, simplify_factor=0.5)
    one = np.zeros(grid.shape_zyx, dtype=bool)
    one[3, 3, 3] = True                                            # a single voxel blurs away
    assert raises(ValueError, C.instance_to_mesh, one, grid, smooth_sigma_nm=200.0, backend="skimage")
    # ... but meshes fine without blur: an octahedron around the voxel centre, 1/6 of the voxel volume
    octahedron = C.instance_to_mesh(one, grid)
    assert len(octahedron.faces) == 8 and np.isclose(octahedron.volume_nm3(), DX * DY * DZ / 6.0)
    assert np.allclose(octahedron.vertices_xyz_nm.mean(axis=0), grid.voxel_centers_nm([3, 3, 3]))


# ---- skeleton --------------------------------------------------------------------------------
def test_teasar_defaults_are_the_ones_documented():
    assert C.TEASAR_DEFAULTS["scale"] == 1.5 and C.TEASAR_DEFAULTS["const"] == 300.0


def test_rod_skeleton_runs_along_the_voxel_centre_line():
    for axis in range(3):
        mask, grid, centre, cross = rod_phantom(axis)
        skeleton = C.instance_to_skeleton(mask, grid, instance_id=3)
        assert isinstance(skeleton, Skeleton) and skeleton.instance_id == 3
        v = skeleton.vertices_xyz_nm
        assert v.dtype == np.float64 and skeleton.edges.dtype == np.int32 and skeleton.radii_nm.shape == (len(v),)
        # one unbranched chain through all 40 voxels of the centre line
        assert len(v) == 40 and len(skeleton.edges) == 39
        assert skeleton.n_end_points() == 2 and skeleton.n_branch_points() == 0
        kji = grid.nm_to_voxel(v)
        for a, c in centre.items():
            assert (kji[:, a] == c).all(), (axis, a)
        assert sorted(kji[:, axis]) == list(range(40))
        # vertices are voxel centres of the corner convention: (index + 0.5) * voxel size
        assert np.allclose(v, grid.voxel_centers_nm(kji), atol=1e-4)
        # cable length and radius, from first principles
        cable, radius = rod_expectation(mask, VOXEL_ZYX, axis, cross)
        assert np.isclose(skeleton.cable_length_nm(), cable, rtol=1e-6), (axis, skeleton.cable_length_nm(), cable)
        assert np.allclose(skeleton.radii_nm, radius, rtol=1e-6), (axis, np.unique(skeleton.radii_nm), radius)
    # the numbers of the docstring: 7 voxels of 6 nm wide -> 24 nm (half width 21 nm)
    mask, grid, _, cross = rod_phantom(0)
    assert cross == {1: 5, 2: 7} and rod_expectation(mask, VOXEL_ZYX, 0, cross) == (39 * 30.0, 24.0)
    assert mask.any(axis=(0, 1)).sum() * DX / 2 == 21.0


def test_rod_skeleton_notices_a_wrong_voxel_size_order():
    """The same rods with the voxel size given in a wrong order: cable or radius is off."""
    for axis in range(3):
        mask, grid, _, cross = rod_phantom(axis)
        cable, radius = rod_expectation(mask, VOXEL_ZYX, axis, cross)
        for wrong in ((6.0, 8.0, 30.0), (8.0, 6.0, 30.0), (6.0, 30.0, 8.0), (30.0, 6.0, 8.0), (8.0, 30.0, 6.0)):
            skeleton = C.instance_to_skeleton(mask, VoxelGrid(mask.shape, wrong), dust_nm3=0.0)
            cable_ok = np.isclose(skeleton.cable_length_nm(), cable, rtol=1e-6)
            radius_ok = np.allclose(skeleton.radii_nm, radius, rtol=1e-6)
            assert not (cable_ok and radius_ok), (axis, wrong)


def test_l_shape_skeleton_ends_where_the_array_says():
    mask, grid = l_phantom()
    skeleton = C.instance_to_skeleton(mask, grid)
    v = skeleton.vertices_xyz_nm
    assert skeleton.n_end_points() == 2 and skeleton.n_branch_points() == 0
    # every vertex is the centre of a voxel of the instance
    kji = grid.nm_to_voxel(v)
    assert mask[tuple(kji.T)].all()
    assert np.allclose(v, grid.voxel_centers_nm(kji), atol=1e-4)
    # the two ends are the centres of the two cut faces: arm x leaves at i = 59, arm y at j = 39
    ends = v[skeleton.degrees() == 1]
    end_x, end_y = ends[np.argmax(ends[:, 0])], ends[np.argmax(ends[:, 1])]
    assert np.allclose(end_x, grid.voxel_centers_nm([5, 10, 59]), atol=1e-4) and np.allclose(end_x, [357.0, 84.0, 165.0])
    assert np.allclose(end_y, grid.voxel_centers_nm([5, 39, 17]), atol=1e-4) and np.allclose(end_y, [105.0, 316.0, 165.0])
    # handedness: going from the y end around the elbow to the x end is a right turn seen from +z,
    # exactly as in the array ((x arm) x (y arm) points to +z)
    elbow = v[np.argmin(v[:, 0] + v[:, 1])]
    assert np.cross(end_x - elbow, end_y - elbow)[2] > 0
    # cable: at least the city-block path between the ends minus the cut corner, at most the city-block path
    city_block = (end_x[0] - end_y[0]) + (end_y[1] - end_x[1])
    assert np.hypot(end_x[0] - end_y[0], end_y[1] - end_x[1]) < skeleton.cable_length_nm() <= city_block + 1e-6
    # a mirrored array gives a mirrored skeleton (the test can tell them apart)
    mirrored = C.instance_to_skeleton(mask[:, :, ::-1].copy(), grid).vertices_xyz_nm
    assert not np.allclose(mirrored[:, 0].max(), v[:, 0].max())


def test_chiral_object_is_not_mirrored_in_any_representation():
    mask, grid = l_phantom(peg=True)
    elbow = grid.index_to_nm([17.5, 10.5, 5.5])                    # centre of the elbow, (x, y, z) nm
    assert np.array_equal(elbow, [105.0, 84.0, 165.0])

    def handedness(points, elbow):
        """Triple product of the three arm directions seen in a point cloud.  The arms are the
        points far from the elbow along x, along y and along z.  Positive if (x arm, y arm,
        z peg) is a right-handed set, as it is in the array; any mirror image gives a negative value."""
        arm_x = points[np.abs(points[:, 0] - elbow[0]) > 60.0].mean(axis=0) - elbow
        arm_y = points[np.abs(points[:, 1] - elbow[1]) > 100.0].mean(axis=0) - elbow
        peg_z = points[np.abs(points[:, 2] - elbow[2]) > 50.0].mean(axis=0) - elbow
        return float(np.dot(np.cross(arm_x, arm_y), peg_z))

    truth = grid.voxel_centers_nm(np.argwhere(mask))
    assert handedness(truth, elbow) > 0
    mesh = C.instance_to_mesh(mask, grid)
    bricks = C.instance_to_bricks(mask, grid)
    gaussians = C.instance_to_gaussians(mask, grid, k=24)
    clouds = (("mesh", mesh.vertices_xyz_nm), ("bricks", bricks.centers_xyz_nm), ("gaussians", gaussians.means_xyz_nm))
    for name, points in clouds:
        assert handedness(points, elbow) > 0, name
        # each arm is where the array has it: x arm at low y and z, y arm at low x and z, peg at low x and y
        far_x, far_y, high_z = points[points[:, 0] > 200.0], points[points[:, 1] > 200.0], points[points[:, 2] > 215.0]
        assert len(far_x) and len(far_y) and len(high_z), name
        assert far_x[:, 1].max() <= 128.0 + 1e-3 and far_x[:, 2].max() <= 210.0 + 1e-3, name
        assert far_y[:, 0].max() <= 150.0 + 1e-3 and far_y[:, 2].max() <= 210.0 + 1e-3, name
        assert high_z[:, 0].max() <= 150.0 + 1e-3 and high_z[:, 1].max() <= 128.0 + 1e-3, name
    # the mesh encloses the voxels (bevels take a few per cent) and winds outward
    voxel_volume = mask.sum() * DX * DY * DZ
    assert 0.9 * voxel_volume < mesh.volume_nm3() < voxel_volume
    assert np.allclose(mesh.vertices_xyz_nm.min(axis=0), [60.0, 40.0, 120.0], atol=1e-3)
    assert np.allclose(mesh.vertices_xyz_nm.max(axis=0), [360.0, 320.0, 300.0], atol=1e-3)
    # the check can tell: an array mirrored along any one axis gives the other handedness
    extent = grid.extent_xyz_nm
    for axis, flipped in ((0, mask[:, :, ::-1]), (1, mask[:, ::-1, :]), (2, mask[::-1, :, :])):
        mirrored_elbow = elbow.copy()
        mirrored_elbow[axis] = extent[axis] - elbow[axis]
        mirrored = C.instance_to_mesh(flipped.copy(), grid)
        assert mirrored.volume_nm3() > 0                           # still wound outward
        assert handedness(mirrored.vertices_xyz_nm, mirrored_elbow) < 0, axis


def test_smooth_skeleton_shortens_a_zigzag_and_keeps_ends_and_branch_points():
    # a chain that advances 6 nm in x per step and hops between two sections 30 nm apart
    n = 201
    zigzag = np.stack([np.arange(n) * 6.0, np.zeros(n), (np.arange(n) % 2) * 30.0], axis=1)
    chain = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int32)
    raw = Skeleton(zigzag, chain, np.ones(n))
    assert raw.cable_length_nm() > 6000.0                          # 200 steps of 30.6 nm
    smooth = C.smooth_skeleton(raw, 120.0)
    assert 1200.0 <= smooth.cable_length_nm() < 1240.0             # the straight distance is 1200 nm
    assert np.array_equal(smooth.vertices_xyz_nm[[0, -1]], zigzag[[0, -1]])          # ends do not move
    assert np.array_equal(smooth.edges, raw.edges) and np.array_equal(smooth.radii_nm, raw.radii_nm)
    assert np.array_equal(raw.vertices_xyz_nm, zigzag)             # the input is not modified
    # a straight chain is left exactly as it is, also with uneven steps
    straight = Skeleton(np.outer(np.cumsum([0, 3, 8, 30, 30, 6, 6, 50]), [0.6, 0.0, 0.8]), chain[:7], np.ones(8))
    assert np.allclose(C.smooth_skeleton(straight, 40.0).vertices_xyz_nm, straight.vertices_xyz_nm, atol=1e-9)
    # a Y: three zig-zag chains meet in vertex 0, which must stay; so must the three tips
    arms = [np.stack([np.arange(1, 21) * 6.0 * c, np.arange(1, 21) * 6.0 * s, (np.arange(1, 21) % 2) * 30.0], axis=1)
            for c, s in ((1.0, 0.0), (-0.5, 0.866), (-0.5, -0.866))]
    vertices = np.concatenate([np.zeros((1, 3))] + arms)
    edges = []
    for a in range(3):
        first = 1 + 20 * a
        edges += [[0, first]] + [[first + m, first + m + 1] for m in range(19)]
    tree = Skeleton(vertices, np.array(edges, dtype=np.int32), np.ones(61))
    assert tree.n_branch_points() == 1 and tree.n_end_points() == 3
    smooth_tree = C.smooth_skeleton(tree, 120.0)
    fixed = [0, 20, 40, 60]
    assert np.array_equal(smooth_tree.vertices_xyz_nm[fixed], vertices[fixed])
    assert smooth_tree.cable_length_nm() < 0.5 * tree.cable_length_nm()
    assert raises(ValueError, C.smooth_skeleton, tree, 0.0)


def test_skeleton_smooth_nm_argument_is_smooth_skeleton():
    mask, grid = l_phantom()
    raw = C.instance_to_skeleton(mask, grid)
    direct = C.instance_to_skeleton(mask, grid, smooth_nm=120.0)
    assert np.array_equal(direct.vertices_xyz_nm, C.smooth_skeleton(raw, 120.0).vertices_xyz_nm)
    assert direct.cable_length_nm() <= raw.cable_length_nm()
    assert np.array_equal(direct.edges, raw.edges)


def test_array_border_is_not_a_wall_unless_the_instance_fills_the_array():
    # a rod that leaves the array at both ends: the radius is the same in the first and the last section
    mask, grid, _, cross = rod_phantom(0)
    skeleton = C.instance_to_skeleton(mask, grid)
    assert np.allclose(skeleton.radii_nm, 24.0)
    # the same rod as the whole array (no background voxel at all): now the border is a wall
    grid_full = VoxelGrid((40, 5, 7), VOXEL_ZYX)
    full = C.instance_to_skeleton(np.ones(grid_full.shape_zyx, dtype=bool), grid_full, dust_nm3=0.0)
    assert full.radii_nm.min() < 24.0 and full.n_end_points() > 2


def test_two_pieces_give_a_forest():
    # two parallel rods along z in one mask: one tree each
    grid = VoxelGrid((40, 13, 40), VOXEL_ZYX)
    mask = np.zeros(grid.shape_zyx, dtype=bool)
    mask[:, 4:9, 4:11] = True
    mask[:, 4:9, 24:31] = True
    skeleton = C.instance_to_skeleton(mask, grid)
    assert skeleton.n_end_points() == 4 and skeleton.n_branch_points() == 0
    assert len(skeleton.vertices_xyz_nm) == 80 and len(skeleton.edges) == 78
    assert len(skeleton.polylines()) == 2
    assert np.isclose(skeleton.cable_length_nm(), 2 * 39 * 30.0)
    kji = grid.nm_to_voxel(skeleton.vertices_xyz_nm)
    assert mask[tuple(kji.T)].all() and set(kji[:, 2]) == {7, 27} and set(kji[:, 1]) == {6}


def test_skeleton_errors_are_clear():
    mask, grid = box_phantom()
    empty = np.zeros(grid.shape_zyx, dtype=bool)
    assert raises(ValueError, C.instance_to_skeleton, empty, grid)
    one = empty.copy()
    one[3, 3, 3] = True
    assert raises(ValueError, C.instance_to_skeleton, one, grid)                    # below the dust volume
    assert raises(ValueError, C.instance_to_skeleton, one, grid, dust_nm3=0.0)      # a voxel has no centre line
    try:
        C.instance_to_skeleton(one, grid)
    except ValueError as error:
        assert "too small" in str(error) and "1 voxels" in str(error)
    assert raises(TypeError, C.instance_to_skeleton, mask, grid, konst=300)         # misspelt TEASAR parameter
    # the box is 3 x 10 x 20 voxels = 864000 nm^3: skipped as dust by default, skeletonised on request
    assert raises(ValueError, C.instance_to_skeleton, mask, grid)
    assert C.measure(mask, grid)["cable_length_nm"] > 0                            # measure() has no dust threshold
    assert len(C.instance_to_skeleton(mask, grid, dust_nm3=0.0).vertices_xyz_nm) >= 2
    assert len(C.instance_to_skeleton(mask, grid, dust_nm3=0.0, const=50.0).vertices_xyz_nm) >= 2


# ---- bricks ------------------------------------------------------------------------------------
def test_bricks_count_and_positions():
    mask, grid = box_phantom()
    everything = C.instance_to_bricks(mask, grid, surface_only=False, instance_id=4)
    assert isinstance(everything, Bricks) and everything.instance_id == 4
    assert len(everything) == 3 * 10 * 20 == everything.n_voxels_total
    assert np.array_equal(everything.kji, np.argwhere(mask))
    shell = C.instance_to_bricks(mask, grid)
    assert len(shell) == 600 - 1 * 8 * 18 and shell.n_voxels_total == 600          # the inside is 1 x 8 x 18
    for bricks in (everything, shell):
        assert np.issubdtype(bricks.kji.dtype, np.integer) and bricks.centers_xyz_nm.dtype == np.float64
        assert mask[tuple(bricks.kji.T)].all()
        # centres map back to their voxel, and are (index + 0.5) * (dx, dy, dz)
        assert np.array_equal(grid.nm_to_voxel(bricks.centers_xyz_nm), bricks.kji)
        assert np.allclose(bricks.centers_xyz_nm, (bricks.kji[:, ::-1] + 0.5) * [DX, DY, DZ])
        assert np.array_equal(bricks.size_xyz_nm, [DX, DY, DZ])
    assert np.allclose(everything.centers_xyz_nm.min(axis=0), BOX_LO + np.array([DX, DY, DZ]) / 2)
    # the shell is exactly the voxels with an outside face neighbour
    inside = np.zeros_like(mask)
    inside[6:7, 11:19, 31:49] = True
    assert np.array_equal(shell.kji, np.argwhere(mask & ~inside))
    # ignoring anisotropy: same voxels, cubes of 6 nm
    cubes = C.instance_to_bricks(mask, grid.as_cubes())
    assert np.array_equal(cubes.kji, shell.kji) and np.array_equal(cubes.size_xyz_nm, [6.0, 6.0, 6.0])
    assert np.allclose(cubes.centers_xyz_nm, shell.centers_xyz_nm * grid.space_scale("index"))


def test_bricks_on_the_array_border_count_as_surface():
    grid = VoxelGrid((4, 5, 6), VOXEL_ZYX)
    mask = np.ones(grid.shape_zyx, dtype=bool)
    shell = C.instance_to_bricks(mask, grid)
    assert len(shell) == 4 * 5 * 6 - 2 * 3 * 4 and shell.n_voxels_total == 120
    assert np.array_equal(grid.nm_to_voxel(shell.centers_xyz_nm), shell.kji)


def test_max_bricks_keeps_an_evenly_spaced_subset():
    mask, grid = box_phantom()
    shell = C.instance_to_bricks(mask, grid)
    some = C.instance_to_bricks(mask, grid, max_bricks=100)
    again = C.instance_to_bricks(mask, grid, max_bricks=100)
    assert len(some) == 100 and some.n_voxels_total == 600
    assert np.array_equal(some.kji, again.kji)                    # deterministic
    assert len(np.unique(some.kji, axis=0)) == 100
    assert {tuple(r) for r in some.kji} <= {tuple(r) for r in shell.kji}
    assert np.array_equal(some.kji[0], shell.kji[0]) and np.array_equal(some.kji[-1], shell.kji[-1])
    assert len(C.instance_to_bricks(mask, grid, max_bricks=10 ** 6)) == len(shell)   # no effect when under the cap
    assert raises(ValueError, C.instance_to_bricks, mask, grid, max_bricks=0)


# ---- gaussians -----------------------------------------------------------------------------------
def test_single_gaussian_of_a_box_is_exact():
    mask, grid = box_phantom()
    g = C.single_gaussian(mask, grid, instance_id=9)
    assert isinstance(g, Gaussians) and len(g) == 1 and g.instance_id == 9
    assert np.allclose(g.means_xyz_nm[0], (BOX_LO + BOX_HI) / 2, atol=1e-9)
    # a uniform box of edge L has variance L^2 / 12: (20 * 6)^2, (10 * 8)^2, (3 * 30)^2 over 12
    assert np.allclose(g.covariances_nm2[0], np.diag([120.0 ** 2, 80.0 ** 2, 90.0 ** 2]) / 12.0, rtol=1e-9, atol=1e-6)
    assert np.array_equal(g.weights, [1.0])
    assert np.array_equal(g.colors_rgb[0], C.PALETTE_RGB[0])
    # without the within-voxel term the z variance of this 3-section box would be 11 % too small
    centres_only = np.var((np.arange(5, 8) + 0.5) * DZ)
    assert np.isclose(centres_only + DZ ** 2 / 12.0, 90.0 ** 2 / 12.0)
    assert np.isclose(1 - centres_only / (90.0 ** 2 / 12.0), 1.0 / 9.0)


def test_single_gaussian_recovers_an_ellipsoid():
    mask, grid, centre, rot, true_cov = ellipsoid_phantom()
    assert grid.voxel_size_zyx_nm == (30.0, 6.0, 6.0) and mask.sum() > 50000
    g = C.single_gaussian(mask, grid)
    assert np.allclose(g.means_xyz_nm[0], centre, atol=0.5)
    # stated tolerance: every covariance entry within 1 % of the largest variance (measured 0.04 %)
    largest = np.linalg.eigvalsh(true_cov).max()
    assert np.abs(g.covariances_nm2[0] - true_cov).max() < 0.01 * largest
    assert angle_deg(long_axis(g), rot[:, 0]) < 0.5
    # semi-axes of the solid ellipsoid = sqrt(5) standard deviations
    _, sigmas = g.principal_axes()
    assert np.allclose(np.sort(sigmas[0]) * np.sqrt(5.0), [150.0, 200.0, 500.0], rtol=0.01)


def test_ignoring_anisotropy_turns_the_principal_axis():
    mask, grid, centre, rot, true_cov = ellipsoid_phantom()
    wrong = C.single_gaussian(mask, grid.as_cubes())               # the index-space mistake
    # what the mistake must give: the true covariance squashed by (1, 1, 1/5)
    expected_cov = grid.covariance_to_space(true_cov, "index")
    expected_axis = np.linalg.eigh(expected_cov)[1][:, -1]
    assert np.abs(wrong.covariances_nm2[0] - expected_cov).max() < 0.01 * np.linalg.eigvalsh(expected_cov).max()
    assert angle_deg(long_axis(wrong), expected_axis) < 0.5
    # the true long axis rises 60 degrees out of the section plane; the mistake shows 14 degrees
    rise = lambda axis: float(np.degrees(np.arctan2(abs(axis[2]), np.hypot(axis[0], axis[1]))))  # noqa: E731
    assert abs(rise(long_axis(C.single_gaussian(mask, grid))) - 60.0) < 0.5
    assert abs(rise(long_axis(wrong)) - 14.4) < 0.5
    assert angle_deg(long_axis(wrong), rot[:, 0]) > 40.0
    # with one part, the two "scaled" variants are this mistake and the two "transformed" ones are right
    ways = C.gaussians_wrong_ways(mask, grid, k=1)
    assert set(ways) == {"correct", "index_fit_scaled", "index_fit_transformed",
                         "same_clusters_scaled", "same_clusters_transformed"}
    for name, g in ways.items():
        assert np.allclose(g.means_xyz_nm[0], centre, atol=0.5), name          # means are always physical
        if name.endswith("_scaled"):
            assert np.allclose(g.covariances_nm2, wrong.covariances_nm2, rtol=1e-9), name
            assert angle_deg(long_axis(g), rot[:, 0]) > 40.0, name
        else:
            assert np.allclose(g.covariances_nm2, ways["correct"].covariances_nm2, rtol=1e-9), name
            assert angle_deg(long_axis(g), rot[:, 0]) < 0.5, name


def test_mixture_keeps_the_total_moments():
    mask, grid, centre, _, _ = ellipsoid_phantom()
    one = C.single_gaussian(mask, grid)
    for k in (2, 8, 64):
        g = C.instance_to_gaussians(mask, grid, k=k, seed=0, instance_id=5)
        assert len(g) == k and g.instance_id == 5
        assert g.means_xyz_nm.shape == (k, 3) and g.covariances_nm2.shape == (k, 3, 3) and g.weights.shape == (k,)
        assert np.isclose(g.weights.sum(), 1.0) and (g.weights > 0).all()
        assert np.allclose(g.covariances_nm2, np.swapaxes(g.covariances_nm2, 1, 2))
        # positive definite: no variance is smaller than the within-voxel term of the thinnest axis
        assert np.linalg.eigvalsh(g.covariances_nm2).min() >= 6.0 ** 2 / 12.0 - 1e-6
        mean, cov = total_moments(g)
        assert np.allclose(mean, one.means_xyz_nm[0], atol=1e-6)
        assert np.allclose(cov, one.covariances_nm2[0], rtol=1e-8, atol=1e-4)
        assert np.array_equal(g.colors_rgb, C.PALETTE_RGB[np.arange(k) % 10])
        # every mean lies inside the bounding box of the instance
        lo, hi = grid.voxel_centers_nm(np.argwhere(mask)).min(axis=0), grid.voxel_centers_nm(np.argwhere(mask)).max(axis=0)
        assert (g.means_xyz_nm >= lo).all() and (g.means_xyz_nm <= hi).all()
    # centres fitted on a small random subset of the voxels: every voxel is still counted
    few = C.instance_to_gaussians(mask, grid, k=8, max_points=500)
    assert len(few) == 8 and np.allclose(total_moments(few)[1], one.covariances_nm2[0], rtol=1e-8, atol=1e-4)
    assert not np.array_equal(few.means_xyz_nm, C.instance_to_gaussians(mask, grid, k=8).means_xyz_nm)
    # k-means in nanometres gives compact parts: about as tall as wide, although a voxel is 5 times taller
    g = C.instance_to_gaussians(mask, grid, k=64)
    tall = np.sqrt(g.covariances_nm2[:, 2, 2] / g.covariances_nm2[:, 0, 0])
    assert 0.7 < np.median(tall) < 1.5, np.median(tall)


def test_gaussians_wrong_ways_separate_clustering_from_covariance():
    mask, grid, _, _, _ = ellipsoid_phantom()
    stretch = 1.0 / grid.space_scale("index")
    assert np.array_equal(stretch, [1.0, 1.0, 5.0])                # cube space -> physical space
    k = 32
    ways = C.gaussians_wrong_ways(mask, grid, k=k, seed=0)
    correct = ways["correct"]
    reference = C.instance_to_gaussians(mask, grid, k=k, seed=0)
    assert np.array_equal(correct.means_xyz_nm, reference.means_xyz_nm)
    assert np.array_equal(correct.covariances_nm2, reference.covariances_nm2)
    careless = C.instance_to_gaussians(mask, grid.as_cubes(), k=k, seed=0)   # the only source of the mistake
    true_mean = total_moments(correct)[0]
    for name, g in ways.items():
        assert np.isclose(g.weights.sum(), 1.0), name
        assert np.allclose(total_moments(g)[0], true_mean, atol=1e-6), name    # means are physical in every variant

    # "index_fit_*": parts and covariances of the careless program; means moved to physical space
    for name in ("index_fit_scaled", "index_fit_transformed"):
        assert np.allclose(ways[name].means_xyz_nm, careless.means_xyz_nm * stretch), name
        assert np.array_equal(ways[name].weights, careless.weights), name
    assert np.array_equal(ways["index_fit_scaled"].covariances_nm2, careless.covariances_nm2)
    d = np.diag(stretch)
    assert np.allclose(ways["index_fit_transformed"].covariances_nm2, d @ careless.covariances_nm2 @ d.T)

    # "same_clusters_*": the parts of the correct fit
    for name in ("same_clusters_scaled", "same_clusters_transformed"):
        assert np.allclose(ways[name].means_xyz_nm, correct.means_xyz_nm, atol=1e-6), name
        assert np.array_equal(ways[name].weights, correct.weights), name
    assert np.allclose(ways["same_clusters_transformed"].covariances_nm2, correct.covariances_nm2, rtol=1e-8, atol=1e-4)
    # only the covariance is wrong: xy block untouched, z variance 25 times too small, xz terms 5 times
    scaled = ways["same_clusters_scaled"].covariances_nm2
    assert np.allclose(scaled[:, :2, :2], correct.covariances_nm2[:, :2, :2], rtol=1e-8, atol=1e-4)
    assert np.allclose(scaled[:, 2, 2] * 25.0, correct.covariances_nm2[:, 2, 2], rtol=1e-8, atol=1e-4)
    assert np.allclose(scaled[:, 0, 2] * 5.0, correct.covariances_nm2[:, 0, 2], rtol=1e-8, atol=1e-4)

    # only the clustering is wrong: the whole mixture still has the right moments ...
    assert np.allclose(total_moments(ways["index_fit_transformed"])[1], total_moments(correct)[1], rtol=1e-8, atol=1e-4)
    # ... but the parts were made compact in index space, so in nanometres they are tall columns
    tall = lambda g: float(np.median(np.sqrt(g.covariances_nm2[:, 2, 2] / g.covariances_nm2[:, 0, 0])))  # noqa: E731
    assert tall(ways["index_fit_transformed"]) > 3.0 * tall(correct)
    # the classic mistake does not even keep the total moments
    assert not np.allclose(total_moments(ways["index_fit_scaled"])[1], total_moments(correct)[1], rtol=0.05)
    assert not np.allclose(total_moments(ways["same_clusters_scaled"])[1], total_moments(correct)[1], rtol=0.01)


def test_on_an_isotropic_grid_all_ways_are_the_same():
    mask, _, _, _, _ = ellipsoid_phantom(shape_zyx=(40, 60, 80), voxel_zyx=(8.0, 8.0, 8.0), semi_nm=(250.0, 120.0, 100.0))
    grid = VoxelGrid(mask.shape, (8.0, 8.0, 8.0))
    ways = C.gaussians_wrong_ways(mask, grid, k=16, seed=3)
    for name, g in ways.items():
        assert np.allclose(g.means_xyz_nm, ways["correct"].means_xyz_nm, atol=1e-6), name
        assert np.allclose(g.covariances_nm2, ways["correct"].covariances_nm2, rtol=1e-8, atol=1e-4), name


def test_kmeans_is_deterministic_and_fast():
    # 1.18 million voxels on a SNEMI3D-sized grid
    grid = VoxelGrid((100, 512, 512), (30.0, 6.0, 6.0))
    mask = np.zeros(grid.shape_zyx, dtype=bool)
    mask[20:80, 100:240, 100:240] = True
    assert mask.sum() == 1176000
    seconds = []
    for _ in range(2):
        start = time.perf_counter()
        a = C.instance_to_gaussians(mask, grid, k=128, seed=0)
        seconds.append(time.perf_counter() - start)
    assert min(seconds) < 3.0, seconds
    assert len(a) == 128
    b = C.instance_to_gaussians(mask, grid, k=128, seed=0)
    assert np.array_equal(a.means_xyz_nm, b.means_xyz_nm) and np.array_equal(a.covariances_nm2, b.covariances_nm2)
    assert np.array_equal(a.weights, b.weights)
    other = C.instance_to_gaussians(mask, grid, k=128, seed=1)
    assert not np.array_equal(a.means_xyz_nm, other.means_xyz_nm)
    # the parts are balanced and the mixture reproduces the box
    assert a.weights.max() < 5.0 / 128
    mean, cov = total_moments(a)
    assert np.allclose(mean, [1020.0, 1020.0, 1500.0], atol=1e-6)
    assert np.allclose(cov, np.diag([840.0 ** 2, 840.0 ** 2, 1800.0 ** 2]) / 12.0, rtol=1e-8, atol=1e-3)


def test_kmeans_with_more_parts_than_voxels():
    grid = VoxelGrid((4, 4, 4), VOXEL_ZYX)
    mask = np.zeros(grid.shape_zyx, dtype=bool)
    mask[1, 1, 1:4] = True                                         # 3 voxels
    g = C.instance_to_gaussians(mask, grid, k=10)
    assert len(g) == 3 and np.allclose(g.weights, 1.0 / 3.0)
    # each part is one voxel: exactly a voxel_gaussian
    v = C.voxel_gaussian(grid, np.argwhere(mask))
    order = np.argsort(g.means_xyz_nm[:, 0])
    assert np.allclose(g.means_xyz_nm[order], v.means_xyz_nm) and np.allclose(g.covariances_nm2, v.covariances_nm2)
    assert raises(ValueError, C.instance_to_gaussians, mask, grid, k=0)
    assert raises(ValueError, C.instance_to_gaussians, mask, grid, max_points=0)


def test_voxel_gaussian_is_a_small_anisotropic_gaussian():
    grid = VoxelGrid((100, 512, 512), (30.0, 6.0, 6.0))
    g = C.voxel_gaussian(grid, [3, 4, 5])
    assert len(g) == 1 and np.array_equal(g.weights, [1.0])
    assert np.allclose(g.means_xyz_nm[0], [5.5 * 6.0, 4.5 * 6.0, 3.5 * 30.0])
    assert np.allclose(g.covariances_nm2[0], np.diag([36.0, 36.0, 900.0]) / 12.0)
    _, sigmas = g.principal_axes()
    assert np.allclose(np.sort(sigmas[0]), np.array([6.0, 6.0, 30.0]) / np.sqrt(12.0))   # 0.29 voxel edges
    assert np.isclose(1.0 / np.sqrt(12.0), 0.29, atol=0.005)
    assert np.isclose(g.elongation()[0], 5.0)                      # dz / dx
    # scale = 3: one standard deviation is half the voxel edge, so the 1-sigma ellipsoid touches the faces
    touching = C.voxel_gaussian(grid, [3, 4, 5], scale=3.0)
    assert np.allclose(np.sqrt(np.diag(touching.covariances_nm2[0])), [3.0, 3.0, 15.0])
    # several voxels at once
    many = C.voxel_gaussian(grid, [[0, 0, 0], [99, 511, 511]])
    assert len(many) == 2 and np.allclose(many.weights, 0.5)
    assert np.allclose(many.means_xyz_nm[1], grid.extent_xyz_nm - np.array([3.0, 3.0, 15.0]))
    # the careless grid makes round Gaussians
    assert np.isclose(C.voxel_gaussian(grid.as_cubes(), [3, 4, 5]).elongation()[0], 1.0)
    assert raises(ValueError, C.voxel_gaussian, grid, [100, 0, 0])                 # outside the array
    assert raises(ValueError, C.voxel_gaussian, grid, [0.5, 0.0, 0.0])             # not an index
    assert raises(ValueError, C.voxel_gaussian, grid, [1, 2])
    assert raises(ValueError, C.voxel_gaussian, grid, [1, 2, 3], scale=0.0)
    assert raises(TypeError, C.voxel_gaussian, (30.0, 6.0, 6.0), [1, 2, 3])


# ---- wrong input ---------------------------------------------------------------------------------
def test_wrong_input_fails_loudly_in_every_converter():
    mask, grid = box_phantom()
    converters = (C.instance_to_bricks, C.instance_to_mesh, C.instance_to_skeleton, C.instance_to_gaussians,
                  C.single_gaussian, C.gaussians_wrong_ways, C.measure, C.compare_anisotropy)
    for convert in converters:
        name = convert.__name__
        assert raises(TypeError, convert, mask.astype(np.uint8), grid), name          # labels, not a mask
        assert raises(TypeError, convert, mask.astype(np.float32), grid), name
        assert raises(ValueError, convert, mask[1:], grid), name                      # shape does not fit the grid
        assert raises(ValueError, convert, mask[0], VoxelGrid((1,) + mask.shape[1:], VOXEL_ZYX)), name   # 2D
        assert raises(ValueError, convert, np.zeros_like(mask), grid), name           # empty
        assert raises(TypeError, convert, mask, VOXEL_ZYX), name                      # a tuple is not a grid


# ---- cross-checks: one representation against another, or against another library ----------------
def rotation_xyz(about_x, about_y, about_z):
    """Rotation matrix Rz Ry Rx (angles in radians)."""
    cx, sx, cy, sy, cz, sz = (fn(a) for a in (about_x, about_y, about_z) for fn in (np.cos, np.sin))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def tilted_ellipsoid_phantom():
    """A solid ellipsoid (semi-axes 400, 220, 150 nm) turned about all three axes on the
    (30, 8, 6) nm grid.  Its centre is a voxel CORNER in the middle of the array, so the
    mask is point symmetric about a point that is not a voxel centre: half a voxel of
    error, (3, 4, 15) nm, cannot hide.  The covariances xy, xz and yz are all positive.
    Returns mask, grid, centre (x, y, z) nm, true covariance (nm^2)."""
    grid = VoxelGrid((50, 150, 170), VOXEL_ZYX)
    centre = grid.index_to_nm([85, 75, 25])
    semi = np.array([400.0, 220.0, 150.0])
    rot = rotation_xyz(0.5, -0.7, 0.9)
    kji = np.indices(grid.shape_zyx).reshape(3, -1).T
    along_axes = (grid.voxel_centers_nm(kji) - centre) @ rot
    mask = (((along_axes / semi) ** 2).sum(axis=1) <= 1.0).reshape(grid.shape_zyx)
    return mask, grid, centre, rot @ np.diag(semi ** 2 / 5.0) @ rot.T


def solid_moments(mesh):
    """Volume (nm^3), centre of mass (nm) and covariance (nm^2) of the solid enclosed by a
    closed mesh, summed over the signed tetrahedra (origin, a, b, c) of its triangles."""
    v, f = mesh.vertices_xyz_nm, mesh.faces
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    six_volumes = np.einsum("ij,ij->i", a, np.cross(b, c))
    volume = six_volumes.sum() / 6.0
    s = a + b + c
    centre = (six_volumes[:, None] * s).sum(axis=0) / (24.0 * volume)
    # integral of p p^T over a tetrahedron (0, a, b, c) = volume / 20 * (a a^T + b b^T + c c^T + s s^T)
    second = sum(np.einsum("n,ni,nj->ij", six_volumes, p, p) for p in (a, b, c, s)) / 120.0
    return volume, centre, second / volume - np.outer(centre, centre)


def principal_direction(points):
    """Unit vector along the largest spread of a point cloud."""
    return np.linalg.eigh(np.cov((points - points.mean(axis=0)).T))[1][:, -1]


def test_solid_moments_helper_on_a_box_written_by_hand():
    # corner (1, 2, 3), edges (4, 5, 6): 12 outward triangles
    corners = np.array([[x, y, z] for z in (3.0, 9.0) for y in (2.0, 7.0) for x in (1.0, 5.0)])
    quads = ([0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5])
    faces = np.array([[q[0], q[1], q[2]] for q in quads] + [[q[0], q[2], q[3]] for q in quads], dtype=np.int32)
    volume, centre, cov = solid_moments(Mesh(corners, faces))
    assert np.isclose(volume, 4.0 * 5.0 * 6.0)
    assert np.allclose(centre, [3.0, 4.5, 6.0])
    assert np.allclose(cov, np.diag([16.0, 25.0, 36.0]) / 12.0)


def test_mesh_bricks_and_gaussian_describe_the_same_solid():
    mask, grid, centre, true_cov = tilted_ellipsoid_phantom()
    largest = np.linalg.eigvalsh(true_cov).max()
    assert np.array_equal(centre, [510.0, 600.0, 750.0])
    assert (true_cov[[0, 0, 1], [1, 2, 2]] > 0.15 * largest).all()                # clearly tilted in xy, xz and yz
    one = C.single_gaussian(mask, grid)
    assert np.allclose(one.means_xyz_nm[0], centre, atol=1e-6)                    # point symmetric mask: exact
    assert np.abs(one.covariances_nm2[0] - true_cov).max() < 0.005 * largest      # measured 0.15 %
    for surface_only in (False, True):
        bricks = C.instance_to_bricks(mask, grid, surface_only=surface_only)
        assert np.allclose(bricks.centers_xyz_nm.mean(axis=0), centre, atol=1e-6), surface_only
    mixture = C.instance_to_gaussians(mask, grid, k=32)
    assert np.allclose(total_moments(mixture)[0], centre, atol=1e-6)
    assert mask[tuple(grid.nm_to_voxel(mixture.means_xyz_nm).T)].all()            # convex solid: every part's mean is inside
    for backend in ("zmesh", "skimage"):
        mesh = C.instance_to_mesh(mask, grid, backend=backend)
        volume, mesh_centre, mesh_cov = solid_moments(mesh)
        assert np.isclose(volume, mesh.volume_nm3(), rtol=1e-9), backend
        assert 0.995 < volume / (mask.sum() * DX * DY * DZ) < 1.0, backend        # only the bevels are missing
        assert np.allclose(mesh_centre, centre, atol=1e-2), (backend, mesh_centre - centre)
        assert np.abs(mesh_cov - one.covariances_nm2[0]).max() < 0.005 * largest, backend   # measured 0.11 %
        # a closed surface: the area vectors of its triangles add up to zero
        v, f = mesh.vertices_xyz_nm, mesh.faces
        area_vectors = 0.5 * np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        assert np.abs(area_vectors.sum(axis=0)).max() < 1e-6 * mesh.area_nm2(), backend
        # the box around the bricks is the box around the mesh (corner convention in both)
        everything = C.instance_to_bricks(mask, grid, surface_only=False)
        assert np.allclose(v.min(axis=0), everything.centers_xyz_nm.min(axis=0) - everything.size_xyz_nm / 2, atol=1e-3)
        assert np.allclose(v.max(axis=0), everything.centers_xyz_nm.max(axis=0) + everything.size_xyz_nm / 2, atol=1e-3)
    # mirror the ARRAY along one axis: the two covariances that involve that axis change sign, nothing else
    for array_axis, column in ((0, 2), (1, 1), (2, 0)):
        flipped = np.flip(mask, axis=array_axis).copy()
        sign = np.ones(3)
        sign[column] = -1.0
        expected = one.covariances_nm2[0] * np.outer(sign, sign)
        assert np.allclose(C.single_gaussian(flipped, grid).covariances_nm2[0], expected, rtol=1e-9, atol=1e-6), array_axis
        assert np.abs(solid_moments(C.instance_to_mesh(flipped, grid))[2] - expected).max() < 0.005 * largest, array_axis
        assert np.abs(expected - one.covariances_nm2[0]).max() > 0.3 * largest      # the check can tell them apart


def test_oblique_rod_skeleton_direction_ends_and_radii():
    from scipy import ndimage

    # a capsule of radius 90 nm from a to b, oblique to all three axes (it goes down in y)
    grid = VoxelGrid((40, 120, 150), VOXEL_ZYX)
    a, b, radius = np.array([150.0, 800.0, 250.0]), np.array([750.0, 200.0, 950.0]), 90.0
    length = float(np.linalg.norm(b - a))
    along = (b - a) / length
    assert length == 1100.0 and along[1] < 0 < along[0] < along[2]
    centres = grid.voxel_centers_nm(np.indices(grid.shape_zyx).reshape(3, -1).T)
    t = np.clip((centres - a) @ along, 0.0, length)
    mask = (np.linalg.norm(centres - (a + t[:, None] * along), axis=1) <= radius).reshape(grid.shape_zyx)

    skeleton = C.instance_to_skeleton(mask, grid)
    v = skeleton.vertices_xyz_nm
    assert skeleton.n_end_points() == 2 and skeleton.n_branch_points() == 0
    direction = principal_direction(v)
    assert angle_deg(direction, along) < 3.0                       # measured 1.0 degree
    # the two ends are at the two tips, and each is at the RIGHT tip (no axis is mirrored)
    ends = v[skeleton.degrees() == 1]
    low, high = ends[np.argmin(ends @ along)], ends[np.argmax(ends @ along)]
    assert np.linalg.norm(low - (a - radius * along)) < radius / 2 and np.linalg.norm(high - (b + radius * along)) < radius / 2
    # every node is near the axis of the capsule (the line through a and b), well away from the wall
    from_a = v - a
    assert np.linalg.norm(from_a - (from_a @ along)[:, None] * along, axis=1).max() < 0.75 * radius  # measured 42 nm
    # radii: exactly the anisotropic distance transform of another library (scipy) at the node's voxel
    distance = ndimage.distance_transform_edt(mask, sampling=VOXEL_ZYX)             # nm, (z, y, x) spacing
    assert np.allclose(skeleton.radii_nm, distance[tuple(grid.nm_to_voxel(v).T)], atol=1e-3)
    assert 0.85 * radius < np.median(skeleton.radii_nm) < radius + DX
    # cable: the chain of voxel centres zig-zags (+29 %); smoothed over 4 sections it is the tip-to-tip length
    true_cable = length + 2 * radius
    assert 1.2 < skeleton.cable_length_nm() / true_cable < 1.4
    assert abs(C.smooth_skeleton(skeleton, 4 * DZ).cable_length_nm() / true_cable - 1) < 0.05

    # ignoring anisotropy turns the rod: its direction becomes the true one scaled by g / (dx, dy, dz)
    cubes = C.instance_to_skeleton(mask, grid.as_cubes())
    predicted = along * grid.space_scale("index")
    assert angle_deg(principal_direction(cubes.vertices_xyz_nm), predicted) < 5.0   # measured 2.1 degrees
    assert angle_deg(predicted, along) > 25.0
    # voxel size in a wrong order: the rod points somewhere else
    for wrong in ((30.0, 6.0, 8.0), (6.0, 8.0, 30.0)):
        other = C.instance_to_skeleton(mask, VoxelGrid(grid.shape_zyx, wrong))
        assert angle_deg(principal_direction(other.vertices_xyz_nm), along) > 8.0, wrong


def test_wrong_ways_use_the_index_space_of_the_grid_when_dx_differs_from_dy():
    mask, grid, _, _ = tilted_ellipsoid_phantom()
    assert np.allclose(grid.space_scale("index"), [1.0, 6.0 / 8.0, 6.0 / 30.0])    # physical nm -> index-space nm
    ways = C.gaussians_wrong_ways(mask, grid, k=16, seed=2)
    correct, scaled = ways["correct"], ways["same_clusters_scaled"]
    # the covariance mistake alone = the correct covariances as grid.py itself moves them to index space
    assert np.allclose(scaled.covariances_nm2, grid.covariance_to_space(correct.covariances_nm2, "index"), rtol=1e-9, atol=1e-6)
    assert np.allclose(scaled.means_xyz_nm, correct.means_xyz_nm, atol=1e-6)
    # here y is squashed as well (dy = 8 nm, g = 6 nm): variances are (6/8)^2 and (6/30)^2 of the true ones
    ratio = scaled.covariances_nm2 / correct.covariances_nm2
    assert np.allclose(ratio[:, 0, 0], 1.0) and np.allclose(ratio[:, 1, 1], 0.5625) and np.allclose(ratio[:, 2, 2], 0.04)
    # the careless fit, moved to index space with grid.py's own functions, is the fit on grid.as_cubes()
    careless = C.instance_to_gaussians(mask, grid.as_cubes(), k=16, seed=2)
    assert np.allclose(grid.to_space(ways["index_fit_scaled"].means_xyz_nm, "index"), careless.means_xyz_nm, atol=1e-6)
    assert np.array_equal(ways["index_fit_scaled"].covariances_nm2, careless.covariances_nm2)
    in_index_space = grid.covariance_to_space(ways["index_fit_transformed"].covariances_nm2, "index")
    assert np.allclose(in_index_space, careless.covariances_nm2, rtol=1e-9, atol=1e-6)
    # "index space", literally: moments of the voxel indices (voxel size 1) times g = 6 nm, g^2 = 36 nm^2
    indices = C.single_gaussian(mask, VoxelGrid(grid.shape_zyx, (1.0, 1.0, 1.0)))
    cubes = C.single_gaussian(mask, grid.as_cubes())
    assert np.allclose(indices.means_xyz_nm * 6.0, cubes.means_xyz_nm, atol=1e-6)
    assert np.allclose(indices.covariances_nm2 * 36.0, cubes.covariances_nm2, rtol=1e-9, atol=1e-6)


def test_blur_sigma_is_in_nanometres_along_every_axis():
    # A plate of thickness T blurred with sigma peaks at erf(T / (2 sqrt(2) sigma)).  It keeps a
    # surface at the level 0.5 only while T > 1.349 sigma -- in nanometres, whatever the voxel size.
    grid = VoxelGrid((13, 81, 101), VOXEL_ZYX)
    for axis, n in ((0, 3), (1, 5), (2, 7)):                       # plates 90, 40 and 42 nm thick
        mask = np.zeros(grid.shape_zyx, dtype=bool)
        plate = [slice(None)] * 3
        first = grid.shape_zyx[axis] // 2 - n // 2
        plate[axis] = slice(first, first + n)
        mask[tuple(plate)] = True
        critical_sigma = n * VOXEL_ZYX[axis] / 1.349
        kept = C.instance_to_mesh(mask, grid, smooth_sigma_nm=0.9 * critical_sigma, backend="skimage")
        assert raises(ValueError, C.instance_to_mesh, mask, grid, smooth_sigma_nm=1.1 * critical_sigma, backend="skimage"), axis
        # the surface that is left is centred on the middle plane of the plate
        middle = (first + n / 2.0) * VOXEL_ZYX[axis]
        across = kept.vertices_xyz_nm[:, 2 - axis]                 # array axis 0 is z = column 2
        assert np.isclose(across.mean(), middle, atol=1e-2) and across.min() < middle < across.max(), axis
        assert across.max() - across.min() < n * VOXEL_ZYX[axis], axis             # thinner than the plate


def test_brick_shell_is_what_an_erosion_leaves_also_at_the_array_border():
    from scipy import ndimage

    whole, _, _, _ = tilted_ellipsoid_phantom()
    mask = np.ascontiguousarray(whole[20:, 60:, :85])              # the ellipsoid cut by three faces of the array
    grid = VoxelGrid(mask.shape, VOXEL_ZYX)
    assert mask[0].any() and mask[:, 0].any() and mask[:, :, -1].any()
    six_neighbours = ndimage.generate_binary_structure(3, 1)
    inside = ndimage.binary_erosion(mask, structure=six_neighbours, border_value=0)   # outside the array = outside
    shell = C.instance_to_bricks(mask, grid)
    assert np.array_equal(shell.kji, np.argwhere(mask & ~inside))
    assert shell.n_voxels_total == mask.sum() and 0 < len(shell) < mask.sum()
    on_cut_face = np.argwhere(mask[:, :, -1])
    assert {(k, j, mask.shape[2] - 1) for k, j in on_cut_face} <= {tuple(r) for r in shell.kji}
    assert np.array_equal(grid.nm_to_voxel(shell.centers_xyz_nm), shell.kji)


def test_smooth_skeleton_is_the_arc_length_average_computed_the_slow_way():
    rng = np.random.default_rng(5)
    n, window = 60, 150.0
    steps = rng.normal(size=(n - 1, 3)) * [DX, DY, DZ] + [10.0, 0.0, 0.0]
    p = np.concatenate([np.zeros((1, 3)), np.cumsum(steps, axis=0)])
    chain = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int32)
    smooth = C.smooth_skeleton(Skeleton(p, chain, np.ones(n)), window).vertices_xyz_nm
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])   # arc length at each vertex
    slow = p.copy()
    for i in range(1, n - 1):
        half = min(window / 2.0, s[i], s[-1] - s[i])
        t = np.linspace(s[i] - half, s[i] + half, 20001)
        q = np.stack([np.interp(t, s, p[:, axis]) for axis in range(3)], axis=1)       # the polyline at arc length t
        slow[i] = (0.5 * (q[:-1] + q[1:])).mean(axis=0)
    assert np.allclose(smooth, slow, atol=1e-4)                    # measured 2e-7 nm
    assert np.linalg.norm(smooth - p, axis=1).max() > 10.0         # and it did move the vertices
    # the order in which the vertices are stored does not matter
    order = rng.permutation(n)
    position = np.argsort(order)
    shuffled = C.smooth_skeleton(Skeleton(p[order], position[chain].astype(np.int32), np.ones(n)), window)
    assert np.allclose(shuffled.vertices_xyz_nm[position], smooth, atol=1e-9)


def test_mean_radius_is_taken_along_the_cable_not_over_the_nodes():
    # Two arms of 300 nm from the origin with the node spacing of 30 x 6 x 6 nm voxels:
    # along x a node every 6 nm (radius 50 nm), along z a node every 30 nm (radius 10 nm).
    along_x = np.stack([np.arange(51) * 6.0, np.zeros(51), np.zeros(51)], axis=1)
    along_z = np.stack([np.zeros(10), np.zeros(10), np.arange(1, 11) * 30.0], axis=1)
    edges = [[m, m + 1] for m in range(50)] + [[0, 51]] + [[51 + m, 52 + m] for m in range(9)]
    radii = np.concatenate([np.full(51, 50.0), np.full(10, 10.0)])
    skeleton = Skeleton(np.concatenate([along_x, along_z]), np.array(edges, dtype=np.int32), radii)
    assert skeleton.cable_length_nm() == 600.0
    assert len(along_x) - 1 == 5 * len(along_z)                    # five times more nodes per nanometre along x
    assert np.isclose(radii.mean(), 43.44, atol=0.01)              # the mean over nodes counts the x arm five times
    # along the cable: 300 nm at 50 nm, the first z edge (30 nm) at (50 + 10) / 2, 270 nm at 10 nm
    assert np.isclose(C.mean_radius_nm(skeleton), (300.0 * 50.0 + 30.0 * 30.0 + 270.0 * 10.0) / 600.0)
    assert np.isclose(C.mean_radius_nm(skeleton), 31.0)
    # measure() reports the mean along the cable
    mask, grid = l_phantom()
    assert C.measure(mask, grid)["mean_radius_nm"] == C.mean_radius_nm(C.instance_to_skeleton(mask, grid))


def test_a_cropped_sample_is_put_back_by_its_origin():
    # the chiral L inside a (30, 8, 6) nm sample, converted from the whole sample and from a crop of it
    mask, grid = l_phantom(peg=True)
    sample_whole = S.Sample("toy", Path("toy.h5"), np.zeros(grid.shape_zyx, dtype=np.uint8), mask.astype(np.uint8) * 3,
                            grid, {}, (0.0, 255.0))
    crop = sample_whole.crop(slice(2, 12), slice(3, 40), slice(6, 60))
    origin = crop.origin_xyz_nm
    assert np.array_equal(origin, [6 * DX, 3 * DY, 2 * DZ]) and np.array_equal(origin, [36.0, 24.0, 60.0])
    assert crop.grid.voxel_size_zyx_nm == VOXEL_ZYX and crop.grid.shape_zyx == (10, 37, 54)
    whole_mask, crop_mask = sample_whole.mask(3), crop.mask(3)
    assert np.array_equal(whole_mask, mask) and crop_mask.sum() == mask.sum()

    mesh, mesh_crop = C.instance_to_mesh(whole_mask, grid), C.instance_to_mesh(crop_mask, crop.grid)
    assert np.array_equal(mesh.faces, mesh_crop.faces)
    assert np.allclose(mesh_crop.vertices_xyz_nm + origin, mesh.vertices_xyz_nm, atol=1e-3)
    bricks, bricks_crop = C.instance_to_bricks(whole_mask, grid), C.instance_to_bricks(crop_mask, crop.grid)
    assert np.allclose(bricks_crop.centers_xyz_nm + origin, bricks.centers_xyz_nm, atol=1e-9)
    assert np.array_equal(bricks_crop.kji + [2, 3, 6], bricks.kji)
    skeleton, skeleton_crop = (C.instance_to_skeleton(whole_mask, grid, dust_nm3=0.0),
                               C.instance_to_skeleton(crop_mask, crop.grid, dust_nm3=0.0))
    assert np.allclose(skeleton_crop.vertices_xyz_nm + origin, skeleton.vertices_xyz_nm, atol=1e-3)
    assert np.array_equal(skeleton_crop.radii_nm, skeleton.radii_nm)
    one, one_crop = C.single_gaussian(whole_mask, grid), C.single_gaussian(crop_mask, crop.grid)
    assert np.allclose(one_crop.means_xyz_nm + origin, one.means_xyz_nm, atol=1e-9)
    assert np.allclose(one_crop.covariances_nm2, one.covariances_nm2, rtol=1e-9, atol=1e-6)


# ---- the anisotropy exhibit ------------------------------------------------------------------------
def test_measure_and_compare_on_a_rod_across_the_sections():
    # a rod along z, 15 x 15 voxels of 6 nm wide, through all 40 sections of 30 nm
    grid = VoxelGrid((40, 29, 29), (30.0, 6.0, 6.0))
    mask = np.zeros(grid.shape_zyx, dtype=bool)
    mask[:, 7:22, 7:22] = True
    m = C.measure(mask, grid)
    assert m["n_voxels"] == 9000 and m["volume_nm3"] == 9000 * 1080.0
    assert np.isclose(m["mesh_volume_nm3"], bevelled_box_volume_nm3(15, 15, 40, 6.0, 6.0, 30.0), rtol=1e-6)
    assert np.isclose(m["mesh_area_nm2"], bevelled_box_area_nm2(15, 15, 40, 6.0, 6.0, 30.0), rtol=1e-6)
    assert np.isclose(m["cable_length_nm"], 39 * 30.0) and np.isclose(m["cable_length_smooth_nm"], 39 * 30.0)
    assert np.isclose(m["mean_radius_nm"], 8 * 6.0)
    assert m["n_branch_points"] == 0 and m["n_end_points"] == 2
    assert m["smooth_nm"] == 4 * 30.0                              # default: 4 x the largest voxel edge
    assert C.measure(mask, grid, smooth_nm=77.0)["smooth_nm"] == 77.0

    rows = C.compare_anisotropy(mask, grid)
    assert [r["quantity"] for r in rows] == list(C.MEASURED)
    assert all(type(r) is dict and set(r) == {"quantity", "correct", "as_cubes", "as_cubes_percent"} for r in rows)
    row = {r["quantity"]: r for r in rows}
    cubes = C.measure(mask, grid.as_cubes(), smooth_nm=120.0)      # the only definition of "ignored"
    for quantity in C.MEASURED:
        assert row[quantity]["correct"] == m[quantity], quantity
        assert row[quantity]["as_cubes"] == cubes[quantity], quantity
    # everything along z shrinks by 5, nothing in the plane changes
    assert np.isclose(row["volume_nm3"]["as_cubes_percent"], -80.0)
    assert np.isclose(row["mesh_volume_nm3"]["as_cubes_percent"], -80.0, atol=1e-4)
    assert np.isclose(row["cable_length_nm"]["as_cubes_percent"], -80.0, atol=1e-4)
    assert np.isclose(row["cable_length_smooth_nm"]["as_cubes_percent"], -80.0, atol=1e-4)
    assert np.isclose(row["mean_radius_nm"]["as_cubes_percent"], 0.0, atol=1e-4)
    assert -80.0 < row["mesh_area_nm2"]["as_cubes_percent"] < -70.0
    assert row["n_branch_points"]["as_cubes_percent"] == 0.0       # 0 and 0

    # with a reference (here: the object itself, as a pair and as a measure() dict)
    for reference in ((mask, grid), C.measure(mask, grid)):
        with_ref = {r["quantity"]: r for r in C.compare_anisotropy(mask, grid, reference=reference)}
        for quantity in C.MEASURED:
            r = with_ref[quantity]
            assert set(r) == {"quantity", "correct", "as_cubes", "as_cubes_percent", "reference",
                              "correct_percent_of_ref", "as_cubes_percent_of_ref"}
            assert r["reference"] == r["correct"] and r["correct_percent_of_ref"] == 0.0
            assert r["as_cubes_percent_of_ref"] == r["as_cubes_percent"]


def test_compare_measures_an_object_that_ignoring_anisotropy_would_call_dust():
    # 16 x 10 x 12 voxels of 30 x 6 x 6 nm: 2.07e6 nm^3 in truth, 4.15e5 nm^3 as cubes of 6 nm.
    # The default dust volume of instance_to_skeleton (1e6 nm^3) lies between the two.
    grid = VoxelGrid((20, 40, 60), (30.0, 6.0, 6.0))
    mask = np.zeros(grid.shape_zyx, dtype=bool)
    mask[2:18, 15:25, 10:22] = True
    assert mask.sum() * 1080.0 == 2073600.0 and mask.sum() * 216.0 == 414720.0
    assert len(C.instance_to_skeleton(mask, grid).vertices_xyz_nm) >= 2
    assert raises(ValueError, C.instance_to_skeleton, mask, grid.as_cubes())       # "dust", by the wrong volume
    rows = {r["quantity"]: r for r in C.compare_anisotropy(mask, grid)}             # both columns are measured
    assert rows["volume_nm3"]["correct"] == 2073600.0 and rows["volume_nm3"]["as_cubes"] == 414720.0
    assert np.isclose(rows["volume_nm3"]["as_cubes_percent"], -80.0)
    assert 0 < rows["cable_length_nm"]["as_cubes"] < 0.5 * rows["cable_length_nm"]["correct"]   # measured 135 and 463 nm


def test_compare_on_an_isotropic_grid_shows_no_change():
    grid = VoxelGrid((40, 29, 29), (8.0, 8.0, 8.0))
    mask = np.zeros(grid.shape_zyx, dtype=bool)
    mask[:, 7:22, 7:22] = True
    assert grid.as_cubes() == grid
    for row in C.compare_anisotropy(mask, grid):
        assert row["as_cubes"] == row["correct"] and row["as_cubes_percent"] == 0.0, row


# ---- real samples ------------------------------------------------------------------------------------
def test_snemi3d_subject_converts_to_every_representation():
    s = sample("snemi3d")
    i = s.skeleton_subjects[0]                                     # the neurite that runs across the sections
    mask, grid = s.mask(i), s.grid
    n = int(mask.sum())
    near = grow_by_one_voxel(mask)
    box_lo = grid.index_to_nm(np.array(s.bbox_zyx(i)[:3])[::-1])
    box_hi = grid.index_to_nm(np.array(s.bbox_zyx(i)[3:])[::-1])

    bricks = C.instance_to_bricks(mask, grid, instance_id=i)
    assert bricks.n_voxels_total == n and 0 < len(bricks) < n
    assert mask[tuple(bricks.kji.T)].all()
    assert np.array_equal(grid.nm_to_voxel(bricks.centers_xyz_nm), bricks.kji)

    mesh = C.instance_to_mesh(mask, grid, instance_id=i)
    assert near[tuple(voxels_of(mesh.vertices_xyz_nm, grid).T)].all()          # in or next to the instance
    assert np.allclose(mesh.vertices_xyz_nm.min(axis=0), box_lo, atol=1e-2)
    assert np.allclose(mesh.vertices_xyz_nm.max(axis=0), box_hi, atol=1e-2)
    voxel_volume = n * 30.0 * 6.0 * 6.0
    assert 0.99 * voxel_volume < mesh.volume_nm3() < voxel_volume
    assert (edge_use_counts(mesh.faces) % 2 == 0).all()                         # closed
    wire = C.instance_to_mesh(mask, grid, simplify_factor=20)
    assert len(wire.faces) < len(mesh.faces) / 10 and abs(wire.volume_nm3() / mesh.volume_nm3() - 1) < 0.01
    assert near[tuple(voxels_of(wire.vertices_xyz_nm, grid).T)].mean() > 0.99

    skeleton = C.instance_to_skeleton(mask, grid, instance_id=i)
    kji = grid.nm_to_voxel(skeleton.vertices_xyz_nm)
    assert mask[tuple(kji.T)].all()                                             # every node inside the instance
    assert np.allclose(skeleton.vertices_xyz_nm, grid.voxel_centers_nm(kji), atol=1e-3)
    assert (skeleton.radii_nm > 0).all() and skeleton.radii_nm.max() < 1000.0
    assert skeleton.n_end_points() >= 2
    assert 3000.0 < skeleton.cable_length_nm() < 30000.0                        # it crosses 3000 nm of sections
    smooth = C.smooth_skeleton(skeleton, 120.0)
    assert 0.75 * skeleton.cable_length_nm() < smooth.cable_length_nm() < skeleton.cable_length_nm()
    assert near[tuple(voxels_of(smooth.vertices_xyz_nm, grid).T)].mean() > 0.98

    gaussians = C.instance_to_gaussians(mask, grid, k=64, instance_id=i)
    one = C.single_gaussian(mask, grid)
    assert len(gaussians) == 64 and np.isclose(gaussians.weights.sum(), 1.0)
    mean, cov = total_moments(gaussians)
    assert np.allclose(mean, one.means_xyz_nm[0], atol=1e-6) and np.allclose(cov, one.covariances_nm2[0], rtol=1e-8, atol=1e-3)
    assert (gaussians.means_xyz_nm >= box_lo).all() and (gaussians.means_xyz_nm <= box_hi).all()
    assert near[tuple(voxels_of(gaussians.means_xyz_nm, grid).T)].mean() > 0.8  # most part centres are in the neurite
    # this neurite runs across the sections: its long axis is closer to z than to the section plane ...
    assert angle_deg(long_axis(one), [0.0, 0.0, 1.0]) < 30.0
    # ... and a program that ignores anisotropy sees it lying in the plane
    assert angle_deg(long_axis(C.single_gaussian(mask, grid.as_cubes())), [0.0, 0.0, 1.0]) > 60.0

    # ignoring anisotropy: the mesh is the same mesh squashed 5 times along z
    cubes = C.instance_to_mesh(mask, grid.as_cubes())
    assert np.allclose(cubes.vertices_xyz_nm, mesh.vertices_xyz_nm * [1.0, 1.0, 0.2], atol=1e-2)
    assert np.isclose(cubes.volume_nm3() / mesh.volume_nm3(), 0.2, rtol=1e-5)
    assert cubes.area_nm2() < 0.6 * mesh.area_nm2()


def test_fib25_strided_by_4_against_its_isotropic_original():
    f = sample("fib25")
    f4 = f.strided(4)
    assert f4.grid.voxel_size_zyx_nm == (32.0, 8.0, 8.0) and f4.grid.shape_zyx == (96, 384, 384)
    i = f.skeleton_subjects[1]                                     # the neurite that runs inside the section plane
    mask, mask4 = f.mask(i), f4.mask(i)
    assert abs(mask4.sum() * 4 / mask.sum() - 1) < 0.01

    # every representation of the strided instance lands on the strided instance
    near = grow_by_one_voxel(mask4)
    mesh4 = C.instance_to_mesh(mask4, f4.grid)
    skeleton4 = C.instance_to_skeleton(mask4, f4.grid)
    bricks4 = C.instance_to_bricks(mask4, f4.grid)
    assert near[tuple(voxels_of(mesh4.vertices_xyz_nm, f4.grid).T)].all()
    assert mask4[tuple(f4.grid.nm_to_voxel(skeleton4.vertices_xyz_nm).T)].all()
    assert np.array_equal(f4.grid.nm_to_voxel(bricks4.centers_xyz_nm), bricks4.kji)
    # and on the ORIGINAL instance too, because both grids describe the same nanometres:
    # a node of the 32 nm skeleton is at most half a thick voxel away from the 8 nm instance
    near_original = grow_by_one_voxel(grow_by_one_voxel(mask))
    assert near_original[tuple(voxels_of(skeleton4.vertices_xyz_nm, f.grid).T)].mean() > 0.95

    reference = C.measure(mask, f.grid, smooth_nm=128.0)
    rows = {r["quantity"]: r for r in C.compare_anisotropy(mask4, f4.grid, reference=reference)}
    # with the correct voxel size the thick sections cost a few per cent ...
    assert abs(rows["volume_nm3"]["correct_percent_of_ref"]) < 1.0
    assert abs(rows["mesh_volume_nm3"]["correct_percent_of_ref"]) < 1.0
    assert 0.0 < rows["mesh_area_nm2"]["correct_percent_of_ref"] < 12.0        # the staircase adds area
    assert abs(rows["cable_length_smooth_nm"]["correct_percent_of_ref"]) < 10.0
    assert abs(rows["mean_radius_nm"]["correct_percent_of_ref"]) < 10.0
    # ... with anisotropy ignored volume is a quarter, area about half, radius less than half
    assert np.isclose(rows["volume_nm3"]["as_cubes_percent"], -75.0)
    assert np.isclose(rows["mesh_volume_nm3"]["as_cubes_percent"], -75.0, atol=1e-3)
    assert -55.0 < rows["mesh_area_nm2"]["as_cubes_percent_of_ref"] < -35.0
    assert rows["mean_radius_nm"]["as_cubes_percent_of_ref"] < -40.0
    assert rows["cable_length_smooth_nm"]["as_cubes"] < rows["cable_length_smooth_nm"]["correct"]

    # where a strided object is drawn: the kept plane stands for its whole slab, so this neurite
    # (which does not reach the z faces) sits (4 / 2 - phase - 1 / 2) * 8 nm further along z
    everything = f.mask(i, largest_component=False)
    assert not everything[0].any() and not everything[-1].any()
    z_original = C.single_gaussian(everything, f.grid).means_xyz_nm[0, 2]
    for phase in range(4):
        shifted = f.strided(4, phase=phase)
        z_strided = C.single_gaussian(shifted.mask(i, largest_component=False), shifted.grid).means_xyz_nm[0, 2]
        assert abs((z_strided - z_original) - (1.5 - phase) * 8.0) < 1.0, (phase, z_strided - z_original)


def run_as_script(namespace) -> int:
    """Call every test_* function; print PASS / FAIL and the time; return the number of failures."""
    failed = 0
    for name, fn in list(namespace.items()):
        if name.startswith("test_") and callable(fn):
            start = time.perf_counter()
            try:
                fn()
                print(f"PASS {name}  [{time.perf_counter() - start:.1f} s]")
            except Exception:
                failed += 1
                traceback.print_exc()
                print("FAIL", name)
    print(f"{failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_as_script(globals()) else 0)
