"""Tests of the contract of ncntm.viz: grid.py (VoxelGrid) and the containers of types.py.

    python -m pytest tests/test_grid.py        or        python tests/test_grid.py

numpy only.  Every number that a docstring of grid.py states is checked here.  The test voxel
is 30 x 8 x 6 nm (dz, dy, dx): all three sizes differ, so a swapped axis changes the result.
"""
import math
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ncntm.viz import grid as G  # noqa: E402
from ncntm.viz.grid import NM_PER_UNIT, SPACES, VoxelGrid  # noqa: E402
from ncntm.viz.types import Gaussians, Mesh, Skeleton, rotation_to_quaternion_wxyz  # noqa: E402

GRID = VoxelGrid((5, 7, 11), (30.0, 8.0, 6.0))             # nz, ny, nx = 5, 7, 11;  dz, dy, dx = 30, 8, 6 nm
SNEMI3D = VoxelGrid((100, 512, 512), (30.0, 6.0, 6.0))
CREMI = VoxelGrid((75, 768, 768), (40.0, 4.0, 4.0))
FIB25 = VoxelGrid((384, 384, 384), (8.0, 8.0, 8.0))


def raises(error, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error:
        return True
    return False


# ---- sizes ---------------------------------------------------------------------------------
def test_units_and_spaces():
    assert NM_PER_UNIT == 1000.0 and SPACES == ("physical", "index")


def test_array_order_is_zyx_and_coordinates_are_xyz():
    assert GRID.shape_zyx == (5, 7, 11) and GRID.voxel_size_zyx_nm == (30.0, 8.0, 6.0)
    assert list(GRID.shape_xyz) == [11, 7, 5] and list(GRID.voxel_size_xyz_nm) == [6.0, 8.0, 30.0]
    assert list(G.zyx_to_xyz([1, 2, 3])) == [3, 2, 1]
    assert list(GRID.extent_xyz_nm) == [66.0, 56.0, 150.0]                 # n * d per axis, (x, y, z)


def test_the_numbers_of_the_docstrings():
    assert list(SNEMI3D.extent_xyz_nm) == [3072.0, 3072.0, 3000.0]         # a cube, although the array is flat
    assert SNEMI3D.anisotropy == 5.0 and CREMI.anisotropy == 10.0 and FIB25.anisotropy == 1.0
    assert FIB25.is_isotropic and not SNEMI3D.is_isotropic
    assert GRID.in_plane_nm == 6.0 and GRID.anisotropy == 5.0              # g = min(dx, dy); dz / g
    assert SNEMI3D.describe() == "512 x 512 x 100 voxels (x, y, z) of 6 x 6 x 30 nm = 3.07 x 3.07 x 3.00 um, anisotropy 5:1"


def test_wrong_input_fails_loudly():
    assert raises(ValueError, VoxelGrid, (5, 7), (30.0, 8.0, 6.0))
    assert raises(ValueError, VoxelGrid, (5, 7, 11), (30.0, 8.0))
    assert raises(ValueError, VoxelGrid, (5, 7, 11), (30.0, 0.0, 6.0))
    assert raises(ValueError, VoxelGrid, (5, 7, 11), (30.0, -8.0, 6.0))
    for method in (GRID.display_voxel_size_xyz_nm, GRID.extent_nm, GRID.space_scale):
        assert raises(ValueError, method, "voxel")


# ---- the corner convention -------------------------------------------------------------------
def test_voxel_is_a_box_from_its_index_and_its_centre_is_at_plus_one_half():
    k, j, i = 2, 3, 4
    assert list(GRID.index_to_nm((i, j, k))) == [24.0, 24.0, 60.0]                 # low corner: (i*dx, j*dy, k*dz)
    assert list(GRID.index_to_nm((i + 1, j + 1, k + 1))) == [30.0, 32.0, 90.0]     # high corner
    assert list(GRID.voxel_centers_nm([k, j, i])) == [27.0, 28.0, 75.0]            # ((i+.5)*dx, (j+.5)*dy, (k+.5)*dz)
    assert list(GRID.index_to_nm(GRID.shape_xyz)) == list(GRID.extent_xyz_nm)      # the array fills [0, n*d]
    kji = np.argwhere(np.ones(GRID.shape_zyx, bool))
    centres = GRID.voxel_centers_nm(kji)
    assert centres.shape == (5 * 7 * 11, 3)
    assert np.array_equal(GRID.nm_to_voxel(centres), kji)                          # centre -> its own voxel
    assert np.allclose(GRID.nm_to_index(centres), kji[:, ::-1] + 0.5)
    # anywhere inside the box of voxel [k, j, i], floor gives [k, j, i]; just below it, the neighbour
    assert list(GRID.nm_to_voxel((24.0, 24.0, 60.0))) == [2, 3, 4] and list(GRID.nm_to_voxel((29.999, 31.999, 89.999))) == [2, 3, 4]
    assert list(GRID.nm_to_voxel((23.999, 24.0, 60.0))) == [2, 3, 3] and list(GRID.nm_to_voxel((-0.001, 0.0, 0.0))) == [0, 0, -1]


# ---- display spaces ----------------------------------------------------------------------------
def test_index_space_draws_every_voxel_as_a_cube_of_the_in_plane_pixel():
    assert list(GRID.display_voxel_size_xyz_nm("physical")) == [6.0, 8.0, 30.0]
    assert list(GRID.display_voxel_size_xyz_nm("index")) == [6.0, 6.0, 6.0]
    assert list(GRID.extent_nm("index")) == [66.0, 42.0, 30.0]                     # (11, 7, 5) voxels of 6 nm
    assert np.allclose(GRID.space_scale("physical"), 1.0) and np.allclose(GRID.space_scale("index"), (1.0, 0.75, 0.2))
    # x and y keep their size, z shrinks by the anisotropy
    for grid, flatter in ((SNEMI3D, 5.0), (CREMI, 10.0), (FIB25, 1.0)):
        physical, index = grid.extent_nm("physical"), grid.extent_nm("index")
        assert np.array_equal(index[:2], physical[:2]) and math.isclose(index[2], physical[2] / flatter)
        assert math.isclose(physical[2] / index[2], grid.anisotropy)


def test_points_and_covariances_move_into_a_space():
    p = np.array([[27.0, 28.0, 75.0], [6.0, 8.0, 30.0]])
    assert np.array_equal(GRID.to_space(p, "physical"), p)
    assert np.allclose(GRID.to_space(p, "index"), [[27.0, 21.0, 15.0], [6.0, 6.0, 6.0]])
    # a point keeps its voxel: the centre of voxel [k, j, i] goes to the centre of the cube [k, j, i]
    kji = np.array([[2, 3, 4], [0, 6, 10]])
    assert np.allclose(GRID.to_space(GRID.voxel_centers_nm(kji), "index"), GRID.voxel_centers_nm(kji, "index"))
    assert np.array_equal(GRID.nm_to_voxel(GRID.voxel_centers_nm(kji, "index"), "index"), kji)
    # covariances: S -> A S A, checked against a point cloud moved point by point
    rng = np.random.default_rng(3)
    cloud = rng.normal(size=(20000, 3)) @ np.array([[9.0, 2.0, 1.0], [0.0, 5.0, 3.0], [0.0, 0.0, 7.0]])
    cov = np.cov(cloud.T)
    assert np.allclose(GRID.covariance_to_space(cov, "index"), np.cov(GRID.to_space(cloud, "index").T), rtol=1e-9)
    assert np.array_equal(GRID.covariance_to_space(cov, "physical"), cov)


def test_index_to_unit_matrix_has_the_voxel_size_on_its_diagonal():
    m = GRID.index_to_unit_matrix("physical")
    assert np.array_equal(m, np.diag([0.006, 0.008, 0.030, 1.0]))                  # micrometres per voxel
    assert np.allclose(m @ [11, 7, 5, 1], [0.066, 0.056, 0.150, 1.0])
    assert np.array_equal(GRID.index_to_unit_matrix("index"), np.diag([0.006, 0.006, 0.006, 1.0]))


def test_round_tube_width_factor():
    """The rule for round tubes in a display space (used by stage.py and emray.py alike)."""
    o = np.zeros((1, 3))
    along = {axis: np.eye(3)[[k]] * 100.0 for k, axis in enumerate("xyz")}
    for axis in "xyz":
        assert SNEMI3D.tube_width_factor(o, along[axis], "physical")[0] == 1.0
        assert SNEMI3D.tube_width_factor(o, along[axis])[0] == 1.0                 # physical is the default
    assert abs(SNEMI3D.tube_width_factor(o, along["z"], "index")[0] - 1.0) < 1e-12          # cross-section in x, y
    for axis in "xy":
        assert abs(SNEMI3D.tube_width_factor(o, along[axis], "index")[0] - math.sqrt(1 / 5)) < 1e-12
    assert abs(math.sqrt(1 / 5) - 0.447) < 5e-4
    diagonal = SNEMI3D.tube_width_factor(o, np.array([[70.0, 0.0, 70.0]]), "index")[0]      # 45 degrees between x and z
    assert abs(diagonal - 0.527) < 5e-4
    assert math.sqrt(1 / 5) > 1 / 5                 # along x: the round tube is higher (z) than the exact image, r / 5
    zero = SNEMI3D.tube_width_factor(o, o, "index")[0]                             # an edge of length zero
    assert abs(zero - (1 / 5) ** (1 / 3)) < 1e-12
    assert np.array_equal(FIB25.tube_width_factor(o, along["x"], "index"), [1.0])  # isotropic: nothing changes
    # The drawn tube keeps the volume of the image of the true tube, for any direction and any
    # voxel size: (width factor)^2 * (length as drawn) = det(A) * (true length).
    for size_zyx in ((30.0, 6.0, 6.0), (30.0, 8.0, 6.0), (40.0, 4.0, 4.0), (8.0, 8.0, 8.0)):
        other = VoxelGrid((4, 4, 4), size_zyx)
        a = other.space_scale("index")
        rng = np.random.default_rng(2)
        p0, p1 = rng.normal(size=(50, 3)) * 100.0, rng.normal(size=(50, 3)) * 100.0
        factor = other.tube_width_factor(p0, p1, "index")
        true_length, drawn_length = np.linalg.norm(p1 - p0, axis=1), np.linalg.norm((p1 - p0) * a, axis=1)
        assert np.allclose(factor ** 2 * drawn_length, np.prod(a) * true_length, rtol=1e-12)
        assert np.array_equal(other.tube_width_factor(p0, p1, "physical"), np.ones(50))
    # The same by brute force, without the formula: fill a tube of radius 1 with points, move them
    # with A, and measure the area of the cross-section perpendicular to the moved axis.
    a = GRID.space_scale("index")                                                  # (1, 0.75, 0.2)
    rng = np.random.default_rng(5)
    for t in ((1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.2, 0.1, 1.0)):
        t = np.array(t) / np.linalg.norm(t)
        u = np.cross(t, [0.3, 0.5, 0.7])
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        n = 200000
        r, phi, s = np.sqrt(rng.random(n)), rng.random(n) * 2.0 * np.pi, rng.random(n) * 10.0
        moved = (s[:, None] * t + (r * np.cos(phi))[:, None] * u + (r * np.sin(phi))[:, None] * v) * a
        axis = a * t / np.linalg.norm(a * t)
        e1 = np.cross(axis, [0.3, 0.5, 0.7])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        section = np.stack([moved @ e1, moved @ e2])               # points of a uniformly filled ellipse
        area_ratio = 4.0 * math.sqrt(np.linalg.det(np.cov(section)))          # its area / pi
        factor = GRID.tube_width_factor(o, t[None], "index")[0]
        assert abs(factor ** 2 - area_ratio) < 0.01 * area_ratio, (t, factor ** 2, area_ratio)


# ---- the deliberately wrong grids ----------------------------------------------------------------
def test_as_cubes_is_the_grid_of_a_program_that_ignores_the_voxel_size():
    cubes = GRID.as_cubes()
    assert cubes.shape_zyx == GRID.shape_zyx and cubes.voxel_size_zyx_nm == (6.0, 6.0, 6.0)
    assert np.array_equal(cubes.extent_xyz_nm, GRID.extent_nm("index"))            # index space, as a grid of its own
    assert FIB25.as_cubes() == FIB25


def test_strided_grid_has_fewer_thicker_voxels():
    assert FIB25.strided(4).shape_zyx == (96, 384, 384) and FIB25.strided(4).voxel_size_zyx_nm == (32.0, 8.0, 8.0)
    assert FIB25.strided(4).anisotropy == 4.0
    assert GRID.strided(2).shape_zyx == (3, 7, 11)                                 # len(range(0, 5, 2)): array[::2]
    assert GRID.strided(3, axis=2).shape_zyx == (5, 7, 4) and GRID.strided(3, axis=2).voxel_size_zyx_nm == (30.0, 8.0, 18.0)


# ---- the containers --------------------------------------------------------------------------------
def test_mesh_area_volume_and_edges():
    # a tetrahedron with three edges of 6, 8 and 30 nm along x, y, z, wound outward
    v = np.array([(0.0, 0.0, 0.0), (6.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 30.0)])
    mesh = Mesh(v, np.array([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)], np.int32))
    assert math.isclose(mesh.volume_nm3(), 6.0 * 8.0 * 30.0 / 6.0)                 # positive: outward winding
    slanted = 0.5 * np.linalg.norm(np.cross(v[2] - v[1], v[3] - v[1]))
    assert math.isclose(mesh.area_nm2(), 0.5 * (6 * 8 + 6 * 30 + 8 * 30) + slanted)
    assert len(mesh.edges()) == 6 and mesh.edges().dtype == np.int32
    inside_out = Mesh(v, mesh.faces[:, ::-1])
    assert math.isclose(inside_out.volume_nm3(), -240.0)


def test_skeleton_graph():
    # a Y: 0 - 1 - 2 with a branch 1 - 3; steps of 6 nm (x), 8 nm (y) and 30 nm (z)
    v = np.array([(0.0, 0.0, 0.0), (6.0, 0.0, 0.0), (6.0, 8.0, 0.0), (6.0, 0.0, 30.0)])
    skeleton = Skeleton(v, np.array([(0, 1), (1, 2), (1, 3)], np.int32), np.array([3.0, 4.0, 3.0, 3.0]))
    assert skeleton.cable_length_nm() == 44.0
    assert list(skeleton.degrees()) == [1, 3, 1, 1]
    assert skeleton.n_branch_points() == 1 and skeleton.n_end_points() == 3
    chains = sorted(tuple(int(i) for i in chain) for chain in skeleton.polylines())
    assert sorted(tuple(sorted(c)) for c in chains) == [(0, 1), (1, 2), (1, 3)]    # three chains, split at the branch point
    assert Skeleton(v, np.zeros((0, 2), np.int32), np.ones(4)).cable_length_nm() == 0.0


def test_gaussian_principal_axes_and_quaternions():
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(8, 3, 3)))
    q[np.linalg.det(q) < 0, :, 0] *= -1.0                                          # proper rotations
    sigma = np.array([3.0, 4.0, 15.0])                                             # nm
    cov = np.einsum("nab,b,ncb->nac", q, sigma ** 2, q)
    gaussians = Gaussians(np.zeros((8, 3)), cov, np.full(8, 1 / 8))
    rotations, sigmas = gaussians.principal_axes()
    assert np.allclose(sigmas, [3.0, 4.0, 15.0]) and np.allclose(np.linalg.det(rotations), 1.0)
    assert np.allclose(np.einsum("nab,nb,ncb->nac", rotations, sigmas ** 2, rotations), cov)   # cov = R diag(sigma^2) R^T
    assert np.allclose(gaussians.elongation(), 5.0) and len(gaussians) == 8
    w, x, y, z = rotation_to_quaternion_wxyz(rotations).T
    back = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)], axis=-2)
    assert np.allclose(back, rotations) and np.all(w >= 0) and np.allclose(w * w + x * x + y * y + z * z, 1.0)


if __name__ == "__main__":
    failed = []
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"PASS  {name}")
            except Exception:
                traceback.print_exc()
                print(f"FAIL  {name}")
                failed.append(name)
    print(f"\n{len(failed)} failed" if failed else "\nall tests passed")
    sys.exit(1 if failed else 0)
