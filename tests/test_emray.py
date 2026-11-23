"""Tests of ncntm.viz.emray (the Warp ray caster) and ncntm.viz.figure (captions, montages, scale bars).

    python -m pytest tests/test_emray.py                       # device: NCNTM_DEVICE, else Warp's preferred one
    NCNTM_DEVICE=cpu python -m pytest tests/test_emray.py      # the same tests on Warp's CPU device

Every expected value is computed here with numpy from an independent model: float64 pixel-centre
rays (``my_rays``), ray/plane and ray/sphere intersections, and plain array indexing.  The volumes
are small random arrays, so that every voxel differs from its neighbours and a shift by one voxel
or by half a voxel cannot go unnoticed.  Scene unit = 1 micrometre; the test voxel is the SNEMI3D
one, 6 x 6 x 30 nm.  Section 16 (the independent review) repeats the essential statements on an
asymmetric phantom of 11 x 7 x 5 voxels of 6 x 8 x 30 nm, where x and y differ as well.
"""
import io
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

wp.config.quiet = True

from ncntm.viz import emray as E  # noqa: E402
from ncntm.viz import figure as F  # noqa: E402
from ncntm.viz.grid import NM_PER_UNIT, SPACES, VoxelGrid  # noqa: E402
from ncntm.viz.types import Bricks, Gaussians, Mesh, Skeleton  # noqa: E402

DEVICE = E.default_device()
ON_CPU = DEVICE == "cpu"
needs_cuda = pytest.mark.skipif(ON_CPU, reason="needs a CUDA device (this run uses the CPU device)")
SAMPLES = Path(os.environ.get("NCNTM_SAMPLES", REPO / "samples"))

RNG = np.random.default_rng(7)
EM = RNG.integers(0, 256, (12, 20, 28), dtype=np.uint8)            # rawvalue [z, y, x]
NZ, NY, NX = EM.shape
N_XYZ = np.array([NX, NY, NZ])
VS = (0.030, 0.006, 0.006)                                         # voxel size (dz, dy, dx) in micrometres
D_XYZ = np.array(VS[::-1])                                         # (dx, dy, dz)
GRID = VoxelGrid(EM.shape, (30.0, 6.0, 6.0))                       # the same voxel, in nanometres
LAB = (RNG.integers(1, 40, EM.shape) * (RNG.random(EM.shape) > 0.2)).astype(np.uint16)     # instance ids, 0 = background
SLICES = [("z", 5), ("y", 13), ("x", 21)]


def render(scene, cam, **options):
    return scene.render(cam, device=DEVICE, **options)


def my_rays(eye, target, up, W, H, mode, fov=None, ortho_h=None):
    """Independent camera model: float64 origin and unit direction of every pixel-centre ray.
    Right-handed, image x = forward x up, image row 0 = top."""
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    X, Y = np.meshgrid((np.arange(W) + 0.5) / W - 0.5, 0.5 - (np.arange(H) + 0.5) / H)
    if mode == "perspective":
        s = 2 * math.tan(math.radians(fov) / 2)
        d = f + X[..., None] * (s * W / H) * r + Y[..., None] * s * u
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return np.broadcast_to(eye, d.shape).copy(), d
    o = eye + X[..., None] * (ortho_h * W / H) * r + Y[..., None] * ortho_h * u
    return o, np.broadcast_to(f, o.shape).copy()


def rays(cam):
    return my_rays(cam.eye, cam.target, cam.up, cam.width, cam.height, cam.mode, cam.fov, cam.ortho_height)


def numpy_slice(vol, axis, index):
    return vol[index] if axis == "z" else (vol[:, index, :] if axis == "y" else vol[:, :, index])


def upsample(sl, shape):
    """Every voxel of a slice as a block of pixels, to the image ``shape`` (H, W)."""
    assert shape[0] % sl.shape[0] == 0 and shape[1] % sl.shape[1] == 0
    return np.repeat(np.repeat(sl, shape[0] // sl.shape[0], axis=0), shape[1] // sl.shape[1], axis=1)


def rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = math.radians(deg)
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


# =================================================================================================
# 1. device handling
# =================================================================================================
def test_default_device_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("NCNTM_DEVICE", "cpu")
    assert E.default_device() == "cpu"
    monkeypatch.setenv("NCNTM_DEVICE", "no-such-device")
    with pytest.raises(Exception):
        E.default_device()
    monkeypatch.delenv("NCNTM_DEVICE")
    monkeypatch.setattr(wp, "get_preferred_device", lambda: wp.get_device("cpu"))
    assert E.default_device() == "cpu"                     # no variable: whatever Warp prefers


def test_no_device_name_is_hard_coded():
    source = Path(E.__file__).read_text()
    assert "cuda:0" not in source and "/home/" not in source
    assert "/home/" not in Path(F.__file__).read_text() and ".ttf" not in Path(F.__file__).read_text()


def test_scene_renders_on_the_default_device_and_reports_times():
    assert E.load_kernels(DEVICE) >= 0.0
    s = E.Scene()
    s.add_spheres([[0, 0, 0]], 1.0)
    cam = E.Camera(width=32, height=24, ssaa=1)
    img = s.render(cam)                                    # no device given
    assert s._gpu["device"] == DEVICE
    assert img.rgba.shape == (24, 32, 4) and img.depth.shape == (24, 32) and img.object_id.shape == (24, 32)
    assert img.rgba.dtype == np.uint8 and img.depth.dtype == np.float32 and img.object_id.dtype == np.int32
    assert 0.0 < img.seconds <= img.wall_seconds           # kernel time is part of the wall time
    img2, seconds = s.render_timed(cam, repeat=3, device=DEVICE)
    assert np.array_equal(img2.rgba, img.rgba)
    assert seconds >= img2.seconds > 0.0                   # wall time of a frame includes the read-back


# =================================================================================================
# 2. depth against an analytic plane and an analytic sphere; smooth normals
# =================================================================================================
@pytest.mark.parametrize("mode", ["perspective", "orthographic"])
def test_depth_of_a_plane(mode):
    cam = E.Camera(eye=(0.7, -3.0, 2.0), target=(0, 0, 0), mode=mode, fov=40, ortho_height=3.0, width=512, height=512, ssaa=1)
    o, d = rays(cam)
    s = E.Scene()
    s.add_quads([[(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)]], object_id=1)
    img = render(s, cam)
    t = -o[..., 2] / d[..., 2]
    assert np.all(np.isfinite(img.depth)) and np.all(img.object_id == 1)
    assert np.abs(img.depth - t).max() < 2e-5


@pytest.mark.parametrize("mode", ["perspective", "orthographic"])
@pytest.mark.parametrize("subdiv", [3, 5, 7])
def test_depth_of_a_sphere_lies_in_the_analytic_bracket(mode, subdiv):
    """The icosphere is inscribed in the unit sphere and contains the sphere of radius rin (the
    smallest distance of a face plane from the centre): the rendered depth must lie between the
    analytic depths of those two spheres."""
    cam = E.Camera(eye=(0.7, -3.0, 2.0), target=(0, 0, 0), mode=mode, fov=40, ortho_height=3.0, width=512, height=512, ssaa=1)
    o, d = rays(cam)
    c = np.array([0.1, 0.2, 0.3])
    s = E.Scene()
    assert s.add_spheres([c], 1.0, subdiv=subdiv, object_id=1) == 20 * 4 ** subdiv
    img = render(s, cam)
    sv, sf = E.icosphere(subdiv)
    fn = np.cross(sv[sf[:, 1]] - sv[sf[:, 0]], sv[sf[:, 2]] - sv[sf[:, 0]])
    rin = float(np.abs((fn / np.linalg.norm(fn, axis=1, keepdims=True) * sv[sf[:, 0]]).sum(1)).min())
    oc = o - c
    b, cc = (oc * d).sum(-1), (oc * oc).sum(-1)
    t_out = -b - np.sqrt(np.maximum(b * b - (cc - 1.0), 0))
    t_in = -b - np.sqrt(np.maximum(b * b - (cc - rin * rin), 0))
    inner = (b * b - (cc - rin * rin)) > 0                      # rays that hit the inner sphere
    assert inner.sum() > 20000 and np.all(np.isfinite(img.depth[inner]))
    assert np.maximum(t_out - img.depth, img.depth - t_in)[inner].max() < 2e-5
    assert not np.any(np.isfinite(img.depth) & ((b * b - (cc - 1.0)) <= 0))     # nothing outside the unit sphere


def test_smooth_normals_give_analytic_lambert_on_a_sphere():
    cam = E.Camera(eye=(0.7, -3.0, 2.0), target=(0, 0, 0), mode="orthographic", ortho_height=2.4, width=512, height=512,
                   ssaa=1, background=(0, 0, 0))
    o, d = rays(cam)
    s = E.Scene()
    s.add_spheres([[0, 0, 0]], 1.0, subdiv=5, color=(1, 1, 1))
    img = render(s, cam, ambient=0.2)
    b = (o * d).sum(-1)
    disc = b * b - ((o * o).sum(-1) - 1.0)
    lam = 0.2 + 0.8 * np.sqrt(np.maximum(disc, 0))              # ambient + (1 - ambient) * |n . view|
    inner = np.isfinite(img.depth) & (disc > 0.05)
    assert np.abs(img.rgba_float[..., 0] - lam)[inner].max() < 0.01


# =================================================================================================
# 3. slices: face-on orthographic pictures must BE the numpy slices
# =================================================================================================
def center_slice_camera(scene, smp, axis, n, ssaa):
    """Hand-made orthographic camera facing a slice so that image rows/columns follow numpy
    rows/columns, with n pixels per voxel along the finest in-plane axis."""
    lo, hi = scene.volume_corners(smp)
    c, ext = 0.5 * (lo + hi), hi - lo
    vs = scene.samplers[smp]["vsize"]
    if axis == "z":      # rows = y, columns = x  -> look along +z, up = -y
        eye, up, w, h, px = c - [0, 0, 2 * ext[2] + 1], (0, -1, 0), ext[0], ext[1], vs[0] / n
    elif axis == "y":    # rows = z, columns = x  -> look along -y, up = -z
        eye, up, w, h, px = c + [0, 2 * ext[1] + 1, 0], (0, 0, -1), ext[0], ext[2], vs[0] / n
    else:                # rows = z, columns = y  -> look along +x, up = -z
        eye, up, w, h, px = c - [2 * ext[0] + 1, 0, 0], (0, 0, -1), ext[1], ext[2], vs[1] / n
    return E.Camera(eye=tuple(eye), target=tuple(c), up=up, mode="orthographic", ortho_height=h,
                    width=int(round(w / px)), height=int(round(h / px)), ssaa=ssaa, background=(1, 0, 1))


@pytest.mark.parametrize("voxel_size", [(1, 1, 1), VS], ids=["unit_voxels", "6x6x30nm"])
@pytest.mark.parametrize("axis,index", SLICES)
@pytest.mark.parametrize("ssaa", [1, 2])
def test_slice_exact_center_convention(voxel_size, axis, index, ssaa):
    s = E.Scene()
    smp = s.add_volume(EM, voxel_size=voxel_size)               # default convention: voxel centres on integers
    s.add_slice(smp, axis, index, object_id=7)
    lo, hi = s.volume_corners(smp)
    size = np.array(voxel_size[::-1], float)
    assert np.allclose(lo, -0.5 * size) and np.allclose(hi, (N_XYZ - 0.5) * size)
    img = render(s, center_slice_camera(s, smp, axis, 2, ssaa))
    exp = upsample(numpy_slice(EM, axis, index), img.rgba.shape)
    assert np.array_equal(img.rgba[..., :3], np.repeat(exp[..., None], 3, axis=2))
    assert np.all(img.rgba[..., 3] == 255) and np.all(img.object_id == 7)


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("axis,index", SLICES)
@pytest.mark.parametrize("orientation", ["numpy", "physical"])
def test_grid_slice_is_the_numpy_slice(space, axis, index, orientation):
    """Corner convention, both display spaces, all three axes: the face-on picture equals the
    numpy slice ("numpy" orientation, row 0 at the top) or its up-down flip ("physical": z up)."""
    s = E.Scene()
    smp = s.add_volume_grid(EM, GRID, space)
    s.add_slice(smp, axis, index, object_id=7)
    d = GRID.display_voxel_size_xyz_nm(space)
    a = "xyz".index(axis)
    u, v = (1 if a == 0 else 0), (1 if a == 2 else 2)           # world axes along image columns and rows
    px_nm = min(d[u], d[v]) / 2
    sl = numpy_slice(EM, axis, index)
    for ssaa in (1, 2):
        cam = E.Camera.for_slice(GRID, axis, space, orientation, pixels_per_voxel=2, ssaa=ssaa, background=(1, 0, 1))
        assert cam.mode == "orthographic" and math.isclose(cam.pixel_size(), px_nm / NM_PER_UNIT)
        assert (cam.height, cam.width) == (round(N_XYZ[v] * d[v] / px_nm), round(N_XYZ[u] * d[u] / px_nm))
        img = render(s, cam)
        exp = upsample(sl if orientation == "numpy" else sl[::-1], img.rgba.shape)
        assert np.array_equal(img.rgba[..., :3], np.repeat(exp[..., None], 3, axis=2))
        assert np.all(img.rgba[..., 3] == 255) and np.all(img.object_id == 7)
    if space == "physical" and axis != "z":                     # a voxel is drawn 2 px wide and 10 px tall: 6 x 30 nm
        assert img.rgba.shape[0] // sl.shape[0] == 10 and img.rgba.shape[1] // sl.shape[1] == 2
    else:                                                       # square in the xy plane and everywhere in index space
        assert img.rgba.shape[0] // sl.shape[0] == 2 and img.rgba.shape[1] // sl.shape[1] == 2


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("axis,index", SLICES)
def test_grid_slice_quad_spans_0_to_n_d_through_the_layer_centre(space, axis, index):
    s = E.Scene()
    smp = s.add_volume_grid(EM, GRID, space)
    s.add_slice(smp, axis, index, object_id=7)
    d = GRID.display_voxel_size_xyz_nm(space) / NM_PER_UNIT    # voxel as drawn, micrometres
    a = "xyz".index(axis)
    exp_lo, exp_hi = np.zeros(3), N_XYZ * d
    exp_lo[a] = exp_hi[a] = (index + 0.5) * d[a]                # the plane passes through the centre of layer `index`
    lo, hi = s.bounds()
    assert np.allclose(lo, exp_lo, rtol=0, atol=1e-12) and np.allclose(hi, exp_hi, rtol=0, atol=1e-12)
    vlo, vhi = s.volume_corners(smp)
    assert np.allclose(vlo, 0.0, atol=1e-15) and np.allclose(vhi, GRID.extent_nm(space) / NM_PER_UNIT, rtol=1e-12)
    # rendered: with a border of 3 background pixels the quad covers exactly the inner rectangle
    cam = E.Camera.for_slice(GRID, axis, space, pixels_per_voxel=2, border=3, background=(1, 0, 1))
    img = render(s, cam)
    expect = np.zeros(img.object_id.shape, bool)
    expect[3:-3, 3:-3] = True
    assert np.array_equal(img.object_id == 7, expect)
    assert np.all(img.rgba[~expect] == (255, 0, 255, 255))


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("orientation", ["numpy", "physical"])
def test_grid_slices_with_three_different_voxel_edges(space, orientation):
    """dx, dy and dz all different (4, 8, 40 nm): an x/y mix-up of sizes or of counts would show."""
    vol = np.random.default_rng(5).integers(0, 256, (5, 6, 7), dtype=np.uint8)
    grid = VoxelGrid(vol.shape, (40.0, 8.0, 4.0))               # (dz, dy, dx)
    d = grid.display_voxel_size_xyz_nm(space)                   # physical: (4, 8, 40); index: (4, 4, 4)
    assert tuple(d) == ((4.0, 8.0, 40.0) if space == "physical" else (4.0, 4.0, 4.0))
    s = E.Scene()
    smp = s.add_volume_grid(vol, grid, space)
    assert np.allclose(s.volume_corners(smp)[1] * NM_PER_UNIT, (7, 6, 5) * d)
    for axis, index, (u, v) in (("z", 2, (0, 1)), ("y", 3, (0, 2)), ("x", 4, (1, 2))):   # world axes of columns, rows
        s.clear_geometry()
        s.add_slice(smp, axis, index)
        cam = E.Camera.for_slice(grid, axis, space, orientation)
        img = render(s, cam)
        px = min(d[u], d[v])                                    # nm per pixel
        sl = numpy_slice(vol, axis, index)
        assert img.rgba.shape[:2] == (sl.shape[0] * d[v] / px, sl.shape[1] * d[u] / px) and math.isclose(cam.pixel_size() * 1000, px)
        exp = np.repeat(np.repeat(sl if orientation == "numpy" else sl[::-1], int(d[v] / px), axis=0), int(d[u] / px), axis=1)
        assert np.array_equal(img.rgba[..., 0], exp)
    m = np.zeros(vol.shape, bool)
    m[1, 2, 3] = True                                           # voxel [k=1, j=2, i=3]
    s = E.Scene()
    s.add_bricks(bricks_of(m, grid), grid, space)
    assert np.allclose(s.bounds()[0] * NM_PER_UNIT, (3, 2, 1) * d) and np.allclose(s.bounds()[1] * NM_PER_UNIT, (4, 3, 2) * d)
    s = E.Scene()
    s.add_voxels(m, d[::-1] / NM_PER_UNIT, convention="corner")
    assert np.allclose(s.bounds()[0] * NM_PER_UNIT, (3, 2, 1) * d) and np.allclose(s.bounds()[1] * NM_PER_UNIT, (4, 3, 2) * d)
    s = E.Scene()
    s.add_voxels(m, d[::-1] / NM_PER_UNIT)                      # centre convention: half a voxel lower on every axis
    assert np.allclose(s.bounds()[0] * NM_PER_UNIT, (2.5, 1.5, 0.5) * d)


@pytest.mark.parametrize("space", SPACES)
def test_full_size_volume_slices_are_exact(space):
    """100 x 512 x 512 voxels of 6 x 6 x 30 nm (the size of the sample crops, about 3 um across),
    2 pixels per voxel: up to 1024 x 1024 pixels, every one equal to the numpy slice."""
    big = np.random.default_rng(11).integers(0, 256, (100, 512, 512), dtype=np.uint8)
    grid = VoxelGrid(big.shape, (30.0, 6.0, 6.0))
    s = E.Scene()
    smp = s.add_volume_grid(big, grid, space)
    for axis, index in (("z", 37), ("y", 200), ("x", 300)):
        s.clear_geometry()
        s.add_slice(smp, axis, index)
        img = render(s, E.Camera.for_slice(grid, axis, space, pixels_per_voxel=2))
        assert np.array_equal(img.rgba[..., 0], upsample(numpy_slice(big, axis, index), img.rgba.shape))
    assert img.rgba.shape[:2] == ((1000, 1024) if space == "physical" else (200, 1024))      # the x slice: rows z, columns y


def test_center_and_corner_conventions_differ_by_half_a_voxel():
    s = E.Scene()
    centre = s.add_volume(EM, voxel_size=VS, origin=(1.0, 2.0, 3.0))
    corner = s.add_volume(EM, voxel_size=VS, origin=(1.0, 2.0, 3.0), convention="corner")
    assert np.allclose(s.volume_corners(corner)[0], (1.0, 2.0, 3.0))
    assert np.allclose(s.volume_corners(corner)[0] - s.volume_corners(centre)[0], 0.5 * D_XYZ)   # 3, 3, 15 nm
    assert np.allclose(s.volume_corners(corner)[1] - s.volume_corners(corner)[0], N_XYZ * D_XYZ)
    with pytest.raises(ValueError, match="convention"):
        s.add_volume(EM, voxel_size=VS, convention="middle")


def test_grid_volume_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        E.Scene().add_volume_grid(EM[:, :, :-1], GRID)
    with pytest.raises(ValueError, match="space"):
        E.Scene().add_volume_grid(EM, GRID, space="voxel")


def test_slice_through_a_lookup_table_is_exact():
    lut = E.lut_window_level(level=0.55, window=0.5)
    s = E.Scene()
    smp = s.add_volume_grid(EM, GRID, "physical", lut=lut)
    s.add_slice(smp, "y", 13)
    img = render(s, E.Camera.for_slice(GRID, "y", "physical", pixels_per_voxel=2, ssaa=2))
    exp = np.floor(upsample(lut[EM[:, 13, :]], img.rgba.shape) * 255 + 0.5).astype(np.uint8)
    assert np.array_equal(img.rgba, exp)


def test_lookup_tables():
    g = np.arange(256) / 255.0
    assert np.allclose(E.lut_grey(), np.stack([g, g, g, np.ones(256)], 1))
    n = E.lut_normalised(20, 220)                               # voxvalue = (raw - min) / (max - min), clamped
    assert n.shape == (256, 4) and np.allclose(n[:, 0], np.clip((np.arange(256) - 20) / 200.0, 0, 1)) and np.all(n[:, 3] == 1)
    assert n[20, 0] == 0.0 and n[220, 0] == 1.0 and np.allclose(n[120, :3], 0.5)
    with pytest.raises(ValueError):
        E.lut_normalised(10, 10)
    w = E.lut_window_level(level=0.5, window=0.5, lo=(0, 0, 1), hi=(1, 0, 0))
    assert np.allclose(w[0], (0, 0, 1, 1)) and np.allclose(w[255], (1, 0, 0, 1))
    assert np.allclose(w[:, 0], np.clip((g - 0.25) / 0.5, 0, 1), atol=1e-6)
    m = E.lut_membrane(dark=0.35, soft=0.12, opacity=0.9)
    assert math.isclose(m[0, 3], 0.9, rel_tol=1e-6) and m[255, 3] == 0.0 and np.all(np.diff(m[:, 3]) <= 1e-7)


def test_voxvalue_slice_through_lut_normalised():
    """voxvalue = (raw - min) / (max - min) drawn as grey, straight from the stored uint8 array."""
    lut = E.lut_normalised(30, 200)
    s = E.Scene()
    s.add_slice(s.add_volume_grid(EM, GRID, "physical", lut=lut), "z", 5)
    img = render(s, E.Camera.for_slice(GRID, "z", "physical"))
    vox = np.clip((EM[5].astype(np.float32) - 30) / 170.0, 0, 1)
    assert np.abs(img.rgba_float[..., 0] - vox).max() < 1e-6


def test_label_slice_matches_the_numpy_twin_of_the_hash():
    lab32 = LAB.astype(np.int32)
    for labels in (lab32, LAB):                                 # int32 and compact uint16 storage give the same picture
        s = E.Scene()
        smp = s.add_volume_grid(labels, GRID, "physical", labels=True, zero_alpha=1.0)
        s.add_slice(smp, "y", 13)
        img = render(s, E.Camera.for_slice(GRID, "y", "physical", pixels_per_voxel=2, ssaa=2))
        ids = upsample(lab32[:, 13, :], img.rgba.shape)
        exp = np.where((ids == 0)[..., None], 0.06, E.label_colors(ids))
        assert len(np.unique(ids)) > 20
        assert np.abs(img.rgba_float[..., :3] - exp).max() < 1e-5


def test_label_focus_and_transparent_background_id():
    s = E.Scene()
    smp = s.add_volume_grid(LAB, GRID, "index", labels=True, focus=7)       # id 0 transparent, only id 7 keeps its colour
    s.add_slice(smp, "z", 5)
    img = render(s, E.Camera.for_slice(GRID, "z", "index", background=(1, 0, 1)))
    ids = LAB[5]
    assert np.all(img.rgba[ids == 0] == (255, 0, 255, 255)) and np.all(img.object_id[ids == 0] == -1)
    assert np.abs(img.rgba_float[ids == 7][:, :3] - E.label_colors(7)).max() < 1e-5
    rest = (ids != 0) & (ids != 7)                              # every other id: 85 % grey + 15 % of its own colour
    rgb = E.label_colors(ids[rest])
    grey = 0.55 + 0.25 * rgb.mean(1, keepdims=True)
    assert np.abs(img.rgba_float[rest][:, :3] - (0.85 * grey + 0.15 * rgb)).max() < 1e-5


def test_float_volume_in_0_1_equals_the_uint8_slice():
    vox = EM.astype(np.float32) / 255.0                         # voxvalue as floats in [0, 1]
    s = E.Scene()
    s.add_slice(s.add_volume_grid(vox, GRID, "physical"), "z", 5)
    img = render(s, E.Camera.for_slice(GRID, "z", "physical"))
    assert np.array_equal(img.rgba[..., 0], EM[5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        E.Scene().add_volume(EM.astype(np.float32))             # not normalised
    with pytest.raises(ValueError):
        E.Scene().add_volume(np.full((2, 2, 2), np.nan, np.float32))
    with pytest.raises(TypeError):
        E.Scene().add_volume(np.zeros((2, 2, 2), complex))
    with pytest.raises(TypeError):
        E.Scene().add_volume(vox, labels=True)                  # labels must be integers


@pytest.mark.parametrize("convention", E.CONVENTIONS)
@pytest.mark.parametrize("mode", ["perspective", "orthographic"])
@pytest.mark.parametrize("axis,idx", SLICES)
def test_oblique_view_of_a_slice_shows_the_right_voxels(convention, mode, axis, idx):
    """Axis-aligned slices seen OBLIQUELY, non-square image, non-zero origin: every pixel shows
    EM[k, j, i] of the voxel whose box contains the ray/plane intersection."""
    org = np.array([0.10, -0.20, 0.05])
    half = 0.5 if convention == "center" else 0.0               # the low face of voxel 0 is half a voxel below a centre origin
    a = "xyz".index(axis)
    s = E.Scene()
    smp = s.add_volume(EM, voxel_size=VS, origin=tuple(org), convention=convention)
    s.add_slice(smp, axis, idx, object_id=11)
    c = org + (0.5 * N_XYZ - half) * D_XYZ
    eye = c + 0.12 * np.array([2.9, -4.1, 3.3])
    kw = dict(fov=28.0) if mode == "perspective" else dict(ortho_height=0.40)
    cam = E.Camera(eye=tuple(eye), target=tuple(c), up=(0, 0, 1), mode=mode, width=320, height=240, ssaa=1, background=(1, 0, 1), **kw)
    img = render(s, cam)
    o, d = rays(cam)
    t = (org[a] + (idx + 0.5 - half) * D_XYZ[a] - o[..., a]) / d[..., a]
    q = (o + d * t[..., None] - org) / D_XYZ + half             # continuous index, voxel [k,j,i] = cube [i,i+1] x [j,j+1] x [k,k+1]
    q[..., a] = idx + 0.5
    inside = (t > 0) & np.all(q >= 0, -1) & np.all(q <= N_XYZ, -1)
    ijk = np.clip(np.floor(q).astype(int), 0, N_XYZ - 1)
    exp = np.where(inside, EM[ijk[..., 2], ijk[..., 1], ijk[..., 0]].astype(np.int64), -1)
    fr = np.abs(q - np.round(q))                                # distance to a voxel boundary, in voxels
    fr[..., a] = 1.0
    sure = fr.min(-1) > 2e-3
    got = np.where(img.object_id == 11, img.rgba[..., 0].astype(int), -1)
    assert inside.sum() > 3000
    assert int(((got != exp) & sure).sum()) == 0
    grey = (img.rgba[..., 0] == img.rgba[..., 1]) & (img.rgba[..., 1] == img.rgba[..., 2])
    assert np.all(grey | (img.object_id != 11))
    assert np.abs(img.depth[inside & sure] - t[inside & sure]).max() < 2e-6


def test_slice_orientation_against_numpy():
    """Which viewing direction reproduces EM[k] as numpy / imshow shows it?  Only looking along +z
    with y pointing down; from above with y up it is the up-down flip.  Never a mirror image."""
    k = 5
    s = E.Scene()
    smp = s.add_volume(EM, voxel_size=VS)
    s.add_slice(smp, "z", k)
    lo, hi = s.volume_corners(smp)
    c = 0.5 * (lo + hi)

    def view(eye, up):
        cam = E.Camera(eye=tuple(eye), target=tuple(c), up=up, mode="orthographic", ortho_height=float(hi[1] - lo[1]),
                       width=NX, height=NY, ssaa=1)
        return render(s, cam).rgba[..., 0]

    assert np.array_equal(view(c - [0, 0, 1], (0, -1, 0)), EM[k])                   # from below, y down
    assert np.array_equal(view(c + [0, 0, 1], (0, 1, 0)), EM[k][::-1])              # from above, y up
    assert np.array_equal(view(c + [0, 0, 1], (0, -1, 0)), EM[k][:, ::-1])          # from above, y down
    i2w = E.index_to_world(VS)
    assert np.allclose(np.diag(i2w)[:3], D_XYZ) and np.linalg.det(i2w[:3, :3]) > 0  # no mirroring in the matrix


def test_closeup_of_a_voxel_window_20_by_100_pixels():
    """A window of 12 x 2 voxels of the xz slice (centre convention, lo/hi restricted), z up:
    each 6 x 30 nm voxel is exactly 20 px wide and 100 px tall."""
    J, I0, K0, NI, NK = 10, 5, 4, 12, 2
    dz, dy, dx = VS
    s = E.Scene()
    smp = s.add_volume(EM, voxel_size=VS)
    s.add_slice(smp, "y", J, lo=(K0, J, I0), hi=(K0 + NK, J + 1, I0 + NI))
    h = 12.8 * dx                                               # 256 px for 12.8 voxels: 20 px per voxel along x
    cc = np.array([(I0 - 0.5) * dx + h / 2, J * dy, (K0 - 0.5) * dz + NK * dz / 2])
    cam = E.Camera(eye=tuple(cc - [0, 10 * h, 0]), target=tuple(cc), mode="orthographic", ortho_height=h, width=256, height=256,
                   ssaa=1, background=(1, 1, 1))
    g = render(s, cam).rgba[..., 0]
    exp = np.full((256, 256), 255, np.uint8)
    exp[28:228, 0:240] = np.repeat(np.repeat(EM[K0:K0 + NK, J, I0:I0 + NI][::-1], 100, axis=0), 20, axis=1)
    assert np.array_equal(g, exp)


@pytest.mark.parametrize("interp", ["nearest", "linear"])
def test_oblique_reslice_against_numpy_and_scipy(interp):
    """An arbitrary quad through the anisotropic volume, seen in perspective: nearest sampling
    against EM[containing voxel], trilinear against scipy.ndimage.map_coordinates."""
    ndimage = pytest.importorskip("scipy.ndimage")
    lo, hi = np.zeros(3), N_XYZ * D_XYZ                         # corner convention
    pc = 0.5 * (lo + hi)
    e1 = np.array([1.0, 0.35, 0.2])
    nrm = np.array([0.25, -0.45, 1.0])
    nrm /= np.linalg.norm(nrm)
    e2 = np.cross(nrm, e1)
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, nrm)
    R = 0.07
    quad = [pc + R * (sa * e1 + sb * e2) for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    eye = pc + nrm * 0.5 + np.array([0.06, -0.04, 0.0])
    s = E.Scene()
    smp = s.add_volume_grid(EM, GRID, "physical", interp=interp)
    s.add_quads([quad], sampler=smp, lit=False, color=(1, 1, 1, 1), object_id=3)
    cam = E.Camera(eye=tuple(eye), target=tuple(pc), up=(0, 1, 0), mode="perspective", fov=30, width=300, height=300, ssaa=1,
                   background=(1, 0, 1))
    img = render(s, cam)
    o, d = rays(cam)
    t = ((pc - o) @ nrm) / (d @ nrm)
    p = o + d * t[..., None]
    q = p / D_XYZ                                               # continuous index, corner convention
    inq = (np.abs((p - pc) @ e1) < 0.999 * R) & (np.abs((p - pc) @ e2) < 0.999 * R)
    inv = inq & np.all(q >= 0, -1) & np.all(q <= N_XYZ, -1)
    got = img.rgba_float[..., 0] * 255.0
    assert inv.sum() > 10000 and q[inv][:, 2].max() - q[inv][:, 2].min() > 2      # the quad crosses several z layers
    if interp == "nearest":
        ijk = np.clip(np.floor(q).astype(int), 0, N_XYZ - 1)
        exp = EM[ijk[..., 2], ijk[..., 1], ijk[..., 0]].astype(float)
        sure = np.abs(q - np.round(q)).min(-1) > 5e-3
        assert np.abs(got - exp)[inv & sure].max() < 1e-3
    else:                                                       # voxel centres are at +0.5: array coordinate = q - 0.5
        exp = ndimage.map_coordinates(EM.astype(np.float64), [q[..., 2] - 0.5, q[..., 1] - 0.5, q[..., 0] - 0.5], order=1, mode="nearest")
        assert np.abs(got - exp)[inv].max() < 0.25


@pytest.mark.parametrize("space", SPACES)
def test_three_orthogonal_slices_in_perspective(space):
    """The standard teaching picture -- three orthogonal slices in a 3D view -- checked pixel by
    pixel against numpy, in index space and in physical space."""
    D = GRID.display_voxel_size_xyz_nm(space) / NM_PER_UNIT
    planes = ((2, NZ // 2, 1), (1, NY // 2, 2), (0, NX // 2, 3))
    s = E.Scene()
    smp = s.add_volume_grid(EM, GRID, space)
    for a, idx, oid in planes:
        s.add_slice(smp, "xyz"[a], idx, object_id=oid)
    lo, hi = s.volume_corners(smp)
    cam = E.Camera.fit(lo, hi, direction=(1, -0.8, 0.55), fov=30, width=320, height=320, ssaa=1, background=(1, 0, 1))
    img = render(s, cam)
    o, d = rays(cam)
    shape = img.object_id.shape
    best_t, second_t = np.full(shape, np.inf), np.full(shape, np.inf)
    exp_v, exp_id, sure = np.full(shape, -1), np.full(shape, -1), np.ones(shape, bool)
    for a, idx, oid in planes:
        t = ((idx + 0.5) * D[a] - o[..., a]) / d[..., a]
        q = (o + d * t[..., None]) / D
        q[..., a] = idx + 0.5
        hit = (t > 0) & np.all(q >= 0, -1) & np.all(q <= N_XYZ, -1)
        ijk = np.clip(np.floor(q).astype(int), 0, N_XYZ - 1)
        fr = np.abs(q - np.round(q))
        fr[..., a] = 1.0
        second_t = np.where(hit, np.where(t < best_t, best_t, np.minimum(second_t, t)), second_t)
        ins = hit & (t < best_t)
        best_t[ins] = t[ins]
        exp_v[ins] = EM[ijk[..., 2], ijk[..., 1], ijk[..., 0]][ins]
        exp_id[ins] = oid
        sure[ins] = (fr.min(-1) > 2e-3)[ins]
    with np.errstate(invalid="ignore"):                                     # (inf - inf where no slice is hit)
        chk = sure & (exp_id >= 0) & (second_t - best_t > 1e-5 * best_t)    # not on a line where two slices cross
    got_v = np.where(img.object_id >= 1, img.rgba[..., 0].astype(int), -1)
    assert chk.sum() > 15000
    assert int(((got_v != exp_v) & chk).sum()) == 0 and int(((img.object_id != exp_id) & chk).sum()) == 0
    assert int(((img.object_id >= 1) & (exp_id < 0)).sum()) == 0            # no slice pixel where none is expected
    ratio = (hi - lo)[2] / (hi - lo)[0]                                     # the box is flat in index space, tall in physical space
    assert math.isclose(ratio, NZ / NX if space == "index" else NZ * 5 / NX, rel_tol=1e-9)


def test_voxel_boundary_lines_sit_on_integer_indices():
    s = E.Scene()
    smp = s.add_volume_grid(EM, GRID, "index", grid_lines=0.0012, grid_color=(1, 0, 0))   # 1.2 nm lines on a 6 nm voxel
    s.add_slice(smp, "z", 5)
    img = render(s, E.Camera.for_slice(GRID, "z", "index", pixels_per_voxel=10))           # 0.6 nm per pixel
    red = np.all(img.rgba[..., :3] == (255, 0, 0), axis=-1)
    col = np.arange(img.rgba.shape[1]) % 10
    row = np.arange(img.rgba.shape[0]) % 10
    expect = ((col == 0) | (col == 9))[None, :] | ((row == 0) | (row == 9))[:, None]      # 1 px on either side of a boundary
    assert np.array_equal(red, expect)


def test_clear_geometry_keeps_the_volumes_on_the_device():
    s = E.Scene()
    smp = s.add_volume_grid(EM, GRID)
    cam = E.Camera.for_slice(GRID, "z")
    s.add_slice(smp, "z", 2)
    assert np.array_equal(render(s, cam).rgba[..., 0], EM[2])
    stored = s._vol_gpu[1]
    s.clear_geometry()
    assert s.num_triangles == 0 and not np.all(np.isfinite(s.bounds()[0]))
    s.add_slice(smp, "z", 9)
    assert np.array_equal(render(s, cam).rgba[..., 0], EM[9])
    assert s._vol_gpu[1] is stored                              # not uploaded again


@pytest.mark.parametrize("name", ["snemi3d", "cremi_a", "fib25"])
def test_sample_file_slices_are_exact_in_both_spaces(name):
    """The same statement on a crop of a real sample file, with the voxel size read from the file."""
    h5py = pytest.importorskip("h5py")
    path = SAMPLES / f"{name}.h5"
    if not path.exists():
        pytest.skip(f"sample {path.name} not found in the samples directory")
    with h5py.File(path, "r") as fh:
        raw = fh["raw"][10:22, 30:50, 40:68]
        lab = fh["instance"][10:22, 30:50, 40:68]
        grid = VoxelGrid(raw.shape, tuple(fh.attrs["voxel_size_zyx_nm"]))
    assert raw.dtype == np.uint8
    for space in SPACES:
        s = E.Scene()
        smp = s.add_volume_grid(raw, grid, space)
        smp_lab = s.add_volume_grid(lab, grid, space, labels=True, zero_alpha=1.0)
        for axis, index in SLICES[:2]:
            s.clear_geometry()
            s.add_slice(smp, axis, index)
            cam = E.Camera.for_slice(grid, axis, space, pixels_per_voxel=1)
            img = render(s, cam)
            assert np.array_equal(img.rgba[..., 0], upsample(numpy_slice(raw, axis, index), img.rgba.shape))
            s.clear_geometry()
            s.add_slice(smp_lab, axis, index)
            ids = upsample(numpy_slice(lab, axis, index), img.rgba.shape).astype(np.int64)
            exp = np.where((ids == 0)[..., None], 0.06, E.label_colors(ids))
            assert np.abs(render(s, cam).rgba_float[..., :3] - exp).max() < 1e-5
        size = grid.display_voxel_size_xyz_nm(space)            # rows of the y slice: z; columns: x
        assert img.rgba.shape[:2] == (round(12 * size[2] / min(size[0], size[2])), round(28 * size[0] / min(size[0], size[2])))


# =================================================================================================
# 4. voxel bricks
# =================================================================================================
def test_one_voxel_brick_is_6_by_30_nm_center_convention():
    m = np.zeros((3, 3, 3), bool)
    m[1, 1, 1] = True
    s = E.Scene()
    assert s.add_voxels(m, voxel_size=VS, object_id=5) == 12    # one voxel: 6 x 6 x 30 nm, centre (6, 6, 30) nm
    cam = E.Camera(eye=(0.006, -1.0, 0.030), target=(0.006, 0.0, 0.030), mode="orthographic", ortho_height=0.064, width=640,
                   height=640, ssaa=1)
    img = render(s, cam)                                        # 10 px per nm
    ys, xs = np.nonzero(img.object_id == 5)
    assert (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) == (60, 300)
    assert len(xs) == 60 * 300                                  # a filled rectangle
    assert np.allclose([xs.mean() + 0.5, ys.mean() + 0.5], cam.project((0.006, 0, 0.030))[0])


@pytest.mark.parametrize("convention,lo_index", [("center", (4.5, 2.5, 1.5)), ("corner", (5, 3, 2))])
def test_single_brick_box(convention, lo_index):
    m = np.zeros((5, 7, 9), bool)
    m[2, 3, 5] = True
    s = E.Scene()
    s.add_voxels(m, VS, convention=convention)
    lo, hi = s.bounds()
    assert s.num_triangles == 12
    assert np.allclose(lo, np.array(lo_index) * D_XYZ, atol=1e-12) and np.allclose(hi, (np.array(lo_index) + 1) * D_XYZ, atol=1e-12)


def bricks_of(mask, grid):
    kji = np.argwhere(mask)
    return Bricks(kji=kji, centers_xyz_nm=grid.voxel_centers_nm(kji), size_xyz_nm=grid.voxel_size_xyz_nm, n_voxels_total=len(kji))


@pytest.mark.parametrize("space", SPACES)
def test_add_bricks_one_brick_is_exactly_the_voxel_box(space):
    """Brick [1,1,1] of a 6 x 6 x 30 nm grid at 1 nm per pixel: x in [6,12] nm, z in [30,60] nm in
    physical space (6 x 30 px); a 6 nm cube with z in [6,12] nm in index space (6 x 6 px)."""
    grid = VoxelGrid((3, 3, 3), (30.0, 6.0, 6.0))
    m = np.zeros((3, 3, 3), bool)
    m[1, 1, 1] = True
    s = E.Scene()
    assert s.add_bricks(bricks_of(m, grid), grid, space, object_id=5) == 12
    dz = 30.0 if space == "physical" else 6.0
    lo, hi = s.bounds()
    assert np.allclose(lo * 1000, (6, 6, dz), atol=1e-9) and np.allclose(hi * 1000, (12, 12, 2 * dz), atol=1e-9)
    cam = E.Camera.for_slice(grid, "y", space, orientation="physical", pixels_per_voxel=6)      # 1 nm per pixel, z up
    assert math.isclose(cam.pixel_size(), 0.001)
    img = render(s, cam)
    height = int(round(3 * dz))
    expect = np.zeros((height, 18), bool)
    expect[height - int(2 * dz):height - int(dz), 6:12] = True
    assert img.object_id.shape == expect.shape and np.array_equal(img.object_id == 5, expect)
    assert math.isclose(float(img.depth[expect].max()), float(img.depth[expect].min()), rel_tol=1e-6)   # a flat face


def blob_mask():
    """A lumpy 3D object: the union of a few random boxes."""
    rng = np.random.default_rng(3)
    m = np.zeros(EM.shape, bool)
    for _ in range(9):
        k, j, i = rng.integers(0, NZ - 3), rng.integers(0, NY - 5), rng.integers(0, NX - 7)
        m[k:k + rng.integers(1, 4), j:j + rng.integers(2, 6), i:i + rng.integers(2, 8)] = True
    return m


@pytest.mark.parametrize("entry", ["add_voxels_center", "add_voxels_corner", "add_bricks_physical", "add_bricks_index"])
def test_brick_silhouette_equals_the_mask_projection(entry):
    """Orthographic side view (along +y, z up): the silhouette is exactly mask.any(axis=1), every
    voxel a dx by dz rectangle at its place."""
    mask = blob_mask()
    D, half = D_XYZ, 0.0                                         # voxel as drawn; index offset of the low face
    s = E.Scene()
    if entry == "add_voxels_center":
        s.add_voxels(mask, VS, object_id=18)
        half = 0.5
    elif entry == "add_voxels_corner":
        s.add_voxels(mask, VS, convention="corner", object_id=18)
    else:
        space = entry.split("_")[-1]
        s.add_bricks(bricks_of(mask, GRID), GRID, space, object_id=18)
        D = GRID.display_voxel_size_xyz_nm(space) / NM_PER_UNIT
    cen = (0.5 * N_XYZ - half) * D
    oh = 1.3 * max(N_XYZ[0] * D[0], N_XYZ[2] * D[2])
    cam = E.Camera(eye=tuple(cen - [0, 1.0, 0]), target=tuple(cen), up=(0, 0, 1), mode="orthographic", ortho_height=oh,
                   width=400, height=400, ssaa=1)
    img = render(s, cam)
    o, _ = rays(cam)
    qi, qk = o[..., 0] / D[0] + half, o[..., 2] / D[2] + half
    ii, kk = np.floor(qi).astype(int), np.floor(qk).astype(int)
    ok = (ii >= 0) & (ii < NX) & (kk >= 0) & (kk < NZ)
    exp = np.zeros(ok.shape, bool)
    exp[ok] = mask.any(axis=1)[kk[ok], ii[ok]]
    sure = (np.abs(qi - np.round(qi)) > 2e-3) & (np.abs(qk - np.round(qk)) > 2e-3)
    assert exp.sum() > 3000
    assert int((((img.object_id == 18) != exp) & sure).sum()) == 0


def test_add_bricks_emits_only_exposed_faces_and_takes_per_brick_styles():
    m = np.zeros((4, 4, 6), bool)
    m[1:3, 1:3, 1:3] = True                                     # a 2 x 2 x 2 block: 24 exposed voxel faces
    m[1, 1, 4] = True                                           # and a lone voxel: 6 faces
    b = bricks_of(m, VoxelGrid(m.shape, (30.0, 6.0, 6.0)))
    grid = VoxelGrid(m.shape, (30.0, 6.0, 6.0))
    s = E.Scene()
    ids = 100 + np.arange(len(b))
    assert s.add_bricks(b, grid, object_id=ids, color=E.label_colors(ids)) == 2 * (24 + 6)
    cam = E.Camera.for_slice(grid, "z", "physical", pixels_per_voxel=4)        # numpy orientation, seen from -z
    img = render(s, cam)
    number = np.full(m.shape, -1)
    number[tuple(b.kji.T)] = ids
    first = np.where(m.any(0), number[m.argmax(0), np.arange(4)[:, None], np.arange(6)[None, :]], -1)   # lowest k along each column
    assert np.array_equal(img.object_id, upsample(first, img.object_id.shape))
    assert s.add_bricks(Bricks(b.kji[:1], b.centers_xyz_nm[:1], b.size_xyz_nm), grid) == 12
    with pytest.raises(ValueError, match="no bricks"):
        s.add_bricks(Bricks(b.kji[:0], b.centers_xyz_nm[:0], b.size_xyz_nm), grid)
    with pytest.raises(ValueError):
        s.add_bricks(Bricks(b.kji, b.centers_xyz_nm[:, :2], b.size_xyz_nm), grid)


def test_bricks_from_the_wrong_grid_show_the_anisotropy_mistake():
    """Bricks computed with grid.as_cubes() (a careless program: every voxel a 6 nm cube) are a
    slab 5 times too flat when they are placed in the true physical scene."""
    mask = np.ones((4, 4, 4), bool)
    grid = VoxelGrid(mask.shape, (30.0, 6.0, 6.0))
    right, wrong = E.Scene(), E.Scene()
    right.add_bricks(bricks_of(mask, grid), grid)
    wrong.add_bricks(bricks_of(mask, grid.as_cubes()), grid)
    assert np.allclose(right.bounds()[1], (0.024, 0.024, 0.120)) and np.allclose(wrong.bounds()[1], (0.024, 0.024, 0.024))
    assert math.isclose(right.bounds()[1][2] / wrong.bounds()[1][2], grid.anisotropy)


def test_outline_width_is_the_same_along_x_and_z_on_an_anisotropic_brick():
    m = np.zeros((3, 3, 3), bool)
    m[1, 1, 1] = True
    s = E.Scene()
    s.add_voxels(m, VS, color=(1, 1, 1), lit=False, wire_width=0.002, wire_color=(1, 0, 0))     # 2 nm lines: 1 nm inside each face
    cam = E.Camera(eye=(0.006, -1.0, 0.030), target=(0.006, 0.0, 0.030), mode="orthographic", ortho_height=0.064, width=640,
                   height=640, ssaa=1)
    im = render(s, cam).rgba                                    # 10 px per nm
    red = (im[..., 0] == 255) & (im[..., 1] == 0)
    row, col = red[320], red[:, 320]
    assert int(np.argmin(row[np.argmax(row):])) == 10 and int(np.argmin(col[np.argmax(col):])) == 10


def test_voxel_colour_function_and_per_face_mesh_styles():
    mask = LAB > 0
    s = E.Scene()
    s.add_voxels(mask, VS, convention="corner", lit=False, color=lambda kji: E.label_colors(LAB[kji[:, 0], kji[:, 1], kji[:, 2]]))
    img = render(s, E.Camera.for_slice(GRID, "z", "physical", background=(0, 0, 0)))      # numpy orientation, seen from -z
    first = mask.argmax(0)                                      # the first set voxel along z in every column
    seen = np.where(mask.any(0), LAB[first, np.arange(NY)[:, None], np.arange(NX)[None, :]], 0)
    exp = np.where((seen > 0)[..., None], E.label_colors(seen), 0.0)
    assert np.abs(img.rgba_float[..., :3] - exp).max() < 1e-6
    s = E.Scene()
    v = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]
    s.add_mesh(v, [(0, 1, 2), (0, 2, 3)], color=[(1, 0, 0), (0, 0, 1)], object_id=[11, 12], lit=False)      # one style per face
    img = render(s, E.Camera.face_on((0, 0, 0), (1, 0, 1), axis="y", orientation="physical", pixel_size=0.1))
    lower = np.tril(np.ones((10, 10), bool), -1)[:, ::-1]       # the triangle (0, 1, 2) is the lower right half (x right, z up)
    upper = np.triu(np.ones((10, 10), bool), 1)[:, ::-1]
    assert np.all(img.object_id[lower] == 11) and np.all(img.object_id[upper] == 12)
    assert np.all(img.rgba[lower][:, :3] == (255, 0, 0)) and np.all(img.rgba[upper][:, :3] == (0, 0, 255))


def test_empty_and_malformed_masks_raise():
    with pytest.raises(ValueError, match="empty"):
        E.Scene().add_voxels(np.zeros((3, 3, 3), bool), VS)
    with pytest.raises(ValueError, match="3D"):
        E.Scene().add_voxels(np.ones((3, 3), bool), VS)
    with pytest.raises(ValueError, match="positive"):
        E.Scene().add_voxels(np.ones((2, 2, 2), bool), (0.03, 0.0, 0.006))


# =================================================================================================
# 5. materials: wire lines, translucency, object ids
# =================================================================================================
def test_wire_width_is_in_world_units():
    s = E.Scene()
    s.add_quads([[(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]], color=(1, 1, 1), lit=False, wire_width=0.1, wire_color=(1, 0, 0))
    cam = E.Camera(eye=(0.5, -3, 0.5), target=(0.5, 0, 0.5), mode="orthographic", ortho_height=1.2, width=240, height=240, ssaa=1)
    img = render(s, cam)                                        # 200 px per unit -> half the line width = 10 px
    red = (img.rgba[..., 0] == 255) & (img.rgba[..., 1] == 0)
    assert red[120].sum() == 20 and red[:, 120].sum() == 20
    assert red[60:180, 60:180].sum() == 0                       # the diagonal of the quad is not an edge


@pytest.mark.parametrize("tilt", [0.0, 60.0])
def test_wire_min_pixels_keeps_lines_a_fixed_number_of_pixels_wide(tilt):
    """Display aid: with wire_min_pixels = 6 each quad draws 3 px inside each of its edges,
    however thin the line is in world units and however the face is tilted."""
    c, sn = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    quad = [(-0.5 * c, -0.5 * sn, -0.5), (0.5 * c, 0.5 * sn, -0.5), (0.5 * c, 0.5 * sn, 0.5), (-0.5 * c, -0.5 * sn, 0.5)]
    cam = E.Camera(eye=(0, -3, 0), target=(0, 0, 0), mode="orthographic", ortho_height=1.28, width=128, height=128, ssaa=1)

    def border(**wire):                                         # red pixels along the middle row / middle column
        s = E.Scene()
        s.add_quads([quad], color=(1, 1, 1), lit=False, wire_color=(1, 0, 0), **wire)
        img = render(s, cam)
        red = (img.rgba[..., 0] == 255) & (img.rgba[..., 1] == 0)
        assert (img.object_id[64] == 0).sum() == round(100 * c)                # the quad is 100 px * cos(tilt) wide on screen
        return int(red[64].sum()), int(red[:, 64].sum())

    assert border(wire_min_pixels=6.0) == (6, 6)                # 3 px inside the left and right (top and bottom) edge
    assert border(wire_width=0.001, wire_min_pixels=6.0) == (6, 6)
    world = border(wire_width=0.12)                             # 0.06 units inside each edge = 6 px face-on, foreshortened when tilted
    assert world == (round(12 * c), 12)
    assert border(wire_width=0.12, wire_min_pixels=10.0) == (max(world[0], 10), 12)     # the wider of the two rules wins


def test_wire_min_pixels_in_perspective_does_not_thin_out_with_distance():
    s = E.Scene()
    for y in (0.0, 30.0):                                       # a quad near the eye and one eleven times farther away
        size = (0.5 if y == 0 else 0.25) * (3.0 + y) / 3.0      # (the far one looks half as big)
        s.add_quads([[(-size, y, -size), (size, y, -size), (size, y, size), (-size, y, size)]], color=(1, 1, 1, 1.0 if y else 0.0),
                    lit=False, wire_color=(1, 0, 0) if y else (0, 0, 1), wire_min_pixels=4.0)
    cam = E.Camera(eye=(0, -3, 0), target=(0, 0, 0), mode="perspective", fov=30, width=200, height=200, ssaa=1)
    img = render(s, cam).rgba
    blue = (img[..., 2] == 255) & (img[..., 0] == 0)            # near quad: transparent fill, blue lines
    red = (img[..., 0] == 255) & (img[..., 1] == 0)             # far quad, seen through the near one
    for lines in (blue, red):
        runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], lines[100].astype(int), [0]]))))[::2]
        assert len(runs) == 2 and np.all((runs >= 1) & (runs <= 3))            # about 2 px inside each vertical edge


def test_translucent_surfaces_composite_front_to_back():
    s = E.Scene()
    for i, c in enumerate([(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5)]):
        s.add_quads([[(-1, i, -1), (1, i, -1), (1, i, 1), (-1, i, 1)]], color=c, lit=False, object_id=i + 1)
    cam = E.Camera(eye=(0, -3, 0), target=(0, 0, 0), mode="orthographic", ortho_height=1.0, width=64, height=64, ssaa=2, background=(1, 1, 1))
    img = render(s, cam)
    assert np.abs(img.rgba_float - np.array([0.5 + 0.125, 0.25 + 0.125, 0.125 + 0.125, 1.0])).max() < 1e-6
    img2 = render(s, cam, max_hits=2)
    assert np.abs(img2.rgba_float - np.array([0.5 + 0.25, 0.25 + 0.25, 0.25, 1.0])).max() < 1e-6
    assert np.all(img.object_id == 1) and abs(float(img.depth[0, 0]) - 3.0) < 1e-5


@pytest.mark.parametrize("nlayers", [10, 16, 17, 30, 200])
def test_fully_transparent_faces_do_not_use_up_max_hits(nlayers):
    s = E.Scene()
    for n in range(nlayers):
        s.add_quads([[(-1, n * 0.01, -1), (1, n * 0.01, -1), (1, n * 0.01, 1), (-1, n * 0.01, 1)]], color=(0, 1, 0, 0.0), lit=False,
                    wire_width=0.01, object_id=1)
    s.add_quads([[(-1, 9, -1), (1, 9, -1), (1, 9, 1), (-1, 9, 1)]], color=(1, 0, 0), lit=False, object_id=2)
    cam = E.Camera(eye=(0.3, -3, 0.2), target=(0.3, 0, 0.2), mode="orthographic", ortho_height=0.1, width=8, height=8, ssaa=1, background=(1, 1, 1))
    im = render(s, cam)                                         # default max_hits = 16
    assert np.all(im.rgba[..., :3] == (255, 0, 0)) and np.all(im.object_id == 2)


def test_object_id_buffer():
    rng = np.random.default_rng(1)
    s = E.Scene()
    gx, gy = np.meshgrid(np.arange(6), np.arange(6))
    ctr = np.stack([gx.ravel() * 1.0, rng.uniform(-2, 2, 36), gy.ravel() * 1.0], axis=1)        # random depth along y
    s.add_boxes(ctr, (0.6, 0.3, 0.4), object_id=100 + np.arange(36), color=E.label_colors(100 + np.arange(36)))
    s.add_spheres(ctr + [0.5, 0, 0.5], 0.15, object_id=200 + np.arange(36))
    tp = np.concatenate([ctr + [-0.3, 0, 0.3], ctr + [0.3, 0, 0.3]])
    s.add_tubes(tp, np.stack([np.arange(36), 36 + np.arange(36)], 1), 0.04, sides=16, object_id=300 + np.arange(36))
    cam = E.Camera(eye=(2.5, -20, 2.5), target=(2.5, 0, 2.5), up=(0, 0, 1), mode="orthographic", ortho_height=7.0, width=700, height=700, ssaa=1)
    img = render(s, cam)
    o, _ = rays(cam)
    exp_id = np.full(o.shape[:2], -1)
    exp_t = np.full(o.shape[:2], np.inf)
    unsure = np.zeros(o.shape[:2], bool)            # thin bands at the facetted silhouettes of spheres and tubes
    for i, c in enumerate(ctr):                     # analytic footprints of the three kinds of primitive
        dx, dz = o[..., 0] - c[0], o[..., 2] - c[2]
        inside = (np.abs(dx) < 0.3) & (np.abs(dz) < 0.2)
        exp_id[inside], exp_t[inside] = 100 + i, c[1] - 0.15 + 20
        r2 = (dx - 0.5) ** 2 + (dz - 0.5) ** 2
        exp_id[r2 < 0.145 ** 2] = 200 + i
        unsure |= (r2 >= 0.145 ** 2) & (r2 <= 0.15 ** 2)
        in_tube = (np.abs(dx) < 0.3) & (np.abs(dz - 0.3) < 0.039)
        exp_id[in_tube] = 300 + i
        unsure |= (np.abs(dx) < 0.3) & (np.abs(dz - 0.3) >= 0.039) & (np.abs(dz - 0.3) <= 0.04)
    sure = ~unsure
    box_px = (exp_id >= 100) & (exp_id < 200)
    assert sure.sum() > 400000 and len(np.unique(img.object_id)) - 1 == 108
    assert int((img.object_id != exp_id)[sure].sum()) == 0
    assert float(np.abs(img.depth[box_px] - exp_t[box_px]).max()) < 1e-4


# =================================================================================================
# 6. Gaussians against the analytic image
# =================================================================================================
ISO = np.eye(3) * 0.3 ** 2
ANISO = rot((1, 2, 3), 50) @ np.diag([0.6, 0.25, 0.08]) ** 2 @ rot((1, 2, 3), 50).T
EM_LIKE = np.diag([0.1, 0.1, 0.5]) ** 2                         # elongated along z, like a 5:1 voxel


@pytest.mark.parametrize("cov", [ISO, ANISO, EM_LIKE], ids=["isotropic", "anisotropic_rotated", "anisotropic_5to1"])
@pytest.mark.parametrize("cutoff,tol_m2", [(6.0, 2e-4), (4.0, 5e-3)])
def test_gaussian_image_against_the_analytic_marginal(cov, cutoff, tol_m2):
    """Orthographic: the alpha image of one Gaussian is opacity * its 2D marginal (same peak, mean
    and covariance), cut off at `cutoff` sigma."""
    mu, opac = (0.3, -0.2, 0.4), 0.9
    s = E.Scene()
    s.add_gaussians([mu], covs=[cov], opacity=opac, color=(1, 1, 1), object_id=9)
    s.build(DEVICE, gaussian_cutoff=cutoff)
    cam = E.Camera(eye=(2.0, -5.0, 3.0), target=(0.1, 0.0, 0.2), mode="orthographic", ortho_height=6.0, width=600, height=600,
                   ssaa=1, background=(0, 0, 0, 0))
    img = render(s, cam, gaussian_min_alpha=0.0)
    r, u, f = cam.basis()
    p = np.stack([r, u])                                       # world -> image plane (right, up)
    o, _ = rays(cam)
    q = (o - np.asarray(cam.eye)) @ p.T                        # pixel-centre coordinates in the image plane
    mu2, cov2 = (np.asarray(mu) - np.asarray(cam.eye)) @ p.T, p @ cov @ p.T
    dq = q - mu2
    d2 = np.einsum("...a,ab,...b->...", dq, np.linalg.inv(cov2), dq)
    exp = opac * np.exp(-0.5 * d2) * (d2 < cutoff ** 2)
    a = img.rgba_float[..., 3].astype(np.float64)
    cen = (a[..., None] * q).sum((0, 1)) / a.sum()
    m2 = np.einsum("yx,yxa,yxb->ab", a, q - cen, q - cen) / a.sum()
    iy, ix = np.unravel_index(np.argmax(a), a.shape)
    peak_px = cam.project(mu)[0]
    assert (iy, ix) == np.unravel_index(np.argmax(exp), exp.shape)
    assert np.abs(a - exp).max() < 2e-4
    assert np.linalg.norm(cen - mu2) < 0.02 * cam.pixel_size() + 1e-4
    assert np.abs(m2 - cov2).max() < tol_m2 * np.abs(cov2).max()
    assert abs(ix + 0.5 - peak_px[0]) <= 1.0 and abs(iy + 0.5 - peak_px[1]) <= 1.0      # the peak is at the projected mean


@pytest.mark.parametrize("cov", [ANISO, EM_LIKE], ids=["rotated", "voxel_5to1"])
def test_gaussian_in_perspective_whole_alpha_image(cov):
    mug = np.array([0.3, -0.2, 0.4])
    s = E.Scene()
    s.add_gaussians([mug], covs=[cov], opacity=0.9, color=(1, 1, 1))
    s.build(DEVICE, gaussian_cutoff=6.0)
    cam = E.Camera(eye=(2.0, -5.0, 3.0), target=(0.1, 0.0, 0.2), mode="perspective", fov=40, width=640, height=480, ssaa=1,
                   background=(0, 0, 0, 0))
    a_img = render(s, cam, gaussian_min_alpha=0.0).rgba_float[..., 3].astype(np.float64)
    o, d = rays(cam)
    P = np.linalg.inv(cov)
    om = o - mug
    A, B, C = np.einsum("...a,ab,...b->...", d, P, d), np.einsum("...a,ab,...b->...", om, P, d), np.einsum("...a,ab,...b->...", om, P, om)
    d2 = C - B * B / A                                          # min over t of (o + t d - mu)^T P (o + t d - mu)
    a_exp = 0.9 * np.exp(-0.5 * d2) * (d2 < 36.0) * ((-B / A) > 0)
    assert np.abs(a_img - a_exp).max() < 2e-4
    iy, ix = np.unravel_index(np.argmax(a_img), a_img.shape)    # the ray through the mean has d2 = 0
    pp = cam.project(mug)[0]
    assert abs(ix + 0.5 - pp[0]) <= 0.75 and abs(iy + 0.5 - pp[1]) <= 0.75 and abs(a_img.max() - 0.9) < 5e-3


TINY_CAM = dict(eye=(0, -10, 0), target=(0, 0, 0), mode="orthographic", ortho_height=0.02, width=4, height=4, ssaa=1, background=(0, 0, 0, 0))


def test_gaussians_are_ordered_and_stop_at_opaque_geometry():
    cam = E.Camera(**TINY_CAM)
    s = E.Scene()
    s.add_gaussians([(0, 2, 0), (0, -2, 0)], covs=[np.eye(3), np.eye(3)], opacity=[0.6, 0.5], color=[(0, 1, 0), (1, 0, 0)])
    img = render(s, cam)
    assert np.abs(img.rgba_float - np.array([0.5, 0.5 * 0.6, 0.0, 1 - 0.5 * 0.4])).max() < 1e-4
    s.add_quads([[(-5, 0, -5), (5, 0, -5), (5, 0, 5), (-5, 0, 5)]], color=(0, 0, 1), lit=False, object_id=3)
    img = render(s, cam)
    assert np.abs(img.rgba_float - np.array([0.5, 0.0, 0.5, 1.0])).max() < 1e-4 and np.all(img.object_id == 3)


def test_more_gaussians_on_one_ray_than_the_buffer_holds():
    n = 300
    assert n > E.GAUSS_BUF
    cam = E.Camera(**TINY_CAM)
    s = E.Scene()
    s.add_gaussians(np.stack([np.zeros(n), np.linspace(-3, 3, n), np.zeros(n)], 1), covs=np.tile(np.eye(3) * 0.01, (n, 1, 1)),
                    opacity=0.01, color=np.tile([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], (n // 2, 1)))
    img = render(s, cam, min_transmittance=0.0)
    o, _ = rays(cam)
    a1 = 0.01 * np.exp(-0.5 * (o[..., 0] ** 2 + o[..., 2] ** 2) / 0.01)      # per-Gaussian alpha on each pixel's ray
    white = sum((1 - a1) ** i * a1 for i in range(0, n, 2))                    # even ones are white, odd ones red
    cover = 1 - (1 - a1) ** n
    assert np.abs(img.rgba_float - np.stack([cover, white, white, cover], axis=-1)).max() < 2e-5


@pytest.mark.parametrize("n", [60, 64, 65, 100, 200, 1000])
def test_gaussians_at_exactly_equal_depth_are_all_composited(n):
    s = E.Scene()
    s.add_gaussians(np.zeros((n, 3)), covs=np.tile(np.eye(3), (n, 1, 1)), opacity=0.02, color=(1, 1, 1))
    a = float(render(s, E.Camera(**TINY_CAM), min_transmittance=0.0, gaussian_min_alpha=0.0).rgba_float[0, 0, 3])
    assert abs(a - (1 - 0.98 ** n)) < 1e-3


def test_coplanar_lattice_of_gaussians_seen_face_on():
    g = np.arange(-12, 13) * 1.0
    gx, gz = np.meshgrid(g, g)
    means = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], 1)
    s = E.Scene()
    s.add_gaussians(means, covs=np.tile(np.eye(3) * 4.0, (len(means), 1, 1)), opacity=0.05, color=(1, 1, 1))
    s.build(DEVICE, gaussian_cutoff=4.0)
    cam = E.Camera(**TINY_CAM)
    a = float(render(s, cam, min_transmittance=0.0, gaussian_min_alpha=0.0).rgba_float[0, 0, 3])
    o, _ = rays(cam)
    d2 = ((means[:, 0] - o[0, 0, 0]) ** 2 + (means[:, 2] - o[0, 0, 2]) ** 2) / 4.0
    al = 0.05 * np.exp(-0.5 * d2) * (d2 < 16.0)
    assert (d2 < 16).sum() > E.GAUSS_BUF and abs(a - (1 - np.prod(1 - al))) < 1e-3


def test_gaussian_input_is_validated():
    s = E.Scene()
    with pytest.raises(ValueError, match="positive definite"):
        s.add_gaussians([(0, 0, 0)], covs=[np.diag([1.0, 1.0, 0.0])])
    with pytest.raises(ValueError, match="symmetric"):
        s.add_gaussians([(0, 0, 0)], covs=[np.array([[1.0, 0.5, 0], [0.0, 1.0, 0], [0, 0, 1.0]])])
    with pytest.raises(ValueError):
        s.add_gaussians([(0, 0, 0)], covs=[np.eye(3) * np.nan])
    with pytest.raises(ValueError):
        s.add_gaussians([(0, 0, 0), (1, 1, 1)], covs=[np.eye(3)])
    with pytest.raises(ValueError):
        s.add_gaussians([(0, 0, 0)])
    with pytest.raises(ValueError, match="positive"):
        s.add_gaussians([(0, 0, 0)], rotations=[np.eye(3)], scales=[(1.0, 0.0, 1.0)])
    assert s.num_gaussians == 0


# =================================================================================================
# 7. direct volume rendering
# =================================================================================================
@pytest.mark.parametrize("convention", E.CONVENTIONS)
def test_volume_rendering_is_beer_lambert_and_stops_at_opaque_geometry(convention):
    vol = np.full((10, 20, 20), 100, np.uint8)
    lut = np.zeros((256, 4), np.float32)
    lut[100] = (1.0, 0.5, 0.25, 0.2)                           # opacity 0.2 per reference length
    s = E.Scene()
    smp = s.add_volume(vol, voxel_size=(0.5, 0.1, 0.1), lut=lut, convention=convention)       # a box 2 x 2 x 5
    lo, hi = s.volume_corners(smp)
    c = 0.5 * (lo + hi)
    assert np.allclose(hi - lo, (2, 2, 5)) and np.allclose(lo, 0.0 if convention == "corner" else (-0.05, -0.05, -0.25))
    s.set_dvr(smp, step=0.02, ref_length=0.5)
    cam = E.Camera(eye=(c[0], -10, c[2]), target=(c[0], 0, c[2]), up=(0, 0, 1), mode="orthographic", ortho_height=1.0, width=8,
                   height=8, ssaa=1, background=(0, 0, 0, 0))
    img = render(s, cam, min_transmittance=0.0)
    a_exp = 1 - 0.8 ** (2.0 / 0.5)                              # 2 units of path
    assert np.abs(img.rgba_float - np.array([a_exp, 0.5 * a_exp, 0.25 * a_exp, a_exp])).max() < 2e-4
    s.add_quads([[(-5, c[1], -5), (5, c[1], -5), (5, c[1], 10), (-5, c[1], 10)]], color=(0, 0, 1), lit=False, object_id=4)
    img = render(s, cam, min_transmittance=0.0)
    a_half = 1 - 0.8 ** (1.0 / 0.5)                             # an opaque wall half way through
    assert np.abs(img.rgba_float - np.array([a_half, 0.5 * a_half, 0.25 * a_half + (1 - a_half), 1.0])).max() < 2e-4
    s.set_dvr(None, 1.0)
    assert np.all(render(s, cam).rgba[..., :3] == (0, 0, 255))
    only = E.Scene()                                            # nothing but the ray-marched volume: its box is the scene box
    only.set_dvr(only.add_volume(vol, voxel_size=(0.5, 0.1, 0.1), lut=lut, convention=convention), step=0.02, ref_length=0.5)
    cam.background = (0, 0, 0, 1)
    fogged = render(only, cam, min_transmittance=0.0, fog=1.0).rgba_float[0, 0]        # fog 0 at the near face, 1 at the far face
    t = np.arange(0.01, 2.0, 0.02)                              # the sample positions inside the box
    w = 0.8 ** ((t - 0.01) / 0.5) * (1 - 0.8 ** (0.02 / 0.5))   # their weights
    assert abs(fogged[0] - (w * (1 - t / 2.0)).sum()) < 2e-3 and abs(fogged[0] - a_exp * 0.55) < 0.05
    with pytest.raises(ValueError):
        s.set_dvr(5, 0.1)
    with pytest.raises(ValueError):
        s.set_dvr(smp, 0.0)


# =================================================================================================
# 8. robustness: value ranges, storage, wrong input
# =================================================================================================
@pytest.mark.parametrize("arr", [np.full((2, 2, 2), 3_000_000_000, np.uint32), np.full((2, 2, 2), 2 ** 32, np.uint64),
                                 np.full((2, 2, 2), -2 ** 40, np.int64)], ids=["uint32_3e9", "uint64_2**32", "int64_-2**40"])
def test_label_ids_beyond_int32_raise(arr):
    with pytest.raises(ValueError, match="int32"):
        E.Scene().add_volume(arr, labels=True)


def test_scalar_volume_outside_0_255_raises():
    with pytest.raises(ValueError, match="256-entry"):
        E.Scene().add_volume(np.full((2, 2, 2), 300, np.uint16))
    with pytest.raises(ValueError, match="256-entry"):
        E.Scene().add_volume(np.full((2, 2, 2), -1, np.int8))
    E.Scene().add_volume(np.full((2, 2, 2), 255, np.uint16))    # in range: fine
    E.Scene().add_volume(np.full((2, 2, 2), 300, np.uint16), labels=True)
    with pytest.raises(ValueError, match="3D"):
        E.Scene().add_volume(np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError, match="lookup table"):
        E.Scene().add_volume(EM, lut=np.zeros((255, 4)))
    with pytest.raises(ValueError, match="interp"):
        E.Scene().add_volume(EM, interp="cubic")


def test_volumes_are_stored_compactly_and_once():
    s = E.Scene()
    n = EM.size
    s.add_volume(EM)                                            # uint8: 1 byte per voxel
    assert s.volume_nbytes == n
    s.add_volume(EM, voxel_size=VS, lut=E.lut_window_level())   # the same array again: not stored twice
    assert s.volume_nbytes == n
    s.add_volume(LAB, labels=True)                              # uint16: 2 bytes per voxel
    assert s.volume_nbytes == 3 * n
    s.add_volume(LAB.astype(np.int64), labels=True)             # any other integer type: 4 bytes per voxel
    assert s.volume_nbytes == 7 * n
    s.add_volume(EM.astype(np.float32) / 255.0)                 # float in [0, 1]: 1 byte per voxel
    s.add_volume(EM > 100)                                      # bool: 1 byte per voxel
    assert s.volume_nbytes == 9 * n
    s.add_slice(0, "z", 0)
    render(s, E.Camera(width=8, height=8, ssaa=1))
    assert [a.dtype for a in s._vol_gpu[1]] == [wp.uint8, wp.uint16, wp.int32]
    assert [a.shape[0] for a in s._vol_gpu[1]] == [3 * n, n, n]


def test_all_storage_types_render_the_same_values():
    small = (EM // 2).astype(np.uint8)                          # 0..127 fits every integer type
    cam = E.Camera.for_slice(GRID, "x", "physical")
    pictures = []
    for dtype in (np.uint8, np.uint16, np.int32, np.int8, np.int64):
        s = E.Scene()
        s.add_volume_grid(EM, GRID)                             # other volumes first, one in each store,
        s.add_volume_grid(LAB, GRID, labels=True)               # so that the volume under test
        s.add_volume_grid(LAB.astype(np.int32), GRID, labels=True)     # never starts at offset 0
        s.add_slice(s.add_volume_grid(small.astype(dtype), GRID), "x", 21)
        pictures.append(render(s, cam).rgba)
    assert np.array_equal(pictures[0][..., 0], upsample(small[:, :, 21], pictures[0].shape))
    assert all(np.array_equal(p, pictures[0]) for p in pictures[1:])


def test_wrong_geometry_input_raises():
    s = E.Scene()
    tri = [(0, 1, 2)]
    with pytest.raises(ValueError, match="NaN"):
        s.add_mesh([(0, 0, 0), (1, 0, 0), (0, np.nan, 0)], tri)
    with pytest.raises(ValueError, match="face ind"):
        s.add_mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 3)])
    with pytest.raises(ValueError, match=r"\(V, 3\)"):
        s.add_mesh([(0, 0), (1, 0), (0, 1)], tri)
    with pytest.raises(ValueError, match=r"\(F, 3\)"):
        s.add_mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2, 0)])
    with pytest.raises(ValueError, match="normals"):
        s.add_mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], tri, normals=[(0, 0, 1)])
    with pytest.raises(ValueError, match="colour"):
        s.add_mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], tri, color=(1, 0))
    with pytest.raises(ValueError, match="object_id"):
        s.add_spheres([(0, 0, 0), (1, 1, 1)], 0.1, object_id=[1])
    with pytest.raises(ValueError, match=r"\(Q, 4, 3\)"):
        s.add_quads([[(0, 0, 0), (1, 0, 0), (0, 1, 0)]])
    with pytest.raises(ValueError, match="edge ind"):
        s.add_tubes([(0, 0, 0), (1, 0, 0)], [(0, 2)], 0.1)
    with pytest.raises(ValueError, match="radii"):
        s.add_tubes([(0, 0, 0), (1, 0, 0)], [(0, 1)], [0.1, 0.1, 0.1])
    with pytest.raises(ValueError):
        s.add_tubes([(0, 0, 0), (1, 0, 0)], [(0, 1)], -0.1)
    with pytest.raises(ValueError, match="caps"):
        s.add_tubes([(0, 0, 0), (1, 0, 0)], [(0, 1)], 0.1, caps="square")
    with pytest.raises(ValueError):
        s.add_spheres([(0, 0, 0)], np.inf)
    with pytest.raises(ValueError, match="sampler"):
        s.add_quads([[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]], sampler=0)
    assert s.num_triangles == 0                                 # nothing half-added
    smp = s.add_volume(EM)
    with pytest.raises(ValueError, match="outside"):
        s.add_slice(smp, "z", NZ)
    with pytest.raises(ValueError, match="axis"):
        s.add_slice(smp, "w", 0)
    with pytest.raises(ValueError, match="window"):
        s.add_slice(smp, "z", 0, lo=(0, 5, 5), hi=(1, 5, 9))
    with pytest.raises(ValueError, match="window"):
        s.add_slice(smp, "z", 0, lo=(0, 0, 0), hi=(1, NY + 1, NX))
    assert s.add_tubes([(0, 0, 0), (0, 0, 0)], [(0, 1)], 0.1) == 0             # an edge of zero length draws nothing


def test_wrong_camera_raises():
    s = E.Scene()
    s.add_spheres([(0, 0, 0)], 1.0)
    for bad in (dict(mode="fisheye"), dict(eye=(1, 1, 1), target=(1, 1, 1)), dict(eye=(0, 0, 5), target=(0, 0, 0), up=(0, 0, 1)),
                dict(width=0), dict(ssaa=0), dict(fov=0.0), dict(mode="orthographic", ortho_height=0.0), dict(background=(1, 1)),
                dict(eye=(np.nan, 0, 0))):
        with pytest.raises(ValueError):
            s.render(E.Camera(**bad), device=DEVICE)
    with pytest.raises(ValueError):
        E.Camera.fit((0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError, match="orientation"):
        E.Camera.face_on((0, 0, 0), (1, 1, 1), orientation="matlab")
    with pytest.raises(ValueError, match="extent"):
        E.Camera.face_on((0, 0, 0), (1, 0, 1), axis="z")


def test_wrong_render_options_raise():
    s = E.Scene()
    s.add_spheres([(0, 0, 0)], 1.0)
    cam = E.Camera(width=8, height=8, ssaa=1)
    for bad in (dict(fog=1.5), dict(hemisphere=-0.1), dict(ambient=2.0), dict(max_hits=0), dict(fog=0.5, fog_range=(2.0, 2.0))):
        with pytest.raises(ValueError):
            s.render(cam, device=DEVICE, **bad)
    with pytest.raises(Exception):
        s.render(cam, device="no-such-device")


def test_empty_scene_renders_the_background():
    img = render(E.Scene(), E.Camera(width=16, height=8, ssaa=1, background=(0.2, 0.4, 0.6)))
    assert np.all(img.rgba == (51, 102, 153, 255)) and np.all(np.isinf(img.depth)) and np.all(img.object_id == -1)


# =================================================================================================
# 9. tubes: end caps, joints, minimum on-screen width
# =================================================================================================
def end_on_camera():
    return E.Camera(eye=(0, -5, 0), target=(0, 0, 0), mode="orthographic", ortho_height=1.0, width=65, height=65, ssaa=1)


def test_tube_ends_are_closed():
    """Looking straight down the axis of a tube from y = -5; the tube runs from y = 0 to y = 2."""
    def view(caps):
        s = E.Scene()
        nf = s.add_tubes([(0, 0, 0), (0, 2, 0)], [(0, 1)], 0.3, sides=16, caps=caps, object_id=4)
        return nf, render(s, end_on_camera())

    nf, img = view("none")                                      # hollow: the far background shows through
    assert nf == 32 and img.object_id[32, 32] == -1
    nf, img = view("flat")                                      # a disc at each end
    assert nf == 32 + 2 * 14
    o, _ = rays(end_on_camera())
    rr = np.hypot(o[..., 0], o[..., 2])
    disc = rr < 0.3 * math.cos(math.pi / 16) - 1e-6             # inside the 16-gon for sure
    assert np.all(img.object_id[disc] == 4) and np.allclose(img.depth[disc], 5.0, atol=1e-5)
    assert np.all(img.object_id[rr > 0.3 + 1e-6] == -1)         # and nothing beyond the radius
    lit = img.rgba_float[32, 32, 0]
    assert np.allclose(img.rgba_float[disc][:, 0], lit, atol=1e-6)             # flat shaded: one colour over the disc
    nf, img = view("round")                                     # a sphere joint on each end point
    assert nf == 32 + 2 * 80
    assert 5.0 - 0.3 - 1e-5 <= img.depth[32, 32] <= 5.0 - 0.3 * 0.9
    nf, _ = view(None)
    assert nf == 32


def test_round_joints_fill_the_gap_where_tubes_meet_at_an_angle():
    pts = [(-1, 0, 0), (0, 0, 0), (0, 0, 1)]                    # a right-angle bend at the origin
    cam = E.Camera(eye=(0, -5, 0), target=(0, 0, 0), mode="orthographic", ortho_height=1.0, width=101, height=101, ssaa=1)
    corner = (58, 42)                                           # pixel at (x, z) = (-0.08, 0.08): outside both tubes, inside the joint sphere
    for caps, covered in (("round", True), ("flat", False)):
        s = E.Scene()
        s.add_tubes(pts, [(0, 1), (1, 2)], 0.2, sides=24, caps=caps, object_id=[7, 8], color=[(1, 0, 0), (0, 1, 0)])
        img = render(s, cam)
        # the outer corner of the bend: x > 0 and z < 0, within the radius of the joint
        outer = img.object_id[50 + 7, 50 + 7]                   # (x, z) = (+0.07, -0.07)
        assert (outer == 7) == covered                          # the joint takes the style of the first edge that uses it
        assert img.object_id[corner[0] - 16, 50] == 8 and img.object_id[50, 30] == 7


def thin_tube_scene(min_pixels, radius=0.001):
    s = E.Scene()
    s.add_tubes([(0, 0, -20), (0, 0, 20)], [(0, 1)], radius, caps="flat", min_pixels=min_pixels, color=(1, 0, 0), lit=False, object_id=3)
    return s


@pytest.mark.parametrize("min_pixels,width", [(0.0, 0), (2.0, 2), (4.0, 4)])
def test_tube_min_pixels_orthographic(min_pixels, width):
    """A tube 0.002 units thick where a pixel is 1 unit: invisible without the display aid,
    exactly `min_pixels` wide with it (its axis lies on a pixel boundary); its length is unchanged."""
    cam = E.Camera(eye=(0, -100, 0), target=(0, 0, 0), mode="orthographic", ortho_height=64.0, width=64, height=64, ssaa=1)
    assert cam.pixel_size() == 1.0
    img = render(thin_tube_scene(min_pixels), cam)
    hit = img.object_id == 3
    per_row = hit.sum(1)
    expect = np.zeros(64, int)
    expect[12:52] = width                                       # z from -20 to 20: rows 12 .. 51
    assert np.array_equal(per_row, expect)
    if width:
        assert np.array_equal(np.flatnonzero(hit[32]), np.arange(32 - width // 2, 32 + width // 2))


def test_tube_min_pixels_follows_the_camera_and_never_thins_a_thick_tube():
    s = thin_tube_scene(2.0)
    far = E.Camera(eye=(0, -100, 0), target=(0, 0, 0), mode="orthographic", ortho_height=64.0, width=64, height=64, ssaa=1)
    near = E.Camera(eye=(0, -100, 0), target=(0, 0, 0), mode="orthographic", ortho_height=0.0064, width=64, height=64, ssaa=1)
    assert (render(s, far).object_id[32] == 3).sum() == 2       # widened to 2 px
    n = int((render(s, near).object_id[32] == 3).sum())         # zoomed in 10000x: true width 0.002 = 20 px
    assert 19 <= n <= 20
    assert (render(s, far).object_id[32] == 3).sum() == 2       # and back: nothing accumulates


def test_thin_diagonal_tube_does_not_break_up_in_perspective():
    def rows_hit(min_pixels):
        s = E.Scene()
        s.add_tubes([(-1, 0, -0.8), (0.9, 3, 1)], [(0, 1)], 1e-4, min_pixels=min_pixels, object_id=3)
        cam = E.Camera(eye=(0, -6, 0), target=(0, 1.5, 0), mode="perspective", fov=30, width=160, height=160, ssaa=1)
        hit = render(s, cam).object_id == 3
        ends = cam.project([(-1, 0, -0.8), (0.9, 3, 1)])
        r0, r1 = sorted(int(round(y)) for y in ends[:, 1])
        return hit[r0 + 2:r1 - 2].sum(1)

    assert (rows_hit(0.0) == 0).mean() > 0.9                    # sub-pixel tube: almost every row misses it
    with_aid = rows_hit(1.5)
    assert len(with_aid) > 40 and np.all(with_aid >= 1) and np.all(with_aid <= 3)


def test_sphere_min_pixels():
    cam = E.Camera(eye=(0, -100, 0), target=(0, 0, 0), mode="orthographic", ortho_height=64.0, width=64, height=64, ssaa=1)
    for min_pixels, lo, hi in ((0.0, 0, 0), (6.0, 24, 32)):     # a disc 6 px across: pi * 3**2 = 28 px
        s = E.Scene()
        s.add_spheres([(0, 0, 0)], 1e-3, subdiv=3, min_pixels=min_pixels, object_id=2)
        assert lo <= int((render(s, cam).object_id == 2).sum()) <= hi


# =================================================================================================
# 10. teaching aids: wire box, axes, scale bar
# =================================================================================================
def test_wire_box_draws_the_twelve_edges():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([4.0, 2.0, 3.0])
    s = E.Scene()
    s.add_wire_box(lo, hi, radius=0.05, object_id=900)
    assert np.allclose(s.bounds()[0], lo - 0.05) and np.allclose(s.bounds()[1], hi + 0.05)     # round corners: the radius sticks out
    cam = E.Camera.face_on(lo - 0.5, hi + 0.5, axis="y", orientation="physical", pixel_size=0.02)     # x right, z up
    img = render(s, cam)
    o, _ = rays(cam)
    x, z = o[..., 0], o[..., 2]
    # seen along y the box is the outline of the rectangle [0, 4] x [0, 3], drawn 0.1 thick with round corners
    out_x, out_z = np.maximum(np.maximum(lo[0] - x, x - hi[0]), 0), np.maximum(np.maximum(lo[2] - z, z - hi[2]), 0)
    in_x, in_z = np.minimum(x - lo[0], hi[0] - x), np.minimum(z - lo[2], hi[2] - z)
    dist = np.where((out_x > 0) | (out_z > 0), np.hypot(out_x, out_z), np.minimum(in_x, in_z))     # distance to the outline
    hit = img.object_id == 900
    assert (dist < 0.044).sum() > 2500 and np.all(hit[dist < 0.044]) and not np.any(hit[dist > 0.051])
    s2 = E.Scene()
    s2.add_wire_box(lo, hi)                                     # default radius: 0.2 % of the diagonal
    assert np.allclose(s2.bounds()[1] - hi, 0.002 * np.linalg.norm(hi - lo))


def test_wire_box_stays_visible_when_zoomed_out():
    s = E.Scene()
    s.add_wire_box((0, 0, 0), (1, 1, 1), radius=1e-5, object_id=900)       # default min_pixels = 1.5
    cam = E.Camera.face_on((-1, 0, -1), (2, 1, 2), axis="y", orientation="physical", pixel_size=0.05)   # the hair is 4e-4 px thick
    hit = render(s, cam).object_id == 900
    rows, cols = np.flatnonzero(hit.any(1)), np.flatnonzero(hit.any(0))
    assert len(rows) >= 20 and np.all(np.diff(rows) == 1) and np.all(np.diff(cols) == 1)       # an unbroken outline
    assert np.all(hit[rows[0]:rows[-1] + 1].sum(1) >= 2)                    # every row crosses at least the two sides


def test_axes_are_x_red_y_green_z_blue():
    assert E.AXIS_COLORS == ((0.9, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.3, 1.0))
    s = E.Scene()
    s.add_axes(origin=(1, 2, 3), lengths=(2.0, 1.0, 0.5), radius=0.02, lit=False, object_id=5)
    lo, hi = s.bounds()
    assert np.allclose(hi, (3.0, 3.0, 3.5), atol=0.051) and np.all(hi >= np.array([3.0, 3.0, 3.5]) - 1e-6)      # the tips are at origin + length
    for axis, mid, rgb in (("y", (2.0, 0, 3.0), (230, 26, 26)), ("y", (1.0, 0, 3.2), (26, 77, 255)), ("x", (0, 2.5, 3.0), (26, 179, 26))):
        cam = E.Camera.face_on((0, 1, 2), (4, 4, 4), axis=axis, orientation="physical", pixel_size=0.01)
        img = render(s, cam)
        px = cam.project((1.0, mid[1], mid[2]) if axis == "x" else (mid[0], 2.0, mid[2]))[0].astype(int)
        assert tuple(img.rgba[px[1], px[0], :3]) == rgb         # the middle of the x / z / y arrow
        assert img.object_id[px[1], px[0]] == 5
    plain = E.Scene()
    assert plain.add_axes(lengths=1.0, heads=False) < s.num_triangles
    assert np.allclose(plain.bounds()[1], 1.0) and np.allclose(plain.bounds()[0], -0.025)      # default radius: 2.5 % of the length
    with pytest.raises(ValueError):
        plain.add_axes(lengths=(1, 0, 1))


@pytest.mark.parametrize("length", [0.5, 1.0, 2.0])
def test_scale_bar_has_exactly_its_length(length):
    s = E.Scene()
    s.add_scale_bar(start=(0.25, 0, 0.5), direction=(5, 0, 0), length=length, radius=0.03, object_id=6)
    lo, hi = s.bounds()
    assert math.isclose(lo[0], 0.25) and math.isclose(hi[0], 0.25 + length)
    cam = E.Camera.face_on((0, -1, 0), (3, 1, 1), axis="y", orientation="physical", pixel_size=0.01)
    img = render(s, cam)
    hit = img.object_id == 6
    cols = np.flatnonzero(hit.any(0))
    assert len(cols) == round(length / cam.pixel_size()) and cols[0] == 25      # flat ends: no more, no less
    assert np.all(img.rgba[hit][:, :3] == 0)                    # unlit black by default
    thin = E.Scene()
    thin.add_scale_bar((0, 0, 0), (0, 0, 1), length)            # default radius: length / 80
    assert math.isclose(thin.bounds()[1][:2].max(), length / 80) and np.allclose([thin.bounds()[0][2], thin.bounds()[1][2]], (0, length))
    with pytest.raises(ValueError):
        s.add_scale_bar((0, 0, 0), (1, 0, 0), 0.0)
    with pytest.raises(ValueError):
        s.add_scale_bar((0, 0, 0), (0, 0, 0), 1.0)


# =================================================================================================
# 11. depth cues: fog and hemisphere ambient (off by default)
# =================================================================================================
def tilted_plane_scene():
    s = E.Scene()
    s.add_quads([[(-1, 0, -1), (1, 2, -1), (1, 2, 1), (-1, 0, 1)]], color=(1, 0, 0), lit=False)
    return s


def test_fog_blends_toward_the_background_with_depth():
    s = tilted_plane_scene()
    cam = E.Camera(eye=(0, -4, 0), target=(0, 0, 0), mode="orthographic", ortho_height=1.5, width=64, height=64, ssaa=1,
                   background=(1, 1, 1))
    plain = render(s, cam)
    assert np.all(plain.rgba == (255, 0, 0, 255))               # off by default
    t = plain.depth                                             # 4 + y: from 4.25 to 5.75 over the picture
    assert 4.24 < t.min() < 4.3 and 5.7 < t.max() < 5.76
    img = render(s, cam, fog=0.8, fog_range=(4.0, 6.0))
    f = 0.8 * (t - 4.0) / 2.0
    assert np.abs(img.rgba_float - np.stack([np.ones_like(f), f, f, np.ones_like(f)], -1)).max() < 1e-5
    img = render(s, cam, fog=1.0)                               # default range: nearest and farthest corner of the scene box
    f = (t - 4.0) / 2.0                                         # the quad spans y = 0 .. 2, i.e. depth 4 .. 6
    assert np.abs(img.rgba_float[..., 1] - f).max() < 1e-5
    img = render(s, cam, fog=0.8, fog_range=(0.0, 1.0))         # everything beyond the range: saturates at `fog`
    assert np.abs(img.rgba_float[..., 1] - 0.8).max() < 1e-6


def test_fog_applies_to_gaussians_and_volume_samples():
    cam = E.Camera(**{**TINY_CAM, "background": (0, 0, 1, 1)})
    s = E.Scene()
    s.add_gaussians([(0, 0, 0)], covs=[np.eye(3)], opacity=0.6, color=(1, 0, 0))
    a = float(0.6 * np.exp(-0.5 * (0.0025 ** 2 * 2)))           # alpha of the pixel-centre ray, 0.0025 off the mean in x and z
    img = render(s, cam, fog=1.0, fog_range=(5.0, 15.0))        # the Gaussian is at depth 10: half fogged
    exp = np.array([0.5 * a, 0.0, 0.5 * a + (1 - a), 1.0])
    assert np.abs(img.rgba_float[0, 0] - exp).max() < 1e-4


def test_hemisphere_ambient_on_a_sphere():
    cam = E.Camera(eye=(0.7, -3.0, 2.0), target=(0, 0, 0), mode="orthographic", ortho_height=2.4, width=200, height=200,
                   ssaa=1, background=(0, 0, 0))
    o, d = rays(cam)
    s = E.Scene()
    s.add_spheres([[0, 0, 0]], 1.0, subdiv=5, color=(1, 1, 1))
    b = (o * d).sum(-1)
    disc = b * b - ((o * o).sum(-1) - 1.0)
    n = o + d * (-b - np.sqrt(np.maximum(disc, 0)))[..., None]  # hit point on the unit sphere = outward normal
    up = cam.basis()[1]
    inner = disc > 0.05
    for a, h in ((0.3, 1.0), (0.5, 0.4)):
        img = render(s, cam, ambient=a, hemisphere=h)
        lam = a * (1 - h * 0.5 * (1 - n @ up)) + (1 - a) * np.abs((n * d).sum(-1))
        assert np.abs(img.rgba_float[..., 0] - lam)[inner].max() < 0.01
    top, bottom = img.rgba_float[40, 100, 0], img.rgba_float[160, 100, 0]
    assert top > bottom + 0.05                                  # lit from above: the lower half is darker


def test_depth_cues_are_off_by_default_and_change_nothing_when_zero():
    s = E.Scene()
    s.add_spheres([[0, 0, 0], [0.5, 1.0, 0.2]], [1.0, 0.6], color=[(1, 0.5, 0.2, 1.0), (0.2, 0.5, 1.0, 0.5)])
    s.add_gaussians([(0.2, -1.5, 0.1)], covs=[np.eye(3) * 0.05], opacity=0.5)
    cam = E.Camera(eye=(0.7, -4.0, 2.0), target=(0, 0, 0), fov=40, width=96, height=96, ssaa=2)
    a, b = render(s, cam), render(s, cam, hemisphere=0.0, fog=0.0, fog_range=(1.0, 2.0))
    assert np.array_equal(a.rgba_float, b.rgba_float) and np.array_equal(a.depth, b.depth)
    c = render(s, cam, hemisphere=1.0, fog=0.5)
    assert np.abs(c.rgba_float - a.rgba_float).max() > 0.05 and np.array_equal(c.depth, a.depth)       # geometry is untouched


# =================================================================================================
# 12. cameras
# =================================================================================================
@pytest.mark.parametrize("mode", ["perspective", "orthographic"])
@pytest.mark.parametrize("width,height", [(120, 120), (180, 120), (120, 180)])
def test_camera_fit_bounding_sphere_just_fills_the_shorter_side(mode, width, height):
    lo, hi = np.array([-1.0, 0.0, 2.0]), np.array([2.0, 1.0, 4.0])
    c, r = 0.5 * (lo + hi), 0.5 * np.linalg.norm(hi - lo)
    cam = E.Camera.fit(lo, hi, direction=(1, -0.8, 0.5), margin=1.0, mode=mode, fov=30, width=width, height=height, ssaa=1)
    corners = E.box_edges(lo, hi)[0]
    px = cam.project(corners)
    assert np.all(px >= -1e-9) and np.all(px[:, 0] <= width + 1e-9) and np.all(px[:, 1] <= height + 1e-9)
    s = E.Scene()
    s.add_spheres([c], r, subdiv=5, object_id=1)                # the bounding sphere itself
    hit = render(s, cam).object_id == 1
    rows, cols = np.flatnonzero(hit.any(1)), np.flatnonzero(hit.any(0))
    short = min(width, height)
    assert short - 2 <= min(len(rows), len(cols)) <= short and max(len(rows), len(cols)) <= short + 1
    if mode == "orthographic":
        assert math.isclose(cam.pixel_size() * short, 2 * r)
    assert np.allclose(cam.target, c)


def test_camera_pixel_size():
    ortho = E.Camera(mode="orthographic", ortho_height=3.0, width=400, height=300)
    assert ortho.pixel_size() == 0.01 and ortho.pixel_size(distance=123.0) == 0.01
    persp = E.Camera(eye=(0, -5, 0), target=(0, 0, 0), fov=40, width=400, height=300)
    with pytest.raises(ValueError, match="distance"):
        persp.pixel_size()
    size = persp.pixel_size(distance=5.0)
    assert math.isclose(size, 2 * math.tan(math.radians(20)) * 5.0 / 300)
    a, b = persp.project([(0, 0, 0), (size, 0, 0)])             # two points one pixel apart at that distance
    assert math.isclose(b[0] - a[0], 1.0, rel_tol=1e-9) and math.isclose(b[1], a[1])


def test_camera_face_on_frames_the_cross_section():
    lo, hi = np.array([1.0, 2.0, 3.0]), np.array([5.0, 4.0, 4.0])
    for axis, (w, h) in (("z", (4, 2)), ("y", (4, 1)), ("x", (2, 1))):
        for orientation in ("numpy", "physical"):
            cam = E.Camera.face_on(lo, hi, axis=axis, orientation=orientation, pixel_size=0.05)
            assert (cam.width, cam.height) == (w * 20, h * 20) and math.isclose(cam.pixel_size(), 0.05) and cam.ssaa == 1
            px = cam.project([lo, hi])
            assert np.allclose(np.sort(px[:, 0]), (0, cam.width)) and np.allclose(np.sort(px[:, 1]), (0, cam.height))
            r, u, f = cam.basis()
            assert np.allclose(np.cross(r, u), -f)              # the basis of an ordinary look-at camera: no mirror
            a = "xyz".index(axis)
            assert abs(f[a]) == 1.0 and r["xyz".index("y" if axis == "x" else "x")] == 1.0
            assert u["xyz".index("y" if axis == "z" else "z")] == (-1.0 if orientation == "numpy" else 1.0)
    cam = E.Camera.face_on(lo, hi, axis=0, max_pixels=200, border=5)           # array axis 0 = z; the longer side gets 200 px
    assert (cam.width, cam.height) == (210, 110) and math.isclose(cam.pixel_size(), 0.02)
    cam = E.Camera.for_slice(GRID, "y", "physical")             # a 6 x 30 nm voxel at 6 nm per pixel: 1 x 5 pixels
    assert (cam.width, cam.height) == (NX, 5 * NZ) and math.isclose(cam.pixel_size(), 0.006)
    cam = E.Camera.for_slice(GRID, "y", "index")                # index space: every voxel one pixel
    assert (cam.width, cam.height) == (NX, NZ) and math.isclose(cam.pixel_size(), 0.006)


# =================================================================================================
# 13. the package types in a display space
# =================================================================================================
def test_wiremesh_skeleton_and_gaussians_move_into_the_display_space():
    grid = VoxelGrid((10, 20, 20), (30.0, 6.0, 6.0))            # 120 x 120 x 300 nm
    v_nm = np.array([(0, 0, 0), (120, 0, 0), (0, 120, 0), (0, 0, 300)], float)
    mesh = Mesh(v_nm, np.array([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)], np.int32))
    skel = Skeleton(v_nm, np.array([(0, 1), (0, 2), (0, 3)], np.int32), np.array([10.0, 5.0, 5.0, 5.0]))
    cov = np.diag([10.0, 10.0, 50.0]) ** 2
    gm = Gaussians(np.array([(60.0, 60.0, 150.0)]), cov[None], np.array([1.0]))
    for space, top in (("physical", 0.300), ("index", 0.060)):  # index space: z is drawn 5 times shorter
        s = E.Scene()
        assert s.add_wiremesh(mesh, grid, space) == 4
        assert np.allclose(s.bounds()[0], 0.0) and np.allclose(s.bounds()[1], (0.120, 0.120, top))
        squash = np.array([1.0, 1.0, top / 0.300])              # the tubes are squashed along z like everything else
        s = E.Scene()
        s.add_skeleton(skel, grid, space, radius=0.002)
        assert np.allclose(s.bounds()[0], -0.002 * squash) and np.allclose(s.bounds()[1], np.array([0.120, 0.120, top]) + 0.002 * squash)
        s = E.Scene()
        s.add_skeleton(skel, grid, space)                       # radii from the skeleton: 10 nm at the root
        assert np.allclose(s.bounds()[0], -0.010 * squash)
        s = E.Scene()
        assert s.add_gaussian_mixture(gm, grid, space) == 1
        mu, w = s._g[0][0], s._g[0][1]
        sigma_z = 0.050 * top / 0.300                           # the covariance is scaled per axis: S -> A S A
        assert np.allclose(mu, (0.060, 0.060, top / 2)) and np.allclose(np.linalg.inv(w[0]) @ np.linalg.inv(w[0]).T,
                                                                     np.diag([0.010, 0.010, sigma_z]) ** 2)
        s = E.Scene()
        s.add_gaussian_mixture(gm, grid, space, sigma_surface=2.0)             # the 2-sigma ellipsoid
        assert np.allclose(s.bounds()[1], (0.080, 0.080, top / 2 + 2 * sigma_z), atol=1e-3)
        assert np.all(s.bounds()[1] <= np.array([0.080, 0.080, top / 2 + 2 * sigma_z]) + 1e-12)


def test_wiremesh_lines_and_gaussian_colours():
    grid = VoxelGrid((10, 20, 20), (30.0, 6.0, 6.0))
    mesh = Mesh(np.array([(0, 0, 150), (120, 0, 150), (120, 120, 150), (0, 120, 150)], float), np.array([(0, 1, 2), (0, 2, 3)], np.int32))
    s = E.Scene()
    s.add_wiremesh(mesh, grid, "physical", color=(1, 1, 1), wire_color=(1, 0, 0), wire_min_pixels=2.0, lit=False)
    cam = E.Camera.face_on((0, 0, 0), (0.120, 0.120, 0.300), axis="z", pixel_size=0.002)       # 60 x 60 px
    img = render(s, cam).rgba
    red = (img[..., 0] == 255) & (img[..., 1] == 0)
    assert red[30, 0] and red[30, 59] and red[0, 30] and red[59, 30] and red[30, 30] and not red[30, 10]   # border and diagonal
    gm = Gaussians(np.array([(60.0, 60.0, 150.0)]), (np.eye(3) * 400.0)[None], np.array([1.0]), colors_rgb=np.array([(0.0, 1.0, 0.0)]))
    s = E.Scene()
    s.add_gaussian_mixture(gm, grid, opacity=0.9)
    img = render(s, E.Camera.face_on((0, 0, 0), (0.120, 0.120, 0.300), axis="z", pixel_size=0.004, background=(0, 0, 0)))
    assert img.rgba[15, 15, 1] > 200 and img.rgba[15, 15, 0] == 0 and img.rgba[15, 15, 2] == 0


# =================================================================================================
# 14. small numpy helpers and PNG output
# =================================================================================================
def test_geometry_helpers():
    v, f = E.icosphere(2)
    assert f.shape == (20 * 4 ** 2, 3) and np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert len(E.unique_edges(f)) == 3 * len(f) // 2            # closed surface: E = 3F/2
    p, e = E.box_edges((0, 0, 0), (1, 2, 3))
    assert p.shape == (8, 3) and e.shape == (12, 2) and np.allclose(sorted(np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1)), [1] * 4 + [2] * 4 + [3] * 4)
    cov = rot((1, 2, 3), 50) @ np.diag([0.6, 0.25, 0.08]) ** 2 @ rot((1, 2, 3), 50).T
    r, sig = E.cov_to_axes([cov])
    assert np.allclose(sorted(sig[0]), [0.08, 0.25, 0.6]) and np.allclose(r[0] @ np.diag(sig[0] ** 2) @ r[0].T, cov)
    m = E.index_to_world((30.0, 6.0, 4.0), origin=(1, 2, 3))
    assert np.allclose(m @ [1, 1, 1, 1], (5, 8, 33, 1))         # (i, j, k) -> (x, y, z): the voxel size is reversed once, here
    with pytest.raises(ValueError):
        E.index_to_world((1, 1))
    colours = E.label_colors(np.arange(1, 200))
    assert colours.shape == (199, 3) and colours.min() >= 0.0 and colours.max() <= 1.0 and len(np.unique(colours, axis=0)) > 190


def test_png_bytes_round_trip(tmp_path):
    from PIL import Image as PILImage
    rgba = RNG.integers(0, 256, (7, 5, 4), dtype=np.uint8)
    assert np.array_equal(np.asarray(PILImage.open(io.BytesIO(E.encode_png(rgba)))), rgba)
    assert np.array_equal(np.asarray(PILImage.open(io.BytesIO(E.encode_png(rgba[..., :3])))), rgba[..., :3])
    with pytest.raises(ValueError):
        E.encode_png(rgba[..., 0])
    s = E.Scene()
    s.add_spheres([[0, 0, 0]], 1.0)
    img = render(s, E.Camera(width=20, height=12, ssaa=1))
    path = img.save(str(tmp_path / "picture.png"))
    assert np.array_equal(np.asarray(PILImage.open(path)), img.rgba)
    assert img._repr_png_()[:8] == b"\x89PNG\r\n\x1a\n"        # what Jupyter displays inline


@needs_cuda
def test_cpu_and_cuda_devices_agree():
    s = E.Scene()
    smp = s.add_volume_grid(EM, GRID, "physical")
    s.add_slice(smp, "z", 5)
    s.add_slice(smp, "x", 14)
    s.add_spheres([(0.05, 0.03, 0.25)], 0.04, color=(1, 0.6, 0.2, 0.6))
    s.add_wire_box(*s.volume_corners(smp))
    s.add_gaussians([(0.12, 0.09, 0.1)], covs=[np.diag([0.02, 0.02, 0.05]) ** 2], opacity=0.7)
    cam = E.Camera.fit(*s.volume_corners(smp), direction=(1, -0.8, 0.5), fov=30, width=128, height=128, ssaa=2)
    gpu, cpu = s.render(cam, device=DEVICE, hemisphere=0.5, fog=0.3), s.render(cam, device="cpu", hemisphere=0.5, fog=0.3)
    differ = np.abs(gpu.rgba.astype(int) - cpu.rgba.astype(int)).max(-1) > 1
    assert differ.mean() < 0.005                                # the two devices differ on a few edge pixels at most
    assert (gpu.object_id != cpu.object_id).mean() < 0.005


# =================================================================================================
# 15. figure.py: captions, panel letters, montages, scale bars (CPU only)
# =================================================================================================
def picture(h=60, w=80, value=None):
    a = RNG.integers(0, 256, (h, w, 4), dtype=np.uint8) if value is None else np.full((h, w, 4), value, np.uint8)
    a[..., 3] = 255
    return a


def test_figure_accepts_arrays_images_and_pil():
    rgba = picture()
    img = E.Image(rgba, np.zeros(rgba.shape[:2], np.float32), np.zeros(rgba.shape[:2], np.int32), rgba / 255.0)
    for source in (rgba, img, F.to_pil(rgba), rgba[..., :3], rgba[..., 0], rgba[..., :3] / 255.0, rgba[..., 0] > 100):
        out = F.to_pil(source)
        assert out.size == (80, 60) and out.mode in ("RGB", "RGBA")
    assert np.array_equal(F.to_array(img), rgba) and np.array_equal(F.to_array(rgba[..., :3]), rgba[..., :3])
    assert np.array_equal(F.to_array(rgba[..., :3] / 255.0), rgba[..., :3])     # float in [0, 1] -> floor(255 v + 0.5)
    grey = F.to_array(rgba[..., 0])
    assert grey.shape == (60, 80, 3) and np.array_equal(grey[..., 1], rgba[..., 0])
    for bad in (rgba.astype(np.uint16), rgba[..., :2], rgba[..., :3] * 2.0, np.zeros((0, 4, 3), np.uint8)):
        with pytest.raises((ValueError, TypeError)):
            F.to_pil(bad)


def test_figure_png_bytes_round_trip():
    from PIL import Image as PILImage
    rgba = picture()
    data = F.to_png_bytes(rgba)
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and np.array_equal(np.asarray(PILImage.open(io.BytesIO(data))), rgba)


def test_caption_adds_a_strip_and_leaves_the_picture_alone():
    rgba = picture(60, 200)
    top = F.to_array(F.caption(rgba, "index space", size=14))
    strip = top.shape[0] - 60
    assert top.shape[1] == 200 and strip >= 14 and np.array_equal(top[strip:], rgba)
    assert np.all(top[:strip, :, :3].max(axis=(0, 1)) == 255) and top[:strip, :, :3].min() < 80        # white strip, dark text
    bottom = F.to_array(F.caption(rgba, "index space", size=14, position="bottom"))
    assert np.array_equal(bottom[:60], rgba) and bottom.shape == top.shape
    two = F.to_array(F.caption(rgba, "first line\nsecond line", size=14))
    wrapped = F.to_array(F.caption(rgba, "a caption that is far too long to fit on a single line of this picture", size=14))
    assert two.shape[0] > top.shape[0] and wrapped.shape[0] > top.shape[0] and wrapped.shape[1] == 200
    assert F.to_array(F.caption(rgba, "short", size=14, lines=2)).shape == two.shape                   # room reserved for two lines
    assert np.array_equal(F.to_array(F.caption(rgba, "default size")), F.to_array(F.caption(rgba, "default size", size=12)))
    wide = picture(60, 640)                                     # default text height: 1/32 of the width, at least 12
    assert np.array_equal(F.to_array(F.caption(wide, "default size")), F.to_array(F.caption(wide, "default size", size=20)))
    with pytest.raises(ValueError):
        F.caption(rgba, "x", position="left")


def test_label_panel_and_letters():
    assert F.letters(4) == ["a", "b", "c", "d"] and F.letters(28)[-3:] == ["z", "aa", "ab"]
    rgba = picture(80, 120, value=(0, 0, 255, 255))
    out = F.to_array(F.label_panel(rgba, "b", size=20))
    changed = np.any(out != rgba, axis=-1)
    assert out.shape == rgba.shape and changed.any()
    ys, xs = np.nonzero(changed)
    assert ys.min() == 0 and xs.min() == 0 and ys.max() < 40 and xs.max() < 40                          # a small box in the top-left corner
    assert np.all(out[0, 0, :3] == 255) and out[ys.min():ys.max(), xs.min():xs.max(), :3].min() < 80    # white box, dark letter
    out = F.to_array(F.label_panel(rgba, "c", size=20, corner="bottom-right"))
    ys, xs = np.nonzero(np.any(out != rgba, axis=-1))
    assert ys.max() == 79 and xs.max() == 119 and ys.min() > 40 and xs.min() > 80
    wide = picture(80, 480)                                     # default letter height: 1/16 of the width, at least 14
    assert np.array_equal(F.to_array(F.label_panel(wide, "a")), F.to_array(F.label_panel(wide, "a", size=30)))
    assert np.array_equal(F.to_array(F.label_panel(rgba, "a")), F.to_array(F.label_panel(rgba, "a", size=14)))
    with pytest.raises(ValueError):
        F.label_panel(rgba, "a", corner="middle")


def test_montage_layout():
    a, b, c = picture(60, 80), picture(60, 80), picture(40, 50)
    out = F.to_array(F.montage([a, b, c], ncols=2, pad=4, background=(10, 20, 30)))
    assert out.shape == (4 + 2 * 64, 4 + 2 * 84, 4)
    assert np.array_equal(out[4:64, 4:84], a) and np.array_equal(out[4:64, 88:168], b)
    assert np.array_equal(out[68:108, 4 + 15:4 + 65], c)        # the smaller picture: centred, top-aligned, not rescaled
    assert np.all(out[0, 0, :3] == (10, 20, 30)) and np.all(out[120, 150, :3] == (10, 20, 30))
    row = F.to_array(F.montage([a, b, c]))                      # default: one row
    assert row.shape == (8 + 60 + 8, 8 + 3 * 88, 4)
    assert F.to_array(F.montage([a] * 5)).shape == (8 + 2 * 68, 8 + 3 * 88, 4)             # 5 pictures: 3 columns, 2 rows
    cap = F.to_array(F.montage([a, b], captions=["short", "a much longer caption that needs a second line"], size=12))
    plain = F.to_array(F.montage([a, b]))
    strip = cap.shape[0] - plain.shape[0]
    assert strip > 24 and cap.shape[1] == plain.shape[1]
    assert np.array_equal(cap[8 + strip:8 + strip + 60, 8:88], a) and np.array_equal(cap[8 + strip:8 + strip + 60, 96:176], b)   # equal strips
    lab = F.to_array(F.montage([a, b], labels=True))
    assert lab.shape == plain.shape and np.any(lab[8:30, 8:30] != plain[8:30, 8:30]) and np.array_equal(lab[40:, :], plain[40:, :])
    both = F.to_array(F.montage([a, b], captions=["short", "a much longer caption that needs a second line"], labels=True, size=12))
    assert both.shape == cap.shape and np.array_equal(both[8 + strip:], cap[8 + strip:])       # the letters are in the strips:
    assert np.any(both[:8 + strip] != cap[:8 + strip])                                          # no picture pixel is covered
    with pytest.raises(ValueError):
        F.montage([])
    with pytest.raises(ValueError):
        F.montage([a, b], captions=["only one"])


def test_side_by_side():
    a, b = picture(60, 80), picture(60, 80)
    out = F.to_array(F.side_by_side(a, b, pad=0))
    assert np.array_equal(out, np.concatenate([a, b], axis=1))
    assert np.array_equal(F.to_array(F.side_by_side([a, b], pad=0)), out)
    assert F.side_by_side(a, b, captions=["index", "physical"], labels=["a", "b"]).size[0] == 8 * 3 + 160


def test_round_and_formatted_lengths():
    assert [F.round_length(x) for x in (0.35, 3.07, 700, 1.0, 0.0199, 5)] == pytest.approx([0.2, 2, 500, 1.0, 0.01, 5])
    assert F.format_length(0.05) == "50 nm" and F.format_length(2) == "2 um" and F.format_length(0.5) == "500 nm"
    assert F.format_length(1.5) == "1.5 um" and F.format_length(1500) == "1.5 mm" and F.format_length(2).isascii()
    with pytest.raises(ValueError):
        F.round_length(0)


@pytest.mark.parametrize("position", ["bottom-right", "bottom-left", "top-right", "top-left"])
def test_scale_bar_overlay_has_the_right_number_of_pixels(position):
    rgba = picture(200, 400, value=(90, 90, 90, 255))
    pixel_size = 0.006                                          # micrometres per pixel: 400 px = 2.4 um
    out = F.to_array(F.scale_bar(rgba, pixel_size, position=position))         # default: round length <= a quarter of the width
    white = np.all(out[..., :3] == 255, axis=-1)
    longest = white.sum(1).max()
    assert longest == round(0.5 / pixel_size) == 83             # a 500 nm bar
    bar_rows = np.flatnonzero(white.sum(1) == longest)
    cols = np.flatnonzero(white[bar_rows[0]])
    assert np.all(np.diff(cols) == 1)                           # one unbroken bar
    assert (cols[-1] > 300) == ("right" in position) and (bar_rows[0] > 100) == ("bottom" in position)
    assert np.any(np.all(out[..., :3] == 0, axis=-1))           # black outline
    out = F.to_array(F.scale_bar(rgba, pixel_size, length=1.0, label="", color=(255, 255, 0)))
    yellow = np.all(out[..., :3] == (255, 255, 0), axis=-1)
    assert yellow.sum(1).max() == round(1.0 / pixel_size) and len(np.flatnonzero(yellow.any(1))) == yellow.sum(0).max()   # only the bar
    with pytest.raises(ValueError, match="fit"):
        F.scale_bar(rgba, pixel_size, length=5.0)
    with pytest.raises(ValueError):
        F.scale_bar(rgba, 0.0)
    with pytest.raises(ValueError):
        F.scale_bar(rgba, pixel_size, position="center")


def test_scale_bar_overlay_on_a_rendered_slice():
    """The overlay agrees with the scene: a 60 nm bar on a face-on slice is 10 voxels of 6 nm long."""
    s = E.Scene()
    s.add_slice(s.add_volume_grid(EM, GRID, "physical"), "z", 5)
    cam = E.Camera.for_slice(GRID, "z", "physical", pixels_per_voxel=8)        # 0.75 nm per pixel
    img = render(s, cam)
    out = F.to_array(F.scale_bar(img, cam.pixel_size(), length=0.060, label="60 nm", color=(255, 255, 0)))
    yellow = np.all(out[..., :3] == (255, 255, 0), axis=-1)
    assert yellow.sum(1).max() == 80                            # 10 voxels x 8 px
    assert out.shape == img.rgba.shape


def test_show_returns_a_pil_image_outside_jupyter_and_displays_inside(monkeypatch):
    from PIL import Image as PILImage
    rgba = picture()
    out = F.show(rgba)                                          # plain Python: the picture comes back
    assert isinstance(out, PILImage.Image) and np.array_equal(np.asarray(out), rgba)
    IPython = pytest.importorskip("IPython")
    shown = []

    class FakeKernelShell:
        kernel = object()

    monkeypatch.setattr(IPython, "get_ipython", lambda: FakeKernelShell())
    monkeypatch.setattr(IPython.display, "display", lambda obj: shown.append(obj))
    assert F.show(rgba) is None                                 # in a notebook: displayed once, nothing returned
    assert len(shown) == 1 and np.array_equal(np.asarray(shown[0]), rgba)


# =================================================================================================
# 16. independent review checks on an ASYMMETRIC phantom: 11 x 7 x 5 voxels of 6 x 8 x 30 nm
#     (dx, dy and dz all different, nx, ny and nz all different, dy/dx not a whole number), so
#     that any swap of two axes, any mirror image and any half-voxel shift changes the numbers.
# =================================================================================================
P_SHAPE = (5, 7, 11)                                                # (nz, ny, nx)
P_GRID = VoxelGrid(P_SHAPE, (30.0, 8.0, 6.0))                       # (dz, dy, dx) in nm
P_D = np.array([6.0, 8.0, 30.0])                                    # (dx, dy, dz) in nm, written out by hand
P_N = np.array([11, 7, 5])                                          # (nx, ny, nz)
P_RAW = np.random.default_rng(123).integers(0, 256, P_SHAPE, dtype=np.uint8)


def drawn_nm(space):
    """The phantom voxel as drawn, (x, y, z) in nm -- written out, not taken from grid.py."""
    return P_D if space == "physical" else np.array([6.0, 6.0, 6.0])


def columns_rows(axis):
    """World axes that run along the image columns and rows of a face-on picture of a slice."""
    return {"z": (0, 1), "y": (0, 2), "x": (1, 2)}[axis]


def nearest_box(o, d, lo, hi):
    """Slab method in float64: for every ray the index of the nearest box (or -1) and its depth."""
    best, best_t = np.full(o.shape[:-1], -1), np.full(o.shape[:-1], np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for n in range(len(lo)):
            t0, t1 = (lo[n] - o) / d, (hi[n] - o) / d
            t_in, t_out = np.minimum(t0, t1).max(-1), np.maximum(t0, t1).min(-1)
            closer = (t_out > t_in) & (t_out > 0) & (t_in < best_t)
            best_t[closer], best[closer] = t_in[closer], n
    return best, best_t


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("direction", [(1, -1.3, 0.7), (-0.6, 1.0, -0.9), (0.2, 0.3, 1.0)])
def test_review_chiral_bricks_in_perspective_against_a_slab_ray_caster(space, direction):
    """A chiral set of voxels (it differs from each of its mirror images) as bricks, seen in
    perspective from three sides: every pixel shows the brick that a float64 slab-method ray
    caster finds, at the same depth.  A swap of x and y, of sizes or of counts, a mirror image
    or a half-voxel shift would change thousands of pixels."""
    kji = np.array([(1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 3, 3), (2, 2, 3), (3, 2, 3), (1, 4, 3), (1, 2, 7), (4, 5, 9)])
    bricks = Bricks(kji=kji, centers_xyz_nm=(kji[:, ::-1] + 0.5) * P_D, size_xyz_nm=P_D)
    d = drawn_nm(space) / 1000.0                                # micrometres
    s = E.Scene()
    s.add_bricks(bricks, P_GRID, space, object_id=10 + np.arange(len(kji)), lit=False)
    cam = E.Camera.fit(np.zeros(3), P_N * d, direction=direction, fov=30, width=300, height=220, ssaa=1)
    img = render(s, cam)
    best, t = nearest_box(*rays(cam), kji[:, ::-1] * d, (kji[:, ::-1] + 1) * d)
    exp = np.where(best >= 0, 10 + best, -1)
    sure = np.ones(exp.shape, bool)                             # pixels whose 3 x 3 neighbourhood shows one brick (or none)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sure &= np.roll(np.roll(exp, dy, 0), dx, 1) == exp
    hit = sure & (best >= 0)
    assert hit.sum() > 500 and len(np.unique(exp[hit])) >= 6
    assert int(((img.object_id != exp) & sure).sum()) == 0
    assert np.abs(img.depth[hit] - t[hit]).max() < 1e-5


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("axis", ["z", "y", "x"])
def test_review_five_representations_register_on_one_voxel(space, axis):
    """Voxel [k=3, j=2, i=8] drawn five ways, face-on at 2 nm per pixel: as a label slice, as a
    brick, as a mesh of its box, and with a skeleton node and a Gaussian at its centre.  All five
    land on the same pixel rectangle, [i*dx, (i+1)*dx] x ... of the voxel as drawn."""
    k, j, i = 3, 2, 8
    idx = np.array([i, j, k])
    d = drawn_nm(space)
    u, v = columns_rows(axis)
    ppv = 4 if (space, axis) == ("physical", "x") else 3        # the finer in-plane edge (6 or 8 nm) / 2 nm
    cam = E.Camera.for_slice(P_GRID, axis, space, pixels_per_voxel=ppv, background=(0, 0, 0))
    assert math.isclose(cam.pixel_size() * 1000, 2.0)
    assert (cam.width, cam.height) == (round(P_N[u] * d[u] / 2), round(P_N[v] * d[v] / 2))
    expect = np.zeros((cam.height, cam.width), bool)
    expect[round(idx[v] * d[v] / 2):round((idx[v] + 1) * d[v] / 2), round(idx[u] * d[u] / 2):round((idx[u] + 1) * d[u] / 2)] = True
    centre_px = (idx[[u, v]] + 0.5) * d[[u, v]] / 2             # (column, row) of the voxel centre
    lo_nm, hi_nm, centre_nm = idx * P_D, (idx + 1) * P_D, (idx + 0.5) * P_D        # true physical nm

    lab = np.zeros(P_SHAPE, np.uint16)
    lab[k, j, i] = 77
    s = E.Scene()
    s.add_slice(s.add_volume_grid(lab, P_GRID, space, labels=True), axis, int(idx["xyz".index(axis)]), object_id=1)
    assert np.array_equal(render(s, cam).object_id == 1, expect)                    # 1. label slice (id 0 is transparent)

    s = E.Scene()
    s.add_bricks(Bricks(np.array([[k, j, i]]), centre_nm[None], P_D), P_GRID, space, object_id=2)
    assert np.array_equal(render(s, cam).object_id == 2, expect)                    # 2. brick

    corners = np.array([[(lo_nm, hi_nm)[(n >> a) & 1][a] for a in range(3)] for n in range(8)])
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    faces = np.array([t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))], np.int32)
    s = E.Scene()
    s.add_wiremesh(Mesh(corners, faces), P_GRID, space, object_id=3)
    assert np.array_equal(render(s, cam).object_id == 3, expect)                    # 3. mesh with vertices in nm

    def centroid_px(weight):
        rows, cols = np.mgrid[0:cam.height, 0:cam.width]
        return np.array([(weight * (cols + 0.5)).sum(), (weight * (rows + 0.5)).sum()]) / weight.sum()

    other_nm = (np.array([1, 5, 1]) + 0.5) * P_D                # centre of voxel [k=1, j=5, i=1]
    skeleton = Skeleton(np.stack([centre_nm, other_nm]), np.array([(0, 1)], np.int32), np.array([4.0, 4.0]))
    s = E.Scene()
    s.add_skeleton(skeleton, P_GRID, space, min_pixels=0.0, color=(1, 1, 1), lit=False)      # 4. a capsule of 4 nm radius between two centres
    ends = np.stack([centre_nm, other_nm]) / P_D * d / 1000.0   # nm -> index -> scene units as drawn
    radius = 0.004 * d / P_D                                    # the radius is squashed with the voxel, per axis
    assert np.allclose(s.bounds()[0], ends.min(0) - radius, atol=1e-9) and np.allclose(s.bounds()[1], ends.max(0) + radius, atol=1e-9)
    cam.ssaa = 4                                                # coverage of the thin capsule: 16 rays per pixel
    middle_px = 0.5 * (centre_px + (np.array([1, 5, 1])[[u, v]] + 0.5) * d[[u, v]] / 2)
    assert np.abs(centroid_px(render(s, cam).rgba_float[..., 0].astype(np.float64)) - middle_px).max() < 0.3
    cam.ssaa = 1

    gaussian = Gaussians(centre_nm[None], (np.diag([1.5, 2.0, 2.5]) ** 2)[None], np.array([1.0]))
    s = E.Scene()
    s.add_gaussian_mixture(gaussian, P_GRID, space, opacity=0.9, color=(1, 1, 1))  # 5. Gaussian
    s.build(DEVICE, gaussian_cutoff=6.0)
    alpha = render(s, cam, gaussian_min_alpha=0.0).rgba_float[..., 0].astype(np.float64)
    assert np.abs(centroid_px(alpha) - centre_px).max() < 0.02


@pytest.mark.parametrize("space", SPACES)
def test_review_gaussian_covariance_against_a_pushed_forward_point_cloud(space):
    """No formula for covariances is used here.  400 000 points drawn from a tilted Gaussian in
    physical nm are moved point by point (nm -> voxel index -> nm as drawn); the picture of the
    Gaussian must have their mean and their second moments, seen along z, y and x.  In index
    space the three axes are scaled by 1, 3/4 and 1/5: one scalar factor cannot pass."""
    r = rot((2, -1, 3), 40)
    cov = r @ np.diag([7.0, 4.0, 2.0]) ** 2 @ r.T               # nm^2
    mean = np.array([33.0, 28.0, 75.0])                         # nm
    points = np.random.default_rng(1).multivariate_normal(mean, cov, 400000) / P_D * drawn_nm(space)
    s = E.Scene()
    s.add_gaussian_mixture(Gaussians(mean[None], cov[None], np.array([1.0])), P_GRID, space, opacity=0.9, color=(1, 1, 1))
    s.build(DEVICE, gaussian_cutoff=6.0)
    for axis in "zyx":
        u, v = columns_rows(axis)
        cam = E.Camera.for_slice(P_GRID, axis, space, pixels_per_voxel=12, background=(0, 0, 0))
        px = cam.pixel_size() * 1000.0                          # nm per pixel
        alpha = render(s, cam, gaussian_min_alpha=0.0).rgba_float[..., 0].astype(np.float64)
        rows, cols = np.mgrid[0:cam.height, 0:cam.width]
        q = np.stack([(cols + 0.5) * px, (rows + 0.5) * px], -1)        # (u, v) position of every pixel, nm as drawn
        centroid = (alpha[..., None] * q).sum((0, 1)) / alpha.sum()
        moments = np.einsum("yx,yxa,yxb->ab", alpha, q - centroid, q - centroid) / alpha.sum()
        expect = np.cov(points[:, [u, v]].T)
        assert np.abs(centroid - points[:, [u, v]].mean(0)).max() < 0.1
        assert np.abs(moments - expect).max() < 0.02 * np.abs(expect).max()
    surface = E.Scene()                                         # the 2-sigma ellipsoid is the image of the physical one
    surface.add_gaussian_mixture(Gaussians(mean[None], cov[None], np.array([1.0])), P_GRID, space, sigma_surface=2.0)
    half = 0.5 * (surface.bounds()[1] - surface.bounds()[0]) * 1000.0
    assert np.allclose(half, 2.0 * points.std(0), rtol=0.02)


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("orientation", ["numpy", "physical"])
def test_review_phantom_slices_are_the_numpy_slices_at_2_nm_per_pixel(space, orientation):
    """6 x 8 x 30 nm voxels at 2 nm per pixel: a voxel is 3 x 4 x 15 pixels (3 x 3 x 3 in index
    space), and every face-on picture is the numpy slice, or its up-down flip with the row axis up."""
    d = drawn_nm(space)
    s = E.Scene()
    smp = s.add_volume_grid(P_RAW, P_GRID, space)
    for axis, index in (("z", 3), ("y", 4), ("x", 9)):
        u, v = columns_rows(axis)
        s.clear_geometry()
        s.add_slice(smp, axis, index)
        cam = E.Camera.for_slice(P_GRID, axis, space, orientation, pixels_per_voxel=round(min(d[u], d[v]) / 2))
        img = render(s, cam)
        block = np.repeat(np.repeat(numpy_slice(P_RAW, axis, index), round(d[v] / 2), axis=0), round(d[u] / 2), axis=1)
        assert np.array_equal(img.rgba[..., 0], block if orientation == "numpy" else block[::-1])


def test_review_voxel_edges_that_are_not_a_whole_number_of_pixels():
    """4 x 4 x 33 nm voxels (the H01 sample) at 4 nm per pixel: 3 layers = 99 nm = 24.75 pixels.
    The picture gets 25 rows of exactly 4 nm, centred on the slice; every pixel shows the voxel
    under its centre, so the layers are 8, 9 and 8 pixels tall; no pixel is left empty."""
    vol = np.random.default_rng(2).integers(0, 256, (3, 5, 6), dtype=np.uint8)
    grid = VoxelGrid(vol.shape, (33.0, 4.0, 4.0))
    s = E.Scene()
    s.add_slice(s.add_volume_grid(vol, grid), "y", 2, object_id=7)
    cam = E.Camera.for_slice(grid, "y", background=(1, 0, 1))
    assert (cam.width, cam.height) == (6, 25) and math.isclose(cam.pixel_size(), 0.004)
    assert abs(cam.ortho_height - 0.099) < 0.5 * cam.pixel_size()           # the frame is within half a pixel of the slice
    img = render(s, cam)
    z_nm = -0.5 + (np.arange(25) + 0.5) * 4.0                   # centre of every pixel row: the 100 nm frame is centred on 0..99 nm
    layer = np.floor(z_nm / 33.0).astype(int)
    assert np.bincount(layer).tolist() == [8, 9, 8]
    assert np.all(img.object_id == 7) and np.array_equal(img.rgba[..., 0], vol[layer, 2, :])
    with pytest.raises(ValueError, match="whole number"):
        E.Camera.for_slice(grid, "y", pixels_per_voxel=1.5)     # would give voxels of unequal size
    with pytest.raises(ValueError, match="whole number"):
        E.Camera.for_slice(grid, "y", pixels_per_voxel=0.5)     # pixel centres would fall on voxel boundaries


def test_review_slice_index_and_sampler_are_checked():
    s = E.Scene()
    smp = s.add_volume_grid(P_RAW, P_GRID)
    with pytest.raises(ValueError, match="whole number"):
        s.add_slice(smp, "z", 2.7)                              # would silently draw layer 3
    for unknown in (-1, 1, 0.0, None):                          # -1 would silently pick the last volume and draw a white quad
        with pytest.raises(ValueError, match="unknown sampler"):
            s.add_slice(unknown, "z", 2)
        with pytest.raises(ValueError, match="unknown sampler"):
            s.volume_corners(unknown)
    assert s.num_triangles == 0
    assert s.add_slice(smp, "z", np.int64(2)) == 2 and s.add_slice(np.int64(smp), "z", 2.0) == 2


def test_review_an_array_changed_in_place_is_not_served_stale():
    vol = P_RAW.copy()
    cam = E.Camera.for_slice(P_GRID, "z", "index")
    s = E.Scene()
    before = s.add_volume_grid(vol, P_GRID, "index")
    assert s.add_volume_grid(vol, P_GRID, "index") != before and s.volume_nbytes == vol.size      # unchanged: stored once
    vol[:] = 255 - vol                                          # the same array object, new values
    after = s.add_volume_grid(vol, P_GRID, "index")
    assert s.volume_nbytes == 2 * vol.size                      # stored anew
    s.add_slice(after, "z", 1)
    assert np.array_equal(render(s, cam).rgba[..., 0], vol[1])
    s.clear_geometry()
    s.add_slice(before, "z", 1)                                 # the earlier sampler keeps the values it was registered with
    assert np.array_equal(render(s, cam).rgba[..., 0], 255 - vol[1])


def test_review_shared_icosphere_cannot_be_modified():
    v, f = E.icosphere(1)
    with pytest.raises(ValueError, match="read-only"):
        v *= 2.0
    with pytest.raises(ValueError, match="read-only"):
        f[0, 0] = 5
    assert np.allclose(np.linalg.norm(E.icosphere(1)[0], axis=1), 1.0)


def test_review_wrong_numbers_of_colours_ids_and_opacities_raise():
    """One value or exactly one per primitive.  Four per-VERTEX colours for two triangles were
    once accepted and coloured the faces with the first two."""
    verts, tris = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [(0, 1, 2), (1, 3, 2)]
    s = E.Scene()
    with pytest.raises(ValueError, match="colour"):
        s.add_mesh(verts, tris, color=np.ones((4, 3)))
    with pytest.raises(ValueError, match="object_id"):
        s.add_mesh(verts, tris, object_id=[1, 2, 3, 4])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        s.add_mesh(verts, tris, color=(255, 128, 0))            # 0..255 colours would draw white
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        s.add_mesh(verts, tris, color=(1.0, 0.5, 0.0, 1.5))
    with pytest.raises(ValueError, match="colour"):
        s.add_gaussians([(0, 0, 0)], covs=[np.eye(3)], color=np.ones((2, 3)))
    with pytest.raises(ValueError, match="opacity"):
        s.add_gaussians([(0, 0, 0)], covs=[np.eye(3)], opacity=[0.5, 0.5])
    with pytest.raises(ValueError, match="opacity"):
        s.add_gaussians([(0, 0, 0)], covs=[np.eye(3)], opacity=1.5)
    with pytest.raises(ValueError, match="no Gaussians"):
        s.add_gaussians(np.zeros((0, 3)), covs=np.zeros((0, 3, 3)))
    with pytest.raises(ValueError, match="lookup table"):
        s.add_volume(P_RAW, lut=E.lut_grey() * 255.0)           # a table in 0..255
    assert s.num_triangles == 0 and s.num_gaussians == 0 and len(s.samplers) == 0
    assert s.add_mesh(verts, tris, color=np.ones((2, 3)), object_id=[1, 2]) == 2


def test_review_empty_or_invalid_package_types_raise():
    empty = np.zeros((0, 3))
    s = E.Scene()
    with pytest.raises(ValueError, match="no faces"):
        s.add_wiremesh(Mesh(np.zeros((3, 3)), np.zeros((0, 3), np.int32)), P_GRID)
    with pytest.raises(ValueError, match="no edges"):
        s.add_skeleton(Skeleton(np.zeros((1, 3)), np.zeros((0, 2), np.int32), np.ones(1)), P_GRID)
    with pytest.raises(ValueError, match="no Gaussians"):
        s.add_gaussian_mixture(Gaussians(empty, np.zeros((0, 3, 3)), np.zeros(0)), P_GRID)
    flat = Gaussians(np.zeros((1, 3)), np.diag([4.0, 4.0, -1.0])[None], np.ones(1))       # not a covariance
    for kw in (dict(), dict(sigma_surface=2.0)):
        with pytest.raises(ValueError, match="positive definite"):
            s.add_gaussian_mixture(flat, P_GRID, **kw)
    good = Gaussians(np.zeros((1, 3)), np.eye(3)[None], np.ones(1))
    with pytest.raises(ValueError, match="sigma_surface"):
        s.add_gaussian_mixture(good, P_GRID, sigma_surface=0.0)
    assert s.num_triangles == 0 and s.num_gaussians == 0


def test_review_voxvalue_as_floats_and_through_the_table_give_the_same_picture():
    """voxvalue = clip((rawvalue - lo) / (hi - lo), 0, 1) reaches the picture two ways: as a float
    volume, and as the stored uint8 through lut_normalised(lo, hi).  Both are the 8-bit picture
    floor(255 * voxvalue + 0.5) of the formula."""
    lo, hi = 37.0, 201.0
    vox = np.clip((P_RAW.astype(np.float32) - np.float32(lo)) / np.float32(hi - lo), 0.0, 1.0)
    cam = E.Camera.for_slice(P_GRID, "z", "index")
    as_floats, through_table = E.Scene(), E.Scene()
    as_floats.add_slice(as_floats.add_volume_grid(vox, P_GRID, "index"), "z", 2)
    through_table.add_slice(through_table.add_volume_grid(P_RAW, P_GRID, "index", lut=E.lut_normalised(lo, hi)), "z", 2)
    exact = np.floor(np.clip((P_RAW[2].astype(np.float64) - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8)
    assert exact.min() == 0 and exact.max() == 255 and len(np.unique(exact)) > 40
    assert np.array_equal(render(as_floats, cam).rgba[..., 0], exact)
    assert np.array_equal(render(through_table, cam).rgba[..., 0], exact)
    assert np.abs(render(through_table, cam).rgba_float[..., 0] - vox[2]).max() < 1e-6        # the table is exact before rounding


def test_review_instance_ids_of_every_integer_type_get_the_same_colours():
    """Instance arrays come out of HDF5 files as uint16, uint32 or uint64: the picture must not
    depend on the type, and large ids that still fit into int32 must hash like the numpy twin."""
    lab = (np.random.default_rng(4).integers(1, 300, P_SHAPE) * (np.random.default_rng(5).random(P_SHAPE) > 0.3)).astype(np.uint16)
    cam = E.Camera.for_slice(P_GRID, "y", "index")

    def picture_of(labels):
        s = E.Scene()
        s.add_slice(s.add_volume_grid(labels, P_GRID, "index", labels=True, zero_alpha=1.0), "y", 3)
        return render(s, cam).rgba_float[..., :3]

    reference = np.where((lab[:, 3, :] == 0)[..., None], 0.06, E.label_colors(lab[:, 3, :]))
    for dtype in (np.uint16, np.uint32, np.uint64, np.int64):
        assert np.abs(picture_of(lab.astype(dtype)) - reference).max() < 1e-5
    for dtype, offset in ((np.uint32, 70_000), (np.int64, 2_000_000_000)):      # beyond 16 bits; just below 2**31
        big = lab.astype(dtype)
        big[lab > 0] += offset
        reference = np.where((big[:, 3, :] == 0)[..., None], 0.06, E.label_colors(big[:, 3, :]))
        assert np.abs(picture_of(big) - reference).max() < 1e-5
        assert np.abs(picture_of(big) - picture_of(lab)).max() > 0.1            # other ids, other colours


@pytest.mark.parametrize("space", SPACES)
def test_review_skeleton_tubes_are_squashed_with_the_voxels(space):
    """6 x 6 x 30 nm voxels, a skeleton edge along x with radii of 15 nm, at 0.5 nm per pixel.
    Physical space: a round tube, 30 nm = 60 pixels across seen along y and along z.
    Index space (z drawn 5 times too short): still 30 nm wide in y, but 6 nm = 12 pixels high in
    z -- the exact image of the round tube, so the skeleton stays inside its squashed instance."""
    skeleton = Skeleton(np.array([(30.0, 60.0, 180.0), (130.0, 60.0, 180.0)]), np.array([(0, 1)], np.int32), np.array([15.0, 15.0]))
    s = E.Scene()
    assert s.add_skeleton(skeleton, GRID, space, min_pixels=0.0, object_id=4) > 0
    squash = 1.0 if space == "physical" else 0.2
    lo, hi = s.bounds()
    assert np.allclose(lo * 1000, (15.0, 45.0, (180.0 - 15.0) * squash)) and np.allclose(hi * 1000, (145.0, 75.0, (180.0 + 15.0) * squash))
    for axis, across_nm, centre_nm in (("y", 30.0 * squash, 180.0 * squash), ("z", 30.0, 60.0)):
        cam = E.Camera.for_slice(GRID, axis, space, pixels_per_voxel=12)       # rows: z (seen along y) or y (seen along z)
        assert math.isclose(cam.pixel_size() * 1000, 0.5)
        hit = render(s, cam).object_id == 4
        rows = np.flatnonzero(hit[:, int(cam.project((0.080, 0.060, 0.180 * squash))[0, 0])])     # the column at x = 80 nm
        assert 0.94 * across_nm / 0.5 - 1 <= len(rows) <= across_nm / 0.5     # a 10-sided tube is a little narrower than its circle
        assert abs(rows.mean() + 0.5 - centre_nm / 0.5) < 0.6
    hair = E.Scene()                                            # the minimum on-screen width still holds for squashed tubes
    hair.add_skeleton(skeleton, GRID, space, radius=1e-6, min_pixels=4.0, object_id=4)
    cam = E.Camera.for_slice(GRID, "y", space)                  # 6 nm per pixel
    rows = np.flatnonzero(render(hair, cam).object_id[:, 13] == 4)             # the column at x = 81 nm
    assert len(rows) == 4 and abs(rows.mean() + 0.5 - 180.0 * squash / 6.0) < 0.01


def test_review_stretched_spheres_are_the_ellipsoids_of_add_ellipsoids():
    """``stretch`` gives the exact affine image, surface normals included: spheres stretched by
    (1, 0.75, 0.2) are drawn pixel for pixel like the ellipsoids A * r with their analytic normals."""
    centres, radii, a = np.array([(0.3, 0.4, 2.0), (-0.5, -0.2, 1.0)]), np.array([0.5, 0.3]), np.array([1.0, 0.75, 0.2])
    cam = E.Camera(eye=(1.5, -3.0, 1.2), target=(0, 0, 0.3), fov=30, width=160, height=120, ssaa=1)
    stretched, ellipsoids = E.Scene(), E.Scene()
    stretched.add_spheres(centres, radii, subdiv=3, stretch=a, color=(1.0, 0.8, 0.6), object_id=[1, 2])
    ellipsoids.add_ellipsoids(centres * a, radii[:, None, None] * np.diag(a), subdiv=3, color=(1.0, 0.8, 0.6), object_id=[1, 2])
    one, two = render(stretched, cam, hemisphere=0.5), render(ellipsoids, cam, hemisphere=0.5)
    assert (one.object_id > 0).sum() > 800 and len(np.unique(one.rgba[one.object_id > 0][:, 0])) > 50      # shaded, not flat
    same = one.object_id == two.object_id                       # (a silhouette pixel may flip with the last bit of a vertex)
    assert (~same).sum() <= 2 and np.abs(one.rgba_float - two.rgba_float)[same].max() < 1e-4
    assert np.allclose(stretched.bounds()[0], ((centres - radii[:, None]) * a).min(0))
    with pytest.raises(ValueError, match="stretch"):
        stretched.add_spheres(centres, radii, stretch=(1.0, 0.0, 1.0))


def test_review_wiremesh_line_width_is_in_scene_units():
    """A rectangle of 66 x 56 nm with 8 nm lines (0.008 scene units), at 1 nm per pixel: each
    face draws half the line inside its edge, 4 pixels."""
    mesh = Mesh(np.array([(0, 0, 60.0), (66, 0, 60.0), (66, 56, 60.0), (0, 56, 60.0)]), np.array([(0, 1, 2), (0, 2, 3)], np.int32))
    s = E.Scene()
    s.add_wiremesh(mesh, P_GRID, "physical", color=(1, 1, 1), wire_color=(1, 0, 0), wire_width=0.008, wire_min_pixels=0.0, lit=False)
    cam = E.Camera.for_slice(P_GRID, "z", "physical", pixels_per_voxel=6)
    assert math.isclose(cam.pixel_size() * 1000, 1.0)
    img = render(s, cam).rgba
    red = (img[..., 0] == 255) & (img[..., 1] == 0)
    assert int(red[28, :12].sum()) == 4 and int(red[:12, 50].sum()) == 4 and int(red[28, -12:].sum()) == 4


@pytest.mark.parametrize("ssaa", [1, 3])
def test_review_min_pixels_counts_output_pixels_not_sub_samples(ssaa):
    s = E.Scene()
    s.add_tubes([(0, 0, -20), (0, 0, 20)], [(0, 1)], 0.001, min_pixels=4.0, color=(1, 0, 0), lit=False)
    cam = E.Camera(eye=(0, -100, 0), target=(0, 0, 0), mode="orthographic", ortho_height=64.0, width=64, height=64, ssaa=ssaa,
                   background=(0, 0, 0))
    coverage = float(render(s, cam).rgba_float[32, :, 0].sum())                 # red pixels along the middle row
    assert abs(coverage - 4.0) < 0.35


def test_review_montage_captions_are_readable_on_a_dark_background():
    a = picture(60, 200, value=(90, 90, 90, 255))
    dark = F.to_array(F.montage([a, a], captions=["index space", "physical space"], background=(0, 0, 0), size=14))
    strip = dark[8:dark.shape[0] - 60 - 8, 8:208, :3]           # the caption strip of the first panel
    assert strip.max() > 200 and strip.min() == 0               # white text on the black strip
    bright = F.to_array(F.montage([a, a], captions=["index space", "physical space"], size=14))
    strip = bright[8:bright.shape[0] - 60 - 8, 8:208, :3]
    assert strip.min() < 80 and strip.max() == 255              # black text on the white strip
    chosen = F.to_array(F.montage([a, a], captions=["index space", "physical space"], background=(0, 0, 0), size=14, color=(255, 0, 0)))
    strip = chosen[8:chosen.shape[0] - 60 - 8, 8:208, :3]
    assert strip[..., 0].max() > 200 and strip[..., 1].max() == 0
    assert F.caption(a, "a text size may be a fraction", size=14.5).size[0] == 200      # (once a TypeError)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
