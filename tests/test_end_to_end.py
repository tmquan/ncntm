"""End-to-end test of ncntm.viz: from a sample file to a picture, through every module.

    python -m pytest tests/test_end_to_end.py        or        python tests/test_end_to_end.py

For each of the three core samples (snemi3d 30 x 6 x 6 nm, cremi_a 40 x 4 x 4 nm, fib25
8 x 8 x 8 nm) and its first skeleton subject (the neurite that runs across the sections):

    samples.load_sample                       rawvalue, voxvalue, instance, the mask of the subject
    convert.instance_to_*                     bricks, surface mesh, wiremesh, skeleton, Gaussians
    stage.Dataset                             one USD stage with all of them and the "space" variants
    Dataset.save  ->  Usd.Stage.Open          written to a .usdc file and read back from it
    usd_to_emray.build_scene  ->  render      drawn by the Warp ray caster in physical and index space
    figure.montage                            one picture per sample: physical beside index

Everything that is asserted is asserted on the stage that was READ BACK from the file.

The registration invariants (both spaces)
    * the box of the scene is grid.extent_nm(space) in scene units; in physical space that is
      grid.extent_xyz_nm / 1000, and the index-space z extent is the physical one divided by the
      anisotropy
    * every representation of the instance lies inside the bounding box of the instance plus one
      voxel: as geometry on the stage, and as pixels of the picture (position from the depth buffer)
    * seen along y with one pixel column per voxel, the bricks and the surface cover exactly the
      voxels of the mask; every skeleton node sits in a voxel of the mask; the Gaussian mixture
      has the mean and the covariance of the mask's voxels
    * a slice seen face-on is the numpy slice of the file

Sizes.  On a CUDA device the whole sample is used (about 3 x 3 x 3 um).  On Warp's CPU device
(NCNTM_DEVICE=cpu) a block of about 0.77 um around the middle of the subject is cut out and the
pictures are small, so that the same assertions run in reasonable time.

Environment
    NCNTM_SAMPLES    directory of the sample files (default <repository>/samples); the three
                     core samples must be there
    NCNTM_DEVICE     Warp device (default: Warp's choice)
    NCNTM_TEST_OUT   directory for the montages (default: a temporary directory)
"""
import atexit
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ncntm.viz.grid import NM_PER_UNIT, SPACES  # noqa: E402

CORE = ("snemi3d", "cremi_a", "fib25")
# written out by hand for the whole samples: voxel (dz, dy, dx) nm, box (x, y, z) um in both spaces
EXPECTED = {
    "snemi3d": dict(voxel_zyx_nm=(30.0, 6.0, 6.0), anisotropy=5.0, physical_um=(3.072, 3.072, 3.0), index_um=(3.072, 3.072, 0.6)),
    "cremi_a": dict(voxel_zyx_nm=(40.0, 4.0, 4.0), anisotropy=10.0, physical_um=(3.072, 3.072, 3.0), index_um=(3.072, 3.072, 0.3)),
    "fib25": dict(voxel_zyx_nm=(8.0, 8.0, 8.0), anisotropy=1.0, physical_um=(3.072, 3.072, 3.072), index_um=(3.072, 3.072, 3.072)),
}
SMALL_EDGE_NM = 768.0          # edge of the block used on the CPU device
ROWS = ("together", "slices", "bricks", "surface", "wiremesh", "skeleton", "gaussian")
TITLES = {"together": "slices, surface, wiremesh, skeleton and Gaussians together",
          "slices": "rawvalue (floor), voxvalue (right wall), instance labels (left wall)",
          "bricks": "instance as voxel bricks", "surface": "instance as a surface (35 % opaque)",
          "wiremesh": "wiremesh", "skeleton": "skeleton (red: branch points, blue: end points)",
          "gaussian": "gaussian mixture"}
GAUSSIAN_REACH = 3.3           # a Gaussian of opacity 0.8 is drawn out to sqrt(2 ln(0.8 * 255)) = 3.26 sigma


class Skip(Exception):
    """Raised by a test that cannot run here (a package is missing)."""


def need(*modules):
    """Import and return the named modules, or skip the test."""
    out = []
    for name in modules:
        try:
            out.append(__import__(name, fromlist=["_"]))
        except ImportError as error:
            if "PYTEST_CURRENT_TEST" in os.environ:
                import pytest

                pytest.skip(f"{name} is not available: {error}")
            raise Skip(f"{name} is not available: {error}")
    return out[0] if len(out) == 1 else out


try:
    import pytest

    each_sample = pytest.mark.parametrize("name", CORE)
except ImportError:                    # the script form below calls every test with every sample
    def each_sample(test):
        return test


@lru_cache(maxsize=None)
def tmp_dir() -> Path:
    path = Path(tempfile.mkdtemp(prefix="ncntm_test_end_to_end_"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def out_dir() -> Path:
    """Where the montages go."""
    path = Path(os.environ.get("NCNTM_TEST_OUT") or tmp_dir() / "pictures")
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=None)
def small() -> bool:
    """True on Warp's CPU device: small blocks and small pictures."""
    emray = need("ncntm.viz.emray")
    return emray.default_device().startswith("cpu")


# ---- from the file to the stage, once per sample -------------------------------------------------
def central_voxel(mask) -> np.ndarray:
    """[k, j, i] of the voxel of the mask that is nearest to the mean of its voxel indices."""
    kji = np.argwhere(mask)
    return kji[np.argmin(((kji - kji.mean(axis=0)) ** 2).sum(axis=1))]


def small_block(sample, subject):
    """A crop of about SMALL_EDGE_NM in every direction around the central voxel of the subject."""
    shape = np.array(sample.grid.shape_zyx)
    half = np.maximum(1, np.rint(0.5 * SMALL_EDGE_NM / np.array(sample.grid.voxel_size_zyx_nm))).astype(int)
    half = np.minimum(half, shape // 2)
    start = np.clip(central_voxel(sample.mask(subject)) - half, 0, shape - 2 * half)
    return sample.crop(*(slice(int(a), int(a + 2 * h)) for a, h in zip(start, half)))


@lru_cache(maxsize=None)
def built(name):
    """Sample -> containers -> stage -> file -> stage read back.  Returns everything the tests use;
    ``seconds`` holds the time of every step."""
    samples, C, S = need("ncntm.viz.samples", "ncntm.viz.convert", "ncntm.viz.stage")
    need("h5py", "zmesh", "kimimaro", "scipy")
    from pxr import Usd

    seconds = {}

    def timed(step, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        seconds[step] = time.perf_counter() - t0
        return result

    sample = timed("load_sample", samples.load_sample, name)
    subject = sample.skeleton_subjects[0]
    if small():
        sample = small_block(sample, subject)
    grid = sample.grid
    mask = timed("mask", sample.mask, subject)
    c = SimpleNamespace(
        bricks=timed("bricks", C.instance_to_bricks, mask, grid, instance_id=subject),
        mesh=timed("mesh", C.instance_to_mesh, mask, grid, instance_id=subject),              # every vertex on a voxel face
        wire=timed("wiremesh", C.instance_to_mesh, mask, grid, simplify_factor=20, instance_id=subject),
        skeleton=timed("skeleton", C.instance_to_skeleton, mask, grid, instance_id=subject),
        gaussians=timed("gaussians", C.instance_to_gaussians, mask, grid, k=16 if small() else 64, instance_id=subject))

    folder = tmp_dir() / name
    nz, ny, nx = grid.shape_zyx
    k_cut = int(central_voxel(mask)[0])                      # a section that cuts the subject
    t0 = time.perf_counter()
    stage = S.new_stage(folder / "authored.usda")
    ds = S.Dataset.from_sample(stage, sample)
    # seen from the overview camera, z = 0 is the floor and y = ny - 1, x = 0 are the two far walls
    ds.add_slices("rawvalue", z=0, upsample=1)
    ds.add_slices("voxvalue", y=ny - 1, upsample=1)
    ds.add_slices("instance", x=0, z=k_cut, upsample=1)
    ds.add_bricks(c.bricks)
    ds.add_surface(c.mesh, opacity=0.35)
    ds.add_wiremesh(c.wire, width_nm=0.003 * float(np.linalg.norm(grid.extent_xyz_nm)))      # about one pixel of the panels
    ds.add_skeleton(c.skeleton)
    ds.add_gaussians(c.gaussians)
    ds.add_frame()
    ds.author_camera()
    seconds["author stage"] = time.perf_counter() - t0
    saved = Path(timed("save", ds.save, folder / "saved" / f"{name}.usdc"))
    reopened = timed("reopen", Usd.Stage.Open, str(saved))
    assert os.path.samefile(reopened.GetRootLayer().realPath, saved)                 # really the file that was written,
    assert not os.path.samefile(saved, stage.GetRootLayer().realPath)                # not the layer it was authored in
    (dataset,) = S.find_datasets(reopened)

    # the box of the instance: of the piece that was converted (sample.mask keeps the largest piece of
    # an id; the table of the file, sample.bbox_zyx, is the box of all its pieces and contains this one)
    voxels = np.argwhere(mask)
    (z0, y0, x0), (z1, y1, x1) = voxels.min(axis=0), voxels.max(axis=0) + 1         # upper bounds exclusive
    all_pieces = np.array(sample.bbox_zyx(subject))
    assert np.all(all_pieces[:3] <= (z0, y0, x0)) and np.all((z1, y1, x1) <= all_pieces[3:])
    i = int(subject)
    walls = ["rawvalue/slice_z0000", f"voxvalue/slice_y{ny - 1:04d}", "instance/slice_x0000"]
    frame = ["frame/bbox", "frame/gnomon", "frame/scalebar"]
    shown = {"slices": [], "bricks": [f"instance/bricks_{i}"], "surface": [f"instance/surface_{i}"],
             "wiremesh": [f"wiremesh/wire_{i}"], "skeleton": [f"skeleton/tubes_{i}", f"skeleton/nodes_{i}"],
             "gaussian": [f"gaussian/splats_{i}"]}
    shown["together"] = shown["surface"] + shown["wiremesh"] + shown["skeleton"] + shown["gaussian"]
    return SimpleNamespace(
        name=name, sample=sample, grid=grid, subject=i, mask=mask, stage=reopened, ds=dataset, saved=saved, seconds=seconds,
        k_cut=k_cut, cut=f"instance/slice_z{k_cut:04d}", walls=walls, frame=frame, shown=shown,
        box_lo_ijk=np.array([x0, y0, z0], dtype=np.float64), box_hi_ijk=np.array([x1, y1, z1], dtype=np.float64),
        whole=not small(), **vars(c))


def instance_box(b, space):
    """(lo, hi) of the bounding box of the subject as drawn in ``space``, scene units (x, y, z):
    the low corner of its first voxel and the high corner of its last one."""
    return b.grid.index_to_nm(b.box_lo_ijk, space) / NM_PER_UNIT, b.grid.index_to_nm(b.box_hi_ijk, space) / NM_PER_UNIT


def voxel(b, space) -> np.ndarray:
    """One voxel as drawn in ``space``, scene units (x, y, z)."""
    return b.grid.display_voxel_size_xyz_nm(space) / NM_PER_UNIT


def show_only(b, names) -> None:
    """Make exactly the prims ``names`` ("layer/prim") of the dataset visible."""
    wanted = set(names)
    for layer, info in b.ds.layers().items():
        b.ds.set_visible(layer, True)
        for prim in info["prims"]:
            b.ds.set_visible(f"{layer}/{prim}", f"{layer}/{prim}" in wanted)
    missing = [what for what in wanted if not b.ds.find(what)]
    assert not missing, missing


def picture_height(b) -> int:
    return 150 if small() else 480


def detail() -> dict:
    """Tessellation of tubes and spheres in the ray caster."""
    return dict(tube_sides=4, sphere_subdiv=1) if small() else dict(tube_sides=10, sphere_subdiv=2)


@lru_cache(maxsize=None)
def panel(name, space, row):
    """One row of the montage in one space, drawn from the stage that was read back: ``pretty``
    for the montage (4 rays per pixel) and ``exact`` with one ray per pixel, whose depth and
    object-id buffers are measured."""
    U = need("ncntm.viz.usd_to_emray")
    b = built(name)
    b.ds.set_space(space)
    show_only(b, b.shown[row] + b.frame + (b.walls if row in ("together", "slices") else []))
    t0 = time.perf_counter()
    scene = U.build_scene(b.stage, **detail())
    build_seconds = time.perf_counter() - t0
    camera = U.camera_from_stage(b.stage, "/World/Cameras/overview", height=picture_height(b), ssaa=2)
    pretty = scene.render(camera)
    camera_exact = U.camera_from_stage(b.stage, "/World/Cameras/overview", height=picture_height(b), ssaa=1)
    exact = scene.render(camera_exact)
    b.ds.set_space("physical")
    return SimpleNamespace(pretty=pretty, exact=exact, camera=camera_exact, objects=dict(scene.objects), skipped=dict(scene.skipped),
                           triangles=scene.num_triangles, gaussians=scene.num_gaussians, build_seconds=build_seconds,
                           volume_boxes=[scene.volume_corners(n) for n in range(len(scene.samplers))])


def pixel_positions(camera, picture) -> np.ndarray:
    """World position (H, W, 3) of what each pixel shows: eye + ray direction * depth, for a
    perspective camera with one ray per pixel.  NaN where the pixel shows nothing."""
    right, up, forward = camera.basis()
    span_y = 2.0 * math.tan(math.radians(camera.fov) / 2.0)
    span_x = span_y * camera.width / camera.height
    u = (np.arange(camera.width) + 0.5) / camera.width - 0.5
    v = 0.5 - (np.arange(camera.height) + 0.5) / camera.height
    direction = forward[None, None, :] + right[None, None, :] * (u[None, :, None] * span_x) + up[None, None, :] * (v[:, None, None] * span_y)
    direction /= np.linalg.norm(direction, axis=2, keepdims=True)
    depth = np.where(np.isfinite(picture.depth), picture.depth, np.nan)
    return np.asarray(camera.eye)[None, None, :] + direction * depth[..., None]


def pixels_of(p, *prim_names) -> np.ndarray:
    """Boolean (H, W): the pixels of ``p.exact`` whose nearest surface belongs to one of the prims
    whose path ends with one of ``prim_names``."""
    ids = [oid for oid, o in p.objects.items() if any(o["path"].endswith("/" + n) or ("/" + n + "/") in o["path"] for n in prim_names)]
    assert ids, (prim_names, [o["path"] for o in p.objects.values()])
    return np.isin(p.exact.object_id, ids)


def world_points(prim, attribute="points") -> np.ndarray:
    """World positions (N, 3), scene units, of a point attribute as composed on the stage."""
    from pxr import Usd, UsdGeom

    m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).T
    return np.array(prim.GetAttribute(attribute).Get(), dtype=np.float64) @ m[:3, :3].T + m[:3, 3]


def outside(points, lo, hi) -> np.ndarray:
    """How far the points reach out of the box lo..hi along x, y, z (0 where they stay inside)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.maximum(0.0, np.maximum((lo - points).max(axis=0), (points - hi).max(axis=0)))


# ---- 1. the file that was written -----------------------------------------------------------------
@each_sample
def test_the_stage_read_back_from_the_file_has_every_layer_and_both_spaces(name):
    S = need("ncntm.viz.stage")
    from pxr import UsdGeom

    b = built(name)
    assert b.saved.suffix == ".usdc" and b.saved.stat().st_size > 10000
    assert UsdGeom.GetStageUpAxis(b.stage) == "Z" and b.stage.GetRootLayer().customLayerData["emScale"]["nanometersPerUnit"] == 1000.0
    assert b.ds.grid == b.grid                                                     # shape and voxel size survive the file
    assert list(b.ds.layers()) == list(S.LAYERS)                                   # all seven layers
    assert b.ds.prim.GetVariantSets().GetVariantSet("space").GetVariantNames() == ["index", "physical"]
    for space in SPACES:
        assert np.allclose(b.ds.voxel_scale(space), b.grid.display_voxel_size_xyz_nm(space))       # nm, in the file
    assert b.ds.prim.GetAttribute("em:sampleFile").Get() == f"{name}.h5"
    offset = np.rint(b.grid.nm_to_index(b.sample.origin_xyz_nm)).astype(int)                     # (i0, j0, k0) of a crop
    assert tuple(b.ds.prim.GetAttribute("em:voxelOffset").Get()) == tuple(offset) and (offset.any() == (not b.whole))
    textures = sorted(p.name for p in (b.saved.parent / "textures").glob("*.png"))               # copied with the file
    assert len(textures) == 4 and all(t.startswith(f"{name}_") for t in textures), textures
    if b.whole:
        e = EXPECTED[name]
        assert b.grid.voxel_size_zyx_nm == e["voxel_zyx_nm"] and b.grid.anisotropy == e["anisotropy"]
    print(f"\n  {name}: {b.sample.describe()}; subject {b.subject}: {int(b.mask.sum())} voxels, {len(b.bricks)} bricks, "
          f"{len(b.mesh.faces)} triangles, {len(b.wire.edges())} wires, {len(b.skeleton.vertices_xyz_nm)} nodes, "
          f"{len(b.gaussians)} Gaussians; file {b.saved.stat().st_size / 1e6:.1f} MB")


# ---- 2. the box of the scene ------------------------------------------------------------------------
@each_sample
def test_scene_box_is_the_grid_extent_and_index_space_is_flatter_by_the_anisotropy(name):
    from pxr import Usd, UsdGeom

    b = built(name)
    nz, ny, nx = b.grid.shape_zyx
    box = {}
    for space in SPACES:
        b.ds.set_space(space)
        extent = b.grid.extent_nm(space) / NM_PER_UNIT                             # (x, y, z) in scene units
        # (a) the transform on the stage: the far corner (nx, ny, nz) of the voxels frame
        assert np.abs((b.ds.index_to_world() @ [nx, ny, nz, 1.0])[:3] - extent).max() < 1e-9
        # (b) what USD computes as the bound of the three wall slices: together they span the box
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bound = [cache.ComputeWorldBound(b.ds.find(wall)).ComputeAlignedRange() for wall in b.walls]
        lo = np.min([np.array(r.GetMin()) for r in bound], axis=0)
        hi = np.max([np.array(r.GetMax()) for r in bound], axis=0)
        assert np.abs(lo[:2]).max() < 1e-9 and np.abs(hi - extent).max() < 1e-6, (space, lo, hi)
        # (c) the ray caster: the box of every volume it samples, and the bounds of the drawn slices
        p = panel(name, space, "slices")
        assert len(p.volume_boxes) == 3                                            # grey, viridis, labels
        for vlo, vhi in p.volume_boxes:
            assert np.abs(vlo).max() < 1e-9 and np.abs(vhi - extent).max() < 1e-6, (space, vlo, vhi)
        # (d) the picture: the frame box as seen in the depth buffer (its lines are 0.3 % of the diagonal wide)
        line = 0.5 * 0.003 * float(np.linalg.norm(b.grid.extent_xyz_nm)) / NM_PER_UNIT
        points = pixel_positions(p.camera, p.exact)[pixels_of(p, "bbox")]
        assert np.abs(points.max(axis=0) - extent).max() <= line + 1e-5, (space, points.max(axis=0), extent)
        assert np.abs(points.min(axis=0)).max() <= line + 1e-5, (space, points.min(axis=0))
        box[space] = extent
    b.ds.set_space("physical")
    assert np.array_equal(box["physical"], b.grid.extent_xyz_nm / NM_PER_UNIT)
    assert math.isclose(box["index"][2], box["physical"][2] / b.grid.anisotropy, rel_tol=1e-12)
    in_plane = b.grid.in_plane_nm / np.array(b.grid.voxel_size_xyz_nm)[:2]         # x, y: g / dx, g / dy (1 for these samples)
    assert np.allclose(box["index"][:2], box["physical"][:2] * in_plane)
    if b.whole:
        assert np.allclose(box["physical"], EXPECTED[name]["physical_um"]) and np.allclose(box["index"], EXPECTED[name]["index_um"])
    print(f"\n  {name}: physical {box['physical']} um, index {box['index']} um, anisotropy {b.grid.anisotropy:g}")


# ---- 3. every representation inside the box of the instance ----------------------------------------
@each_sample
def test_every_representation_lies_inside_the_instance_box_plus_one_voxel_on_the_stage(name):
    """Geometry read from the composed stage, in world coordinates, in both spaces."""
    from pxr import Usd, UsdGeom

    b = built(name)
    i = b.subject
    report = []
    for space in SPACES:
        b.ds.set_space(space)
        lo, hi = instance_box(b, space)
        one = voxel(b, space)
        a = b.grid.space_scale(space)

        bricks = UsdGeom.PointInstancer(b.ds.find(f"instance/bricks_{i}"))
        t = Usd.TimeCode.Default()
        matrices = np.array(bricks.ComputeInstanceTransformsAtTime(t, t, UsdGeom.PointInstancer.ExcludeProtoXform))
        to_world = np.array(UsdGeom.Xformable(bricks).ComputeLocalToWorldTransform(t))
        world = matrices @ to_world                                                # row-vector matrices, as USD has them
        centres, half = world[:, 3, :3], 0.5 * np.abs(world[:, :3, :3]).sum(axis=1)                # a unit cube per instance
        assert np.allclose(half, 0.5 * one)                                        # every brick is one voxel as drawn
        found = {
            "bricks": np.concatenate([centres - half, centres + half]),
            "surface": world_points(b.ds.find(f"instance/surface_{i}")),
            "wiremesh": world_points(b.ds.find(f"wiremesh/wire_{i}")),
            "skeleton": world_points(b.ds.find(f"skeleton/tubes_{i}")),
            "skeleton nodes": world_points(b.ds.find(f"skeleton/nodes_{i}"), "positions"),
            "gaussian": world_points(b.ds.find(f"gaussian/splats_{i}"), "em:gaussian:means"),
        }
        for what, points in found.items():
            out = outside(points, lo, hi)
            exact = what in ("bricks", "surface", "skeleton", "skeleton nodes", "gaussian")       # only simplification moves points
            allowed = 1e-6 * np.ones(3) if exact else one
            assert np.all(out <= allowed), (name, space, what, out, allowed)
            report.append(f"{what:15s} {space:8s}: out of the instance box by {np.round(out / one, 3)} voxels")
        # bricks and surface touch every face of the box: it is THEIR bounding box
        for what in ("bricks", "surface"):
            assert np.abs(found[what].min(axis=0) - lo).max() < 1e-6 and np.abs(found[what].max(axis=0) - hi).max() < 1e-6
        # the same instance in the other space is this one times A
        assert np.allclose(found["surface"], b.mesh.vertices_xyz_nm * a / NM_PER_UNIT, atol=1e-6)
    b.ds.set_space("physical")
    print(f"\n  {name}:\n  " + "\n  ".join(report))


@each_sample
def test_every_representation_lies_inside_the_instance_box_plus_one_voxel_in_the_picture(name):
    """The same statement about what the ray caster shows: the position of every pixel of a
    representation, from the depth buffer.  What may reach out of the box, and how far:

        bricks, surface   nothing (one voxel is allowed)
        wiremesh          half the width of a wire
        skeleton          a tube or node marker is as thick as the neurite: by its own radius where
                          a node lies near the border of the array (the tissue goes on there)
        gaussian          a Gaussian has no edge: it is drawn out to 3.3 standard deviations"""
    b = built(name)
    i = b.subject
    report = []
    for space in SPACES:
        b.ds.set_space(space)
        lo, hi = instance_box(b, space)
        one = voxel(b, space)
        nodes = world_points(b.ds.find(f"skeleton/nodes_{i}"), "positions")
        marker = 1.25 * b.skeleton.radii_nm[:, None] / NM_PER_UNIT                 # marker_scale 1.25, never larger in index space
        means = world_points(b.ds.find(f"gaussian/splats_{i}"), "em:gaussian:means")
        sigma = np.sqrt(np.einsum("nii->ni", np.array(b.ds.find(f"gaussian/splats_{i}").GetAttribute("em:gaussian:covariances").Get()))) / NM_PER_UNIT
        wire = 0.5 * b.ds.find(f"wiremesh/wire_{i}").GetAttribute("em:widthNm").Get() / NM_PER_UNIT
        reach = {
            "bricks": np.zeros(3), "surface": np.zeros(3), "wiremesh": np.full(3, wire),
            "skeleton": np.maximum(outside(nodes - marker, lo, hi), outside(nodes + marker, lo, hi)),
            "gaussian": np.maximum(outside(means - GAUSSIAN_REACH * sigma, lo, hi), outside(means + GAUSSIAN_REACH * sigma, lo, hi)),
        }
        b.ds.set_space("physical")
        for row, allowed in reach.items():
            p = panel(name, space, row)
            hit = pixels_of(p, *[what.split("/")[1] for what in b.shown[row]])
            assert hit.sum() >= 20, (name, space, row, hit.sum())                  # it is in the picture
            points = pixel_positions(p.camera, p.exact)[hit]
            assert np.isfinite(points).all()
            out = outside(points, lo, hi)
            assert np.all(out <= allowed + one), (name, space, row, out, allowed + one)
            report.append(f"{row:9s} {space:8s}: {hit.sum():6d} pixels, out of the instance box by {np.round(out / one, 2)} voxels "
                          f"(allowed {np.round((allowed + one) / one, 2)})")
    print(f"\n  {name}:\n  " + "\n  ".join(report))


# ---- 4. voxel-exact registration -------------------------------------------------------------------
@each_sample
def test_bricks_and_surface_seen_along_y_cover_exactly_the_voxels_of_the_mask(name):
    """Orthographic view along y with one pixel column per voxel (emray.Camera.for_slice): the
    picture rows are z.  A pixel must show the instance exactly where the mask has a voxel
    anywhere along y: ``mask.any(axis=1)``, each voxel drawn dz / g rows high in physical space
    and one row high in index space.  Bricks: exact, their faces lie between the rays.  Surface:
    marching cubes puts it between the voxel centres, so it encloses the centre line of every
    voxel of the mask and of no other voxel."""
    U, emray = need("ncntm.viz.usd_to_emray", "ncntm.viz.emray")
    b = built(name)
    i = b.subject
    nz, ny, nx = b.grid.shape_zyx
    column = b.mask.any(axis=1)                                                    # (nz, nx)
    for space in SPACES:
        rows = int(round(b.grid.display_voxel_size_xyz_nm(space)[2] / b.grid.in_plane_nm))       # picture rows per voxel
        assert rows == (int(round(b.grid.anisotropy)) if space == "physical" else 1)             # 5, 10, 1 for the three samples
        expected = np.repeat(column, rows, axis=0)
        camera = emray.Camera.for_slice(b.grid, "y", space)
        assert (camera.height, camera.width) == expected.shape == (nz * rows, nx)
        b.ds.set_space(space)
        for what in (f"instance/bricks_{i}", f"instance/surface_{i}"):
            show_only(b, [what])
            seen = U.build_scene(b.stage, **detail()).render(camera).object_id == 1
            wrong = int(np.count_nonzero(seen != expected))
            assert wrong == 0, (name, space, what, wrong, int(expected.sum()))
    b.ds.set_space("physical")


@each_sample
def test_skeleton_nodes_wires_and_the_cut_section_agree_with_the_mask(name):
    U, emray, S = need("ncntm.viz.usd_to_emray", "ncntm.viz.emray", "ncntm.viz.stage")
    from scipy import ndimage

    b = built(name)
    i = b.subject
    near = ndimage.binary_dilation(b.mask, structure=np.ones((3, 3, 3), bool))     # the mask and its 26 neighbours
    shape = np.array(b.grid.shape_zyx)
    for space in SPACES:
        b.ds.set_space(space)
        # every skeleton node, read from the stage, lies in a voxel of the mask (it is the centre of one)
        nodes_nm = world_points(b.ds.find(f"skeleton/nodes_{i}"), "positions") * NM_PER_UNIT
        kji = b.grid.nm_to_voxel(nodes_nm, space)
        assert np.all(b.mask[tuple(kji.T)]), (name, space, int((~b.mask[tuple(kji.T)]).sum()))
        assert np.allclose(b.grid.nm_to_index(nodes_nm, space) % 1.0, 0.5, atol=1e-3)            # ... its centre
        # every end of a wire lies in the mask or in a voxel next to it (the surface runs between the two)
        wire_nm = world_points(b.ds.find(f"wiremesh/wire_{i}")) * NM_PER_UNIT
        kji = np.clip(b.grid.nm_to_voxel(wire_nm, space), 0, shape - 1)
        assert np.all(near[tuple(kji.T)]), (name, space, int((~near[tuple(kji.T)]).sum()))
        # the instance slice that cuts the subject, face-on: the subject's colour exactly on mask[k]
        show_only(b, [b.cut])
        camera = emray.Camera.for_slice(b.grid, "z", space)
        picture = U.build_scene(b.stage).render(camera)                            # reads the HDF5 file itself
        labels = b.sample.instance[b.k_cut]
        assert np.array_equal(picture.rgba[..., :3], S.label_rgb8(labels)), (name, space)
        subject_colour = np.all(picture.rgba[..., :3] == S.label_rgb8(np.array([i]))[0], axis=-1)
        assert np.array_equal(subject_colour, labels == i) and (labels == i).any()
    b.ds.set_space("physical")


@each_sample
def test_slices_seen_face_on_are_the_numbers_of_the_file(name):
    """rawvalue: the stored uint8 number as a grey.  voxvalue: that number through the window of
    the file and the viridis map.  instance: the colour of the id.  In physical space a section
    (a row of the y and x slices) is dz / g picture rows high."""
    U, emray, S = need("ncntm.viz.usd_to_emray", "ncntm.viz.emray", "ncntm.viz.stage")
    b = built(name)
    nz, ny, nx = b.grid.shape_zyx
    raw, instance = b.sample.rawvalue, b.sample.instance
    viridis = S.slice_lut8("viridis", b.sample.norm_window)
    expected = {b.walls[0]: ("z", np.repeat(raw[0][..., None], 3, axis=2)),
                b.walls[1]: ("y", viridis[raw[:, ny - 1, :]]),
                b.walls[2]: ("x", S.label_rgb8(instance[:, :, 0]))}
    # the viridis row of a voxel is its voxvalue: round(255 * voxvalue), up to a tie of the rounding
    row = S.slice_lut8("grey", b.sample.norm_window)[raw[:, ny - 1, :]][..., 0].astype(int)
    assert np.abs(row - np.rint(b.sample.voxvalue()[:, ny - 1, :].astype(np.float64) * 255.0)).max() <= 1
    for space in SPACES:
        b.ds.set_space(space)
        rows = int(round(b.grid.display_voxel_size_xyz_nm(space)[2] / b.grid.in_plane_nm))
        for wall, (axis, colours) in expected.items():
            show_only(b, [wall])
            picture = U.build_scene(b.stage).render(emray.Camera.for_slice(b.grid, axis, space))
            assert np.array_equal(picture.rgba[..., :3], np.repeat(colours, 1 if axis == "z" else rows, axis=0)), (name, space, wall)
    b.ds.set_space("physical")


@each_sample
def test_gaussian_mixture_on_the_stage_has_the_moments_of_the_mask(name):
    """Mean and covariance of the whole mixture (law of total covariance) equal those of the
    voxels of the mask: centroid of the voxel centres, covariance of the centres plus the
    box term diag(dx^2, dy^2, dz^2) / 12.  In index space both are moved by A: A mu and A S A."""
    b = built(name)
    centres = b.grid.voxel_centers_nm(np.argwhere(b.mask))                         # (N, 3) nm
    mean_nm = centres.mean(axis=0)
    cov_nm2 = np.cov((centres - mean_nm).T, bias=True) + np.diag(b.grid.voxel_size_xyz_nm ** 2 / 12.0)
    prim = b.ds.find(f"gaussian/splats_{b.subject}")
    weights = np.array(prim.GetAttribute("em:gaussian:weights").Get(), dtype=np.float64)
    assert abs(weights.sum() - 1.0) < 1e-5
    for space in SPACES:
        b.ds.set_space(space)
        a = b.grid.space_scale(space)
        mu = np.array(prim.GetAttribute("em:gaussian:means").Get(), dtype=np.float64)              # nm, as drawn
        cov = np.array(prim.GetAttribute("em:gaussian:covariances").Get(), dtype=np.float64)
        mixture_mean = (weights[:, None] * mu).sum(axis=0) / weights.sum()
        spread = mu - mixture_mean
        mixture_cov = (weights[:, None, None] * (cov + spread[:, :, None] * spread[:, None, :])).sum(axis=0) / weights.sum()
        assert np.abs(mixture_mean - mean_nm * a).max() < 0.05, (name, space, mixture_mean, mean_nm * a)     # nm (weights are float32)
        wanted = cov_nm2 * a[:, None] * a[None, :]
        assert np.abs(mixture_cov - wanted).max() < 1e-4 * np.abs(wanted).max(), (name, space)
    b.ds.set_space("physical")


@each_sample
def test_a_block_converted_on_its_own_goes_back_to_its_place_by_its_origin(name):
    """Converters return coordinates relative to the array they were given.  For a crop,
    ``sample.origin_xyz_nm`` (nm, where the corner of its voxel [0, 0, 0] lies in the file) puts
    them back: geometry of the block + origin is geometry in the grid of the whole sample."""
    samples, C = need("ncntm.viz.samples", "ncntm.viz.convert")
    whole = samples.load_sample(name)
    subject = whole.skeleton_subjects[0]
    block = small_block(whole, subject)
    assert block.origin_xyz_nm.any()                                               # it is not at the corner of the file
    k0, j0, i0 = (int(n) for n in whole.grid.nm_to_voxel(block.origin_xyz_nm))     # first voxel of the block
    nz, ny, nx = block.grid.shape_zyx
    inside = whole.instance[k0:k0 + nz, j0:j0 + ny, i0:i0 + nx] == subject          # the subject inside the block, all pieces
    assert np.array_equal(block.instance == subject, inside)
    # bricks: every voxel of the block's instance, moved by the origin, is that voxel of the whole sample
    bricks = C.instance_to_bricks(block.instance == subject, block.grid, surface_only=False)
    kji = whole.grid.nm_to_voxel(bricks.centers_xyz_nm + block.origin_xyz_nm)
    assert np.array_equal(kji, bricks.kji + [k0, j0, i0]) and len(kji) == inside.sum()
    assert np.all(whole.instance[tuple(kji.T)] == subject)
    # surface, skeleton and Gaussians of the block against those of the whole sample
    mask = block.mask(subject)
    box = np.argwhere(mask)
    lo_nm = whole.grid.index_to_nm((box.min(axis=0) + [k0, j0, i0])[::-1])          # (x, y, z) nm in the whole sample
    hi_nm = whole.grid.index_to_nm((box.max(axis=0) + [k0, j0, i0] + 1)[::-1])
    mesh = C.instance_to_mesh(mask, block.grid)
    vertices = mesh.vertices_xyz_nm + block.origin_xyz_nm
    assert np.allclose(vertices.min(axis=0), lo_nm) and np.allclose(vertices.max(axis=0), hi_nm)
    skeleton = C.instance_to_skeleton(mask, block.grid)
    kji = whole.grid.nm_to_voxel(skeleton.vertices_xyz_nm + block.origin_xyz_nm)
    assert np.all(whole.instance[tuple(kji.T)] == subject)                         # every node in a voxel of the subject
    gaussian = C.single_gaussian(mask, block.grid)
    centroid = whole.grid.voxel_centers_nm(np.argwhere(mask) + [k0, j0, i0]).mean(axis=0)
    assert np.abs(gaussian.means_xyz_nm[0] + block.origin_xyz_nm - centroid).max() < 1e-6       # nm


# ---- 5. the pictures ---------------------------------------------------------------------------------
@each_sample
def test_montage_physical_beside_index(name):
    """One picture per sample for a person to look at: a row per representation, physical space
    on the left and index space on the right, drawn with the same camera."""
    F, S = need("ncntm.viz.figure", "ncntm.viz.stage")
    b = built(name)
    pictures, captions = [], []
    e = b.grid.extent_xyz_nm / NM_PER_UNIT
    told = {"physical": f"physical space, {e[0]:.2f} x {e[1]:.2f} x {e[2]:.2f} um",
            "index": f"index space, every voxel drawn as a {b.grid.in_plane_nm:g} nm cube: "
                     + ("the same picture, the voxels ARE cubes" if b.grid.is_isotropic else f"z {b.grid.anisotropy:g} times too short")}
    for row in ROWS:
        pair = {space: panel(name, space, row) for space in SPACES}
        for space in SPACES:
            p = pair[space]
            assert not p.skipped, p.skipped                                        # the adapter drew every prim it met
            assert p.pretty.rgba.shape == (picture_height(b), picture_height(b) * 4 // 3, 4)
            ink = np.any(p.pretty.rgba[..., :3] < 250, axis=-1).mean()
            assert ink > 0.002, (name, space, row, ink)                            # not an empty picture
            layers = {o["layer"] for o in p.objects.values()}
            wanted = {"frame"} | {what.split("/")[0] for what in b.shown[row]} | ({"rawvalue", "voxvalue", "instance"} if row in ("together", "slices") else set())
            assert layers == wanted, (name, space, row, layers)
            pictures.append(p.pretty)
            captions.append(f"{TITLES[row]}: {told[space]}")
        difference = np.abs(pair["physical"].pretty.rgba_float - pair["index"].pretty.rgba_float).max()
        if b.grid.is_isotropic:
            assert difference == 0.0, (name, row, difference)                      # index space IS physical space
        else:
            assert difference > 0.2, (name, row, difference)
    labels = S.format_length_nm(b.ds.find("frame/scalebar").GetAttribute("em:lengthNm").Get())
    montage = F.montage(pictures, captions=captions, ncols=2, labels=True)
    montage = F.caption(montage, f"{name}{'' if b.whole else ' (a block of it)'}: {b.sample.grid.describe()}; instance {b.subject}; "
                                 f"black bar and coloured axes (x red, y green, z blue): {labels} of tissue each", size=18)
    path = out_dir() / f"e2e_{name}{'' if b.whole else '_small'}.png"
    montage.save(path)
    assert path.stat().st_size > 20000
    width, height = montage.size
    assert width > 2 * pictures[0].rgba.shape[1] and height > len(ROWS) * pictures[0].rgba.shape[0]
    total = sum(panel(name, space, row).build_seconds for row in ROWS for space in SPACES)
    print(f"\n  {path}: {width} x {height} pixels; {total:.1f} s to build the {2 * len(ROWS)} scenes; "
          f"steps: " + ", ".join(f"{step} {seconds:.2f} s" for step, seconds in b.seconds.items()))


if __name__ == "__main__":
    failed = []
    for test_name, test in list(globals().items()):
        if test_name.startswith("test_") and callable(test):
            for sample_name in CORE:
                try:
                    test(sample_name)
                    print(f"PASS  {test_name}[{sample_name}]")
                except Skip as reason:
                    print(f"SKIP  {test_name}[{sample_name}]: {reason}")
                except Exception:
                    traceback.print_exc()
                    print(f"FAIL  {test_name}[{sample_name}]")
                    failed.append(f"{test_name}[{sample_name}]")
    print(f"\n{len(failed)} failed" if failed else "\nall tests passed")
    sys.exit(1 if failed else 0)
