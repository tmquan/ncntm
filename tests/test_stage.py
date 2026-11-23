"""Tests of ncntm.viz.stage (USD authoring) and ncntm.viz.usd_to_emray (drawing the stage with Warp).

    python -m pytest tests/test_stage.py        or        python tests/test_stage.py

Two stages are built once and shared by the tests:
    toy()    a 40 x 32 x 10 voxel volume of 6 x 6 x 30 nm with hand-made containers (fast)
    real()   samples/snemi3d.h5 with every layer for its first skeleton subject

Environment
    NCNTM_SAMPLES    directory of the sample files (default <repository>/samples)
    NCNTM_DEVICE     Warp device for the pictures (default: Warp's choice); with "cpu" the
                     real stage is built with fewer bricks and coarser tubes
    NCNTM_TEST_OUT   directory for the pictures of all layers (default: a temporary directory)

Tests that need a package that is not installed (warp, h5py, the converters' libraries) are
skipped, so the authoring tests also run in a plain OpenUSD environment.
"""
import atexit
import math
import os
import shutil
import sys
import tempfile
import traceback
import warnings
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pxr import Gf, Usd, UsdGeom, UsdShade  # noqa: E402

from ncntm.viz import stage as S  # noqa: E402
from ncntm.viz.grid import NM_PER_UNIT, SPACES, VoxelGrid  # noqa: E402
from ncntm.viz.types import Bricks, Gaussians, Mesh, Skeleton  # noqa: E402


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


def raises(error, fn, *args, **kwargs):
    """True if fn(*args, **kwargs) raises ``error`` (so the tests also run without pytest)."""
    try:
        fn(*args, **kwargs)
    except error:
        return True
    return False


@lru_cache(maxsize=None)
def tmp_dir() -> Path:
    path = Path(tempfile.mkdtemp(prefix="ncntm_test_stage_"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def out_dir() -> Path:
    """Where the pictures for inspection go."""
    path = Path(os.environ.get("NCNTM_TEST_OUT") or tmp_dir() / "pictures")
    path.mkdir(parents=True, exist_ok=True)
    return path


def world_matrix(prim) -> np.ndarray:
    """Local-to-world of a prim as a column-vector 4x4, straight from USD."""
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).T


def world_points(prim, attribute="points") -> np.ndarray:
    """World positions (N, 3), stage units, of a point attribute as composed on the stage."""
    m = world_matrix(prim)
    return np.array(prim.GetAttribute(attribute).Get(), dtype=np.float64) @ m[:3, :3].T + m[:3, 3]


def world_size(prim) -> np.ndarray:
    """Edge lengths (x, y, z) of the world bounding box of a prim, from UsdGeom.BBoxCache."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return np.array(cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize())


def instance_world_matrices(instancer_prim) -> np.ndarray:
    """(N, 4, 4) column-vector instance-to-world matrices of a PointInstancer, from USD."""
    instancer = UsdGeom.PointInstancer(instancer_prim)
    t = Usd.TimeCode.Default()
    local = np.array(instancer.ComputeInstanceTransformsAtTime(t, t, UsdGeom.PointInstancer.ExcludeProtoXform))
    return np.swapaxes(local @ world_matrix(instancer_prim).T, 1, 2)


def box_mesh(lo, hi, instance_id=0) -> Mesh:
    """A closed box as 12 triangles wound counter-clockwise seen from outside."""
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
             [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]]
    return Mesh(S._box_points(lo, hi), np.array(faces, dtype=np.int32), instance_id=instance_id)


# ---- the two shared stages -----------------------------------------------------------------
TOY_GRID = VoxelGrid((10, 32, 40), (30.0, 6.0, 6.0))


def toy_volumes():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, TOY_GRID.shape_zyx, dtype=np.uint8)
    instance = np.zeros(TOY_GRID.shape_zyx, dtype=np.uint8)
    instance[2:8, 8:24, 10:30] = 3                              # one block: z 2..7, y 8..23, x 10..29
    instance[0, 0, :5] = 7
    return raw, instance


def toy_containers(grid=TOY_GRID):
    """Hand-made containers for instance 3 of the toy volume, in the coordinates of ``grid``."""
    _, instance = toy_volumes()
    kji = np.argwhere(instance == 3)
    bricks = Bricks(kji, grid.voxel_centers_nm(kji), grid.voxel_size_xyz_nm, instance_id=3, n_voxels_total=len(kji))
    mesh = box_mesh(grid.index_to_nm((10, 8, 2)), grid.index_to_nm((30, 24, 8)), instance_id=3)
    # node 1 is a branch point (four edges), node 2 an ordinary node, the others are end points;
    # edges 0, 1, 4 run along x, edge 2 along y, edge 3 along z
    nodes_kji = np.array([[5, 16, 12], [5, 16, 18], [5, 16, 24], [5, 10, 18], [3, 16, 18], [5, 16, 28]])
    skeleton = Skeleton(grid.voxel_centers_nm(nodes_kji), np.array([[0, 1], [1, 2], [1, 3], [1, 4], [2, 5]], dtype=np.int32),
                        np.array([18.0, 30.0, 24.0, 12.0, 20.0, 15.0]), instance_id=3)
    covariances = np.array([np.diag([400.0, 100.0, 2500.0]), [[900.0, 300.0, 100.0], [300.0, 400.0, 50.0], [100.0, 50.0, 1600.0]]])
    gaussians = Gaussians(grid.voxel_centers_nm(np.array([[5, 16, 14], [4, 14, 26]])), covariances, np.array([0.6, 0.4]),
                          colors_rgb=np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]]), instance_id=3)
    return SimpleNamespace(bricks=bricks, mesh=mesh, skeleton=skeleton, gaussians=gaussians)


def build_toy(path, **gaussian_options):
    raw, instance = toy_volumes()
    stage = S.new_stage(path)
    ds = S.Dataset(stage, "toy", TOY_GRID, raw=raw, instance=instance, norm_window=(20.0, 200.0))
    c = toy_containers()
    slices = {kind: ds.add_slices(kind, z=5, y=16, x=20, upsample=3) for kind in ("rawvalue", "voxvalue", "instance")}
    prims = SimpleNamespace(
        slices=slices, bricks=ds.add_bricks(c.bricks), surface=ds.add_surface(c.mesh, opacity=0.4),
        wire=ds.add_wiremesh(c.mesh), frame=ds.add_frame())
    prims.tubes, prims.nodes = ds.add_skeleton(c.skeleton)
    prims.splats, prims.ellipsoids = ds.add_gaussians(c.gaussians, **gaussian_options)
    prims.camera = ds.author_camera().GetPrim()
    ds.save()
    return SimpleNamespace(stage=stage, ds=ds, prims=prims, path=Path(path), raw=raw, instance=instance, grid=TOY_GRID,
                           volumes={"toy": {"raw": raw, "instance": instance}}, **vars(c))


@lru_cache(maxsize=None)
def toy():
    return build_toy(tmp_dir() / "toy" / "toy.usda")


def device() -> str:
    U = need("ncntm.viz.usd_to_emray")
    return U.render_device()


@lru_cache(maxsize=None)
def real():
    """samples/snemi3d.h5 with every layer for its first skeleton subject."""
    samples, C = need("ncntm.viz.samples", "ncntm.viz.convert")
    need("h5py", "zmesh", "kimimaro")
    small = device().startswith("cpu")
    sample = samples.load_sample("snemi3d")
    subject = sample.skeleton_subjects[0]
    mask = sample.mask(subject)
    c = SimpleNamespace(
        bricks=C.instance_to_bricks(mask, sample.grid, max_bricks=2000 if small else 20000, instance_id=subject),
        mesh=C.instance_to_mesh(mask, sample.grid, instance_id=subject),                         # every vertex on a voxel face
        wire=C.instance_to_mesh(mask, sample.grid, simplify_factor=20, instance_id=subject),     # few enough edges to see
        skeleton=C.instance_to_skeleton(mask, sample.grid, instance_id=subject),
        gaussians=C.instance_to_gaussians(mask, sample.grid, instance_id=subject))
    path = tmp_dir() / "real" / "snemi3d.usda"
    stage = S.new_stage(path)
    ds = S.Dataset.from_sample(stage, sample)
    slices = {"rawvalue": ds.add_slices("rawvalue", z=50, y=300, x=200, upsample=2),
              "voxvalue": ds.add_slices("voxvalue", z=20, y=400, x=100, upsample=2),
              "instance": ds.add_slices("instance", z=80, y=256, x=320, upsample=2)}
    prims = SimpleNamespace(slices=slices, bricks=ds.add_bricks(c.bricks), surface=ds.add_surface(c.mesh, opacity=0.35),
                            wire=ds.add_wiremesh(c.wire), frame=ds.add_frame())
    prims.tubes, prims.nodes = ds.add_skeleton(c.skeleton)
    prims.splats, prims.ellipsoids = ds.add_gaussians(c.gaussians)
    prims.camera = ds.author_camera().GetPrim()
    ds.save()
    return SimpleNamespace(stage=stage, ds=ds, prims=prims, path=path, sample=sample, grid=sample.grid, subject=subject,
                           mask=mask, small=small, **vars(c))


def leaf_prims(built) -> list:
    """Every drawable prim that the add_* methods returned (slices, bricks, ..., frame parts)."""
    found = []

    def collect(value):
        if isinstance(value, Usd.Prim):
            if not value.IsA(UsdGeom.Camera):
                found.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(vars(built.prims))
    return found


class showing:
    """Context manager: only the given prims are visible, everything else the build functions
    returned is hidden; the visibility found is restored on exit."""

    def __init__(self, built, *prims):
        self.built, self.keep = built, [p.GetPath() for p in prims]

    def __enter__(self):
        self.saved = []
        for prim in leaf_prims(self.built):
            attribute = UsdGeom.Imageable(prim).CreateVisibilityAttr()
            self.saved.append((attribute, attribute.Get()))
            attribute.Set(UsdGeom.Tokens.inherited if prim.GetPath() in self.keep else UsdGeom.Tokens.invisible)
        return self

    def __exit__(self, *exc):
        for attribute, token in self.saved:
            attribute.Set(token)


# ---- colours, lookup tables, small helpers -------------------------------------------------
def test_srgb_and_linear_are_inverse():
    c = np.linspace(0.0, 1.0, 101)
    assert np.abs(S.linear_to_srgb(S.srgb_to_linear(c)) - c).max() < 1e-12
    assert abs(float(S.srgb_to_linear(0.5)) - 0.21404114) < 1e-7            # mid grey is 21 % linear


def test_colour_tables():
    grey = S.colormap_rgb8("grey")
    assert grey.shape == (256, 3) and grey.dtype == np.uint8
    assert np.array_equal(grey, np.repeat(np.arange(256)[:, None], 3, axis=1))
    viridis = S.colormap_rgb8("viridis")
    assert viridis.shape == (256, 3) and viridis.dtype == np.uint8
    assert tuple(viridis[0]) == (0x44, 0x01, 0x54) and tuple(viridis[255]) == (0xfd, 0xe7, 0x25)
    linear = S.srgb_to_linear(viridis / 255.0)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(luminance[2:] > luminance[:-2])                          # brighter from any row to the next but one
    assert np.diff(luminance).min() > -5e-4 and luminance[255] > 40 * luminance[0]      # single rows: 8-bit rounding
    assert np.abs(viridis.astype(int) - grey.astype(int)).max(axis=1).min() > 30     # never a grey
    assert raises(ValueError, S.colormap_rgb8, "jet")


def test_slice_lut_is_identity_for_rawvalue_and_windowed_for_voxvalue():
    assert np.array_equal(S.slice_lut8("grey", (0.0, 255.0)), S.colormap_rgb8("grey"))        # v -> (v, v, v)
    lut = S.slice_lut8("viridis", (20.0, 200.0))
    viridis = S.colormap_rgb8("viridis")
    assert np.array_equal(lut[:21], np.repeat(viridis[:1], 21, axis=0))      # at and below the window: t = 0
    assert np.array_equal(lut[200:], np.repeat(viridis[255:], 56, axis=0))   # at and above: t = 1
    assert np.array_equal(lut[110], viridis[round(255 * 90 / 180)])          # the middle of the window
    assert raises(ValueError, S.slice_lut8, "viridis", (5.0, 5.0))


def test_label_colours_are_those_of_the_ray_caster():
    emray = need("ncntm.viz.emray")
    ids = np.concatenate([np.arange(0, 5000), [2 ** 16 - 1, 2 ** 31 - 1, 2 ** 32 - 1]])
    assert np.array_equal(S.label_rgb(ids), emray.label_colors(ids))
    rgb8 = S.label_rgb8(ids)
    assert rgb8.dtype == np.uint8 and tuple(rgb8[0]) == (15, 15, 15)         # id 0 is dark: floor(255 * 0.06 + 0.5)
    assert rgb8[1:].max(axis=1).min() >= 178                                # every other id is bright (value >= 0.7)


def test_write_png_round_trips():
    Image = need("PIL.Image")
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (7, 13, 3), dtype=np.uint8)
    path = tmp_dir() / "random.png"
    S.write_png(path, rgb)
    assert np.array_equal(np.asarray(Image.open(path).convert("RGB")), rgb)
    assert raises(ValueError, S.write_png, path, rgb[..., 0])


def test_tube_width_factor():
    grid = VoxelGrid((100, 512, 512), (30.0, 6.0, 6.0))                     # SNEMI3D: A = (1, 1, 1/5)
    o = np.zeros((1, 3))
    along = {axis: np.eye(3)[[k]] * 100.0 for k, axis in enumerate("xyz")}
    for axis in "xyz":
        assert S.tube_width_factor(grid, "physical", o, along[axis])[0] == 1.0
    assert abs(S.tube_width_factor(grid, "index", o, along["z"])[0] - 1.0) < 1e-12        # cross-section in x, y
    for axis in "xy":
        assert abs(S.tube_width_factor(grid, "index", o, along[axis])[0] - math.sqrt(1 / 5)) < 1e-12
    assert abs(math.sqrt(1 / 5) - 0.447) < 5e-4
    # the factor squared is the area of the image of the unit disc perpendicular to the edge
    rng = np.random.default_rng(2)
    t = rng.normal(size=3)
    t /= np.linalg.norm(t)
    u = np.cross(t, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    a = grid.space_scale("index")
    area = np.linalg.norm(np.cross(a * u, a * v))
    assert abs(S.tube_width_factor(grid, "index", o, t[None])[0] ** 2 - area) < 1e-12
    assert np.isfinite(S.tube_width_factor(grid, "index", o, o)[0])          # an edge of length zero


def test_round_scale_bar_lengths():
    assert S.nice_length_nm(3072.0) == 500.0            # a quarter of 3072 is 768
    assert S.nice_length_nm(800.0) == 200.0
    assert S.nice_length_nm(4000.0) == 1000.0
    assert S.nice_length_nm(8.0, fraction=1.0) == 5.0
    assert raises(ValueError, S.nice_length_nm, 0.0)
    assert S.format_length(500.0) == "500 nm" and S.format_length(2000.0) == "2 um"


# ---- stage metadata and layout ----------------------------------------------------------------
def test_stage_metadata_records_the_true_scale():
    t = toy()
    stage = t.stage
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetDefaultPrim().GetPath() == "/World"
    scale = stage.GetRootLayer().customLayerData["emScale"]
    assert scale["nanometersPerUnit"] == NM_PER_UNIT == 1000.0
    assert scale["trueMetersPerUnit"] == 1e-6 and scale["authoredMetersPerUnit"] == 1.0 and scale["magnification"] == 1e6
    assert scale["axisOrder"] == "xyz" and scale["voxelOrigin"] == "corner"
    prim = t.ds.prim
    assert tuple(prim.GetAttribute("em:voxelSizeNm").Get()) == (6.0, 6.0, 30.0)          # (dx, dy, dz)
    assert tuple(prim.GetAttribute("em:shapeVoxels").Get()) == (40, 32, 10)              # (nx, ny, nz)
    assert prim.GetAttribute("em:anisotropy").Get() == 5.0 and prim.GetAttribute("em:indexEdgeNm").Get() == 6.0
    assert prim.GetAttribute("em:arrayAxisOrder").Get() == "zyx" and prim.GetAttribute("em:voxelOrigin").Get() == "corner"
    assert tuple(prim.GetAttribute("em:normWindow").Get()) == (20.0, 200.0)
    # the dataset frame is nanometres under a uniform scale, authored in double precision
    assert tuple(prim.GetAttribute("xformOp:scale").Get()) == (1e-3, 1e-3, 1e-3)
    assert str(prim.GetAttribute("xformOp:scale").GetTypeName()) == "double3"
    assert str(t.ds.voxels.GetAttribute("xformOp:scale").GetTypeName()) == "double3"
    assert np.array_equal(world_matrix(t.ds.physical), np.diag([1e-3, 1e-3, 1e-3, 1.0]))


def test_the_voxel_scale_is_the_anisotropy_and_lives_only_in_the_variants():
    t = toy()
    ds, layer = t.ds, t.stage.GetRootLayer()
    assert ds.prim.GetVariantSets().GetNames() == ["space"]
    assert sorted(ds.prim.GetVariantSet("space").GetVariantNames()) == ["index", "physical"]
    assert np.array_equal(ds.voxel_scale("physical"), [6.0, 6.0, 30.0])
    assert np.array_equal(ds.voxel_scale("index"), [6.0, 6.0, 6.0])
    # a local opinion would be stronger than the variants and switch the switch off: there is none,
    # neither on the voxel scale nor on any attribute that the variants carry
    assert layer.GetAttributeAtPath("/World/toy/voxels.xformOp:scale").default is None
    per_space = []

    def visit(path):
        if path.IsPropertyPath() and path.ContainsPrimVariantSelection():
            per_space.append(path)

    layer.Traverse("/World/toy", visit)
    names = {path.StripAllVariantSelections().pathString for path in per_space}
    assert "/World/toy/voxels.xformOp:scale" in names and "/World/toy/physical/skeleton/tubes_3.points" in names
    assert len(names) >= 20
    for name in names:
        spec = layer.GetAttributeAtPath(name)
        assert spec is not None and spec.default is None, name
    for space in SPACES:
        ds.set_space(space)
        assert ds.space == space
        assert np.array_equal(np.array(ds.voxels.GetAttribute("xformOp:scale").Get()), t.grid.display_voxel_size_xyz_nm(space))
        assert np.allclose(ds.index_to_world(), t.grid.index_to_unit_matrix(space), rtol=1e-15, atol=0)
    ds.set_space("physical")
    assert raises(ValueError, ds.set_space, "morph")


def test_a_local_opinion_would_defeat_the_variant():
    """The trap the helpers avoid, shown on a throw-away stage."""
    t = build_toy(tmp_dir() / "trap" / "toy.usda")
    scale = t.ds.voxels.GetAttribute("xformOp:scale")
    scale.Set(Gf.Vec3d(1, 2, 3))                       # e.g. a script, or a drag in a viewport
    t.ds.set_space("index")
    assert tuple(scale.Get()) == (1.0, 2.0, 3.0)       # the switch does nothing any more
    scale.Clear()
    assert tuple(scale.Get()) == (6.0, 6.0, 6.0)


def test_layers_are_tagged_and_switchable():
    t = toy()
    layers = t.ds.layers()
    assert list(layers) == list(S.LAYERS) == ["rawvalue", "voxvalue", "instance", "skeleton", "wiremesh", "gaussian", "frame"]
    for name, info in layers.items():
        group = t.stage.GetPrimAtPath(info["path"])
        assert group.GetAttribute("em:layer").Get() == name and info["visible"]
        frame = "voxels" if name in ("rawvalue", "voxvalue", "instance") else "physical"
        assert info["path"] == f"/World/toy/{frame}/{name}"
    assert layers["instance"]["prims"] == ["slice_z0005", "slice_y0016", "slice_x0020", "bricks_3", "surface_3"]
    assert layers["gaussian"]["prims"] == ["splats_3", "ellipsoids_3"]
    assert layers["frame"]["prims"] == ["bbox", "gnomon", "scalebar"]
    t.ds.set_visible("skeleton", False)
    assert not t.ds.layers()["skeleton"]["visible"]
    assert UsdGeom.Imageable(t.prims.tubes).ComputeVisibility() == UsdGeom.Tokens.invisible
    t.ds.set_visible("skeleton", True)
    assert UsdGeom.Imageable(t.prims.tubes).ComputeVisibility() == UsdGeom.Tokens.inherited
    assert t.ds.find("instance/bricks_3") == t.prims.bricks
    assert t.ds.find("voxels/instance/bricks_3") == t.prims.bricks
    assert t.ds.find("/World/toy/voxels/instance/bricks_3") == t.prims.bricks
    assert raises(KeyError, t.ds.set_visible, "meshes", False)


def test_describe_shows_the_tree_and_the_scale_in_each_space():
    text = toy().ds.describe()
    assert "40 x 32 x 10 voxels (x, y, z) of 6 x 6 x 30 nm" in text
    assert "physical (6, 6, 30) nm" in text and "index (6, 6, 6) nm" in text
    for layer in S.LAYERS:
        assert f"[layer {layer}]" in text
    for name in ("slice_z0005", "bricks_3", "surface_3", "wire_3", "tubes_3", "nodes_3", "splats_3", "ellipsoids_3",
                 "bbox", "gnomon", "scalebar"):
        assert name in text
    assert "proto_cube" not in text and "(invisible)" in text


def test_wrong_input_fails_loudly():
    t = toy()
    stage = Usd.Stage.CreateInMemory()
    assert raises(TypeError, S.Dataset, stage, "x", ((2, 2, 2), (1, 1, 1)))                      # not a VoxelGrid
    assert raises(ValueError, S.Dataset, stage, "x", TOY_GRID, raw=np.zeros((3, 3, 3), np.uint8))  # wrong shape
    ds = S.Dataset(stage, "x", TOY_GRID)
    assert raises(ValueError, ds.add_slices, "rawvalue", z=1)                                    # no array given
    raw, instance = toy_volumes()
    assert raises(ValueError, ds.add_slices, "rawvalue", z=1, volume=raw)                        # stage has no file
    assert len(ds.add_slices("rawvalue", z=1, volume=raw, textures=False)) == 1
    assert raises(ValueError, ds.add_slices, "membrane", z=1, volume=raw)
    assert raises(ValueError, ds.add_slices, "rawvalue", volume=raw, textures=False)             # no index
    assert raises(IndexError, ds.add_slices, "rawvalue", z=10, volume=raw, textures=False)       # nz = 10
    assert raises(ValueError, ds.add_slices, "rawvalue", z=1, volume=raw.astype(np.float32), textures=False)
    assert raises(ValueError, ds.add_slices, "instance", z=1, volume=instance.astype(np.float32), textures=False)
    c = toy_containers()
    empty = Bricks(np.zeros((0, 3), int), np.zeros((0, 3)), TOY_GRID.voxel_size_xyz_nm)
    assert raises(ValueError, ds.add_bricks, empty)
    nan_vertices = c.mesh.vertices_xyz_nm.copy()
    nan_vertices[0, 0] = np.nan
    assert raises(ValueError, ds.add_surface, Mesh(nan_vertices, c.mesh.faces))
    assert raises(ValueError, ds.add_surface, Mesh(c.mesh.vertices_xyz_nm, c.mesh.faces + 100))    # no such vertices
    assert raises(ValueError, ds.add_surface, Mesh(c.mesh.vertices_xyz_nm[:, :2], c.mesh.faces))
    assert raises(ValueError, ds.add_surface, c.mesh, opacity=0.0)
    assert raises(ValueError, ds.add_wiremesh, c.mesh, width_nm=-1.0)
    s = c.skeleton
    assert raises(ValueError, ds.add_skeleton, Skeleton(s.vertices_xyz_nm, s.edges, s.radii_nm[:-1]))
    assert raises(ValueError, ds.add_skeleton, Skeleton(s.vertices_xyz_nm, s.edges, -s.radii_nm))
    assert raises(ValueError, ds.add_skeleton, Skeleton(s.vertices_xyz_nm, s.edges + 9, s.radii_nm))
    g = c.gaussians
    assert raises(ValueError, ds.add_gaussians, Gaussians(g.means_xyz_nm, g.covariances_nm2[:1], g.weights))
    assert raises(ValueError, ds.add_gaussians, Gaussians(g.means_xyz_nm, -g.covariances_nm2, g.weights))
    skew = g.covariances_nm2.copy()
    skew[0, 0, 1] += 50.0
    assert raises(ValueError, ds.add_gaussians, Gaussians(g.means_xyz_nm, skew, g.weights))
    assert raises(ValueError, ds.add_gaussians, g, opacity=1.5)
    assert raises(ValueError, ds.add_gaussians, g, show="nothing")
    assert raises(ValueError, S.author_camera, stage, eye=(0, 0, 1), target=(0, 0, 1))
    assert raises(ValueError, S.author_camera, stage, eye=(0, 0, 1), target=(0, 0, 0))           # up is the view direction
    assert raises(ValueError, t.ds.voxel_scale, "morph")
    assert raises(ValueError, S.Dataset.from_prim, t.ds.voxels)


# ---- the frames -----------------------------------------------------------------------------------
def check_round_trip(built):
    grid, ds = built.grid, built.ds
    rng = np.random.default_rng(3)
    nz, ny, nx = grid.shape_zyx
    kji = np.stack([rng.integers(0, n, 500) for n in (nz, ny, nx)], axis=1)
    kji = np.concatenate([kji, [[0, 0, 0], [nz - 1, ny - 1, nx - 1]]])
    worst = 0.0
    for space in SPACES:
        ds.set_space(space)
        m = UsdGeom.Xformable(ds.voxels).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world = np.array([m.Transform(Gf.Vec3d(i + 0.5, j + 0.5, k + 0.5)) for k, j, i in kji.tolist()])
        expected = grid.voxel_centers_nm(kji, space) / NM_PER_UNIT
        worst = max(worst, np.abs(world - expected).max())
        assert np.abs(world - expected).max() < 1e-9                       # stage units
        assert np.array_equal(grid.nm_to_voxel(world * NM_PER_UNIT, space), kji)
        back = np.array([m.GetInverse().Transform(Gf.Vec3d(*p)) for p in world.tolist()])
        assert np.array_equal(np.floor(back).astype(int)[:, ::-1], kji)
        assert np.abs((world_matrix(ds.voxels) @ np.c_[kji[:, ::-1] + 0.5, np.ones(len(kji))].T).T[:, :3] - expected).max() < 1e-9
    ds.set_space("physical")
    return worst


def test_voxel_centres_round_trip_through_the_voxels_frame_toy():
    check_round_trip(toy())


def test_voxel_centres_round_trip_through_the_voxels_frame_snemi3d():
    worst = check_round_trip(real())
    print(f"\n  max |USD - closed form| = {worst:.2e} stage units over 502 voxel centres, both spaces")


def test_world_bounding_box_in_both_spaces():
    r = real()
    for space, expected in (("physical", (3.072, 3.072, 3.0)), ("index", (3.072, 3.072, 0.6))):
        r.ds.set_space(space)
        assert np.abs(world_size(r.ds.voxels) - expected).max() < 1e-9          # micrometres
        assert np.allclose(r.grid.extent_nm(space) / NM_PER_UNIT, expected, rtol=0, atol=1e-12)
        corners = world_points(r.prims.frame["bbox"])
        assert np.abs(np.ptp(corners, axis=0) - expected).max() < 1e-6          # the wire box is that box
    r.ds.set_space("physical")


def test_slices_sit_at_voxel_centres_and_say_what_they_show():
    t = toy()
    for kind, (dataset, colormap) in S.SLICE_KINDS.items():
        for prim, axis, index in zip(t.prims.slices[kind], "zyx", (5, 16, 20)):
            assert prim.GetAttribute("em:slice:kind").Get() == kind
            assert prim.GetAttribute("em:slice:dataset").Get() == dataset
            assert prim.GetAttribute("em:slice:colorMap").Get() == colormap
            assert prim.GetAttribute("em:slice:axis").Get() == axis and prim.GetAttribute("em:slice:index").Get() == index
            assert tuple(prim.GetAttribute("em:slice:window").Get()) == ((20.0, 200.0) if kind == "voxvalue" else (0.0, 255.0))
            points = np.array(prim.GetAttribute("points").Get())
            a = "xyz".index(axis)
            assert np.all(points[:, a] == index + 0.5)                       # through the centre of the layer
            others = [b for b in range(3) if b != a]
            assert np.array_equal(points[:, others].min(axis=0), [0, 0])
            assert np.array_equal(points[:, others].max(axis=0), t.grid.shape_xyz[others])
            assert not prim.GetAttribute("normals").HasAuthoredValue()
            assert UsdGeom.Mesh(prim).GetSubdivisionSchemeAttr().Get() == "none"


def test_slice_textures_are_crisp_copies_of_the_array():
    Image = need("PIL.Image")
    t = toy()
    viridis_lut = S.slice_lut8("viridis", (20.0, 200.0))
    for kind in S.SLICE_KINDS:
        volume = t.instance if kind == "instance" else t.raw
        for prim, image in zip(t.prims.slices[kind], (volume[5], volume[:, 16, :], volume[:, :, 20])):
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            surface = material.ComputeSurfaceSource()[0]
            assert surface.GetIdAttr().Get() == "UsdPreviewSurface"
            assert tuple(surface.GetInput("diffuseColor").Get()) == (0.0, 0.0, 0.0)
            texture = UsdShade.Shader(surface.GetInput("emissiveColor").GetConnectedSources()[0][0].source.GetPrim())
            assert texture.GetIdAttr().Get() == "UsdUVTexture" and texture.GetInput("sourceColorSpace").Get() == "sRGB"
            asset = texture.GetInput("file").Get()
            assert asset.path.startswith("./textures/") and Path(asset.resolvedPath).is_file()
            pixels = np.asarray(Image.open(asset.resolvedPath).convert("RGB"))
            expected = {"rawvalue": np.repeat(image[..., None], 3, axis=2), "voxvalue": viridis_lut[image],
                        "instance": S.label_rgb8(image)}[kind]
            assert pixels.shape == (3 * image.shape[0], 3 * image.shape[1], 3)       # upsample = 3
            assert np.array_equal(pixels, np.repeat(np.repeat(expected, 3, axis=0), 3, axis=1))
            # texture coordinates: the corner at array row 0, column 0 is point 0 and has (s, t) = (0, 1)
            st = np.array(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get())
            assert np.array_equal(st, [[0, 1], [1, 1], [1, 0], [0, 0]])


def test_bricks_are_exact_voxel_boxes():
    t = toy()
    kji = t.bricks.kji
    assert np.array_equal(np.array(t.prims.bricks.GetAttribute("positions").Get()), kji[:, ::-1] + 0.5)
    assert np.all(np.array(t.prims.bricks.GetAttribute("scales").Get()) == 1.0)
    assert not t.prims.bricks.GetAttribute("orientations").HasAuthoredValue()
    assert UsdGeom.Cube(t.prims.bricks.GetChild("proto_cube")).GetSizeAttr().Get() == 1.0
    for space in SPACES:
        t.ds.set_space(space)
        matrices = instance_world_matrices(t.prims.bricks)
        size = t.grid.display_voxel_size_xyz_nm(space) / NM_PER_UNIT
        assert np.allclose(matrices[:, :3, :3], np.diag(size)[None], rtol=1e-12, atol=0)       # no rotation, no shear
        assert np.abs(matrices[:, :3, 3] - t.grid.voxel_centers_nm(kji, space) / NM_PER_UNIT).max() < 1e-9
        # the instancer's box is the box of the block of voxels z 2..7, y 8..23, x 10..29
        assert np.abs(world_size(t.prims.bricks) - np.array([20, 16, 6]) * size).max() < 1e-9
    t.ds.set_space("physical")


def test_bricks_made_with_the_wrong_grid_are_drawn_where_their_coordinates_say():
    t = build_toy(tmp_dir() / "wrong" / "toy.usda")
    wrong = toy_containers(TOY_GRID.as_cubes()).bricks          # a converter that was told 6 x 6 x 6 nm
    prim = t.ds.add_bricks(wrong, name="bricks_wrong")
    assert np.allclose(np.array(prim.GetAttribute("scales").Get()), [1.0, 1.0, 0.2])
    matrices = instance_world_matrices(prim)
    assert np.allclose(matrices[:, :3, :3], np.diag([0.006, 0.006, 0.006])[None], rtol=1e-6)     # cubes
    assert np.abs(matrices[:, :3, 3] - wrong.centers_xyz_nm / NM_PER_UNIT).max() < 1e-7
    assert abs(world_size(prim)[2] - 6 * 0.006) < 1e-7           # 6 sections, 36 nm instead of 180 nm


def test_surface_points_are_index_coordinates_without_normals():
    t = toy()
    points = np.array(t.prims.surface.GetAttribute("points").Get())
    assert np.array_equal(points, t.mesh.vertices_xyz_nm / t.grid.voxel_size_xyz_nm)
    assert np.array_equal(points.min(axis=0), [10, 8, 2]) and np.array_equal(points.max(axis=0), [30, 24, 8])
    assert not t.prims.surface.GetAttribute("normals").HasAuthoredValue()
    assert UsdGeom.Mesh(t.prims.surface).GetSubdivisionSchemeAttr().Get() == "none"
    surface = UsdShade.MaterialBindingAPI(t.prims.surface).ComputeBoundMaterial()[0].ComputeSurfaceSource()[0]
    assert abs(surface.GetInput("opacity").Get() - 0.4) < 1e-7                  # opacity is on the material for RTX
    assert abs(UsdGeom.Mesh(t.prims.surface).GetDisplayOpacityAttr().Get()[0] - 0.4) < 1e-7
    for space in SPACES:
        t.ds.set_space(space)
        assert np.abs(world_points(t.prims.surface) - t.grid.to_space(t.mesh.vertices_xyz_nm, space) / NM_PER_UNIT).max() < 1e-9
    t.ds.set_space("physical")


def test_wiremesh_and_skeleton_tubes():
    t = toy()
    wire, tubes = UsdGeom.BasisCurves(t.prims.wire), UsdGeom.BasisCurves(t.prims.tubes)
    edges = t.mesh.edges()
    assert len(edges) == 18                                                    # a triangulated box
    for curves, n_edges in ((wire, 18), (tubes, 5)):
        assert curves.GetTypeAttr().Get() == "linear" and curves.GetWidthsInterpolation() == "vertex"
        assert list(curves.GetCurveVertexCountsAttr().Get()) == [2] * n_edges
        assert not curves.GetPrim().GetAttribute("normals").HasAuthoredValue()
    default_width = 0.001 * np.linalg.norm(t.grid.extent_xyz_nm)               # 0.1 % of the box diagonal
    assert abs(t.prims.wire.GetAttribute("em:widthNm").Get() - default_width) < 1e-9
    s = t.skeleton
    order = s.edges.reshape(-1)
    for space in SPACES:
        t.ds.set_space(space)
        factor = S.tube_width_factor(t.grid, space, s.vertices_xyz_nm[s.edges[:, 0]], s.vertices_xyz_nm[s.edges[:, 1]])
        assert np.allclose(np.array(tubes.GetWidthsAttr().Get()), 2.0 * s.radii_nm[order] * np.repeat(factor, 2), rtol=1e-6)
        assert np.abs(np.array(tubes.GetPointsAttr().Get()) - t.grid.to_space(s.vertices_xyz_nm, space)[order]).max() < 1e-3
        wire_factor = S.tube_width_factor(t.grid, space, t.mesh.vertices_xyz_nm[edges[:, 0]], t.mesh.vertices_xyz_nm[edges[:, 1]])
        assert np.allclose(np.array(wire.GetWidthsAttr().Get()), default_width * np.repeat(wire_factor, 2), rtol=1e-6)
        assert np.abs(np.array(wire.GetPointsAttr().Get()) - t.grid.to_space(t.mesh.vertices_xyz_nm, space)[edges.reshape(-1)]).max() < 1e-3
    t.ds.set_space("physical")
    assert np.allclose(np.array(tubes.GetWidthsAttr().Get()), 2.0 * s.radii_nm[order], rtol=1e-6)      # width = 2 x radius
    t.ds.set_space("index")       # edges 0, 1, 4 run along x, edge 2 along y, edge 3 along z
    widths = np.array(tubes.GetWidthsAttr().Get()).reshape(-1, 2)
    in_plane = [0, 1, 2, 4]
    assert np.allclose(widths[in_plane], 2.0 * s.radii_nm[s.edges[in_plane]] * math.sqrt(1 / 5), rtol=1e-6)
    assert np.allclose(widths[3], 2.0 * s.radii_nm[s.edges[3]], rtol=1e-6)        # along z: unchanged
    t.ds.set_space("physical")


def test_skeleton_node_markers():
    t = toy()
    s = t.skeleton
    instancer = UsdGeom.PointInstancer(t.prims.nodes)
    assert [p.name for p in instancer.GetPrototypesRel().GetTargets()] == ["proto_node", "proto_branch_point", "proto_end_point"]
    assert list(instancer.GetProtoIndicesAttr().Get()) == [2, 1, 0, 2, 2, 2]    # end, branch, ordinary, end, end, end
    marker = s.radii_nm * np.array([1.5, 1.5, 1.0, 1.5, 1.5, 1.5])              # branch and end points are 1.5 times larger
    for space in SPACES:
        t.ds.set_space(space)
        matrices = instance_world_matrices(t.prims.nodes)
        a = t.grid.space_scale(space)
        assert np.allclose(matrices[:, :3, :3], marker[:, None, None] * np.diag(a)[None] / NM_PER_UNIT, rtol=1e-6)
        assert np.abs(matrices[:, :3, 3] - t.grid.to_space(s.vertices_xyz_nm, space) / NM_PER_UNIT).max() < 1e-7
    t.ds.set_space("physical")


def covariance_from_splat(prim) -> np.ndarray:
    """R diag(scales^2) R^T from the schema attributes; the numpy layout of a Vt quaternion array is (x, y, z, w)."""
    U = need("ncntm.viz.usd_to_emray")
    rotations = U._quat_xyzw_to_matrix(np.array(prim.GetAttribute("orientations").Get(), dtype=np.float64))
    sigmas = np.array(prim.GetAttribute("scales").Get(), dtype=np.float64)
    return np.einsum("nab,nb,ncb->nac", rotations, sigmas ** 2, rotations)


def test_gaussians_are_transformed_as_covariances():
    t = toy()
    g = t.gaussians
    splats, ellipsoids = t.prims.splats, t.prims.ellipsoids
    assert abs(ellipsoids.GetAttribute("em:kSigma").Get() - math.sqrt(5.0)) < 1e-15
    assert np.allclose(np.array(splats.GetAttribute("em:gaussian:opacities").Get()), 0.8)
    assert np.allclose(np.array(splats.GetAttribute("em:gaussian:displayRgb").Get()), g.colors_rgb)
    for space in SPACES:
        t.ds.set_space(space)
        a = t.grid.space_scale(space)
        mean = g.means_xyz_nm * a                                  # A mu
        cov = g.covariances_nm2 * a[:, None] * a[None, :]          # A S A^T
        assert np.array_equal(np.array(splats.GetAttribute("em:gaussian:means").Get()), mean)
        assert np.array_equal(np.array(splats.GetAttribute("em:gaussian:covariances").Get()), cov)
        # the k-sigma ellipsoids: M M^T / k^2 is the covariance (orientations are half floats)
        matrices = instance_world_matrices(ellipsoids)
        linear = matrices[:, :3, :3] * NM_PER_UNIT
        from_instances = linear @ np.swapaxes(linear, 1, 2) / 5.0
        assert np.abs(from_instances - cov).max() / np.abs(cov).max() < 3e-3
        assert np.abs(matrices[:, :3, 3] * NM_PER_UNIT - mean).max() < 1e-3
        assert np.allclose(np.linalg.det(linear), 5.0 ** 1.5 * np.sqrt(np.linalg.det(cov)), rtol=5e-3)   # proper: det > 0
    t.ds.set_space("physical")


def test_splat_prim_follows_the_particle_field_schema():
    t = toy()
    if not S.splat_schema_available():
        raise Skip("this USD build has no ParticleField3DGaussianSplat schema")
    need("ncntm.viz.usd_to_emray")
    g, prim = t.gaussians, t.prims.splats
    assert prim.GetTypeName() == "ParticleField3DGaussianSplat" and prim.IsA(UsdGeom.Gprim)
    definition = Usd.SchemaRegistry().FindConcretePrimDefinition("ParticleField3DGaussianSplat")
    for name, type_name in (("positions", "point3f[]"), ("orientations", "quatf[]"), ("scales", "float3[]"),
                            ("opacities", "float[]"), ("radiance:sphericalHarmonicsDegree", "int"),
                            ("radiance:sphericalHarmonicsCoefficients", "float3[]")):
        attribute = prim.GetAttribute(name)
        assert name in definition.GetPropertyNames()
        assert str(attribute.GetTypeName()) == type_name and not attribute.IsCustom(), name
    assert prim.GetAttribute("radiance:sphericalHarmonicsDegree").Get() == 0          # the schema default is 3
    harmonics = prim.GetAttribute("radiance:sphericalHarmonicsCoefficients")
    assert harmonics.GetMetadata("interpolation") == "vertex" and harmonics.GetMetadata("elementSize") == 1
    rgb = 0.5 + S.SH_C0 * np.array(harmonics.Get())
    assert np.abs(rgb - g.colors_rgb).max() < 1e-6
    assert abs(S.SH_C0 - 1.0 / (2.0 * math.sqrt(math.pi))) < 1e-15
    assert np.allclose(np.array(prim.GetAttribute("opacities").Get()), 0.8)
    counts = {len(prim.GetAttribute(n).Get()) for n in ("positions", "orientations", "scales", "opacities")}
    assert counts == {len(g)}
    for space in SPACES:
        t.ds.set_space(space)
        a = t.grid.space_scale(space)
        cov = g.covariances_nm2 * a[:, None] * a[None, :]
        assert np.abs(covariance_from_splat(prim) - cov).max() / np.abs(cov).max() < 1e-5
        assert np.abs(np.array(prim.GetAttribute("positions").Get()) - g.means_xyz_nm * a).max() < 1e-3
        quaternions = np.array(prim.GetAttribute("orientations").Get())
        assert np.abs(np.linalg.norm(quaternions, axis=1) - 1.0).max() < 1e-6
        assert np.array(prim.GetAttribute("scales").Get()).min() > 0
    t.ds.set_space("physical")
    # the uniform dataset scale is the only transform above the splat prim
    lengths = np.linalg.norm(world_matrix(prim)[:3, :3], axis=0)
    assert np.allclose(lengths, 1e-3, rtol=1e-12)


def test_frame_gnomon_and_scale_bar():
    t = toy()
    frame, extent = t.prims.frame, t.grid.extent_xyz_nm                       # 240 x 192 x 300 nm
    bar = 50.0                                                                # largest 1-2-5 length within 240 / 4
    assert frame["scalebar"].GetAttribute("em:lengthNm").Get() == bar
    assert frame["scalebar"].GetAttribute("em:label").Get() == "50 nm"
    assert frame["scalebar"].GetAttribute("em:axis").Get() == "x"
    width = list(UsdGeom.BasisCurves(frame["bbox"]).GetWidthsAttr().Get())[0]
    assert abs(width - 0.003 * np.linalg.norm(extent)) < 1e-4                 # 0.3 % of the box diagonal
    for space, arms_nm in (("physical", (50.0, 50.0, 50.0)), ("index", (50.0, 50.0, 10.0))):
        t.ds.set_space(space)
        for k, axis in enumerate("xyz"):
            size = world_size(frame["gnomon"].GetChild(axis)) * NM_PER_UNIT
            assert abs(size[k] - arms_nm[k]) < 1e-3                           # nm drawn for 50 nm of tissue
            assert np.allclose(np.delete(size, k), bar / 12.0, atol=1e-3)     # thickness: the same in both spaces
        assert abs(world_size(frame["scalebar"])[0] * NM_PER_UNIT - bar) < 1e-3
        corners = world_points(frame["bbox"]) * NM_PER_UNIT
        assert np.abs(corners.min(axis=0)).max() < 1e-3 and np.abs(corners.max(axis=0) - t.grid.extent_nm(space)).max() < 1e-3
        assert np.abs(world_size(frame["bbox"]) * NM_PER_UNIT - (t.grid.extent_nm(space) + width)).max() < 1e-3
    t.ds.set_space("physical")
    colours = [S.linear_to_srgb(np.array(UsdGeom.Mesh(frame["gnomon"].GetChild(axis)).GetDisplayColorAttr().Get()[0])) for axis in "xyz"]
    assert [int(np.argmax(c)) for c in colours] == [0, 1, 2]                  # x red, y green, z blue
    for prim in (frame["bbox"], frame["scalebar"], frame["gnomon"].GetChild("z")):      # annotation is unlit
        surface = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0].ComputeSurfaceSource()[0]
        assert max(surface.GetInput("emissiveColor").Get()) > 0 and tuple(surface.GetInput("diffuseColor").Get()) == (0, 0, 0)


def test_switching_the_space_moves_every_layer_together():
    scipy_ndimage = need("scipy.ndimage")
    r = real()
    grid, p = r.grid, r.prims
    near = scipy_ndimage.binary_dilation(r.mask, structure=np.ones((3, 3, 3), bool))
    index_coordinates, inside = {}, {}
    for space in SPACES:
        r.ds.set_space(space)
        world = {
            "skeleton tubes": world_points(p.tubes), "skeleton nodes": world_points(p.nodes, "positions"),
            "surface vertices": world_points(p.surface), "wire vertices": world_points(p.wire),
            "brick centres": world_points(p.bricks, "positions"),
            "gaussian means": world_points(p.splats, "em:gaussian:means"),
            "ellipsoid centres": world_points(p.ellipsoids, "positions"),
            "bounding box": world_points(p.frame["bbox"]),
        }
        if p.splats.HasAttribute("positions"):
            world["splat positions"] = world_points(p.splats, "positions")
        for name, points in world.items():
            index_coordinates[name, space] = grid.nm_to_index(points * NM_PER_UNIT, space)       # continuous (i, j, k)
            kji = np.clip(grid.nm_to_voxel(points * NM_PER_UNIT, space), 0, np.array(grid.shape_zyx) - 1)
            inside[name, space] = (r.mask[tuple(kji.T)], near[tuple(kji.T)])
    r.ds.set_space("physical")
    for name in ("skeleton tubes", "skeleton nodes", "surface vertices", "wire vertices", "brick centres",
                 "gaussian means", "ellipsoid centres", "bounding box"):
        a, b = index_coordinates[name, "physical"], index_coordinates[name, "index"]
        assert np.abs(a - b).max() < 2e-3, name                 # the same place in the array, in voxels
        on_faces = name in ("surface vertices", "wire vertices", "bounding box")     # no containing voxel: round-off decides
        for k in ((1,) if on_faces else (0, 1)):
            assert np.array_equal(inside[name, "physical"][k], inside[name, "index"][k]), name
    for space in SPACES:
        assert inside["skeleton tubes", space][0].all() and inside["skeleton nodes", space][0].all()
        assert inside["brick centres", space][0].all()
        assert inside["surface vertices", space][1].all()        # on a face of a voxel of the instance
        assert inside["wire vertices", space][1].mean() > 0.97   # simplified mesh: vertices moved a little
        assert inside["gaussian means", space][0].mean() > 0.9   # the mean of a bent part can lie outside it
        assert np.array_equal(inside["gaussian means", space][0], inside["ellipsoid centres", space][0])
        box = index_coordinates["bounding box", space]
        assert np.abs(box.min(axis=0)).max() < 1e-3 and np.abs(box.max(axis=0) - grid.shape_xyz).max() < 1e-3
    assert np.array_equal(np.array(p.bricks.GetAttribute("positions").Get()), r.bricks.kji[:, ::-1] + 0.5)
    print(f"\n  inside the instance: gaussian means {inside['gaussian means', 'physical'][0].mean():.3f}, "
          f"wire vertices within one voxel {inside['wire vertices', 'physical'][1].mean():.4f}")


def test_real_dataset_records_where_its_data_is():
    r = real()
    prim = r.ds.prim
    assert prim.GetAttribute("em:sampleFile").Get() == "snemi3d.h5"             # a file name, not a path
    assert tuple(prim.GetAttribute("em:voxelOffset").Get()) == (0, 0, 0)
    assert tuple(prim.GetAttribute("em:normWindow").Get()) == r.sample.norm_window
    assert tuple(prim.GetAttribute("em:shapeVoxels").Get()) == (512, 512, 100)


# ---- saving, reopening, validation ------------------------------------------------------------
def test_saved_files_reopen_with_variants_and_relative_textures():
    r = real()
    text = r.path.read_text()
    assert str(tmp_dir()) not in text and str(REPO) not in text                # no absolute path in the file
    for extension in ("usda", "usdc"):
        target = tmp_dir() / f"export_{extension}" / f"snemi3d.{extension}"
        assert r.ds.save(target) == str(target)
        stage = Usd.Stage.Open(str(target))
        datasets = S.find_datasets(stage)
        assert [d.name for d in datasets] == ["snemi3d"]
        ds = datasets[0]
        assert ds.grid == r.grid and ds.norm_window == r.sample.norm_window
        assert ds.prim.GetVariantSets().GetNames() == ["space"]
        assert sorted(ds.prim.GetVariantSet("space").GetVariantNames()) == ["index", "physical"]
        assert np.array_equal(ds.voxel_scale("physical"), [6, 6, 30]) and np.array_equal(ds.voxel_scale("index"), [6, 6, 6])
        assert list(ds.layers()) == list(S.LAYERS)
        tubes = stage.GetPrimAtPath(r.prims.tubes.GetPath())
        for space, z_extent in (("physical", 3.0), ("index", 0.6)):
            ds.set_space(space)
            assert abs(world_size(ds.voxels)[2] - z_extent) < 1e-9
            expected = r.grid.to_space(r.skeleton.vertices_xyz_nm, space)[r.skeleton.edges.reshape(-1)]
            assert np.abs(np.array(tubes.GetAttribute("points").Get()) - expected).max() < 1e-3
        textures = []
        for prim in stage.Traverse():
            if prim.GetAttribute("inputs:file"):
                asset = prim.GetAttribute("inputs:file").Get()
                assert asset.path.startswith("./textures/")
                resolved = Path(asset.resolvedPath)
                assert resolved.is_file() and resolved.parent == target.parent / "textures"
                textures.append(resolved.name)
        assert len(textures) == 9 and "snemi3d_rawvalue_z0050.png" in textures
        assert stage.GetRootLayer().customLayerData["emScale"]["nanometersPerUnit"] == 1000.0
        assert stage.GetPrimAtPath("/World/Cameras/overview").IsA(UsdGeom.Camera)


def test_stage_export_would_lose_the_variants():
    """Why ``save`` exports the root layer: Usd.Stage.Export flattens the composition."""
    t = toy()
    flat = tmp_dir() / "flat" / "toy.usda"
    flat.parent.mkdir(exist_ok=True)
    t.stage.Export(str(flat))
    assert Usd.Stage.Open(str(flat)).GetPrimAtPath("/World/toy").GetVariantSets().GetNames() == []
    kept = tmp_dir() / "kept" / "toy.usdc"
    t.ds.save(kept)
    assert Usd.Stage.Open(str(kept)).GetPrimAtPath("/World/toy").GetVariantSets().GetNames() == ["space"]


def test_toy_stage_passes_the_usd_validators():
    t = toy()
    for space in SPACES:
        t.ds.set_space(space)
        assert S.validate(t.stage) == []
    t.ds.set_space("physical")
    # the validators do find things: a stage without metadata fails
    assert len(S.validate(Usd.Stage.CreateInMemory())) > 0


def test_real_stage_passes_the_usd_validators():
    r = real()
    for space in SPACES:
        r.ds.set_space(space)
        assert S.validate(r.stage) == []
    r.ds.set_space("physical")
    for extension in ("usda", "usdc"):
        target = tmp_dir() / f"validate_{extension}" / f"snemi3d.{extension}"
        r.ds.save(target)
        assert S.validate(Usd.Stage.Open(str(target))) == []


def test_rerunning_a_cell_replaces_what_it_made():
    path = tmp_dir() / "rerun" / "toy.usda"
    first = build_toy(path)
    n_prims = len(list(first.stage.Traverse()))
    c = toy_containers()
    first.ds.add_skeleton(c.skeleton)                                # the same names again: replaced, not doubled
    first.ds.add_gaussians(c.gaussians, splats=False)                # now without the schema prim
    assert len(list(first.stage.Traverse())) == n_prims
    assert first.stage.GetPrimAtPath("/World/toy/physical/gaussian/splats_3").GetTypeName() == "Scope"
    assert not first.stage.GetPrimAtPath("/World/toy/physical/gaussian/splats_3").HasAttribute("positions")
    second = build_toy(path)                                         # new_stage on a path that is still open
    assert len(list(second.stage.Traverse())) == n_prims
    assert S.validate(second.stage) == []


# ---- without the ParticleField schema ------------------------------------------------------------
def test_stage_without_the_splat_schema_still_opens_and_draws():
    built = build_toy(tmp_dir() / "fallback" / "toy.usda", splats=False)
    carrier, ellipsoids = built.prims.splats, built.prims.ellipsoids
    assert carrier.GetTypeName() == "Scope" and ellipsoids.GetTypeName() == "PointInstancer"
    assert not any(carrier.HasAttribute(n) for n in ("positions", "orientations", "scales", "opacities"))
    assert UsdGeom.Imageable(carrier).ComputeVisibility() == "invisible"       # show="auto": the fallback is shown
    assert UsdGeom.Imageable(ellipsoids).ComputeVisibility() == "inherited"
    text = built.path.read_text()
    assert "ParticleField" not in text
    for extension in ("usda", "usdc"):
        target = tmp_dir() / f"fallback_{extension}" / f"toy.{extension}"
        built.ds.save(target)
        stage = Usd.Stage.Open(str(target))
        assert S.validate(stage) == []
        prim = stage.GetPrimAtPath(carrier.GetPath())
        assert np.array_equal(np.array(prim.GetAttribute("em:gaussian:covariances").Get()), built.gaussians.covariances_nm2)
    U = need("ncntm.viz.usd_to_emray")
    scene = U.build_scene(built.stage, layers=["gaussian"], volumes=built.volumes)
    assert scene.num_gaussians == 0 and scene.num_triangles > 0               # ellipsoids only
    built.ds.set_visible("gaussian/splats_3", True)                            # the Warp picture can still show Gaussians
    built.ds.set_visible("gaussian/ellipsoids_3", False)
    scene = U.build_scene(built.stage, layers=["gaussian"], volumes=built.volumes)
    assert scene.num_gaussians == 2 and scene.num_triangles == 0
    if S.splat_schema_available():
        assert raises(ValueError, built.ds.add_gaussians, built.gaussians, show="splat")
    else:
        assert raises(RuntimeError, built.ds.add_gaussians, built.gaussians, splats=True)


# ---- the Warp adapter -----------------------------------------------------------------------------
def face_on_camera(corners, rows, columns):
    """An orthographic camera that looks straight at a slice quad so that the picture has the
    layout of the array: quad point 0 (array row 0, column 0) top-left, columns to the right,
    rows downwards.  Pixels are square; a voxel covers px_row x px_col of them."""
    emray = need("ncntm.viz.emray")
    right, down = corners[1] - corners[0], corners[3] - corners[0]
    width, height = np.linalg.norm(right), np.linalg.norm(down)
    pixel = min(width / columns, height / rows)
    px_col, px_row = int(round(width / columns / pixel)), int(round(height / rows / pixel))
    right, down = right / width, down / height
    forward = np.cross(-down, right)                        # emray: right = forward x up, with up = -down
    centre = corners.mean(axis=0)
    camera = emray.Camera(eye=tuple(centre - forward), target=tuple(centre), up=tuple(-down), mode="orthographic",
                          ortho_height=height, width=columns * px_col, height=rows * px_row, ssaa=1)
    return camera, px_row, px_col


def check_slices_face_on(built, volumes=None):
    """Every slice, drawn alone and face on, in both spaces: the picture is the numpy slice."""
    U = need("ncntm.viz.usd_to_emray")
    need("h5py")
    raw, instance = (built.sample.raw, built.sample.instance) if volumes is None else (built.raw, built.instance)
    worst = 0
    for space in SPACES:
        built.ds.set_space(space)
        for kind, prims in built.prims.slices.items():
            volume = instance if kind == "instance" else raw
            for prim in prims:
                axis, index = prim.GetAttribute("em:slice:axis").Get(), prim.GetAttribute("em:slice:index").Get()
                image = volume[index] if axis == "z" else volume[:, index, :] if axis == "y" else volume[:, :, index]
                window = tuple(prim.GetAttribute("em:slice:window").Get())
                expected = {"rawvalue": lambda: np.repeat(image[..., None], 3, axis=2),
                            "voxvalue": lambda: S.slice_lut8("viridis", window)[image],
                            "instance": lambda: S.label_rgb8(image)}[kind]()
                camera, px_row, px_col = face_on_camera(world_points(prim), *image.shape)
                with showing(built, prim):
                    scene = U.build_scene(built.stage, volumes=volumes)
                    picture = scene.render(camera)
                assert list(scene.objects.values()) == [{"path": str(prim.GetPath()), "layer": kind, "type": "Mesh"}]
                assert np.all(picture.object_id == 1) and np.allclose(picture.depth, 1.0, atol=1e-4)
                seen = picture.rgba[px_row // 2::px_row, px_col // 2::px_col, :3]         # the middle of every voxel
                difference = np.abs(seen.astype(int) - expected.astype(int)).max()
                worst = max(worst, difference)
                assert difference == 0, (space, kind, axis, index, difference)
                # every pixel of a voxel has the voxel's colour: no filtering, crisp edges
                assert np.array_equal(picture.rgba[..., :3], np.repeat(np.repeat(expected, px_row, axis=0), px_col, axis=1))
                assert (px_row, px_col) == ((1, 1) if axis == "z" or space == "index" else (5, 1))
    built.ds.set_space("physical")
    return worst


def test_face_on_render_of_each_toy_slice_equals_the_numpy_slice():
    t = toy()
    assert check_slices_face_on(t, volumes=t.volumes) == 0


def test_face_on_render_of_each_real_slice_equals_the_hdf5_slice():
    r = real()
    assert len([p for group in r.prims.slices.values() for p in group]) == 9
    assert check_slices_face_on(r) == 0                # the adapter reads samples/snemi3d.h5 itself
    # voxvalue slices show sample.voxvalue() through the colour map
    index = r.prims.slices["voxvalue"][0].GetAttribute("em:slice:index").Get()
    t = np.rint(r.sample.voxvalue()[index] * 255.0).astype(int)
    assert np.array_equal(S.colormap_rgb8("viridis")[t], S.slice_lut8("viridis", r.sample.norm_window)[r.sample.raw[index]])


def test_slices_of_a_crop_and_of_a_derived_sample():
    samples, U = need("ncntm.viz.samples", "ncntm.viz.usd_to_emray")
    need("h5py")
    crop = samples.load_sample("snemi3d").crop(slice(10, 30), slice(100, 164), slice(200, 248))
    stage = S.new_stage(tmp_dir() / "crop" / "crop.usda")
    ds = S.Dataset.from_sample(stage, crop, name="crop")
    assert ds.prim.GetAttribute("em:sampleFile").Get() == "snemi3d.h5"
    assert tuple(ds.prim.GetAttribute("em:voxelOffset").Get()) == (200, 100, 10)       # (i0, j0, k0)
    prims = {"rawvalue": ds.add_slices("rawvalue", z=7, y=30), "instance": ds.add_slices("instance", x=11)}
    built = SimpleNamespace(stage=stage, ds=ds, sample=crop, prims=SimpleNamespace(slices=prims))
    assert check_slices_face_on(built) == 0            # read from the file through em:voxelOffset
    # a strided sample has voxels that are not in the file: the arrays must be handed over
    thick = crop.strided(2)
    stage2 = S.new_stage(tmp_dir() / "strided" / "strided.usda")
    ds2 = S.Dataset.from_sample(stage2, thick, name="thick")
    assert ds2.prim.GetAttribute("em:sampleFile").Get() == ""
    assert tuple(ds2.prim.GetAttribute("em:voxelSizeNm").Get()) == (6.0, 6.0, 60.0)
    ds2.add_slices("rawvalue", z=3)
    assert raises(LookupError, U.build_scene, stage2)
    assert U.build_scene(stage2, volumes={"thick": thick}).num_triangles == 2
    assert raises(ValueError, U.build_scene, stage2, volumes={"thick": crop})          # wrong shape


def pixel_positions(camera, picture):
    """World position of what each pixel shows: eye + ray direction * depth (perspective camera,
    one ray per pixel).  Rows without a hit are NaN."""
    right, up, forward = camera.basis()
    span_y = 2.0 * math.tan(math.radians(camera.fov) / 2.0)
    span_x = span_y * camera.width / camera.height
    u = (np.arange(camera.width) + 0.5) / camera.width - 0.5
    v = 0.5 - (np.arange(camera.height) + 0.5) / camera.height
    direction = forward[None, None, :] + right[None, None, :] * (u[None, :, None] * span_x) + up[None, None, :] * (v[:, None, None] * span_y)
    direction /= np.linalg.norm(direction, axis=2, keepdims=True)
    depth = np.where(np.isfinite(picture.depth), picture.depth, np.nan)
    return np.asarray(camera.eye)[None, None, :] + direction * depth[..., None]


def test_object_and_depth_buffers_show_each_layer_inside_the_box():
    U = need("ncntm.viz.usd_to_emray")
    r = real()
    p = r.prims
    height = 120 if r.small else 360
    # how far each kind may reach out of the array box, in micrometres:
    marker = 1.5 * r.skeleton.radii_nm.max() / NM_PER_UNIT         # a node marker at the border of the array
    wire = r.prims.wire.GetAttribute("em:widthNm").Get() / NM_PER_UNIT
    report = []
    for space in SPACES:
        r.ds.set_space(space)
        camera = U.camera_from_stage(r.stage, "/World/Cameras/overview", height=height, ssaa=1)
        lo, hi = np.zeros(3), r.grid.extent_nm(space) / NM_PER_UNIT
        means = np.array(p.splats.GetAttribute("em:gaussian:means").Get()) / NM_PER_UNIT
        deviations = np.sqrt(np.einsum("nii->ni", np.array(p.splats.GetAttribute("em:gaussian:covariances").Get()))) / NM_PER_UNIT

        def reach(k):          # of the k-sigma boxes of the Gaussians
            return max(0.0, (lo - (means - k * deviations)).max(), ((means + k * deviations) - hi).max())

        cases = [("bricks", p.bricks, 1e-4), ("surface", p.surface, 1e-4), ("wire", p.wire, wire),
                 ("skeleton tubes", p.tubes, marker), ("skeleton nodes", p.nodes, marker),
                 ("gaussians", p.splats, reach(2.0)),               # fuzzy: half opacity is reached within 2 sigma
                 ("ellipsoids", p.ellipsoids, reach(math.sqrt(5.0)) + 1e-3)]
        for name, prim, tolerance in cases:
            with showing(r, prim):
                scene = U.build_scene(r.stage, tube_sides=4 if r.small else 10, sphere_subdiv=1 if r.small else 2)
                picture = scene.render(camera)
            assert [o["path"] for o in scene.objects.values()] == [str(prim.GetPath())]
            hit = picture.object_id == 1
            assert set(np.unique(picture.object_id)) == {-1, 1}
            assert hit.mean() > 0.002, (name, space, hit.mean())            # it is in the picture
            assert np.array_equal(hit, np.isfinite(picture.depth))
            points = pixel_positions(camera, picture)[hit]
            outside = max(0.0, (lo - points).max(), (points - hi).max())
            assert outside <= tolerance, (name, space, outside, tolerance)
            report.append(f"{name:15s} {space:8s}: {hit.mean():.3f} of the picture, z {points[:, 2].min():.3f} .. "
                          f"{points[:, 2].max():.3f} um, out of the box by {outside:.4f} (allowed {tolerance:.4f})")
    r.ds.set_space("physical")
    print("\n  " + "\n  ".join(report))


def test_picture_of_all_layers_in_both_spaces():
    U = need("ncntm.viz.usd_to_emray")
    r = real()
    height = 180 if r.small else 900
    for space in SPACES:
        r.ds.set_space(space)
        detail = dict(tube_sides=4 if r.small else 10, sphere_subdiv=1 if r.small else 2)
        camera = U.default_camera(r.stage, height=height)
        scene = U.build_scene(r.stage, **detail)
        picture = scene.render(camera)
        assert {o["layer"] for o in scene.objects.values()} == set(S.LAYERS)            # every layer is in the scene
        # the object-id buffer holds the nearest surface of each pixel, so only the outer layers appear in it
        layers_seen = {scene.objects[i]["layer"] for i in np.unique(picture.object_id) if i > 0}
        assert {"rawvalue", "voxvalue", "instance", "frame"} <= layers_seen, (space, layers_seen)
        # the Gaussians are inside the instance: they change the colours, not the nearest surface
        without = U.build_scene(r.stage, layers=[n for n in S.LAYERS if n != "gaussian"], **detail).render(camera)
        assert np.abs(picture.rgba_float - without.rgba_float).max() > 0.05
        assert picture.rgba.shape == (height, height * 4 // 3, 4)
        path = out_dir() / f"snemi3d_all_layers_{space}.png"
        picture.save(str(path))
        assert path.stat().st_size > 10000
        print(f"\n  {path}: {scene.num_triangles} triangles, {scene.num_gaussians} Gaussians, {picture.seconds:.2f} s on {device()}")
    r.ds.set_space("physical")


def test_camera_authored_on_the_stage_drives_the_warp_picture():
    U, emray = need("ncntm.viz.usd_to_emray", "ncntm.viz.emray")
    r = real()
    direction = (1.0, -1.0, 0.6)
    fitted = emray.Camera.fit(np.zeros(3), r.grid.extent_xyz_nm / NM_PER_UNIT, direction=direction)   # the same rule
    camera = U.camera_from_stage(r.stage, "/World/Cameras/overview", height=300)
    assert camera.mode == "perspective" and abs(camera.fov - 35.0) < 1e-5       # focalLength is a float attribute
    assert (camera.width, camera.height) == (400, 300)
    assert np.allclose(camera.eye, fitted.eye, rtol=1e-12)
    for mine, theirs in zip(camera.basis(), fitted.basis()):
        assert np.allclose(mine, theirs, atol=1e-12)
    usd_camera = UsdGeom.Camera(r.prims.camera)
    distance = np.linalg.norm(np.asarray(fitted.eye) - np.asarray(fitted.target))
    assert np.allclose(usd_camera.GetClippingRangeAttr().Get(), (0.01 * distance, 100.0 * distance), rtol=1e-6)
    assert np.array_equal(np.asarray(U.default_camera(r.stage, height=300).eye), np.asarray(camera.eye))
    # the centre of the box is in the middle of the picture, in the Warp camera and in USD's own camera
    centre = 0.5 * r.grid.extent_xyz_nm / NM_PER_UNIT
    assert np.allclose(camera.project(centre), [[200.0, 150.0]], atol=1e-6)
    gf = usd_camera.GetCamera()
    ndc = gf.frustum.ComputeViewMatrix().Transform(Gf.Vec3d(*centre))
    assert abs(ndc[0]) < 1e-9 and abs(ndc[1]) < 1e-9 and ndc[2] < 0             # on the axis, in front (-Z)
    # an orthographic camera: the picture height in stage units survives the tenths-of-a-unit apertures
    t = build_toy(tmp_dir() / "camera" / "toy.usda")
    S.author_camera(t.stage, "side", eye=(0.12, -1.0, 0.15), target=(0.12, 0.0, 0.15), projection="orthographic",
                    ortho_height=0.4, aspect=2.0)
    side = U.camera_from_stage(t.stage, "/World/Cameras/side", height=100)
    assert side.mode == "orthographic" and abs(side.ortho_height - 0.4) < 1e-6 and side.width == 200
    assert np.allclose(side.basis()[2], (0, 1, 0)) and np.allclose(side.basis()[1], (0, 0, 1))
    assert raises(ValueError, U.camera_from_stage, t.stage, "/World/toy")
    assert S.validate(t.stage) == []


def test_visibility_purpose_layers_and_unknown_prims():
    U = need("ncntm.viz.usd_to_emray")
    t = build_toy(tmp_dir() / "adapter" / "toy.usda")
    scene = U.build_scene(t.stage, volumes=t.volumes)
    paths = {o["path"] for o in scene.objects.values()}
    shown, hidden = (t.prims.splats, t.prims.ellipsoids) if S.splat_schema_available() else (t.prims.ellipsoids, t.prims.splats)
    assert str(shown.GetPath()) in paths and str(hidden.GetPath()) not in paths            # invisible prims are left out
    assert not any("proto_" in path or "/looks/" in path for path in paths)
    everything = len(paths)
    # layers
    scene = U.build_scene(t.stage, layers=["skeleton", "frame"], volumes=t.volumes)
    assert {o["layer"] for o in scene.objects.values()} == {"skeleton", "frame"}
    assert U.build_scene(t.stage, layers="rawvalue", volumes=t.volumes).num_triangles == 6      # three quads
    # visibility of a layer and of one prim
    t.ds.set_visible("instance", False)
    t.ds.set_visible("frame/scalebar", False)
    scene = U.build_scene(t.stage, volumes=t.volumes)
    assert {o["layer"] for o in scene.objects.values()} == set(S.LAYERS) - {"instance"}
    assert len(scene.objects) == everything - 6
    t.ds.set_visible("instance", True)
    t.ds.set_visible("frame/scalebar", True)
    # purpose: guides are not drawn
    UsdGeom.Imageable(t.prims.wire).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    assert len(U.build_scene(t.stage, volumes=t.volumes).objects) == everything - 1
    UsdGeom.Imageable(t.prims.wire).GetPurposeAttr().Clear()
    # prim types the ray caster has no drawing for are skipped with one warning that names them
    UsdGeom.Cylinder.Define(t.stage, "/World/extra/can")
    UsdGeom.Cone.Define(t.stage, "/World/extra/hat")
    cubic = UsdGeom.BasisCurves.Define(t.stage, "/World/extra/cubic")
    cubic.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    cubic.CreateCurveVertexCountsAttr([4])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scene = U.build_scene(t.stage, volumes=t.volumes)
    assert len(scene.objects) == everything
    caught = [w for w in caught if "build_scene" in str(w.message)]
    assert len(caught) == 1
    message = str(caught[0].message)
    for needle in ("prim type Cylinder", "/World/extra/can", "prim type Cone", "/World/extra/hat",
                   "BasisCurves that are not linear", "/World/extra/cubic"):
        assert needle in message
    # a clean stage gives no warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        U.build_scene(toy().stage, volumes=toy().volumes)
    assert [w for w in caught if "build_scene" in str(w.message)] == []


def test_adapter_reads_plain_gprims_and_a_foreign_splat_prim():
    U, emray = need("ncntm.viz.usd_to_emray", "ncntm.viz.emray")
    stage = Usd.Stage.CreateInMemory()
    S.configure_stage(stage)
    scaled = UsdGeom.Xform.Define(stage, "/World/scaled")
    scaled.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    scaled.AddScaleOp().Set(Gf.Vec3f(2.0, 3.0, 4.0))
    cube = UsdGeom.Cube.Define(stage, "/World/scaled/cube")                 # size 2 by default: [-1, 1]^3
    sphere = UsdGeom.Sphere.Define(stage, "/World/scaled/sphere")           # radius 1
    sphere.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*S.srgb_to_linear((0.2, 0.4, 0.6)))])
    points = UsdGeom.Points.Define(stage, "/World/points")
    points.CreatePointsAttr([(0, 0, 0), (0, 0, 5)])
    points.CreateWidthsAttr([1.0, 2.0])
    quad = UsdGeom.Mesh.Define(stage, "/World/quad")                        # a polygon with five corners
    quad.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1.5, 1, 0), (0.5, 2, 0), (-0.5, 1, 0)])
    quad.CreateFaceVertexCountsAttr([5])
    quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4])
    scene = U.build_scene(stage)
    kinds = {o["type"]: i for i, o in scene.objects.items()}
    assert set(kinds) == {"Cube", "Sphere", "Points", "Mesh"}
    lo, hi = scene.bounds()
    assert np.allclose(lo, [-1.0, -1.0, -1.0], atol=1e-6)                   # the cube: 1 - 2, 2 - 3, 3 - 4
    assert np.allclose(hi, [23.0, 5.0, 7.0], atol=1e-2)                     # the sphere: x = 1 + 2 * (10 + 1)
    assert scene.num_triangles == 12 + 320 + 2 * 320 + 3                    # cube, sphere, two points, pentagon
    camera = emray.Camera(eye=(21.0, 2.0, 40.0), target=(21.0, 2.0, 3.0), up=(0, 1, 0), mode="orthographic",
                          ortho_height=8.0, width=64, height=64, ssaa=1)
    picture = scene.render(camera, ambient=1.0)                             # no shading: the colour as authored
    assert picture.object_id[32, 32] == kinds["Sphere"]
    assert np.array_equal(picture.rgba[32, 32, :3], [51, 102, 153])         # 0.2, 0.4, 0.6 as display values
    assert abs(picture.depth[32, 32] - (40.0 - 7.0)) < 0.1                  # the top of the ellipsoid, z = 3 + 4
    # a splat prim without em:gaussian:* attributes: covariance = R diag(scales^2) R^T
    if not S.splat_schema_available():
        return
    t = toy()
    stage2 = Usd.Stage.CreateInMemory()
    S.configure_stage(stage2)
    prim = stage2.DefinePrim("/World/splats", "ParticleField3DGaussianSplat")
    for name in ("positions", "orientations", "scales", "opacities", "radiance:sphericalHarmonicsDegree",
                 "radiance:sphericalHarmonicsCoefficients"):
        source = t.prims.splats.GetAttribute(name)
        prim.CreateAttribute(name, source.GetTypeName(), False).Set(source.Get())
    scene = U.build_scene(stage2)
    assert scene.num_gaussians == 2
    g = t.gaussians
    reference = emray.Scene()                  # the same Gaussians, given to the ray caster directly, in nm
    reference.add_gaussians(g.means_xyz_nm, covs=g.covariances_nm2, opacity=0.8, color=g.colors_rgb)
    camera = emray.Camera.fit(g.means_xyz_nm.min(axis=0) - 150.0, g.means_xyz_nm.max(axis=0) + 150.0, width=96, height=96, ssaa=1)
    seen, expected = scene.render(camera), reference.render(camera, device=U.render_device())
    assert expected.rgba_float[..., :3].min() < 0.5                         # the Gaussians are in the picture
    assert np.abs(seen.rgba_float - expected.rgba_float).max() < 2e-3


def test_render_device_follows_the_environment():
    U, wp = need("ncntm.viz.usd_to_emray", "warp")
    saved = os.environ.pop("NCNTM_DEVICE", None)
    try:
        assert U.render_device() == str(wp.get_preferred_device())
        os.environ["NCNTM_DEVICE"] = "cpu"
        assert U.render_device() == "cpu"
        assert U.render_device("cuda:1") == "cuda:1"
    finally:
        os.environ.pop("NCNTM_DEVICE", None)
        if saved is not None:
            os.environ["NCNTM_DEVICE"] = saved


def test_render_in_one_call():
    U = need("ncntm.viz.usd_to_emray")
    t = toy()
    picture = U.render(t.stage, volumes=t.volumes, height=120)
    assert picture.rgba.shape == (120, 160, 4)
    assert set(np.unique(picture.object_id)) - {-1} != set()
    same = U.render(t.stage, camera="/World/Cameras/overview", volumes=t.volumes, height=120)
    assert np.array_equal(picture.rgba, same.rgba)


if __name__ == "__main__":
    failed = []
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"PASS  {name}")
            except Skip as reason:
                print(f"SKIP  {name}: {reason}")
            except Exception:
                traceback.print_exc()
                print(f"FAIL  {name}")
                failed.append(name)
    print(f"\n{len(failed)} failed" if failed else "\nall tests passed")
    sys.exit(1 if failed else 0)
