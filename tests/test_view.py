"""Tests of ncntm.viz.view: one call for a picture of the stage, with Warp or with RTX.

    python -m pytest tests/test_view.py        or        python tests/test_view.py

What is tested, and with what
    the Warp picture       against usd_to_emray.build_scene + emray used directly, in both spaces
    the fallback           no session, a Kit-free session, a real kernel in tier "off" and one in
                           tier "usd": the Warp picture comes back and why() says why; backend="rtx"
                           raises "no renderer: ..."
    the stage afterwards   root layer, session layer, selected variant and edit target are as before
    the scene cache        one build for repeated pictures; a change of the stage builds again
    the RTX path           with a STAND-IN for the renderer (class StandIn): it draws the stage it
                           is handed through the camera prim it is handed, with the Warp ray caster.
                           That shows that the camera, the space, the layers and the clean-up are
                           right.  It shows nothing about RTX itself: no RTX frame can be produced on
                           a machine without a graphics device.

One small stage is used throughout (``build``): a 48 x 40 x 12 voxel volume of 6 x 6 x 30 nm with
hand-made containers, no HDF5 file.

The two kernel tests execute ``in_kernel`` of this file inside a real Jupyter kernel of the Isaac
Sim interpreter, started the way tests/test_kit.py starts its kernels (kernel spec in a temporary
prefix, Kit's files in a temporary root, crash reporter and telemetry off).  They are skipped with
the reason where test_kit.py's kernel tests are skipped (no jupyter_client, no Isaac Sim, the
licence variable OMNI_KIT_ACCEPT_EULA not set).

Environment
    NCNTM_DEVICE     Warp device for the pictures (default: Warp's choice); the kernels inherit it
    NCNTM_TEST_GPU   GPU index the kernels may use (default 0)
    NCNTM_TEST_OUT   directory for the pictures to look at (default: a temporary directory)
"""
import atexit
import hashlib
import json
import os
import shutil
import subprocess
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

SIZE = (160, 120)                       # (width, height) of the test pictures
CAMERA = "/World/Cameras/overview"
HEAVY = ("pxr", "warp", "PIL", "h5py", "scipy", "isaacsim", "omni", "carb")


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


def raised(error, fn, *args, **kwargs) -> str:
    """The message of the ``error`` that fn(*args, **kwargs) raises; an AssertionError if it does not."""
    try:
        fn(*args, **kwargs)
    except error as caught:
        return str(caught)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {error.__name__}")


@lru_cache(maxsize=None)
def tmp_dir() -> Path:
    path = Path(tempfile.mkdtemp(prefix="ncntm_test_view_"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def out_dir() -> Path:
    """Where the pictures for inspection go."""
    path = Path(os.environ.get("NCNTM_TEST_OUT") or tmp_dir() / "pictures")
    path.mkdir(parents=True, exist_ok=True)
    return path


def digest(picture) -> str:
    return hashlib.sha256(np.ascontiguousarray(picture).tobytes()).hexdigest()[:16]


# ---- the stage of these tests -------------------------------------------------------------------------
def build(path):
    """A small stage with every layer.  No two drawn surfaces lie in one plane, so that a
    picture does not depend on which of two coinciding surfaces a ray meets first."""
    from ncntm.viz import stage as S
    from ncntm.viz.grid import VoxelGrid
    from ncntm.viz.types import Bricks, Gaussians, Mesh, Skeleton

    grid = VoxelGrid((12, 40, 48), (30.0, 6.0, 6.0))                          # (z, y, x) voxels of (dz, dy, dx) nm
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, grid.shape_zyx, dtype=np.uint8)
    instance = np.zeros(grid.shape_zyx, dtype=np.uint8)
    instance[3:9, 10:30, 12:36] = 3                                            # the object that is drawn in every form
    instance[9:11, 30:36, 38:44] = 7                                           # a second one, drawn as bricks only
    stage = S.new_stage(path)
    ds = S.Dataset(stage, "demo", grid, raw=raw, instance=instance, norm_window=(20.0, 200.0))
    ds.add_slices("rawvalue", z=1, upsample=2)                                 # floor
    ds.add_slices("voxvalue", y=39, upsample=2)                                # far wall
    ds.add_slices("instance", x=0, upsample=2)                                 # other far wall
    kji = np.argwhere(instance == 7)
    ds.add_bricks(Bricks(kji, grid.voxel_centers_nm(kji), grid.voxel_size_xyz_nm, instance_id=7, n_voxels_total=len(kji)))
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
             [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]]
    mesh = Mesh(S._box_points(grid.index_to_nm((12, 10, 3)), grid.index_to_nm((36, 30, 9))), np.array(faces, dtype=np.int32),
                instance_id=3)
    ds.add_surface(mesh, opacity=0.4)
    ds.add_wiremesh(mesh, width_nm=3.0)
    nodes = np.array([[6, 20, 14], [6, 20, 24], [6, 20, 34], [6, 12, 24], [4, 20, 24]])
    ds.add_skeleton(Skeleton(grid.voxel_centers_nm(nodes), np.array([[0, 1], [1, 2], [1, 3], [1, 4]], dtype=np.int32),
                             np.array([15.0, 24.0, 15.0, 12.0, 12.0]), instance_id=3))
    ds.add_gaussians(Gaussians(grid.voxel_centers_nm(np.array([[6, 16, 18], [5, 24, 30]])),
                               np.array([np.diag([400.0, 100.0, 900.0]), np.diag([100.0, 400.0, 400.0])]), np.array([0.5, 0.5]),
                               colors_rgb=np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]]), instance_id=3))
    ds.add_frame()
    ds.author_camera()
    ds.save()
    return SimpleNamespace(stage=stage, ds=ds, path=Path(path), grid=grid, volumes={"demo": {"raw": raw, "instance": instance}})


_FRESH = [0]


def fresh(name=None):
    """A stage of its own for a test that changes it."""
    _FRESH[0] += 1
    return build(tmp_dir() / (name or f"stage_{_FRESH[0]}") / "demo.usda")


@lru_cache(maxsize=None)
def shared():
    """The stage of the tests that only look."""
    need("pxr", "warp")
    return fresh("shared")


def direct(b, height=SIZE[1], **build_options):
    """The picture as usd_to_emray and emray give it, without view."""
    U = need("ncntm.viz.usd_to_emray")
    return U.build_scene(b.stage, volumes=b.volumes, **build_options).render(U.camera_from_stage(b.stage, CAMERA, height=height)).rgba


def state_of(stage) -> dict:
    """Everything view must leave alone, as text."""
    from ncntm.viz import stage as S

    return {"root": stage.GetRootLayer().ExportToString(), "session": stage.GetSessionLayer().ExportToString(),
            "space": [ds.space for ds in S.find_datasets(stage)], "dirty": stage.GetRootLayer().dirty,
            "edit_target": stage.GetEditTarget().GetLayer().identifier}


class StandIn:
    """Stands in for a ``kit.Session`` whose RTX renderer works.  ``capture_rtx`` draws the stage
    the session holds, as it is composed at that moment, through the camera PRIM it is handed, with
    the Warp ray caster, and notes what it found on the stage.  ``holder`` is anything with a
    ``stage`` and an ``update``: a SimpleNamespace here, a real session inside a kernel."""

    tier = "full"

    def __init__(self, holder, volumes=None, ssaa=2, fail=None):
        self.holder, self.volumes, self.ssaa, self.fail, self.seen, self.updates = holder, volumes, ssaa, fail, [], 0

    @property
    def stage(self):
        return self.holder.stage

    def renderer_status(self):
        from ncntm.viz import kit

        return kit.RendererStatus(True, "a stand-in for RTX")

    def update(self, n=1):
        self.updates += n
        if hasattr(self.holder, "update"):
            self.holder.update(n)

    def capture_rtx(self, camera_path, width=1280, height=720, subframes=32, mode="RealTimePathTracing", spp=None):
        from pxr import UsdGeom, UsdLux

        from ncntm.viz import stage as S
        from ncntm.viz import usd_to_emray as U

        if self.fail is not None:
            raise self.fail
        stage = self.stage
        camera = U.camera_from_stage(stage, camera_path, height=height, ssaa=self.ssaa)
        hidden = [str(p.GetPath()) for p in stage.Traverse()
                  if UsdGeom.Imageable(p) and UsdGeom.Imageable(p).GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible]
        self.seen.append({
            "camera": str(camera_path), "size": [int(width), int(height)], "camera_width": camera.width,
            "subframes": subframes, "mode": mode, "spaces": [ds.space for ds in S.find_datasets(stage)],
            "lights": [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)], "hidden": hidden,
            "clip": [float(v) for v in UsdGeom.Camera(stage.GetPrimAtPath(camera_path)).GetClippingRangeAttr().Get()],
            "edit_target_is_root": stage.GetEditTarget().GetLayer() == stage.GetRootLayer()})
        return U.build_scene(stage, volumes=self.volumes).render(camera).rgba


def holding(b):
    """A stand-in session that holds the stage of ``b`` the way Isaac Sim does after
    session.open_stage(path): as another Usd.Stage on the same root layer."""
    from pxr import Usd

    return StandIn(SimpleNamespace(stage=Usd.Stage.Open(b.stage.GetRootLayer())), b.volumes)


def kit_free_session():
    """A real ``kit.Session`` of tier "off" that is not registered as the session of this process."""
    from ncntm.viz import kit

    probe = kit.GraphicsProbe(False, "no graphics device (says this test)")
    return kit.Session("off", kit.TierChoice("off", "argument", "a test asked for it", probe))


class no_session:
    """Context manager: this process has no Kit session, whatever ran before."""

    def __enter__(self):
        from ncntm.viz import kit

        self.state = kit._state()
        self.saved = self.state["session"]
        self.state["session"] = None

    def __exit__(self, *exc):
        self.state["session"] = self.saved


# ---- 1. the module ------------------------------------------------------------------------------------
def test_importing_view_loads_neither_pxr_nor_warp():
    """``from ncntm.viz import kit, view`` in the first cell must be harmless: pxr and warp have
    to come after kit.start_kit()."""
    code = ("import json, sys\nsys.path.insert(0, sys.argv[1])\n"
            "from ncntm.viz import kit, view\n"
            f"print(json.dumps(sorted(m for m in {HEAVY!r} if m in sys.modules)))\n")
    done = subprocess.run([sys.executable, "-c", code, str(REPO)], capture_output=True, text=True, timeout=300,
                          stdin=subprocess.DEVNULL, cwd=str(REPO.parent), env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == [], done.stdout


def test_view_imports_no_isaac_sim():
    import ast

    tree = ast.parse((REPO / "ncntm" / "viz" / "view.py").read_text())
    tops = {(n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0] for n in ast.walk(tree)
            if (isinstance(n, ast.ImportFrom) and n.level == 0 and n.module) or isinstance(n, ast.Import)
            for a in (n.names if isinstance(n, ast.Import) else [None])}
    assert not tops & {"isaacsim", "omni", "carb"}, tops


# ---- 2. the Warp picture ------------------------------------------------------------------------------
def test_warp_picture_is_what_usd_to_emray_and_emray_give_in_both_spaces():
    view = need("ncntm.viz.view")
    need("pxr", "warp")
    b = fresh()
    before = state_of(b.stage)
    physical = view.render(b.stage, size=SIZE, backend="warp", volumes=b.volumes)
    index = view.render(b.stage, size=SIZE, space="index", backend="warp", volumes=b.volumes)
    assert state_of(b.stage) == before                                      # the stage is still in physical space
    assert isinstance(physical, view.Picture) and isinstance(physical, np.ndarray)
    assert physical.shape == (SIZE[1], SIZE[0], 4) and physical.dtype == np.uint8
    assert physical.backend == "warp" and view.last_backend() == "warp"
    assert np.array_equal(physical, direct(b))                              # the stage as it is: physical
    assert np.array_equal(view.render(b.stage, size=SIZE, space="physical", backend="warp", volumes=b.volumes), physical)
    assert np.abs(physical.astype(int) - index.astype(int)).max() > 50      # the two spaces are different pictures
    b.ds.set_space("index")                                                 # now select index space on the stage itself
    assert np.array_equal(direct(b), index)
    assert np.array_equal(view.render(b.stage, size=SIZE, backend="warp", volumes=b.volumes), index)       # space=None: as selected
    assert np.array_equal(view.render(b.stage, size=SIZE, space="physical", backend="warp", volumes=b.volumes), physical)
    assert b.ds.space == "index"
    b.ds.set_space("physical")
    ink = np.any(physical[..., :3] < 250, axis=-1).mean()
    assert ink > 0.05, ink                                                  # not an empty picture
    physical.save(out_dir() / "view_physical.png")
    index.save(out_dir() / "view_index.png")
    print(f"\n  physical {digest(physical)}, index {digest(index)}, {100 * ink:.0f} % of the pixels are drawn")


def test_layers_and_build_options_reach_the_ray_caster():
    view, U = need("ncntm.viz.view", "ncntm.viz.usd_to_emray")
    b = shared()
    camera = U.camera_from_stage(b.stage, CAMERA, height=SIZE[1])
    some = view.render(b.stage, size=SIZE, layers=["instance", "frame"], backend="warp", volumes=b.volumes)
    assert np.array_equal(some, U.build_scene(b.stage, layers=["instance", "frame"], volumes=b.volumes).render(camera).rgba)
    assert not np.array_equal(some, view.render(b.stage, size=SIZE, backend="warp", volumes=b.volumes))
    one = view.render(b.stage, size=SIZE, layers="frame", backend="warp")   # a single name; the frame needs no volume
    assert np.array_equal(one, U.build_scene(b.stage, layers=["frame"]).render(camera).rgba)
    coarse = view.render(b.stage, size=SIZE, backend="warp", volumes=b.volumes, tube_sides=4, sphere_subdiv=1)
    assert np.array_equal(coarse, direct(b, tube_sides=4, sphere_subdiv=1))
    # keywords of emray.Scene.render and of the camera pass through
    dim = view.render(b.stage, size=SIZE, backend="warp", volumes=b.volumes, ambient=0.9, ssaa=1, background=(0.0, 0.0, 0.0))
    wanted = U.build_scene(b.stage, volumes=b.volumes).render(
        U.camera_from_stage(b.stage, CAMERA, height=SIZE[1], ssaa=1, background=(0.0, 0.0, 0.0)), ambient=0.9).rgba
    assert np.array_equal(dim, wanted) and tuple(dim[0, 0]) == (0, 0, 0, 255)
    assert "unknown layer" in raised(ValueError, view.render, b.stage, layers=["instanse"], backend="warp")
    assert "no_such_option" in raised(TypeError, view.render, b.stage, size=SIZE, backend="warp", volumes=b.volumes, no_such_option=1)


def test_camera_size_and_what_stands_for_a_stage():
    view, U, emray = need("ncntm.viz.view", "ncntm.viz.usd_to_emray", "ncntm.viz.emray")
    b = shared()
    assert view.render(b.stage, backend="warp", volumes=b.volumes).shape == (720, 960, 4)        # the default size
    wide = view.render(b.stage, size=(200, 100), backend="warp", volumes=b.volumes)
    assert wide.shape == (100, 200, 4)
    assert view.camera_for(b.stage, size=(200, 100)).fov == view.camera_for(b.stage, size=SIZE).fov   # vertical field of view kept
    by_path = view.render(b.stage, CAMERA, size=SIZE, backend="warp", volumes=b.volumes)
    assert np.array_equal(by_path, view.render(b.stage, size=SIZE, backend="warp", volumes=b.volumes))    # overview is the default
    own = emray.Camera.for_slice(b.grid, "y", "physical", pixels_per_voxel=2)                   # a camera with a size of its own
    face_on = view.render(b.stage, own, size=(7, 5), backend="warp", volumes=b.volumes)
    assert face_on.shape == (own.height, own.width, 4) == (12 * 5 * 2, 48 * 2, 4)               # 5 rows per voxel: 30 nm / 6 nm
    assert np.array_equal(face_on, U.build_scene(b.stage, volumes=b.volumes).render(own).rgba)
    assert np.array_equal(view.render(b.ds, size=SIZE, backend="warp", volumes=b.volumes), by_path)     # a Dataset stands for its stage
    assert "Usd.Stage" in raised(TypeError, view.render, "demo.usda")
    assert "backend" in raised(ValueError, view.render, b.stage, backend="hydra")
    assert "space" in raised(ValueError, view.render, b.stage, space="voxel", backend="warp")
    assert "size" in raised(ValueError, view.render, b.stage, size=(0, 10), backend="warp")
    # what render does not return: depth and object ids come from the kept scene
    image = view.scene(b.stage, volumes=b.volumes).render(view.camera_for(b.stage, size=SIZE))
    assert np.array_equal(image.rgba, by_path) and image.depth.shape == (SIZE[1], SIZE[0]) and image.object_id.max() > 0


def test_picture_is_an_array_that_shows_and_saves():
    view, figure = need("ncntm.viz.view", "ncntm.viz.figure")
    from PIL import Image as PILImage

    b = shared()
    picture = view.render(b.stage, size=SIZE, backend="warp", volumes=b.volumes)
    assert picture._repr_png_()[:8] == b"\x89PNG\r\n\x1a\n"                  # shows inline in Jupyter
    path = picture.save(tmp_dir() / "picture.png")
    assert np.array_equal(np.asarray(PILImage.open(path)), picture)
    assert type(picture.rgba) is np.ndarray and np.array_equal(figure.to_array(picture), picture)
    assert figure.side_by_side(picture, picture, captions=["a", "b"]).width > 2 * SIZE[0]
    part = picture[10:20, :, :3]
    assert part.backend == "warp" and part._repr_png_()[:4] == b"\x89PNG"    # a crop is a picture too
    assert picture[0, 0]._repr_png_() is None and float(picture.mean()) > 0  # one pixel is not


# ---- 3. the stage afterwards --------------------------------------------------------------------------
def test_stage_is_left_as_it_was_found():
    """space and layers of a picture are not selections on the stage: root layer, session layer,
    selected variant and edit target are the same afterwards, and the switch works as before."""
    view = need("ncntm.viz.view")
    from pxr import Usd, UsdGeom

    b = fresh()
    before = state_of(b.stage)
    assert before["session"].strip() == "#usda 1.0" and before["space"] == ["physical"] and not before["dirty"]
    for space in (None, "physical", "index"):
        for layers in (None, ["skeleton", "frame"]):
            view.render(b.stage, size=SIZE, space=space, layers=layers, backend="warp", volumes=b.volumes)
            assert state_of(b.stage) == before, (space, layers)
    # a picture that fails half-way leaves nothing behind either (index space is selected, then the build fails)
    view.clear_cache()
    raised(LookupError, view.render, b.stage, size=SIZE, space="index", backend="warp")           # no volumes for the slices
    assert state_of(b.stage) == before and view._SCENES == [] and view._WATCHES == []
    # the switch still works: a session-layer selection left behind would overrule it
    b.ds.set_space("index")
    assert b.ds.space == "index" and np.allclose(b.ds.index_to_world()[2, 2], 0.006)
    b.ds.set_space("physical")
    assert np.allclose(b.ds.index_to_world()[2, 2], 0.030)
    # what somebody else put into the session layer stays there, and is in force while the picture is drawn
    with Usd.EditContext(b.stage, b.stage.GetSessionLayer()):
        UsdGeom.Imageable(b.ds.find("skeleton")).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    with_opinion = state_of(b.stage)
    assert "invisible" in with_opinion["session"]
    hidden = view.render(b.stage, size=SIZE, space="index", backend="warp", volumes=b.volumes)
    assert state_of(b.stage) == with_opinion
    assert np.array_equal(hidden, view.render(b.stage, size=SIZE, space="index", layers=[n for n in b.ds.layers() if n != "skeleton"],
                                              backend="warp", volumes=b.volumes))
    assert state_of(b.stage) == with_opinion


# ---- 4. no renderer -----------------------------------------------------------------------------------
def test_auto_gives_the_warp_picture_and_says_why_when_no_renderer_exists():
    view = need("ncntm.viz.view")
    b = shared()
    reference = view.render(b.stage, size=SIZE, backend="warp", volumes=b.volumes)
    assert view.last_backend() == "warp" and "warp" in view.why() and reference.why == view.why()
    with no_session():                                                       # kit.start_kit() was never called
        picture = view.render(b.stage, size=SIZE, volumes=b.volumes)
        assert np.array_equal(picture, reference) and picture.backend == "warp" == view.last_backend()
        assert "no Isaac Sim session" in view.why() and "start_kit" in view.why() and picture.why == view.why()
        message = raised(RuntimeError, view.render, b.stage, size=SIZE, backend="rtx", volumes=b.volumes)
        assert message.startswith("no renderer: ") and "no Isaac Sim session" in message
    off = kit_free_session()                                                 # Isaac Sim not running: tier "off"
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # an absent renderer is no reason for a warning
        picture = view.render(b.stage, size=SIZE, session=off, volumes=b.volumes)
    assert np.array_equal(picture, reference) and view.last_backend() == "warp"
    assert view.why() == off.renderer_status().reason and "tier 'off'" in view.why()
    message = raised(RuntimeError, view.render, b.stage, size=SIZE, backend="rtx", session=off, volumes=b.volumes)
    assert message == "no renderer: " + off.renderer_status().reason        # the session's own sentence
    assert message == raised(RuntimeError, off.capture_rtx, CAMERA, 64, 64)


def test_compare_backends_without_rtx_is_the_warp_picture_with_the_reason():
    view, figure = need("ncntm.viz.view", "ncntm.viz.figure")
    b = shared()
    off = kit_free_session()
    montage = view.compare_backends(b.stage, size=(480, 360), session=off, volumes=b.volumes)
    assert view.last_backend() == "warp" and view.why() == off.renderer_status().reason
    assert montage.size[0] == 480 and montage.size[1] > 360                 # one picture, a caption above it
    strip = montage.size[1] - 360
    assert np.array_equal(np.asarray(montage)[strip:], view.render(b.stage, size=(480, 360), backend="warp", volumes=b.volumes))
    assert np.asarray(montage)[:strip, :, :3].min() < 100                   # there is text in the caption
    montage.save(out_dir() / "view_compare_no_rtx.png")
    with no_session():
        view.compare_backends(b.stage, size=SIZE, volumes=b.volumes)
        assert "no Isaac Sim session" in view.why()


# ---- 5. the scene cache -------------------------------------------------------------------------------
class counting_builds:
    """Context manager: counts the calls of usd_to_emray.build_scene."""

    def __enter__(self):
        from ncntm.viz import usd_to_emray as U

        self.module, self.original, self.count = U, U.build_scene, 0

        def build_scene(*args, **kwargs):
            self.count += 1
            return self.original(*args, **kwargs)

        U.build_scene = build_scene
        return self

    def __exit__(self, *exc):
        self.module.build_scene = self.original


def test_scene_is_built_once_and_again_when_the_stage_changes():
    view = need("ncntm.viz.view")
    from pxr import Usd

    b = fresh()
    view.clear_cache()
    options = dict(size=SIZE, backend="warp", volumes=b.volumes)
    with counting_builds() as builds:
        first = view.render(b.stage, **options)
        for _ in range(3):
            assert np.array_equal(view.render(b.stage, **options), first)
        assert builds.count == 1                                             # four pictures, one scene
        assert view.scene(b.stage, volumes=b.volumes) is view.scene(b.stage, volumes=b.volumes)
        view.render(b.stage, size=(80, 60), backend="warp", volumes=b.volumes)          # another camera: the same scene
        view.render(b.stage, space="physical", **options)                    # the space the stage is in anyway
        assert builds.count == 1
        index = view.render(b.stage, space="index", **options)               # another space: another scene ...
        view.render(b.stage, layers=["frame"], **options)                    # ... other layers too
        assert builds.count == 3
        view.render(b.stage, **options), view.render(b.stage, space="index", **options)
        assert builds.count == 3                                             # view's own switch of the space dropped nothing
        # a change of the stage: the kept scenes are stale
        b.ds.set_visible("skeleton", False)
        changed = view.render(b.stage, **options)
        assert builds.count == 4 and not np.array_equal(changed, first)
        assert np.array_equal(changed, direct(b)) and builds.count == 5      # (the build of direct() is counted as well)
        b.ds.set_visible("skeleton", True)
        assert np.array_equal(view.render(b.stage, **options), first) and builds.count == 6
        # a change made through ANOTHER stage on the same root layer (Isaac Sim's, after open_stage) is noticed too
        other = Usd.Stage.Open(b.stage.GetRootLayer())
        other.GetPrimAtPath(str(b.ds.find("gaussian").GetPath())).GetAttribute("visibility").Set("invisible")
        assert not np.array_equal(view.render(b.stage, **options), first) and builds.count == 7
        other.GetPrimAtPath(str(b.ds.find("gaussian").GetPath())).GetAttribute("visibility").Set("inherited")
        # the selection on the stage is part of what a scene is: switch it and switch it back
        b.ds.set_space("index")
        assert np.array_equal(view.render(b.stage, **options), index)
        b.ds.set_space("physical")
        assert np.array_equal(view.render(b.stage, **options), first)
        count = builds.count
        view.clear_cache()
        assert np.array_equal(view.render(b.stage, **options), first) and builds.count == count + 1


def test_cache_keeps_a_bounded_number_of_scenes():
    view = need("ncntm.viz.view")
    b = shared()
    view.clear_cache()
    options = dict(size=SIZE, backend="warp", volumes=b.volumes)
    choices = [dict(tube_sides=n) for n in range(3, 3 + view.MAX_SCENES + 2)]
    with counting_builds() as builds:
        for choice in choices:
            view.render(b.stage, **options, **choice)
        assert builds.count == len(choices) and len(view._SCENES) == view.MAX_SCENES and len(view._WATCHES) == 1
        view.render(b.stage, **options, **choices[-1])                       # the newest is kept
        assert builds.count == len(choices)
        view.render(b.stage, **options, **choices[0])                        # the oldest went
        assert builds.count == len(choices) + 1
    other = fresh()                                                          # scenes of a second stage push the first one out
    for choice in choices[:view.MAX_SCENES]:
        view.render(other.stage, **dict(options, volumes=other.volumes), **choice)
    assert len(view._SCENES) == view.MAX_SCENES and [w.stage == other.stage for w in view._WATCHES] == [True]
    view.clear_cache()
    assert view._SCENES == [] and view._WATCHES == []


# ---- 6. the RTX path, with a stand-in for the renderer ------------------------------------------------
def test_rtx_path_hands_the_renderer_the_same_camera_space_and_layers():
    """StandIn.capture_rtx draws what Isaac Sim's stage shows at that moment through the camera prim
    view authored.  If view prepared the stage correctly, that IS the Warp picture."""
    view = need("ncntm.viz.view")
    b = fresh()
    session = holding(b)
    held = session.stage
    assert not held == b.stage and held.GetRootLayer() == b.stage.GetRootLayer()        # two stages, one root layer
    before, before_held = state_of(b.stage), state_of(held)
    for space in (None, "index"):
        for layers in (None, ["instance", "skeleton", "frame"], ["rawvalue"]):
            warp = view.render(b.stage, size=SIZE, space=space, layers=layers, backend="warp", volumes=b.volumes)
            rtx = view.render(b.stage, size=SIZE, space=space, layers=layers, session=session, subframes=7, mode="PathTracing")
            assert rtx.backend == "rtx" == view.last_backend() and view.why() == "" == rtx.why
            assert np.array_equal(rtx, warp), (space, layers, int(np.abs(rtx.astype(int) - warp.astype(int)).max()))
            seen = session.seen[-1]
            assert seen["spaces"] == [space or "physical"] and seen["size"] == list(SIZE) and seen["camera_width"] == SIZE[0]
            assert seen["camera"] == f"/World/Cameras/{view.RTX_CAMERA}" and seen["subframes"] == 7 and seen["mode"] == "PathTracing"
            assert seen["lights"] == [view.RTX_LIGHT]                        # the stage has no light of its own
            assert seen["edit_target_is_root"]                               # the renderer is called with the stage's own edit target
            near, far = seen["clip"]
            assert near < 0.1 and far > 10.0                                 # the 0.3 um block, about 1.5 units from the eye, is inside
            groups = [h for h in seen["hidden"] if h.split("/")[-1] in ("rawvalue", "voxvalue", "instance", "skeleton",
                                                                         "wiremesh", "gaussian", "frame")]
            assert len(groups) == (0 if layers is None else 7 - len(layers)), (layers, groups)
            assert state_of(held) == before_held and state_of(b.stage) == before         # nothing is left on either stage
            assert not held.GetPrimAtPath(seen["camera"]) and not held.GetPrimAtPath(view.RTX_LIGHT)
    assert session.updates == 6 * 3                                          # Kit is advanced before and after each frame
    # a camera of its own, orthographic and pixel-exact
    emray = need("ncntm.viz.emray")
    own = emray.Camera.for_slice(b.grid, "x", "physical", pixels_per_voxel=2)
    session.ssaa = own.ssaa
    assert np.array_equal(view.render(b.stage, own, session=session), view.render(b.stage, own, backend="warp", volumes=b.volumes))
    assert state_of(held) == before_held


def test_rtx_path_adds_no_light_to_a_stage_that_has_one_and_takes_the_light_option():
    view = need("ncntm.viz.view")
    from pxr import UsdLux

    b = fresh()
    session = holding(b)
    view.render(b.stage, size=SIZE, session=session, light=2.5)
    assert session.seen[-1]["lights"] == [view.RTX_LIGHT]
    UsdLux.DistantLight.Define(b.stage, "/World/Sun")
    view.render(b.stage, size=SIZE, session=session)
    assert session.seen[-1]["lights"] == ["/World/Sun"]


def test_rtx_needs_the_stage_isaac_sim_holds():
    view = need("ncntm.viz.view")
    from pxr import Usd

    b = shared()
    reference = view.render(b.stage, size=SIZE, backend="warp", volumes=b.volumes)
    for held, said in ((None, "holds no stage"), (Usd.Stage.CreateInMemory(), "holds another stage")):
        session = StandIn(SimpleNamespace(stage=held))
        with warnings.catch_warnings():
            warnings.simplefilter("error")                                   # a how-to, not a failure: no warning
            picture = view.render(b.stage, size=SIZE, session=session, volumes=b.volumes)
        assert np.array_equal(picture, reference) and view.last_backend() == "warp"
        assert said in view.why() and "session.open_stage(path)" in view.why()
        assert said in raised(view.NotHeld, view.render, b.stage, size=SIZE, backend="rtx", session=session)
        assert session.seen == []                                            # the renderer was never asked
    assert Usd.Stage.CreateInMemory()                                        # (a live handle is true ...)
    assert "Usd.Stage" in raised(TypeError, view.render, None)


def test_failed_capture_gives_the_warp_picture_a_warning_and_a_clean_stage():
    view = need("ncntm.viz.view")
    b = fresh()
    session = holding(b)
    session.fail = RuntimeError("no renderer: RTX returned an empty frame (see kit.log)")
    before = state_of(session.stage)
    reference = view.render(b.stage, size=SIZE, backend="warp", volumes=b.volumes)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        picture = view.render(b.stage, size=SIZE, space="index", layers=["frame"], session=session)
    assert picture.backend == "warp" == view.last_backend() and "RTX returned an empty frame" in view.why()
    assert [w.category for w in caught] == [RuntimeWarning] and "Warp ray caster drew it instead" in str(caught[0].message)
    assert state_of(session.stage) == before                                 # camera, light, space and visibility are taken back
    assert "empty frame" in raised(RuntimeError, view.render, b.stage, size=SIZE, backend="rtx", session=session)
    assert state_of(session.stage) == before
    session.fail = None
    assert np.array_equal(view.render(b.stage, size=SIZE, session=session), reference)


def test_compare_backends_with_two_pictures():
    view = need("ncntm.viz.view")
    b = shared()
    session = holding(b)
    montage = view.compare_backends(b.stage, size=(400, 300), session=session, volumes=b.volumes)
    assert montage.size[0] > 2 * 400 and montage.size[1] > 300
    assert view.last_backend() == "rtx" and view.why() == ""
    montage.save(out_dir() / "view_compare_stand_in.png")
    assert session.seen[-1]["size"] == [400, 300]


def test_missing_textures_are_the_ones_rtx_would_not_find():
    view = need("ncntm.viz.view")
    from pxr import Usd

    b = shared()
    assert view.missing_textures(b.stage) == [] and view.missing_textures(b.ds) == []
    moved = tmp_dir() / "moved_without_textures" / "demo.usda"
    moved.parent.mkdir(parents=True, exist_ok=True)
    assert b.stage.GetRootLayer().Export(str(moved))                         # what session.save(path) does: the layer only
    missing = view.missing_textures(Usd.Stage.Open(str(moved)))
    assert len(missing) == 3 and all(path.startswith("./textures/demo_") and path.endswith(".png") for path in missing)
    copied = tmp_dir() / "moved_with_textures" / "demo.usda"
    b.ds.save(copied)                                                        # Dataset.save copies them
    assert view.missing_textures(Usd.Stage.Open(str(copied))) == []


# ---- 7. inside real kernels ---------------------------------------------------------------------------
def in_kernel(session, folder) -> dict:
    """Runs INSIDE a Jupyter kernel, after kit.start_kit(): the same steps with Kit (tier "usd")
    and without (tier "off").  Returns plain numbers and strings; the asserts are in the tests."""
    from pxr import Usd

    from ncntm.viz import emray, view

    folder = Path(folder)
    facts = {"tier": session.tier, "usd": session.usd_version, "device": emray.default_device()}
    b = build(folder / "demo.usda")
    options = dict(size=SIZE, volumes=b.volumes)
    before = state_of(b.stage)

    # 1. Isaac Sim holds nothing yet: auto gives the Warp picture and says why
    facts["held_at_start"] = session.stage is not None
    picture = view.render(b.stage, **options)
    facts["auto"] = {"backend": picture.backend, "why": picture.why, "last": view.last_backend(), "said": view.why(),
                     "equals_direct": bool(np.array_equal(picture, direct(b))), "digest": digest(picture)}
    try:
        view.render(b.stage, backend="rtx", **options)
        facts["rtx"] = "a picture"
    except RuntimeError as error:
        facts["rtx"] = str(error)
    facts["status"] = session.renderer_status().reason
    index = view.render(b.stage, space="index", **options)
    facts["index_digest"], facts["left_alone"] = digest(index), state_of(b.stage) == before

    # 2. the stage as Isaac Sim holds it (tier "off": a second plain stage on the same layer)
    held = session.open_stage(str(b.path))
    session.update(2)
    facts["same_root_layer"] = held.GetRootLayer().identifier == b.stage.GetRootLayer().identifier
    facts["same_stage_object"] = bool(held == b.stage)
    before_held = state_of(held)
    facts["held_equals"] = bool(np.array_equal(view.render(held, **options), picture))
    facts["held_index_equals"] = bool(np.array_equal(view.render(held, space="index", **options), index))
    facts["held_left_alone"] = state_of(held) == before_held
    facts["missing_textures"] = view.missing_textures(held)

    # 3. the cache listens to this stage: a change made on the authoring stage reaches Isaac Sim's stage
    kept = view.scene(held, volumes=b.volumes)
    facts["scene_kept"] = view.scene(held, volumes=b.volumes) is kept
    b.ds.set_visible("skeleton", False)
    session.update(2)
    facts["scene_rebuilt"] = view.scene(held, volumes=b.volumes) is not kept
    facts["change_seen"] = not np.array_equal(view.render(held, **options), picture)
    b.ds.set_visible("skeleton", True)
    session.update(2)
    facts["change_undone"] = bool(np.array_equal(view.render(held, **options), picture))

    # 4. the code of tier "full" without a renderer (only where Kit runs; as tests/test_kit.py drives it)
    if session.app is not None:
        session.tier, session._renderer = "full", None
        try:
            pretended = view.render(held, **options)
            facts["full"] = {"backend": pretended.backend, "why": pretended.why, "equals": bool(np.array_equal(pretended, picture))}
            try:
                view.render(held, backend="rtx", **options)
                facts["full"]["rtx"] = "a picture"
            except RuntimeError as error:
                facts["full"]["rtx"] = str(error)
        finally:
            session.tier, session._renderer = "usd", None

    # 5. the RTX path on the stage Isaac Sim really holds, with the stand-in renderer
    stand_in = StandIn(session, b.volumes)
    before_held = state_of(session.stage)                                   # (step 3 changed the root layer)
    drawn = view.render(b.stage, space="index", layers=["instance", "skeleton", "frame"], session=stand_in, **options)
    wanted = view.render(b.stage, space="index", layers=["instance", "skeleton", "frame"], backend="warp", **options)
    facts["stand_in"] = {"backend": drawn.backend, "equals": bool(np.array_equal(drawn, wanted)), "seen": stand_in.seen[-1],
                         "updates": stand_in.updates, "left_alone": state_of(session.stage) == before_held,
                         "camera_gone": not session.stage.GetPrimAtPath(f"/World/Cameras/{view.RTX_CAMERA}")}
    other = build(folder / "other" / "demo.usda")                           # a stage Isaac Sim does not hold
    try:
        view.render(other.stage, backend="rtx", session=stand_in)
        facts["not_held"] = "a picture"
    except view.NotHeld as error:
        facts["not_held"] = str(error)

    # 6. the picture a newcomer looks at first
    montage = view.compare_backends(held, size=(480, 360), session=session, volumes=b.volumes)
    montage.save(folder / "compare.png")
    facts["compare"] = {"size": list(montage.size), "last": view.last_backend(), "why": view.why()}
    facts["kit_errors"] = session.kit_errors()
    facts["live"] = bool(Usd.Stage.Open(str(b.path)))
    return facts


@lru_cache(maxsize=None)
def kernel_facts(tier) -> dict:
    """Start a kernel, start Kit in ``tier`` in it, run ``in_kernel`` there.  Skips where a kernel cannot be had."""
    K = need("test_kit")
    try:
        kernel = K.Kernel(f"view {tier}", env={"NCNTM_VIEW_OUT": str(out_dir() / f"kernel_{tier}")})
    except K.Skip as reason:                                                 # the script form of test_kit.py skips this way
        raise Skip(str(reason)) from None
    try:
        kernel.timing["tier"] = tier
        cell = kernel.run(
            K.start_cell(tier)
            + "sys.path.insert(0, os.path.join(os.environ['NCNTM_TEST_REPO'], 'tests'))\n"
              "import warnings\nwarnings.simplefilter('ignore')\n"
              "import test_view\n"
              "print('RESULT ' + json.dumps(test_view.in_kernel(s, os.environ['NCNTM_VIEW_OUT'])))\n", timeout=900)
    finally:
        timing = kernel.shutdown()
    facts = K.result(cell)
    facts["exit_code"] = timing["exit_code"]
    return facts


def check_kernel(facts, tier, no_renderer):
    assert facts["tier"] == tier
    auto = facts["auto"]
    assert auto["backend"] == "warp" == auto["last"] and auto["equals_direct"]
    assert no_renderer in auto["why"] and auto["said"] == auto["why"] == facts["status"]
    assert facts["rtx"] == "no renderer: " + facts["status"]                # explicit RTX: the session's clear error
    assert facts["left_alone"] and facts["held_left_alone"]
    assert facts["same_root_layer"] and not facts["same_stage_object"]      # two stages, one root layer
    assert facts["held_equals"] and facts["held_index_equals"] and facts["missing_textures"] == []
    assert facts["scene_kept"] and facts["scene_rebuilt"] and facts["change_seen"] and facts["change_undone"]
    stand_in = facts["stand_in"]
    assert stand_in["backend"] == "rtx" and stand_in["equals"] and stand_in["left_alone"] and stand_in["camera_gone"]
    assert stand_in["seen"]["spaces"] == ["index"] and stand_in["updates"] == 3
    assert "holds another stage" in facts["not_held"] and "other" in facts["not_held"]
    assert facts["compare"]["size"][0] == 480 and facts["compare"]["last"] == "warp" and facts["compare"]["why"] == facts["status"]
    assert facts["kit_errors"] == [] and facts["live"]
    assert (out_dir() / f"kernel_{tier}" / "compare.png").stat().st_size > 5000


def test_kernel_without_kit_falls_back_to_warp():
    facts = kernel_facts("off")
    check_kernel(facts, "off", "Kit is not running (tier 'off')")
    assert "full" not in facts
    print(f"\n  tier off: OpenUSD {facts['usd']}, device {facts['device']}, picture {facts['auto']['digest']}")


def test_kernel_with_kit_falls_back_to_warp_and_draws_the_same_picture():
    facts = kernel_facts("usd")
    check_kernel(facts, "usd", "tier 'usd' loads no renderer")
    assert facts["exit_code"] == 0
    full = facts["full"]                                                     # tier "full" on a machine without a graphics device
    assert full["backend"] == "warp" and full["equals"]
    assert full["why"].startswith("no Hydra render engine is attached to the USD context")
    assert full["rtx"] == "no renderer: " + full["why"]
    off = kernel_facts("off")
    assert facts["usd"] != off["usd"]                                        # Kit's own OpenUSD, not the interpreter's
    assert facts["auto"]["digest"] == off["auto"]["digest"] and facts["index_digest"] == off["index_digest"]
    print(f"\n  tier usd: OpenUSD {facts['usd']} (tier off: {off['usd']}), device {facts['device']}, the same picture "
          f"{facts['auto']['digest']} with and without Kit")


if __name__ == "__main__":
    failed = []
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"PASS  {name}", flush=True)
            except Skip as reason:
                print(f"SKIP  {name}: {reason}", flush=True)
            except Exception:
                traceback.print_exc()
                print(f"FAIL  {name}", flush=True)
                failed.append(name)
    print(f"\n{len(failed)} failed" if failed else "\nall tests passed")
    sys.exit(1 if failed else 0)
