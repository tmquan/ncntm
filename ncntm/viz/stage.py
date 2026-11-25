"""stage -- the USD stage is the single scene description of ``ncntm.viz``.

The same stage is drawn twice: here by the Warp ray caster (``usd_to_emray.build_scene``
reads the composed stage) and on an RTX-capable machine by Isaac Sim.  A wrong transform,
variant or prim therefore shows up in a Warp picture.

Only ``pxr`` (OpenUSD) and ``numpy`` are imported: no Kit, no Warp, no imaging library.
Inside an Isaac Sim kernel import this module only AFTER ``SimulationApp(...)`` exists
(otherwise the wrong ``pxr`` build is loaded).

Stage
    upAxis Z;  X, Y, Z = x, y, z of the data (no flip, no rotation).
    1 stage unit = 1 micrometre of tissue (``grid.NM_PER_UNIT`` nm).
    metersPerUnit = 1.0: a MAGNIFIED MODEL (1 um is drawn 1 m large), because that is the
    configuration Isaac Sim is built around.  The true scale is recorded in the root
    layer's ``customLayerData["emScale"]`` and in ``em:*`` attributes of every dataset.

One dataset = three prims
    /World/<name>            local frame: NANOMETRES (x, y, z) from the corner of the array.
                             xformOp:scale = 1 / NM_PER_UNIT, uniform.
    /World/<name>/voxels     local frame: VOXEL INDEX (i, j, k), corner convention (voxel
                             [k, j, i] is the box [i, i+1] x [j, j+1] x [k, k+1]).
                             xformOp:scale = the voxel size in nm, e.g. (6, 6, 30).
                             THIS ONE ATTRIBUTE IS THE ANISOTROPY.
    /World/<name>/physical   local frame: nanometres.  For everything that must not sit
                             under a non-uniform scale: tubes, spheres, Gaussians, rulers.

The variant set "space" on /World/<name>
    "physical"   voxels.xformOp:scale = (dx, dy, dz)        the truth
    "index"      voxels.xformOp:scale = (g, g, g), g = min(dx, dy)
                 every voxel drawn as a cube: what a program shows that ignores the voxel size
    Content under ``voxels`` follows that one attribute.  Content under ``physical`` carries
    its own per-space data inside the two variants: points p become A p and covariances S
    become A S A^T with A = diag(grid.space_scale(space)), so both frames always agree.

Layers
    Each representation is a group prim tagged ``em:layer`` = rawvalue | voxvalue | instance
    | skeleton | wiremesh | gaussian | frame, switched with ``Dataset.set_visible``.

Colours given to the functions here are display colours (sRGB values in [0, 1], what you
want to see).  USD stores linear values, so ``displayColor`` and material inputs are
written through ``srgb_to_linear``; the Warp adapter converts back.
"""
from __future__ import annotations

import math
import os
import shutil
import struct
import zlib

import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, Vt

from .grid import NM_PER_UNIT, SPACES, VoxelGrid
from .types import Gaussians, rotation_to_quaternion_wxyz

LAYERS = ("rawvalue", "voxvalue", "instance", "skeleton", "wiremesh", "gaussian", "frame")
SPACE_VARIANT_SET = "space"
SLICE_KINDS = {"rawvalue": ("raw", "grey"), "voxvalue": ("raw", "viridis"), "instance": ("instance", "labels")}
SH_C0 = 0.28209479177387814          # 1 / (2 sqrt(pi)): degree-0 spherical harmonic
K_SIGMA = math.sqrt(5.0)             # semi-axes of the solid ellipsoid with the same covariance
SPLAT_TYPE = "ParticleField3DGaussianSplat"
LABEL_ZERO_GREY = 0.06               # display grey of label 0 (background) in instance slices
# exposure of an authored camera: these five values multiply the picture by exactly 1
CAMERA_EXPOSURE = {"exposure": 0.0, "exposure:fStop": 1.0, "exposure:iso": 100.0,
                   "exposure:responsivity": 1.0, "exposure:time": 1.0}

_DOUBLE = UsdGeom.XformOp.PrecisionDouble     # UsdGeom's default precision for scale is FLOAT
_T = Sdf.ValueTypeNames


# ---- colours ---------------------------------------------------------------------------
def srgb_to_linear(rgb) -> np.ndarray:
    """Display colour (sRGB, [0, 1]) -> linear value as USD stores it."""
    c = np.asarray(rgb, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb) -> np.ndarray:
    """Inverse of ``srgb_to_linear``."""
    c = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def label_rgb(ids) -> np.ndarray:
    """Instance id -> display colour, float32 (..., 3) in [0, 1], by integer hashing.
    The same hash as the Warp ray caster (``emray.label_colors``), so an instance has one
    colour in every layer and in both renderers."""
    h = np.asarray(ids).astype(np.uint64)
    h = (h * 2654435761) & 0xFFFFFFFF
    h = h ^ (h >> 15)
    h = (h * 2246822519) & 0xFFFFFFFF
    h = (h ^ (h >> 13)).astype(np.uint32)
    hue = (h & np.uint32(0xFFFF)).astype(np.float32) / np.float32(65536.0)
    sat = np.float32(0.45) + np.float32(0.45) * ((h >> np.uint32(16)) & np.uint32(0xFF)).astype(np.float32) / np.float32(255.0)
    val = np.float32(0.70) + np.float32(0.30) * ((h >> np.uint32(24)) & np.uint32(0xFF)).astype(np.float32) / np.float32(255.0)
    out = []
    for n in (5.0, 3.0, 1.0):
        k = np.mod(np.float32(n) + hue * np.float32(6.0), np.float32(6.0))
        out.append(val - val * sat * np.clip(np.minimum(k, np.float32(4.0) - k), 0.0, 1.0))
    return np.stack(out, axis=-1).astype(np.float32)


def _to_rgb8(rgb) -> np.ndarray:
    """[0, 1] floats -> uint8 with the rounding the ray caster uses: floor(255 v + 0.5)."""
    return np.floor(np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)


def label_rgb8(ids) -> np.ndarray:
    """Instance ids (any shape) -> uint8 (..., 3) texture colours; id 0 (background) is dark."""
    ids = np.asarray(ids)
    rgb = _to_rgb8(label_rgb(ids))
    rgb[ids == 0] = _to_rgb8(LABEL_ZERO_GREY)
    return rgb


# viridis at 17 equally spaced positions; colormap_rgb8 interpolates linearly between them
_VIRIDIS = ("440154 48186a 472d7b 424086 3b528b 33638d 2c728e 26828e 21918c "
            "1fa088 28ae80 3fbc73 5ec962 84d44b addc30 d8e219 fde725")


def colormap_rgb8(name: str) -> np.ndarray:
    """(256, 3) uint8 table for a value t in [0, 1] at row round(255 t).
    "grey": row v is (v, v, v).  "viridis": a perceptual map (dark blue -> green -> yellow)
    whose luminance rises with t (strictly from any row to the next but one; single rows
    are limited by 8-bit rounding).  No row is a grey, so it cannot be mistaken for rawvalue."""
    if name == "grey":
        return np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    if name == "viridis":
        anchors = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in _VIRIDIS.split()], dtype=np.float64)
        t = np.linspace(0.0, 1.0, 256)
        cols = [np.interp(t, np.linspace(0.0, 1.0, len(anchors)), anchors[:, c]) for c in range(3)]
        return np.rint(np.stack(cols, axis=1)).astype(np.uint8)
    raise ValueError(f"unknown colour map {name!r}: use 'grey' or 'viridis'")


def slice_lut8(colormap: str, window=(0.0, 255.0)) -> np.ndarray:
    """(256, 3) uint8 colour of every possible stored uint8 number v:
    t = clip((v - lo) / (hi - lo), 0, 1) with (lo, hi) = ``window``, then row round(255 t)
    of ``colormap_rgb8(colormap)``.

    rawvalue   colormap "grey", window (0, 255): the identity, v -> (v, v, v).
    voxvalue   colormap "viridis", window = the normalisation window of the volume, so t is
               the voxvalue of the voxel."""
    lo, hi = (float(w) for w in window)
    if not hi > lo:
        raise ValueError(f"window must be (lo, hi) with hi > lo, got {window}")
    t = np.clip((np.arange(256, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return colormap_rgb8(colormap)[np.rint(t * 255.0).astype(np.int64)]


def write_png(path, rgb) -> None:
    """Write an (H, W, 3) uint8 array as an 8-bit RGB PNG (zlib only; row 0 is the top row)."""
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"write_png needs an (H, W, 3) array, got shape {a.shape}")
    h, w, _ = a.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), a.reshape(h, w * 3)], axis=1).tobytes()

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


# ---- small converters to USD value types -----------------------------------------------
def _vec3f(a):
    return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(a, dtype=np.float32).reshape(-1, 3))


def _floats(a):
    return Vt.FloatArray.FromNumpy(np.ascontiguousarray(a, dtype=np.float32).reshape(-1))


def _ints(a):
    return Vt.IntArray.FromNumpy(np.ascontiguousarray(a, dtype=np.int32).reshape(-1))


def _extent(points, pad=0.0):
    """USD extent [min, max] of points (N, 3), grown by ``pad`` (scalar or (3,)) on every side."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return _vec3f(np.stack([p.min(axis=0) - pad, p.max(axis=0) + pad]))


def _quats(q_wxyz, half: bool):
    """(N, 4) quaternions (w, x, y, z) -> Vt.QuathArray (half) or Vt.QuatfArray.
    TRAP: the numpy layout of a Vt quaternion array is (x, y, z, w), real part LAST,
    although Gf.Quatf(w, x, y, z) and .usda text are real-first."""
    xyzw = np.asarray(q_wxyz, dtype=np.float64).reshape(-1, 4)[:, [1, 2, 3, 0]]
    if half:
        return Vt.QuathArray.FromNumpy(np.ascontiguousarray(xyzw, dtype=np.float16))
    return Vt.QuatfArray.FromNumpy(np.ascontiguousarray(xyzw, dtype=np.float32))


def _checked(name, a, shape_tail, kind="f"):
    """A non-empty, finite array whose trailing dimensions are ``shape_tail``, or a clear error."""
    a = np.asarray(a, dtype=np.float64 if kind == "f" else np.int64)
    if a.ndim != 1 + len(shape_tail) or a.shape[1:] != tuple(shape_tail):
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, shape_tail))}), got {a.shape}")
    if len(a) == 0:
        raise ValueError(f"{name} is empty: nothing to draw")
    if kind == "f" and not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")
    return a


def _rgb(color) -> np.ndarray:
    c = np.asarray(color, dtype=np.float64)
    if c.shape != (3,) or not np.isfinite(c).all() or c.min() < 0.0 or c.max() > 1.0:
        raise ValueError(f"a colour is three numbers in [0, 1], got {color!r}")
    return c


# ---- the stage ---------------------------------------------------------------------------
def configure_stage(stage: Usd.Stage) -> Usd.Stage:
    """Give ``stage`` (a new one, or the one Isaac Sim has open) the metadata of this package:
    upAxis Z, metersPerUnit 1.0, defaultPrim /World, and the true scale in
    ``customLayerData["emScale"]`` (existing customLayerData is kept)."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    layer = stage.GetRootLayer()
    data = dict(layer.customLayerData)
    data["emScale"] = {                                   # plain floats and strings only: a Python
        "nanometersPerUnit": float(NM_PER_UNIT),          # list here cannot be stored in .usdc
        "trueMetersPerUnit": NM_PER_UNIT / 1e9,           # what metersPerUnit would honestly be
        "authoredMetersPerUnit": 1.0,                     # what it says
        "magnification": 1e9 / NM_PER_UNIT,               # 1 um of tissue is drawn 1 m large
        "axisOrder": "xyz",                               # USD X, Y, Z = data x, y, z (arrays are z, y, x)
        "voxelOrigin": "corner",                          # voxel (i, j, k) covers [i, i+1]; centre at +0.5
        "convention": "ncntm.viz.stage/1",
    }
    layer.customLayerData = data
    return stage


def new_stage(path) -> Usd.Stage:
    """Create a stage stored in the file ``path`` (.usda or .usdc) and configure it.
    The file is needed from the start because slice textures are written next to it and
    referenced by relative path.  An existing file of that name is replaced."""
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    layer = Sdf.Layer.Find(path)                # still open in this process (a re-run notebook cell)
    if layer is None:
        layer = Sdf.Layer.CreateNew(path)
    else:
        layer.Clear()
    return configure_stage(Usd.Stage.Open(layer))


def save_stage(stage: Usd.Stage, path=None) -> str:
    """Save the root layer; returns the file written.

    With ``path``: write a copy there (.usda or .usdc by extension) and copy the textures it
    references, so the relative paths keep resolving.  The LAYER is exported, never the
    stage: ``Usd.Stage.Export`` flattens composition and the "space" variant set would be
    gone from the file."""
    root = stage.GetRootLayer()
    if path is None or os.path.abspath(os.fspath(path)) == root.realPath:
        if not root.realPath:
            raise ValueError("this stage has no file yet: give save a path")
        root.Save()
        return root.realPath
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not root.Export(path):
        raise IOError(f"could not write {path}")
    source_dir, target_dir = os.path.dirname(root.realPath), os.path.dirname(path)
    if root.realPath and source_dir != target_dir:
        for prim in stage.Traverse():                        # texture readers: UsdUVTexture inputs:file
            file_attr = prim.GetAttribute("inputs:file")
            asset = file_attr.Get() if file_attr else None
            if asset and not os.path.isabs(asset.path):
                os.makedirs(os.path.dirname(os.path.join(target_dir, asset.path)), exist_ok=True)
                shutil.copyfile(os.path.join(source_dir, asset.path), os.path.join(target_dir, asset.path))
    return path


def splat_schema_available() -> bool:
    """True if this USD knows the prim type ParticleField3DGaussianSplat (OpenUSD >= 26.03, or
    Kit 110 through its schema extension omni.usd.schema.usd_particle_field)."""
    return Usd.SchemaRegistry().FindConcretePrimDefinition(SPLAT_TYPE) is not None


def validate(stage: Usd.Stage) -> list:
    """Run every validator this USD build has; returns a list of (severity, name, message).
    An empty list means the stage passed.  Uses UsdValidation (OpenUSD >= 25.02); older
    builds fall back to UsdUtils.ComplianceChecker on the saved file."""
    try:
        from pxr import UsdValidation
    except ImportError:
        UsdValidation = None
    if UsdValidation is not None:
        registry = UsdValidation.ValidationRegistry()
        context = UsdValidation.ValidationContext(registry.GetOrLoadAllValidators())
        return [(str(e.GetType()).rsplit(".", 1)[-1], e.GetName(), e.GetMessage()) for e in context.Validate(stage)]
    from pxr import UsdUtils
    checker = UsdUtils.ComplianceChecker()
    checker.CheckCompliance(stage.GetRootLayer().realPath)
    return [("Error", "ComplianceChecker", m) for m in list(checker.GetErrors()) + list(checker.GetFailedChecks())]


def author_camera(stage: Usd.Stage, name: str = "overview", *, eye, target, up=(0.0, 0.0, 1.0),
                  projection: str = "perspective", fov: float = 35.0, ortho_height: float = 1.0,
                  aspect: float = 4.0 / 3.0) -> UsdGeom.Camera:
    """Author /World/Cameras/<name>, a look-at camera.  ``eye`` and ``target`` are world
    positions in stage units (micrometres).

    perspective    ``fov`` is the VERTICAL field of view in degrees
    orthographic   ``ortho_height`` is the world height of the picture in stage units
    ``aspect`` = width / height of the picture.  The clipping range is authored explicitly
    (near 1 % and far 100 times the eye-target distance): Kit's defaults are meant for
    metre-sized scenes and would cut a micrometre-sized one away.

    A USD camera looks down its local -Z with +Y up, and its apertures are in tenths of a
    stage unit.  The five exposure attributes are authored with the values that give an
    exposure scale of exactly 1 (exposure 0, f-stop 1, ISO 100, responsivity 1, time 1): RTX
    has other defaults for some of them, and the slices are emissive, so their brightness
    would depend on which default applies."""
    if projection not in ("perspective", "orthographic"):
        raise ValueError(f"projection must be 'perspective' or 'orthographic', got {projection!r}")
    eye, target = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)
    forward = target - eye
    distance = float(np.linalg.norm(forward))
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("eye and target must be two different finite points")
    forward /= distance
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(right) < 1e-12:
        raise ValueError("up must not be parallel to the viewing direction")
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    UsdGeom.Scope.Define(stage, "/World/Cameras")
    camera = UsdGeom.Camera.Define(stage, f"/World/Cameras/{Tf.MakeValidIdentifier(name)}")
    matrix = Gf.Matrix4d(1.0)                        # Gf matrices are ROW-vector matrices:
    matrix.SetRow3(0, Gf.Vec3d(*right))              # rows 0..2 are the camera's X, Y, Z axes,
    matrix.SetRow3(1, Gf.Vec3d(*true_up))            # row 3 is its position
    matrix.SetRow3(2, Gf.Vec3d(*(-forward)))
    matrix.SetRow3(3, Gf.Vec3d(*eye))
    camera.ClearXformOpOrder()
    camera.AddTransformOp(_DOUBLE).Set(matrix)
    if projection == "perspective":
        vertical = 24.0                              # any value: only aperture / focal length matters
        camera.CreateFocalLengthAttr(0.5 * vertical / math.tan(math.radians(fov) / 2.0))
    else:
        vertical = 10.0 * float(ortho_height)        # tenths of a stage unit
        camera.CreateFocalLengthAttr(50.0)
    camera.CreateProjectionAttr(projection)
    camera.CreateVerticalApertureAttr(vertical)
    camera.CreateHorizontalApertureAttr(vertical * float(aspect))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01 * distance, 100.0 * distance))
    for key, value in CAMERA_EXPOSURE.items():       # schema attributes of UsdGeom.Camera (custom = False)
        camera.GetPrim().CreateAttribute(key, _T.Float, False).Set(value)
    return camera


# ---- geometry that does not need a stage ---------------------------------------------------
def nice_length_nm(extent_nm: float, fraction: float = 0.25) -> float:
    """Largest length of the form 1, 2 or 5 times a power of ten that is not longer than
    ``fraction`` of ``extent_nm``: a round number for a scale bar."""
    target = float(extent_nm) * fraction
    if not target > 0:
        raise ValueError("extent must be positive")
    power = 10.0 ** math.floor(math.log10(target))
    return float(max(m * power for m in (1.0, 2.0, 5.0) if m * power <= target * (1 + 1e-12)))


def format_length_nm(nm: float) -> str:
    """A length in NANOMETRES as a label: "500 nm" below one micrometre, "2 um" from there on.
    (``figure.format_length`` does the same for a length in micrometres, the unit of pictures.)"""
    return f"{nm / 1000.0:g} um" if nm >= 1000.0 else f"{nm:g} nm"


def _gaussian_extent(means, covariances, k: float):
    """USD extent of the k-sigma ellipsoids of Gaussians: each reaches k sqrt(S_aa) along axis a."""
    half = k * np.sqrt(np.einsum("nii->ni", covariances))
    return _vec3f(np.stack([(means - half).min(axis=0), (means + half).max(axis=0)]))


_BOX_FACES = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 1, 2, 6, 5, 0, 4, 7, 3]     # outward quads
_BOX_EDGES = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]     # 12 segments


def _box_points(lo, hi) -> np.ndarray:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return np.array([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                     (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], dtype=np.float64)


# ---- the dataset ---------------------------------------------------------------------------
class Dataset:
    """One EM volume on a stage: the prims /World/<name>, .../voxels, .../physical, the
    "space" variant set, and the methods that add the layers.

    ``grid`` says what the array is (shape (z, y, x), voxel size (dz, dy, dx) nm).
    ``raw`` / ``instance`` are the (z, y, x) arrays behind the slices; they are only needed
    by ``add_slices`` (to write the textures).  ``sample_file`` is the name of the HDF5 file
    they were read from and ``voxel_offset_xyz`` the first voxel read from it: the Warp
    adapter finds the data again through these two.  ``origin_xyz_nm`` places the corner of
    the array in the world (layout only; the same in both spaces).
    """

    def __init__(self, stage: Usd.Stage, name: str, grid: VoxelGrid, *, raw=None, instance=None,
                 norm_window=(0.0, 255.0), sample_file: str = "", voxel_offset_xyz=(0, 0, 0),
                 origin_xyz_nm=(0.0, 0.0, 0.0)):
        if not isinstance(grid, VoxelGrid):
            raise TypeError("grid must be a VoxelGrid")
        for label, volume in (("raw", raw), ("instance", instance)):
            if volume is not None and tuple(np.shape(volume)) != grid.shape_zyx:
                raise ValueError(f"{label} has shape {np.shape(volume)} but the grid says {grid.shape_zyx} (z, y, x)")
        self.stage, self.grid = stage, grid
        self.name = Tf.MakeValidIdentifier(name)
        self.volumes = {"raw": raw, "instance": instance}
        self.norm_window = (float(norm_window[0]), float(norm_window[1]))
        if not stage.GetRootLayer().customLayerData.get("emScale"):
            raise ValueError("this stage does not have the metadata of this package: make it with "
                             "new_stage(path), or call configure_stage(stage) first")
        stage.RemovePrim(f"/World/{self.name}")                # a re-run cell starts the dataset afresh

        d = grid.voxel_size_xyz_nm
        root = UsdGeom.Xform.Define(stage, f"/World/{self.name}")
        root.AddTranslateOp(_DOUBLE).Set(Gf.Vec3d(*(np.asarray(origin_xyz_nm, dtype=np.float64) / NM_PER_UNIT)))
        root.AddOrientOp(_DOUBLE).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        root.AddScaleOp(_DOUBLE).Set(Gf.Vec3d(1.0 / NM_PER_UNIT))             # nm -> stage unit, UNIFORM
        self.prim = root.GetPrim()
        self.prim.SetDocumentation("Local frame: nanometres (x, y, z) from the corner of the array. Uniform scale only.")
        for key, type_name, value in (
            ("em:voxelSizeNm", _T.Double3, Gf.Vec3d(*d)),                     # (dx, dy, dz)
            ("em:shapeVoxels", _T.Int3, Gf.Vec3i(*(int(n) for n in grid.shape_xyz))),          # (nx, ny, nz)
            ("em:indexEdgeNm", _T.Double, grid.in_plane_nm),                  # g: cube edge in index space
            ("em:anisotropy", _T.Double, grid.anisotropy),
            ("em:nanometersPerUnit", _T.Double, float(NM_PER_UNIT)),
            ("em:arrayAxisOrder", _T.Token, "zyx"),
            ("em:voxelOrigin", _T.Token, "corner"),
            ("em:sampleFile", _T.String, str(sample_file)),
            ("em:voxelOffset", _T.Int3, Gf.Vec3i(*(int(v) for v in voxel_offset_xyz))),        # (i0, j0, k0)
            ("em:normWindow", _T.Double2, Gf.Vec2d(*self.norm_window)),       # rawvalue lo -> voxvalue 0, hi -> 1
        ):
            self.prim.CreateAttribute(key, type_name).Set(value)

        voxels = UsdGeom.Xform.Define(stage, self.prim.GetPath().AppendChild("voxels"))
        voxels.AddTranslateOp(_DOUBLE).Set(Gf.Vec3d(0.0))
        voxels.AddOrientOp(_DOUBLE).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        scale_op = voxels.AddScaleOp(_DOUBLE)          # NO value here: a local opinion would beat the variants
        self.voxels = voxels.GetPrim()
        self.voxels.SetDocumentation("Local frame: voxel index (i, j, k), corner convention. xformOp:scale is the "
                                     "voxel size in nm, set by the 'space' variant of the parent: the anisotropy.")
        self.physical = UsdGeom.Xform.Define(stage, self.prim.GetPath().AppendChild("physical")).GetPrim()
        self.physical.SetDocumentation("Local frame: nanometres. Prims here carry per-space data in the 'space' variants.")
        self.looks = UsdGeom.Scope.Define(stage, self.prim.GetPath().AppendChild("looks")).GetPrim()

        variant_set = self.prim.GetVariantSets().AddVariantSet(SPACE_VARIANT_SET)
        for space in SPACES:
            variant_set.AddVariant(space)
        variant_set.SetVariantSelection("physical")
        self._author(scale_op.GetAttr(), lambda space: Gf.Vec3d(*grid.display_voxel_size_xyz_nm(space)))

    @classmethod
    def from_sample(cls, stage: Usd.Stage, sample, name: str = None, **kw) -> "Dataset":
        """A dataset for a ``samples.Sample`` (whole file or a crop of it).  For a sample whose
        voxels are not voxels of the file (``strided``) no file is recorded, and the Warp
        adapter must be given the arrays (``build_scene(stage, volumes={name: sample})``)."""
        file_voxel_size = sample.attrs.get("voxel_size_zyx_nm")
        from_file = file_voxel_size is not None and np.allclose(file_voxel_size, sample.grid.voxel_size_zyx_nm)
        offset = np.rint(sample.grid.nm_to_index(sample.origin_xyz_nm)).astype(int)
        return cls(stage, name or sample.name, sample.grid, raw=sample.raw, instance=sample.instance,
                   norm_window=sample.norm_window, sample_file=os.path.basename(str(sample.path)) if from_file else "",
                   voxel_offset_xyz=offset if from_file else (0, 0, 0), **kw)

    @classmethod
    def from_prim(cls, prim: Usd.Prim) -> "Dataset":
        """The Dataset object of a dataset prim that is already on a stage (a reopened file).
        Nothing is authored.  It has no arrays: pass ``volume=`` to ``add_slices``."""
        size = prim.GetAttribute("em:voxelSizeNm").Get() if prim else None
        if size is None:
            raise ValueError(f"{prim.GetPath() if prim else prim} is not a dataset prim (no em:voxelSizeNm)")
        self = cls.__new__(cls)
        self.stage, self.prim, self.name = prim.GetStage(), prim, prim.GetName()
        self.grid = VoxelGrid(tuple(prim.GetAttribute("em:shapeVoxels").Get())[::-1], tuple(size)[::-1])
        self.volumes = {"raw": None, "instance": None}
        self.norm_window = tuple(prim.GetAttribute("em:normWindow").Get())
        self.voxels, self.physical, self.looks = (prim.GetChild(n) for n in ("voxels", "physical", "looks"))
        return self

    # ---- the space switch --------------------------------------------------------------------
    @property
    def space(self) -> str:
        """The selected variant of "space": "physical" or "index"."""
        return self.prim.GetVariantSets().GetVariantSet(SPACE_VARIANT_SET).GetVariantSelection()

    def set_space(self, space: str) -> None:
        """Select "physical" (true voxel size) or "index" (every voxel a cube).  This one call
        changes voxels.xformOp:scale and the per-space data of every prim under ``physical``."""
        if space not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, got {space!r}")
        self.prim.GetVariantSets().GetVariantSet(SPACE_VARIANT_SET).SetVariantSelection(space)

    def voxel_scale(self, space: str) -> np.ndarray:
        """voxels.xformOp:scale (x, y, z) in nm as stored in the ``space`` variant of the file."""
        path = self.prim.GetPath().AppendVariantSelection(SPACE_VARIANT_SET, space) \
            .AppendPath("voxels").AppendProperty("xformOp:scale")
        spec = self.stage.GetRootLayer().GetAttributeAtPath(path)
        if spec is None:
            raise ValueError(f"the variant {space!r} holds no voxel scale")
        return np.array(spec.default, dtype=np.float64)

    def index_to_world(self) -> np.ndarray:
        """4x4 matrix (column vectors: p' = M @ [i, j, k, 1]) from continuous voxel index
        coordinates to world coordinates in stage units, READ FROM THE STAGE at the current
        space.  Equals ``grid.index_to_unit_matrix(space)`` for a dataset at the origin."""
        m = UsdGeom.Xformable(self.voxels).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array(m, dtype=np.float64).T                  # Gf matrices are row-vector: transpose

    def _variant_target(self, space: str) -> Usd.EditTarget:
        path = self.prim.GetPath().AppendVariantSelection(SPACE_VARIANT_SET, space)
        return Usd.EditTarget.ForLocalDirectVariant(self.stage.GetRootLayer(), path)

    def _author(self, attr: Usd.Attribute, value_in_space) -> None:
        """Write ``value_in_space(space)`` into each variant of "space".  The attribute must
        exist and must have NO value outside the variants: local opinions are stronger than
        variant opinions, so a single ``attr.Set(...)`` elsewhere would disable the switch."""
        for space in SPACES:
            with Usd.EditContext(self.stage, self._variant_target(space)):
                attr.Set(value_in_space(space))

    def _remove(self, path: Sdf.Path) -> None:
        """Remove a prim of this dataset together with what the two variants say about it."""
        self.stage.RemovePrim(path)
        for space in SPACES:
            with Usd.EditContext(self.stage, self._variant_target(space)):
                self.stage.RemovePrim(path)

    # ---- layers ----------------------------------------------------------------------------------
    def _new_prim_path(self, layer: str, frame: Usd.Prim, name: str) -> Sdf.Path:
        """Path <frame>/<layer>/<name> for a prim about to be defined.  Creates the layer group
        (tagged em:layer) if needed and removes an older prim of that name (re-run cells)."""
        group_path = frame.GetPath().AppendChild(layer)
        if not self.stage.GetPrimAtPath(group_path):
            group = UsdGeom.Xform.Define(self.stage, group_path).GetPrim()
            group.CreateAttribute("em:layer", _T.Token).Set(layer)
        path = group_path.AppendChild(Tf.MakeValidIdentifier(name))
        self._remove(path)
        return path

    def layers(self) -> dict:
        """{layer name: {"path": group prim path, "visible": bool, "prims": [child names]}} of
        the layers present, in the order of ``LAYERS``."""
        found = {}
        for prim in Usd.PrimRange(self.prim):
            tag = prim.GetAttribute("em:layer")
            if tag and tag.Get():
                found[tag.Get()] = {
                    "path": str(prim.GetPath()),
                    "visible": UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible,
                    "prims": [child.GetName() for child in prim.GetChildren()],
                }
        return {layer: found[layer] for layer in LAYERS if layer in found}

    def find(self, what: str) -> Usd.Prim:
        """The prim meant by ``what``: a layer name ("skeleton"), "<layer>/<prim>"
        ("gaussian/ellipsoids_10"), a path relative to the dataset, or an absolute path."""
        present = self.layers()
        head, _, rest = str(what).partition("/")
        candidates = [str(what)] if str(what).startswith("/") else [f"{self.prim.GetPath()}/{what}"]
        if head in present:
            candidates.insert(0, present[head]["path"] + ("/" + rest if rest else ""))
        for path in candidates:
            prim = self.stage.GetPrimAtPath(path)
            if prim:
                return prim
        raise KeyError(f"no layer or prim {what!r} in {self.prim.GetPath()}; layers present: {', '.join(present) or 'none'}")

    def set_visible(self, what: str, visible: bool = True) -> None:
        """Show or hide a layer or one prim (see ``find``) by its USD ``visibility``."""
        token = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
        UsdGeom.Imageable(self.find(what)).CreateVisibilityAttr().Set(token)

    def save(self, path=None) -> str:
        """Save the stage (see ``save_stage``)."""
        return save_stage(self.stage, path)

    def describe(self) -> str:
        """The prim tree of the dataset with layer tags and the voxel scale in each space."""
        scales = "   ".join(f"{space} ({', '.join(f'{v:g}' for v in self.voxel_scale(space))}) nm" for space in SPACES)
        lines = [
            f"{self.prim.GetPath()}   {self.grid.describe()}",
            f"  space = {self.space!r}   (variants: {', '.join(SPACES)})",
            f"  voxels.xformOp:scale   {scales}   <- the anisotropy",
            f"  1 stage unit = {NM_PER_UNIT:g} nm; metersPerUnit = {UsdGeom.GetStageMetersPerUnit(self.stage):g} (magnified model)",
        ]
        frames = {"voxels": "frame: voxel index (i, j, k)", "physical": "frame: nanometres (x, y, z)"}
        for prim in Usd.PrimRange(self.prim):
            rel = prim.GetPath().MakeRelativePath(self.prim.GetPath())
            if prim == self.prim or rel.pathString.startswith("looks/") or prim.GetParent().IsA(UsdGeom.PointInstancer):
                continue
            depth = rel.pathElementCount
            tag = prim.GetAttribute("em:layer")
            kind = prim.GetAttribute("em:kind")
            note = [prim.GetTypeName()]
            if tag and tag.Get():
                note.append(f"[layer {tag.Get()}]")
            if kind and kind.Get():
                note.append(kind.Get())
            if depth == 1 and prim.GetName() in frames:
                note.append(frames[prim.GetName()])
            if UsdGeom.Imageable(prim) and UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible:
                note.append("(invisible)")
            lines.append(f"  {'  ' * depth}{prim.GetName():<{max(1, 26 - 2 * depth)}} {'  '.join(note)}")
        return "\n".join(lines)

    def author_camera(self, name: str = "overview", direction=(1.0, -1.0, 0.6), *, fov: float = 35.0,
                      margin: float = 1.05, projection: str = "perspective", aspect: float = 4.0 / 3.0) -> UsdGeom.Camera:
        """A camera that frames the PHYSICAL box of the dataset: it looks at the centre of the
        box from ``direction`` (centre -> eye) so that the bounding sphere of the box, times
        ``margin``, just touches the shorter side of the picture.  The camera does not move when
        the space is switched, so index-space and physical-space pictures can be compared.
        The same rule as ``emray.Camera.fit``."""
        offset = np.array(self.prim.GetAttribute("xformOp:translate").Get(), dtype=np.float64)
        extent = self.grid.extent_xyz_nm / NM_PER_UNIT
        centre, radius = offset + 0.5 * extent, 0.5 * float(np.linalg.norm(extent)) * margin
        d = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
        narrow = min(1.0, float(aspect))                 # < 1: the picture is narrower than high
        if projection == "perspective":
            distance = radius / math.sin(math.atan(narrow * math.tan(math.radians(fov) / 2.0)))
        else:
            distance = 4.0 * radius
        return author_camera(self.stage, name, eye=centre + d * distance, target=centre, projection=projection,
                             fov=fov, ortho_height=2.0 * radius / narrow, aspect=aspect)

    # ---- materials ---------------------------------------------------------------------------------
    def _material(self, name: str, *, color=None, opacity: float = 1.0, unlit: bool = False,
                  texture: str = None) -> UsdShade.Material:
        """A UsdPreviewSurface material under <dataset>/looks.

        texture   relative path of a PNG: shown as is (texture -> emissiveColor, diffuse black)
        unlit     ``color`` -> emissiveColor (annotation: independent of the lights)
        else      ``color`` -> diffuseColor, matte
        ``opacity`` < 1 goes to the material because RTX ignores primvars:displayOpacity."""
        path = self.looks.GetPath().AppendChild(Tf.MakeValidIdentifier(name))
        material = UsdShade.Material.Define(self.stage, path)
        surface = UsdShade.Shader.Define(self.stage, path.AppendChild("surface"))
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", _T.Float).Set(1.0)
        surface.CreateInput("metallic", _T.Float).Set(0.0)
        surface.CreateInput("useSpecularWorkflow", _T.Int).Set(1)
        surface.CreateInput("specularColor", _T.Color3f).Set(Gf.Vec3f(0.0))
        black, linear = Gf.Vec3f(0.0), (Gf.Vec3f(*srgb_to_linear(color)) if color is not None else None)
        if texture is not None:
            reader = UsdShade.Shader.Define(self.stage, path.AppendChild("st_reader"))
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", _T.String).Set("st")
            reader.CreateOutput("result", _T.Float2)
            image = UsdShade.Shader.Define(self.stage, path.AppendChild("texture"))
            image.CreateIdAttr("UsdUVTexture")
            image.CreateInput("file", _T.Asset).Set(Sdf.AssetPath(texture))
            image.CreateInput("sourceColorSpace", _T.Token).Set("sRGB")      # the PNG holds display values
            image.CreateInput("wrapS", _T.Token).Set("clamp")
            image.CreateInput("wrapT", _T.Token).Set("clamp")
            image.CreateInput("st", _T.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            image.CreateOutput("rgb", _T.Float3)
            surface.CreateInput("diffuseColor", _T.Color3f).Set(black)
            surface.CreateInput("emissiveColor", _T.Color3f).ConnectToSource(image.ConnectableAPI(), "rgb")
        elif unlit:
            surface.CreateInput("diffuseColor", _T.Color3f).Set(black)
            surface.CreateInput("emissiveColor", _T.Color3f).Set(linear)
        else:
            surface.CreateInput("diffuseColor", _T.Color3f).Set(linear)
        if opacity < 1.0:
            surface.CreateInput("opacity", _T.Float).Set(float(opacity))
            surface.CreateInput("opacityThreshold", _T.Float).Set(0.0)       # 0 = fractional opacity
        surface.CreateOutput("surface", _T.Token)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        return material

    def _paint(self, gprim, color, opacity: float = 1.0, unlit: bool = False) -> None:
        """Give a gprim its colour twice: as primvars:displayColor (read by the Warp adapter and
        by viewers without materials) and as a bound material (what RTX shades with)."""
        color = _rgb(color)
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*srgb_to_linear(color))])
        if opacity < 1.0:
            gprim.CreateDisplayOpacityAttr([float(opacity)])
        hexcode = "".join(f"{int(v):02x}" for v in _to_rgb8(color))
        name = f"c_{hexcode}_a{int(round(opacity * 100)):03d}" + ("_unlit" if unlit else "")
        material = self._material(name, color=color, opacity=opacity, unlit=unlit)
        UsdShade.MaterialBindingAPI.Apply(gprim.GetPrim()).Bind(material)
        if unlit:        # annotation must not darken the data
            UsdGeom.PrimvarsAPI(gprim).CreatePrimvar("doNotCastShadows", _T.Bool).Set(True)

    def _texture_dir(self) -> str:
        real_path = self.stage.GetRootLayer().realPath
        if not real_path:
            raise ValueError("slice textures are written next to the stage file, but this stage has no file: "
                             "create it with new_stage(path), or call add_slices(..., textures=False)")
        return os.path.join(os.path.dirname(real_path), "textures")

    # ---- rawvalue, voxvalue, instance: slices ------------------------------------------------------
    def add_slices(self, kind: str = "rawvalue", z=None, y=None, x=None, *, volume=None, upsample: int = 4,
                   textures: bool = True) -> list:
        """Slices of the volume as quads in the VOXELS frame (index units), each through the
        centre of the chosen voxel layer: ``z=50`` is the section [k = 50], drawn at height
        k + 0.5; ``y`` / ``x`` are reslices [:, j, :] / [:, :, i].  Each of z, y, x is an index
        or a list of indices.  Returns the prims.

        kind "rawvalue"  dataset /raw, identity grey: the uint8 number exactly as stored
             "voxvalue"  dataset /raw through the normalisation window and the viridis map
             "instance"  dataset /instance, colour by id hash, id 0 dark

        Every quad says in custom attributes (em:slice:*) which dataset, axis, index, colour
        map and window it shows; the Warp adapter samples the HDF5 data with exactly these.
        For RTX each quad also gets a PNG in ``textures/`` next to the stage file (referenced by
        relative path, nearest-neighbour enlarged ``upsample`` times so that voxels stay crisp
        under texture filtering) on the emissiveColor of a UsdPreviewSurface material.

        The quad spans [0, n] index units and needs no voxel size: the voxels frame supplies it.
        Texture rows: USD puts st (0, 0) at the lower-left of the image and PNG row 0 is the top
        row, so the corner at array row 0 gets t = 1.  Slices of two kinds at the same index
        lie in the same plane: show one of them at a time (``set_visible``)."""
        if kind not in SLICE_KINDS:
            raise ValueError(f"kind must be one of {tuple(SLICE_KINDS)}, got {kind!r}")
        dataset, colormap = SLICE_KINDS[kind]
        volume = self._slice_volume(kind, dataset, volume)
        if int(upsample) < 1:
            raise ValueError("upsample must be a positive integer")
        nx, ny, nz = (int(n) for n in self.grid.shape_xyz)
        wanted = []                                             # every index is checked before anything is authored
        for axis, indices, n_axis in (("z", z, nz), ("y", y, ny), ("x", x, nx)):
            for index in np.atleast_1d(indices if indices is not None else []).tolist():
                if index != int(index) or not 0 <= index < n_axis:
                    raise IndexError(f"{axis} = {index} is outside 0 .. {n_axis - 1}")
                wanted.append((axis, int(index)))
        if not wanted:
            raise ValueError("add_slices: give at least one of z=, y=, x=")
        texture_dir = self._texture_dir() if textures else None
        window = self.norm_window if kind == "voxvalue" else (0.0, 255.0)

        prims = []
        for axis, index in wanted:
            c = index + 0.5                                     # the centre of voxel layer `index`
            if axis == "z":      # image = volume[k]        rows = y, columns = x
                image, corners = volume[index], [(0, 0, c), (nx, 0, c), (nx, ny, c), (0, ny, c)]
            elif axis == "y":    # image = volume[:, j, :]  rows = z, columns = x
                image, corners = volume[:, index, :], [(0, c, 0), (nx, c, 0), (nx, c, nz), (0, c, nz)]
            else:                # image = volume[:, :, i]  rows = z, columns = y
                image, corners = volume[:, :, index], [(c, 0, 0), (c, ny, 0), (c, ny, nz), (c, 0, nz)]
            prim = self._add_textured_quad(self._new_prim_path(kind, self.voxels, f"slice_{axis}{index:04d}"), corners)
            for key, type_name, value in (
                ("em:kind", _T.Token, "slice"),
                ("em:slice:kind", _T.Token, kind),
                ("em:slice:dataset", _T.Token, dataset),             # HDF5 dataset name
                ("em:slice:axis", _T.Token, axis),                   # array axis the slice is normal to
                ("em:slice:index", _T.Int, index),
                ("em:slice:colorMap", _T.Token, colormap),
                ("em:slice:window", _T.Double2, Gf.Vec2d(*window)),  # stored numbers mapped to 0 and 1
            ):
                prim.CreateAttribute(key, type_name).Set(value)
            if textures:
                rgb = label_rgb8(image) if kind == "instance" else slice_lut8(colormap, window)[image]
                file_name = f"{self.name}_{kind}_{axis}{index:04d}.png"
                os.makedirs(texture_dir, exist_ok=True)
                write_png(os.path.join(texture_dir, file_name), rgb.repeat(int(upsample), axis=0).repeat(int(upsample), axis=1))
                material = self._material(f"{kind}_{axis}{index:04d}", texture=f"./textures/{file_name}")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            prims.append(prim)
        return prims

    def _slice_volume(self, kind: str, dataset: str, volume) -> np.ndarray:
        """The (z, y, x) array a slice of ``kind`` shows: ``volume`` or the dataset's own, checked."""
        volume = self.volumes[dataset] if volume is None else volume
        if volume is None:
            raise ValueError(f"no /{dataset} array for {kind!r} slices: this dataset has none, pass volume=")
        volume = np.asarray(volume)
        if tuple(volume.shape) != self.grid.shape_zyx:
            raise ValueError(f"volume has shape {volume.shape} but the grid says {self.grid.shape_zyx} (z, y, x)")
        if dataset == "raw" and volume.dtype != np.uint8:
            raise ValueError(f"{kind} slices show the stored uint8 numbers, got dtype {volume.dtype}")
        if dataset == "instance" and not np.issubdtype(volume.dtype, np.integer):
            raise ValueError(f"instance slices need integer labels, got dtype {volume.dtype}")
        return volume

    def _add_textured_quad(self, path: Sdf.Path, corners) -> Usd.Prim:
        """A quad Mesh whose four points are ``corners``, in order around it.  Texture
        coordinates: point 0 is the image's top-left corner (s, t) = (0, 1), point 1 its
        top-right, point 2 its bottom-right, point 3 its bottom-left."""
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(_vec3f(corners))
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)       # no normals authored
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr(_extent(corners))
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", _T.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 1), Gf.Vec2f(1, 1), Gf.Vec2f(1, 0), Gf.Vec2f(0, 0)]))
        return mesh.GetPrim()

    # ---- instance: bricks and surface ------------------------------------------------------------
    def _instance_color(self, color, instance_id):
        return _rgb(label_rgb(np.array([int(instance_id)]))[0] if color is None else color)

    def add_bricks(self, bricks, color=None, name: str = None) -> Usd.Prim:
        """instance, voxel by voxel: a PointInstancer of unit cubes in the VOXELS frame, one
        per voxel of ``bricks``.  No orientations are authored, so under the voxel scale every
        instance is an exact box of the voxel size.

        Positions are ``bricks.centers_xyz_nm`` divided by the voxel size of THIS dataset, i.e.
        (i + 0.5, j + 0.5, k + 0.5) for bricks made with the same grid, and the cube edge is
        ``bricks.size_xyz_nm`` divided by it, i.e. (1, 1, 1).  Bricks made with a wrong grid
        (``grid.as_cubes()``) are drawn where their wrong coordinates put them."""
        centers = _checked("bricks.centers_xyz_nm", bricks.centers_xyz_nm, (3,))
        d = self.grid.voxel_size_xyz_nm
        size = np.asarray(bricks.size_xyz_nm, dtype=np.float64).reshape(3) / d            # index units
        if not (np.isfinite(size).all() and size.min() > 0):
            raise ValueError(f"bricks.size_xyz_nm must be three positive numbers, got {bricks.size_xyz_nm}")
        centers = centers / d                                                             # index units
        color = self._instance_color(color, bricks.instance_id)
        name = name or f"bricks_{int(bricks.instance_id)}"
        instancer = UsdGeom.PointInstancer.Define(self.stage, self._new_prim_path("instance", self.voxels, name))
        cube = UsdGeom.Cube.Define(self.stage, instancer.GetPath().AppendChild("proto_cube"))
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
        self._paint(cube, color)
        instancer.CreatePrototypesRel().SetTargets([cube.GetPath()])
        instancer.CreateProtoIndicesAttr(Vt.IntArray(len(centers), 0))
        instancer.CreatePositionsAttr(_vec3f(centers))
        instancer.CreateScalesAttr(_vec3f(np.broadcast_to(size, centers.shape)))
        instancer.CreateExtentAttr(_extent(centers, pad=0.5 * size))
        prim = instancer.GetPrim()
        prim.CreateAttribute("em:kind", _T.Token).Set("bricks")
        prim.CreateAttribute("em:instanceId", _T.Int).Set(int(bricks.instance_id))
        prim.CreateAttribute("em:voxelsTotal", _T.Int64).Set(int(bricks.n_voxels_total))
        return prim

    def _mesh_arrays(self, mesh):
        vertices = _checked("mesh.vertices_xyz_nm", mesh.vertices_xyz_nm, (3,))
        faces = _checked("mesh.faces", mesh.faces, (3,), kind="i")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("mesh.faces refers to vertices that do not exist")
        return vertices, faces

    def add_surface(self, mesh, color=None, opacity: float = 1.0, name: str = None) -> Usd.Prim:
        """instance as a solid surface: a triangle Mesh in the VOXELS frame.  Its points are
        ``mesh.vertices_xyz_nm`` divided by the voxel size (index units); the voxel scale puts
        them back.  No normals are authored: under a non-uniform scale an authored normal would
        have to be transformed by the inverse transpose, and a renderer that shades from the
        transformed triangles cannot get that wrong.  ``opacity`` < 1 makes it see-through."""
        vertices, faces = self._mesh_arrays(mesh)
        if not 0.0 < opacity <= 1.0:
            raise ValueError(f"opacity must be in (0, 1], got {opacity}")
        color = self._instance_color(color, mesh.instance_id)
        points = vertices / self.grid.voxel_size_xyz_nm
        name = name or f"surface_{int(mesh.instance_id)}"
        gprim = UsdGeom.Mesh.Define(self.stage, self._new_prim_path("instance", self.voxels, name))
        gprim.CreatePointsAttr(_vec3f(points))
        gprim.CreateFaceVertexCountsAttr(Vt.IntArray(len(faces), 3))
        gprim.CreateFaceVertexIndicesAttr(_ints(faces))
        gprim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)      # the fallback, catmullClark, would smooth it
        gprim.CreateDoubleSidedAttr(True)
        gprim.CreateExtentAttr(_extent(points))
        self._paint(gprim, color, opacity)
        prim = gprim.GetPrim()
        prim.CreateAttribute("em:kind", _T.Token).Set("surface")
        prim.CreateAttribute("em:instanceId", _T.Int).Set(int(mesh.instance_id))
        return prim

    # ---- wiremesh and skeleton: tubes in the physical frame --------------------------------------
    def _add_tubes(self, layer, name, vertices_nm, edges, radii_nm, color, kind) -> UsdGeom.BasisCurves:
        """Linear BasisCurves in the PHYSICAL frame: one 2-point curve per edge, widths per point.

        physical space   points = the vertices in nm, width = 2 radius
        index space      points = A p, width = 2 radius * ``grid.tube_width_factor`` of the edge
        Widths are diameters in nm of the local frame; no normals, so renderers draw tubes."""
        order = edges.reshape(-1)                                        # point k of the prim = vertex order[k]
        radii = np.broadcast_to(np.asarray(radii_nm, dtype=np.float64), (len(vertices_nm),))

        def points(space):
            return self.grid.to_space(vertices_nm, space)[order]

        def widths(space):
            factor = self.grid.tube_width_factor(vertices_nm[edges[:, 0]], vertices_nm[edges[:, 1]], space)
            return 2.0 * radii[order] * np.repeat(factor, 2)

        curves = UsdGeom.BasisCurves.Define(self.stage, self._new_prim_path(layer, self.physical, name))
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)                     # the fallback is cubic bezier
        curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curves.CreateCurveVertexCountsAttr(Vt.IntArray(len(edges), 2))
        self._author(curves.CreatePointsAttr(), lambda space: _vec3f(points(space)))
        self._author(curves.CreateWidthsAttr(), lambda space: _floats(widths(space)))
        curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)             # one width per point
        self._author(curves.CreateExtentAttr(), lambda space: _extent(points(space), pad=0.5 * widths(space).max()))
        self._paint(curves, color)
        curves.GetPrim().CreateAttribute("em:kind", _T.Token).Set(kind)
        return curves

    def add_wiremesh(self, mesh, width_nm: float = None, color=None, name: str = None) -> Usd.Prim:
        """wiremesh: the unique edges of the triangles as tubes ``width_nm`` wide, in the
        PHYSICAL frame (a tube under the non-uniform voxel scale would have no defined shape).
        Default width: 0.1 % of the diagonal of the dataset box.  Default colour: the instance
        colour, darkened."""
        vertices, _ = self._mesh_arrays(mesh)
        edges = np.asarray(mesh.edges(), dtype=np.int64)
        if width_nm is None:
            width_nm = 0.001 * float(np.linalg.norm(self.grid.extent_xyz_nm))
        if not width_nm > 0:
            raise ValueError(f"width_nm must be positive, got {width_nm}")
        color = _rgb(0.55 * self._instance_color(None, mesh.instance_id) if color is None else color)
        curves = self._add_tubes("wiremesh", name or f"wire_{int(mesh.instance_id)}", vertices, edges,
                                 0.5 * float(width_nm), color, "wire")
        prim = curves.GetPrim()
        prim.CreateAttribute("em:widthNm", _T.Double).Set(float(width_nm))
        prim.CreateAttribute("em:instanceId", _T.Int).Set(int(mesh.instance_id))
        return prim

    def add_skeleton(self, skeleton, color=(0.12, 0.12, 0.12), branch_color=(0.90, 0.10, 0.10),
                     end_color=(0.10, 0.35, 0.95), marker_scale: float = 1.25, name: str = None) -> tuple:
        """skeleton: tubes and node markers in the PHYSICAL frame.  Returns (tubes, nodes).

        tubes   linear BasisCurves, one 2-point curve per edge; the width at a node is
                2 x its radius (``skeleton.radii_nm``: the distance to the centre of the nearest
                voxel outside the instance, about half a voxel more than to the surface)
        nodes   a PointInstancer of unit spheres, one per node, scaled to the node radius;
                branch points (more than two edges) and end points (one edge) are
                ``marker_scale`` times larger and have their own colours

        In index space the node positions become A p, the node spheres become the ellipsoids
        A maps them to (scale = radius * A), and the tube widths follow ``grid.tube_width_factor``
        (round tubes of the same volume as the exact image; ``emray.Scene.add_skeleton`` draws
        the same)."""
        vertices = _checked("skeleton.vertices_xyz_nm", skeleton.vertices_xyz_nm, (3,))
        edges = _checked("skeleton.edges", skeleton.edges, (2,), kind="i")
        radii = np.asarray(skeleton.radii_nm, dtype=np.float64)
        if radii.shape != (len(vertices),) or not np.isfinite(radii).all() or radii.min() <= 0:
            raise ValueError("skeleton.radii_nm must be one positive finite radius per node")
        if edges.min() < 0 or edges.max() >= len(vertices):
            raise ValueError("skeleton.edges refers to nodes that do not exist")
        color, branch_color, end_color = _rgb(color), _rgb(branch_color), _rgb(end_color)
        if not marker_scale > 0:
            raise ValueError("marker_scale must be positive")
        tag = name or str(int(skeleton.instance_id))
        tubes = self._add_tubes("skeleton", f"tubes_{tag}", vertices, edges, radii, color, "tubes").GetPrim()
        tubes.CreateAttribute("em:instanceId", _T.Int).Set(int(skeleton.instance_id))

        degree = np.bincount(edges.ravel(), minlength=len(vertices))
        proto_index = np.where(degree > 2, 1, np.where(degree == 1, 2, 0))        # 0 ordinary, 1 branch, 2 end
        marker_radii = radii * np.where(proto_index > 0, float(marker_scale), 1.0)
        nodes = UsdGeom.PointInstancer.Define(self.stage, self._new_prim_path("skeleton", self.physical, f"nodes_{tag}"))
        targets = []
        for proto_name, proto_color in (("node", color), ("branch_point", branch_color), ("end_point", end_color)):
            sphere = UsdGeom.Sphere.Define(self.stage, nodes.GetPath().AppendChild(f"proto_{proto_name}"))
            sphere.CreateRadiusAttr(1.0)
            sphere.CreateExtentAttr([(-1, -1, -1), (1, 1, 1)])
            self._paint(sphere, proto_color)
            targets.append(sphere.GetPath())
        nodes.CreatePrototypesRel().SetTargets(targets)
        nodes.CreateProtoIndicesAttr(_ints(proto_index))

        def scales(space):          # a sphere of radius r under A is the ellipsoid with semi-axes r * A
            return marker_radii[:, None] * self.grid.space_scale(space)[None, :]

        self._author(nodes.CreatePositionsAttr(), lambda space: _vec3f(self.grid.to_space(vertices, space)))
        self._author(nodes.CreateScalesAttr(), lambda space: _vec3f(scales(space)))
        self._author(nodes.CreateExtentAttr(), lambda space: _vec3f(np.stack([
            (self.grid.to_space(vertices, space) - scales(space)).min(axis=0),
            (self.grid.to_space(vertices, space) + scales(space)).max(axis=0)])))
        nodes.GetPrim().CreateAttribute("em:kind", _T.Token).Set("nodes")
        nodes.GetPrim().CreateAttribute("em:instanceId", _T.Int).Set(int(skeleton.instance_id))
        return tubes, nodes.GetPrim()

    # ---- gaussian ----------------------------------------------------------------------------------
    def add_gaussians(self, gaussians, k_sigma: float = K_SIGMA, opacity=0.8, color=None, splats: bool = None,
                      show: str = "auto", name: str = None) -> tuple:
        """gaussian: a mixture of 3D Gaussians in the PHYSICAL frame, authored twice.
        Returns (gaussian prim, ellipsoid prim).

        <layer>/splats_<id>      "the Gaussian prim".  If this USD has the schema it is a
            ParticleField3DGaussianSplat: positions = means, scales = the three standard
            deviations (linear, in nm of the local frame), orientations = the principal axes
            (covariance = R diag(scales^2) R^T), opacities linear in [0, 1], colour as the
            degree-0 spherical harmonic c0 = (rgb - 0.5) / SH_C0.  Without the schema it is a
            plain Scope.  Either way it carries em:gaussian:means, em:gaussian:covariances
            (full 3x3, nm^2), em:gaussian:opacities and em:gaussian:displayRgb, from which the
            Warp adapter draws true Gaussians without decomposing anything again.
        <layer>/ellipsoids_<id>  the fallback every renderer can draw: a PointInstancer of unit
            spheres scaled to the ``k_sigma`` ellipsoids (default sqrt(5): the solid ellipsoid
            with the same covariance).  Pure translate-rotate-scale instances.

        A Gaussian cannot inherit the anisotropy from a parent scale (position, rotation and
        three scales cannot express a sheared matrix), so everything is authored per space:
        mean A mu, covariance A S A^T, decomposed again.

        opacity  one value or one per Gaussian;  color  one colour or one per Gaussian
                 (default ``gaussians.colors_rgb``, else the instance colour)
        splats   None = use the schema if this USD has it, False = take the fallback path
        show     which of the two prims is visible: "splats", "ellipsoids", "both", or "auto"
                 (splats if the schema prim was authored, else ellipsoids)"""
        means, covs, opacities, colors = self._gaussian_arrays(gaussians, opacity, color)
        if show not in ("auto", "splats", "ellipsoids", "both"):
            raise ValueError(f"show must be 'auto', 'splats', 'ellipsoids' or 'both', got {show!r}")
        if not k_sigma > 0:
            raise ValueError("k_sigma must be positive")
        use_schema = splat_schema_available() if splats is None else bool(splats)
        if use_schema and not splat_schema_available():
            raise RuntimeError(f"this USD build has no {SPLAT_TYPE} schema: call add_gaussians(..., splats=False)")
        tag = name or str(int(gaussians.instance_id))

        cache = {}

        def in_space(space):
            """(means, covariances, quaternions (w, x, y, z), standard deviations, rotations) as
            drawn in ``space``, in nm: mean A mu and covariance A S A^T, decomposed into
            principal axes (covariance = R diag(sigma^2) R^T, axes in the columns of R)."""
            if space not in cache:
                mu = self.grid.to_space(means, space)
                cov = self.grid.covariance_to_space(covs, space)
                rotations, sigmas = Gaussians(mu, cov, gaussians.weights).principal_axes()
                cache[space] = (mu, cov, rotation_to_quaternion_wxyz(rotations), sigmas, rotations)
            return cache[space]

        prim = self.stage.DefinePrim(self._new_prim_path("gaussian", self.physical, f"splats_{tag}"),
                                     SPLAT_TYPE if use_schema else "Scope")
        prim.CreateAttribute("em:kind", _T.Token).Set("gaussians")
        prim.CreateAttribute("em:gaussian:opacities", _T.FloatArray).Set(_floats(opacities))
        prim.CreateAttribute("em:gaussian:displayRgb", _T.Float3Array).Set(_vec3f(colors))
        prim.CreateAttribute("em:gaussian:weights", _T.FloatArray).Set(_floats(gaussians.weights))
        self._author(prim.CreateAttribute("em:gaussian:means", _T.Point3dArray),
                     lambda space: Vt.Vec3dArray.FromNumpy(np.ascontiguousarray(in_space(space)[0])))
        self._author(prim.CreateAttribute("em:gaussian:covariances", _T.Matrix3dArray),
                     lambda space: Vt.Matrix3dArray.FromNumpy(np.ascontiguousarray(in_space(space)[1])))
        if use_schema:
            self._author_splat_attributes(prim, in_space, opacities, colors)
        ellipsoids = self._add_ellipsoids(f"ellipsoids_{tag}", in_space, colors, k_sigma)
        for p in (prim, ellipsoids):
            p.CreateAttribute("em:instanceId", _T.Int).Set(int(gaussians.instance_id))

        if show == "auto":
            show = "splats" if use_schema else "ellipsoids"
        for p, shown in ((prim, show in ("splats", "both")), (ellipsoids, show in ("ellipsoids", "both"))):
            UsdGeom.Imageable(p).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited if shown else UsdGeom.Tokens.invisible)
        return prim, ellipsoids

    def _gaussian_arrays(self, gaussians, opacity, color) -> tuple:
        """(means (K, 3) nm, covariances (K, 3, 3) nm^2, opacities (K,), colours (K, 3)) of a
        ``types.Gaussians``, checked: finite, symmetric positive definite, values in [0, 1]."""
        means = _checked("gaussians.means_xyz_nm", gaussians.means_xyz_nm, (3,))
        covs = _checked("gaussians.covariances_nm2", gaussians.covariances_nm2, (3, 3))
        if len(covs) != len(means):
            raise ValueError(f"{len(means)} means but {len(covs)} covariances")
        if not np.allclose(covs, np.swapaxes(covs, 1, 2), rtol=1e-6, atol=1e-9 * np.abs(covs).max()):
            raise ValueError("gaussians.covariances_nm2 must be symmetric")
        if np.linalg.eigvalsh(covs).min() <= 0:
            raise ValueError("gaussians.covariances_nm2 must be positive definite")
        opacities = np.broadcast_to(np.asarray(opacity, dtype=np.float64), (len(means),))
        if not np.isfinite(opacities).all() or opacities.min() < 0 or opacities.max() > 1:
            raise ValueError("opacity must be in [0, 1]")
        if color is None:
            color = self._instance_color(None, gaussians.instance_id) if gaussians.colors_rgb is None else gaussians.colors_rgb
        colors = np.broadcast_to(np.asarray(color, dtype=np.float64), (len(means), 3))
        if not np.isfinite(colors).all() or colors.min() < 0 or colors.max() > 1:
            raise ValueError("gaussian colours must be in [0, 1]")
        return means, covs, opacities, colors

    def _author_splat_attributes(self, prim, in_space, opacities, colors) -> None:
        """The attributes of the ParticleField3DGaussianSplat schema, per space.  Written with
        the generic attribute API (custom=False: they belong to the schema), because inside
        Kit the schema is a codeless plugin without a Python class."""
        def schema_attribute(attr_name, type_name, variability=Sdf.VariabilityVarying):
            return prim.CreateAttribute(attr_name, type_name, False, variability)

        schema_attribute("opacities", _T.FloatArray).Set(_floats(opacities))
        # the schema default is degree 3 (16 coefficients per Gaussian): say 0, or the colours are ignored
        schema_attribute("radiance:sphericalHarmonicsDegree", _T.Int, Sdf.VariabilityUniform).Set(0)
        harmonics = schema_attribute("radiance:sphericalHarmonicsCoefficients", _T.Float3Array)
        harmonics.Set(_vec3f((colors - 0.5) / SH_C0))
        UsdGeom.Primvar(harmonics).SetElementSize(1)
        UsdGeom.Primvar(harmonics).SetInterpolation(UsdGeom.Tokens.vertex)       # RTX insists on vertex
        # for viewers that draw the prim as plain points; linear, like every displayColor of this module
        display = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("displayColor", _T.Color3fArray, UsdGeom.Tokens.vertex)
        display.Set(_vec3f(srgb_to_linear(colors)))
        self._author(schema_attribute("positions", _T.Point3fArray), lambda space: _vec3f(in_space(space)[0]))
        self._author(schema_attribute("orientations", _T.QuatfArray), lambda space: _quats(in_space(space)[2], half=False))
        self._author(schema_attribute("scales", _T.Float3Array), lambda space: _vec3f(in_space(space)[3]))
        self._author(UsdGeom.Boundable(prim).CreateExtentAttr(), lambda space: _gaussian_extent(*in_space(space)[:2], 3.0))

    def _add_ellipsoids(self, name, in_space, colors, k_sigma) -> Usd.Prim:
        """The k-sigma ellipsoids of Gaussians as a PointInstancer of unit spheres in the
        PHYSICAL frame: position = mean, orientation = principal axes, scale = k_sigma * the
        standard deviations, all per space.  One sphere prototype per colour, chosen by
        protoIndices (the way of colouring instances that every renderer supports)."""
        instancer = UsdGeom.PointInstancer.Define(self.stage, self._new_prim_path("gaussian", self.physical, name))
        unique_colors, proto_index = np.unique(np.round(colors, 4), axis=0, return_inverse=True)
        targets = []
        for c, proto_color in enumerate(unique_colors):
            sphere = UsdGeom.Sphere.Define(self.stage, instancer.GetPath().AppendChild(f"proto_{c:03d}"))
            sphere.CreateRadiusAttr(1.0)
            sphere.CreateExtentAttr([(-1, -1, -1), (1, 1, 1)])
            self._paint(sphere, proto_color)
            targets.append(sphere.GetPath())
        instancer.CreatePrototypesRel().SetTargets(targets)
        instancer.CreateProtoIndicesAttr(_ints(proto_index))
        self._author(instancer.CreatePositionsAttr(), lambda space: _vec3f(in_space(space)[0]))
        # 'orientations' (half floats) is the attribute every renderer reads; 'orientationsf' is newer
        self._author(instancer.CreateOrientationsAttr(), lambda space: _quats(in_space(space)[2], half=True))
        self._author(instancer.CreateScalesAttr(), lambda space: _vec3f(k_sigma * in_space(space)[3]))

        def extent(space):
            """The extent as UsdGeom computes it for a PointInstancer: the box around the boxes of
            the instances, each the box [-1, 1]^3 of the unit sphere under R diag(k sigma).  A
            little larger than the ellipsoids themselves, so no renderer culls an instance."""
            mu, _, _, sigmas, rotations = in_space(space)
            half = np.einsum("nab,nb->na", np.abs(rotations), k_sigma * sigmas)
            return _vec3f(np.stack([(mu - half).min(axis=0), (mu + half).max(axis=0)]))

        self._author(instancer.CreateExtentAttr(), extent)
        prim = instancer.GetPrim()
        prim.CreateAttribute("em:kind", _T.Token).Set("ellipsoids")
        prim.CreateAttribute("em:kSigma", _T.Double).Set(float(k_sigma))
        return prim

    # ---- frame: bounding box, axes, scale bar ------------------------------------------------------
    def _add_bar(self, path, anchor_nm, axis: int, length_nm: float, thickness_nm: float, color) -> UsdGeom.Mesh:
        """A box that stands for ``length_nm`` of TISSUE along data axis ``axis`` (0 = x),
        starting at ``anchor_nm`` (true nm).  Drawn in a space it is length_nm * A[axis] long;
        its square cross-section of ``thickness_nm`` is the same in every space."""
        anchor = np.asarray(anchor_nm, dtype=np.float64)

        def points(space):
            lo = self.grid.to_space(anchor, space) - 0.5 * thickness_nm
            hi = lo + thickness_nm
            lo[axis] = self.grid.to_space(anchor, space)[axis]
            hi[axis] = lo[axis] + length_nm * self.grid.space_scale(space)[axis]
            return _box_points(lo, hi)

        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr(_BOX_FACES)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self._author(mesh.CreatePointsAttr(), lambda space: _vec3f(points(space)))
        self._author(mesh.CreateExtentAttr(), lambda space: _extent(points(space)))
        self._paint(mesh, color, unlit=True)
        prim = mesh.GetPrim()
        prim.CreateAttribute("em:lengthNm", _T.Double).Set(float(length_nm))
        prim.CreateAttribute("em:axis", _T.Token).Set("xyz"[axis])
        prim.CreateAttribute("em:label", _T.String).Set(format_length_nm(length_nm))
        return mesh

    def add_frame(self, bar_nm: float = None, color=(0.35, 0.35, 0.35)) -> dict:
        """frame: what makes a picture readable.  Returns {"bbox", "gnomon", "scalebar"}.

        bbox      the 12 edges of the array box as tubes (width 0.3 % of the box diagonal)
        gnomon    three arms from the corner (0, 0, 0): x red, y green, z blue, each standing
                  for ``bar_nm`` of tissue.  Equal in physical space; in index space the z arm
                  is anisotropy times shorter: the most direct picture of the mistake
        scalebar  a dark bar along x of the same round length, in front of the box; its length
                  and label are in em:lengthNm and em:label (USD has no text prim)

        ``bar_nm`` defaults to the largest 1-2-5 length within a quarter of the x extent.
        Everything is in the PHYSICAL frame with per-space points; line widths and bar
        thicknesses are annotation and stay the same in both spaces."""
        extent = self.grid.extent_xyz_nm
        diagonal = float(np.linalg.norm(extent))
        bar_nm = nice_length_nm(extent[0]) if bar_nm is None else float(bar_nm)
        if not bar_nm > 0:
            raise ValueError("bar_nm must be positive")
        color = _rgb(color)
        corners = _box_points(np.zeros(3), extent)

        def box_points(space):
            return self.grid.to_space(corners, space)[_BOX_EDGES]

        width = 0.003 * diagonal
        bbox = UsdGeom.BasisCurves.Define(self.stage, self._new_prim_path("frame", self.physical, "bbox"))
        bbox.CreateTypeAttr(UsdGeom.Tokens.linear)
        bbox.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        bbox.CreateCurveVertexCountsAttr(Vt.IntArray(12, 2))
        bbox.CreateWidthsAttr([width])
        bbox.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        self._author(bbox.CreatePointsAttr(), lambda space: _vec3f(box_points(space)))
        self._author(bbox.CreateExtentAttr(), lambda space: _extent(box_points(space), pad=0.5 * width))
        self._paint(bbox, color, unlit=True)
        bbox.GetPrim().CreateAttribute("em:kind", _T.Token).Set("bbox")

        thickness = bar_nm / 12.0
        gnomon_path = self._new_prim_path("frame", self.physical, "gnomon")
        gnomon = UsdGeom.Xform.Define(self.stage, gnomon_path).GetPrim()
        gnomon.CreateAttribute("em:kind", _T.Token).Set("gnomon")
        for axis, arm_color in enumerate(((0.90, 0.15, 0.15), (0.10, 0.65, 0.20), (0.15, 0.35, 0.95))):
            self._add_bar(gnomon_path.AppendChild("xyz"[axis]), np.zeros(3), axis, bar_nm, thickness, arm_color)

        anchor = np.array([extent[0] - bar_nm, -3.0 * thickness, 0.0])
        bar = self._add_bar(self._new_prim_path("frame", self.physical, "scalebar"), anchor, 0, bar_nm, thickness,
                            (0.10, 0.10, 0.10))
        bar.GetPrim().CreateAttribute("em:kind", _T.Token).Set("scalebar")
        return {"bbox": bbox.GetPrim(), "gnomon": gnomon, "scalebar": bar.GetPrim()}


def find_datasets(stage: Usd.Stage) -> list:
    """The datasets on a stage (prims with em:voxelSizeNm), as Dataset objects."""
    return [Dataset.from_prim(prim) for prim in stage.Traverse() if prim.HasAttribute("em:voxelSizeNm")]
