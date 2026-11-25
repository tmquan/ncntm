"""emray -- a small ray caster (NVIDIA Warp, CUDA or CPU) for teaching EM-connectomics representations.

It draws, exactly and with voxel anisotropy handled in the open, the six representations
of an electron-microscopy volume:

    rawvalue  the uint8 number as stored, in array index space      -> volume-sampled slices
    voxvalue  the number normalised to [0,1] on a physical voxel     -> slices + lookup table
    instance  voxel label = instance id                              -> label slices, voxel bricks
    skeleton  nodes + edges                                          -> tubes + spheres
    wiremesh  wireframe surface mesh                                 -> wire material / tubes
    gaussian  3D Gaussians                                           -> true Gaussians or k-sigma ellipsoids

Dependencies: ``warp`` and ``numpy`` only.  No Kit app, no RTX, no graphics context.

Conventions (the whole module follows them)
-------------------------------------------
* Volumes are numpy arrays indexed ``[k, j, i]`` = ``[z, y, x]``.  Voxel sizes are given in the
  same order, ``(dz, dy, dx)``.  Points, vectors and matrices are in world order ``(x, y, z)``.
* The renderer has no unit of its own: it draws the numbers it is given ("scene units").  The
  package draws 1 scene unit = 1 micrometre (``grid.NM_PER_UNIT`` nanometres).
* Where is voxel ``[k, j, i]``?  Two conventions are supported and always named:

    "corner"  the voxel is the box  x in [i*dx, (i+1)*dx],  y in [j*dy, (j+1)*dy],  z in [k*dz, (k+1)*dz]
              (plus ``origin``); its centre is at +0.5 voxel.  This is the convention of ``grid.py``
              and of every entry point that takes a ``VoxelGrid``: :meth:`Scene.add_volume_grid`,
              :meth:`Scene.add_bricks`, :meth:`Camera.for_slice`.
    "center"  the voxel is centred on ``origin + (i*dx, j*dy, k*dz)``; the volume spans -0.5 .. n-0.5
              voxels.  This is the default of the array-based entry points :meth:`Scene.add_volume`
              and :meth:`Scene.add_voxels` (what kimimaro, zmesh and skimage return).

  The two differ by half a voxel: (dx/2, dy/2, dz/2), i.e. 3 nm in x and 15 nm in z for a
  6 x 6 x 30 nm voxel.  A slice with index k is always drawn through the CENTRE of voxel layer k.
* Anisotropy never hides inside the renderer: it enters only through the voxel size of a volume
  and through the coordinates you hand to the tessellators.
* Image row 0 is the top of the picture; ``depth`` is the distance t along the unit ray
  (perspective: distance from the eye; orthographic: distance from the eye plane).
* Colours are display-referred (no gamma step), so an unlit material reproduces its input exactly.
* The Warp device is the ``device`` argument, else the environment variable ``NCNTM_DEVICE``
  (e.g. ``cuda:1`` or ``cpu``), else Warp's preferred device (see :func:`default_device`).

Typical use
-----------
    scene = Scene()
    s = scene.add_volume_grid(em, grid, space="physical", lut=lut_grey())
    scene.add_slice(s, "z", 50, object_id=1)
    scene.add_wire_box(*scene.volume_corners(s))
    cam = Camera.fit(*scene.bounds(), direction=(1, -1, 0.6))
    img = scene.render(cam)                           # img.rgba, img.depth, img.object_id
    img.save("picture.png")
"""
from __future__ import annotations

import functools
import math
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass

import numpy as np
import warp as wp

from .grid import NM_PER_UNIT

__all__ = ["Scene", "Camera", "Image", "default_device", "load_kernels", "index_to_world", "lut_grey",
           "lut_normalised", "lut_window_level", "lut_membrane", "label_colors", "icosphere", "unique_edges",
           "box_edges", "cov_to_axes", "save_png", "encode_png", "MAT_LIT", "MAT_UNLIT", "MAT_VOLUME", "MAT_WIRE",
           "AXIS_COLORS", "CONVENTIONS"]

MAT_LIT, MAT_UNLIT, MAT_VOLUME, MAT_WIRE = 0, 1, 2, 3   # per-face material ids
GAUSS_BUF = 64          # Gaussians kept sorted per ray per pass (more are handled by further passes)
CONVENTIONS = ("center", "corner")
AXIS_COLORS = ((0.9, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.3, 1.0))     # x red, y green, z blue
_FAR = 1.0e30
_vecf = wp.types.vector(length=GAUSS_BUF, dtype=wp.float32)
_veci = wp.types.vector(length=GAUSS_BUF, dtype=wp.int32)


# --------------------------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------------------------
def default_device():
    """Name of the Warp device used when no ``device`` is passed: the environment variable
    ``NCNTM_DEVICE`` if it is set (``"cuda:1"``, ``"cpu"``, ...), otherwise Warp's preferred device
    (the first CUDA device if there is one, else the CPU)."""
    name = os.environ.get("NCNTM_DEVICE", "").strip()
    return str(wp.get_device(name).alias) if name else str(wp.get_preferred_device().alias)


def _device_name(device):
    """Canonical name ("cuda:1", "cpu") of ``device``; ``None`` means :func:`default_device`."""
    return default_device() if device is None else str(wp.get_device(device).alias)


def load_kernels(device=None):
    """Compile (or load from Warp's cache) the two kernels of this module for ``device`` and return
    the seconds it took.  Call it once at the top of a notebook so that the one-off compilation
    is not mistaken for the time of the first picture."""
    t0 = time.perf_counter()
    wp.load_module(sys.modules[__name__], device=_device_name(device))
    return time.perf_counter() - t0


# --------------------------------------------------------------------------------------------
# numpy helpers: voxel grids, lookup tables, label colours, small geometry utilities
# --------------------------------------------------------------------------------------------
def _convention(name):
    if name not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {name!r}")
    return name


def _vec3(value, what):
    """``value`` as three finite floats, or a clear error."""
    v = np.asarray(value, np.float64)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{what} must be three finite numbers, got {value!r}")
    return v


def index_to_world(voxel_size, origin=(0.0, 0.0, 0.0)):
    """4x4 matrix taking an index vector ``(i, j, k, 1)`` to world ``(x, y, z, 1)``:
    ``x = origin_x + i*dx`` and so on.  Its diagonal IS the voxel size.

    ``voxel_size`` is ``(dz, dy, dx)`` (array order), ``origin`` is a world position (x, y, z).
    What an integer index means is the caller's convention: with "center", ``origin`` is the
    centre of voxel ``[0, 0, 0]`` and integers are voxel centres; with "corner", ``origin`` is the
    low corner of voxel ``[0, 0, 0]`` and integers are voxel corners (centres at +0.5).
    """
    size = _vec3(voxel_size, "voxel_size (dz, dy, dx)")
    if size.min() <= 0:
        raise ValueError(f"voxel size must be positive, got {voxel_size!r}")
    dz, dy, dx = size
    m = np.diag([dx, dy, dz, 1.0])
    m[:3, 3] = _vec3(origin, "origin (x, y, z)")
    return m


def lut_grey():
    """Identity lookup table for *rawvalue*: stored number v -> grey v/255, opaque."""
    g = np.arange(256, dtype=np.float32) / 255.0
    return np.stack([g, g, g, np.ones_like(g)], axis=1)


def lut_normalised(vmin, vmax):
    """Lookup table for *voxvalue*: stored number v -> grey ``(v - vmin) / (vmax - vmin)`` clamped
    to [0, 1], opaque.  ``vmin`` / ``vmax`` are the per-volume normalisation limits in stored
    units (0..255), e.g. the HDF5 attributes ``intensity_norm_min`` / ``intensity_norm_max``."""
    if not vmax > vmin:
        raise ValueError(f"need vmax > vmin, got vmin={vmin!r}, vmax={vmax!r}")
    g = np.clip((np.arange(256, dtype=np.float64) - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
    return np.stack([g, g, g, np.ones_like(g)], axis=1)


def lut_window_level(level=0.5, window=1.0, lo=(0, 0, 0), hi=(1, 1, 1)):
    """Lookup table for *voxvalue*: v/255 in [0,1] mapped linearly from ``level-window/2`` (colour
    ``lo``) to ``level+window/2`` (colour ``hi``) and clamped outside."""
    v = np.arange(256, dtype=np.float64) / 255.0
    t = np.clip((v - (level - 0.5 * window)) / max(window, 1e-9), 0.0, 1.0)[:, None]
    rgb = np.asarray(lo, float)[None] * (1 - t) + np.asarray(hi, float)[None] * t
    return np.concatenate([rgb, np.ones((256, 1))], axis=1).astype(np.float32)


def lut_membrane(dark=0.35, soft=0.12, opacity=0.9, color=(0.05, 0.05, 0.08), bright_color=(0.9, 0.9, 0.9)):
    """Transfer function for direct volume rendering of EM: values darker than ``dark`` (membranes,
    in [0,1]) are opaque, brighter cytoplasm fades out over ``soft``.  Alpha is the opacity
    accumulated over the reference length given to :meth:`Scene.set_dvr`."""
    v = np.arange(256, dtype=np.float64) / 255.0
    t = np.clip((dark + 0.5 * soft - v) / max(soft, 1e-9), 0.0, 1.0)
    a = opacity * t * t * (3 - 2 * t)
    rgb = np.asarray(color, float)[None] * (1 - v[:, None]) + np.asarray(bright_color, float)[None] * v[:, None]
    return np.concatenate([rgb, a[:, None]], axis=1).astype(np.float32)


def label_colors(ids):
    """Hash-to-colour for instance ids (numpy twin of the kernel function; same colours)."""
    h = np.asarray(ids).astype(np.uint64)                   # 64-bit with explicit masks = uint32 wrap-around
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


@functools.lru_cache(maxsize=None)
def icosphere(subdiv=2):
    """Unit icosphere: (vertices (V,3), faces (F,3)); F = 20 * 4**subdiv.  The arrays are computed
    once per ``subdiv`` and shared by all callers, so they are read-only: copy them to change them."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], dtype=np.float64)
    f = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                  (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
                  (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)], dtype=np.int64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdiv):
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        ue, inv = np.unique(edges, axis=0, return_inverse=True)
        mid = v[ue].mean(axis=1)
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        ab, bc, ca = len(v) + inv.reshape(3, -1)
        a, b, c = f.T
        v = np.concatenate([v, mid])
        f = np.concatenate([np.stack(x, 1) for x in ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))])
    v.setflags(write=False)
    f.setflags(write=False)
    return v, f


def unique_edges(faces):
    """Unique undirected edges (E,2) of a triangle list -- e.g. to draw a wireframe with tubes."""
    f = np.asarray(faces).reshape(-1, 3)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def box_edges(lo, hi):
    """(points (8,3), edges (12,2)) of the axis-aligned box lo..hi -- for outlines drawn as tubes."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    p = np.array([[(lo, hi)[(n >> a) & 1][a] for a in range(3)] for n in range(8)])
    e = [(n, n | (1 << a)) for n in range(8) for a in range(3) if not n & (1 << a)]
    return p, np.array(e)


def cov_to_axes(covs):
    """Covariances (N,3,3) -> (rotations R (N,3,3) with principal axes in columns, sigmas (N,3)),
    so that ``cov = R diag(sigma**2) R^T``."""
    w, r = np.linalg.eigh(np.asarray(covs, np.float64).reshape(-1, 3, 3))
    return r, np.sqrt(np.maximum(w, 1e-20))


def _checked_covariances(covs, n):
    """``covs`` as an (n, 3, 3) float64 array, or a clear error if they are not ``n`` finite,
    symmetric, positive definite matrices."""
    covs = np.asarray(covs, np.float64)
    if covs.size != 9 * n or not np.all(np.isfinite(covs)):
        raise ValueError(f"need one finite 3x3 covariance per Gaussian ({n}), got shape {covs.shape}")
    covs = covs.reshape(n, 3, 3)
    asymmetry = np.abs(covs - covs.transpose(0, 2, 1)).max(axis=(1, 2))
    if np.any(asymmetry > 1e-5 * np.abs(covs).max(axis=(1, 2))):            # (float32 round-off is tolerated)
        raise ValueError("covariance matrices must be symmetric")
    if np.linalg.eigvalsh(covs).min() <= 0:
        raise ValueError("covariance matrices must be positive definite (every variance > 0)")
    return covs


def encode_png(rgba):
    """An (H,W,3|4) uint8 array as PNG bytes (pure zlib, no imaging library needed)."""
    a = np.ascontiguousarray(rgba, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"expected an (H, W, 3|4) array, got shape {a.shape}")
    h, w, c = a.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), a.reshape(h, w * c)], axis=1).tobytes()

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if c == 4 else 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def save_png(path, rgba):
    """Write an (H,W,3|4) uint8 array as a PNG file."""
    with open(path, "wb") as fh:
        fh.write(encode_png(rgba))


_BOX_QUADS = np.array([[[(s if a == ax else (u if a == (ax + 1) % 3 else v)) for a in range(3)]
                        for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                       for ax in range(3) for s in (-1, 1)], dtype=np.float64)   # (6 faces, 4 corners, 3) signs


def _world_axis(axis):
    """World axis 0, 1, 2 (= x, y, z) of a slice normal given as "x" / "y" / "z" or as the ARRAY
    axis 2 / 1 / 0."""
    if isinstance(axis, str):
        if axis not in ("x", "y", "z"):
            raise ValueError(f'axis must be "x", "y" or "z" (or array axis 0, 1, 2), got {axis!r}')
        return "xyz".index(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f'axis must be "x", "y" or "z" (or array axis 0, 1, 2), got {axis!r}')
    return 2 - int(axis)


# --------------------------------------------------------------------------------------------
# camera and image
# --------------------------------------------------------------------------------------------
@dataclass
class Camera:
    """Look-at camera.  ``mode`` is "perspective" (``fov`` = vertical field of view in degrees) or
    "orthographic" (``ortho_height`` = world height of the image).  ``ssaa`` = s renders s*s rays
    per pixel and box-averages them.  ``background`` is RGB or RGBA (alpha < 1 gives a transparent
    background)."""
    eye: tuple = (3.0, -3.0, 2.0)
    target: tuple = (0.0, 0.0, 0.0)
    up: tuple = (0.0, 0.0, 1.0)
    mode: str = "perspective"
    fov: float = 35.0
    ortho_height: float = 1.0
    width: int = 1024
    height: int = 1024
    ssaa: int = 2
    background: tuple = (1.0, 1.0, 1.0, 1.0)

    def check(self):
        """Raise ``ValueError`` if the camera cannot produce a picture."""
        if self.mode not in ("perspective", "orthographic"):
            raise ValueError(f'camera mode must be "perspective" or "orthographic", got {self.mode!r}')
        if int(self.width) < 1 or int(self.height) < 1 or int(self.ssaa) < 1:
            raise ValueError(f"camera width, height and ssaa must be at least 1, got {self.width}, {self.height}, {self.ssaa}")
        if self.mode == "perspective" and not 0.0 < float(self.fov) < 180.0:
            raise ValueError(f"camera fov must be between 0 and 180 degrees, got {self.fov!r}")
        if self.mode == "orthographic" and not float(self.ortho_height) > 0.0:
            raise ValueError(f"camera ortho_height must be positive, got {self.ortho_height!r}")
        if len(self.background) not in (3, 4):
            raise ValueError("camera background must be RGB or RGBA")
        self.basis()
        return self

    def basis(self):
        """(right, up, forward) unit vectors; image x runs along right, image rows run along -up."""
        f = _vec3(self.target, "camera target") - _vec3(self.eye, "camera eye")
        if np.linalg.norm(f) == 0.0:
            raise ValueError("camera eye and target are the same point")
        f /= np.linalg.norm(f)
        r = np.cross(f, _vec3(self.up, "camera up"))
        if np.linalg.norm(r) < 1e-9:
            raise ValueError("camera up vector is parallel to the viewing direction")
        r /= np.linalg.norm(r)
        return r, np.cross(r, f), f

    @classmethod
    def fit(cls, lo, hi, direction=(1.0, -1.0, 0.6), margin=1.05, mode="perspective", fov=35.0, **kw):
        """Frame the box lo..hi: look at its centre from ``direction`` (centre -> eye) so that the
        bounding sphere of the box (radius times ``margin``) just fills the image, i.e. touches the
        shorter image side, in either ``mode``.  One rule for every scene, which is what makes
        index-space and physical-space pictures comparable."""
        lo, hi = _vec3(lo, "lo"), _vec3(hi, "hi")
        c, r = 0.5 * (lo + hi), 0.5 * float(np.linalg.norm(hi - lo)) * margin
        if not r > 0:
            raise ValueError("cannot fit a camera to an empty box (lo == hi)")
        d = _vec3(direction, "direction")
        d = d / np.linalg.norm(d)
        aspect = min(1.0, kw.get("width", cls.width) / kw.get("height", cls.height))    # < 1: the image is narrow
        if mode == "perspective":
            half = math.atan(aspect * math.tan(math.radians(fov) / 2))      # half angle of the shorter side
            dist = r / math.sin(half)
        else:
            dist = 4.0 * r
        return cls(eye=tuple(c + d * dist), target=tuple(c), mode=mode, fov=fov, ortho_height=2.0 * r / aspect, **kw)

    @classmethod
    def face_on(cls, lo, hi, axis="z", orientation="numpy", pixel_size=None, max_pixels=1024, border=0, **kw):
        """Orthographic camera that looks straight at the box lo..hi along ``axis`` ("z", "y", "x" =
        array axis 0, 1, 2) and frames its cross-section exactly, with square pixels.

        ``orientation="numpy"``     the picture is the numpy slice as ``matplotlib.imshow`` shows it:
                                    row 0 at the top.  z slice: columns = x, rows = y (seen from -z);
                                    y slice: columns = x, rows = z; x slice: columns = y, rows = z.
        ``orientation="physical"``  the same slice with the row axis pointing UP (z up for y and x
                                    slices, y up for a z slice seen from +z): ``numpy_slice[::-1]``.
                                    A rotation of the viewpoint, never a mirror image.

        ``pixel_size`` is the edge of one pixel in scene units (default: the longer side gets
        ``max_pixels`` pixels); ``border`` adds that many background pixels on every side.

        The pixel size is always exact (a scale bar is right); the picture has
        ``round(extent / pixel_size)`` pixels along each side and is centred on the box.  If an
        extent is not a whole number of pixels, the frame therefore differs from the box by less
        than half a pixel on each side; every pixel centre still lies inside the box."""
        lo, hi = _vec3(lo, "lo"), _vec3(hi, "hi")
        if orientation not in ("numpy", "physical"):
            raise ValueError(f'orientation must be "numpy" or "physical", got {orientation!r}')
        a = _world_axis(axis)
        u = 1 if a == 0 else 0                       # world axis along the image columns
        v = 1 if a == 2 else 2                       # world axis along the image rows
        w, h = hi[u] - lo[u], hi[v] - lo[v]
        if not (w > 0 and h > 0):
            raise ValueError(f"the box has no extent across axis {axis!r}: lo={lo}, hi={hi}")
        if pixel_size is None:
            pixel_size = max(w, h) / max_pixels
        right, up = np.zeros(3), np.zeros(3)
        right[u] = 1.0
        up[v] = -1.0 if orientation == "numpy" else 1.0
        fwd = np.cross(up, right)                    # as for every look-at camera: a turn of the viewpoint, never a mirror
        nx, ny = max(1, int(round(w / pixel_size))) + 2 * border, max(1, int(round(h / pixel_size))) + 2 * border
        c = 0.5 * (lo + hi)
        dist = (hi[a] - lo[a]) + w + h
        kw.setdefault("ssaa", 1)
        return cls(eye=tuple(c - fwd * dist), target=tuple(c), up=tuple(up), mode="orthographic",
                   ortho_height=ny * pixel_size, width=nx, height=ny, **kw)

    @classmethod
    def for_slice(cls, grid, axis="z", space="physical", orientation="numpy", pixels_per_voxel=1, border=0, **kw):
        """:meth:`face_on` for a whole slice of a ``VoxelGrid`` drawn in ``space`` (corner convention:
        the array fills [0, n*d]).  The pixel is the smaller in-plane voxel edge divided by
        ``pixels_per_voxel``.  When the other in-plane voxel edge is a whole multiple of it (6 and
        30 nm: 1 x 5 pixels per voxel) every voxel is a block of whole pixels and the picture IS
        the numpy slice.  Otherwise (4 and 33 nm) each pixel still shows the voxel under its
        centre, but the voxels are 8 or 9 pixels tall in turn (see :meth:`face_on`).
        ``pixels_per_voxel`` is a whole number: with a fraction the voxels of one picture would
        have unequal sizes, or pixel centres would fall on voxel boundaries.  For a reduced
        overview use :meth:`face_on` with ``max_pixels``."""
        if pixels_per_voxel != int(pixels_per_voxel) or pixels_per_voxel < 1:
            raise ValueError(f"pixels_per_voxel must be a whole number >= 1, got {pixels_per_voxel!r}")
        size = grid.display_voxel_size_xyz_nm(space) / NM_PER_UNIT      # (dx, dy, dz) as drawn, scene units
        a = _world_axis(axis)
        in_plane = [size[b] for b in range(3) if b != a]
        return cls.face_on(np.zeros(3), grid.extent_nm(space) / NM_PER_UNIT, axis=axis, orientation=orientation,
                           pixel_size=min(in_plane) / pixels_per_voxel, border=border, **kw)

    def pixel_size(self, distance=None):
        """Edge length of one pixel in scene units.  Orthographic: ``ortho_height / height``, the
        same everywhere in the picture.  Perspective: a pixel covers more the further away it
        looks, so the ``distance`` in front of the eye (along the viewing direction) is required:
        ``2 * tan(fov/2) * distance / height``."""
        if self.mode == "orthographic":
            return float(self.ortho_height) / int(self.height)
        if distance is None:
            raise ValueError("a perspective camera has no single pixel size: pass the distance from the eye")
        return 2.0 * math.tan(math.radians(self.fov) / 2) * float(distance) / int(self.height)

    def project(self, p):
        """World point(s) -> pixel coordinates (x, y) (continuous; pixel centres at +0.5)."""
        r, u, f = self.basis()
        q = np.atleast_2d(np.asarray(p, float)) - np.asarray(self.eye, float)
        x, y, z = q @ r, q @ u, q @ f
        if self.mode == "perspective":
            th = 2.0 * math.tan(math.radians(self.fov) / 2)
            sx, sy = x / (z * th * self.width / self.height), y / (z * th)
        else:
            sx, sy = x / (self.ortho_height * self.width / self.height), y / self.ortho_height
        return np.stack([(sx + 0.5) * self.width, (0.5 - sy) * self.height], axis=-1)


@dataclass
class Image:
    """Result of a render: ``rgba`` uint8 (H,W,4), ``depth`` float32 (H,W; inf = nothing),
    ``object_id`` int32 (H,W; -1 = nothing), ``rgba_float`` float32 before 8-bit rounding
    (premultiplied if the background is transparent).  With supersampling, depth and object id are
    those of the nearest sub-sample of the pixel.  Shows inline in Jupyter.

    Two times are recorded: ``seconds`` is the ray-casting kernel alone (launch + wait for the
    device); ``wall_seconds`` is the whole :meth:`Scene.render` call, which adds the read-back of
    the buffers and the 8-bit conversion -- what a notebook user waits for."""
    rgba: np.ndarray
    depth: np.ndarray
    object_id: np.ndarray
    rgba_float: np.ndarray
    seconds: float = 0.0
    wall_seconds: float = 0.0

    def save(self, path):
        save_png(path, self.rgba)
        return path

    def _repr_png_(self):
        return encode_png(self.rgba)


# --------------------------------------------------------------------------------------------
# kernel-side structures and functions
# --------------------------------------------------------------------------------------------
@wp.struct
class _Sampler:
    w2i: wp.mat44        # world (x,y,z) -> continuous index (i,j,k), CORNER convention: voxel [k,j,i] is the cube [i,i+1] x [j,j+1] x [k,k+1]
    vsize: wp.vec3       # voxel size along x,y,z (world units), for grid lines
    store: wp.int32      # packed array that holds the voxels: 0 = uint8, 1 = uint16, 2 = int32
    offset: wp.int32     # start of this volume in that array
    nx: wp.int32
    ny: wp.int32
    nz: wp.int32
    labels: wp.int32     # 0: scalar through LUT, 1: integer labels through the colour hash
    linear: wp.int32     # 0: nearest (crisp voxels), 1: trilinear (scalar volumes only)
    lut: wp.int32        # row of the LUT table
    zero_alpha: wp.float32   # labels: alpha of id 0 (0 = transparent, 1 = dark)
    focus: wp.int32      # labels: if >= 0 only this id keeps its colour, the others are greyed
    grid: wp.float32     # width (world units) of lines drawn on voxel boundaries, 0 = none
    grid_rgb: wp.vec3


@wp.struct
class _Cam:
    eye: wp.vec3
    right: wp.vec3
    up: wp.vec3
    fwd: wp.vec3
    ortho: wp.int32
    span_x: wp.float32   # ortho: world width/height of the image; perspective: 2*tan(fov/2)*(aspect, 1)
    span_y: wp.float32
    width: wp.int32
    height: wp.int32
    ss: wp.int32
    bg: wp.vec4


@wp.struct
class _Params:
    has_mesh: wp.int32
    max_hits: wp.int32       # surface hits composited per ray
    max_steps: wp.int32      # hard bound on surface queries per ray (transparent faces included)
    eps: wp.float32          # restart offset after a hit
    ambient: wp.float32
    hemi: wp.float32         # hemisphere ambient: 0 = uniform ambient, 1 = no ambient from below
    fog: wp.float32          # depth cueing: 0 = off, 1 = the far end is the background colour
    fog_near: wp.float32     # depth t where the fog starts ...
    fog_inv: wp.float32      # ... and 1 / (far - near)
    t_min: wp.float32        # stop when transmittance falls below this
    depth_alpha: wp.float32  # a surface registers in depth/id buffers if its alpha exceeds this
    n_gauss: wp.int32
    g_cut2: wp.float32       # squared Mahalanobis cut-off of a Gaussian
    g_amin: wp.float32       # smallest alpha that is composited
    dvr: wp.int32            # sampler used for direct volume rendering, -1 = off
    dvr_step: wp.float32     # ray-marching step (world units)
    dvr_ref: wp.float32      # length over which the LUT alpha accumulates
    dvr_id: wp.int32


@wp.func
def _hsv(hue: float, sat: float, val: float, n: float):
    k = wp.mod(n + hue * 6.0, 6.0)
    return val - val * sat * wp.clamp(wp.min(k, 4.0 - k), 0.0, 1.0)


@wp.func
def _label_rgb(label: int):
    h = wp.uint32(label) * wp.uint32(-1640531535)       # = 2654435761 mod 2**32 (written as an int32 literal)
    h = h ^ (h >> wp.uint32(15))
    h = h * wp.uint32(-2048144777)                      # = 2246822519 mod 2**32
    h = h ^ (h >> wp.uint32(13))
    hue = float(h & wp.uint32(65535)) / 65536.0
    sat = 0.45 + 0.45 * float((h >> wp.uint32(16)) & wp.uint32(255)) / 255.0
    val = 0.70 + 0.30 * float((h >> wp.uint32(24)) & wp.uint32(255)) / 255.0
    return wp.vec3(_hsv(hue, sat, val, 5.0), _hsv(hue, sat, val, 3.0), _hsv(hue, sat, val, 1.0))


@wp.func
def _voxel(vol8: wp.array(dtype=wp.uint8), vol16: wp.array(dtype=wp.uint16), vol32: wp.array(dtype=wp.int32),
           s: _Sampler, i: int, j: int, k: int):
    """Stored number of voxel [k, j, i] (indices clamped to the volume)."""
    ii = wp.clamp(i, 0, s.nx - 1)
    jj = wp.clamp(j, 0, s.ny - 1)
    kk = wp.clamp(k, 0, s.nz - 1)
    n = s.offset + (kk * s.ny + jj) * s.nx + ii
    v = int(0)
    if s.store == 0:
        v = int(vol8[n])
    elif s.store == 1:
        v = int(vol16[n])
    else:
        v = vol32[n]
    return v


@wp.func
def _sample(vol8: wp.array(dtype=wp.uint8), vol16: wp.array(dtype=wp.uint16), vol32: wp.array(dtype=wp.int32),
            luts: wp.array2d(dtype=wp.vec4), s: _Sampler, p: wp.vec3):
    """RGBA of the volume at world point p (alpha 0 outside the volume)."""
    q = wp.transform_point(s.w2i, p)    # continuous index (i, j, k); voxel [k,j,i] is the cube [i,i+1] x [j,j+1] x [k,k+1]
    if q[0] < 0.0 or q[1] < 0.0 or q[2] < 0.0 or q[0] > float(s.nx) or q[1] > float(s.ny) or q[2] > float(s.nz):
        return wp.vec4(0.0)
    i = int(wp.floor(q[0]))             # the voxel that contains p
    j = int(wp.floor(q[1]))
    k = int(wp.floor(q[2]))
    c = wp.vec4(0.0)
    if s.labels == 1:
        label = _voxel(vol8, vol16, vol32, s, i, j, k)
        if label == 0:
            c = wp.vec4(0.06, 0.06, 0.06, s.zero_alpha)
        else:
            rgb = _label_rgb(label)
            if s.focus >= 0 and label != s.focus:
                g = 0.55 + 0.25 * (rgb[0] + rgb[1] + rgb[2]) / 3.0
                rgb = wp.vec3(g, g, g) * 0.85 + rgb * 0.15
            c = wp.vec4(rgb[0], rgb[1], rgb[2], 1.0)
    else:
        v = float(0.0)
        if s.linear == 0:
            v = float(_voxel(vol8, vol16, vol32, s, i, j, k))
        else:                           # trilinear between voxel CENTRES, which sit at +0.5
            x = q[0] - 0.5
            y = q[1] - 0.5
            z = q[2] - 0.5
            i0 = int(wp.floor(x))
            j0 = int(wp.floor(y))
            k0 = int(wp.floor(z))
            fx = x - float(i0)
            fy = y - float(j0)
            fz = z - float(k0)
            for dk in range(2):
                for dj in range(2):
                    for di in range(2):
                        w = (wp.abs(1.0 - float(di) - fx) * wp.abs(1.0 - float(dj) - fy)
                             * wp.abs(1.0 - float(dk) - fz))
                        v += w * float(_voxel(vol8, vol16, vol32, s, i0 + di, j0 + dj, k0 + dk))
        v = wp.clamp(v, 0.0, 255.0)
        l0 = int(wp.floor(v))
        l1 = wp.min(l0 + 1, 255)
        f = v - float(l0)
        c = luts[s.lut, l0] * (1.0 - f) + luts[s.lut, l1] * f
    if s.grid > 0.0:                    # lines on voxel boundaries (integer index), width in world units
        for a in range(3):
            fr = q[a] - wp.floor(q[a])
            if wp.min(fr, 1.0 - fr) * s.vsize[a] < 0.5 * s.grid:
                c = wp.vec4(s.grid_rgb[0], s.grid_rgb[1], s.grid_rgb[2], wp.max(c[3], 1.0))
    return c


@wp.func
def _ray_box(s: _Sampler, o: wp.vec3, d: wp.vec3):
    """Entry and exit t of the ray through the box of a volume (t is the same in index space)."""
    oi = wp.transform_point(s.w2i, o)
    di = wp.transform_vector(s.w2i, d)
    hi = wp.vec3(float(s.nx), float(s.ny), float(s.nz))
    t0 = float(0.0)
    t1 = float(_FAR)
    for a in range(3):
        if wp.abs(di[a]) < 1.0e-12:
            if oi[a] < 0.0 or oi[a] > hi[a]:
                t1 = float(-1.0)
        else:
            ta = (0.0 - oi[a]) / di[a]
            tb = (hi[a] - oi[a]) / di[a]
            t0 = wp.max(t0, wp.min(ta, tb))
            t1 = wp.min(t1, wp.max(ta, tb))
    return t0, t1


@wp.func
def _on_wire(p: wp.vec3, d: wp.vec3, a: wp.vec3, b: wp.vec3, half_width: float, half_px: float):
    """Is the surface point p (seen along the ray direction d) on the line drawn along the edge
    a-b?  Yes if its distance to the edge, measured on the surface, is below ``half_width`` (world
    units) -- or, the display aid, if its distance to the edge AS SEEN ON SCREEN (measured
    perpendicular to the ray) is below ``half_px`` (world size of half the minimum pixel width)."""
    e = b - a
    le = wp.max(wp.length(e), 1.0e-20)
    hit = wp.length(wp.cross(e, p - a)) / le < half_width
    if half_px > 0.0:
        n = wp.cross(d, e)
        ln = wp.length(n)
        if ln > 1.0e-6 * le:            # (an edge seen exactly end-on has no on-screen direction)
            if wp.abs(wp.dot(p - a, n)) / ln < half_px:
                hit = True
    return hit


@wp.func
def _fogged(rgb: wp.vec3, t: float, rp: _Params, bg: wp.vec4):
    """Depth cueing: blend a colour seen at depth t toward the background colour."""
    f = rp.fog * wp.clamp((t - rp.fog_near) * rp.fog_inv, 0.0, 1.0)
    return rgb * (1.0 - f) + wp.vec3(bg[0], bg[1], bg[2]) * f


@wp.kernel
def _inflate_kernel(cam: _Cam, idx: wp.array(dtype=wp.int32), ctr: wp.array(dtype=wp.vec3),
                    dirs: wp.array(dtype=wp.vec3), rad: wp.array(dtype=wp.float32),
                    min_px: wp.array(dtype=wp.float32), pts: wp.array(dtype=wp.vec3)):
    """Display aid for thin tubes and small spheres: move every listed vertex to
    ``centre + direction * max(radius, half the minimum on-screen width)`` for THIS camera."""
    t = wp.tid()
    px = cam.span_y / float(cam.height)                 # world size of one pixel (orthographic)
    if cam.ortho == 0:                                  # perspective: it grows with the depth of the centre
        px = px * wp.max(wp.dot(ctr[t] - cam.eye, cam.fwd), 0.0)
    pts[idx[t]] = ctr[t] + dirs[t] * wp.max(rad[t], 0.5 * min_px[t] * px)


@wp.kernel
def _render_kernel(cam: _Cam, rp: _Params, mesh: wp.uint64,
                   pts: wp.array(dtype=wp.vec3), tri: wp.array(dtype=wp.int32), nrm: wp.array(dtype=wp.vec3),
                   f_rgba: wp.array(dtype=wp.vec4), f_rgba2: wp.array(dtype=wp.vec4),
                   f_mat: wp.array(dtype=wp.int32), f_obj: wp.array(dtype=wp.int32),
                   f_aux: wp.array(dtype=wp.vec4),
                   vol8: wp.array(dtype=wp.uint8), vol16: wp.array(dtype=wp.uint16), vol32: wp.array(dtype=wp.int32),
                   smp: wp.array(dtype=_Sampler), luts: wp.array2d(dtype=wp.vec4),
                   g_bvh: wp.uint64, g_mean: wp.array(dtype=wp.vec3), g_w: wp.array(dtype=wp.mat33),
                   g_rgbo: wp.array(dtype=wp.vec4), g_obj: wp.array(dtype=wp.int32),
                   out_rgba: wp.array2d(dtype=wp.vec4), out_depth: wp.array2d(dtype=wp.float32),
                   out_id: wp.array2d(dtype=wp.int32)):
    py, px = wp.tid()
    acc = wp.vec4(0.0)
    best_t = float(_FAR)
    best_id = int(-1)
    buf_t = _vecf()
    buf_a = _vecf()
    buf_i = _veci()
    for sy in range(cam.ss):
        for sx in range(cam.ss):
            # ---- primary ray of this sub-sample -------------------------------------------------
            u = (float(px) + (float(sx) + 0.5) / float(cam.ss)) / float(cam.width) - 0.5
            v = 0.5 - (float(py) + (float(sy) + 0.5) / float(cam.ss)) / float(cam.height)
            o = cam.eye
            d = cam.fwd
            if cam.ortho == 1:
                o = cam.eye + cam.right * (u * cam.span_x) + cam.up * (v * cam.span_y)
            else:
                d = wp.normalize(cam.fwd + cam.right * (u * cam.span_x) + cam.up * (v * cam.span_y))
            col = wp.vec3(0.0)          # accumulated (premultiplied) colour
            tr = float(1.0)             # transmittance
            hit_t = float(_FAR)
            hit_id = int(-1)
            t_cur = float(0.0)
            g_more = int(0)             # Gaussian stream: sorted buffer buf[g_p:g_n], refilled on demand
            if rp.n_gauss > 0:
                g_more = int(1)
            g_n = int(0)
            g_p = int(0)
            g_tmin = float(0.0)
            g_imin = int(2147483647)    # (g_tmin, g_imin): key of the last Gaussian already delivered
            gi = int(0)
            v_t = float(_FAR)           # next ray-marching sample
            v_end = float(-1.0)
            if rp.dvr >= 0:
                b0, b1 = _ray_box(smp[rp.dvr], o, d)
                if b1 > b0:
                    v_t = b0 + 0.5 * rp.dvr_step
                    v_end = b1
            n_hit = int(0)              # surface hits composited so far (fully transparent ones do not count)
            for step in range(rp.max_steps):
                # ---- 1. next surface along the ray ---------------------------------------------
                t_s = float(_FAR)
                face = int(-1)
                bu = float(0.0)
                bv = float(0.0)
                if rp.has_mesh == 1 and n_hit < rp.max_hits:
                    r = wp.mesh_query_ray(mesh, o + d * t_cur, d, _FAR)
                    if r.result:
                        t_s = t_cur + r.t
                        face = r.face
                        bu = r.u
                        bv = r.v
                # ---- 2. participating media (Gaussians, volume samples) in front of it, in order
                while tr > rp.t_min:
                    if g_p == g_n and g_more == 1:          # refill: nearest GAUSS_BUF beyond g_tmin
                        g_n = int(0)
                        g_p = int(0)
                        query = wp.bvh_query_ray(g_bvh, o, d)
                        while wp.bvh_query_next(query, gi):
                            ol = g_w[gi] * (o - g_mean[gi])     # ray in the canonical (unit) frame
                            dl = g_w[gi] * d
                            tg = -wp.dot(ol, dl) / wp.dot(dl, dl)   # parameter of maximum response
                            if tg > g_tmin or (tg == g_tmin and gi > g_imin):
                                cp = ol + dl * tg
                                d2 = wp.dot(cp, cp)                 # min squared distance, canonical
                                ga = g_rgbo[gi][3] * wp.exp(-0.5 * d2)
                                if d2 < rp.g_cut2 and ga >= rp.g_amin:
                                    pos = int(g_n)
                                    if g_n < GAUSS_BUF:
                                        g_n += 1
                                    else:
                                        pos = int(GAUSS_BUF - 1)
                                        if tg > buf_t[pos] or (tg == buf_t[pos] and gi > buf_i[pos]):
                                            pos = int(-1)
                                    while pos > 0:
                                        if buf_t[pos - 1] > tg or (buf_t[pos - 1] == tg and buf_i[pos - 1] > gi):
                                            buf_t[pos] = buf_t[pos - 1]
                                            buf_a[pos] = buf_a[pos - 1]
                                            buf_i[pos] = buf_i[pos - 1]
                                            pos -= 1
                                        else:
                                            break
                                    if pos >= 0:
                                        buf_t[pos] = tg
                                        buf_a[pos] = ga
                                        buf_i[pos] = gi
                        if g_n < GAUSS_BUF:
                            g_more = int(0)
                        if g_n > 0:
                            g_tmin = buf_t[g_n - 1]
                            g_imin = buf_i[g_n - 1]
                    t_g = float(_FAR)
                    if g_p < g_n:
                        t_g = buf_t[g_p]
                    if t_g >= t_s and v_t >= t_s:
                        break
                    ma = float(0.0)
                    mc = wp.vec3(0.0)
                    mt = float(t_g)
                    mid = int(rp.dvr_id)
                    if t_g <= v_t:                          # a Gaussian comes first
                        gi = buf_i[g_p]
                        ma = wp.min(buf_a[g_p], 0.999)
                        mc = wp.vec3(g_rgbo[gi][0], g_rgbo[gi][1], g_rgbo[gi][2])
                        mid = g_obj[gi]
                        g_p += 1
                    else:                                   # a volume sample comes first
                        sv = _sample(vol8, vol16, vol32, luts, smp[rp.dvr], o + d * v_t)
                        ma = 1.0 - wp.pow(1.0 - wp.min(sv[3], 0.999), rp.dvr_step / rp.dvr_ref)
                        mc = wp.vec3(sv[0], sv[1], sv[2])
                        mt = v_t
                        v_t += rp.dvr_step
                        if v_t >= v_end:
                            v_t = float(_FAR)
                    if rp.fog > 0.0:
                        mc = _fogged(mc, mt, rp, cam.bg)
                    col += mc * (tr * ma)
                    tr *= 1.0 - ma
                    if hit_id == -1 and hit_t >= _FAR and tr < 0.5:     # media: depth where opacity passes 1/2
                        hit_t = mt
                        hit_id = mid
                if face < 0 or tr <= rp.t_min:
                    break
                # ---- 3. shade the surface hit and composite it ---------------------------------
                p = o + d * t_s
                p0 = pts[tri[face * 3]]
                p1 = pts[tri[face * 3 + 1]]
                p2 = pts[tri[face * 3 + 2]]
                n = wp.normalize(wp.cross(p1 - p0, p2 - p0))
                ns = nrm[tri[face * 3]] * bu + nrm[tri[face * 3 + 1]] * bv + nrm[tri[face * 3 + 2]] * (1.0 - bu - bv)
                if wp.length_sq(ns) > 1.0e-12:
                    n = wp.normalize(ns)
                nv = n                                      # the normal on the side we look at
                if wp.dot(n, d) > 0.0:
                    nv = -n
                amb = rp.ambient * (1.0 - rp.hemi * 0.5 * (1.0 - wp.dot(nv, cam.up)))   # hemisphere: less from below
                lam = amb + (1.0 - rp.ambient) * wp.abs(wp.dot(n, d))                   # + two-sided headlight
                c = f_rgba[face]
                aux = f_aux[face]
                mat = f_mat[face]
                rgb = wp.vec3(c[0], c[1], c[2])
                a = c[3]
                if mat == MAT_LIT:
                    rgb = rgb * lam
                elif mat == MAT_VOLUME:
                    sv = _sample(vol8, vol16, vol32, luts, smp[int(aux[0])], p)
                    rgb = wp.vec3(sv[0] * c[0], sv[1] * c[1], sv[2] * c[2])
                    a = sv[3] * c[3]
                    if aux[2] > 0.5:
                        rgb = rgb * lam
                elif mat == MAT_WIRE:                       # aux = (line width, real-edge bits, lit bits, min pixels)
                    mask = int(aux[1])
                    flags = int(aux[2])
                    half_px = float(0.0)
                    if aux[3] > 0.0:                        # half the minimum line width, as a world length here
                        half_px = 0.5 * aux[3] * cam.span_y / float(cam.height)
                        if cam.ortho == 0:
                            half_px = half_px * t_s * wp.dot(d, cam.fwd)
                    line = int(0)
                    if (mask & 1) != 0 and _on_wire(p, d, p0, p1, 0.5 * aux[0], half_px):
                        line = int(1)
                    if (mask & 2) != 0 and _on_wire(p, d, p1, p2, 0.5 * aux[0], half_px):
                        line = int(1)
                    if (mask & 4) != 0 and _on_wire(p, d, p2, p0, 0.5 * aux[0], half_px):
                        line = int(1)
                    if line == 1:
                        c2 = f_rgba2[face]
                        rgb = wp.vec3(c2[0], c2[1], c2[2])
                        a = c2[3]
                        if (flags & 2) != 0:
                            rgb = rgb * lam
                    elif (flags & 1) != 0:
                        rgb = rgb * lam
                if a > 0.0:
                    n_hit += 1
                    if hit_t >= _FAR and a > rp.depth_alpha:
                        hit_t = t_s
                        hit_id = f_obj[face]
                    if rp.fog > 0.0:
                        rgb = _fogged(rgb, t_s, rp, cam.bg)
                    col += rgb * (tr * a)
                    tr *= 1.0 - a
                t_cur = t_s + rp.eps
            if tr <= rp.t_min:
                tr = float(0.0)
            acc += wp.vec4(col[0] + tr * cam.bg[0] * cam.bg[3], col[1] + tr * cam.bg[1] * cam.bg[3],
                           col[2] + tr * cam.bg[2] * cam.bg[3], 1.0 - tr + tr * cam.bg[3])
            if hit_t < best_t:
                best_t = hit_t
                best_id = hit_id
    out_rgba[py, px] = acc / float(cam.ss * cam.ss)
    out_depth[py, px] = best_t
    out_id[py, px] = best_id


# --------------------------------------------------------------------------------------------
# the scene: a triangle soup with per-face attributes, volumes, Gaussians
# --------------------------------------------------------------------------------------------
def _per(value, n, width):
    """A colour, or one colour per primitive, as an (n, width) float32 array; a missing alpha
    becomes 1.  Components are floats in [0, 1]; anything else raises (0..255 colours would draw
    white, a wrong number of colours would colour the wrong primitives)."""
    a = np.asarray(value, np.float32)
    if a.ndim == 1:
        a = np.broadcast_to(a, (n, a.shape[0]))
    if a.ndim != 2 or a.shape[1] not in (width - 1, width) or len(a) != n:
        raise ValueError(f"a colour must be RGB or RGBA, one value or one per primitive ({n}); got shape {a.shape}")
    if not np.all(np.isfinite(a)) or (a.size and (a.min() < -1e-6 or a.max() > 1.0 + 1e-6)):
        raise ValueError("colour components must be finite numbers in [0, 1] (not 0..255)")
    if a.shape[1] == width - 1:
        a = np.concatenate([a, np.ones((len(a), 1), np.float32)], axis=1)
    return np.clip(a, 0.0, 1.0).astype(np.float32)          # (the clip only removes round-off, see the check above)


_PER_PRIMITIVE = (("color", 2), ("wire_color", 2), ("object_id", 1))      # style key, ndim of a per-primitive array


def _style_take(style, index):
    """Copy of ``style`` in which arrays that hold one value per primitive are re-indexed by
    ``index`` (used when a tessellator drops, repeats or regroups primitives)."""
    out = dict(style)
    for key, ndim in _PER_PRIMITIVE:
        if key in out and np.ndim(out[key]) == ndim:
            out[key] = np.asarray(out[key])[index]
    return out


def _stretched(centres, dirs, radii, stretch):
    """A round surface described by ``point = centre + radius * dir`` (``dir`` a unit vector, also
    the surface normal) after the scaling p -> stretch * p along x, y and z.  Returns the same
    description of the image, ``(centres, unit dirs, radii)``, and its unit normals."""
    a = _vec3(stretch, "stretch")
    if a.min() <= 0:
        raise ValueError(f"stretch factors must be positive, got {stretch!r}")
    offset = dirs * a                                   # image of the unit vector from the centre to the point
    length = np.linalg.norm(offset, axis=1)
    normals = dirs / a                                  # normals transform with the inverse matrix
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return centres * a, offset / length[:, None], radii * length, normals


def _exposed_faces(mask):
    """For a boolean volume ``mask[z,y,x]``: yield ``(row of _BOX_QUADS, kji)`` for each of the six
    face directions, where ``kji`` (n,3) are the set voxels whose neighbour in that direction is
    unset (or outside the array) -- the faces that can be seen."""
    m = np.pad(np.asarray(mask, bool), 1)
    for axis in range(3):                                   # array axis: 0=z, 1=y, 2=x
        for si, sgn in enumerate((-1, 1)):
            yield (2 - axis) * 2 + si, np.argwhere(m & ~np.roll(m, -sgn, axis=axis)) - 1


class Scene:
    """Everything that is drawn.  Opaque/translucent geometry is ONE merged triangle soup (one
    ``wp.Mesh``); every ``add_*`` tessellator appends to it and returns the number of triangles it
    added.  Coordinates are scene units, (x, y, z).  Style keywords shared by all of them:

    ``color``       RGB or RGBA, one value or one per primitive (alpha < 1 = translucent)
    ``object_id``   integer written to the object-id buffer, one value or one per primitive
    ``lit``         True: headlight Lambert + ambient, two-sided (material 0); False: flat (material 1)
    ``sampler``     id returned by :meth:`add_volume`: colour comes from the volume (material 2)
    ``wire_width``  > 0: draw the triangle edges as lines of this width in world units (material 3);
                    ``wire_color`` is the line colour, ``color`` the fill (alpha 0 = pure wireframe),
                    ``wire_lit`` shades the lines too
    ``wire_min_pixels``  > 0: DISPLAY AID.  The lines are never drawn thinner than this many pixels
                    on screen, whatever the zoom (each face draws half of the line on its side of
                    the edge).  The geometry is unchanged; the picture is no longer to scale for the
                    line width.  May be used with ``wire_width=0``: lines of constant pixel width.

    Two dictionaries are free for whoever fills the scene; the renderer never reads them:
    ``objects``     {object id: description}: what the numbers of the object-id buffer mean
                    (``usd_to_emray.build_scene`` writes {"path", "layer", "type"} of each prim)
    ``skipped``     {reason: [names]}: what could not be drawn
    """

    def __init__(self):
        self.objects, self.skipped = {}, {}
        self._vol = ([], [], [])                # packed voxel stores: uint8, uint16, int32
        self._vol_size = [0, 0, 0]
        self._vol_ids = {}
        self.samplers, self._luts = [], []
        self._dvr = None
        self._vol_gpu = None
        self.clear_geometry()

    def clear_geometry(self):
        """Drop all triangles and Gaussians but keep the registered volumes (they stay on the
        device): the cheap way to redraw, e.g. to move a slice from a notebook slider."""
        self._v, self._n, self._f = [], [], []
        self._rgba, self._rgba2, self._mat, self._obj, self._aux = [], [], [], [], []
        self._inflate = []
        self._nv, self._g = 0, []
        self._lo, self._hi = np.full(3, np.inf), np.full(3, -np.inf)
        self._gpu = None
        self.build_seconds = 0.0

    # ---- low level ----------------------------------------------------------------------------
    def _emit(self, verts, faces, normals=None, prim=None, edge_mask=7, *, color=(0.8, 0.8, 0.8, 1.0),
              object_id=0, lit=True, sampler=-1, wire_width=0.0, wire_color=(0.0, 0.0, 0.0, 1.0),
              wire_lit=False, wire_min_pixels=0.0, inflate=None):
        """Append triangles.  ``prim`` gives, per face, the index of the primitive it belongs to;
        per-primitive ``color`` / ``object_id`` arrays are expanded through it.  ``inflate`` =
        (centres (V,3), unit directions (V,3), radii (V,), min_pixels) marks vertices that the
        renderer may push outward to keep a minimum on-screen size (see :meth:`add_tubes`)."""
        verts = np.asarray(verts, np.float64)
        faces = np.asarray(faces)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(f"vertices must have shape (V, 3), got {verts.shape}")
        if faces.size and (faces.ndim != 2 or faces.shape[1] != 3):
            raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
        faces = faces.astype(np.int64).reshape(-1, 3)
        nf = len(faces)
        if nf == 0:
            return 0
        if not np.all(np.isfinite(verts)):
            raise ValueError("vertices contain NaN or infinity")
        if faces.min() < 0 or faces.max() >= len(verts):
            raise ValueError(f"face indices must lie in 0..{len(verts) - 1}, got {faces.min()}..{faces.max()}")
        if normals is not None and np.shape(normals) != verts.shape:
            raise ValueError(f"normals must have the shape of the vertices {verts.shape}, got {np.shape(normals)}")
        if wire_width < 0 or wire_min_pixels < 0:
            raise ValueError("wire_width and wire_min_pixels must not be negative")
        prim = np.arange(nf) if prim is None else np.asarray(prim)
        npr = int(prim.max()) + 1
        rgba, rgba2 = _per(color, npr, 4)[prim], _per(wire_color, npr, 4)[prim]
        aux = np.zeros((nf, 4), np.float32)
        mat = MAT_LIT if lit else MAT_UNLIT
        if sampler >= 0:
            if sampler >= len(self.samplers):
                raise ValueError(f"unknown sampler {sampler}: {len(self.samplers)} volume(s) registered")
            mat, aux[:, 0], aux[:, 2] = MAT_VOLUME, sampler, float(bool(lit))
        elif wire_width > 0 or wire_min_pixels > 0:
            mat = MAT_WIRE
            aux[:, 0], aux[:, 1], aux[:, 2], aux[:, 3] = wire_width, edge_mask, int(bool(lit)) + 2 * int(bool(wire_lit)), wire_min_pixels
        if np.ndim(object_id):
            obj = np.asarray(object_id, np.int32)
            if obj.ndim != 1 or len(obj) != npr:
                raise ValueError(f"object_id must be one integer or one per primitive ({npr}), got shape {obj.shape}")
            obj = obj[prim]
        else:
            obj = np.full(nf, object_id, np.int32)
        if inflate is not None:
            centres, dirs, radii, min_pixels = inflate
            self._inflate.append((self._nv + np.arange(len(verts)), np.asarray(centres, np.float64).reshape(-1, 3),
                                  np.asarray(dirs, np.float64).reshape(-1, 3), np.asarray(radii, np.float64).reshape(-1),
                                  np.full(len(verts), float(min_pixels))))
        self._v.append(verts.astype(np.float32))
        self._n.append(np.zeros_like(verts, dtype=np.float32) if normals is None else np.asarray(normals, np.float32))
        self._f.append((faces + self._nv).astype(np.int32))
        self._rgba.append(rgba)
        self._rgba2.append(rgba2)
        self._mat.append(np.full(nf, mat, np.int32))
        self._obj.append(obj)
        self._aux.append(aux)
        self._nv += len(verts)
        self._lo, self._hi = np.minimum(self._lo, verts.min(0)), np.maximum(self._hi, verts.max(0))
        self._gpu = None
        return nf

    # ---- tessellators -------------------------------------------------------------------------
    def add_mesh(self, vertices, faces, normals=None, **style):
        """Arbitrary triangle mesh; ``normals`` (per vertex) switch on smooth shading.  A per-face
        ``color`` / ``object_id`` array is allowed."""
        return self._emit(vertices, faces, normals, **style)

    def add_quads(self, corners, **style):
        """Quads given as (Q,4,3) corners in order around the quad.  The diagonal is not a wire edge."""
        q = np.asarray(corners, np.float64)
        if q.size % 12 or (q.ndim >= 2 and q.shape[-1] != 3):
            raise ValueError(f"quads must have shape (Q, 4, 3), got {q.shape}")
        q = q.reshape(-1, 4, 3)
        base = 4 * np.arange(len(q))[:, None]
        faces = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]])
        prim = np.tile(np.arange(len(q)), 2)
        mask = np.repeat([3.0, 6.0], len(q))            # bit e = edge (vertex e, vertex e+1) is real
        return self._emit(q.reshape(-1, 3), faces, None, prim, mask, **style)

    def add_boxes(self, centers, sizes, **style):
        """Axis-aligned boxes with centre (N,3) and full size (3,) or (N,3) -- e.g. voxel bricks with
        a non-cubic size (dx, dy, dz)."""
        c = np.asarray(centers, np.float64).reshape(-1, 3)
        h = 0.5 * np.broadcast_to(np.asarray(sizes, np.float64), c.shape)
        quads = c[:, None, None, :] + h[:, None, None, :] * _BOX_QUADS[None]
        return self.add_quads(quads, **_style_take(style, np.repeat(np.arange(len(c)), 6)))

    def add_voxels(self, mask, voxel_size, origin=(0.0, 0.0, 0.0), convention="center", **style):
        """Voxel bricks of a boolean volume ``mask[z,y,x]``: only faces between a set voxel and an
        unset one are emitted, each as a quad of the true (anisotropic) voxel face size.

        ``voxel_size`` is ``(dz, dy, dx)``.  With ``convention="center"`` the brick of voxel
        ``[k,j,i]`` is centred on ``origin + (i*dx, j*dy, k*dz)``; with ``"corner"`` it is the box
        from ``origin + (i*dx, j*dy, k*dz)`` to ``origin + ((i+1)*dx, (j+1)*dy, (k+1)*dz)``.
        ``color`` may be a function ``f(kji) -> (n,3|4)`` of the (n,3) voxel indices."""
        mask = np.asarray(mask)
        if mask.ndim != 3:
            raise ValueError(f"mask must be a 3D array [z, y, x], got shape {mask.shape}")
        if not mask.any():
            raise ValueError("the mask is empty: there is no voxel to draw")
        size = index_to_world(voxel_size).diagonal()[:3]            # (dx, dy, dz)
        first = _vec3(origin, "origin") + (0.5 * size if _convention(convention) == "corner" else 0.0)
        quads, kjis = [], []
        for face, kji in _exposed_faces(mask):
            ctr = kji[:, ::-1] * size + first                       # centre of voxel [k,j,i], as (x, y, z)
            quads.append(ctr[:, None, :] + 0.5 * size * _BOX_QUADS[face][None])
            kjis.append(kji)
        if callable(style.get("color")):
            style["color"] = style["color"](np.concatenate(kjis))
        return self.add_quads(np.concatenate(quads), **style)

    def add_tubes(self, points, edges, radii, sides=10, caps="flat", min_pixels=0.0, stretch=None, **style):
        """Truncated cones along the edges (E,2) of a point set, radius per point (or one value):
        skeleton edges, wire edges, axes.  Smooth-shaded.

        ``caps``  "flat"  a disc closes both ends of every tube (its length is exact);
                  "round" a sphere of the point's radius sits on every point used by an edge: closes
                          the ends and fills the gaps where tubes meet at an angle (skeletons, boxes);
                  "none"  open ends, the inside of the tube can be seen.
        ``min_pixels`` > 0: DISPLAY AID.  A tube narrower than this many pixels on screen is drawn
                  this wide (its vertices are pushed outward for the camera of each render), so
                  thin lines neither vanish nor break up when zoomed out.  The picture is then not
                  to scale for the tube radius; lengths and positions are unchanged.
        ``stretch`` = (ax, ay, az): the finished round tubes -- points, radii and joints -- are
                  scaled by these factors along x, y and z about the origin, so their
                  cross-sections become ellipses: the exact image of the tubes in a display
                  space whose axes are scaled differently (see :meth:`add_skeleton`)."""
        p = np.asarray(points, np.float64).reshape(-1, 3)
        e = np.asarray(edges, np.int64).reshape(-1, 2)
        if np.ndim(radii) > 1 or (np.ndim(radii) == 1 and len(radii) != len(p)):
            raise ValueError(f"radii must be one number or one per point ({len(p)}), got shape {np.shape(radii)}")
        r = np.broadcast_to(np.asarray(radii, np.float64), (len(p),))
        if caps is None:
            caps = "none"
        if caps not in ("flat", "round", "none"):
            raise ValueError(f'caps must be "flat", "round" or "none", got {caps!r}')
        if sides < 3:
            raise ValueError("a tube needs at least 3 sides")
        if e.size and (e.min() < 0 or e.max() >= len(p)):
            raise ValueError(f"edge indices must lie in 0..{len(p) - 1}, got {e.min()}..{e.max()}")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)) or (r.size and r.min() < 0):
            raise ValueError("tube points and radii must be finite, radii not negative")
        ax = p[e[:, 1]] - p[e[:, 0]]
        ln = np.linalg.norm(ax, axis=1)
        ok = ln > 1e-12                                 # edges of zero length are dropped
        style = _style_take(style, ok)
        e, ax = e[ok], ax[ok] / ln[ok, None]
        if len(e) == 0:
            return 0
        helper = np.where(np.abs(ax[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        u = np.cross(ax, helper)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        w = np.cross(ax, u)
        th = np.arange(sides) * (2 * np.pi / sides)
        ring = np.cos(th)[None, :, None] * u[:, None, :] + np.sin(th)[None, :, None] * w[:, None, :]
        # per tube 2 * sides vertices: a ring around its first end, then a ring around its second end
        ends = np.repeat(p[e], sides, axis=1).reshape(-1, 3)        # the end point each vertex belongs to
        rads = np.repeat(r[e], sides, axis=1).reshape(-1)           # ... and the radius there
        dirs = np.concatenate([ring, ring], 1).reshape(-1, 3)       # unit direction from the axis to the vertex
        normals = dirs                                              # ... which is also the surface normal there
        if stretch is not None:
            ends, dirs, rads, normals = _stretched(ends, dirs, rads, stretch)
        verts = ends + rads[:, None] * dirs
        inflate = (ends, dirs, rads, min_pixels) if min_pixels > 0 else None
        s, s1 = np.arange(sides), (np.arange(sides) + 1) % sides
        local = np.concatenate([np.stack([s, s1, sides + s], 1), np.stack([s1, sides + s1, sides + s], 1)])
        faces = (np.arange(len(e))[:, None, None] * 2 * sides + local[None]).reshape(-1, 3)
        nf = self._emit(verts, faces, normals, np.repeat(np.arange(len(e)), 2 * sides), inflate=inflate, **style)
        if caps == "flat":                              # the two rings again, as flat-shaded triangle fans
            fan = np.arange(1, sides - 1)
            local = np.concatenate([np.stack([0 * fan, fan + 1, fan], 1), np.stack([sides + 0 * fan, sides + fan, sides + fan + 1], 1)])
            faces = (np.arange(len(e))[:, None, None] * 2 * sides + local[None]).reshape(-1, 3)
            nf += self._emit(verts, faces, None, np.repeat(np.arange(len(e)), 2 * (sides - 2)), inflate=inflate, **style)
        elif caps == "round":
            used = np.unique(e)
            owner = np.zeros(len(p), np.int64)          # a joint takes the style of the first edge that uses it
            owner[e[::-1].ravel()] = np.repeat(np.arange(len(e))[::-1], 2)
            nf += self.add_spheres(p[used], r[used], subdiv=1, min_pixels=min_pixels, stretch=stretch,
                                   **_style_take(style, owner[used]))
        return nf

    def add_spheres(self, centers, radii, subdiv=2, min_pixels=0.0, stretch=None, **style):
        """Icospheres (skeleton nodes): centres (N,3), radius per sphere or one value.
        ``min_pixels`` > 0 is the display aid of :meth:`add_tubes`: a sphere is never drawn smaller
        than this many pixels across.  ``stretch`` = (ax, ay, az) scales the finished spheres
        along x, y and z about the origin (they become ellipsoids), as in :meth:`add_tubes`."""
        c = np.asarray(centers, np.float64).reshape(-1, 3)
        if np.ndim(radii) > 1 or (np.ndim(radii) == 1 and len(radii) != len(c)):
            raise ValueError(f"radii must be one number or one per sphere ({len(c)}), got shape {np.shape(radii)}")
        r = np.broadcast_to(np.asarray(radii, np.float64), (len(c),))
        if not np.all(np.isfinite(r)) or (r.size and r.min() < 0):
            raise ValueError("sphere radii must be finite and not negative")
        sv, sf = icosphere(subdiv)
        ctr = np.repeat(c, len(sv), axis=0)
        dirs = np.tile(sv, (len(c), 1))
        rads = np.repeat(r, len(sv))
        normals = dirs
        if stretch is not None:
            ctr, dirs, rads, normals = _stretched(ctr, dirs, rads, stretch)
        faces = (np.arange(len(c))[:, None, None] * len(sv) + sf[None]).reshape(-1, 3)
        return self._emit(ctr + rads[:, None] * dirs, faces, normals, np.repeat(np.arange(len(c)), len(sf)),
                          inflate=(ctr, dirs, rads, min_pixels) if min_pixels > 0 else None, **style)

    def add_ellipsoids(self, centers, matrices, subdiv=3, radii=None, **style):
        """Ellipsoids ``centre + M @ unit_sphere`` with M (N,3,3) -- e.g. the k-sigma surface of a
        Gaussian is ``M = R @ diag(k * sigma)`` (see :func:`cov_to_axes`)."""
        if matrices is None:
            return self.add_spheres(centers, radii, subdiv=subdiv, **style)
        c = np.asarray(centers, np.float64).reshape(-1, 3)
        sv, sf = icosphere(subdiv)
        m = np.asarray(matrices, np.float64).reshape(-1, 3, 3)
        if len(m) != len(c) or not np.all(np.isfinite(m)):
            raise ValueError(f"need one finite 3x3 matrix per ellipsoid ({len(c)}), got {m.shape}")
        verts = c[:, None, :] + np.einsum("nab,vb->nva", m, sv)
        normals = np.einsum("nba,vb->nva", np.linalg.inv(m), sv)        # M^-T v
        normals /= np.linalg.norm(normals, axis=2, keepdims=True)
        faces = (np.arange(len(c))[:, None, None] * len(sv) + sf[None]).reshape(-1, 3)
        return self._emit(verts.reshape(-1, 3), faces, normals.reshape(-1, 3), np.repeat(np.arange(len(c)), len(sf)),
                          **style)

    # ---- teaching aids: real geometry in the scene ----------------------------------------------
    def add_wire_box(self, lo, hi, radius=None, color=(0.15, 0.15, 0.15), min_pixels=1.5, **style):
        """The 12 edges of the box lo..hi as thin tubes with round corners: the frame of a picture
        (for a volume: ``add_wire_box(*scene.volume_corners(sampler))``).  ``radius`` is in scene
        units (default 0.2 % of the box diagonal); ``min_pixels`` keeps the lines visible at any
        zoom (display aid, see :meth:`add_tubes`)."""
        lo, hi = _vec3(lo, "lo"), _vec3(hi, "hi")
        p, e = box_edges(lo, hi)
        if radius is None:
            radius = 0.002 * float(np.linalg.norm(hi - lo))
        return self.add_tubes(p, e, radius, caps="round", min_pixels=min_pixels, color=color, **style)

    def add_axes(self, origin=(0.0, 0.0, 0.0), lengths=1.0, radius=None, heads=True, min_pixels=1.5,
                 colors=AXIS_COLORS, **style):
        """Three arrows from ``origin`` along +x (red), +y (green) and +z (blue).  ``lengths`` is one
        length or three (scene units), measured from the origin to the tip.  ``radius`` is the shaft
        radius (default 2.5 % of the longest arrow, thick enough to show on an edge of
        :meth:`add_wire_box`); ``heads=False`` draws plain rods."""
        o = _vec3(origin, "origin")
        length = np.broadcast_to(np.asarray(lengths, np.float64), (3,))
        if not np.all(np.isfinite(length)) or length.min() <= 0:
            raise ValueError(f"axis lengths must be positive, got {lengths!r}")
        if radius is None:
            radius = 0.025 * float(length.max())
        colors = _per(colors, 3, 4)
        head = np.minimum(6.0 * radius, 0.3 * length) if heads else np.zeros(3)     # length of each arrow head
        shaft_end, tip = o + np.diag(length - head), o + np.diag(length)
        nf = self.add_tubes(np.concatenate([o[None], shaft_end]), [(0, 1), (0, 2), (0, 3)], radius, caps="flat",
                            min_pixels=min_pixels, color=colors, **style)
        nf += self.add_spheres(o[None], radius, subdiv=1, min_pixels=min_pixels, color=(0.1, 0.1, 0.1), **style)
        if heads:
            pts = np.stack([shaft_end, tip], 1).reshape(-1, 3)
            nf += self.add_tubes(pts, [(0, 1), (2, 3), (4, 5)], np.tile([2.5 * radius, 2.5e-3 * radius], 3), caps="flat",
                                 color=colors, **style)
        return nf

    def add_scale_bar(self, start, direction, length, radius=None, color=(0.0, 0.0, 0.0), min_pixels=2.0, **style):
        """A straight bar of exactly ``length`` scene units from ``start`` along ``direction``, with
        flat ends: a scale bar that lives in the scene and is foreshortened with it.  Choose a round
        length (e.g. 1.0 = 1 micrometre).  ``radius`` defaults to ``length / 80``; the bar is drawn
        unlit so that its colour is exact."""
        s, d = _vec3(start, "start"), _vec3(direction, "direction")
        if not (np.isfinite(length) and length > 0) or np.linalg.norm(d) == 0:
            raise ValueError("a scale bar needs a positive length and a non-zero direction")
        style.setdefault("lit", False)
        return self.add_tubes([s, s + d / np.linalg.norm(d) * length], [(0, 1)], length / 80.0 if radius is None else radius,
                              caps="flat", min_pixels=min_pixels, color=color, **style)

    # ---- volumes ------------------------------------------------------------------------------
    def _store_volume(self, data):
        """Copy ``data`` into one of the three packed stores and return its record
        (store, offset, shape, min, max).  An array object that was registered before is not stored
        a second time -- unless its values were changed in place since: then the new values are
        stored (samplers made earlier keep showing the old ones)."""
        arr = np.ascontiguousarray(data)
        if arr.ndim != 3 or arr.size == 0:
            raise ValueError(f"a volume must be a non-empty 3D array [z, y, x], got shape {arr.shape}")
        if arr.dtype.kind == "f":                       # voxvalue as floats in [0, 1]: stored as 0..255
            if not np.all(np.isfinite(arr)):
                raise ValueError("float volumes must not contain NaN or infinity")
            if arr.min() < 0.0 or arr.max() > 1.0:
                raise ValueError("float volumes must be normalised to [0, 1] (got %g .. %g)" % (arr.min(), arr.max()))
            arr = np.rint(arr.astype(np.float64) * 255.0).astype(np.uint8)
        elif arr.dtype.kind not in "iub":
            raise TypeError("unsupported volume dtype %s" % arr.dtype)
        lo, hi = int(arr.min()), int(arr.max())
        if lo < -2 ** 31 or hi > 2 ** 31 - 1:
            raise ValueError("volume values %d .. %d do not fit into int32: renumber the ids first, e.g. "
                             "ids, compact = np.unique(labels, return_inverse=True)" % (lo, hi))
        store = 0 if arr.dtype in (np.uint8, np.bool_) else (1 if arr.dtype == np.uint16 else 2)
        packed = arr.astype((np.uint8, np.uint16, np.int32)[store]).reshape(-1)     # always a copy
        known = self._vol_ids.get(id(data))
        if known is not None and known[0] == store and np.array_equal(known[5], packed):
            return known[:5]
        if self._vol_size[store] + packed.size > 2 ** 31 - 1:
            raise ValueError("more than 2**31 voxels of one storage type in a scene")
        record = (store, self._vol_size[store], arr.shape, lo, hi, packed, data)    # holding `data` keeps its id unique
        self._vol_ids[id(data)] = record
        self._vol[store].append(packed)
        self._vol_size[store] += packed.size
        return record[:5]

    def add_volume(self, data, voxel_size=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), lut=None, labels=False,
                   interp="nearest", zero_alpha=0.0, focus=-1, grid=0.0, grid_color=(1.0, 0.85, 0.0),
                   convention="center"):
        """Register a volume ``data[z,y,x]`` placed in the world by ``voxel_size=(dz,dy,dx)`` and
        ``origin`` (x, y, z); returns a sampler id for :meth:`add_slice`, ``sampler=`` and
        :meth:`set_dvr`.

        ``convention="center"`` (default here): ``origin`` is the CENTRE of voxel ``[0,0,0]``, the
        volume fills ``origin - size/2 .. origin + (n - 1/2)*size``.  ``convention="corner"``:
        ``origin`` is the low CORNER of voxel ``[0,0,0]``, the volume fills ``origin .. origin +
        n*size`` (the convention of ``grid.py``; see :meth:`add_volume_grid`).

        Values: a SCALAR volume (``labels=False``) is looked up in a 256-entry table ``lut``
        ((256,4) RGBA, default identity grey), so it must be uint8-like 0..255, or float in [0,1]
        (stored as ``rint(255*v)``).  A LABEL volume (``labels=True``) holds integer ids coloured
        by a hash (id 0 gets alpha ``zero_alpha``; ``focus`` keeps one id coloured and greys the
        rest); ids must fit into int32.  Anything else raises.

        Memory on the device: 1 byte per voxel for uint8 / bool / float input, 2 for uint16, 4 for
        every other integer type.  The same array object may be registered several times (index
        space, physical space, other LUTs): its voxels are stored once.  The values are copied
        when the volume is registered; changing the array afterwards does not change the picture
        (register it again to show the new values).

        ``interp`` is "nearest" (crisp voxels) or "linear" (trilinear between voxel centres; scalar
        volumes only, labels are never interpolated).  ``grid`` > 0 draws the voxel boundaries as
        lines of that width (world units) in ``grid_color``."""
        if interp not in ("nearest", "linear"):
            raise ValueError(f'interp must be "nearest" or "linear", got {interp!r}')
        if labels and np.asarray(data).dtype.kind not in "iub":
            raise TypeError("label volumes must be integer arrays, got %s" % np.asarray(data).dtype)
        table = lut_grey() if lut is None else np.asarray(lut, np.float32)
        if table.shape != (256, 4) or not np.all(np.isfinite(table)) or table.min() < 0.0 or table.max() > 1.0:
            raise ValueError(f"a lookup table must be a (256, 4) RGBA array of numbers in [0, 1], got shape {table.shape}")
        i2w = index_to_world(voxel_size, origin)
        vsize = np.diag(i2w)[:3].copy()                 # (dx, dy, dz): the voxel size enters here
        corner = i2w[:3, 3].copy()                      # low corner of voxel [0, 0, 0] ...
        if _convention(convention) == "center":
            corner -= 0.5 * vsize                       # ... which lies half a voxel below its centre
        store, offset, shape, lo, hi = self._store_volume(data)
        if not labels and (lo < 0 or hi > 255):
            raise ValueError("a scalar volume is looked up in a 256-entry table and must hold 0..255 (or floats in "
                             "[0, 1]); got %d .. %d.  Normalise it, or pass labels=True for instance ids." % (lo, hi))
        self._luts.append(table)
        self.samplers.append(dict(convention=convention, corner=corner, shape=shape, store=store, offset=offset,
                                  labels=int(labels), linear=int(interp == "linear"), lut=len(self._luts) - 1,
                                  zero_alpha=zero_alpha, focus=focus, grid=grid, grid_rgb=grid_color, vsize=vsize))
        self._gpu = None
        return len(self.samplers) - 1

    @property
    def volume_nbytes(self):
        """Bytes the registered volumes occupy on the device (1, 2 or 4 per voxel, see :meth:`add_volume`)."""
        return int(self._vol_size[0] + 2 * self._vol_size[1] + 4 * self._vol_size[2])

    def _sampler(self, sampler):
        """The record of a registered volume, or a clear error for an id that :meth:`add_volume`
        did not return (a negative id would otherwise silently pick a volume from the end)."""
        if isinstance(sampler, bool) or not isinstance(sampler, (int, np.integer)) or not 0 <= sampler < len(self.samplers):
            raise ValueError(f"unknown sampler {sampler!r}: {len(self.samplers)} volume(s) registered")
        return self.samplers[sampler]

    def volume_corners(self, sampler):
        """World-space (lo, hi) corners of the box occupied by a registered volume."""
        s = self._sampler(sampler)
        return s["corner"].copy(), s["corner"] + np.array(s["shape"][::-1], float) * s["vsize"]

    def add_slice(self, sampler, axis, index, lo=None, hi=None, **style):
        """One slice of a registered volume as a volume-sampled quad at its true position: the plane
        through the CENTRE of voxel layer ``index``, spanning the full extent of the voxels.
        ``axis`` is "z", "y" or "x" (array axis 0, 1, 2); ``lo``/``hi`` optionally restrict it to the
        voxel range ``[lo, hi)`` given as (k, j, i)."""
        s = self._sampler(sampler)
        a = _world_axis(axis)                                                   # world axis of the normal
        n = np.array(s["shape"][::-1], float)                                   # (nx, ny, nz)
        if index != int(index):
            raise ValueError(f"a slice index is a whole number (the number of a voxel layer), got {index!r}")
        if not 0 <= index <= n[a] - 1:
            raise ValueError(f"slice index {index} is outside 0..{int(n[a]) - 1} along axis {axis!r}")
        qlo = np.zeros(3) if lo is None else np.asarray(lo, float)[::-1]        # voxel corners, as (i, j, k)
        qhi = n if hi is None else np.asarray(hi, float)[::-1]
        b, c = (a + 1) % 3, (a + 2) % 3
        if not all(0 <= qlo[x] < qhi[x] <= n[x] for x in (b, c)):
            raise ValueError(f"the slice window lo={lo}, hi={hi} (k, j, i) is empty or outside the volume {s['shape']}")
        idx = np.zeros((4, 3))
        idx[:, a] = index + 0.5                                                 # the centre of the layer
        idx[:, b] = [qlo[b], qhi[b], qhi[b], qlo[b]]
        idx[:, c] = [qlo[c], qlo[c], qhi[c], qhi[c]]
        world = s["corner"] + idx * s["vsize"]
        style.setdefault("color", (1.0, 1.0, 1.0, 1.0))
        style.setdefault("lit", False)
        return self.add_quads(world[None], sampler=sampler, **style)

    def set_dvr(self, sampler, step, ref_length=None, object_id=-1):
        """Direct volume rendering of a registered volume: the ray is marched through its box in
        steps of ``step`` (world units); the LUT alpha is the opacity accumulated over
        ``ref_length`` (default: one step).  Marching stops at opaque geometry.  ``None`` = off."""
        if sampler is None:
            self._dvr = None
            return
        if not 0 <= int(sampler) < len(self.samplers):
            raise ValueError(f"unknown sampler {sampler}: {len(self.samplers)} volume(s) registered")
        if not step > 0:
            raise ValueError(f"the ray-marching step must be positive, got {step!r}")
        self._dvr = (int(sampler), float(step), float(ref_length or step), int(object_id))

    # ---- Gaussians ----------------------------------------------------------------------------
    def add_gaussians(self, means, covs=None, opacity=0.8, color=(1.0, 0.5, 0.1), rotations=None, scales=None,
                      object_id=0):
        """3D Gaussians rendered as Gaussians: per ray ``alpha = opacity * exp(-d2/2)`` with d2 the
        smallest squared Mahalanobis distance between the ray and the mean, composited front to
        back in order of the point of maximum response.  Give ``covs`` (N,3,3) or ``rotations``
        (N,3,3, axes in columns) + ``scales`` (N,3 standard deviations).  ``opacity`` (the peak
        alpha, in [0, 1]) and ``color`` are one value or one per Gaussian."""
        mu = np.asarray(means, np.float64).reshape(-1, 3)
        if len(mu) == 0:
            raise ValueError("there are no Gaussians to draw")
        opac = np.asarray(opacity, np.float32)
        if opac.shape not in ((), (len(mu),)) or not np.all(np.isfinite(opac)) or opac.min() < 0.0 or opac.max() > 1.0:
            raise ValueError(f"opacity must be one number in [0, 1] or one per Gaussian ({len(mu)}), got {opacity!r}")
        if covs is not None:
            rotations, scales = cov_to_axes(_checked_covariances(covs, len(mu)))
        elif rotations is None or scales is None:
            raise ValueError("give either covs, or rotations and scales")
        rot = np.asarray(rotations, np.float64).reshape(-1, 3, 3)
        sig = np.broadcast_to(np.asarray(scales, np.float64), mu.shape)
        if len(rot) != len(mu) or not (np.all(np.isfinite(mu)) and np.all(np.isfinite(rot)) and np.all(np.isfinite(sig))):
            raise ValueError("Gaussian means, rotations and scales must be finite, one rotation per mean")
        if sig.size and sig.min() <= 0:
            raise ValueError("Gaussian scales (standard deviations) must be positive")
        w = np.einsum("na,nba->nab", 1.0 / sig, rot)                    # diag(1/sigma) R^T : world -> canonical
        ext = np.sqrt(np.einsum("nab,nb->na", rot ** 2, sig ** 2))       # sqrt(diag(cov)) = 1-sigma half extent
        rgbo = np.concatenate([_per(color, len(mu), 4)[:, :3], np.broadcast_to(opac, (len(mu),))[:, None]], axis=1)
        self._g.append((mu, w, ext, rgbo, np.broadcast_to(np.asarray(object_id, np.int32), (len(mu),))))
        self._lo, self._hi = np.minimum(self._lo, (mu - 3 * ext).min(0)), np.maximum(self._hi, (mu + 3 * ext).max(0))
        self._gpu = None
        return len(mu)

    # ---- the package's representations, in a display space ---------------------------------------
    # Every method below takes coordinates in TRUE PHYSICAL NANOMETRES (types.py) together with the
    # VoxelGrid and a display space, and does exactly one thing before calling a tessellator above:
    #     scene position = physical nm * grid.space_scale(space) / NM_PER_UNIT
    # space_scale is (1, 1, 1) for "physical" and (g/dx, g/dy, g/dz) for "index" (voxels drawn as
    # cubes of the in-plane pixel size g): that factor is the whole difference between the spaces.
    # It applies to every point of a drawn surface, so sizes are squashed with the positions: a
    # brick becomes a cube, a covariance S becomes A S A.  The one exception is the round tube of
    # a skeleton, which a USD curve cannot squash: see add_skeleton.
    def add_volume_grid(self, data_zyx, grid, space="physical", lut=None, labels=False, interp="nearest",
                        zero_alpha=0.0, focus=-1, grid_lines=0.0, grid_color=(1.0, 0.85, 0.0)):
        """Register ``data_zyx`` on a ``VoxelGrid`` (CORNER convention): voxel ``[k,j,i]`` is drawn as
        the box ``[i*dx, (i+1)*dx] x [j*dy, (j+1)*dy] x [k*dz, (k+1)*dz]`` with (dx, dy, dz) =
        ``grid.display_voxel_size_xyz_nm(space)`` converted to scene units, so the array fills
        ``[0, n*d]``.  "physical": the true voxel size; "index": cubes of the in-plane pixel size.
        ``grid_lines`` > 0 draws the voxel boundaries (width in scene units).  Other arguments and
        the returned sampler id are those of :meth:`add_volume`."""
        if tuple(np.shape(data_zyx)) != tuple(grid.shape_zyx):
            raise ValueError(f"the array has shape {np.shape(data_zyx)} but the grid describes {tuple(grid.shape_zyx)}")
        size_xyz = grid.display_voxel_size_xyz_nm(space) / NM_PER_UNIT      # one voxel as drawn, in scene units
        return self.add_volume(data_zyx, voxel_size=size_xyz[::-1], origin=(0.0, 0.0, 0.0), lut=lut, labels=labels,
                               interp=interp, zero_alpha=zero_alpha, focus=focus, grid=grid_lines,
                               grid_color=grid_color, convention="corner")

    def add_bricks(self, bricks, grid, space="physical", **style):
        """*instance* as voxel bricks (``types.Bricks``): every brick is exactly its voxel box, centre
        ``bricks.centers_xyz_nm`` and size ``bricks.size_xyz_nm`` moved into ``space``.  Only faces
        between a listed voxel and an unlisted one are tessellated.  ``color`` / ``object_id`` may
        hold one value per brick."""
        kji = np.asarray(bricks.kji, np.int64)
        ctr_nm = np.asarray(bricks.centers_xyz_nm, np.float64)
        size_nm = np.asarray(bricks.size_xyz_nm, np.float64)
        if kji.ndim != 2 or kji.shape[1] != 3 or ctr_nm.shape != kji.shape or size_nm.shape != (3,):
            raise ValueError(f"bricks need kji (N,3), centers_xyz_nm (N,3), size_xyz_nm (3,); got {kji.shape}, {ctr_nm.shape}, {size_nm.shape}")
        if len(kji) == 0:
            raise ValueError("there are no bricks to draw")
        if not np.all(np.isfinite(ctr_nm)) or not np.all(np.isfinite(size_nm)) or size_nm.min() <= 0:
            raise ValueError("brick centres must be finite and brick sizes positive")
        scale = grid.space_scale(space) / NM_PER_UNIT               # physical nm -> scene units in `space`
        ctr, size = ctr_nm * scale, size_nm * scale
        k0 = kji.min(0)
        number = np.full(tuple(kji.max(0) - k0 + 1), -1, np.int32)  # which brick sits in each voxel of the bounding box
        number[tuple((kji - k0).T)] = np.arange(len(kji))
        quads, owner = [], []
        for face, local in _exposed_faces(number >= 0):
            n = number[tuple(local.T)]
            quads.append(ctr[n][:, None, :] + 0.5 * size * _BOX_QUADS[face][None])
            owner.append(n)
        return self.add_quads(np.concatenate(quads), **_style_take(style, np.concatenate(owner)))

    def add_wiremesh(self, mesh, grid, space="physical", color=(0.8, 0.8, 0.8, 1.0), wire_color=(0.0, 0.0, 0.0, 1.0),
                     wire_width=0.0, wire_min_pixels=1.0, **style):
        """*wiremesh* (``types.Mesh``): the surface triangles with their edges drawn as lines.
        ``color`` is the fill (alpha 0 = wires only, front and back visible); the lines are
        ``wire_width`` scene units wide but at least ``wire_min_pixels`` on screen (display aid).
        Where the triangles of a dense mesh are about one pixel in size the lines fill the whole
        surface: pass ``wire_min_pixels=0`` with a ``wire_width``, or draw a simplified mesh."""
        if len(mesh.faces) == 0:
            raise ValueError("the mesh has no faces to draw")
        v = np.asarray(mesh.vertices_xyz_nm, np.float64) * (grid.space_scale(space) / NM_PER_UNIT)
        return self.add_mesh(v, mesh.faces, color=color, wire_color=wire_color, wire_width=wire_width,
                             wire_min_pixels=wire_min_pixels, **style)

    def add_skeleton(self, skeleton, grid, space="physical", radius=None, min_pixels=1.5, exact=False, **style):
        """*skeleton* (``types.Skeleton``): one tube per edge and a joint on every node.  ``radius``
        is the tube radius, a PHYSICAL length in scene units (micrometres), one value or one per
        node; ``None`` uses the measured ``skeleton.radii_nm`` (see ``types.Skeleton``).
        ``color`` / ``object_id`` may hold one value per edge.

        "physical" space: round tubes of that radius with a sphere on every node.

        "index" space: the nodes move like every other point, p -> A p with
        A = ``grid.space_scale(space)``.  A radius is a physical length and has no meaning there,
        so there are two pictures:

        ``exact=False`` (default): what the USD stage draws (``stage.Dataset.add_skeleton``, and
            therefore RTX): ROUND tubes, each edge with its width scaled by
            ``grid.tube_width_factor`` (the round tube with the volume of the exact image), and
            the node spheres as the ellipsoids A maps them to.  With 6 x 6 x 30 nm voxels a tube
            of 15 nm radius along x is drawn with 15 * sqrt(1/5) = 6.7 nm radius: narrower than
            the exact image across the section plane and higher along z, so it can stand out of
            its squashed instance.
        ``exact=True``: the exact image of the physical tubes: every point of their surface is
            multiplied by A, so the cross-sections become ellipses (that tube: 30 nm wide in y,
            6 nm high in z) and the skeleton stays inside its squashed instance.

        ``min_pixels`` keeps thin tubes visible at any zoom (display aid, see :meth:`add_tubes`)."""
        e = np.asarray(skeleton.edges, np.int64).reshape(-1, 2)
        if len(e) == 0:
            raise ValueError("the skeleton has no edges to draw")
        p_nm = np.asarray(skeleton.vertices_xyz_nm, np.float64).reshape(-1, 3)      # true physical positions
        if radius is None:
            radius = np.asarray(skeleton.radii_nm, np.float64) / NM_PER_UNIT
        a = grid.space_scale(space)                                                 # (1, 1, 1) in physical space
        if exact or space == "physical":
            return self.add_tubes(p_nm / NM_PER_UNIT, e, radius, caps="round", min_pixels=min_pixels, stretch=a, **style)
        if e.min() < 0 or e.max() >= len(p_nm):
            raise ValueError(f"edge indices must lie in 0..{len(p_nm) - 1}, got {e.min()}..{e.max()}")
        if np.ndim(radius) > 1 or (np.ndim(radius) == 1 and len(radius) != len(p_nm)):
            raise ValueError(f"radius must be one number or one per node ({len(p_nm)}), got shape {np.shape(radius)}")
        r = np.broadcast_to(np.asarray(radius, np.float64), (len(p_nm),))
        # one round tube per edge, as on the stage: both ends are the edge's own points, so that
        # an edge along x and an edge along z that meet in a node can have different widths there
        factor = grid.tube_width_factor(p_nm[e[:, 0]], p_nm[e[:, 1]], space)        # (E,)
        ends = (p_nm * a / NM_PER_UNIT)[e].reshape(-1, 3)                           # (2 E, 3): A p in scene units
        nf = self.add_tubes(ends, np.arange(2 * len(e)).reshape(-1, 2), (r[e] * factor[:, None]).reshape(-1),
                            caps="flat", min_pixels=min_pixels, **style)
        used = np.unique(e)
        owner = np.zeros(len(p_nm), np.int64)           # a joint takes the style of the first edge that uses it
        owner[e[::-1].ravel()] = np.repeat(np.arange(len(e))[::-1], 2)
        return nf + self.add_spheres(p_nm[used] / NM_PER_UNIT, r[used], subdiv=1, min_pixels=min_pixels, stretch=a,
                                     **_style_take(style, owner[used]))

    def add_gaussian_mixture(self, gaussians, grid, space="physical", opacity=0.8, color=None, object_id=0,
                             sigma_surface=None):
        """*gaussian* (``types.Gaussians``): means move as ``A*mu`` and covariances as ``A S A`` with
        A = diag(grid.space_scale(space)) -- never by one scalar.  Drawn as true Gaussians, or, with
        ``sigma_surface=k``, as their k-sigma ellipsoid surfaces (opaque, lit).  ``color`` defaults
        to ``gaussians.colors_rgb`` or, without it, to one hashed colour per component.
        ``opacity`` is the peak alpha, one value or one per component; ``gaussians.weights`` are
        not used."""
        mu = np.asarray(grid.to_space(gaussians.means_xyz_nm, space), np.float64).reshape(-1, 3) / NM_PER_UNIT
        if len(mu) == 0:
            raise ValueError("there are no Gaussians to draw")
        cov = _checked_covariances(gaussians.covariances_nm2, len(mu))
        cov = grid.covariance_to_space(cov, space) / NM_PER_UNIT ** 2
        if color is None:
            color = gaussians.colors_rgb if gaussians.colors_rgb is not None else label_colors(np.arange(len(mu)) + 1)
        if sigma_surface is None:
            return self.add_gaussians(mu, covs=cov, opacity=opacity, color=color, object_id=object_id)
        if not sigma_surface > 0:
            raise ValueError(f"sigma_surface must be positive, got {sigma_surface!r}")
        rot, sig = cov_to_axes(cov)
        return self.add_ellipsoids(mu, rot * (sigma_surface * sig)[:, None, :], subdiv=2, color=color, object_id=object_id)

    # ---- device side ----------------------------------------------------------------------------
    def bounds(self):
        """(lo, hi) of everything added so far."""
        return self._lo.copy(), self._hi.copy()

    @property
    def num_triangles(self):
        return int(sum(len(f) for f in self._f))

    @property
    def num_gaussians(self):
        return int(sum(len(g[0]) for g in self._g))

    def build(self, device=None, gaussian_cutoff=4.0):
        """Upload everything and build the acceleration structures (triangle BVH inside wp.Mesh,
        wp.Bvh over the ``gaussian_cutoff``-sigma boxes of the Gaussians).  ``device=None`` uses
        :func:`default_device`."""
        t0 = time.perf_counter()
        device = _device_name(device)
        g = dict(device=device, cutoff=float(gaussian_cutoff))
        cat = lambda parts, shape, dtype: (np.concatenate(parts) if parts else np.zeros(shape, dtype))
        arr = lambda a, dtype: wp.array(np.ascontiguousarray(a), dtype=dtype, device=device)
        g["has_mesh"] = int(self.num_triangles > 0)
        g["pts"] = arr(cat(self._v, (1, 3), np.float32), wp.vec3)
        g["tri"] = arr(cat(self._f, (1, 3), np.int32).reshape(-1), wp.int32)
        g["nrm"] = arr(cat(self._n, (1, 3), np.float32), wp.vec3)
        g["rgba"] = arr(cat(self._rgba, (1, 4), np.float32), wp.vec4)
        g["rgba2"] = arr(cat(self._rgba2, (1, 4), np.float32), wp.vec4)
        g["mat"] = arr(cat(self._mat, (1,), np.int32), wp.int32)
        g["obj"] = arr(cat(self._obj, (1,), np.int32), wp.int32)
        g["aux"] = arr(cat(self._aux, (1, 4), np.float32), wp.vec4)
        g["mesh"] = wp.Mesh(g["pts"], g["tri"]) if g["has_mesh"] else None
        g["inflate"] = None                                 # vertices with a minimum on-screen size
        if self._inflate and g["has_mesh"]:
            idx, ctr, dirs, rad, mpx = (np.concatenate([part[i] for part in self._inflate]) for i in range(5))
            g["inflate"] = (arr(idx.astype(np.int32), wp.int32), arr(ctr.astype(np.float32), wp.vec3),
                            arr(dirs.astype(np.float32), wp.vec3), arr(rad.astype(np.float32), wp.float32),
                            arr(mpx.astype(np.float32), wp.float32))
        if self._vol_gpu is None or self._vol_gpu[0] != (device, tuple(self._vol_size)):   # voxels are uploaded once
            self._vol_gpu = ((device, tuple(self._vol_size)),
                             tuple(arr(cat(self._vol[i], (1,), t), w)
                                   for i, (t, w) in enumerate(((np.uint8, wp.uint8), (np.uint16, wp.uint16), (np.int32, wp.int32)))))
        g["vol"] = self._vol_gpu[1]
        g["luts"] = arr(np.stack(self._luts) if self._luts else np.zeros((1, 256, 4), np.float32), wp.vec4)
        smp = []
        for s in self.samplers or [dict(corner=np.zeros(3), shape=(1, 1, 1), store=0, offset=0, labels=0, linear=0, lut=0,
                                        zero_alpha=0.0, focus=-1, grid=0.0, grid_rgb=(0, 0, 0), vsize=np.ones(3))]:
            c2w = np.diag([*s["vsize"], 1.0])               # index (corner convention) -> world
            c2w[:3, 3] = s["corner"]
            k = _Sampler()
            k.w2i = wp.mat44(np.linalg.inv(c2w).astype(np.float32))
            k.vsize = wp.vec3(*[float(x) for x in s["vsize"]])
            k.store, k.offset, (k.nz, k.ny, k.nx) = s["store"], s["offset"], s["shape"]
            k.labels, k.linear, k.lut = s["labels"], s["linear"], s["lut"]
            k.zero_alpha, k.focus, k.grid = float(s["zero_alpha"]), int(s["focus"]), float(s["grid"])
            k.grid_rgb = wp.vec3(*[float(x) for x in s["grid_rgb"]])
            smp.append(k)
        g["smp"] = wp.array(smp, dtype=_Sampler, device=device)
        g["n_gauss"] = self.num_gaussians
        if self._g:
            mu, w, ext, rgbo, obj = (np.concatenate([p[i] for p in self._g]) for i in range(5))
        else:
            mu, w, ext, rgbo, obj = np.zeros((1, 3)), np.eye(3)[None], np.ones((1, 3)), np.zeros((1, 4)), np.zeros(1)
        g["g_mean"], g["g_w"] = arr(mu.astype(np.float32), wp.vec3), arr(w.astype(np.float32), wp.mat33)
        g["g_rgbo"], g["g_obj"] = arr(rgbo.astype(np.float32), wp.vec4), arr(obj.astype(np.int32), wp.int32)
        g["g_lo"] = arr((mu - gaussian_cutoff * ext).astype(np.float32), wp.vec3)
        g["g_hi"] = arr((mu + gaussian_cutoff * ext).astype(np.float32), wp.vec3)
        g["g_bvh"] = wp.Bvh(g["g_lo"], g["g_hi"]) if self._g else None
        wp.synchronize_device(device)
        self._gpu, self.build_seconds = g, time.perf_counter() - t0
        return self

    def render(self, camera, device=None, max_hits=16, ambient=0.3, min_transmittance=1.0e-3, depth_alpha=0.0,
               gaussian_min_alpha=1.0 / 255.0, hemisphere=0.0, fog=0.0, fog_range=None):
        """Render with ``camera``; returns an :class:`Image`.  ``max_hits`` bounds the number of
        successive (translucent) surface hits composited per ray; ``depth_alpha`` is the alpha a
        surface must exceed to be written to the depth / object-id buffers.

        Lighting of lit surfaces: ``ambient + (1 - ambient) * |n . view|`` (a headlight).  Two
        optional depth cues, both off by default so that colours stay exact:

        ``hemisphere`` in [0, 1]: the ambient term becomes ``ambient * (1 - hemisphere * (1 - n.up)/2)``
            with n the normal facing the viewer and up the camera's up vector: surfaces facing down
            in the picture get less ambient light (none at all for hemisphere = 1).
        ``fog`` in [0, 1]: every colour is blended toward the background colour by
            ``fog * clamp((t - near) / (far - near), 0, 1)`` with t its depth along the ray.
            ``fog_range=(near, far)``; default: the nearest and farthest corner of the scene box
            (:meth:`bounds`, plus the box of a ray-marched volume)."""
        t_wall = time.perf_counter()
        camera.check()
        if not (0.0 <= ambient <= 1.0 and 0.0 <= hemisphere <= 1.0 and 0.0 <= fog <= 1.0):
            raise ValueError(f"ambient, hemisphere and fog must lie in [0, 1], got {ambient!r}, {hemisphere!r}, {fog!r}")
        if int(max_hits) < 1:
            raise ValueError(f"max_hits must be at least 1, got {max_hits!r}")
        if fog_range is not None and not float(fog_range[1]) > float(fog_range[0]):
            raise ValueError(f"fog_range must be (near, far) with far > near, got {fog_range!r}")
        if self._gpu is None or (device is not None and _device_name(device) != self._gpu["device"]):
            self.build(device)
        g = self._gpu
        device = g["device"]
        right, up, fwd = camera.basis()
        cam = _Cam()
        cam.eye, cam.right, cam.up, cam.fwd = (wp.vec3(*[float(x) for x in v]) for v in (camera.eye, right, up, fwd))
        cam.ortho = int(camera.mode != "perspective")
        span = camera.ortho_height if cam.ortho else 2.0 * math.tan(math.radians(camera.fov) / 2)
        cam.span_x, cam.span_y = span * camera.width / camera.height, span
        cam.width, cam.height, cam.ss = int(camera.width), int(camera.height), int(camera.ssaa)
        bg = tuple(camera.background) + (1.0,) * (4 - len(camera.background))
        cam.bg = wp.vec4(*[float(x) for x in bg])
        rp = _Params()
        rp.has_mesh, rp.max_hits, rp.ambient, rp.hemi = g["has_mesh"], int(max_hits), float(ambient), float(hemisphere)
        rp.max_steps = max(8 * int(max_hits), 256) + 1
        lo, hi = self.bounds()
        if self._dvr:                                       # the ray-marched volume is part of the scene too
            vlo, vhi = self.volume_corners(self._dvr[0])
            lo, hi = np.minimum(lo, vlo), np.maximum(hi, vhi)
        have_bounds = bool(np.all(np.isfinite(lo)))
        diag = float(np.linalg.norm(hi - lo)) if have_bounds else 1.0
        rp.eps, rp.t_min, rp.depth_alpha = 1.0e-5 * diag, float(min_transmittance), float(depth_alpha)
        rp.n_gauss, rp.g_cut2, rp.g_amin = g["n_gauss"], g["cutoff"] ** 2, float(gaussian_min_alpha)
        rp.dvr, rp.dvr_step, rp.dvr_ref, rp.dvr_id = self._dvr if self._dvr else (-1, 1.0, 1.0, -1)
        near, far = (0.0, 1.0) if fog_range is None else (float(fog_range[0]), float(fog_range[1]))
        if fog > 0 and fog_range is None and have_bounds:   # depth of the 8 corners of the scene box
            rel = box_edges(lo, hi)[0] - np.asarray(camera.eye, float)
            t = rel @ fwd if cam.ortho else np.linalg.norm(rel, axis=1)
            near, far = float(t.min()), float(t.max())
        rp.fog, rp.fog_near, rp.fog_inv = float(fog), near, 1.0 / max(far - near, 1e-20)
        if g["inflate"] is not None:                        # minimum on-screen size of tubes and spheres, for this camera
            wp.launch(_inflate_kernel, dim=len(g["inflate"][0]), device=device, inputs=[cam, *g["inflate"]], outputs=[g["pts"]])
            g["mesh"].refit()
        shape = (int(camera.height), int(camera.width))
        out = (wp.empty(shape, dtype=wp.vec4, device=device), wp.empty(shape, dtype=wp.float32, device=device),
               wp.empty(shape, dtype=wp.int32, device=device))
        t0 = time.perf_counter()
        wp.launch(_render_kernel, dim=shape, device=device, outputs=list(out),
                  inputs=[cam, rp, g["mesh"].id if g["mesh"] else wp.uint64(0), g["pts"], g["tri"], g["nrm"],
                          g["rgba"], g["rgba2"], g["mat"], g["obj"], g["aux"], *g["vol"], g["smp"], g["luts"],
                          g["g_bvh"].id if g["g_bvh"] else wp.uint64(0), g["g_mean"], g["g_w"], g["g_rgbo"],
                          g["g_obj"]])
        wp.synchronize_device(device)
        seconds = time.perf_counter() - t0
        rgbaf, depth, oid = (o.numpy() for o in out)        # read back to host memory
        x = rgbaf.copy()
        if bg[3] < 1.0:                                     # un-premultiply for a straight-alpha PNG
            x[..., :3] /= np.maximum(x[..., 3:], 1e-6)
        np.clip(x, 0.0, 1.0, out=x)                         # 8 bit: floor(255 * v + 0.5)
        x *= 255.0
        x += 0.5
        depth[depth >= _FAR] = np.inf
        return Image(x.astype(np.uint8), depth, oid, rgbaf, seconds, time.perf_counter() - t_wall)

    def render_timed(self, camera, repeat=1, **render_options):
        """Render and return ``(image, seconds)`` where seconds is the honest wall-clock time of one
        frame as a notebook user experiences it: kernel launch, waiting for the device, read-back
        of colour, depth and object id to numpy, and the 8-bit conversion.  The scene is uploaded
        and its BVH built BEFORE the clock starts (that time is ``scene.build_seconds``).  With
        ``repeat`` > 1 the frame is rendered that often and the fastest is reported (the first
        frame of a process also pays for compiling / loading the kernel)."""
        device = render_options.get("device")
        if self._gpu is None or (device is not None and _device_name(device) != self._gpu["device"]):
            self.build(device)
        best = None
        for _ in range(max(1, int(repeat))):
            t0 = time.perf_counter()
            image = self.render(camera, **render_options)
            seconds = time.perf_counter() - t0
            best = seconds if best is None else min(best, seconds)
        return image, best
