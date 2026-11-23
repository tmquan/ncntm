"""emray -- a small CUDA ray caster (NVIDIA Warp) for teaching EM-connectomics representations.

It draws, exactly and with voxel anisotropy handled in the open, the six representations
of an electron-microscopy volume:

    rawvalue  the uint8 number as stored, in array index space      -> volume-sampled slices
    voxvalue  the number normalised to [0,1] on a physical voxel     -> slices + window/level LUT
    instance  voxel label = instance id                              -> label slices, voxel bricks
    skeleton  nodes + edges                                          -> tubes + spheres
    wiremesh  wireframe surface mesh                                 -> wire material / tubes
    gaussian  3D Gaussians                                           -> true Gaussians or k-sigma ellipsoids

Dependencies: ``warp`` and ``numpy`` only.  No Kit app, no RTX, no graphics context.

Conventions (the whole module follows them)
-------------------------------------------
* Volumes are numpy arrays indexed ``[k, j, i]`` = ``[z, y, x]``.  Voxel sizes are given in the
  same order, ``(dz, dy, dx)``.  Points, vectors and matrices are in world order ``(x, y, z)``.
* Voxel ``[k, j, i]`` has its CENTRE at ``origin + (i*dx, j*dy, k*dz)`` and occupies the box of
  size ``(dx, dy, dz)`` around it (so the volume spans ``-0.5 .. n-0.5`` voxels on each axis).
  "Index space" is simply ``voxel_size = (1, 1, 1)``; "physical space" is e.g. ``(0.030, 0.006, 0.006)``
  micrometres.  Anisotropy never hides inside the renderer: it enters only through the 4x4
  index-to-world matrix of a volume sampler and through the coordinates you hand to the tessellators.
* Image row 0 is the top of the picture; ``depth`` is the distance t along the unit ray
  (perspective: distance from the eye; orthographic: distance from the eye plane).
* Colours are display-referred (no gamma step), so an unlit material reproduces its input exactly.

Typical use
-----------
    scene = Scene()
    s = scene.add_volume(em, voxel_size=(0.030, 0.006, 0.006), lut=lut_grey())
    scene.add_slice(s, "z", 50, object_id=1)
    scene.add_tubes(nodes, edges, radii, color=(1, 0.8, 0), object_id=2)
    cam = Camera.fit(*scene.bounds(), direction=(1, -1, 0.6))
    img = scene.render(cam, device="cuda:0")          # img.rgba, img.depth, img.object_id
    img.save("picture.png")
"""
from __future__ import annotations

import functools
import math
import struct
import time
import zlib
from dataclasses import dataclass

import numpy as np
import warp as wp

__all__ = ["Scene", "Camera", "Image", "index_to_world", "lut_grey", "lut_window_level", "lut_membrane",
           "label_colors", "icosphere", "unique_edges", "box_edges", "cov_to_axes", "save_png", "encode_png",
           "MAT_LIT", "MAT_UNLIT", "MAT_VOLUME", "MAT_WIRE"]

MAT_LIT, MAT_UNLIT, MAT_VOLUME, MAT_WIRE = 0, 1, 2, 3   # per-face material ids
GAUSS_BUF = 64          # Gaussians kept sorted per ray per pass (more are handled by further passes)
_FAR = 1.0e30
_vecf = wp.types.vector(length=GAUSS_BUF, dtype=wp.float32)
_veci = wp.types.vector(length=GAUSS_BUF, dtype=wp.int32)


# --------------------------------------------------------------------------------------------
# numpy helpers: voxel grids, lookup tables, label colours, small geometry utilities
# --------------------------------------------------------------------------------------------
def index_to_world(voxel_size, origin=(0.0, 0.0, 0.0)):
    """4x4 matrix taking an index vector ``(i, j, k, 1)`` to world ``(x, y, z, 1)``.

    ``voxel_size`` is ``(dz, dy, dx)`` (array order), ``origin`` is the world position (x, y, z)
    of the centre of voxel ``[0, 0, 0]``.
    """
    dz, dy, dx = (float(v) for v in voxel_size)
    m = np.diag([dx, dy, dz, 1.0])
    m[:3, 3] = origin
    return m


def lut_grey():
    """Identity lookup table for *rawvalue*: stored number v -> grey v/255, opaque."""
    g = np.arange(256, dtype=np.float32) / 255.0
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
    """Unit icosphere: (vertices (V,3), faces (F,3)); F = 20 * 4**subdiv."""
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


def encode_png(rgba):
    """An (H,W,3|4) uint8 array as PNG bytes (pure zlib, no imaging library needed)."""
    a = np.ascontiguousarray(rgba, dtype=np.uint8)
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

    def basis(self):
        """(right, up, forward) unit vectors; image x runs along right, image rows run along -up."""
        f = np.asarray(self.target, float) - np.asarray(self.eye, float)
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(self.up, float))
        r /= np.linalg.norm(r)
        return r, np.cross(r, f), f

    @classmethod
    def fit(cls, lo, hi, direction=(1.0, -1.0, 0.6), margin=1.05, mode="perspective", fov=35.0, **kw):
        """Frame the box lo..hi: look at its centre from ``direction`` (centre -> eye) so that the
        bounding sphere of the box just fills the image.  One rule for every scene, which is what
        makes index-space and physical-space pictures comparable."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c, r = 0.5 * (lo + hi), 0.5 * float(np.linalg.norm(hi - lo)) * margin
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        dist = r / math.sin(math.radians(fov) / 2) if mode == "perspective" else 4.0 * r
        return cls(eye=tuple(c + d * dist), target=tuple(c), mode=mode, fov=fov, ortho_height=2.0 * r, **kw)

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
    (premultiplied if the background is transparent), ``seconds`` render time.  With supersampling,
    depth and object id are those of the nearest sub-sample of the pixel.  Shows inline in Jupyter."""
    rgba: np.ndarray
    depth: np.ndarray
    object_id: np.ndarray
    rgba_float: np.ndarray
    seconds: float = 0.0

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
    w2i: wp.mat44        # world (x,y,z) -> continuous index (i,j,k); integers are voxel centres
    vsize: wp.vec3       # voxel size along x,y,z (world units), for grid lines
    offset: wp.int32     # start of this volume in the packed voxel array
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
def _voxel(vol: wp.array(dtype=wp.int32), s: _Sampler, i: int, j: int, k: int):
    ii = wp.clamp(i, 0, s.nx - 1)
    jj = wp.clamp(j, 0, s.ny - 1)
    kk = wp.clamp(k, 0, s.nz - 1)
    return vol[s.offset + (kk * s.ny + jj) * s.nx + ii]


@wp.func
def _sample(vol: wp.array(dtype=wp.int32), luts: wp.array2d(dtype=wp.vec4), s: _Sampler, p: wp.vec3):
    """RGBA of the volume at world point p (alpha 0 outside the volume)."""
    q = wp.transform_point(s.w2i, p)
    if (q[0] < -0.5 or q[1] < -0.5 or q[2] < -0.5 or q[0] > float(s.nx) - 0.5 or q[1] > float(s.ny) - 0.5
            or q[2] > float(s.nz) - 0.5):
        return wp.vec4(0.0)
    i = int(wp.floor(q[0] + 0.5))       # nearest voxel = round to the nearest centre
    j = int(wp.floor(q[1] + 0.5))
    k = int(wp.floor(q[2] + 0.5))
    c = wp.vec4(0.0)
    if s.labels == 1:
        label = _voxel(vol, s, i, j, k)
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
            v = float(_voxel(vol, s, i, j, k))
        else:
            i0 = int(wp.floor(q[0]))
            j0 = int(wp.floor(q[1]))
            k0 = int(wp.floor(q[2]))
            fx = q[0] - float(i0)
            fy = q[1] - float(j0)
            fz = q[2] - float(k0)
            for dk in range(2):
                for dj in range(2):
                    for di in range(2):
                        w = (wp.abs(1.0 - float(di) - fx) * wp.abs(1.0 - float(dj) - fy)
                             * wp.abs(1.0 - float(dk) - fz))
                        v += w * float(_voxel(vol, s, i0 + di, j0 + dj, k0 + dk))
        v = wp.clamp(v, 0.0, 255.0)
        l0 = int(wp.floor(v))
        l1 = wp.min(l0 + 1, 255)
        f = v - float(l0)
        c = luts[s.lut, l0] * (1.0 - f) + luts[s.lut, l1] * f
    if s.grid > 0.0:                    # lines on voxel boundaries, width in world units
        for a in range(3):
            fr = q[a] + 0.5 - wp.floor(q[a] + 0.5)
            if wp.min(fr, 1.0 - fr) * s.vsize[a] < 0.5 * s.grid:
                c = wp.vec4(s.grid_rgb[0], s.grid_rgb[1], s.grid_rgb[2], wp.max(c[3], 1.0))
    return c


@wp.func
def _ray_box(s: _Sampler, o: wp.vec3, d: wp.vec3):
    """Entry and exit t of the ray through the box of a volume (t is the same in index space)."""
    oi = wp.transform_point(s.w2i, o)
    di = wp.transform_vector(s.w2i, d)
    hi = wp.vec3(float(s.nx) - 0.5, float(s.ny) - 0.5, float(s.nz) - 0.5)
    t0 = float(0.0)
    t1 = float(_FAR)
    for a in range(3):
        if wp.abs(di[a]) < 1.0e-12:
            if oi[a] < -0.5 or oi[a] > hi[a]:
                t1 = float(-1.0)
        else:
            ta = (-0.5 - oi[a]) / di[a]
            tb = (hi[a] - oi[a]) / di[a]
            t0 = wp.max(t0, wp.min(ta, tb))
            t1 = wp.min(t1, wp.max(ta, tb))
    return t0, t1


@wp.func
def _edge_dist(p: wp.vec3, a: wp.vec3, b: wp.vec3):
    e = b - a
    return wp.length(wp.cross(e, p - a)) / wp.max(wp.length(e), 1.0e-20)


@wp.kernel
def _render_kernel(cam: _Cam, rp: _Params, mesh: wp.uint64,
                   pts: wp.array(dtype=wp.vec3), tri: wp.array(dtype=wp.int32), nrm: wp.array(dtype=wp.vec3),
                   f_rgba: wp.array(dtype=wp.vec4), f_rgba2: wp.array(dtype=wp.vec4),
                   f_mat: wp.array(dtype=wp.int32), f_obj: wp.array(dtype=wp.int32),
                   f_aux: wp.array(dtype=wp.vec4),
                   vol: wp.array(dtype=wp.int32), smp: wp.array(dtype=_Sampler), luts: wp.array2d(dtype=wp.vec4),
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
                        sv = _sample(vol, luts, smp[rp.dvr], o + d * v_t)
                        ma = 1.0 - wp.pow(1.0 - wp.min(sv[3], 0.999), rp.dvr_step / rp.dvr_ref)
                        mc = wp.vec3(sv[0], sv[1], sv[2])
                        mt = v_t
                        v_t += rp.dvr_step
                        if v_t >= v_end:
                            v_t = float(_FAR)
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
                lam = rp.ambient + (1.0 - rp.ambient) * wp.abs(wp.dot(n, d))    # two-sided headlight
                c = f_rgba[face]
                aux = f_aux[face]
                mat = f_mat[face]
                rgb = wp.vec3(c[0], c[1], c[2])
                a = c[3]
                if mat == MAT_LIT:
                    rgb = rgb * lam
                elif mat == MAT_VOLUME:
                    sv = _sample(vol, luts, smp[int(aux[0])], p)
                    rgb = wp.vec3(sv[0] * c[0], sv[1] * c[1], sv[2] * c[2])
                    a = sv[3] * c[3]
                    if aux[2] > 0.5:
                        rgb = rgb * lam
                elif mat == MAT_WIRE:
                    mask = int(aux[1])
                    dist = float(_FAR)
                    if (mask & 1) != 0:
                        dist = wp.min(dist, _edge_dist(p, p0, p1))
                    if (mask & 2) != 0:
                        dist = wp.min(dist, _edge_dist(p, p1, p2))
                    if (mask & 4) != 0:
                        dist = wp.min(dist, _edge_dist(p, p2, p0))
                    if dist < 0.5 * aux[0]:
                        c2 = f_rgba2[face]
                        rgb = wp.vec3(c2[0], c2[1], c2[2])
                        a = c2[3]
                        if aux[3] > 0.5:
                            rgb = rgb * lam
                    elif aux[2] > 0.5:
                        rgb = rgb * lam
                if a > 0.0:
                    n_hit += 1
                    if hit_t >= _FAR and a > rp.depth_alpha:
                        hit_t = t_s
                        hit_id = f_obj[face]
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
    """Broadcast a colour-like value to (n, width); a missing alpha becomes 1."""
    a = np.asarray(value, np.float32)
    if a.ndim == 1:
        a = np.broadcast_to(a, (n, a.shape[0]))
    if a.shape[1] == width - 1:
        a = np.concatenate([a, np.ones((len(a), 1), np.float32)], axis=1)
    return a.astype(np.float32)


class Scene:
    """Everything that is drawn.  Opaque/translucent geometry is ONE merged triangle soup (one
    ``wp.Mesh``); every ``add_*`` tessellator appends to it.  Style keywords shared by all of them:

    ``color``       RGB or RGBA, one value or one per primitive (alpha < 1 = translucent)
    ``object_id``   integer written to the object-id buffer, one value or one per primitive
    ``lit``         True: headlight Lambert + ambient, two-sided (material 0); False: flat (material 1)
    ``sampler``     id returned by :meth:`add_volume`: colour comes from the volume (material 2)
    ``wire_width``  > 0: draw the triangle edges as lines of this width in world units (material 3);
                    ``wire_color`` is the line colour, ``color`` the fill (alpha 0 = pure wireframe),
                    ``wire_lit`` shades the lines too
    """

    def __init__(self):
        self._vol, self._vol_ids, self._vol_size = [], {}, 0
        self.samplers, self._luts = [], []
        self._dvr = None
        self._vol_gpu = None
        self.clear_geometry()

    def clear_geometry(self):
        """Drop all triangles and Gaussians but keep the registered volumes (they stay on the GPU):
        the cheap way to redraw, e.g. to move a slice from a notebook slider."""
        self._v, self._n, self._f = [], [], []
        self._rgba, self._rgba2, self._mat, self._obj, self._aux = [], [], [], [], []
        self._nv, self._g = 0, []
        self._lo, self._hi = np.full(3, np.inf), np.full(3, -np.inf)
        self._gpu = None
        self.build_seconds = 0.0

    # ---- low level ----------------------------------------------------------------------------
    def _emit(self, verts, faces, normals=None, prim=None, edge_mask=7, *, color=(0.8, 0.8, 0.8, 1.0),
              object_id=0, lit=True, sampler=-1, wire_width=0.0, wire_color=(0.0, 0.0, 0.0, 1.0),
              wire_lit=False):
        """Append triangles.  ``prim`` gives, per face, the index of the primitive it belongs to;
        per-primitive ``color`` / ``object_id`` arrays are expanded through it."""
        verts = np.asarray(verts, np.float64).reshape(-1, 3)
        faces = np.asarray(faces, np.int64).reshape(-1, 3)
        nf = len(faces)
        if nf == 0:
            return 0
        prim = np.arange(nf) if prim is None else np.asarray(prim)
        npr = int(prim.max()) + 1
        rgba = _per(color, npr, 4)[prim]
        aux = np.zeros((nf, 4), np.float32)
        mat = MAT_LIT if lit else MAT_UNLIT
        if sampler >= 0:
            mat, aux[:, 0], aux[:, 2] = MAT_VOLUME, sampler, float(bool(lit))
        elif wire_width > 0:
            mat, aux[:, 0], aux[:, 1], aux[:, 2], aux[:, 3] = MAT_WIRE, wire_width, edge_mask, float(bool(lit)), float(wire_lit)
        self._v.append(verts.astype(np.float32))
        self._n.append(np.zeros_like(verts, dtype=np.float32) if normals is None else np.asarray(normals, np.float32))
        self._f.append((faces + self._nv).astype(np.int32))
        self._rgba.append(rgba)
        self._rgba2.append(_per(wire_color, npr, 4)[prim])
        self._mat.append(np.full(nf, mat, np.int32))
        self._obj.append(np.broadcast_to(np.asarray(object_id, np.int32), (npr,))[prim] if np.ndim(object_id)
                         else np.full(nf, object_id, np.int32))
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
        q = np.asarray(corners, np.float64).reshape(-1, 4, 3)
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
        for key in ("color", "object_id", "wire_color"):
            if key in style and np.ndim(style[key]) == (1 if key == "object_id" else 2):
                style[key] = np.repeat(np.asarray(style[key]), 6, axis=0)
        return self.add_quads(quads, **style)

    def add_voxels(self, mask, voxel_size, origin=(0.0, 0.0, 0.0), **style):
        """Voxel bricks of a boolean volume ``mask[z,y,x]``: only faces between a set voxel and an
        unset one are emitted, each as a quad of the true (anisotropic) voxel face size.  ``color``
        may be a function ``f(kji) -> (n,3|4)`` of the (n,3) voxel indices."""
        m = np.pad(np.asarray(mask, bool), 1)
        size = np.asarray(voxel_size, np.float64)[::-1]          # (dx, dy, dz)
        quads, kjis = [], []
        for axis in range(3):                                   # array axis: 0=z, 1=y, 2=x
            for si, sgn in enumerate((-1, 1)):
                kji = np.argwhere(m & ~np.roll(m, -sgn, axis=axis)) - 1
                ctr = kji[:, ::-1] * size + np.asarray(origin, np.float64)
                quads.append(ctr[:, None, :] + 0.5 * size * _BOX_QUADS[(2 - axis) * 2 + si][None])
                kjis.append(kji)
        if callable(style.get("color")):
            style["color"] = style["color"](np.concatenate(kjis))
        return self.add_quads(np.concatenate(quads), **style)

    def add_tubes(self, points, edges, radii, sides=10, **style):
        """Truncated cones along the edges (E,2) of a point set, radius per point (or one value):
        skeleton edges, wire edges, axes.  Smooth-shaded, open-ended (put spheres on the nodes)."""
        p = np.asarray(points, np.float64).reshape(-1, 3)
        e = np.asarray(edges, np.int64).reshape(-1, 2)
        r = np.broadcast_to(np.asarray(radii, np.float64), (len(p),))
        ax = p[e[:, 1]] - p[e[:, 0]]
        ln = np.linalg.norm(ax, axis=1)
        ok = ln > 1e-12
        for key in ("color", "object_id", "wire_color"):
            if key in style and np.ndim(style[key]) == (1 if key == "object_id" else 2):
                style[key] = np.asarray(style[key])[ok]
        e, ax = e[ok], ax[ok] / ln[ok, None]
        helper = np.where(np.abs(ax[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        u = np.cross(ax, helper)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        w = np.cross(ax, u)
        th = np.arange(sides) * (2 * np.pi / sides)
        ring = np.cos(th)[None, :, None] * u[:, None, :] + np.sin(th)[None, :, None] * w[:, None, :]
        va = p[e[:, 0]][:, None, :] + r[e[:, 0]][:, None, None] * ring
        vb = p[e[:, 1]][:, None, :] + r[e[:, 1]][:, None, None] * ring
        s, s1 = np.arange(sides), (np.arange(sides) + 1) % sides
        local = np.concatenate([np.stack([s, s1, sides + s], 1), np.stack([s1, sides + s1, sides + s], 1)])
        faces = (np.arange(len(e))[:, None, None] * 2 * sides + local[None]).reshape(-1, 3)
        return self._emit(np.concatenate([va, vb], 1).reshape(-1, 3), faces,
                          np.concatenate([ring, ring], 1).reshape(-1, 3), np.repeat(np.arange(len(e)), 2 * sides),
                          **style)

    def add_spheres(self, centers, radii, subdiv=2, **style):
        """Icospheres (skeleton nodes): centres (N,3), radius per sphere or one value."""
        return self.add_ellipsoids(centers, None, subdiv=subdiv, radii=radii, **style)

    def add_ellipsoids(self, centers, matrices, subdiv=3, radii=None, **style):
        """Ellipsoids ``centre + M @ unit_sphere`` with M (N,3,3) -- e.g. the k-sigma surface of a
        Gaussian is ``M = R @ diag(k * sigma)`` (see :func:`cov_to_axes`)."""
        c = np.asarray(centers, np.float64).reshape(-1, 3)
        sv, sf = icosphere(subdiv)
        if matrices is None:
            m = np.broadcast_to(np.asarray(radii, np.float64), (len(c),))[:, None, None] * np.eye(3)[None]
        else:
            m = np.asarray(matrices, np.float64).reshape(-1, 3, 3)
        verts = c[:, None, :] + np.einsum("nab,vb->nva", m, sv)
        normals = np.einsum("nba,vb->nva", np.linalg.inv(m), sv)        # M^-T v
        normals /= np.linalg.norm(normals, axis=2, keepdims=True)
        faces = (np.arange(len(c))[:, None, None] * len(sv) + sf[None]).reshape(-1, 3)
        return self._emit(verts.reshape(-1, 3), faces, normals.reshape(-1, 3), np.repeat(np.arange(len(c)), len(sf)),
                          **style)

    # ---- volumes ------------------------------------------------------------------------------
    def add_volume(self, data, voxel_size=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), lut=None, labels=False,
                   interp="nearest", zero_alpha=0.0, focus=-1, grid=0.0, grid_color=(1.0, 0.85, 0.0)):
        """Register a volume ``data[z,y,x]`` (integer values; uint8 for EM, ids for labels) placed in
        the world by ``voxel_size=(dz,dy,dx)`` and ``origin``; returns a sampler id.  The same
        array may be registered several times (index space, physical space, other LUTs): the voxels
        are stored once on the GPU.  ``lut`` is a (256,4) RGBA table for scalar volumes (default
        identity grey); ``labels=True`` colours ids through the hash instead (id 0 gets alpha
        ``zero_alpha``; ``focus`` keeps one id coloured and greys the rest).  ``interp`` is
        "nearest" (crisp voxels) or "linear".  ``grid`` > 0 draws voxel boundaries with that width."""
        key = id(data)
        if key not in self._vol_ids:
            arr = np.ascontiguousarray(data)
            if arr.dtype.kind == "f":                   # voxvalue as floats in [0, 1]: stored as 0..255
                if labels:
                    raise TypeError("label volumes must be integer arrays, got %s" % arr.dtype)
                if arr.size and (arr.min() < 0.0 or arr.max() > 1.0):
                    raise ValueError("float volumes must be normalised to [0, 1] (got %g .. %g)" % (arr.min(), arr.max()))
                arr = np.rint(arr.astype(np.float64) * 255.0)
            elif arr.dtype.kind not in "iub":
                raise TypeError("unsupported volume dtype %s" % arr.dtype)
            elif arr.size and (int(arr.min()) < -2 ** 31 or int(arr.max()) > 2 ** 31 - 1):
                raise ValueError("volume values do not fit into int32")
            arr = arr.astype(np.int32)
            self._vol_ids[key] = (self._vol_size, arr.shape, data)      # keeps `data` alive so its id stays unique
            self._vol.append(arr.reshape(-1))
            self._vol_size += arr.size
        offset, shape = self._vol_ids[key][:2]
        self._luts.append(lut_grey() if lut is None else np.asarray(lut, np.float32).reshape(256, 4))
        i2w = index_to_world(voxel_size, origin)
        self.samplers.append(dict(i2w=i2w, shape=shape, offset=offset, labels=int(labels),
                                  linear=int(interp == "linear"), lut=len(self._luts) - 1, zero_alpha=zero_alpha,
                                  focus=focus, grid=grid, grid_rgb=grid_color, vsize=np.diag(i2w)[:3]))
        self._gpu = None
        return len(self.samplers) - 1

    def volume_corners(self, sampler):
        """World-space (lo, hi) corners of the box occupied by a registered volume."""
        s = self.samplers[sampler]
        n = np.array(s["shape"][::-1], float)
        return s["i2w"][:3, 3] - 0.5 * s["vsize"], s["i2w"][:3, 3] + (n - 0.5) * s["vsize"]

    def add_slice(self, sampler, axis, index, lo=None, hi=None, **style):
        """One slice of a registered volume as a volume-sampled quad at its true position.
        ``axis`` is "z", "y" or "x" (array axis 0, 1, 2), ``index`` the slice number; ``lo``/``hi``
        optionally restrict it to the voxel range ``[lo, hi)`` given as (k, j, i)."""
        s = self.samplers[sampler]
        a = "xyz".index(axis) if isinstance(axis, str) else 2 - int(axis)       # world axis of the normal
        n = np.array(s["shape"][::-1], float)                                   # (nx, ny, nz)
        qlo = -0.5 * np.ones(3) if lo is None else np.asarray(lo, float)[::-1] - 0.5
        qhi = n - 0.5 if hi is None else np.asarray(hi, float)[::-1] - 0.5
        b, c = (a + 1) % 3, (a + 2) % 3
        idx = np.zeros((4, 3))
        idx[:, a] = index
        idx[:, b] = [qlo[b], qhi[b], qhi[b], qlo[b]]
        idx[:, c] = [qlo[c], qlo[c], qhi[c], qhi[c]]
        world = idx @ s["i2w"][:3, :3].T + s["i2w"][:3, 3]
        style.setdefault("color", (1.0, 1.0, 1.0, 1.0))
        style.setdefault("lit", False)
        return self.add_quads(world[None], sampler=sampler, **style)

    def set_dvr(self, sampler, step, ref_length=None, object_id=-1):
        """Direct volume rendering of a registered volume: the ray is marched through its box in
        steps of ``step`` (world units); the LUT alpha is the opacity accumulated over
        ``ref_length`` (default: one step).  Marching stops at opaque geometry.  ``None`` = off."""
        self._dvr = None if sampler is None else (int(sampler), float(step), float(ref_length or step), int(object_id))

    # ---- Gaussians ----------------------------------------------------------------------------
    def add_gaussians(self, means, covs=None, opacity=0.8, color=(1.0, 0.5, 0.1), rotations=None, scales=None,
                      object_id=0):
        """3D Gaussians rendered as Gaussians: per ray ``alpha = opacity * exp(-d2/2)`` with d2 the
        smallest squared Mahalanobis distance between the ray and the mean, composited front to
        back in order of the point of maximum response.  Give ``covs`` (N,3,3) or ``rotations``
        (N,3,3, axes in columns) + ``scales`` (N,3 standard deviations)."""
        mu = np.asarray(means, np.float64).reshape(-1, 3)
        if covs is not None:
            rotations, scales = cov_to_axes(covs)
        rot = np.asarray(rotations, np.float64).reshape(-1, 3, 3)
        sig = np.broadcast_to(np.asarray(scales, np.float64), mu.shape)
        w = np.einsum("na,nba->nab", 1.0 / sig, rot)                    # diag(1/sigma) R^T : world -> canonical
        ext = np.sqrt(np.einsum("nab,nb->na", rot ** 2, sig ** 2))       # sqrt(diag(cov)) = 1-sigma half extent
        rgbo = np.concatenate([_per(color, len(mu), 4)[:, :3],
                               np.broadcast_to(np.asarray(opacity, np.float32), (len(mu),))[:, None]], axis=1)
        self._g.append((mu, w, ext, rgbo, np.broadcast_to(np.asarray(object_id, np.int32), (len(mu),))))
        self._lo, self._hi = np.minimum(self._lo, (mu - 3 * ext).min(0)), np.maximum(self._hi, (mu + 3 * ext).max(0))
        self._gpu = None
        return len(mu)

    # ---- GPU side -----------------------------------------------------------------------------
    def bounds(self):
        """(lo, hi) of everything added so far."""
        return self._lo.copy(), self._hi.copy()

    @property
    def num_triangles(self):
        return int(sum(len(f) for f in self._f))

    @property
    def num_gaussians(self):
        return int(sum(len(g[0]) for g in self._g))

    def build(self, device="cuda:0", gaussian_cutoff=4.0):
        """Upload everything and build the acceleration structures (triangle BVH inside wp.Mesh,
        wp.Bvh over the ``gaussian_cutoff``-sigma boxes of the Gaussians)."""
        t0 = time.perf_counter()
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
        if self._vol_gpu is None or self._vol_gpu[0] != (device, self._vol_size):      # voxels are uploaded once
            self._vol_gpu = ((device, self._vol_size), arr(cat(self._vol, (1,), np.int32), wp.int32))
        g["vol"] = self._vol_gpu[1]
        g["luts"] = arr(np.stack(self._luts) if self._luts else np.zeros((1, 256, 4), np.float32), wp.vec4)
        smp = []
        for s in self.samplers or [dict(i2w=np.eye(4), shape=(1, 1, 1), offset=0, labels=0, linear=0, lut=0,
                                        zero_alpha=0.0, focus=-1, grid=0.0, grid_rgb=(0, 0, 0), vsize=np.ones(3))]:
            k = _Sampler()
            k.w2i = wp.mat44(np.linalg.inv(s["i2w"]).astype(np.float32))
            k.vsize = wp.vec3(*[float(x) for x in s["vsize"]])
            k.offset, (k.nz, k.ny, k.nx) = s["offset"], s["shape"]
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
               gaussian_min_alpha=1.0 / 255.0):
        """Render with ``camera``; returns an :class:`Image`.  ``max_hits`` bounds the number of
        successive (translucent) surface hits composited per ray; ``depth_alpha`` is the alpha a
        surface must exceed to be written to the depth / object-id buffers."""
        if self._gpu is None or (device is not None and device != self._gpu["device"]):
            self.build(device or "cuda:0")
        g = self._gpu
        device = g["device"]
        right, up, fwd = camera.basis()
        cam = _Cam()
        cam.eye, cam.right, cam.up, cam.fwd = (wp.vec3(*[float(x) for x in v]) for v in (camera.eye, right, up, fwd))
        cam.ortho = int(camera.mode != "perspective")
        span = camera.ortho_height if cam.ortho else 2.0 * math.tan(math.radians(camera.fov) / 2)
        cam.span_x, cam.span_y = span * camera.width / camera.height, span
        cam.width, cam.height, cam.ss = camera.width, camera.height, int(camera.ssaa)
        bg = tuple(camera.background) + (1.0,) * (4 - len(camera.background))
        cam.bg = wp.vec4(*[float(x) for x in bg])
        rp = _Params()
        rp.has_mesh, rp.max_hits, rp.ambient = g["has_mesh"], int(max_hits), float(ambient)
        rp.max_steps = max(8 * int(max_hits), 256) + 1
        diag = float(np.linalg.norm(self._hi - self._lo)) if np.all(np.isfinite(self._lo)) else 1.0
        rp.eps, rp.t_min, rp.depth_alpha = 1.0e-5 * diag, float(min_transmittance), float(depth_alpha)
        rp.n_gauss, rp.g_cut2, rp.g_amin = g["n_gauss"], g["cutoff"] ** 2, float(gaussian_min_alpha)
        rp.dvr, rp.dvr_step, rp.dvr_ref, rp.dvr_id = self._dvr if self._dvr else (-1, 1.0, 1.0, -1)
        shape = (camera.height, camera.width)
        out = (wp.zeros(shape, dtype=wp.vec4, device=device), wp.zeros(shape, dtype=wp.float32, device=device),
               wp.zeros(shape, dtype=wp.int32, device=device))
        t0 = time.perf_counter()
        wp.launch(_render_kernel, dim=shape, device=device, outputs=list(out),
                  inputs=[cam, rp, g["mesh"].id if g["mesh"] else wp.uint64(0), g["pts"], g["tri"], g["nrm"],
                          g["rgba"], g["rgba2"], g["mat"], g["obj"], g["aux"], g["vol"], g["smp"], g["luts"],
                          g["g_bvh"].id if g["g_bvh"] else wp.uint64(0), g["g_mean"], g["g_w"], g["g_rgbo"],
                          g["g_obj"]])
        wp.synchronize_device(device)
        seconds = time.perf_counter() - t0
        rgbaf, depth, oid = (o.numpy() for o in out)
        rgb8 = rgbaf.copy()
        if bg[3] < 1.0:                                     # un-premultiply for a straight-alpha PNG
            rgb8[..., :3] /= np.maximum(rgb8[..., 3:], 1e-6)
        rgba = np.floor(np.clip(rgb8, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        depth = np.where(depth >= _FAR, np.inf, depth).astype(np.float32)
        return Image(rgba, depth, oid, rgbaf, seconds)
