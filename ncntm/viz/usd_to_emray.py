"""usd_to_emray -- draw a USD stage with the Warp ray caster.

``build_scene(stage)`` reads the COMPOSED stage (at the current selection of the "space"
variant) and hands what it finds to ``emray.Scene``:

    Mesh                      triangles           points through the world matrix of the prim
    Mesh with em:slice:*      volume-sampled quad the HDF5 data named by the attributes
    BasisCurves (linear)      tubes               radius = width / 2
    PointInstancer of Cube    boxes               the 8 corners through every instance matrix
    PointInstancer of Sphere  ellipsoids          from the instance matrices
    Points                    spheres             radius = width / 2
    the Gaussian prim         true Gaussians      em:gaussian:* (or the splat attributes)

World matrices come from ``UsdGeom.XformCache`` (so the non-uniform scale of the voxels
frame is applied here, by USD, and nowhere else); invisible prims and prims whose purpose
is not default or render are left out; prim types that are not listed are skipped with one
warning that lists them.

Nothing here knows about voxel sizes: if the stage is wrong, the picture is wrong.

Units: world coordinates are stage units (1 unit = 1 micrometre).  Matrices are numpy 4x4
in COLUMN-vector convention (p' = M @ [x, y, z, 1]); USD's Gf matrices are row-vector, so
every matrix read from USD is transposed once, in ``_world_matrix``.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

from . import emray
from .samples import samples_dir
from .stage import LAYERS, SH_C0, SPLAT_TYPE, linear_to_srgb, slice_lut8

_NOT_DRAWN = {"", "Xform", "Scope", "Material", "Shader", "NodeGraph", "Camera", "GeomSubset"}
_VOLUMES = {}        # (file, its modification time, dataset, box) -> array read from HDF5, kept for the process

# the 6 faces of the cube [-1, 1]^3, 4 corners each, in order around the face
_CUBE_QUADS = np.array([[[(s if a == axis else (u if a == (axis + 1) % 3 else v)) for a in range(3)]
                         for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                        for axis in range(3) for s in (-1, 1)], dtype=np.float64)


# ---- reading the stage -----------------------------------------------------------------------
def _world_matrix(cache: UsdGeom.XformCache, prim: Usd.Prim) -> np.ndarray:
    """Local-to-world matrix of ``prim`` as a COLUMN-vector 4x4 (the transpose of USD's)."""
    return np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T


def _to_world(m: np.ndarray, points) -> np.ndarray:
    """Local points (N, 3) -> world points (N, 3)."""
    return np.asarray(points, dtype=np.float64).reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


def _layer_of(prim: Usd.Prim) -> str:
    """Value of em:layer on the prim or its nearest ancestor that has one ("" if none)."""
    while prim and not prim.IsPseudoRoot():
        tag = prim.GetAttribute("em:layer")
        if tag and tag.Get():
            return tag.Get()
        prim = prim.GetParent()
    return ""


def _dataset_prim(prim: Usd.Prim) -> Usd.Prim:
    """The dataset prim (the ancestor that carries em:voxelSizeNm) of a slice quad."""
    while prim and not prim.IsPseudoRoot():
        if prim.HasAttribute("em:voxelSizeNm"):
            return prim
        prim = prim.GetParent()
    raise ValueError("a slice quad must sit under a dataset prim (one with em:voxelSizeNm)")


def _surface_shader(prim: Usd.Prim):
    """The surface shader of the material bound to ``prim``, or None."""
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    shader = material.ComputeSurfaceSource()[0] if material else None
    return shader if shader else None


def _style(prim: Usd.Prim) -> tuple:
    """(display colours (n, 3), opacity, lit) of a gprim.

    colours   primvars:displayColor, converted from USD's linear values to display values;
              n is 1 or one per element (face, curve, point), as authored.  Grey if absent.
    opacity   primvars:displayOpacity if authored, else the ``opacity`` of the bound
              UsdPreviewSurface, else 1.
    lit       False if the bound material emits (emissiveColor set or connected): such a
              material shows its colour independently of the light."""
    gprim = UsdGeom.Gprim(prim)
    color = gprim.GetDisplayColorAttr().Get()
    rgb = linear_to_srgb(np.array(color, dtype=np.float64).reshape(-1, 3)) if color else np.full((1, 3), 0.8)
    display_opacity = gprim.GetDisplayOpacityAttr().Get()
    opacity, lit = (float(display_opacity[0]) if display_opacity else None), True
    shader = _surface_shader(prim)
    if shader is not None:
        emissive = shader.GetInput("emissiveColor")
        if emissive and (emissive.HasConnectedSource() or max(emissive.Get() or (0.0,)) > 0.0):
            lit = False
        material_opacity = shader.GetInput("opacity")
        if opacity is None and material_opacity and material_opacity.Get() is not None:
            opacity = float(material_opacity.Get())
    return rgb, (1.0 if opacity is None else opacity), lit


def _rgba(rgb, opacity, n=None) -> np.ndarray:
    """One RGBA (4,) for a single colour, else (n, 4); a colour count other than 1 or n falls
    back to the mean colour."""
    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    if len(rgb) != 1 and len(rgb) != n:
        rgb = rgb.mean(axis=0, keepdims=True)
    out = np.concatenate([rgb, np.full((len(rgb), 1), opacity)], axis=1)
    return out[0] if len(out) == 1 else out


def _triangulate(counts, indices) -> tuple:
    """Polygons (faceVertexCounts, faceVertexIndices) -> (triangles (T, 3), face of each triangle (T,)),
    by fans: a polygon with n corners gives n - 2 triangles."""
    counts, indices = np.asarray(counts, dtype=np.int64), np.asarray(indices, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    triangles, faces = [], []
    for n in np.unique(counts[counts >= 3]):
        which = np.flatnonzero(counts == n)
        polygons = indices[starts[which][:, None] + np.arange(n)[None, :]]
        for k in range(1, n - 1):
            triangles.append(polygons[:, [0, k, k + 1]])
            faces.append(which)
    if not triangles:
        return np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
    return np.concatenate(triangles), np.concatenate(faces)


def _quat_xyzw_to_matrix(q) -> np.ndarray:
    """Unit quaternions (N, 4) in the numpy layout of a Vt quaternion array, (x, y, z, w),
    -> rotation matrices (N, 3, 3), column convention."""
    x, y, z, w = (np.asarray(q, dtype=np.float64) / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)], axis=-2)


# ---- the volumes behind the slices -----------------------------------------------------------
def load_volume(dataset_prim: Usd.Prim, dataset: str, volumes=None) -> np.ndarray:
    """The (z, y, x) array that the slices of a dataset prim show.

    ``dataset`` is the HDF5 dataset name, "raw" or "instance".  The array is looked up
    1. in ``volumes``: {dataset prim name: a Sample, or a dict {"raw": ..., "instance": ...}};
    2. in the HDF5 file named by em:sampleFile on the dataset prim, searched next to the
       stage file and then in the samples directory ($NCNTM_SAMPLES, else <repository>/samples).
       The box read is em:voxelOffset (i0, j0, k0) + em:shapeVoxels (nx, ny, nz).
    A file is read once per process (again only if it was modified since) and the same array
    object is returned afterwards."""
    name = dataset_prim.GetName()
    nx, ny, nz = dataset_prim.GetAttribute("em:shapeVoxels").Get()
    if volumes is not None and name in volumes:
        source = volumes[name]
        array = source[dataset] if isinstance(source, dict) else getattr(source, dataset)
        if array is None:
            raise LookupError(f"volumes[{name!r}] has no {dataset!r} array")
    else:
        file_name = dataset_prim.GetAttribute("em:sampleFile").Get()
        if not file_name:
            raise LookupError(f"dataset {dataset_prim.GetPath()} records no HDF5 file (its arrays were derived in "
                              f"memory): pass build_scene(stage, volumes={{{name!r}: sample}})")
        stage_file = dataset_prim.GetStage().GetRootLayer().realPath
        places = ([Path(stage_file).parent] if stage_file else []) + [samples_dir()]
        path = next((p / file_name for p in places if (p / file_name).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"{file_name} not found in {', '.join(str(p) for p in places)}: set the environment "
                                    f"variable NCNTM_SAMPLES, or pass build_scene(stage, volumes={{{name!r}: sample}})")
        i0, j0, k0 = dataset_prim.GetAttribute("em:voxelOffset").Get()
        # the modification time is part of the key: a file that was written again is read again
        key = (str(path), path.stat().st_mtime_ns, dataset, (k0, j0, i0, nz, ny, nx))
        if key not in _VOLUMES:
            import h5py

            with h5py.File(path, "r") as f:                 # read-only
                if dataset not in f:
                    raise KeyError(f"{path} has no dataset /{dataset}")
                _VOLUMES[key] = f[dataset][k0:k0 + nz, j0:j0 + ny, i0:i0 + nx]
        array = _VOLUMES[key]
    if tuple(array.shape) != (nz, ny, nx):
        raise ValueError(f"the /{dataset} array for {dataset_prim.GetPath()} has shape {tuple(array.shape)}, "
                         f"the stage says {(nz, ny, nx)} (z, y, x)")
    return array


# ---- the scene builder -------------------------------------------------------------------------
class _Builder:
    """State of one ``build_scene`` call: the emray scene, the transform cache, the table of
    object ids, and one method per prim type."""

    def __init__(self, volumes, tube_sides, sphere_subdiv):
        self.scene = emray.Scene()                # scene.objects: object id -> {"path", "layer", "type"}
        self.cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        self.volumes, self.tube_sides, self.sphere_subdiv = volumes, tube_sides, sphere_subdiv
        self.samplers = {}                        # (array, matrix, colour map, window) -> emray sampler id
        self.skipped = self.scene.skipped         # reason -> [prim paths]

    def object_id(self, prim: Usd.Prim, layer: str) -> int:
        """A new id for the object-id buffer; ids start at 1 (-1 means "nothing")."""
        oid = len(self.scene.objects) + 1
        self.scene.objects[oid] = {"path": str(prim.GetPath()), "layer": layer, "type": prim.GetTypeName()}
        return oid

    def skip(self, prim: Usd.Prim, reason: str) -> None:
        self.skipped.setdefault(reason, []).append(str(prim.GetPath()))

    def length_scale(self, prim: Usd.Prim, m: np.ndarray) -> float:
        """World length of one local unit, for widths and radii.  Under a non-uniform scale
        a width has no defined shape: the geometric mean is used and the prim is reported."""
        s = np.linalg.norm(m[:3, :3], axis=0)                 # lengths of the images of the local axes
        if s.max() > s.min() * (1.0 + 1e-6):
            self.skip(prim, "width under a non-uniform scale, drawn with the mean scale")
        return float(np.prod(s) ** (1.0 / 3.0))

    # -- Mesh ----------------------------------------------------------------------------------
    def mesh(self, prim, m, layer):
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts, indices = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts:
            return self.skip(prim, "Mesh without points or faces")
        if prim.HasAttribute("em:slice:axis"):
            return self.slice(prim, m, layer, np.array(points, dtype=np.float64))
        world = _to_world(m, points)
        triangles, face = _triangulate(counts, indices)
        rgb, opacity, lit = _style(prim)
        rgba = _rgba(rgb, opacity, n=len(counts))
        if rgba.ndim == 2:                                  # one colour per face -> one per triangle
            rgba = rgba[face]
        self.scene.add_mesh(world, triangles, color=rgba, lit=lit, object_id=self.object_id(prim, layer))

    def slice(self, prim, m, layer, corners):
        """A slice quad: its colour is the volume sampled at the world position of each ray hit.

        ``corners`` (4, 3) are the points of the quad: voxel index coordinates (corner
        convention).  So ``m``, the world matrix of the quad, is also the index-to-world matrix
        of the volume: its diagonal is the size of a voxel as drawn, its translation the low
        corner of voxel [0, 0, 0].  The volume is sampled where the quad IS; that this is the
        plane the attributes name (through the centre of voxel layer em:slice:index) is checked."""
        if len(corners) != 4:
            return self.skip(prim, "slice that is not a quad")
        said = {name: prim.GetAttribute(f"em:slice:{name}").Get() for name in ("dataset", "axis", "index", "colorMap", "window")}
        if None in said.values() or said["axis"] not in ("x", "y", "z"):
            raise ValueError(f"{prim.GetPath()}: a slice quad needs em:slice:dataset, axis (x, y or z), index, colorMap "
                             f"and window; found {said}")
        if np.abs(corners[:, "xyz".index(said["axis"])] - (said["index"] + 0.5)).max() > 1e-6:
            raise ValueError(f"{prim.GetPath()} says it is the slice {said['axis']} = {said['index']}, but its points do "
                             f"not lie in the plane {said['axis']} = {said['index']} + 0.5 (voxel index units)")
        world_corners = _to_world(m, corners)
        colormap, window = said["colorMap"], tuple(said["window"])
        array = load_volume(_dataset_prim(prim), said["dataset"], self.volumes)
        linear = m[:3, :3]
        if np.abs(linear - np.diag(np.diag(linear))).max() > 1e-9 * np.abs(linear).max() or np.diag(linear).min() <= 0:
            raise NotImplementedError(f"{prim.GetPath()}: the ray caster samples axis-aligned volumes only, "
                                      "but the voxels frame is rotated or mirrored in the world")
        key = (id(array), m.tobytes(), colormap, window)
        if key not in self.samplers:
            placement = dict(voxel_size=(linear[2, 2], linear[1, 1], linear[0, 0]),      # (dz, dy, dx), world units
                             origin=m[:3, 3], convention="corner")
            if colormap == "labels":                    # ids through the colour hash, id 0 dark and opaque
                self.samplers[key] = self.scene.add_volume(array, labels=True, zero_alpha=1.0, **placement)
            else:                                       # stored numbers through the same table as the texture
                lut = np.concatenate([slice_lut8(colormap, window) / 255.0, np.ones((256, 1))], axis=1)
                self.samplers[key] = self.scene.add_volume(array, lut=lut, **placement)
        self.scene.add_quads(world_corners[None], sampler=self.samplers[key], color=(1.0, 1.0, 1.0, 1.0), lit=False,
                             object_id=self.object_id(prim, layer))

    # -- BasisCurves and Points ----------------------------------------------------------------
    def _radii(self, prim, schema, m, n_points, counts=None):
        """World radius of every point from ``widths`` (diameters in local units), or None."""
        widths = schema.GetWidthsAttr().Get()
        if not widths:
            return None
        widths, interpolation = np.array(widths, dtype=np.float64), schema.GetWidthsInterpolation()
        if len(widths) == 1 or interpolation == UsdGeom.Tokens.constant:
            widths = np.full(n_points, widths[0])
        elif interpolation == UsdGeom.Tokens.uniform and counts is not None:
            widths = np.repeat(widths, counts)                  # one width per curve
        if len(widths) != n_points:
            return None
        return 0.5 * widths * self.length_scale(prim, m)

    def curves(self, prim, m, layer):
        curves = UsdGeom.BasisCurves(prim)
        if curves.GetTypeAttr().Get() != UsdGeom.Tokens.linear:
            return self.skip(prim, "BasisCurves that are not linear")
        points, counts = curves.GetPointsAttr().Get(), curves.GetCurveVertexCountsAttr().Get()
        if not points or not counts:
            return self.skip(prim, "BasisCurves without points")
        counts = np.array(counts, dtype=np.int64)
        radii = self._radii(prim, curves, m, len(points), counts)
        if radii is None:
            return self.skip(prim, "BasisCurves without usable widths")
        first = np.cumsum(counts) - counts
        k = np.arange(len(points) - 1)
        last_of_curve = np.zeros(len(points), dtype=bool)
        last_of_curve[first + counts - 1] = True
        edges = np.stack([k, k + 1], axis=1)[~last_of_curve[:-1]]          # consecutive points of one curve
        if curves.GetWrapAttr().Get() == UsdGeom.Tokens.periodic:
            edges = np.concatenate([edges, np.stack([first + counts - 1, first], axis=1)])
        rgb, opacity, lit = _style(prim)
        curve_of_edge = np.searchsorted(first, edges[:, 0], side="right") - 1
        if len(rgb) == len(counts) and len(counts) > 1:          # one colour per curve
            rgb = rgb[curve_of_edge]
        elif len(rgb) == len(points) and len(rgb) > 1:           # one colour per point: mean of the two ends
            rgb = 0.5 * (rgb[edges[:, 0]] + rgb[edges[:, 1]])
        self.scene.add_tubes(_to_world(m, points), edges, radii, sides=self.tube_sides,
                             color=_rgba(rgb, opacity, n=len(edges)), lit=lit, object_id=self.object_id(prim, layer))

    def points(self, prim, m, layer):
        points = UsdGeom.Points(prim)
        positions = points.GetPointsAttr().Get()
        if not positions:
            return self.skip(prim, "Points without points")
        radii = self._radii(prim, points, m, len(positions))
        if radii is None:
            return self.skip(prim, "Points without usable widths")
        rgb, opacity, lit = _style(prim)
        self.scene.add_spheres(_to_world(m, positions), radii, subdiv=self.sphere_subdiv,
                               color=_rgba(rgb, opacity, n=len(positions)), lit=lit, object_id=self.object_id(prim, layer))

    # -- Cube, Sphere, PointInstancer ------------------------------------------------------------
    def shapes(self, proto, matrices, oid):
        """Draw ``proto`` (a Cube or a Sphere) once per matrix.  ``matrices`` are
        (N, 4, 4) prototype-to-world matrices in USD's ROW-vector convention: row a of the
        upper 3x3 is the image of the local axis a, row 3 is the centre.  Any affine matrix
        is exact here: boxes become parallelepipeds, spheres become ellipsoids.
        The colour is the prototype's (per-instance primvars on an instancer are not read)."""
        linear, centre = matrices[:, :3, :3], matrices[:, 3, :3]
        rgb, opacity, lit = _style(proto)
        style = dict(color=_rgba(rgb, opacity), lit=lit, object_id=oid)
        if proto.IsA(UsdGeom.Cube):
            half = 0.5 * float(UsdGeom.Cube(proto).GetSizeAttr().Get())
            corners = centre[:, None, None, :] + half * np.einsum("fca,nab->nfcb", _CUBE_QUADS, linear)
            self.scene.add_quads(corners.reshape(-1, 4, 3), **style)
        elif proto.IsA(UsdGeom.Sphere):
            radius = float(UsdGeom.Sphere(proto).GetRadiusAttr().Get())
            self.scene.add_ellipsoids(centre, radius * np.swapaxes(linear, 1, 2), subdiv=self.sphere_subdiv, **style)
        else:
            self.skip(proto, f"PointInstancer prototype of type {proto.GetTypeName() or 'none'}")

    def gprim(self, prim, m, layer):
        """A Cube or Sphere that stands alone (not a prototype)."""
        self.shapes(prim, m.T[None], self.object_id(prim, layer))

    def instancer(self, prim, m, layer):
        instancer = UsdGeom.PointInstancer(prim)
        time = Usd.TimeCode.Default()
        # instance -> instancer frame, including the prototype's own transform; USD applies, in
        # this order: prototype transform, scales, orientations, positions
        local = np.array(instancer.ComputeInstanceTransformsAtTime(time, time, UsdGeom.PointInstancer.IncludeProtoXform),
                         dtype=np.float64).reshape(-1, 4, 4)
        indices = np.array(instancer.GetProtoIndicesAttr().Get() or [], dtype=np.int64)
        if len(local) == 0 or len(local) != len(indices):
            return self.skip(prim, "PointInstancer without instances (or with masked instances)")
        world = local @ m.T                                  # row-vector matrices compose left to right
        oid = self.object_id(prim, layer)
        for p, target in enumerate(instancer.GetPrototypesRel().GetTargets()):
            proto = prim.GetStage().GetPrimAtPath(target)
            if proto and np.any(indices == p):
                self.shapes(proto, world[indices == p], oid)

    # -- Gaussians ---------------------------------------------------------------------------------
    def gaussians(self, prim, m, layer):
        """True Gaussians.  From em:gaussian:* if the prim has them (the exact means and full
        covariances written by ``stage.Dataset.add_gaussians``); else from the attributes of a
        ParticleField3DGaussianSplat: covariance = R diag(scales^2) R^T, colour from the
        degree-0 spherical harmonic, rgb = 0.5 + SH_C0 * c0.  Both are in the local frame of
        the prim: means go through the world matrix M, covariances become L S L^T with L the
        upper 3x3 of M."""
        means = prim.GetAttribute("em:gaussian:means").Get() if prim.HasAttribute("em:gaussian:means") else None
        if means is not None:
            means = np.array(means, dtype=np.float64)
            covs = np.array(prim.GetAttribute("em:gaussian:covariances").Get(), dtype=np.float64).reshape(-1, 3, 3)
            opacity = np.array(prim.GetAttribute("em:gaussian:opacities").Get(), dtype=np.float64)
            rgb = np.array(prim.GetAttribute("em:gaussian:displayRgb").Get(), dtype=np.float64)
        else:
            get = lambda name: prim.GetAttribute(name).Get() if prim.HasAttribute(name) else None      # noqa: E731
            positions, orientations, scales = get("positions"), get("orientations"), get("scales")
            if not positions or not orientations or not scales:
                return self.skip(prim, f"{SPLAT_TYPE} without positions, orientations and scales")
            means = np.array(positions, dtype=np.float64)
            rotations = _quat_xyzw_to_matrix(np.array(orientations, dtype=np.float64).reshape(-1, 4))
            sigmas = np.array(scales, dtype=np.float64)
            covs = np.einsum("nab,nb,ncb->nac", rotations, sigmas ** 2, rotations)
            opacity = np.array(get("opacities") or np.ones(len(means)), dtype=np.float64)
            degree = get("radiance:sphericalHarmonicsDegree")
            per_gaussian = (3 if degree is None else int(degree)) + 1          # schema default: degree 3
            harmonics = get("radiance:sphericalHarmonicsCoefficients")
            rgb = np.full((len(means), 3), 0.5)
            if harmonics and len(harmonics) == len(means) * per_gaussian ** 2:
                c0 = np.array(harmonics, dtype=np.float64).reshape(len(means), per_gaussian ** 2, 3)[:, 0]
                rgb = np.clip(0.5 + SH_C0 * c0, 0.0, 1.0)
        if not (len(means) == len(covs) == len(opacity) == len(rgb)) or len(means) == 0:
            return self.skip(prim, "Gaussian prim whose arrays differ in length")
        linear = m[:3, :3]
        self.scene.add_gaussians(_to_world(m, means), covs=linear @ covs @ linear.T, opacity=opacity, color=rgb,
                                 object_id=self.object_id(prim, layer))


def build_scene(stage: Usd.Stage, layers=None, device=None, volumes=None, *, tube_sides: int = 10,
                sphere_subdiv: int = 2) -> emray.Scene:
    """The composed ``stage`` as an ``emray.Scene``, built on ``device`` and ready to render.

    layers    None: everything; else a list of layer names (values of em:layer, e.g.
              ["rawvalue", "frame"]): only prims of these layers are drawn ("" = prims without
              a tag).  A name that is neither one of ``stage.LAYERS`` nor on the stage raises
              ValueError
    device    Warp device; default ``emray.default_device()`` (the environment variable
              NCNTM_DEVICE, else Warp's preferred device)
    volumes   arrays for the slices of datasets that have no HDF5 file, see ``load_volume``
    tube_sides, sphere_subdiv   tessellation of tubes and spheres

    Two dictionaries of the scene are filled: ``scene.objects``, {object id: {"path", "layer",
    "type"}}, to read the object-id buffer of a rendered image (one id per prim, from 1), and
    ``scene.skipped``, {reason: [prim paths]}, the content of the warning.  Which variant of
    "space" is drawn is whatever is selected on the stage: switch it with
    ``Dataset.set_space`` and build again."""
    if layers is not None:
        layers = [layers] if isinstance(layers, str) else list(layers)
        on_stage = {prim.GetAttribute("em:layer").Get() for prim in stage.Traverse() if prim.HasAttribute("em:layer")}
        unknown = [name for name in layers if name not in LAYERS and name not in on_stage and name != ""]
        if unknown:                                    # a misspelt name would give an empty picture without a word
            raise ValueError(f"unknown layer(s) {unknown}: this stage has {sorted(on_stage)}, the package's layers "
                             f"are {list(LAYERS)} ('' selects prims without a layer tag)")
    builder = _Builder(volumes, tube_sides, sphere_subdiv)
    drawn_purposes = (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
    prims = iter(Usd.PrimRange(stage.GetPseudoRoot()))
    for prim in prims:
        if prim.IsPseudoRoot():
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable and (imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
                          or imageable.ComputePurpose() not in drawn_purposes):
            prims.PruneChildren()                      # an invisible prim hides its whole subtree
            continue
        type_name = prim.GetTypeName()
        if type_name in ("Material", "PointInstancer"):
            prims.PruneChildren()                      # shaders; prototypes are drawn through their instancer
        layer = _layer_of(prim)
        if layers is not None and layer not in layers:
            continue
        if type_name == SPLAT_TYPE or prim.HasAttribute("em:gaussian:means"):
            draw = builder.gaussians
        else:
            draw = {"Mesh": builder.mesh, "BasisCurves": builder.curves, "Points": builder.points,
                    "PointInstancer": builder.instancer, "Cube": builder.gprim, "Sphere": builder.gprim}.get(type_name)
        if draw is not None:
            draw(prim, _world_matrix(builder.cache, prim), layer)
        elif type_name not in _NOT_DRAWN and not type_name.endswith("Light"):
            builder.skip(prim, f"prim type {type_name}")
    if builder.skipped:
        lines = [f"  {reason}: {', '.join(paths[:5])}{' ...' if len(paths) > 5 else ''} ({len(paths)})"
                 for reason, paths in builder.skipped.items()]
        warnings.warn("build_scene left these prims out or approximated them:\n" + "\n".join(lines), stacklevel=2)
    return builder.scene.build(device)


# ---- cameras -------------------------------------------------------------------------------------
def camera_from_stage(stage: Usd.Stage, camera_path, height: int = 720, ssaa: int = 2,
                      background=(1.0, 1.0, 1.0, 1.0)) -> emray.Camera:
    """The ``emray.Camera`` that sees what the UsdGeom.Camera at ``camera_path`` sees.

    A USD camera looks down its local -Z axis with +Y up; its position and axes are read
    from its world matrix.  perspective: vertical field of view = 2 atan(verticalAperture /
    (2 focalLength)).  orthographic: picture height = verticalAperture / 10 stage units
    (apertures are in tenths of a unit).  The picture is ``height`` pixels high and
    height * horizontalAperture / verticalAperture pixels wide."""
    prim = stage.GetPrimAtPath(str(camera_path))
    if not prim or not prim.IsA(UsdGeom.Camera):
        raise ValueError(f"{camera_path} is not a camera on this stage")
    camera = UsdGeom.Camera(prim)
    m = _world_matrix(UsdGeom.XformCache(Usd.TimeCode.Default()), prim)
    eye, up_length = m[:3, 3], float(np.linalg.norm(m[:3, 1]))
    up, forward = m[:3, 1] / up_length, -m[:3, 2] / np.linalg.norm(m[:3, 2])
    vertical, horizontal = camera.GetVerticalApertureAttr().Get(), camera.GetHorizontalApertureAttr().Get()
    common = dict(eye=tuple(eye), target=tuple(eye + forward), up=tuple(up), height=int(height),
                  width=int(round(height * horizontal / vertical)), ssaa=ssaa, background=tuple(background))
    if camera.GetProjectionAttr().Get() == UsdGeom.Tokens.orthographic:
        return emray.Camera(mode="orthographic", ortho_height=vertical / 10.0 * up_length, **common)
    fov = math.degrees(2.0 * math.atan(vertical / (2.0 * camera.GetFocalLengthAttr().Get())))
    return emray.Camera(mode="perspective", fov=fov, **common)


def default_camera(stage: Usd.Stage, height: int = 720, **kw) -> emray.Camera:
    """The camera to use when none is named: /World/Cameras/overview if it exists, else the
    first camera on the stage, else (no camera authored) one that frames the world bound of
    the stage with ``emray.Camera.fit`` at 4:3."""
    overview = stage.GetPrimAtPath("/World/Cameras/overview")
    cameras = ([overview] if overview else []) + [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    if cameras:
        return camera_from_stage(stage, cameras[0].GetPath(), height=height, **kw)
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]) \
        .ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    if bound.IsEmpty():
        raise ValueError("the stage has neither a camera nor anything to frame")
    return emray.Camera.fit(np.array(bound.GetMin()), np.array(bound.GetMax()), height=int(height),
                            width=int(round(height * 4 / 3)), **kw)


def render(stage: Usd.Stage, camera=None, layers=None, device=None, volumes=None, height: int = 720, **render_kw) -> emray.Image:
    """Build the scene and render it in one call.  ``camera``: an ``emray.Camera``, the path
    of a camera on the stage, or None for ``default_camera(stage)``."""
    if camera is None:
        camera = default_camera(stage, height=height)
    elif not isinstance(camera, emray.Camera):
        camera = camera_from_stage(stage, camera, height=height)
    return build_scene(stage, layers=layers, device=device, volumes=volumes).render(camera, **render_kw)
