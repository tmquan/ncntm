"""The sample crops: one small HDF5 file per dataset, read into a ``Sample``.

File layout (written by ``scripts/make_samples.py``, described in ``samples/manifest.json``)
    /raw                    uint8 (z, y, x)   EM intensity exactly as in the source file
    /instance               unsigned integer (z, y, x), compact ids 0..K, 0 = background,
                            1 = the largest instance (absent for image-only samples)
    /instance_voxel_counts  (K+1,)   voxels of id i
    /instance_bbox_zyx      (K+1, 6) z0, y0, x0, z1, y1, x1 of id i, upper bounds exclusive
    /instance_original_ids  (K+1,)   id of instance i in the source dataset
    root attributes         voxel_size_zyx_nm, intensity_norm_min, intensity_norm_max,
                            skeleton_subject_ids, label_convention, ...

Names used here are the names of the representations:
    rawvalue   the stored uint8 number, in array index space: ``sample.rawvalue[k, j, i]``
    voxvalue   that number normalised to [0, 1] and attached to a physical voxel of size
               (dz, dy, dx) nm: ``sample.voxvalue()`` together with ``sample.grid``
    instance   voxel label = instance id i: ``sample.instance``, ``sample.mask(i)``

Arrays are (z, y, x).  All geometry (which voxel is where, how big it is) is in
``sample.grid``, a ``VoxelGrid``; this module never multiplies by a voxel size itself.

Where the files are: the directory named by the environment variable ``NCNTM_SAMPLES``
if it is set, otherwise ``<repository>/samples``.  Files are opened read-only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

from .grid import VoxelGrid

STRIDE_MODES = ("stride", "slab_mean")


# ---- where the files are ---------------------------------------------------------------
def samples_dir() -> Path:
    """Directory that holds ``manifest.json`` and the ``*.h5`` files.

    ``$NCNTM_SAMPLES`` if set, else ``<repository>/samples`` (found from this file's
    location, so it does not depend on the current working directory)."""
    env = os.environ.get("NCNTM_SAMPLES")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "samples"


def manifest() -> dict:
    """``samples/manifest.json`` as a dict (key ``"samples"``: one entry per file)."""
    path = samples_dir() / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no manifest.json in {samples_dir()}; set the environment variable NCNTM_SAMPLES "
            "to the directory that holds the sample files"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_samples() -> list:
    """Sample names in manifest order (the three core samples come first)."""
    return [entry["name"] for entry in manifest()["samples"]]


# ---- the sample ------------------------------------------------------------------------
@dataclass(eq=False)
class Sample:
    """One crop: image, optional instance segmentation, and the grid that places them.

    name          sample name, e.g. "snemi3d"
    path          the HDF5 file it was read from
    raw           uint8 (z, y, x): the rawvalue
    instance      unsigned integer (z, y, x), 0 = background; None for image-only samples
    grid          VoxelGrid: shape and voxel size (dz, dy, dx) in nm
    attrs         root attributes of the file as plain Python values
    norm_window   (lo, hi): rawvalue lo maps to voxvalue 0, hi to 1
    origin_xyz_nm (x, y, z) nm: where the corner of voxel [0, 0, 0] of THIS sample lies
                  in the frame of the file.  Zero unless the sample was cropped; add it to
                  converted geometry to put a crop back where it came from.
    derivation    the crop / stride steps that produced this sample from the file
    """

    name: str
    path: Path
    raw: np.ndarray
    instance: Optional[np.ndarray]
    grid: VoxelGrid
    attrs: dict
    norm_window: tuple
    origin_xyz_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    derivation: tuple = ()
    # per-id tables (K+1,) and (K+1, 6); None means "compute from the instance array when asked"
    _voxel_counts: Optional[np.ndarray] = field(default=None, repr=False)
    _bboxes_zyx: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        if self.raw.ndim != 3 or self.raw.dtype != np.uint8:
            raise ValueError(f"raw must be a uint8 (z, y, x) array, got {self.raw.dtype} {self.raw.shape}")
        if tuple(self.raw.shape) != tuple(self.grid.shape_zyx):
            raise ValueError(f"raw has shape {self.raw.shape} but the grid says {self.grid.shape_zyx}")
        if self.instance is not None:
            if self.instance.shape != self.raw.shape:
                raise ValueError(f"instance has shape {self.instance.shape}, raw has {self.raw.shape}")
            if not np.issubdtype(self.instance.dtype, np.integer):
                raise ValueError(f"instance must be an integer array, got {self.instance.dtype}")
        lo, hi = self.norm_window
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError(f"norm_window must be (lo, hi) with hi > lo, got {self.norm_window}")

    def __repr__(self) -> str:
        return f"Sample({self.describe()})"

    def describe(self) -> str:
        """One line: name, grid, number of instances, how it was derived."""
        what = f"{len(self.instance_ids())} instances" if self.has_instances else "image only"
        how = "".join(f" -> {step}" for step in self.derivation)
        return f"{self.name}{how}: {self.grid.describe()}, {what}"

    # ---- rawvalue and voxvalue -----------------------------------------------------------
    @property
    def rawvalue(self) -> np.ndarray:
        """The stored uint8 array itself (not a copy), indexed [k, j, i] = (z, y, x)."""
        return self.raw

    def voxvalue(self, equalize_sections: bool = False) -> np.ndarray:
        """float32 (z, y, x) in [0, 1]:  clip((rawvalue - lo) / (hi - lo), 0, 1)  with
        (lo, hi) = ``norm_window``, the per-volume window stored in the file.

        equalize_sections=True first gives every z section (``raw[k]``) the same mean and
        standard deviation (the medians over the sections).  SNEMI3D needs it: its section
        means range from 110 to 207 grey levels, which shows as banding in any view that
        cuts across the sections.  It changes the values, so it is for display only."""
        v = self.raw.astype(np.float32)
        if equalize_sections:
            mean = v.mean(axis=(1, 2), keepdims=True)
            std = v.std(axis=(1, 2), keepdims=True)
            v = (v - mean) / np.maximum(std, 1e-6) * np.median(std) + np.median(mean)
        lo, hi = self.norm_window
        v -= np.float32(lo)
        v /= np.float32(hi - lo)
        return np.clip(v, 0.0, 1.0, out=v)

    # ---- instances -----------------------------------------------------------------------
    @property
    def has_instances(self) -> bool:
        return self.instance is not None

    def _need_instances(self):
        if self.instance is None:
            raise ValueError(f"sample {self.name!r} is image-only: it has no instance segmentation")

    def _tables(self):
        """(voxel_counts, bboxes_zyx) indexed by instance id; computed once when not read
        from the file (a cropped or strided sample has other counts and boxes than the file)."""
        self._need_instances()
        if self._voxel_counts is None:
            from scipy import ndimage

            counts = np.bincount(self.instance.ravel())
            boxes = np.zeros((len(counts), 6), dtype=np.int64)
            for i, sl in enumerate(ndimage.find_objects(self.instance), start=1):
                if sl is not None:
                    boxes[i] = [s.start for s in sl] + [s.stop for s in sl]
            self._voxel_counts, self._bboxes_zyx = counts, boxes
        return self._voxel_counts, self._bboxes_zyx

    def instance_ids(self) -> np.ndarray:
        """Ids present in this sample, largest first (ties: smaller id first).
        For a sample straight from the file this is 1, 2, ..., K."""
        counts, _ = self._tables()
        ids = np.flatnonzero(counts[1:] > 0) + 1
        order = np.lexsort((ids, -counts[ids].astype(np.int64)))
        return ids[order]

    def _check_id(self, i) -> int:
        counts, _ = self._tables()
        i = int(i)
        if i < 1 or i >= len(counts) or counts[i] == 0:
            raise ValueError(f"instance id {i} is not present in sample {self.name!r} (0 is background)")
        return i

    def voxel_count(self, i) -> int:
        """Number of voxels labelled i (all pieces)."""
        i = self._check_id(i)
        return int(self._tables()[0][i])

    def bbox_zyx(self, i) -> tuple:
        """(z0, y0, x0, z1, y1, x1): bounding box of instance i in array indices,
        upper bounds exclusive, so ``instance[z0:z1, y0:y1, x0:x1]`` contains it."""
        i = self._check_id(i)
        return tuple(int(v) for v in self._tables()[1][i])

    def mask(self, i, largest_component: bool = True) -> np.ndarray:
        """bool (z, y, x) array, same shape as the sample: True where the label is i.

        An id is not always one piece (a neurite can leave the crop and come back, and a
        strided sample loses thin bridges).  With largest_component=True only the largest
        6-connected piece (voxels joined through faces) is kept; the others are dropped."""
        z0, y0, x0, z1, y1, x1 = self.bbox_zyx(i)
        box = self.instance[z0:z1, y0:y1, x0:x1] == int(i)
        if largest_component:
            from scipy import ndimage

            pieces, n = ndimage.label(box)  # default structure: 6-connectivity
            if n > 1:
                sizes = np.bincount(pieces.ravel())
                sizes[0] = 0
                box = pieces == sizes.argmax()
        out = np.zeros(self.instance.shape, dtype=bool)
        out[z0:z1, y0:y1, x0:x1] = box
        return out

    @property
    def skeleton_subjects(self) -> tuple:
        """The three ids recommended for skeletons, from the file attribute
        ``skeleton_subject_ids``: (a neurite that runs across the sections, one that runs
        inside the section plane, the most branched one).  Empty for image-only samples."""
        return tuple(int(i) for i in self.attrs.get("skeleton_subject_ids", ()))

    # ---- derived samples ---------------------------------------------------------------------
    def crop(self, z=slice(None), y=slice(None), x=slice(None)) -> "Sample":
        """Sub-block ``[z, y, x]`` (three slices with step 1) as a new Sample.

        The new sample has its own grid starting at 0; ``origin_xyz_nm`` remembers where
        it was cut out.  Arrays are views of this sample's arrays (nothing is copied)."""
        start, stop = _crop_bounds(self.grid.shape_zyx, z, y, x)
        box = tuple(slice(a, b) for a, b in zip(start, stop))
        shape = tuple(b - a for a, b in zip(start, stop))
        # corner of the first kept voxel, (x, y, z) nm: integer index coordinates are corners
        shift_xyz_nm = self.grid.index_to_nm(start[::-1])
        return replace(
            self,
            raw=self.raw[box],
            instance=None if self.instance is None else self.instance[box],
            grid=VoxelGrid(shape, self.grid.voxel_size_zyx_nm),
            origin_xyz_nm=self.origin_xyz_nm + shift_xyz_nm,
            derivation=self.derivation + (f"crop[{start[0]}:{stop[0]}, {start[1]}:{stop[1]}, {start[2]}:{stop[2]}]",),
            _voxel_counts=None,
            _bboxes_zyx=None,
        )

    def strided(self, step: int, axis: int = 0, phase: int = 0, mode: str = "stride") -> "Sample":
        """The controlled anisotropy experiment: the same tissue with ``step`` times
        thicker voxels along array axis ``axis`` (0 = z).  fib25 (8 x 8 x 8 nm) with
        step 4 becomes a 32 x 8 x 8 nm volume whose truth is known.

        The array is cut into slabs of ``step`` planes (an incomplete last slab is dropped).
        Labels ALWAYS keep plane ``phase`` of each slab (a label cannot be averaged).
        The image keeps the same plane with mode="stride" (what a microscope that images
        one thin plane every ``step`` would store) or the rounded mean of the slab with
        mode="slab_mean" (a section that integrates over its thickness).

        The new grid is ``grid.strided(step, axis)`` of the slab-complete block."""
        step, axis, phase = int(step), int(axis), int(phase)
        if step < 1 or axis not in (0, 1, 2) or not 0 <= phase < step:
            raise ValueError(f"strided: need step >= 1, axis in 0..2 and 0 <= phase < step, got {step}, {axis}, {phase}")
        if mode not in STRIDE_MODES:
            raise ValueError(f"strided: mode must be one of {STRIDE_MODES}, got {mode!r}")
        n_slabs = self.grid.shape_zyx[axis] // step
        if n_slabs < 1:
            raise ValueError(f"strided: axis {axis} has {self.grid.shape_zyx[axis]} voxels, fewer than step {step}")
        whole = [slice(None)] * 3
        whole[axis] = slice(0, n_slabs * step)          # complete slabs only
        block = self.crop(*whole)
        keep = [slice(None)] * 3
        keep[axis] = slice(phase, None, step)           # plane `phase` of every slab
        keep = tuple(keep)
        if mode == "stride":
            raw = block.raw[keep]
        else:
            split = list(block.raw.shape)
            split[axis:axis + 1] = [n_slabs, step]      # (..., n_slabs, step, ...)
            raw = np.rint(block.raw.reshape(split).mean(axis=axis + 1)).astype(np.uint8)
        return replace(
            block,
            raw=np.ascontiguousarray(raw),
            instance=None if block.instance is None else np.ascontiguousarray(block.instance[keep]),
            grid=block.grid.strided(step, axis),
            derivation=self.derivation + (f"strided(step={step}, axis={axis}, phase={phase}, mode={mode!r})",),
            _voxel_counts=None,
            _bboxes_zyx=None,
        )


# ---- reading a file ----------------------------------------------------------------------
def _crop_bounds(shape_zyx, z, y, x):
    """Three slices with step 1 -> (start, stop) lists of array indices, checked."""
    start, stop = [], []
    for name, sl, n in zip("zyx", (z, y, x), shape_zyx):
        if not isinstance(sl, slice):
            raise TypeError(f"crop: {name} must be a slice, got {sl!r}")
        a, b, step = sl.indices(n)
        if step != 1:
            raise ValueError(f"crop: {name} has step {step}; use strided() to subsample")
        if b <= a:
            raise ValueError(f"crop: {name} = {sl} selects nothing from {n} voxels")
        start.append(a)
        stop.append(b)
    return start, stop


def _plain(value):
    """HDF5 attribute -> plain Python value (numpy scalars and arrays do not print well)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_sample(name: str, crop=None) -> Sample:
    """Read sample ``name`` (see ``list_samples()``) fully into memory.

    crop: optional (z, y, x) tuple of slices with step 1; only that block is read from
    the file, and the result equals ``load_sample(name).crop(z, y, x)``."""
    import h5py

    files = {entry["name"]: entry["file"] for entry in manifest()["samples"]}
    if name not in files:
        raise KeyError(f"unknown sample {name!r}; available: {', '.join(files)}")
    path = samples_dir() / files[name]
    if not path.is_file():
        raise FileNotFoundError(f"sample file {path} is listed in the manifest but missing")

    with h5py.File(path, "r") as f:                      # read-only
        attrs = {key: _plain(f.attrs[key]) for key in f.attrs}
        if attrs.get("axis_order", "zyx") != "zyx":
            raise ValueError(f"{path.name}: axis_order is {attrs['axis_order']!r}, this package expects 'zyx'")
        has_instances = "instance" in f
        if crop is None:
            box = (slice(None),) * 3
        else:
            if len(crop) != 3:
                raise ValueError("crop must be three slices (z, y, x)")
            start, stop = _crop_bounds(f["raw"].shape, *crop)
            box = tuple(slice(a, b) for a, b in zip(start, stop))
        raw = f["raw"][box]
        instance = f["instance"][box] if has_instances else None
        # the stored tables describe the whole file, so they are only used for an uncropped read
        whole = has_instances and crop is None
        counts = f["instance_voxel_counts"][...].astype(np.int64) if whole else None
        boxes = f["instance_bbox_zyx"][...].astype(np.int64) if whole else None

    grid = VoxelGrid(raw.shape, attrs["voxel_size_zyx_nm"])
    z0, y0, x0 = (0, 0, 0) if crop is None else start
    return Sample(
        name=name,
        path=path,
        raw=raw,
        instance=instance,
        grid=grid,
        attrs=attrs,
        norm_window=(float(attrs["intensity_norm_min"]), float(attrs["intensity_norm_max"])),
        origin_xyz_nm=grid.index_to_nm((x0, y0, z0)),     # corner of the first voxel read
        derivation=() if crop is None else (f"crop[{z0}:{stop[0]}, {y0}:{stop[1]}, {x0}:{stop[2]}]",),
        _voxel_counts=counts,
        _bboxes_zyx=boxes,
    )
