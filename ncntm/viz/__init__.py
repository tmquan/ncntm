"""ncntm.viz: six representations of EM connectomics data, drawn with their true voxel shape.

    rawvalue   the uint8 number exactly as stored in the file, in array index space
    voxvalue   that number normalised to [0, 1], attached to a physical voxel of (dz, dy, dx) nm
    instance   instance segmentation: voxel label = instance id i
    skeleton   centre line of an instance, with radii
    wiremesh   wireframe surface mesh of an instance
    gaussian   mixture of 3D Gaussians

(see docs/TERMINOLOGY.md).  Arrays are (z, y, x); coordinates are (x, y, z) in true physical
nanometres; voxel [k, j, i] is the box from (i*dx, j*dy, k*dz) to ((i+1)*dx, ...); a scene
unit is one micrometre.  All of this is fixed in ``grid.py``.

The modules, in the order of a notebook
    samples        load_sample(name) -> Sample: rawvalue, voxvalue(), instance, mask(i), grid
    grid           VoxelGrid: the one place where indices become nanometres; "physical" / "index" space
    convert        instance -> Bricks, Mesh (wiremesh), Skeleton, Gaussians  (containers of types.py)
    kit            start_kit(): Isaac Sim inside a Jupyter kernel            [imports nothing heavy]
    stage          Dataset: the USD stage with the "space" variant set       [imports pxr]
    usd_to_emray   build_scene(stage): draw that stage with the ray caster   [imports pxr and warp]
    emray          Scene, Camera: the Warp ray caster (CUDA or CPU)           [imports warp]
    figure         montage, side_by_side, caption, scale_bar, show           [imports Pillow]
    view           render(stage): one picture, by RTX where Isaac Sim can draw, else by the
                   ray caster; compare_backends                [imports the above when used]

``import ncntm.viz`` loads numpy and nothing else: the names of ``__all__`` (samples, grid,
convert, types).  The other modules, and the names of theirs listed in ``LAZY`` (Scene,
Camera, Dataset, new_stage, save_stage, find_datasets, build_scene, default_camera,
camera_from_stage, montage, side_by_side, caption, scale_bar, show, start_kit, render,
compare_backends), are imported the first time they are used:

    import ncntm.viz as viz
    sample = viz.load_sample("snemi3d")                  # numpy, h5py
    mesh = viz.instance_to_mesh(sample.mask(10), sample.grid)
    session = viz.kit.start_kit()                        # BEFORE the first use of pxr (see kit.py)
    stage = viz.new_stage("scene.usda")                  # imports pxr now
    picture = viz.build_scene(stage).render(viz.default_camera(stage))      # imports warp now

``from ncntm.viz import *`` gives the numpy-only names of ``__all__``.
"""
import importlib

from .convert import (compare_anisotropy, gaussians_wrong_ways, instance_to_bricks, instance_to_gaussians,
                      instance_to_mesh, instance_to_skeleton, mean_radius_nm, measure, single_gaussian,
                      smooth_skeleton, voxel_gaussian)
from .grid import NM_PER_UNIT, SPACES, VoxelGrid
from .samples import Sample, list_samples, load_sample, samples_dir
from .types import Bricks, Gaussians, Mesh, Skeleton

# imported with the package: numpy only (h5py, scipy, zmesh, kimimaro are imported inside the
# functions that use them)
__all__ = [
    "NM_PER_UNIT", "SPACES", "VoxelGrid",
    "Bricks", "Mesh", "Skeleton", "Gaussians",
    "Sample", "samples_dir", "list_samples", "load_sample",
    "instance_to_bricks", "instance_to_mesh", "instance_to_skeleton", "smooth_skeleton", "instance_to_gaussians",
    "single_gaussian", "voxel_gaussian", "gaussians_wrong_ways", "measure", "mean_radius_nm", "compare_anisotropy",
]

MODULES = ("grid", "types", "samples", "convert", "kit", "stage", "usd_to_emray", "emray", "figure", "view")

# name -> module that defines it; that module is imported when the name is first used
LAZY = {
    "Scene": "emray", "Camera": "emray",
    "Dataset": "stage", "new_stage": "stage", "save_stage": "stage", "find_datasets": "stage",
    "build_scene": "usd_to_emray", "default_camera": "usd_to_emray", "camera_from_stage": "usd_to_emray",
    "montage": "figure", "side_by_side": "figure", "caption": "figure", "scale_bar": "figure", "show": "figure",
    "start_kit": "kit",
    "render": "view", "compare_backends": "view",
}


def __getattr__(name):
    """Called by Python for a name the package does not have yet: import its module now."""
    if name in MODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in LAZY:
        value = getattr(importlib.import_module(f"{__name__}.{LAZY[name]}"), name)
        globals()[name] = value                  # from now on a plain attribute of the package
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(MODULES) | set(LAZY))
