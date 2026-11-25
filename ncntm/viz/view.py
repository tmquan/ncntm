"""view -- one call that gives a picture of the USD stage, whatever machine the notebook runs on.

    picture = view.render(stage)                      # RTX where Isaac Sim can draw, else the Warp ray caster
    picture = view.render(stage, space="index")       # every voxel drawn as a cube; the stage is left as it was
    view.last_backend(), view.why()                   # "warp" or "rtx", and why RTX did not draw the last picture
    view.compare_backends(stage)                      # both pictures side by side, or one picture and the reason

Two programs can draw the stage (``ncntm.viz.stage``):

    "warp"   the Warp ray caster of this package: ``usd_to_emray.build_scene`` reads the composed
             stage and ``emray`` draws it, on a CUDA device or on the CPU.  Works everywhere.
    "rtx"    the RTX renderer of Isaac Sim, through ``session.capture_rtx`` of ``ncntm.viz.kit``.
             Needs Isaac Sim in tier "full" and a machine whose GPU can create a graphics context.
    "auto"   "rtx" when ``session.renderer_available`` says RTX can draw, else "warp".

A notebook must not stop because RTX is missing or fails: with ``backend="auto"`` the Warp
picture is returned and the reason is kept (``why()``).  Only ``backend="rtx"`` raises.

What RTX draws is the stage ISAAC SIM holds (``session.stage``), so for an RTX picture the stage
given here must be that stage, or share its root layer.  The lines that work in every tier:

    dataset.save()                                    # the file, with its textures next to it
    stage = session.open_stage(path)                  # in tier "off" this is a plain Usd.Stage.Open
    picture = view.render(stage)

view never opens a stage in Isaac Sim by itself: that replaces the stage Isaac Sim holds, whose
handles then are dead (using one aborts the process) and whose unsaved content is lost.  A stage
that Isaac Sim does not hold is drawn by Warp and ``why()`` says what to do.
Slices are textured quads.  RTX reads the PNG files ``./textures/*.png`` next to the stage file
(written by ``Dataset.add_slices``, copied by ``Dataset.save(path)`` but NOT by
``session.save(path)``); Warp reads the HDF5 sample instead.  ``missing_textures(stage)`` lists
the texture files RTX would not find.

The stage is left as it was found.  ``space`` and, for RTX, ``layers``, the camera and a light
are written as opinions into the SESSION layer of the stage (stronger than the file, never saved
with it) and removed again before ``render`` returns, also when the picture fails.  The root
layer and the selected variant are not touched, so ``dataset.space`` and ``dataset.set_space``
work as before.  (An opinion that somebody else put into the session layer on the same
attribute is kept, and put back after the picture.)

The Warp scene of a stage is kept and reused until the stage changes (``scene``; a USD notice
tells).  Measured for a whole sample of 3 um on the development host: building the scene costs
0.1 to 1.4 s on a CUDA device and 1 to 2 s on the CPU device, a 960 x 720 frame 8 to 40 ms on
CUDA and 0.5 to 3 s on the CPU.  At most ``MAX_SCENES`` scenes are kept; each holds its
triangles and volumes on the Warp device and a reference to its stage.  ``clear_cache()`` drops
them.  Not noticed: an HDF5 sample file that was rewritten while its scene is kept.

This module imports numpy and ``kit`` (standard library only) when it is loaded, and pxr, warp
and Pillow when a function needs them: ``from ncntm.viz import kit, view`` before
``kit.start_kit()`` is harmless.  Nothing here imports Isaac Sim.

TESTED AND NOT TESTED.  The Warp path is tested against ``usd_to_emray`` + ``emray`` used
directly, with and without Isaac Sim running (tests/test_view.py).  No RTX frame was ever
produced while this was written: the development host cannot create a graphics context.  The
RTX path was run with a stand-in for the renderer that draws the stage it is handed through the
camera it is handed (so the camera, the space, the layers and the clean-up are tested), and
against the real session up to its "no renderer" answer.  The light added for RTX
(``RTX_LIGHT_INTENSITY``) and everything about how an RTX picture looks are untested.
"""
from __future__ import annotations

import contextlib
import dataclasses
import warnings

import numpy as np

from . import kit
from .grid import SPACES

__all__ = ["render", "compare_backends", "last_backend", "why", "scene", "camera_for", "missing_textures",
           "clear_cache", "Picture", "NotHeld", "BACKENDS", "MAX_SCENES"]

BACKENDS = ("auto", "warp", "rtx")
MAX_SCENES = 6                              # Warp scenes kept, over all stages; the least recently used goes first
BUILD_OPTIONS = ("device", "volumes", "tube_sides", "sphere_subdiv")      # -> usd_to_emray.build_scene
CAMERA_OPTIONS = ("ssaa", "background")                                   # -> the emray.Camera of a Warp picture
RTX_OPTIONS = ("subframes", "mode", "spp", "light")                       # -> session.capture_rtx, and the light
RTX_CAMERA = "ncntm_view_camera"            # /World/Cameras/<this>: exists only while RTX draws
RTX_LIGHT = "/World/ncntm_view_light"       # a dome light, only while RTX draws and only if the stage has no light
# Intensity of that light.  With the exposure the stage's cameras are authored with (scale 1), a matte surface under
# a uniform white dome of intensity 1 should come out in its own colour and the background white, which is how the
# Warp picture looks.  This is reasoning from the USD conventions, NOT a measurement: no RTX frame was ever seen.
RTX_LIGHT_INTENSITY = 1.0

_LAST = {"backend": "", "why": ""}          # of the last picture
_WATCHES = []                               # one _Watch per stage that has scenes in _SCENES
_SCENES = []                                # [watch, key, scene, volumes], most recently used last


class Picture(np.ndarray):
    """What ``render`` returns: an (H, W, 4) uint8 RGBA array, row 0 at the top.

    A numpy array in every respect, plus what a notebook wants from a picture: it shows inline
    in Jupyter, ``picture.save(path)`` writes a PNG, ``picture.backend`` is "warp" or "rtx" and
    ``picture.why`` says why RTX did not draw it ("" if it did).  ``picture.rgba`` is the plain
    array, so every function of ``figure`` takes a Picture as it takes an ``emray.Image``."""

    def __new__(cls, rgba, backend: str = "", why: str = ""):
        picture = np.ascontiguousarray(rgba, dtype=np.uint8).view(cls)
        picture.backend, picture.why = backend, why
        return picture

    def __array_finalize__(self, source):
        self.backend, self.why = getattr(source, "backend", ""), getattr(source, "why", "")

    @property
    def rgba(self) -> np.ndarray:
        return np.asarray(self)

    def _repr_png_(self):
        if self.ndim != 3 or self.shape[2] not in (3, 4) or 0 in self.shape:      # a slice of a picture that is no picture
            return None
        from . import figure
        return figure.to_png_bytes(self.rgba)

    def save(self, path):
        """Write the picture as a PNG file; returns ``path``."""
        from . import figure
        figure.to_pil(self.rgba).save(path, format="PNG")
        return path


def last_backend() -> str:
    """Who drew the last picture: "warp" or "rtx" ("" before the first one)."""
    return _LAST["backend"]


def why() -> str:
    """Why RTX did not draw the last picture, in one sentence ("" if it did)."""
    return _LAST["why"]


# ---- the stage ------------------------------------------------------------------------------------
def _stage(stage_or_dataset):
    """The Usd.Stage meant: the stage itself, or the stage of a ``stage.Dataset``."""
    stage = getattr(stage_or_dataset, "stage", stage_or_dataset)
    if stage is None or not hasattr(stage, "GetRootLayer"):
        raise TypeError(f"render needs a Usd.Stage or a stage.Dataset, got {type(stage_or_dataset).__name__}")
    if not stage:                               # an expired handle: every method call on it would abort the process
        raise RuntimeError("this stage handle is dead: Isaac Sim has replaced its stage since (new_stage / open_stage). "
                           "Take the stage again: stage = session.stage")
    return stage


def _spaces(stage, space) -> tuple:
    """((dataset path, space as it is to be drawn), ...) for the datasets that have the "space" variant set."""
    from .stage import SPACE_VARIANT_SET, find_datasets
    if space is not None and space not in SPACES:
        raise ValueError(f"space must be None (as selected on the stage) or one of {SPACES}, got {space!r}")
    return tuple(sorted((str(ds.prim.GetPath()), space or ds.space) for ds in find_datasets(stage)
                        if ds.prim.GetVariantSets().HasVariantSet(SPACE_VARIANT_SET)))


def _layers(stage, layers):
    """``layers`` as a tuple of layer names, or None for all; a name the stage and the package
    do not know raises ValueError (it would give an empty picture without a word)."""
    from .stage import LAYERS
    if layers is None:
        return None
    layers = (layers,) if isinstance(layers, str) else tuple(layers)
    on_stage = {prim.GetAttribute("em:layer").Get() for prim in stage.Traverse() if prim.HasAttribute("em:layer")}
    unknown = [name for name in layers if name not in LAYERS and name not in on_stage and name != ""]
    if unknown:
        raise ValueError(f"unknown layer(s) {unknown}: this stage has {sorted(on_stage)}, the package's layers "
                         f"are {list(LAYERS)} ('' selects prims without a layer tag)")
    return layers


def missing_textures(stage_or_dataset) -> list:
    """Paths of the texture files the stage refers to and that are NOT where it says: RTX would
    draw those slices without their picture (Warp does not use them).  They are written next to
    the stage file by ``Dataset.add_slices`` and travel with ``Dataset.save(path)``."""
    import os
    missing = []
    for prim in _stage(stage_or_dataset).Traverse():
        attribute = prim.GetAttribute("inputs:file")
        asset = attribute.Get() if attribute else None
        if asset is not None and asset.path and not (asset.resolvedPath and os.path.isfile(asset.resolvedPath)):
            missing.append(asset.path)
    return missing


class _Watch:
    """Listens to one stage: a change that view did not make itself drops the scenes kept for it."""

    def __init__(self, stage):
        from pxr import Tf, Usd
        self.stage, self.own = stage, 0             # own > 0: view is editing the session layer of this stage
        self.listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._changed, stage)

    def _changed(self, notice, sender):
        if not self.own:
            _forget(self)

    @staticmethod
    def of(stage, create=False):
        for watch in list(_WATCHES):
            if not watch.stage:                     # its stage is gone (Isaac Sim opened another one)
                _forget(watch)
        for watch in _WATCHES:
            if watch.stage == stage:
                return watch
        if create:
            _WATCHES.append(_Watch(stage))
            return _WATCHES[-1]
        return None


def _forget(watch) -> None:
    """Drop the scenes of one stage and stop listening to it."""
    _SCENES[:] = [entry for entry in _SCENES if entry[0] is not watch]
    if watch in _WATCHES:
        _WATCHES.remove(watch)
        watch.listener.Revoke()


def clear_cache() -> None:
    """Drop every kept Warp scene (their memory on the Warp device is freed with them)."""
    for watch in list(_WATCHES):
        _forget(watch)
    del _SCENES[:]


@contextlib.contextmanager
def _temporarily(stage, paths):
    """Yields the SESSION layer of ``stage`` for opinions that last as long as the block: on
    exit the session layer is as it was before.  ``paths`` are the prims that will be touched:
    what the session layer says below their root prims is copied aside first and put back
    afterwards.  Author with ``with Usd.EditContext(stage, layer):``."""
    from pxr import Sdf
    layer = stage.GetSessionLayer()
    roots = sorted({Sdf.Path(str(path)).GetPrefixes()[0] for path in paths})
    aside = Sdf.Layer.CreateAnonymous("ncntm_view_aside")
    for root in roots:
        if layer.GetPrimAtPath(root):
            Sdf.CopySpec(layer, root, aside, root)
    watch = _Watch.of(stage)
    if watch is not None:
        watch.own += 1                              # these edits are no reason to drop the kept scenes
    try:
        yield layer
    finally:
        try:
            with Sdf.ChangeBlock():
                for root in roots:
                    if layer.GetPrimAtPath(root):
                        del layer.rootPrims[root.name]
                    if aside.GetPrimAtPath(root):
                        Sdf.CopySpec(aside, root, layer, root)
        finally:
            if watch is not None:
                watch.own -= 1


def _to_switch(stage, spaces) -> list:
    """[(Dataset, space), ...] for the datasets of ``stage`` that are not selected as ``spaces`` wants them drawn."""
    from .stage import Dataset
    datasets = [(Dataset.from_prim(stage.GetPrimAtPath(path)), space) for path, space in spaces]
    return [(dataset, space) for dataset, space in datasets if dataset.space != space]


# ---- Warp -----------------------------------------------------------------------------------------
def scene(stage_or_dataset, space=None, layers=None, device=None, volumes=None, tube_sides: int = 10, sphere_subdiv: int = 2):
    """The ``emray.Scene`` of the stage in ``space``, built on ``device`` and ready to render:
    the kept one if the stage has not changed since it was built, else a new one.

    For what ``render`` does not return, the depth and object-id buffers:
    ``view.scene(stage).render(view.camera_for(stage))`` is an ``emray.Image``.
    Arguments as for ``usd_to_emray.build_scene``; ``space`` as for ``render``."""
    from pxr import Usd
    from . import emray, usd_to_emray
    stage = _stage(stage_or_dataset)
    spaces, layers = _spaces(stage, space), _layers(stage, layers)
    device = emray.default_device() if device is None else str(device)
    key = (spaces, layers, device, int(tube_sides), int(sphere_subdiv),
           None if volumes is None else tuple(sorted((name, id(value)) for name, value in volumes.items())))
    watch = _Watch.of(stage, create=True)
    for entry in _SCENES:
        if entry[0] is watch and entry[1] == key:
            _SCENES.remove(entry)
            _SCENES.append(entry)                   # most recently used last
            return entry[2]
    switch = _to_switch(stage, spaces)
    try:
        with _temporarily(stage, [dataset.prim.GetPath() for dataset, _ in switch]) as session_layer:
            with Usd.EditContext(stage, session_layer):
                for dataset, wanted in switch:
                    dataset.set_space(wanted)
            built = usd_to_emray.build_scene(stage, layers=layers, device=device, volumes=volumes,
                                             tube_sides=tube_sides, sphere_subdiv=sphere_subdiv)
    except BaseException:
        if not any(entry[0] is watch for entry in _SCENES):
            _forget(watch)                          # nothing is kept for this stage: do not hold on to it
        raise
    _SCENES.append([watch, key, built, volumes])    # (volumes is kept so that the ids in the key stay unique)
    for entry in _SCENES[:-MAX_SCENES]:
        _SCENES.remove(entry)
        if not any(other[0] is entry[0] for other in _SCENES):
            _forget(entry[0])
    return built


def camera_for(stage_or_dataset, camera=None, size=(960, 720), ssaa: int = 2, background=(1.0, 1.0, 1.0, 1.0)):
    """The ``emray.Camera`` a picture of ``size`` = (width, height) pixels is drawn with.

    camera   None: /World/Cameras/overview, else the first camera of the stage, else one that
             frames everything on the stage (``usd_to_emray.default_camera``);
             a path: that camera prim of the stage;
             an ``emray.Camera``: used as it is, with ITS width and height (``size`` is ignored).
    The vertical field of view is the camera's.  A picture wider or narrower than the camera's
    aperture (4:3 for the cameras ``Dataset.author_camera`` writes) shows more or less at the sides."""
    from . import emray, usd_to_emray
    stage = _stage(stage_or_dataset)
    if isinstance(camera, emray.Camera):
        return camera.check()
    width, height = (int(n) for n in size)
    if width < 1 or height < 1:
        raise ValueError(f"size must be (width, height) in pixels, got {size!r}")
    if camera is None:
        found = usd_to_emray.default_camera(stage, height=height, ssaa=ssaa, background=tuple(background))
    else:
        found = usd_to_emray.camera_from_stage(stage, camera, height=height, ssaa=ssaa, background=tuple(background))
    return dataclasses.replace(found, width=width)


# ---- RTX ------------------------------------------------------------------------------------------
class NotHeld(RuntimeError):
    """The stage to draw is not the stage Isaac Sim holds (the message says what to do)."""


def _kit_stage(stage, session):
    """The stage Isaac Sim holds, if it shows the root layer of ``stage``; else NotHeld."""
    held = session.stage
    name = stage.GetRootLayer().realPath or "an in-memory stage"
    if held is None:
        raise NotHeld(f"Isaac Sim holds no stage, and RTX draws the stage Isaac Sim holds. Save this one ({name}) and "
                      "open it there: stage = session.open_stage(path)")
    if held.GetRootLayer().identifier != stage.GetRootLayer().identifier:
        raise NotHeld(f"Isaac Sim holds another stage ({held.GetRootLayer().realPath or 'an in-memory stage'}) than "
                      f"the one to draw ({name}), and RTX draws the stage Isaac Sim holds. Open this one there: "
                      "stage = session.open_stage(path). That replaces the stage Isaac Sim holds now: save that one "
                      "first and do not use handles to it afterwards")
    return held


def _hidden(stage, layers) -> list:
    """Paths of the prims to hide so that only ``layers`` are drawn: the groups tagged with another
    layer, and geometry without a tag unless "" is asked for (what ``build_scene(layers=...)`` leaves out)."""
    from pxr import Usd, UsdGeom
    hide = []
    prims = iter(Usd.PrimRange(stage.GetPseudoRoot()))
    for prim in prims:
        tag = prim.GetAttribute("em:layer").Get() if prim.HasAttribute("em:layer") else None
        if tag:
            if tag not in layers:
                hide.append(prim.GetPath())
            prims.PruneChildren()                       # a layer group is shown or hidden as a whole
        elif prim.IsA(UsdGeom.Gprim) and "" not in layers:
            hide.append(prim.GetPath())
    return hide


def _render_rtx(stage, camera, spaces, layers, session, subframes: int = 32, mode: str = "RealTimePathTracing",
                spp=None, light: float = None):
    """One RTX frame of the stage Isaac Sim holds, in ``spaces``, with only ``layers`` visible,
    seen through ``camera`` (an emray.Camera).  Everything this needs on the stage is authored in
    its session layer and taken back: the selected space, the hidden layers, a camera prim that
    is the USD twin of ``camera``, and a white dome light if the stage has no light at all.

    UNTESTED with a real RTX renderer (see the module docstring)."""
    from pxr import Usd, UsdGeom, UsdLux
    from . import stage as S
    held = _kit_stage(stage, session)
    right, up, forward = camera.basis()
    eye = np.asarray(camera.eye, dtype=np.float64)
    # the clipping range is authored from the eye-target distance: put the target behind the middle of the scene
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]) \
        .ComputeWorldBound(held.GetPseudoRoot()).ComputeAlignedRange()
    reach = 1.0 if bound.IsEmpty() else float(np.linalg.norm(np.array(bound.GetMidpoint()) - eye)
                                              + 0.5 * np.linalg.norm(np.array(bound.GetSize())))
    hide = [] if layers is None else _hidden(held, layers)
    switch = _to_switch(held, spaces)
    with _temporarily(held, [dataset.prim.GetPath() for dataset, _ in switch] + hide + ["/World"]) as session_layer:
        with Usd.EditContext(held, session_layer):
            for dataset, wanted in switch:
                dataset.set_space(wanted)
            for path in hide:
                UsdGeom.Imageable(held.GetPrimAtPath(path)).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            twin = S.author_camera(held, RTX_CAMERA, eye=eye, target=eye + forward * max(reach, 1e-6), up=up,
                                   projection=camera.mode, fov=camera.fov, ortho_height=camera.ortho_height,
                                   aspect=camera.width / camera.height).GetPath()
            if not any(prim.HasAPI(UsdLux.LightAPI) for prim in held.Traverse()):
                dome = UsdLux.DomeLight.Define(held, RTX_LIGHT)
                dome.CreateIntensityAttr(RTX_LIGHT_INTENSITY if light is None else float(light))
        session.update(2)                               # let Kit take the changes in before the frame is drawn
        frame = session.capture_rtx(str(twin), width=int(camera.width), height=int(camera.height),
                                    subframes=subframes, mode=mode, spp=spp)
    session.update(1)                                   # ... and their removal after it
    return frame


def _no_rtx(session) -> str:
    """Why RTX cannot draw in this process ("" if it can)."""
    if session is None:
        return "no Isaac Sim session in this process (kit.start_kit() was not called)"
    status = session.renderer_status()
    return "" if status.available else status.reason


# ---- the picture ----------------------------------------------------------------------------------
def render(stage_or_dataset, camera=None, size=(960, 720), space=None, layers=None, backend: str = "auto",
           session=None, **options) -> Picture:
    """A picture of the stage: a ``Picture``, i.e. an (H, W, 4) uint8 RGBA array.

    stage_or_dataset   a Usd.Stage, or a ``stage.Dataset`` (which stands for its stage)
    camera, size       see ``camera_for``; size is (width, height) in pixels
    space      None: as selected on the stage.  "physical" or "index": every dataset of the stage
               is drawn in that space for this picture; the selection on the stage is not changed
    layers     None: everything visible.  Else the layer names to draw, e.g. ["rawvalue", "frame"]
               ("" = prims without a layer tag); the visibility on the stage is not changed
    backend    "auto" (RTX if it can draw, else Warp), "warp", or "rtx" (raises RuntimeError
               "no renderer: ..." if RTX cannot draw, ``NotHeld`` if Isaac Sim holds another
               stage, and whatever the capture raises)
    session    the ``kit.Session``; default ``kit.current_session()``
    options    for Warp: ``device``, ``volumes``, ``tube_sides``, ``sphere_subdiv`` (build_scene),
               ``ssaa``, ``background`` (camera) and the keywords of ``emray.Scene.render``;
               for RTX: ``subframes``, ``mode``, ``spp`` (``session.capture_rtx``) and ``light``
               (intensity of the dome light added when the stage has none).  Options of the
               backend that does not draw are ignored.

    Afterwards ``last_backend()`` says who drew the picture and ``why()`` why it was not RTX;
    the same two texts are ``picture.backend`` and ``picture.why``."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    stage = _stage(stage_or_dataset)
    spaces, layers = _spaces(stage, space), _layers(stage, layers)
    build = {name: options.pop(name) for name in BUILD_OPTIONS if name in options}
    look = {name: options.pop(name) for name in CAMERA_OPTIONS if name in options}
    rtx = {name: options.pop(name) for name in RTX_OPTIONS if name in options}
    view_camera = camera_for(stage, camera, size, **look)
    reason = 'backend="warp" was asked for'
    if backend != "warp":
        session = kit.current_session() if session is None else session
        reason = _no_rtx(session)
        if reason and backend == "rtx":
            raise RuntimeError("no renderer: " + reason)
        if not reason:
            try:
                frame = _render_rtx(stage, view_camera, spaces, layers, session, **rtx)
            except NotHeld as error:                              # not a failure: the message says what to do
                if backend == "rtx":
                    raise
                reason = str(error)
            except Exception as error:
                if backend == "rtx":
                    raise
                reason = f"RTX did not deliver the picture ({type(error).__name__}: {error})"
                warnings.warn(reason + "; the Warp ray caster drew it instead.", RuntimeWarning, stacklevel=2)
            else:
                _LAST.update(backend="rtx", why="")
                return Picture(frame, "rtx", "")
    image = scene(stage, space, layers, **build).render(view_camera, **options)
    _LAST.update(backend="warp", why=reason)
    return Picture(image.rgba, "warp", reason)


def compare_backends(stage_or_dataset, camera=None, size=(960, 720), space=None, layers=None, session=None,
                     text_size: int = None, **options):
    """The picture to look at first on a new machine: the Warp picture and the RTX picture of
    the same stage through the same camera, side by side with captions; or, where RTX cannot
    draw, the Warp picture alone with the reason in its caption.  Returns a ``PIL.Image``
    (shows inline in Jupyter, has ``.save(path)``).  Arguments as for ``render``.

    The Warp picture is the reference: it is computed from the composed stage by code that is
    tested against the arrays of the sample.  In the RTX picture the outlines must fall on the
    same pixels; how its colours and shading compare was never seen (no RTX frame was produced
    while this was written), so the caption reports the mean difference instead of judging it."""
    from . import emray, figure
    stage = _stage(stage_or_dataset)
    warp = render(stage, camera, size, space, layers, backend="warp", **options)
    height, width = warp.shape[:2]
    drawn = ", ".join(sorted({s for _, s in _spaces(stage, space)})) or "no dataset on the stage"
    told = f"{drawn} space" + (" (as selected on the stage)" if space is None else "")
    device = options.get("device") or emray.default_device()
    text_size = max(14, width // 60) if text_size is None else text_size
    first = (f"Warp ray caster on {device}: {width} x {height} pixels, {told}. This is the reference picture, "
             "computed from the USD stage.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rtx = render(stage, camera, size, space, layers, backend="rtx", session=session, **options)
    except Exception as error:
        text = str(error)
        if text.startswith("no renderer: "):
            reason = text[len("no renderer: "):]
        else:
            reason = text if isinstance(error, NotHeld) else f"RTX did not deliver the picture ({type(error).__name__}: {text})"
        _LAST.update(backend="warp", why=reason)
        return figure.caption(warp, f"{first} There is no RTX picture beside it: {reason}", size=text_size)
    second = (f"RTX ({options.get('mode', 'RealTimePathTracing')}, {options.get('subframes', 32)} sub-frames): the same "
              "stage through the same camera. Look at the outlines first: box, slices and objects must sit where they "
              "sit in (a).")
    if rtx.shape == warp.shape:
        difference = np.abs(rtx[..., :3].astype(np.int16) - warp[..., :3].astype(np.int16))
        second += (f" Mean absolute difference to (a): {difference.mean():.1f} of 255 grey levels; "
                   f"{100.0 * np.mean(difference.max(axis=2) > 32):.1f} % of the pixels differ by more than 32.")
    else:
        second += f" RTX delivered {rtx.shape[1]} x {rtx.shape[0]} pixels."
    missing = missing_textures(stage)
    if missing:
        second += f" {len(missing)} texture file(s) of the slices are not next to the stage file: RTX cannot show those slices."
    return figure.side_by_side(warp, rtx, captions=[first, second], labels=True, size=text_size)
