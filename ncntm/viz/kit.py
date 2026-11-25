"""kit -- the one place in ``ncntm.viz`` that knows about NVIDIA Isaac Sim / Omniverse Kit.

Nothing else in the package imports ``isaacsim`` or ``omni``.  A notebook starts with

    from ncntm.viz import kit
    session = kit.start_kit()            # tier "auto"; re-running the cell returns the same session
    print(session.describe())
    stage = session.open_stage("scene.usda")        # or session.new_stage()

and from there on uses plain ``pxr`` on ``session.stage``.  The same lines run on a machine
without Isaac Sim (tier "off": ``session.app`` is None and the stages are plain pxr stages).

ORDER MATTERS: these lines run before anything imports ``pxr`` or ``warp``.
``ncntm.viz.stage`` and ``ncntm.viz.usd_to_emray`` import pxr, ``ncntm.viz.emray`` imports warp,
so they are imported AFTER ``kit.start_kit()``, on a later line or in a later cell
(``from ncntm.viz import kit, stage`` on one line is the mistake to avoid).  If the order is
wrong, start_kit says so and names the module that came too early.  (pxr first breaks every
Kit tier; warp first breaks the tiers that bring their own Warp, "core" and "full".)

Tiers
    "off"    no Kit.  Plain OpenUSD from the interpreter.  Always possible.
    "usd"    Kit + omni.usd + the ParticleField (Gaussian splat) schema, nothing that needs a
             graphics device (apps/ncntm.kit_usd.kit).  About half a second.
    "core"   optional: "usd" + Fabric/USDRT + isaacsim.core.experimental + PhysX, no renderer
             (apps/ncntm.kit_core.kit).  About three seconds.
    "full"   the stock Isaac Sim application with its RTX renderer: needed for capture_rtx,
             the viewport and livestream.  Tens of seconds.
    "auto"   "full" when a hardware graphics device can be initialised (see probe_graphics),
             otherwise "usd"; "off" when Isaac Sim is not installed.  The environment variable
             NCNTM_KIT (off | usd | core | full) overrides the probe; an explicit ``tier``
             argument overrides both.
Which Warp ``import warp`` gives afterwards depends on the tier (``session.warp_version``): in
"off" and "usd" the interpreter's own (1.19.0 in the pip install; a source build has none
unless one is installed into it), in "core" and "full" the one Kit ships (1.17.0).  The Warp
ray caster of this package (ncntm.viz.emray) must therefore run with both.

What start_kit takes care of (each line is a failure that was measured with Isaac Sim 6.2.0
inside a Jupyter kernel; do not remove one without measuring again)
    * ``pxr`` imported before Kit: Kit ships its own OpenUSD (25.11) and breaks when another
      build (26.08 in the same interpreter) is already loaded.  start_kit refuses to start.
    * ``warp`` imported before Kit: tiers "core" and "full" load the Warp that Isaac Sim ships
      (1.17.0, extension omni.warp.core).  With the interpreter's Warp (1.19.0) already
      imported the two are mixed and the first Warp array fails with an ImportError.
      start_kit refuses to start those tiers; tier "usd" has no Warp of its own and only
      warns that the order would not work elsewhere.
    * the licence: ``import isaacsim`` asks for it on stdin, which ends a kernel.  start_kit
      checks first and raises a readable error.  It never accepts the licence for you.
    * nest_asyncio: applied by ``import isaacsim`` when it sees Jupyter; mandatory.  start_kit
      sets ISAAC_JUPYTER_KERNEL=1 inside a kernel so that this does not depend on the front end.
    * sys.argv: SimulationApp forwards it to Kit.  The kernel's "-f <connection file>" is
      removed and "--portable-root DIR" is added; without it Kit writes logs, cache and data
      INTO the Isaac Sim installation.  DIR is ``root``, else NCNTM_KIT_ROOT, else
      ~/.cache/ncntm/kit.
    * TMPDIR: while Kit starts it points to DIR/tmp, which is where Kit's native code then
      keeps its temporary files (otherwise /tmp/carb.* and /var/tmp/st_python*).  Afterwards
      the variable is put back: left in place it broke Warp, whose kernel compiler (NVRTC of
      CUDA 12.9, the one of Kit's Warp 1.17) refuses a TMPDIR longer than about 120
      characters ("name of directory for temporary files is too long").
    * the experience files: apps/*.kit are ignored by the repository (.gitignore has *.kit),
      so a clone does not have them.  Their text is kept in this module as well; a missing
      file is written to DIR/apps and used from there.
    * a folder nobody may list: the stock application creates
      DIR/data/documents/Kit/shared/screenshots with mode 000, at which ``rm -r`` and ``du``
      give up.  start_kit gives the owner access to it.
    * print(): Kit replaces sys.stdout unless --/app/python/interceptSysStdOutput=false.
    * crash reporter and telemetry: stock Kit uploads crash dumps and usage events.  Every
      start passes PRIVACY_ARGS, excludes the telemetry extension and does not load the crash
      reporter plugin.
    * CUDA: SimulationApp's ``active_gpu`` does not restrict CUDA.  With ``gpu=N`` start_kit
      sets CUDA_VISIBLE_DEVICES=N before the first CUDA use (the GPU is then "cuda:0") and
      warns if CUDA was initialised earlier.  On a machine with a single GPU ``gpu=0`` sets
      nothing: there is nothing to restrict, and RTX warns against the variable.
    * a second SimulationApp silently replaces the open stage: start_kit returns the running
      session instead.
    * SimulationApp.close() deadlocks inside a kernel: ``session.close()`` only explains.
    * leaving: Kit's own teardown at interpreter exit crashes or outlives Jupyter's grace
      period.  An exit hook flushes open files, closes h5py files, shuts logging down,
      removes the folders of tempfile.TemporaryDirectory objects (Warp keeps a 70 MB
      precompiled header in one) and then leaves with os._exit.  RLIMIT_CORE is set to 0 so
      that a crash leaves no core file.
    * Ctrl-C / "interrupt" while Kit starts leaves a half-initialised application.  The
      interrupt is held back until the start has finished; if a start fails anyway the
      session is marked tainted and start_kit asks for a kernel restart.
    * stage handles die after new_stage / open_stage: ``session.stage`` always asks Kit again.
    * "Run All": with ipykernel 7 the cells queued behind a cell that drives Kit were silently
      dropped.  protect_queued_cells() keeps them (see there; NCNTM_KIT_PROTECT_QUEUE=0 leaves
      asyncio untouched).

Not done here on purpose: nothing is saved at kernel exit (call ``session.save``), and atexit
handlers registered before start_kit do not run (the exit hook leaves the process first).
Nothing prunes the Kit folder either: every start leaves one log in DIR/logs (about 0.2 MB in
tier "usd", 0.3 MB in "core", 1.3 MB in "full") and an empty carb.* folder in DIR/tmp;
delete DIR whenever no kernel is running.  Two kernels may share DIR; if they start within the
same second they also share the log file, and ``session.kit_errors()`` then counts both.
Tier "off" changes nothing about how the process ends; inside a kernel it only sets the soft
core-file limit to 0 (see start_kit).

The RTX path (``capture_rtx``, ``start_livestream``, a positive ``renderer_available``) could
not produce a picture on the development host: its GPUs cannot create a graphics context.
The names were read in the installed Isaac Sim 6.2.0 files, and the capture sequence was run
there in the stock application up to the point where the renderer returns an empty frame;
what a working renderer returns is untested.
"""
from __future__ import annotations

import atexit
import builtins
import collections
import contextlib
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import weakref
from dataclasses import dataclass, field

TIERS = ("off", "usd", "core", "full")
APPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps")
EXPERIENCES = {"usd": "ncntm.kit_usd.kit", "core": "ncntm.kit_core.kit", "full": ""}   # "" = stock isaacsim.exp.base.python.kit
KIT_WARP_TIERS = ("core", "full")               # tiers in which Kit loads its own Warp (extension omni.warp.core)
ENV_TIER, ENV_ROOT, ENV_GPU = "NCNTM_KIT", "NCNTM_KIT_ROOT", "NCNTM_KIT_GPU"
ENV_PROTECT = "NCNTM_KIT_PROTECT_QUEUE"         # "0": do not touch asyncio (see protect_queued_cells)
EGL_VENDOR_VARIABLE = "__EGL_VENDOR_LIBRARY_DIRS"       # libglvnd: where EGL looks for drivers; conda sets it to its own folder
EULA_VARIABLE = "OMNI_KIT_ACCEPT_EULA"
EULA_URL = "https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html"
RENDER_MODES = ("RealTimePathTracing", "PathTracing")

# Nothing leaves the machine: no crash-dump upload, no usage telemetry, no structured log.
PRIVACY_ARGS = (
    "--/crashreporter/enabled=false",
    "--/crashreporter/skipOldDumpUpload=true",
    "--/telemetry/enableAnonymousData=false",
    "--/telemetry/enableAnonymousAppName=false",
    "--/structuredLog/enable=false",
)
PRIVACY_EXCLUDED_EXTENSIONS = ("omni.kit.telemetry",)
STDOUT_ARG = "--/app/python/interceptSysStdOutput=false"          # keep print() in the cell that prints
QUIET_ARGS = ("--/log/outputStreamLevel=Error", "--/app/enableStdoutOutput=false")


def excluded_extensions_arg(names=PRIVACY_EXCLUDED_EXTENSIONS) -> str:
    return "--/app/extensions/excluded=[" + ",".join(f'"{n}"' for n in names) + "]"


# The experience files apps/*.kit, as text.  The files are what Kit is started with when they are there; this copy
# is what start_kit writes to <root>/apps when they are not (the repository ignores *.kit, so a clone has none).
# tests/test_kit.py compares the two: change them together.
EXPERIENCE_TEXT = {
    "usd": r'''# ncntm tier "usd": Isaac Sim's Kit runtime with omni.usd and nothing that needs a graphics device.
#
# What you get: the UsdContext (new / open / save stage, stage events, omni.kit.commands with undo,
# timeline), Kit's OpenUSD 25.11 and the ParticleField (3D Gaussian splat) schema for it.
# What you do not get: omni.gpu_foundation, RTX, viewport, Replicator, omni.ui, PhysX, Fabric.
# Measured on the development host (Isaac Sim 6.2.0 pip install): 24 extensions, 0.3 - 0.5 s, 0 errors.
#
# This file contains NO path.  Extension folders are given on the command line by
# ncntm.viz.kit.start_kit():  SimulationApp passes  --ext-folder $ISAAC_PATH/exts  and
# --ext-folder $ISAAC_PATH/apps  itself, and start_kit adds  --ext-folder $ISAAC_PATH/extscache
# (see kit.isaac_extension_folders).  ISAAC_PATH is set by "import isaacsim" (pip install) or
# by python.sh (binary and source builds).
# The same text is kept in ncntm/viz/kit.py (EXPERIENCE_TEXT): start_kit writes this file from it where the
# file is missing (the repository ignores *.kit).  Change both together; tests/test_kit.py compares them.
[package]
title = "ncntm Kit USD (no renderer)"
description = "Kit + omni.usd: stage authoring and inspection without gpu_foundation, RTX, viewport or UI"
version = "0.2.0"
keywords = ["app", "usd"]

[dependencies]
"isaacsim.simulation_app" = {}              # the SimulationApp helper (depends on omni.usd)
"omni.usd" = {}                             # UsdContext, stage events, omni.kit.commands, undo, timeline
"omni.kit.usd.layers" = {}                  # imported by isaacsim.simulation_app.utils
"omni.kit.loop-isaac" = {}                  # Isaac loop runner (the notebook steps Kit by hand)
"omni.kit.primitive.mesh" = {}              # CreateMeshPrim command
"omni.usd.schema.usd_particle_field" = {}   # ParticleField3DGaussianSplat schema for Kit's OpenUSD 25.11

[settings.app]
name = "ncntm-kit-usd"
version = "0.2.0"
content.emptyStageOnStart = false
fastShutdown = true
file.ignoreUnsavedStage = true
hangDetector.enabled = false
settings.persistent = false
enableDeveloperWarnings = false
runLoops.main.rateLimitEnabled = false
useFabricSceneDelegate = false          # no usdrt in this tier
extensions.registryEnabled = false      # never try to download extensions

[settings.exts."omni.usd"]
locking.onClose = false

[settings.persistent]
app.stage.upAxis = "Z"
simulation.defaultMetersPerUnit = 1.0
''',
    "core": r'''# ncntm tier "core" (optional): tier "usd" plus Fabric / USDRT and the isaacsim.core.experimental
# stage, prim and xform wrappers, which bring PhysX and Kit's bundled Warp with them.
#
# The GPU foundation is loaded but never asked for a rendering device: no RTX, no viewport, no UI.
# Measured on the development host (Isaac Sim 6.2.0 pip install): 69 extensions, 3.1 s, 0 errors.
# Differences to tier "usd" that a notebook can see: "import warp" gives Kit's Warp (1.17.0)
# instead of the pip one, and playing the timeline adds a /PhysicsScene prim to the stage.
# ncntm.viz itself needs nothing from this tier; it is here for people who want the Isaac Sim
# core API next to the same stage.
#
# This file contains NO path; see ncntm.kit_usd.kit for how extension folders are passed.
# The same text is kept in ncntm/viz/kit.py (EXPERIENCE_TEXT): start_kit writes this file from it where the
# file is missing (the repository ignores *.kit).  Change both together; tests/test_kit.py compares them.
[package]
title = "ncntm Kit core (no renderer)"
description = "Kit + omni.usd + usdrt + isaacsim.core.experimental.* without RTX, viewport or UI"
version = "0.2.0"
keywords = ["app", "usd"]

[dependencies]
"isaacsim.simulation_app" = {}
"omni.usd" = {}
"omni.kit.usd.layers" = {}
"omni.kit.loop-isaac" = {}
"omni.kit.primitive.mesh" = {}
"omni.usd.schema.usd_particle_field" = {}
"usdrt.scenegraph" = {}                     # Fabric / USDRT
"omni.hydra.usdrt_delegate" = {}            # usdrt population utilities (needed by useFabricSceneDelegate)
"isaacsim.core.experimental.utils" = {}     # stage / prim / xform helpers (loads omni.warp.core)
"isaacsim.core.experimental.prims" = {}     # XformPrim, GeomPrim ... (loads PhysX and the simulation manager)
"isaacsim.core.version" = {}

[settings.app]
name = "ncntm-kit-core"
version = "0.2.0"
content.emptyStageOnStart = false
fastShutdown = true
file.ignoreUnsavedStage = true
hangDetector.enabled = false
settings.persistent = false
enableDeveloperWarnings = false
runLoops.main.rateLimitEnabled = false
useFabricSceneDelegate = true
extensions.registryEnabled = false      # never try to download extensions

[settings.exts."omni.usd"]
locking.onClose = false

[settings.persistent]
app.stage.upAxis = "Z"
simulation.defaultMetersPerUnit = 1.0
''',
}


# ---- process-wide state ---------------------------------------------------------------------
def _state() -> dict:
    """One dictionary per PROCESS, kept on ``builtins`` so that reloading this module
    (%autoreload) cannot forget the running application and start a second one."""
    return builtins.__dict__.setdefault(
        "_NCNTM_KIT_STATE", {"session": None, "tainted": "", "probe": None, "exit_code": None, "hook": False})


def current_session():
    """The session of this process, or None before the first start_kit()."""
    return _state()["session"]


def in_jupyter() -> bool:
    """True when THIS process is an IPython kernel (Jupyter, VS Code, nbconvert).

    Not true for a script that a notebook starts ("!python script.py", subprocess): such a
    child inherits JPY_PARENT_PID from the kernel but is an ordinary script, and must keep an
    ordinary script's exit status."""
    kernelapp = sys.modules.get("ipykernel.kernelapp")
    if kernelapp is not None:
        try:
            return bool(kernelapp.IPKernelApp.initialized())
        except Exception:
            return True
    shell = getattr(builtins, "get_ipython", None)              # a kernel that is not ipykernel's own application
    try:
        return shell is not None and hasattr(shell(), "kernel")
    except Exception:
        return False


# ---- is Isaac Sim there, and may it be used? ----------------------------------------------------
def isaacsim_install() -> dict:
    """Where the ``isaacsim`` Python package is, WITHOUT importing it (importing it loads
    Carbonite, asks for the licence and can end the process).

    Returns {"found": bool, "package": folder, "found_at": folder, "layout": "pip" | "build" | ""}.
    "pip" is the wheel layout (<site-packages>/isaacsim with kit/, exts/, extscache/ inside);
    "build" is a binary or source build, where the package is <root>/python_packages/isaacsim
    and python.sh exports ISAAC_PATH, CARB_APP_PATH and EXP_PATH.

    ``package`` has symbolic links resolved, as ``isaacsim/__init__.py`` itself does to tell
    the layouts apart; ``found_at`` is the folder as it stands on sys.path.  They differ in a
    SOURCE build, where <repo>/_build/<platform>/<config>/python_packages is a link to
    <repo>/source/python_packages (premake5.lua of Isaac Sim 6.2.0)."""
    try:
        spec = importlib.util.find_spec("isaacsim")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.origin:
        return {"found": False, "package": "", "found_at": "", "layout": ""}
    found_at = os.path.dirname(os.path.abspath(spec.origin))
    package = os.path.dirname(os.path.abspath(os.path.realpath(spec.origin)))
    build = package.split(os.sep)[-2:] == ["python_packages", "isaacsim"]
    return {"found": True, "package": package, "found_at": found_at, "layout": "build" if build else "pip"}


def licence_markers(install: dict = None) -> list:
    """The files by which Isaac Sim itself records that the licence was accepted.

    pip install      <package>/kit/EULA_ACCEPTED, first line "yes" (kit_app.check_eula; without
                     it and without the variable ``import isaacsim`` asks on stdin).
    source build     <repo>/.eula_accepted, an empty file written by build.sh / repo.sh through
                     tools/eula_check.sh.  ``import isaacsim`` does not ask in a build at all.
    binary build     nothing: the licence is accepted at download.  Only the variable counts.

    A LAYOUT ASSUMPTION (read in the Isaac Sim 6.2.0 source, run only in a stand-in tree): a
    source build is <repo>/_build/<platform>/<config> (the folder that holds python.sh, which
    python.sh exports as ISAAC_PATH) and its Python package is <repo>/source/python_packages."""
    install = install or isaacsim_install()
    if not install["found"]:
        return []
    if install["layout"] == "pip":
        return [os.path.join(install["package"], "kit", "EULA_ACCEPTED")]
    repositories = []
    source = os.path.dirname(os.path.dirname(install["package"]))                   # <repo>/source
    if os.path.basename(source) == "source":
        repositories.append(os.path.dirname(source))
    for build in (os.environ.get("ISAAC_PATH", ""), os.path.dirname(os.path.dirname(install["found_at"]))):
        build = os.path.abspath(build) if build else ""                             # <repo>/_build/<platform>/<config>
        if build and os.path.basename(os.path.dirname(os.path.dirname(build))) == "_build":
            repositories.append(os.path.dirname(os.path.dirname(os.path.dirname(build))))
    markers = []
    for repository in repositories:
        marker = os.path.join(repository, ".eula_accepted")
        if marker not in markers:
            markers.append(marker)
    return markers


def licence_status() -> tuple:
    """(accepted, how).  The NVIDIA Omniverse licence counts as accepted when the variable
    OMNI_KIT_ACCEPT_EULA says so, or when Isaac Sim's own marker file exists (licence_markers:
    written by its interactive prompt for a pip install, by build.sh for a source build).
    This module never sets the variable and never writes a marker."""
    if os.environ.get(EULA_VARIABLE, "N").lower() in ("y", "yes", "1"):
        return True, f"{EULA_VARIABLE} is set"
    for marker in licence_markers():
        try:
            with open(marker) as f:
                first = f.readline().strip().lower()
        except OSError:
            continue
        if first in ("y", "yes", "1") or os.path.basename(marker) == ".eula_accepted":
            return True, f"marker file {marker}"
    return False, f"{EULA_VARIABLE} is not set"


# ---- the cheap graphics probe -----------------------------------------------------------------
@dataclass
class GraphicsProbe:
    """Result of probe_graphics(): can this machine create a hardware graphics context?"""

    hardware: bool                    # an NVIDIA EGL device was initialised
    reason: str                       # one sentence, for people
    devices: int = 0                  # EGL devices enumerated
    nvidia_devices: int = 0           # of these, devices of the NVIDIA driver
    initialised: int = 0              # NVIDIA devices on which eglInitialize succeeded
    errors: tuple = ()                # EGL error names seen, e.g. ("EGL_BAD_ACCESS",)
    seconds: float = 0.0
    details: list = field(default_factory=list)       # one dict per device

    def __str__(self) -> str:
        return self.reason


_EGL_PROBE = r'''
import ctypes, ctypes.util, json, sys, time
try:                                        # a graphics driver that crashes must not leave a core file
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except Exception:
    pass
out = {"library": "", "devices": [], "error": "", "client_extensions": ""}
def finish():
    sys.stdout.write("NCNTM_EGL_PROBE " + json.dumps(out) + "\n"); sys.stdout.flush(); sys.exit(0)
ERR = {0x3000: "EGL_SUCCESS", 0x3001: "EGL_NOT_INITIALIZED", 0x3002: "EGL_BAD_ACCESS", 0x3003: "EGL_BAD_ALLOC",
       0x3004: "EGL_BAD_ATTRIBUTE", 0x3008: "EGL_BAD_DISPLAY", 0x300A: "EGL_BAD_MATCH", 0x300C: "EGL_BAD_PARAMETER"}
egl = None
for name in ("libEGL.so.1", ctypes.util.find_library("EGL")):
    if not name:
        continue
    try:
        egl = ctypes.CDLL(name); out["library"] = name; break
    except OSError as e:
        out["error"] = str(e)
if egl is None:
    out["error"] = "libEGL could not be loaded (" + (out["error"] or "not found") + ")"; finish()
egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]; egl.eglGetProcAddress.restype = ctypes.c_void_p
egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]; egl.eglQueryString.restype = ctypes.c_char_p
egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
egl.eglInitialize.restype = ctypes.c_uint
egl.eglTerminate.argtypes = [ctypes.c_void_p]; egl.eglTerminate.restype = ctypes.c_uint
egl.eglGetError.restype = ctypes.c_int
EGL_EXTENSIONS, EGL_VENDOR, EGL_PLATFORM_DEVICE_EXT = 0x3055, 0x3053, 0x313F
client = (egl.eglQueryString(None, EGL_EXTENSIONS) or b"").decode()
out["client_extensions"] = client
names = client.split()
if not (("EGL_EXT_device_enumeration" in names or "EGL_EXT_device_base" in names) and "EGL_EXT_platform_device" in names):
    out["error"] = "this libEGL cannot enumerate devices (no EGL_EXT_device_enumeration / EGL_EXT_platform_device)"; finish()
def fn(name, restype, *argtypes):
    p = egl.eglGetProcAddress(name)
    return ctypes.CFUNCTYPE(restype, *argtypes)(p) if p else None
query = fn(b"eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))
qstr = fn(b"eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
display = fn(b"eglGetPlatformDisplayEXT", ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
if not (query and display):
    out["error"] = "eglQueryDevicesEXT / eglGetPlatformDisplayEXT are not exported by this libEGL"; finish()
n = ctypes.c_int(0); devs = (ctypes.c_void_p * 64)()
if not query(64, devs, ctypes.byref(n)):
    out["error"] = "eglQueryDevicesEXT failed: " + ERR.get(egl.eglGetError(), "unknown error"); finish()
done = False
for i in range(n.value):
    ext = (qstr(devs[i], EGL_EXTENSIONS) or b"").decode() if qstr else ""
    software = "EGL_MESA_device_software" in ext.split()
    d = {"index": i, "extensions": ext, "nvidia": "EGL_NV_" in ext, "software": software,
         "tried": False, "initialised": False, "error": "", "vendor": "", "version": "", "seconds": 0.0}
    if d["nvidia"] and not done:                    # only a device of the NVIDIA driver can run RTX
        t0 = time.perf_counter(); d["tried"] = True
        dpy = display(EGL_PLATFORM_DEVICE_EXT, devs[i], None)
        major, minor = ctypes.c_int(0), ctypes.c_int(0)
        ok = egl.eglInitialize(dpy, ctypes.byref(major), ctypes.byref(minor)) if dpy else 0
        if ok:
            d["initialised"] = True; d["version"] = "%d.%d" % (major.value, minor.value)
            d["vendor"] = (egl.eglQueryString(dpy, EGL_VENDOR) or b"").decode()
            egl.eglTerminate(dpy); done = True      # one working device is the answer
        else:
            code = egl.eglGetError(); d["error"] = ERR.get(code, hex(code))
        d["seconds"] = round(time.perf_counter() - t0, 3)
    out["devices"].append(d)
finish()
'''


def probe_graphics(timeout: float = 10.0, refresh: bool = False) -> GraphicsProbe:
    """Can a hardware graphics context be created here?  A cheap stand-in for "would RTX start".

    Asks libEGL (through ctypes) for its devices and calls eglInitialize on the ones that
    belong to the NVIDIA driver (device extension EGL_NV_*), stopping at the first success.
    It runs in a short-lived child process, so that a driver that crashes or hangs cannot
    take the kernel with it and no graphics library is loaded into the process Kit will
    start in.  Missing libEGL, a libEGL without device enumeration, a time-out and a crash
    all give ``hardware = False`` with the reason.

    If __EGL_VENDOR_LIBRARY_DIRS is set (a conda environment points it at its own folder) and
    EGL lists no NVIDIA device at all, the question is asked a second time without the
    variable: it hides the system's drivers from EGL in this process, not from Vulkan, which
    is what RTX uses.  The reason says when that second answer is the one given.

    On the development host: 9 devices, 8 of them NVIDIA, eglInitialize fails on every one
    with EGL_BAD_ACCESS -> False.  A True result means "worth trying tier full"; whether RTX
    really renders is what ``session.renderer_available`` measures."""
    state = _state()
    if state["probe"] is not None and not refresh:
        return state["probe"]
    t0 = time.perf_counter()
    vendor_dirs = os.environ.get(EGL_VENDOR_VARIABLE)
    data, problem = _run_egl_probe(timeout, dict(os.environ))
    hint = ""
    seen = data is not None and not data["error"] and any(d["nvidia"] for d in data["devices"])
    if vendor_dirs and data is not None and not seen:
        # The variable is set and EGL shows no NVIDIA device.  A conda environment sets it to its own folder, which
        # hides the NVIDIA driver from EGL (not from Vulkan, which RTX uses): ask again the way the machine answers
        # without it.  (If NVIDIA devices WERE listed, the variable hides nothing and there is nothing to ask again.)
        hint = f" (note: {EGL_VENDOR_VARIABLE}={vendor_dirs} limits which EGL drivers this process finds)"
        plain = {k: v for k, v in os.environ.items() if k != EGL_VENDOR_VARIABLE}
        again, _ = _run_egl_probe(timeout, plain)
        if again is not None and not again["error"] and any(d["nvidia"] for d in again["devices"]):
            data = again
            hint = (f" (seen with {EGL_VENDOR_VARIABLE}={vendor_dirs} ignored: that value hides the system's "
                    "EGL drivers from this process)")
    seconds = round(time.perf_counter() - t0, 3)
    if data is None:
        probe = GraphicsProbe(False, problem + hint, seconds=seconds)
    elif data["error"]:
        probe = GraphicsProbe(False, data["error"] + hint, seconds=seconds)
    else:
        devices = data["devices"]
        nvidia = [d for d in devices if d["nvidia"]]
        good = [d for d in nvidia if d["initialised"]]
        errors = tuple(sorted({d["error"] for d in nvidia if d["error"]}))
        other = [d for d in devices if not d["nvidia"]]
        if good:
            reason = (f"an NVIDIA EGL device initialised (EGL {good[0]['version']}, {len(nvidia)} NVIDIA "
                      f"device{'s' if len(nvidia) != 1 else ''} found){hint}")
        elif nvidia:
            reason = (f"{len(nvidia)} NVIDIA EGL device{'s' if len(nvidia) != 1 else ''} found, but eglInitialize "
                      f"failed on {'all of them' if len(nvidia) > 1 else 'it'} ({', '.join(errors) or 'no error code'}): "
                      f"the driver cannot create a graphics context on this machine{hint}")
        elif other:
            kinds = "software (Mesa)" if all(d["software"] for d in other) else "not NVIDIA"
            reason = f"no NVIDIA EGL device: the {len(other)} device{'s' if len(other) != 1 else ''} found {'is' if len(other) == 1 else 'are'} {kinds}{hint}"
        else:
            reason = "libEGL lists no devices" + hint
        probe = GraphicsProbe(bool(good), reason, devices=len(devices), nvidia_devices=len(nvidia),
                              initialised=len(good), errors=errors, seconds=seconds, details=devices)
    state["probe"] = probe
    return probe


def _run_egl_probe(timeout: float, env: dict) -> tuple:
    """Run _EGL_PROBE in a child process.  Returns (data, problem): the dictionary it printed,
    or None and a sentence saying why there is none."""
    try:
        done = subprocess.run([sys.executable, "-c", _EGL_PROBE], capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, env=env)
        for line in done.stdout.splitlines():
            if line.startswith("NCNTM_EGL_PROBE "):
                return json.loads(line.split(" ", 1)[1]), ""
        return None, f"the EGL probe ended with exit code {done.returncode} and no result"
    except subprocess.TimeoutExpired:
        return None, f"the EGL probe did not finish within {timeout:g} s"
    except (OSError, ValueError) as e:
        return None, f"the EGL probe could not run ({e})"


# ---- which tier ------------------------------------------------------------------------------
@dataclass
class TierChoice:
    """Which tier, and why (so that a notebook can say it)."""

    tier: str                         # "off" | "usd" | "core" | "full"
    source: str                       # "argument" | "NCNTM_KIT" | "probe" | "no Isaac Sim"
    reason: str
    probe: GraphicsProbe = None       # set when the graphics probe was consulted

    def __str__(self) -> str:
        return f"tier '{self.tier}' ({self.source}): {self.reason}"


def choose_tier(tier: str = "auto") -> TierChoice:
    """Decide the tier without starting anything.

    Order: an explicit ``tier`` argument, then the environment variable NCNTM_KIT, then the
    probe: "full" if probe_graphics() finds a usable NVIDIA graphics device, else "usd".
    Without an importable ``isaacsim`` the automatic answer is "off"."""
    tier = str(tier).lower()
    if tier not in TIERS + ("auto",):
        raise ValueError(f"tier must be 'auto' or one of {TIERS}, got {tier!r}")
    if tier != "auto":
        return TierChoice(tier, "argument", f"start_kit(tier={tier!r}) was asked for")
    env = os.environ.get(ENV_TIER, "").strip().lower()
    if env and env != "auto":
        if env not in TIERS:
            raise ValueError(f"the environment variable {ENV_TIER} must be one of {TIERS} (or auto), got {env!r}")
        return TierChoice(env, ENV_TIER, f"the environment variable {ENV_TIER}={env}")
    if not isaacsim_install()["found"]:
        return TierChoice("off", "no Isaac Sim", f"the package 'isaacsim' cannot be imported by {sys.executable}")
    probe = probe_graphics()
    if probe.hardware:
        return TierChoice("full", "probe", probe.reason, probe)
    return TierChoice("usd", "probe", probe.reason, probe)


# ---- pieces of the start ------------------------------------------------------------------------
def default_root() -> str:
    """Folder for Kit's logs, cache, data and temporary files: NCNTM_KIT_ROOT, else
    ~/.cache/ncntm/kit.  Never inside the Isaac Sim installation."""
    return os.path.abspath(os.path.expanduser(os.environ.get(ENV_ROOT) or os.path.join("~", ".cache", "ncntm", "kit")))


def isaac_extension_folders() -> list:
    """Extra ``--ext-folder`` arguments for the slim experiences (apps/*.kit carry no path).

    SimulationApp itself passes $ISAAC_PATH/exts and $ISAAC_PATH/apps.  The Kit SDK
    extensions that the slim experiences need (omni.usd, omni.kit.usd.layers, the
    ParticleField schema, ...) are in $ISAAC_PATH/extscache.

    THE ONE LAYOUT ASSUMPTION OF THIS MODULE.  Verified by running: the pip install
    (ISAAC_PATH = <site-packages>/isaacsim).  Read, not run: binary and source builds, where
    python.sh sets ISAAC_PATH to the folder that holds python.sh (for a source build
    <repo>/_build/<platform>/release) and repo.toml puts the extension cache into
    <that folder>/extscache -- the same folder the stock experience lists as
    "${app}/../extscache".  If a build keeps its extensions elsewhere, this is the function
    to change; call it after ``import isaacsim``."""
    isaac_path = os.environ.get("ISAAC_PATH", "")
    if not isaac_path:
        raise RuntimeError("ISAAC_PATH is not set: 'import isaacsim' sets it for a pip install, python.sh for a build")
    folder = os.path.join(os.path.abspath(isaac_path), "extscache")
    if not os.path.isdir(folder):
        raise RuntimeError(
            f"Isaac Sim's extension cache was expected in {folder} but is not there. Tier 'full' does not need it "
            "(start_kit(tier='full')); for the slim tiers adapt ncntm.viz.kit.isaac_extension_folders() to this layout.")
    return [folder]


def experience_file(tier: str, root: str) -> str:
    """The .kit file Kit is started with in ``tier`` ("" for tier "full": the stock application).

    apps/<name>.kit of this package when it is there.  When it is not (the repository ignores
    *.kit, so a fresh clone has no such file), the same text, kept in EXPERIENCE_TEXT, is
    written to <root>/apps/<name>.kit and that file is used."""
    name = EXPERIENCES[tier]
    if not name:
        return ""
    packaged = os.path.join(APPS_DIR, name)
    if os.path.isfile(packaged):
        return packaged
    folder = os.path.join(root, "apps")
    path = os.path.join(folder, name)
    text = EXPERIENCE_TEXT[tier]
    try:
        with open(path) as f:
            if f.read() == text:
                return path
    except OSError:
        pass
    os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


def _cuda_user() -> str:
    """Name of what has already initialised CUDA in this process, or ''."""
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_initialized():
                return "torch"
        except Exception:
            pass
    for name in ("warp._src.context", "warp.context"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "runtime", None) is not None:
            return "warp"
    try:
        with open("/proc/self/maps") as f:
            if any("libcuda.so" in line for line in f):
                return "a CUDA library (libcuda.so is loaded)"
    except OSError:
        pass
    return ""


def gpu_count() -> int:
    """Number of GPUs the NVIDIA driver knows (its /proc folder), without touching CUDA; 0 if unknown."""
    try:
        return len(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        return 0


def _select_gpu(gpu) -> tuple:
    """Restrict CUDA to one GPU.  Returns (gpu, cuda_index, note): the GPU asked for, the
    index that GPU has inside this process and a sentence for describe().

    On a machine with ONE GPU (a DGX Spark) ``gpu=0`` has nothing to restrict and the
    environment is left alone: RTX logs "Setting CUDA_VISIBLE_DEVICES can lead to undesired
    behavior or crashes" when the variable is set."""
    if gpu is None and os.environ.get(ENV_GPU, "").strip():
        gpu = os.environ[ENV_GPU]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu is None:
        if visible is not None:
            return None, 0, f"CUDA_VISIBLE_DEVICES = {visible} (from the environment)"
        count = gpu_count()
        if count > 1:
            return None, 0, (f"CUDA_VISIBLE_DEVICES not set: all {count} GPUs are visible to Kit and CUDA "
                             f"(start_kit(gpu=N) or {ENV_GPU}=N restricts this process to one)")
        return None, 0, "CUDA_VISIBLE_DEVICES not set" + (": the one GPU of this machine is cuda:0" if count == 1 else "")
    gpu = int(gpu)
    if gpu < 0:
        raise ValueError(f"gpu must be a non-negative index, got {gpu}")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")          # the order nvidia-smi shows
    if visible == str(gpu):
        return gpu, 0, f"CUDA_VISIBLE_DEVICES = {gpu} (already set): GPU {gpu} is cuda:0 in this process"
    if visible is None and gpu == 0 and gpu_count() == 1:
        return gpu, 0, "this machine has one GPU: it is cuda:0 and CUDA_VISIBLE_DEVICES is left unset"
    user = _cuda_user()
    if user:
        warnings.warn(
            f"CUDA was already initialised by {user} before start_kit(gpu={gpu}); CUDA_VISIBLE_DEVICES cannot restrict "
            f"this process any more (it is {visible!r}). Kit and Warp may see other GPUs. Call start_kit() before "
            "the first CUDA use.", RuntimeWarning, stacklevel=3)
        return gpu, gpu, f"CUDA was initialised earlier by {user}: NOT restricted, GPU {gpu} is cuda:{gpu} if all GPUs are visible"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    was = f" (the environment had {visible!r})" if visible is not None else ""
    return gpu, 0, f"CUDA_VISIBLE_DEVICES = {gpu}{was}: GPU {gpu} is cuda:0 in this process"


def _no_core_files(soft_only: bool = False) -> None:
    """RLIMIT_CORE to 0, so that a crash leaves no core file.  ``soft_only`` keeps the hard
    limit, so that the notebook can raise the limit again if it wants core files."""
    try:
        import resource
        hard = resource.getrlimit(resource.RLIMIT_CORE)[1] if soft_only else 0
        resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
    except (ImportError, ValueError, OSError):
        pass


def _flush_c_stdio() -> None:
    try:
        import ctypes
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


def flush_open_files() -> int:
    """Flush every open writable Python file, close every open h5py file and shut logging
    down.  Returns how many objects were handled.  This is what the exit hook does before
    it leaves; it is public so that a notebook can call it before a risky step."""
    import gc
    import logging
    handled = 0
    h5py = sys.modules.get("h5py")
    for obj in gc.get_objects():
        try:
            if h5py is not None and isinstance(obj, h5py.File):
                if obj.id.valid:
                    obj.close()
                    handled += 1
            elif isinstance(obj, io.IOBase) and not obj.closed and obj.writable():
                obj.flush()
                handled += 1
        except Exception:
            pass
    try:
        logging.shutdown()
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    _flush_c_stdio()
    return handled


def _open_closed_folders(root: str) -> int:
    """Give the owner access to folders of the Kit folder that have none; returns how many.

    Measured: the stock application creates <root>/data/documents/Kit/shared/screenshots with
    mode 000.  Not even its owner can list it, so ``du``, ``rm -r`` and shutil.rmtree stop
    there and the Kit folder cannot be deleted."""
    opened = 0
    for folder, names, _ in os.walk(os.path.join(root, "data", "documents"), onerror=lambda error: None):
        for name in names:
            path = os.path.join(folder, name)
            try:
                if not os.path.islink(path) and not os.access(path, os.R_OK | os.W_OK | os.X_OK):
                    os.chmod(path, 0o700)
                    opened += 1
            except OSError:
                pass
    return opened


def _remove_temporary_directories() -> int:
    """Remove the folders that ``tempfile.TemporaryDirectory`` objects still hold; returns how
    many.  An ordinary interpreter exit does this through their finalizers, os._exit does not.
    Measured: Warp keeps the precompiled header of its kernel compiler in one (wp_pch_*,
    70 MB), and every kernel that had compiled a Warp kernel left that folder in /tmp.
    Only for the moment the process ends: the folders are in use until then."""
    import gc
    removed = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, tempfile.TemporaryDirectory):
                obj.cleanup()
                removed += 1
        except Exception:
            pass
    return removed


def _exit_hook() -> None:
    """atexit: save what Python holds in buffers, then leave WITHOUT Kit's teardown.

    Kit's own shutdown at interpreter exit crashes (SIGSEGV) or takes longer than the five
    seconds Jupyter waits before it kills the kernel.  A bare os._exit loses unflushed files
    (measured: an open text file was 0 bytes, an open HDF5 file unreadable), hence the flush,
    and leaves temporary folders behind, hence their removal.
    Exit status: 0 in a kernel.  In a plain script: the code given to sys.exit(), 1 after an
    uncaught exception, else 0 (``raise SystemExit(n)`` cannot be seen from here)."""
    state = _state()
    code = 0
    if not state.get("kernel"):                     # decided when the hook was installed, not while the process ends
        recorded = state.get("exit_code")
        if isinstance(recorded, int):
            code = recorded
        elif recorded is not None:
            code = 1
        elif getattr(sys, "last_exc", None) is not None or getattr(sys, "last_value", None) is not None:
            code = 1
    flush_open_files()
    _remove_temporary_directories()
    session = state.get("session")
    if session is not None and getattr(session, "root", ""):
        _open_closed_folders(session.root)
    os._exit(code)


def _install_exit_hook(app=None) -> None:
    """Make the process leave through _exit_hook.  ``app`` is the SimulationApp, or None after
    a start that failed half-way: Carbonite is loaded then, and its teardown at interpreter
    exit crashes just the same (measured: the kernel ended with SIGSEGV)."""
    state = _state()
    if state["hook"]:
        return
    if app is not None:
        try:
            atexit.unregister(app._atexit_close)    # SimulationApp's own hook would start the teardown
        except Exception:
            pass
    state["kernel"] = in_jupyter()
    if not state["kernel"]:                         # a script's exit status must survive os._exit
        original_exit = sys.exit

        def exit_and_remember(code=0):
            _state()["exit_code"] = code
            original_exit(code)

        sys.exit = exit_and_remember
    atexit.register(_exit_hook)                     # registered last -> runs first
    state["hook"] = True


def protect_queued_cells() -> bool:
    """Keep cells that are queued behind a running cell ("Run All") from being lost.

    Measured with ipykernel 7.4.0 and a live Kit: every ``app.update()`` runs the kernel's own
    asyncio loop from inside the running cell (that is what nest_asyncio is for).  ipykernel 7
    runs all cells in ONE shared ``contextvars.Context``; when the loop, re-entered by Kit,
    picks up the request for the next cell, the first step of its task cannot enter that
    context ("cannot enter context: ... is already entered"), asyncio logs the error and the
    request is gone: the cell never runs and never answers.

    The cure: an event-loop callback whose context is in use by an outer frame is not run and
    not dropped but kept, in order, and run as soon as the context is free (10 ms polling).
    Every other callback runs exactly as before.  This replaces ``asyncio.events.Handle._run``
    for the process; it is applied once, by start_kit, inside a kernel only.  Returns True if
    the protection is (already) in place.

    Measured with ipykernel 7.4.0 only.  If another ipykernel misbehaves with it, set the
    environment variable NCNTM_KIT_PROTECT_QUEUE=0 in the kernel: nothing is replaced then,
    and cells must be run one at a time (nbconvert and "Run Cell" do that; "Run All" may
    lose cells)."""
    from asyncio import events
    if getattr(events.Handle._run, "_ncntm_protected", False):
        return True
    if os.environ.get(ENV_PROTECT, "1").strip().lower() in ("0", "no", "false", "off"):
        return False
    original = events.Handle._run
    waiting = weakref.WeakKeyDictionary()           # loop -> deque of postponed handles

    def nothing():
        pass

    def busy(context) -> bool:
        try:
            context.run(nothing)
        except RuntimeError:                        # only raised when the context is already entered
            return True
        return False

    def drain(loop):
        queue = waiting.get(loop)
        while queue:
            handle = queue[0]
            if not handle._cancelled:
                if busy(handle._context):
                    loop.call_later(0.01, drain, loop)
                    return
                original(handle)
            queue.popleft()

    def run(self):
        loop = self._loop
        queue = waiting.get(loop)
        context = self._context
        if busy(context) or (queue and any(h._context is context for h in queue)):      # the second keeps the order
            if queue is None:
                queue = waiting[loop] = collections.deque()
            if not queue:
                loop.call_later(0.01, drain, loop)
            queue.append(self)
            return
        original(self)

    run._ncntm_protected = True
    try:
        events.Handle._run = run
    except (AttributeError, TypeError):
        return False
    return True


class _HoldInterrupt:
    """Hold SIGINT back while Kit starts.

    An interrupt in the middle of SimulationApp() surfaces inside some extension's start-up,
    is swallowed there and leaves a half-initialised application that still reports
    "started".  SimulationApp also installs its own SIGINT handler (shut Kit down, exit the
    process), which must never be active in a kernel.  Inside this block SIGINT only sets a
    flag and SimulationApp's handler is not installed; the caller raises KeyboardInterrupt
    once the start is complete."""

    def __init__(self):
        self.pending = False
        self.active = False

    def _note(self, signum, frame):
        self.pending = True

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            self._signal = signal.signal
            self._previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._note)

            def guarded(signum, handler):
                if signum == signal.SIGINT:
                    return self._note
                return self._signal(signum, handler)

            signal.signal = guarded
            self.active = True
        return self

    def __exit__(self, *exc):
        if self.active:
            signal.signal = self._signal
            signal.signal(signal.SIGINT, self._previous if self._previous is not None else signal.default_int_handler)
            self.active = False
        return False


@contextlib.contextmanager
def _captured_output(path, quiet: bool):
    """While Kit starts, what Python prints to sys.stdout goes into a buffer, which is yielded:
    SimulationApp prints there the argument list it hands to Kit, and _launch reads it.
    ``quiet`` True: what goes to file descriptors 1 and 2 (Kit's native code) is sent into the
    file ``path`` as well, so the cell shows nothing.  ``quiet`` False: the native output is
    left alone and the buffered prints are shown as soon as the start is over."""
    buffer = io.StringIO()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    saved = None
    if quiet:
        try:
            target = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            saved = (os.dup(1), os.dup(2))
            os.dup2(target, 1)
            os.dup2(target, 2)
            os.close(target)
        except OSError:
            saved = None                            # no usable descriptors: capture Python's prints only
    try:
        with contextlib.redirect_stdout(buffer):
            yield buffer
    finally:
        if saved is not None:
            _flush_c_stdio()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
        if not quiet:
            try:
                sys.stdout.write(buffer.getvalue())
                sys.stdout.flush()
            except Exception:
                pass


def _newest_kit_log(root: str) -> str:
    newest, when = "", 0.0
    for folder, _, files in os.walk(os.path.join(root, "logs"), onerror=lambda e: None):
        for name in files:
            if name.startswith("kit_") and name.endswith(".log"):
                path = os.path.join(folder, name)
                try:
                    modified = os.path.getmtime(path)
                except OSError:
                    continue
                if modified > when:
                    newest, when = path, modified
    return newest


def _start_errors(capture_file: str, log: str, limit: int = 12) -> list:
    """What Kit said went wrong while it started: the [Error] lines, with the lines that
    continue them, of its console output during the start (quiet=True keeps that in a file),
    else of its log file."""
    for path in (capture_file, log):
        if not path:
            continue
        try:
            with open(path, errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        said, taking = [], False
        for line in lines:
            stamped = line[:4].isdigit() and line[4:5] == "-"           # "2026-11-25T02:48:30Z [23ms] [Error] ..."
            if "[Error]" in line:
                taking = True
                said.append(line.split("[Error]", 1)[1].strip()[:240])
            elif stamped or not line.strip():
                taking = False
            elif taking:
                said.append("  " + line.strip()[:240])
        if said:
            return said[:limit]
    return []


def _launch(tier: str, config: dict, root: str, quiet: bool):
    """Construct SimulationApp.  The only place that does.  Returns (app, info)."""
    capture_file = os.path.join(root, "logs", f"ncntm_start_{os.getpid()}.txt")
    os.makedirs(os.path.dirname(capture_file), exist_ok=True)
    saved_argv, saved_out, saved_err = sys.argv, sys.stdout, sys.stderr
    sys.argv = [(sys.argv[0] if sys.argv else "") or "ncntm", "--portable-root", root]
    hold = _HoldInterrupt()
    printed = ""
    t0 = time.perf_counter()
    try:
        with hold, _captured_output(capture_file, quiet) as buffer:
            try:
                try:
                    from isaacsim import SimulationApp      # applies nest_asyncio when it sees Jupyter
                except SystemExit as e:                     # isaacsim leaves with sys.exit(message) (licence, LD_PRELOAD)
                    raise RuntimeError(
                        f"'import isaacsim' ended the start: {e.code}. Nothing was started; fix the environment "
                        "of the kernel (scripts/install_kernel.py writes it into the kernel spec) and run the cell again."
                    ) from None
                if in_jupyter() and not getattr(builtins, "ISAAC_LAUNCHED_FROM_JUPYTER", False):
                    raise RuntimeError("isaacsim did not notice that it runs under Jupyter, so nest_asyncio was not "
                                       "applied; Kit would wreck the kernel's event loop. Kit was not started.")
                folders = [] if tier == "full" else isaac_extension_folders()
                for folder in reversed(folders):
                    config["extra_args"][:0] = ["--ext-folder", folder]
                experience = experience_file(tier, root)
                import_seconds = time.perf_counter() - t0
                _state()["tainted"] = f"the start of tier '{tier}' did not finish"       # until proven otherwise
                app = SimulationApp(config, experience=experience)
            finally:
                printed = buffer.getvalue() if buffer is not None else ""
    except RuntimeError:
        raise
    except BaseException as e:
        state = _state()
        if state["tainted"]:                                # SimulationApp() was entered and did not come back
            log = _newest_kit_log(root)
            state["tainted"] = (f"the start of tier '{tier}' was cut short by {type(e).__name__}"
                                + (f" (Kit log: {log})" if log else ""))
            _install_exit_hook()                            # Carbonite is loaded: its teardown at exit would crash
            if isinstance(e, SystemExit):                   # SimulationApp gave up (sys.exit): say what Kit said
                said = _start_errors(capture_file if quiet else "", log)
                told = "\nKit said:\n    " + "\n    ".join(said) if said else ""
                if tier == "full":
                    advice = "start_kit(tier='usd') starts Kit without the renderer, start_kit(tier='off') works without Kit."
                else:
                    advice = (f"If an extension could not be resolved, this Isaac Sim does not provide what the experience "
                              f"{EXPERIENCES[tier]} asks for: start_kit(tier='full') starts the stock application instead, "
                              "start_kit(tier='off') works without Kit.")
                raise RuntimeError(
                    f"Kit did not start (tier '{tier}').{told}\n" + (f"Full log: {log}\n" if log else "")
                    + "What to do: restart the kernel first (a half-started Kit cannot be started again in this "
                    + "process). " + advice) from None
        raise
    finally:
        sys.argv = saved_argv
        sys.stdout, sys.stderr = saved_out, saved_err       # a no-op unless Kit replaced them after all
    _state()["tainted"] = ""
    argv = []                                               # what SimulationApp really handed to Kit:
    for marker in ("Starting kit application with the following args: ",          # its own list ...
                   "Passing the following args to the base kit application: "):  # ... extended by sys.argv
        for line in printed.splitlines():
            if line.startswith(marker):
                try:
                    import ast
                    argv += [str(a) for a in ast.literal_eval(line[len(marker):].strip())]
                except (ValueError, SyntaxError):
                    argv = []
                break
        else:
            argv = []
        if not argv:
            break
    info = {"seconds": round(time.perf_counter() - t0, 2), "import_seconds": round(import_seconds, 3),
            "argv": argv or list(config["extra_args"]), "argv_from_kit": bool(argv), "capture_file": capture_file if quiet else "",
            "interrupted": hold.pending}
    return app, info


# ---- OpenUSD -----------------------------------------------------------------------------------
@dataclass
class PxrOrigin:
    """Which OpenUSD build this process uses."""

    imported: bool                    # False: no pxr module is loaded yet (and none was loaded to find out)
    version: str = ""                 # "25.11"
    path: str = ""                    # folder of the pxr package
    is_kit: bool = False              # True: Kit's own build (extension omni.usd.libs)

    def __str__(self) -> str:
        if not self.imported:
            return "OpenUSD: pxr is not imported yet"
        return f"OpenUSD {self.version} ({'Kit' if self.is_kit else 'interpreter'} build, {self.path})"


def _loaded(package: str) -> bool:
    """True when ``package`` or one of its modules is in sys.modules."""
    return any(name == package or name.startswith(package + ".") for name in list(sys.modules))


def _pxr_loaded() -> bool:
    return _loaded("pxr")


def _importers(package: str, limit: int = 6) -> list:
    """Names of the loaded modules (outside ``package`` itself) that imported it: the ones to
    move behind start_kit().  "__main__" is the notebook or script itself."""
    found = []
    for name, module in list(sys.modules.items()):
        if module is None or name == package or name.startswith(package + ".") or name == __name__:
            continue
        try:
            values = list(vars(module).values())
        except TypeError:
            continue
        for value in values:
            owner = getattr(value, "__name__", "") if isinstance(value, type(sys)) else ""
            if isinstance(owner, str) and (owner == package or owner.startswith(package + ".")):
                found.append(name)
                break
    found.sort(key=lambda name: (not name.startswith("ncntm."), name != "__main__", name))
    return found[:limit]


def _pxr_importers(limit: int = 6) -> list:
    return _importers("pxr", limit)


def _said_by(package: str) -> str:
    """" (by: ncntm.viz.stage, a cell of this notebook)", or "" when nobody can be named."""
    names = ["a cell of this notebook" if name == "__main__" else name for name in _importers(package)]
    return f" (by: {', '.join(names)})" if names else ""


def pxr_origin(load: bool = None) -> PxrOrigin:
    """Version and origin of the OpenUSD in use.

    The same interpreter can hold two builds: Kit's (25.11 in Isaac Sim 6.2, without the
    typed UsdVol.ParticleField classes) and the one of site-packages (26.08).  Which one
    ``import pxr`` gives depends on whether Kit runs.  ``load`` says whether pxr may be
    imported to answer: by default only when that cannot do harm (Kit runs, or a Kit-free
    session exists); before start_kit() the answer is "not imported", because importing pxr
    then would make a later start of Kit impossible."""
    session = current_session()
    if load is None:
        load = session is not None
    if not _pxr_loaded() and not load:
        return PxrOrigin(False)
    try:
        from pxr import Usd
    except ImportError:
        return PxrOrigin(False)
    _, year, month = Usd.GetVersion()                           # (0, 25, 11) is OpenUSD 25.11
    version = f"{year}.{month:02d}"
    path = os.path.dirname(os.path.dirname(os.path.abspath(Usd.__file__)))
    is_kit = any(part.startswith("omni.usd.libs") for part in path.split(os.sep))
    return PxrOrigin(True, version, path, is_kit)


# ---- the session -----------------------------------------------------------------------------------
def renderer_complaint(error_lines) -> str:
    """The most telling renderer error among the [Error] lines of a Kit log: a line that says
    no graphics device could be created, else the first line about RTX, the GPU or Vulkan."""
    about = [line.split("[Error]", 1)[1].strip() for line in error_lines
             if "[Error]" in line and any(word in line.lower() for word in ("rtx", "gpu", "vulkan"))]
    for line in about:
        if "device" in line.lower() and "creat" in line.lower():
            return line[:200]
    return about[0][:200] if about else ""


def rgba_from_annotator(data, width: int, height: int):
    """What Replicator's "rgb" annotator handed back -> an (height, width, 4) uint8 array that
    belongs to the caller, or an array of size 0 when it holds no picture.

    ``annotator.get_data()`` (omni.replicator.core 1.13.38) gives an ndarray (H, W, 4), or a
    dict {"data": ndarray, "info": ...}, or an EMPTY array when nothing was rendered (the
    development host: shape (0,)), or None after it logged an error.  The pixels are copied:
    the annotator's own buffer does not outlive ``detach()``."""
    import numpy as np
    empty = np.zeros((0, 0, 4), dtype=np.uint8)
    if isinstance(data, dict):
        data = data.get("data")
    if data is None:
        return empty
    if not isinstance(data, np.ndarray) and hasattr(data, "numpy"):     # a Warp array (annotator on a CUDA device)
        data = data.numpy()
    try:
        array = np.array(data, dtype=np.uint8)                          # np.array copies
    except (TypeError, ValueError):
        return empty
    if array.size == 0:
        return empty
    width, height = int(width), int(height)
    if array.ndim == 3 and array.shape[2] == 4:
        if array.shape[:2] != (height, width):
            warnings.warn(f"RTX rendered {array.shape[1]} x {array.shape[0]} pixels where {width} x {height} were asked "
                          "for; the picture is returned as rendered.", RuntimeWarning, stacklevel=3)
        return array
    if array.size == height * width * 4:
        return array.reshape(height, width, 4)
    raise RuntimeError(f"the rgb annotator returned an array of shape {array.shape}; expected ({height}, {width}, 4)")


@dataclass
class RendererStatus:
    """Whether RTX can draw, and why (not)."""

    available: bool
    reason: str
    engines: tuple = ()               # Hydra engines attached to the USD context
    seconds: float = 0.0              # time the check took

    def __str__(self) -> str:
        return ("RTX renderer: available (" if self.available else "RTX renderer: not available on this machine because ") \
            + self.reason + (")" if self.available else "")


class Session:
    """What start_kit() returns.  One per process.

    tier                 "off" | "usd" | "core" | "full"
    app                  the SimulationApp, or None for tier "off"
    choice               TierChoice: how the tier was decided
    usd_version          "25.11" inside Kit, the interpreter's OpenUSD otherwise
    warp_version         version ``import warp`` gives in this process ("" if not installed)
    renderer_available   True only when a Hydra engine is attached AND a test frame has pixels
    device               Warp / torch name of the session GPU ("cuda:0" after start_kit(gpu=N))
    describe()           the above as text for the notebook

    stage                the current stage, asked from Kit every time (old handles die)
    new_stage()          an empty Z-up stage, 1 unit = 1 m;  open_stage(path);  save(path)
    update(n)            advance Kit by n frames (nothing advances between calls)
    capture_rtx(...)     RGBA picture from RTX; RuntimeError("no renderer ...") without one
    """

    TEST_FRAME_TRIES = 3              # renderer_status(): test frames tried before the answer is no

    def __init__(self, tier: str, choice: TierChoice, app=None, root: str = "", gpu=None, cuda_index: int = 0,
                 gpu_note: str = "", info: dict = None, livestream: bool = False):
        info = info or {}
        self.tier, self.choice, self.app, self.root = tier, choice, app, root
        self.gpu, self.gpu_note = gpu, gpu_note
        self.device = f"cuda:{cuda_index}"
        self.start_seconds = info.get("seconds")
        self.launch_args = list(info.get("argv", []))           # the arguments Kit was started with
        self.launch_args_from_kit = bool(info.get("argv_from_kit"))
        self.start_output_file = info.get("capture_file", "")   # Kit's console output during the start (quiet=True)
        self.livestream_prepared = bool(livestream)
        self.log_file, self.kit_version, self.extensions = "", "", 0
        self._stage = None                                      # tier "off" only
        self._renderer = None
        self._livestream = None
        if app is not None:
            import carb.settings
            import omni.kit.app
            kit_app = omni.kit.app.get_app()
            self.log_file = carb.settings.get_settings().get("/log/file") or ""
            self.kit_version = str(kit_app.get_kit_version()).split("+")[0]
            self.extensions = sum(1 for e in kit_app.get_extension_manager().get_extensions() if e["enabled"])
            self._real_close = app.close
            app.close = self.close                              # the real one deadlocks in a kernel

    # ---- what is running ---------------------------------------------------------------------
    @property
    def kit_running(self) -> bool:
        return self.app is not None

    @property
    def usd_version(self) -> str:
        return pxr_origin(load=True).version

    @property
    def warp_version(self) -> str:
        module = sys.modules.get("warp")
        if module is not None:
            return str(getattr(module, "__version__", ""))
        try:
            from importlib import metadata
            return metadata.version("warp-lang")
        except Exception:
            return ""

    def privacy(self) -> dict:
        """What Kit says about sending data out, read back from the running application."""
        if self.app is None:
            return {}
        import carb.settings
        import omni.kit.app
        settings = carb.settings.get_settings()
        manager = omni.kit.app.get_app().get_extension_manager()
        return {
            "crashreporter_enabled": bool(settings.get("/crashreporter/enabled")),
            "crashreporter_plugin_requested": bool(self.app.config.get("enable_crashreporter")),
            "telemetry_anonymous_data": bool(settings.get("/telemetry/enableAnonymousData")),
            "telemetry_anonymous_app_name": bool(settings.get("/telemetry/enableAnonymousAppName")),
            "structured_log": bool(settings.get("/structuredLog/enable")),
            "telemetry_extension_enabled": any(bool(manager.is_extension_enabled(n)) for n in PRIVACY_EXCLUDED_EXTENSIONS),
        }

    def log_lines(self, text: str, limit: int = 50) -> list:
        """The lines of Kit's log file so far that contain ``text`` (at most ``limit``), e.g.
        ``session.log_lines("geometry streaming")`` or ``session.log_lines("[Warning]")``."""
        if not self.log_file:
            return []
        try:
            with open(self.log_file, errors="replace") as f:
                return [line.rstrip() for line in f if text in line][:limit]
        except OSError:
            return []

    def kit_errors(self, limit: int = 50) -> list:
        """The [Error] lines of Kit's log file so far (at most ``limit``)."""
        return self.log_lines("[Error]", limit)

    def renderer_status(self, refresh: bool = False) -> RendererStatus:
        """Can RTX draw?  Tiers without a renderer answer at once.  In tier "full" the answer
        is yes only if a Hydra engine is attached to the USD context and a 64 x 64 test frame
        comes back with pixels (up to three tries, a few Kit frames apart: a renderer that has
        just started may still be loading).

        A yes, and a no for want of an engine, are kept (``refresh=True`` measures again).  A
        no although an engine is attached is NOT kept: the next call tries again.

        On the development host no engine is attached and the answer is no within
        milliseconds.  The positive path is UNTESTED on the development host; expect the very
        first test frame on a machine with RTX to take a while (shaders are compiled into
        ``<root>/cache`` on first use)."""
        if self._renderer is not None and not refresh:
            return self._renderer
        t0 = time.perf_counter()
        keep = True
        if self.tier != "full":
            probe = self.choice.probe or probe_graphics()
            if self.tier == "off":
                why = "Kit is not running (tier 'off')"
            else:
                why = f"tier '{self.tier}' loads no renderer"
            if probe.hardware:
                why += "; a graphics device was found, so start_kit(tier='full') in a new kernel should give RTX"
            else:
                why += (f"; the graphics probe says: {probe.reason}. To try RTX all the same: "
                        f"start_kit(tier='full') in a new kernel, or {ENV_TIER}=full")
            status = RendererStatus(False, why)
        else:
            import omni.usd
            context = omni.usd.get_context()
            try:
                engines = tuple(context.get_attached_hydra_engine_names())
            except Exception as e:
                engines = ()
                note = f" ({type(e).__name__}: {e})"
            else:
                note = ""
            if not engines:
                why = "no Hydra render engine is attached to the USD context" + note
                said = renderer_complaint(self.kit_errors())
                if said:
                    why += "; Kit's log says: " + said
                status = RendererStatus(False, why)
            elif self.stage is None:
                keep = False
                status = RendererStatus(False, f"engine {', '.join(engines)} is attached, but no stage is open to draw a "
                                               "test frame of: call new_stage() or open_stage(path) first", engines)
            else:
                keep = False
                status = None
                for attempt in range(self.TEST_FRAME_TRIES):
                    try:
                        frame = self._render(None, 64, 64, 2, None, None)
                    except Exception as e:
                        status = RendererStatus(False, f"engine {', '.join(engines)} is attached but a test frame failed "
                                                       f"({type(e).__name__}: {e}); see {self.log_file}", engines)
                        break
                    if frame.size:
                        keep = True
                        status = RendererStatus(True, f"engine {', '.join(engines)}, test frame "
                                                      f"{frame.shape[1]} x {frame.shape[0]}", engines)
                        break
                    self.update(10)                             # give a renderer that is still loading some frames
                if status is None:
                    status = RendererStatus(False, f"engine {', '.join(engines)} is attached but {self.TEST_FRAME_TRIES} "
                                                   f"test frames came back empty; see {self.log_file}", engines)
        status.seconds = round(time.perf_counter() - t0, 3)
        self._renderer = status if keep else None
        return status

    @property
    def renderer_available(self) -> bool:
        return self.renderer_status().available

    def describe(self) -> str:
        """Several lines for the notebook: what runs, why, with which OpenUSD, Warp and GPU,
        and whether RTX can draw (with the reason if not)."""
        lines = []
        if self.app is None:
            lines.append("Isaac Sim: not running (tier 'off'); stages are plain OpenUSD stages")
        else:
            lines.append(f"Isaac Sim: Kit {self.kit_version}, tier '{self.tier}', {self.extensions} extensions, "
                         f"started in {self.start_seconds} s")
        if self.choice.source == "probe":                       # the reason is the graphics probe's: it is in the RTX line
            found = self.choice.probe is not None and self.choice.probe.hardware
            lines.append("  tier chosen automatically: "
                         + (self.choice.reason if found else "no usable graphics device was found (see the RTX line)")
                         + f"; {ENV_TIER} or start_kit(tier=...) overrides the choice")
        elif self.choice.source == "no Isaac Sim":
            lines.append(f"  tier 'off' because {self.choice.reason}")
        else:
            lines.append(f"  tier chosen by {self.choice.source}: {self.choice.reason}")
        origin = pxr_origin(load=True)
        usd = f"OpenUSD {origin.version} ({'Kit build' if origin.is_kit else 'interpreter build'})" if origin.imported \
            else "OpenUSD: pxr is not installed"
        lines.append(f"  {usd}   Warp {self.warp_version or 'not installed'}")
        lines.append(f"  GPU: {self.gpu_note}")
        lines.append("  " + str(self.renderer_status()))
        if self.app is not None:
            privacy = self.privacy()
            off = not any(privacy.values())
            lines.append("  crash reporter and telemetry: " + ("off" if off else f"NOT all off: {privacy}"))
            errors = self.kit_errors()
            lines.append(f"  Kit files: {self.root}   log: {self.log_file}   errors in the log: {len(errors)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<ncntm.viz.kit.Session tier={self.tier!r} kit={'running' if self.app is not None else 'off'}>"

    # ---- stages ----------------------------------------------------------------------------------
    def _context(self):
        import omni.usd
        return omni.usd.get_context()

    @property
    def stage(self):
        """The current stage (Usd.Stage), or None if there is none.  Fetched from Kit on every
        access: a handle kept across new_stage() / open_stage() is dead, and calling a method
        on it aborts the process."""
        if self.app is None:
            return self._stage
        return self._context().get_stage()

    def update(self, n: int = 1) -> None:
        """Advance Kit by ``n`` frames.  Does nothing in tier "off"."""
        if self.app is not None:
            for _ in range(int(n)):
                self.app.update()

    def new_stage(self):
        """Replace the current stage by an empty one (Z up, metersPerUnit 1.0, the same in every
        tier) and return it.  Kit holds it in memory until save(path)."""
        from pxr import Usd, UsdGeom
        if self.app is None:
            self._stage = Usd.Stage.CreateInMemory()
        else:
            if not self._context().new_stage():
                raise RuntimeError("Kit could not create a new stage")
            self._after_stage_change()
        stage = self.stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        return stage

    def _after_stage_change(self) -> None:
        self.update(2)
        if self.tier == "full":
            self.app.reset_render_settings()            # a new stage brings its own render settings: put the session's back

    def open_stage(self, path):
        """Open a USD file as the current stage and return it.  In a Kit tier the stage belongs
        to Kit's UsdContext (what RTX draws); in tier "off" it is ``Usd.Stage.Open``.  Either
        way it is the composed stage: variant sets, references and relative asset paths work
        as in the file."""
        path = os.path.abspath(os.fspath(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        if self.app is None:
            from pxr import Usd
            self._stage = Usd.Stage.Open(path)
            if not self._stage:
                raise IOError(f"OpenUSD could not open {path}")
            return self._stage
        context = self._context()
        if not context.open_stage(path):
            raise IOError(f"Kit could not open {path}; see {self.log_file}")
        for _ in range(600):                                    # until Kit has loaded every file of the stage
            self.app.update()
            if context.get_stage_loading_status()[2] <= 0:
                break
        self._after_stage_change()
        return self.stage

    def save(self, path=None) -> str:
        """Write the ROOT LAYER of the current stage; returns the file written.

        Without ``path`` the file the stage was opened from is saved in place.  With ``path``
        a copy is written (.usda or .usdc by extension) and the stage stays open on the old
        layer, so handles stay valid.  The layer is exported, never the stage:
        ``Usd.Stage.Export`` would flatten the composition and the variant sets would be gone.
        Relative asset paths are written as they are: a copy in another folder needs the
        assets copied too (``ncntm.viz.stage.save_stage`` does that).

        Tier "full": the viewport of the stock application keeps prims of its own under
        /Render in the root layer of every stage (a RenderProduct, RenderSettings, RenderVars),
        and they are written with it, as in every file saved from an Omniverse application.
        What a notebook authors is under its own root prim; read and compare that."""
        stage = self.stage
        if stage is None:
            raise RuntimeError("there is no stage to save: call new_stage() or open_stage(path) first")
        layer = stage.GetRootLayer()
        if path is None or os.path.abspath(os.fspath(path)) == layer.realPath:
            if not layer.realPath:
                raise ValueError("this stage has no file yet: give save() a path")
            if not layer.Save():
                raise IOError(f"could not write {layer.realPath}")
            return layer.realPath
        path = os.path.abspath(os.fspath(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not layer.Export(path):
            raise IOError(f"could not write {path}")
        return path

    def close(self, *args, **kwargs) -> None:
        """Disabled on purpose.  SimulationApp.close() waits for the kernel's own event loop
        task and never returns inside Jupyter.  Kit ends when the kernel is shut down or
        restarted; that exit is immediate and flushes open files.  Save with save(path)."""
        if self.app is None:
            print("close(): nothing to close, Kit is not running (tier 'off'). Nothing is saved automatically: "
                  "use session.save(path).")
            return
        print("close() is disabled: inside Jupyter SimulationApp.close() deadlocks. Kit ends when the kernel is "
              "shut down or restarted (open files are flushed). Nothing is saved automatically: use session.save(path).")

    # ---- RTX -------------------------------------------------------------------------------------
    def _render(self, camera_path, width, height, subframes, mode, spp):
        """One Replicator frame as an (H, W, 4) uint8 array (size 0 if nothing was rendered).

        The sequence is the one of Isaac Sim's own isaacsim.test.utils.image_capture (6.2.0):
        render product, "rgb" annotator, attach, orchestrator step, get_data, detach, destroy.
        Render settings this call changes are put back before it returns.
        NOT RUN on the development host beyond the import and the camera checks."""
        import carb.settings
        import omni.replicator.core as rep
        from pxr import Usd, UsdGeom
        stage = self.stage
        if stage is None:
            raise RuntimeError("there is no stage to render: call new_stage() or open_stage(path) first")
        temporary = None
        if camera_path is None:                                 # the renderer test: any camera will do
            camera_path = "/OmniverseKit_Persp"                 # Kit's own perspective camera (session layer)
            if not stage.GetPrimAtPath(camera_path):
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    temporary = UsdGeom.Camera.Define(stage, "/ncntm_renderer_test_camera").GetPath()
                camera_path = str(temporary)
        else:
            prim = stage.GetPrimAtPath(str(camera_path))
            if not prim:
                raise ValueError(f"there is no prim {camera_path} on the stage")
            if not prim.IsA(UsdGeom.Camera):
                raise ValueError(f"{camera_path} is a {prim.GetTypeName() or 'typeless'} prim; capture_rtx needs a Camera")
        settings = carb.settings.get_settings()
        before = {}                                             # setting -> its value before this call

        def change(key, value):
            before.setdefault(key, settings.get(key))
            settings.set(key, value)

        rep.orchestrator.set_capture_on_play(False)             # capture only when step() is called
        if mode is not None:
            change("/rtx/rendermode", mode)
            if mode == "PathTracing":
                # Samples per pixel per sub-frame: the caller's, else what the session was started with -- not the
                # setting as it stands, which Replicator rewrites when it stops (it puts back what it found at its
                # first step: _set_capture_settings / _restore_settings in its orchestrator.py).
                configured = getattr(self.app, "config", {}).get("samples_per_pixel_per_frame")
                per_frame = int(spp) if spp is not None else int(configured or 64)
                change("/rtx/pathtracing/spp", per_frame)
                change("/rtx/pathtracing/clampSpp", per_frame)                      # SimulationApp keeps these two equal
                change("/rtx/pathtracing/totalSpp", per_frame * int(subframes))     # accumulate over all subframes
            else:
                change("/rtx/post/dlss/execMode", 2)            # DLSS "Quality": Performance damages edges in small pictures
        product = annotator = None
        try:
            product = rep.create.render_product(str(camera_path), (int(width), int(height)), force_new=True)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")                  # LdrColor: uint8 (H, W, 4) RGBA
            annotator.attach(product)
            rep.orchestrator.step(rt_subframes=int(subframes), delta_time=0.0)      # 0.0: the timeline does not advance
            # copied here, before detach(): the annotator's buffer is gone afterwards
            return rgba_from_annotator(annotator.get_data(do_array_copy=True), width, height)
        finally:
            for release in (getattr(annotator, "detach", None), getattr(product, "destroy", None)):
                try:
                    if release is not None:
                        release()
                except Exception as e:                          # a failed release must not cost the picture
                    warnings.warn(f"could not release a Replicator object after the capture ({type(e).__name__}: {e})",
                                  RuntimeWarning, stacklevel=2)
            for key, value in before.items():
                if value is not None:
                    settings.set(key, value)
            if temporary is not None:
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    stage.RemovePrim(temporary)

    def capture_rtx(self, camera_path, width: int = 1280, height: int = 720, subframes: int = 32,
                    mode: str = "RealTimePathTracing", spp: int = None):
        """Render the current stage with RTX through the camera prim ``camera_path`` and return
        an (height, width, 4) uint8 RGBA array.

        A Replicator render product with the "rgb" annotator is created for the call and one
        orchestrator step is taken with ``subframes`` sub-frames (the same frame rendered that
        many times, so that the real-time denoiser settles or the path tracer accumulates);
        the timeline does not advance.  Capture-on-play is switched off.
        mode   "RealTimePathTracing" (RTX Real-Time 2.0) or "PathTracing" (RTX Interactive;
               ``spp`` samples per pixel per sub-frame, default: what the application was
               started with, 64; the picture accumulates ``spp * subframes`` samples).
        Gaussian splats converge slowly: NVIDIA's own configuration uses 800 accumulation
        steps; raise ``subframes`` if the picture is noisy.  If splat prims do not show at all,
        look at ``session.log_lines("geometry streaming")``: RTX may ask for a new kernel
        started with ``start_kit(tier="full", extra_args=["--/UJITSO/geometry=true"])``.

        Raises RuntimeError("no renderer ...") when RTX is absent (every tier but "full", or a
        machine without a graphics device) or when the frame comes back empty, and ValueError
        when ``camera_path`` is not a Camera prim of the current stage.
        The successful path is UNTESTED on the development host."""
        if mode not in RENDER_MODES:
            raise ValueError(f"mode must be one of {RENDER_MODES}, got {mode!r}")
        if int(width) < 1 or int(height) < 1 or int(subframes) < 1:
            raise ValueError("width, height and subframes must be positive")
        if spp is not None and int(spp) < 1:
            raise ValueError("spp must be positive")
        status = self.renderer_status()
        if not status.available:
            raise RuntimeError("no renderer: " + status.reason)
        frame = self._render(camera_path, width, height, subframes, mode, spp)
        if frame.size == 0:
            self._renderer = None
            raise RuntimeError("no renderer: RTX returned an empty frame (see " + self.log_file + ")")
        return frame

    def start_livestream(self, public_ip: str = "", signal_port: int = 49100, stream_port: int = 47998) -> dict:
        """OPTIONAL, UNTESTED on the development host.  Enable WebRTC streaming of the viewport
        (extension omni.kit.livestream.app) so that the Isaac Sim WebRTC Streaming Client can
        show it next to the notebook.  Returns the connection facts.

        Needs tier "full" with a working renderer and a session started with
        ``start_kit(livestream=True)`` (the UI must stay visible; that is decided at start).
        One client per application, NVENC required, no authentication or encryption, media
        over UDP (an SSH TCP tunnel is not enough).  The picture only moves while Kit is
        updated: call ``session.update(n)`` in a loop."""
        if not self.renderer_available:
            raise RuntimeError("no renderer: " + self.renderer_status().reason)
        if not self.livestream_prepared:
            raise RuntimeError("this session was started without livestream=True; restart the kernel and call "
                               "start_kit(tier='full', livestream=True)")
        if self._livestream is None:
            import carb.settings
            import omni.kit.app
            settings = carb.settings.get_settings()
            base = "/exts/omni.kit.livestream.app/primaryStream"        # read by the extension when it starts
            settings.set(base + "/signalPort", int(signal_port))
            settings.set(base + "/streamPort", int(stream_port))
            if public_ip:
                settings.set(base + "/publicIp", str(public_ip))
            settings.set("/app/window/drawMouse", True)
            manager = omni.kit.app.get_app().get_extension_manager()
            if not manager.set_extension_enabled_immediate("omni.kit.livestream.app", True):
                raise RuntimeError("the extension omni.kit.livestream.app could not be enabled; see " + self.log_file)
            self.update(10)
            self._livestream = {"signal_port_tcp": int(signal_port), "stream_port_udp": int(stream_port),
                                "public_ip": public_ip, "client": "Isaac Sim WebRTC Streaming Client"}
        return dict(self._livestream)


# ---- start ---------------------------------------------------------------------------------------
def _refuse_if_impossible(tier: str) -> dict:
    install = isaacsim_install()
    if not install["found"]:
        raise RuntimeError(
            f"tier '{tier}' needs Isaac Sim, but the package 'isaacsim' cannot be imported by {sys.executable}. "
            "Select the kernel 'ncntm-isaacsim' (scripts/install_kernel.py registers it), or work without Kit: "
            "start_kit(tier='off').")
    if _pxr_loaded():
        origin = pxr_origin(load=False)
        by = _said_by("pxr")
        session = current_session()
        if not by and session is not None and not session.kit_running:
            by = " (by the Kit-free session this kernel already has)"
        raise RuntimeError(
            f"pxr (OpenUSD {origin.version} from {origin.path}) was imported in this process before Kit was started{by}. "
            "Kit brings its own OpenUSD build and cannot start once another one is loaded (it fails with undefined "
            "symbols and the kernel dies later).\n"
            "What to do: Restart the kernel and let these be the first two lines that run:\n"
            "    from ncntm.viz import kit\n"
            "    session = kit.start_kit()\n"
            "Import pxr and the modules that use it (ncntm.viz.stage, ncntm.viz.usd_to_emray) only AFTER that, "
            "on a later line or in a later cell.\n"
            "Or stay without Kit in this kernel: kit.start_kit(tier='off').")
    if tier in KIT_WARP_TIERS and _loaded("warp"):
        version = str(getattr(sys.modules.get("warp"), "__version__", "") or "")
        raise RuntimeError(
            f"warp{' ' + version if version else ''} was imported in this process before Kit was started"
            f"{_said_by('warp')}. Tier '{tier}' loads the Warp that Isaac Sim ships (extension omni.warp.core), and two "
            "Warp builds in one process do not mix: the first Warp array then fails with an ImportError.\n"
            "What to do: Restart the kernel and let these be the first two lines that run:\n"
            "    from ncntm.viz import kit\n"
            "    session = kit.start_kit()\n"
            "Import warp and the modules that use it (ncntm.viz.emray, ncntm.viz.usd_to_emray) only AFTER that, "
            "on a later line or in a later cell.\n"
            "Tier 'usd' brings no Warp of its own and can start as things are: kit.start_kit(tier='usd').")
    accepted, how = licence_status()
    if not accepted:
        markers = licence_markers(install)
        looked = (f" No acceptance recorded by Isaac Sim itself was found either (looked for {', '.join(markers)})."
                  if install["layout"] == "build" and markers else "")
        raise RuntimeError(
            f"Isaac Sim was not started: the NVIDIA Omniverse licence is not accepted in this process ({how}).{looked} "
            f"Read {EULA_URL} ; to accept, put {EULA_VARIABLE}=YES into the environment of the kernel "
            "(scripts/install_kernel.py --accept-eula writes it into the kernel spec, or export it in the shell that "
            "starts Jupyter) and restart the kernel. To work without Isaac Sim: start_kit(tier='off').")
    return install


def start_kit(tier: str = "auto", gpu=None, root=None, quiet: bool = True, *, livestream: bool = False,
              extra_args=()) -> Session:
    """Start Isaac Sim (Kit) in this process, or return the session that already runs.

    tier    "auto" | "off" | "usd" | "core" | "full"  (module docstring; choose_tier())
    gpu     index of the one GPU CUDA may use (nvidia-smi order), or None to leave the
            environment as it is.  NCNTM_KIT_GPU gives a default.
    root    folder for Kit's logs, cache, data and temporary files
            (default NCNTM_KIT_ROOT, else ~/.cache/ncntm/kit)
    quiet   True: Kit's start-up output goes to a file (``session.start_output_file``) and its
            console log level is Error; the cell shows nothing but your own prints.
    livestream   tier "full" only: keep the UI so that ``session.start_livestream()`` can
            stream it (untested on the development host)
    extra_args   more Kit command-line arguments, e.g. ["--/UJITSO/geometry=true"]

    ALWAYS returns a Session, never None: for tier "off" (asked for, or because Isaac Sim is
    not installed) it is the Kit-free session, ``session.app is None``.  Must be the first
    thing in the process that touches pxr, warp, omni or CUDA: in a notebook the first cell is

        from ncntm.viz import kit            # imports neither pxr nor warp nor Isaac Sim
        session = kit.start_kit()

    and ``ncntm.viz.stage`` / ``ncntm.viz.usd_to_emray`` / ``ncntm.viz.emray`` / ``pxr`` /
    ``warp`` are imported after it.
    Raises RuntimeError with what to do when Kit cannot be started (pxr already imported;
    warp already imported and the tier is "core" or "full"; licence not accepted; Kit itself
    gave up, with what it said; a start that was cut short earlier).

    Tier "full" on a machine with RTX: the first start with a new ``root`` compiles the RTX
    shaders into ``<root>/cache`` and can take minutes; later starts take tens of seconds."""
    state = _state()
    choice = choose_tier(tier)
    running = state["session"]
    explicit = choice.source in ("argument", ENV_TIER)
    if running is not None and running.kit_running:             # never a second SimulationApp
        if explicit and choice.tier != running.tier:
            print(f"Kit already runs in this process as tier '{running.tier}'; tier '{choice.tier}' needs a kernel "
                  "restart. Returning the running session.")
        return running
    if state["tainted"]:
        raise RuntimeError(f"Kit is in an undefined state: {state['tainted']}. Restart the kernel, then run "
                           "start_kit() once and let it finish.")
    if running is not None and not (explicit and choice.tier != "off"):
        return running                                          # the Kit-free session stays unless Kit is asked for by name
    if choice.tier == "off":
        gpu, cuda_index, note = _select_gpu(gpu)
        if in_jupyter():
            # ipykernel 7.4.0 now and then aborts inside libzmq while a kernel WITHOUT Kit shuts down (measured: 1 in
            # 4 to 12 kernels; with Kit the exit hook leaves before that code runs).  Harmless, but each abort would
            # leave a core file of about 12 MB.
            _no_core_files(soft_only=True)
        running = state["session"] = Session("off", choice, gpu=gpu, cuda_index=cuda_index, gpu_note=note)
        return running

    _refuse_if_impossible(choice.tier)
    if livestream and choice.tier != "full":
        raise ValueError("livestream=True needs tier 'full'")
    if choice.tier not in KIT_WARP_TIERS and choice.source != "argument" and _loaded("warp"):
        # works here, but the same notebook is refused where the automatic choice is another tier: say so now
        warnings.warn(
            f"warp was imported before kit.start_kit(){_said_by('warp')}. Tier '{choice.tier}' does not mind, but where "
            "start_kit() chooses tier 'full' (a machine whose RTX renderer works) this order is refused: import warp "
            "and the modules that use it (ncntm.viz.emray, ncntm.viz.usd_to_emray) after start_kit().",
            RuntimeWarning, stacklevel=2)
    root = os.path.abspath(os.path.expanduser(os.fspath(root))) if root else default_root()
    for folder in (root, os.path.join(root, "tmp")):
        os.makedirs(folder, exist_ok=True)
    if not os.access(root, os.W_OK):
        raise RuntimeError(f"the Kit folder {root} is not writable: pass root= or set {ENV_ROOT}")
    gpu, cuda_index, note = _select_gpu(gpu)
    _no_core_files()
    if in_jupyter():
        os.environ.setdefault("ISAAC_JUPYTER_KERNEL", "1")      # isaacsim then applies nest_asyncio
        protect_queued_cells()                                  # "Run All" must not lose cells

    args = [STDOUT_ARG, *PRIVACY_ARGS, excluded_extensions_arg()]
    if quiet:
        args += list(QUIET_ARGS)
    args += [str(a) for a in extra_args]
    config = {
        "headless": True,
        "multi_gpu": False,                 # one GPU for the renderer
        "physics_gpu": cuda_index,          # a CUDA index inside this process
        "fast_shutdown": False,             # forced off under Jupyter anyway
        "enable_crashreporter": False,      # do not even load the crash reporter plugin
        "extra_args": args,
    }
    if choice.tier == "full":
        config["renderer"] = RENDER_MODES[0]
        if gpu is not None:
            config["active_gpu"] = gpu      # /renderer/activeGpu counts graphics devices, which CUDA_VISIBLE_DEVICES does not hide
        if livestream:
            config.update({"hide_ui": False, "window_width": 1920, "window_height": 1080, "display_options": 3286})

    # TMPDIR for the time of the start only: Kit's native code (Carbonite, OpenUSD) reads it once and keeps the
    # folder.  Left in place it broke Warp: NVRTC of CUDA 12.9 (Kit's Warp 1.17) refuses a TMPDIR longer than about
    # 120 characters, so every kernel compilation failed in a session with a long root.
    tempfile.gettempdir()                                       # Python keeps the temp folder it has
    saved_tmpdir = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = os.path.join(root, "tmp")
    try:
        app, info = _launch(choice.tier, config, root, quiet)
    finally:
        if saved_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = saved_tmpdir
    _install_exit_hook(app)
    try:
        session = state["session"] = Session(choice.tier, choice, app=app, root=root, gpu=gpu, cuda_index=cuda_index,
                                             gpu_note=note, info=info, livestream=livestream)
    except BaseException as e:              # Kit runs, but nothing holds it: a second start would replace the stage
        state["tainted"] = f"Kit started, but the session could not be set up ({type(e).__name__}: {e})"
        raise
    _open_closed_folders(root)              # the stock application makes a folder nobody may list
    if info["interrupted"]:
        raise KeyboardInterrupt("interrupt received while Kit was starting. The start was completed first, so Kit is "
                                "intact: run the cell again to get the session.")
    return session


def _main(argv=None) -> int:
    """``python -m ncntm.viz.kit``: say what start_kit() would do here, without starting Kit."""
    install = isaacsim_install()
    print("interpreter :", sys.executable)
    print("isaacsim    :", f"{install['layout']} layout, {install['package']}" if install["found"] else "not importable")
    if install["found"] and install["found_at"] != install["package"]:
        print("              found on sys.path as", install["found_at"])
    accepted, how = licence_status()
    print("licence     :", "accepted" if accepted else "NOT accepted", f"({how})")
    if not accepted and licence_markers(install):
        print("              Isaac Sim's own record would be:", ", ".join(licence_markers(install)))
    probe = probe_graphics()
    print("graphics    :", "usable" if probe.hardware else "not usable", f"- {probe.reason} [{probe.seconds} s]")
    print("start_kit() :", choose_tier("auto"))
    print("Kit folder  :", default_root())
    return 0


if __name__ == "__main__":
    sys.exit(_main())
