#!/usr/bin/env python3
"""Arrival test of ncntm.viz: run it once on a new machine, before the notebooks.

    python scripts/selftest.py [--out DIR] [--tier auto|off|usd|core|full] [--gpu N]

Run it with the interpreter the notebooks will use (the Python of Isaac Sim, or its python.sh).
A plain Python that has only the extra packages works too: Isaac Sim and RTX are then skipped.

It prints one line per check, PASS, FAIL or SKIP with the reason, and writes into --out
(default: <repository>/build/selftest, which git ignores)

    report.json                 every line, the versions, the measurements, session.describe()
    picture_physical.png        the stage of the test drawn by the Warp ray caster, true voxel size
    picture_index.png           the same stage with every voxel drawn as a cube (5 times too flat)
    compare_backends.png        the Warp picture beside the RTX picture, or beside the reason
    rtx_physical.png            only where RTX can draw
    stage/snemi3d.usdc          the stage that was built, with its textures (open it in Isaac Sim)

The checks, in the order they run
    python, packages    interpreter, platform, versions; the packages ncntm.viz cannot do without
    samples             every sample file is there and has the SHA-256 sum of samples/manifest.json
    Isaac Sim           kit.start_kit(tier, gpu) and session.describe().  It comes before anything
                        imports pxr or warp: Isaac Sim brings its own OpenUSD (see ncntm/viz/kit.py)
    converters          instance -> bricks, surface, skeleton, Gaussians for one neurite of snemi3d
    stage               a USD stage with every layer, saved, read back from the file
    Warp                the device of the ray caster and its kernels
    picture physical    the box of the stage, measured in the picture, is 3.072 x 3.072 x 3.0 um
    picture index       ... and 3.072 x 3.072 x 0.6 um when the voxel size is ignored
    side by side        compare_backends.png: the picture to look at first
    RTX                 a frame with its mean and standard deviation, or SKIP with the session's reason

Exit code 0: no check failed.  A check that cannot run on this machine (no Isaac Sim, no RTX, a
package that only Isaac Sim brings) is skipped and does not count as a failure; a missing
package that ncntm.viz itself needs, a wrong sample file or a wrong measurement does.

Nothing is uploaded: Isaac Sim is started through kit.start_kit only, which switches its crash
reporter and telemetry off (the check "Isaac Sim" reads the switches back).  Files are written
into --out and into Isaac Sim's own folder (NCNTM_KIT_ROOT, default ~/.cache/ncntm/kit).

Environment: NCNTM_SAMPLES (samples directory), NCNTM_DEVICE (Warp device, e.g. cpu),
NCNTM_KIT (tier, when --tier is auto), NCNTM_KIT_GPU (when --gpu is not given), NCNTM_KIT_ROOT.

What a PASS means for RTX: a frame came back and is not one flat colour.  Whether it LOOKS right
is for a person to judge in compare_backends.png: no RTX frame existed when this was written.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SAMPLE = "snemi3d"
VOXEL_ZYX_NM = (30.0, 6.0, 6.0)                                         # SNEMI3D: sections 30 nm apart, pixels of 6 nm
BOX_UM = {"physical": (3.072, 3.072, 3.0), "index": (3.072, 3.072, 0.6)}        # (x, y, z): 512 x 512 x 100 voxels
NEEDED = {"numpy": "numpy", "h5py": "h5py", "scipy": "scipy", "zmesh": "zmesh", "kimimaro": "kimimaro", "PIL": "Pillow"}
REPORTED = ("numpy", "h5py", "scipy", "scikit-image", "zmesh", "kimimaro", "Pillow", "warp-lang", "usd-core", "usd-exchange",
            "isaacsim", "torch", "ipykernel", "jupyter_client", "nbconvert", "nbformat")
PICTURE = (960, 720)                                                    # (width, height) of the saved pictures
MEASURE = (640, 480)                                                    # ... of the pictures the box is measured in


class Skip(Exception):
    """A check that cannot run on this machine; the message is the reason."""


def shown(path) -> str:
    """A path as it is printed: relative to the repository when it is inside it."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def need(state, *keys):
    """What earlier checks left in ``state``, or Skip naming the check that did not deliver it."""
    for key in keys:
        if state.get(key) is None:
            raise Skip(f"needs the {key}, which an earlier check did not deliver")
    return [state[key] for key in keys] if len(keys) > 1 else state[keys[0]]


def module(name):
    """Import a module, or Skip with the reason."""
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise Skip(f"{name} cannot be imported by this interpreter ({error})") from None


# ---- the checks -------------------------------------------------------------------------------------
def check_python(state):
    info = {"python": platform.python_version(), "executable": sys.executable, "machine": platform.machine(),
            "system": platform.system(), "release": platform.release(), "libc": " ".join(platform.libc_ver()),
            "repository": str(REPO)}
    state["report"]["environment"] = info
    return f"Python {info['python']} ({info['executable']}) on {info['machine']} {info['system']} {info['release']}"


def check_packages(state):
    versions = {}
    for name in REPORTED:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    state["report"]["packages"] = versions
    missing = [f"{module_name} (package {package})" for module_name, package in NEEDED.items()
               if importlib.util.find_spec(module_name) is None]            # looked up, not imported
    assert not missing, ("ncntm.viz needs these and this interpreter cannot import them: " + ", ".join(missing)
                         + ". Install the extra packages of the hand-over into this interpreter")
    told = ", ".join(f"{name} {version}" for name, version in versions.items() if version)
    absent = [name for name, version in versions.items() if version is None]
    return told + (f"; not installed as packages of this interpreter: {', '.join(absent)}" if absent else "")


def check_samples(state):
    from ncntm.viz import samples                                           # numpy only

    folder = samples.samples_dir()
    entries = samples.manifest()["samples"]                                 # FileNotFoundError says what to do
    wrong, total = [], 0
    for entry in entries:
        path = folder / entry["file"]
        if not path.is_file():
            wrong.append(f"{entry['file']} is missing")
            continue
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                sha.update(block)
        total += path.stat().st_size
        if sha.hexdigest() != entry["sha256"]:
            wrong.append(f"{entry['file']} has SHA-256 {sha.hexdigest()[:12]}..., the manifest says {entry['sha256'][:12]}...")
    state["report"]["samples"] = {"folder": str(folder), "files": [entry["file"] for entry in entries], "bytes": total}
    assert not wrong, f"in {shown(folder)}: " + "; ".join(wrong)
    return f"{len(entries)} files, {total / 1e6:.1f} MB in {shown(folder)}; every SHA-256 sum is the one in manifest.json"


def check_isaac_sim(state, tier, gpu):
    from ncntm.viz import kit                                               # standard library only

    try:
        session = kit.start_kit(tier=tier, gpu=gpu)
        failure = ""
    except RuntimeError as error:                                           # licence, a broken install, ...: go on without it
        failure = str(error)
        session = kit.start_kit(tier="off", gpu=gpu)
    state["session"] = session
    state["report"]["session"] = {
        "tier": session.tier, "choice": str(session.choice), "describe": session.describe(), "kit_root": session.root,
        "kit_version": session.kit_version, "extensions": session.extensions, "start_seconds": session.start_seconds,
        "privacy": session.privacy(), "kit_errors": session.kit_errors(), "usd": session.usd_version,
        "renderer": str(session.renderer_status())}
    assert not failure, f"Isaac Sim did not start, the checks below run without it. start_kit said: {failure}"
    if session.app is None:
        raise Skip(f"not running, plain OpenUSD is used instead ({session.choice.reason})")
    switches = session.privacy()
    assert not any(switches.values()), f"crash reporter or telemetry are NOT all off: {switches}"
    return (f"Kit {session.kit_version}, tier '{session.tier}', {session.extensions} extensions, started in "
            f"{session.start_seconds} s; OpenUSD {session.usd_version}; crash reporter and telemetry off; "
            f"{len(session.kit_errors())} [Error] lines in Kit's log")


def check_converters(state):
    for name in ("h5py", "scipy", "zmesh", "kimimaro"):
        module(name)
    import numpy as np

    from ncntm.viz import convert as C
    from ncntm.viz import samples

    sample = samples.load_sample(SAMPLE)
    grid = sample.grid
    assert grid.voxel_size_zyx_nm == VOXEL_ZYX_NM and grid.anisotropy == 5.0, grid.describe()
    assert tuple(grid.extent_xyz_nm / 1000.0) == BOX_UM["physical"] and tuple(grid.extent_nm("index") / 1000.0) == BOX_UM["index"]
    subject = sample.skeleton_subjects[0]                                   # the neurite that runs across the sections
    mask = sample.mask(subject)
    voxels = np.argwhere(mask)
    lo_nm = grid.index_to_nm(voxels.min(axis=0)[::-1])                      # the box of the neurite, (x, y, z) nm
    hi_nm = grid.index_to_nm(voxels.max(axis=0)[::-1] + 1)
    centres = grid.voxel_centers_nm(voxels)

    bricks = C.instance_to_bricks(mask, grid, instance_id=subject)
    assert len(bricks) > 0 and mask[tuple(bricks.kji.T)].all()              # every brick is a voxel of the neurite
    mesh = C.instance_to_mesh(mask, grid, instance_id=subject)
    assert np.allclose(mesh.vertices_xyz_nm.min(axis=0), lo_nm, atol=1e-3) and np.allclose(mesh.vertices_xyz_nm.max(axis=0), hi_nm, atol=1e-3)
    wire = C.instance_to_mesh(mask, grid, simplify_factor=20, instance_id=subject)
    assert 0 < len(wire.faces) < len(mesh.faces)
    skeleton = C.instance_to_skeleton(mask, grid, instance_id=subject)
    assert mask[tuple(grid.nm_to_voxel(skeleton.vertices_xyz_nm).T)].all()  # every node sits in a voxel of the neurite
    gaussians = C.instance_to_gaussians(mask, grid, k=64, instance_id=subject)
    weights = np.asarray(gaussians.weights, dtype=np.float64)
    off_centre = np.abs((weights[:, None] * gaussians.means_xyz_nm).sum(axis=0) / weights.sum() - centres.mean(axis=0)).max()
    assert abs(weights.sum() - 1.0) < 1e-5 and off_centre < 0.05, (weights.sum(), off_centre)       # nm
    # the mistake this package is about: the same voxels placed with a grid that ignores the section thickness
    flat = C.instance_to_bricks(mask, grid.as_cubes(), instance_id=subject)
    height = np.ptp(bricks.centers_xyz_nm[:, 2]) + bricks.size_xyz_nm[2]
    height_flat = np.ptp(flat.centers_xyz_nm[:, 2]) + flat.size_xyz_nm[2]
    assert abs(height / height_flat - grid.anisotropy) < 1e-9, (height, height_flat)

    state.update(sample=sample, subject=int(subject), bricks=bricks, mesh=mesh, wire=wire, skeleton=skeleton, gaussians=gaussians)
    state["report"]["converters"] = {
        "sample": sample.describe(), "instance": int(subject), "voxels": int(mask.sum()), "bricks": len(bricks),
        "triangles": int(len(mesh.faces)), "wiremesh_edges": int(len(wire.edges())), "skeleton_nodes": int(len(skeleton.vertices_xyz_nm)),
        "skeleton_cable_um": float(skeleton.cable_length_nm() / 1000.0), "gaussians": len(gaussians),
        "height_um": float(height / 1000.0), "height_as_cubes_um": float(height_flat / 1000.0)}
    return (f"{SAMPLE} ({grid.describe()}), instance {subject}: {int(mask.sum()):,} voxels -> {len(bricks):,} bricks, "
            f"{len(mesh.faces):,} triangles, {len(skeleton.vertices_xyz_nm):,} skeleton nodes ({skeleton.cable_length_nm() / 1000.0:.1f} um "
            f"of cable), {len(gaussians)} Gaussians; it is {height / 1000.0:.2f} um high, and {height_flat / 1000.0:.2f} um when the "
            f"voxels are taken for cubes")


def check_stage(state, out):
    sample, subject = need(state, "sample", "subject")
    module("pxr")
    import numpy as np
    from pxr import Usd

    from ncntm.viz import stage as S
    from ncntm.viz import view

    grid = sample.grid
    nz, ny, nx = grid.shape_zyx
    stage = S.new_stage(out / "work" / "authored.usdc")
    ds = S.Dataset.from_sample(stage, sample)
    ds.add_slices("rawvalue", z=0, upsample=1)                              # floor: the stored numbers as greys
    ds.add_slices("voxvalue", y=ny - 1, upsample=1)                         # far wall: normalised, colour-mapped
    ds.add_slices("instance", x=0, upsample=1)                              # other far wall: instance labels
    ds.add_bricks(state["bricks"])
    ds.add_surface(state["mesh"], opacity=0.35)
    ds.add_wiremesh(state["wire"], width_nm=0.003 * float(np.linalg.norm(grid.extent_xyz_nm)))
    ds.add_skeleton(state["skeleton"])
    ds.add_gaussians(state["gaussians"])
    ds.add_frame()
    ds.author_camera()
    ds.set_visible(f"instance/bricks_{subject}", False)                     # in the file, but hidden: bricks and wires
    ds.set_visible("wiremesh", False)                                       # would cover the surface in the pictures
    saved = Path(ds.save(out / "stage" / f"{SAMPLE}.usdc"))                 # a copy elsewhere, textures included
    session = state.get("session")
    reopened = session.open_stage(saved) if session is not None else Usd.Stage.Open(str(saved))
    assert os.path.samefile(reopened.GetRootLayer().realPath, saved)        # read back from the file that was written,
    assert not os.path.samefile(saved, stage.GetRootLayer().realPath)       # not the layer it was authored in

    (dataset,) = S.find_datasets(reopened)
    assert dataset.grid == grid, dataset.grid.describe()                    # shape and voxel size survive the file
    assert list(dataset.layers()) == list(S.LAYERS), list(dataset.layers())
    variants = dataset.prim.GetVariantSets().GetVariantSet(S.SPACE_VARIANT_SET).GetVariantNames()
    assert sorted(variants) == ["index", "physical"] and dataset.space == "physical", (variants, dataset.space)
    scale = {space: tuple(float(v) for v in dataset.voxel_scale(space)) for space in ("physical", "index")}
    assert scale == {"physical": (6.0, 6.0, 30.0), "index": (6.0, 6.0, 6.0)}, scale       # (x, y, z) nm: the anisotropy
    assert view.missing_textures(reopened) == [], view.missing_textures(reopened)
    textures = sorted(p.name for p in (saved.parent / "textures").glob("*.png"))
    assert len(textures) == 3, textures

    by = "Isaac Sim" if session is not None and session.app is not None else "plain OpenUSD"
    state.update(stage=reopened, dataset=dataset)
    state["report"]["stage"] = {"file": str(saved), "bytes": saved.stat().st_size, "layers": list(dataset.layers()),
                                "voxel_scale_xyz_nm": scale, "textures": textures, "reopened_by": by,
                                "splat_schema": bool(S.splat_schema_available())}
    return (f"7 layers, saved as {shown(saved)} ({saved.stat().st_size / 1e6:.1f} MB, 3 slice textures), read back by {by}; "
            f"voxel scale (x, y, z) {scale['physical']} nm in physical and {scale['index']} nm in index space")


def check_warp(state):
    warp = module("warp")
    warp.config.log_level = getattr(warp, "LOG_WARNING", 30)                # no banner between the lines of this test
    from ncntm.viz import emray

    device = emray.default_device()
    seconds = emray.load_kernels(device)
    name = str(warp.get_device(device).name)
    state["device"] = device
    state["report"]["warp"] = {"version": str(warp.__version__), "device": device, "device_name": name, "kernel_seconds": seconds,
                               "kit_build": "omni.warp.core" in (warp.__file__ or "")}
    return f"Warp {warp.__version__} on {device} ({name}); the kernels of the ray caster were ready in {seconds:.1f} s"


def box_in_picture(scene, camera):
    """(lo, hi) of the frame box as the depth buffer of a picture shows it: the world position of
    every pixel of the box lines is eye + ray direction * depth (perspective, one ray per pixel)."""
    import numpy as np

    image = scene.render(camera)
    ids = [oid for oid, o in scene.objects.items() if o["path"].endswith("/frame/bbox")]
    hit = np.isin(image.object_id, ids)
    assert ids and hit.sum() > 100, f"the box lines cover {int(hit.sum())} pixels"
    right, up, forward = camera.basis()
    span_y = 2.0 * math.tan(math.radians(camera.fov) / 2.0)
    u = ((np.arange(camera.width) + 0.5) / camera.width - 0.5) * span_y * camera.width / camera.height
    v = (0.5 - (np.arange(camera.height) + 0.5) / camera.height) * span_y
    direction = forward[None, None, :] + right[None, None, :] * u[None, :, None] + up[None, None, :] * v[:, None, None]
    direction /= np.linalg.norm(direction, axis=2, keepdims=True)
    points = np.asarray(camera.eye)[None, :] + direction[hit] * image.depth[hit][:, None]
    return points.min(axis=0), points.max(axis=0)


def check_picture(state, out, space):
    stage, dataset = need(state, "stage", "dataset")
    need(state, "device")
    import numpy as np

    from ncntm.viz import view

    wanted = np.array(BOX_UM[space])
    # 1. exact: the box of every volume the ray caster samples for the slices (from the transforms on the stage)
    scene = view.scene(stage, space=space)
    boxes = [scene.volume_corners(n) for n in range(len(scene.samplers))]
    assert len(boxes) == 3, len(boxes)                                      # grey, viridis, labels
    for lo, hi in boxes:
        assert np.abs(lo).max() < 1e-9 and np.abs(hi - wanted).max() < 1e-6, (space, lo, hi)
    # 2. in the picture: the lines of the frame box, from the depth buffer (they are 0.3 % of the diagonal wide)
    camera = view.camera_for(stage, size=MEASURE, ssaa=1)
    assert camera.mode == "perspective"
    lo, hi = box_in_picture(view.scene(stage, space=space, layers=["frame"]), camera)
    line = 0.003 * float(np.linalg.norm(dataset.grid.extent_xyz_nm)) / 1000.0
    assert np.abs(lo).max() <= line and np.abs(hi - wanted).max() <= line, (space, lo, hi, wanted, line)
    # 3. the picture for a person
    view.render(stage, size=PICTURE, space=space, backend="warp")
    t0 = time.perf_counter()
    picture = view.render(stage, size=PICTURE, space=space, backend="warp")
    frame_seconds = time.perf_counter() - t0
    drawn = float(np.any(picture[..., :3] < 250, axis=-1).mean())
    assert drawn > 0.02, f"only {100 * drawn:.1f} % of the pixels show anything"
    path = picture.save(out / f"picture_{space}.png")
    assert dataset.space == "physical"                                      # the stage itself was not switched
    state["report"].setdefault("pictures", {})[space] = {
        "file": str(path), "box_um": [float(v) for v in hi - lo], "box_lo_um": [float(v) for v in lo], "expected_um": list(BOX_UM[space]),
        "tolerance_um": line, "drawn_fraction": drawn, "frame_seconds": frame_seconds, "size": list(PICTURE)}
    size = hi - lo
    return (f"the box is {wanted[0]:g} x {wanted[1]:g} x {wanted[2]:g} um in the scene the ray caster draws, and "
            f"{size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} um in the depth buffer of the picture, measured over the outside of "
            f"its {1000 * line:.0f} nm wide lines; {shown(path)} {PICTURE[0]} x {PICTURE[1]}, {100 * drawn:.0f} % of it drawn, "
            f"{frame_seconds:.2f} s a frame")


def check_side_by_side(state, out):
    stage = need(state, "stage")
    need(state, "device")
    from ncntm.viz import view

    path = out / "compare_backends.png"
    view.compare_backends(stage, size=PICTURE, session=state.get("session")).save(path)
    both = view.last_backend() == "rtx"
    state["report"]["compare_backends"] = {"file": str(path), "rtx_in_it": both, "why": view.why()}
    return f"{shown(path)}: the Warp picture " + ("beside the RTX picture" if both else "with the reason why there is no RTX picture")


def check_rtx(state, out):
    session = state.get("session")
    if session is None:
        raise Skip("there is no session (see the check 'Isaac Sim')")
    status = session.renderer_status()
    if not status.available:
        raise Skip(status.reason)                                           # the session's own sentence
    stage = need(state, "stage")
    import numpy as np

    from ncntm.viz import view

    t0 = time.perf_counter()
    frame = view.render(stage, size=PICTURE, backend="rtx", session=session)
    seconds = time.perf_counter() - t0
    path = frame.save(out / "rtx_physical.png")
    mean, deviation = float(frame[..., :3].mean()), float(frame[..., :3].std())
    state["report"]["rtx"] = {"file": str(path), "shape": list(frame.shape), "mean": mean, "std": deviation, "seconds": seconds,
                              "reason": status.reason, "missing_textures": view.missing_textures(stage)}
    assert frame.ndim == 3 and frame.shape[2] == 4 and frame.size > 0, frame.shape
    assert deviation > 0.0, f"the RTX frame is one flat colour (mean {mean:.1f}): nothing was drawn"
    return (f"RTX frame {frame.shape[1]} x {frame.shape[0]} in {seconds:.1f} s, mean {mean:.1f}, standard deviation "
            f"{deviation:.1f} (of 255); saved {shown(path)}. Compare it with the Warp picture in compare_backends.png")


# ---- running them -----------------------------------------------------------------------------------
def run(state, name, check, *args) -> str:
    """Run one check, print its line, note it in the report.  Returns PASS, FAIL or SKIP."""
    t0 = time.perf_counter()
    try:
        detail, status = check(state, *args), "PASS"
    except Skip as reason:
        detail, status = str(reason), "SKIP"
    except Exception as error:                                              # an assertion, or anything a broken install raises
        detail, status = (str(error) if isinstance(error, AssertionError) and str(error) else f"{type(error).__name__}: {error}"), "FAIL"
        state["report"].setdefault("tracebacks", {})[name] = traceback.format_exc()
    seconds = time.perf_counter() - t0
    state["report"]["checks"].append({"check": name, "status": status, "detail": detail, "seconds": round(seconds, 2)})
    print(f"{status}  {name}: {detail}  [{seconds:.1f} s]", flush=True)
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Arrival test of ncntm.viz: one line per check, a JSON report and pictures.")
    parser.add_argument("--out", default=str(REPO / "build" / "selftest"),
                        help="folder for report.json, the pictures and the stage (default: <repository>/build/selftest)")
    parser.add_argument("--tier", default="auto", choices=("auto", "off", "usd", "core", "full"),
                        help="how much of Isaac Sim to start: off = none, usd / core = without renderer, full = with RTX, "
                             "auto = full where a graphics device is found, else usd, else off (default: auto)")
    parser.add_argument("--gpu", type=int, default=None, help="index of the one GPU to use (nvidia-smi order); default: NCNTM_KIT_GPU, else all")
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = {"report": {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "arguments": {"out": str(out), "tier": args.tier, "gpu": args.gpu},
                        "environment_variables": {name: os.environ.get(name) for name in
                                                  ("NCNTM_SAMPLES", "NCNTM_DEVICE", "NCNTM_KIT", "NCNTM_KIT_GPU", "NCNTM_KIT_ROOT",
                                                   "CUDA_VISIBLE_DEVICES")},
                        "checks": []}}
    print(f"ncntm.viz arrival test; output in {shown(out)}", flush=True)
    t0 = time.perf_counter()
    run(state, "python", check_python)
    run(state, "packages", check_packages)
    run(state, "samples", check_samples)
    run(state, "Isaac Sim", check_isaac_sim, args.tier, args.gpu)           # before anything imports pxr or warp
    if state.get("session") is not None:
        print("       " + state["report"]["session"]["describe"].replace("\n", "\n       "), flush=True)
    run(state, "converters", check_converters)
    run(state, "stage", check_stage, out)
    run(state, "Warp", check_warp)
    run(state, "picture physical", check_picture, out, "physical")
    run(state, "picture index", check_picture, out, "index")
    run(state, "side by side", check_side_by_side, out)
    run(state, "RTX", check_rtx, out)

    report = state["report"]
    counts = {status: sum(1 for c in report["checks"] if c["status"] == status) for status in ("PASS", "FAIL", "SKIP")}
    report.update(seconds=round(time.perf_counter() - t0, 1), counts=counts, exit_code=1 if counts["FAIL"] else 0)
    session = state.get("session")
    if session is not None:                                                 # as they stand at the end of the run
        report["session"]["kit_errors"] = session.kit_errors()
        report["session"]["renderer"] = str(session.renderer_status())
    path = out / "report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped in {report['seconds']} s; "
          f"report: {shown(path)}", flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())                # sys.exit, not "raise SystemExit": Isaac Sim's exit hook keeps this status
