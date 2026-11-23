#!/usr/bin/env python3
"""Register the Jupyter kernel "ncntm-isaacsim" for THIS machine.

    python scripts/install_kernel.py --accept-eula            # after reading the licence
    python scripts/install_kernel.py --dry-run                # show what would be written
    python scripts/install_kernel.py --uninstall

The kernel is the Python of Isaac Sim, so that a notebook can start Isaac Sim inside its own
process (``ncntm.viz.kit.start_kit``).  A kernel spec is a folder with one file, kernel.json:
the command that starts the kernel and the environment variables it gets.  Paths in it are
machine-specific, which is why the file is generated here and not shipped.

Where Isaac Sim is looked for (the first that fits; ``--python`` overrides)
    1. the interpreter running this script, if it can import ``isaacsim``
    2. $ISAACSIM_PYTHON_EXE, $ISAACSIM_PATH/python.sh, $ISAAC_PATH/python.sh
    3. ``python`` / ``python3`` on PATH, if it can import ``isaacsim``
    4. ~/isaacsim/env_isaacsim/bin/python, ~/isaacsim/python.sh, ~/isaac-sim/python.sh,
       ~/IsaacSim/_build/linux-*/release/python.sh, ~/isaacsim/_build/linux-*/release/python.sh
A pip install is an interpreter; a binary or source build is started through its python.sh
wrapper, which then is the kernel's executable (it sets PYTHONPATH, LD_LIBRARY_PATH, ISAAC_PATH).
Run through python.sh itself ("./python.sh scripts/install_kernel.py"), the script finds that
python.sh again and does not register Kit's bare Python.  For a wrapper the spec also gets
"interrupt_mode": "message", because the kernel then is a child of the shell script.

Environment written into the kernel spec
    ISAAC_JUPYTER_KERNEL=1         always: Isaac Sim then makes asyncio re-entrant (mandatory)
    CUDA_DEVICE_ORDER=PCI_BUS_ID   always: CUDA numbers GPUs like nvidia-smi
    LD_LIBRARY_PATH                only with --lib-path, or if ~/isaacsim/compat_libs exists
    __EGL_VENDOR_LIBRARY_DIRS      only with --egl-vendor-dirs, or if the current value points into
                                   a conda environment (which hides the NVIDIA graphics driver)
    LD_PRELOAD                     only with --preload, or on aarch64 if libgomp exists (Isaac Sim
                                   refuses to import without it there)
    OMNI_KIT_ACCEPT_EULA=YES       only with --accept-eula
Two properties of kernel.json decide how these are written: a value REPLACES the variable the
Jupyter server would pass on (it is not appended), and "${VAR}" is left as literal text when
VAR is unset.  So every value is written out in full and no "${...}" is used.

Only the Python standard library is used; any Python 3.8+ can run this script.
"""
import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys

DEFAULT_NAME = "ncntm-isaacsim"
EULA_VARIABLE = "OMNI_KIT_ACCEPT_EULA"
EULA_URL = "https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html"
SYSTEM_EGL_VENDOR_DIRS = ("/etc/glvnd/egl_vendor.d", "/usr/share/glvnd/egl_vendor.d")     # libglvnd's own search order
MARKER = "NCNTM_KERNEL_PROBE "

# Runs in the TARGET interpreter.  Imports ipykernel only: importing isaacsim would start Carbonite
# and ask for the licence.
PROBE = r"""
import glob, importlib.util, json, os, platform, sys
out = {"python": sys.version.split()[0], "prefix": sys.prefix, "executable": sys.executable, "machine": platform.machine()}
try:
    import ipykernel
    out["ipykernel"] = ipykernel.__version__
except Exception as e:
    out["ipykernel"] = None; out["ipykernel_error"] = "%s: %s" % (type(e).__name__, e)
def spec_folder(name, resolve=True):
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        spec = None
    if spec is None or not spec.origin:
        return ""
    return os.path.dirname(os.path.abspath(os.path.realpath(spec.origin) if resolve else spec.origin))
package = spec_folder("isaacsim")
out["isaacsim"] = package                                   # links resolved, as isaacsim/__init__.py does
out["isaacsim_found_at"] = spec_folder("isaacsim", False)   # as it stands on sys.path
out["isaac_path"] = os.environ.get("ISAAC_PATH", "")        # exported by python.sh
version = ""
up = os.path.dirname
folders = [package, up(up(package)), up(up(up(package))), out["isaac_path"], up(up(out["isaacsim_found_at"]))]
for folder in folders:                                      # pip: in the package; builds: next to python.sh; source: repository
    try:
        version = open(os.path.join(folder, "VERSION")).read().strip() if folder else ""
    except OSError:
        continue
    if version:
        break
out["isaacsim_version"] = version
torch = spec_folder("torch")
out["torch_gomp"] = sorted(glob.glob(os.path.join(torch + ".libs", "libgomp*"))) if torch else []
print("NCNTM_KERNEL_PROBE " + json.dumps(out))
"""


def say(*parts):
    print(*parts)
    sys.stdout.flush()


def is_arm() -> bool:
    machine = platform.machine().lower()
    return sys.platform == "linux" and ("arm" in machine or "aarch" in machine)


def user_data_dir() -> str:
    """Jupyter's per-user data folder, by the rules of jupyter_core (which may not be installed
    in the Python running this script)."""
    if os.environ.get("JUPYTER_DATA_DIR"):
        return os.environ["JUPYTER_DATA_DIR"]
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Jupyter")
    if os.name == "nt":
        return os.path.join(os.environ.get("APPDATA") or home, "jupyter")
    return os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share"), "jupyter")


def is_wrapper(path: str) -> bool:
    """True for a launcher script such as Isaac Sim's python.sh, False for a Python binary."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"#!"
    except OSError:
        return path.endswith(".sh")


def run_probe(executable: str, timeout: float = 120.0) -> dict:
    """Ask the target what it is.  Returns {} if it could not be asked."""
    try:
        done = subprocess.run([executable, "-c", PROBE], capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"probe_error": f"{type(e).__name__}: {e}"}
    for line in done.stdout.splitlines():
        if line.startswith(MARKER):
            try:
                return json.loads(line[len(MARKER):])
            except ValueError:
                break
    tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or [""]
    return {"probe_error": f"exit code {done.returncode}: {tail[0][:200]}"}


def wrapper_of_build(probe: dict) -> str:
    """python.sh of the binary / source build whose interpreter answered ``probe``, or ''.

    A build is recognised as Isaac Sim does it: the package is .../python_packages/isaacsim
    once links are resolved.  python.sh is NOT next to that folder in a source build, where
    <repo>/_build/<platform>/<config>/python_packages is a link to <repo>/source/python_packages
    (premake5.lua of Isaac Sim 6.2.0).  So it is looked for, in this order, in the folder
    python.sh exports as ISAAC_PATH, two folders above the package as it stands on sys.path,
    and four folders above the interpreter (<root>/kit/python/bin/python3)."""
    package = probe.get("isaacsim") or ""
    executable = probe.get("executable") or ""
    kit_python = os.path.join("kit", "python") in executable            # the test isaacsim/__init__.py uses
    if package.split(os.sep)[-2:] != ["python_packages", "isaacsim"] and not kit_python:
        return ""
    roots = [probe.get("isaac_path") or ""]
    for path, levels in ((probe.get("isaacsim_found_at") or "", 2), (package, 2), (executable if kit_python else "", 4)):
        for _ in range(levels):
            path = os.path.dirname(path)
        roots.append(path)
    for root in roots:
        wrapper = os.path.join(root, "python.sh") if root else ""
        if wrapper and os.path.isfile(wrapper) and os.access(wrapper, os.X_OK):
            return os.path.abspath(wrapper)
    return ""


def find_isaac_python(explicit: str = None):
    """Returns (path, kind, how, probe): kind is "interpreter" or "wrapper"; probe is what the
    target said about itself.  Raises SystemExit with advice when nothing is found."""
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if os.path.isdir(path):                             # a folder: a build, a virtual environment, or a source checkout
            inside = [os.path.join(path, name) for name in ("python.sh", os.path.join("bin", "python"), os.path.join("bin", "python3"))]
            inside += sorted(glob.glob(os.path.join(path, "_build", "linux-*", "release", "python.sh")))
            for candidate in inside:
                if os.path.isfile(candidate):
                    path = candidate
                    break
        if not os.path.isfile(path):
            raise SystemExit(f"--python: {path} does not exist")
        if not os.access(path, os.X_OK):
            raise SystemExit(f"--python: {path} is not executable")
        probe = run_probe(path)
        if is_wrapper(path):
            return path, "wrapper", "given with --python", probe
        launcher = wrapper_of_build({**probe, "executable": probe.get("executable") or path})
        if launcher:                                        # Kit's own Python only works through python.sh
            return launcher, "wrapper", f"given with --python (Kit's own Python: started through {launcher})", run_probe(launcher)
        return path, "interpreter", "given with --python", probe

    tried = []

    def interpreter(path, how):
        if not path or not os.path.isfile(path) or not os.access(path, os.X_OK):
            return None
        probe = run_probe(path)
        tried.append(f"{path}: " + (probe.get("probe_error") or ("isaacsim found" if probe.get("isaacsim") else "cannot import isaacsim")))
        if not probe.get("isaacsim"):
            return None
        wrapper = wrapper_of_build(probe)                   # Kit's own Python: must be started through python.sh
        if wrapper:
            return wrapper, "wrapper", how + f" (a build: its launcher is {wrapper})", run_probe(wrapper)
        if probe["isaacsim"].split(os.sep)[-2:] == ["python_packages", "isaacsim"]:
            tried.append(f"{path}: a binary / source build, but its python.sh was not found; give it with --python")
            return None
        return path, "interpreter", how, probe

    def wrapper(path, how):
        if not path or not os.path.isfile(path) or not os.access(path, os.X_OK):
            return None
        tried.append(f"{path}: launcher script")
        return path, "wrapper", how, run_probe(path)

    found = interpreter(os.path.abspath(sys.executable), "the interpreter running this script")
    for variable, tail in (("ISAACSIM_PYTHON_EXE", ""), ("ISAACSIM_PATH", "python.sh"), ("ISAAC_PATH", "python.sh")):
        if found:
            break
        value = os.environ.get(variable)
        if value:
            path = os.path.abspath(os.path.expanduser(os.path.join(value, tail) if tail else value))
            found = (wrapper if is_wrapper(path) else interpreter)(path, f"${variable}")
    for name in ("python", "python3"):
        if found:
            break
        on_path = shutil.which(name)
        if on_path and os.path.abspath(on_path) != os.path.abspath(sys.executable):
            found = interpreter(os.path.abspath(on_path), f"'{name}' on PATH")
    home = os.path.expanduser("~")
    patterns = ("isaacsim/env_isaacsim/bin/python", "isaacsim/python.sh", "isaac-sim/python.sh",
                "IsaacSim/_build/linux-*/release/python.sh", "isaacsim/_build/linux-*/release/python.sh")
    for pattern in patterns:
        if found:
            break
        for path in sorted(glob.glob(os.path.join(home, pattern))):
            found = (wrapper if is_wrapper(path) else interpreter)(path, f"found at ~/{pattern}")
            if found:
                break
    if not found:
        lines = "\n  ".join(tried) or "(nothing to try)"
        raise SystemExit(
            "Isaac Sim was not found. Looked at:\n  " + lines + "\n"
            "Give its Python explicitly:  --python /path/to/venv/bin/python   (pip install)\n"
            "                        or:  --python /path/to/isaacsim/python.sh (binary or source build)")
    return found


def conda_prefix_of(folder: str) -> str:
    """The conda environment a folder lies in ('' if none): the nearest ancestor with conda-meta/."""
    folder = os.path.abspath(folder)
    while True:
        if os.path.isdir(os.path.join(folder, "conda-meta")):
            return folder
        parent = os.path.dirname(folder)
        if parent == folder:
            return ""
        folder = parent


def kernel_environment(args, probe: dict, kind: str = "interpreter"):
    """Returns (env, notes).  Every decision is explained in ``notes``.  ``kind`` is "wrapper"
    when the kernel is started through python.sh, which adds its own folders to the variables."""
    env = {"ISAAC_JUPYTER_KERNEL": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    notes = []

    lib_paths = [os.path.abspath(os.path.expanduser(p)) for p in (args.lib_path or [])]
    if lib_paths:
        notes.append("LD_LIBRARY_PATH: from --lib-path")
    else:
        compat = os.path.join(os.path.expanduser("~"), "isaacsim", "compat_libs")
        if os.path.isdir(compat):
            lib_paths = [compat]
            notes.append(f"LD_LIBRARY_PATH: {compat} exists on this machine (libraries this OS lacks)")
    if lib_paths:
        for folder in lib_paths:
            if not os.path.isdir(folder):
                notes.append(f"WARNING: library folder {folder} does not exist")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_paths)
        inherited = os.environ.get("LD_LIBRARY_PATH")
        if inherited and kind == "wrapper":
            notes.append("note: kernel.json replaces the LD_LIBRARY_PATH of the Jupyter server; the launcher script then "
                         "appends the folders of Isaac Sim itself. Anything else the kernel needs: --lib-path")
        elif inherited:
            notes.append(f"note: the kernel will NOT see the LD_LIBRARY_PATH of the Jupyter server (kernel.json replaces it). "
                         f"This shell has {inherited!r}; add what the kernel needs with --lib-path")

    current = os.environ.get("__EGL_VENDOR_LIBRARY_DIRS")
    if args.egl_vendor_dirs:
        env["__EGL_VENDOR_LIBRARY_DIRS"] = args.egl_vendor_dirs
        notes.append("__EGL_VENDOR_LIBRARY_DIRS: from --egl-vendor-dirs")
    elif current:
        conda = [conda_prefix_of(d) for d in current.split(os.pathsep) if d]
        if any(conda):
            system = [d for d in SYSTEM_EGL_VENDOR_DIRS if os.path.isdir(d)]
            if system:
                env["__EGL_VENDOR_LIBRARY_DIRS"] = os.pathsep.join(system)
                notes.append(f"__EGL_VENDOR_LIBRARY_DIRS: the current value {current!r} points into the conda environment "
                             f"{next(c for c in conda if c)}, which hides the system's graphics drivers; replaced by the system folders")
            else:
                notes.append(f"WARNING: __EGL_VENDOR_LIBRARY_DIRS={current!r} points into a conda environment but no system "
                             f"folder {SYSTEM_EGL_VENDOR_DIRS} exists to replace it; start Jupyter with the variable unset")

    preload = list(args.preload or [])
    if preload:
        notes.append("LD_PRELOAD: from --preload")
    elif is_arm():
        # the exact strings Isaac Sim compares LD_PRELOAD with: glob("/lib/*/libgomp.so.1") and torch's own copy
        preload = sorted(glob.glob("/lib/*/libgomp.so.1")) + [p for p in probe.get("torch_gomp", []) if os.path.isfile(p)]
        if preload:
            notes.append("LD_PRELOAD: aarch64 - Isaac Sim needs libgomp loaded first (DGX Spark playbook)")
        else:
            notes.append("WARNING: aarch64, but /lib/*/libgomp.so.1 was not found; 'import isaacsim' will ask for LD_PRELOAD")
    if preload:
        env["LD_PRELOAD"] = os.pathsep.join(preload)

    if args.accept_eula:
        env[EULA_VARIABLE] = "YES"
        notes.append(f"{EULA_VARIABLE}=YES: you passed --accept-eula")
    return env, notes


def kernels_dir(args, probe: dict) -> str:
    if args.prefix:
        return os.path.join(os.path.abspath(os.path.expanduser(args.prefix)), "share", "jupyter", "kernels")
    if args.sys_prefix:
        prefix = probe.get("prefix")
        if not prefix:
            raise SystemExit("--sys-prefix: the target interpreter did not tell its prefix; use --prefix DIR")
        return os.path.join(prefix, "share", "jupyter", "kernels")
    return os.path.join(user_data_dir(), "kernels")


def how_to_select(name: str, display_name: str, folder: str, args) -> str:
    lines = [
        "How to use it",
        f"  JupyterLab / Notebook : Kernel > Change Kernel... > \"{display_name}\"  (or the launcher tile of that name)",
        f"  VS Code               : open the notebook > Select Kernel (top right) > Jupyter Kernel... > \"{display_name}\"",
        "                          (not listed? Command Palette > 'Developer: Reload Window')",
        f"  command line          : jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name={name} NOTEBOOK.ipynb",
        f"  check                 : jupyter kernelspec list     should show  {name}  {folder}",
    ]
    if args.prefix:
        share = os.path.dirname(os.path.dirname(folder))
        lines.append(f"  --prefix was used: Jupyter and VS Code only find the kernel when started with  JUPYTER_PATH={share}")
    elif args.sys_prefix:
        lines.append("  --sys-prefix was used: only a Jupyter that runs in that environment lists the kernel")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Register the Jupyter kernel 'ncntm-isaacsim' (the Python of Isaac Sim) for this machine.",
        epilog="Run with --dry-run first to see the kernel.json without writing it.")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--user", action="store_true", help="install for the current user (default)")
    where.add_argument("--prefix", metavar="DIR", help="install into DIR/share/jupyter/kernels")
    where.add_argument("--sys-prefix", action="store_true", help="install into the environment of the Isaac Sim interpreter")
    parser.add_argument("--python", metavar="PATH", help="Isaac Sim's interpreter (pip install) or its python.sh (binary / source build)")
    parser.add_argument("--name", default=DEFAULT_NAME, help=f"kernel name (default {DEFAULT_NAME})")
    parser.add_argument("--display-name", metavar="TEXT", help="name shown in the kernel menus")
    parser.add_argument("--lib-path", metavar="DIR", action="append", help="folder for the kernel's LD_LIBRARY_PATH (repeatable)")
    parser.add_argument("--egl-vendor-dirs", metavar="DIRS", help="value for the kernel's __EGL_VENDOR_LIBRARY_DIRS")
    parser.add_argument("--preload", metavar="LIB", action="append", help="library for the kernel's LD_PRELOAD (repeatable)")
    parser.add_argument("--accept-eula", action="store_true",
                        help=f"write {EULA_VARIABLE}=YES into the kernel spec: you accept the NVIDIA Omniverse licence")
    parser.add_argument("--dry-run", action="store_true", help="print what would be written; write nothing")
    parser.add_argument("--uninstall", action="store_true", help="remove the kernel spec")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9._\-]+", args.name):
        parser.error(f"--name {args.name!r}: only letters, digits, '.', '_' and '-' are allowed in a kernel name")

    if args.uninstall:
        probe = run_probe(os.path.abspath(os.path.expanduser(args.python))) if args.sys_prefix and args.python else {}
        if args.sys_prefix and not probe.get("prefix"):
            _, _, _, probe = find_isaac_python(args.python)
        folder = os.path.join(kernels_dir(args, probe), args.name)
        if not os.path.isfile(os.path.join(folder, "kernel.json")):
            say(f"no kernel spec at {folder}: nothing to remove")
            return 0
        if args.dry_run:
            say(f"(dry run) would remove {folder}")
            return 0
        shutil.rmtree(folder)
        say(f"removed {folder}")
        return 0

    path, kind, how, probe = find_isaac_python(args.python)
    say(f"Isaac Sim : {path}")
    say(f"            {'launcher script of a binary / source build' if kind == 'wrapper' else 'interpreter of a pip install'}; {how}")
    if probe.get("probe_error"):
        say(f"WARNING: the target could not be asked about itself ({probe['probe_error']})")
    version = (probe.get("isaacsim_version") or "").split("-")[0].split("+")[0]
    if probe.get("python"):
        say(f"            Python {probe['python']}" + (f", Isaac Sim {version}" if version else "")
            + ("" if probe.get("isaacsim") else "  -- WARNING: this Python cannot import 'isaacsim'"))

    env, notes = kernel_environment(args, probe, kind)
    display_name = args.display_name or (f"ncntm (Isaac Sim {version})" if version else "ncntm (Isaac Sim)")
    spec = {
        "argv": [path, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": display_name,
        "language": "python",
        "metadata": {"debugger": False},
        "env": env,
    }
    if kind == "wrapper":
        # python.sh runs the interpreter as a CHILD process.  A front end that interrupts by sending SIGINT to
        # the process it started would hit the shell, not Python; "message" makes it ask the kernel instead.
        spec["interrupt_mode"] = "message"
        notes.append('interrupt_mode "message": the kernel is a child of the launcher script, so "Interrupt" is sent '
                     "as a message to the kernel instead of a signal to the script")
    folder = os.path.join(kernels_dir(args, probe), args.name)
    target = os.path.join(folder, "kernel.json")
    text = json.dumps(spec, indent=1) + "\n"

    say(f"\n{target}" + ("   (dry run: not written)" if args.dry_run else ""))
    say(text.rstrip())
    for note in notes:
        say("  - " + note)
    if not args.dry_run:
        existed = os.path.isfile(target)
        os.makedirs(folder, exist_ok=True)
        with open(target, "w") as f:
            f.write(text)
        say(f"\nwritten: {target}" + ("   (replaced the kernel spec that was there)" if existed else ""))

    if probe.get("ipykernel"):
        say(f"\nipykernel {probe['ipykernel']} is importable in the target: the kernel can start.")
    else:
        reason = probe.get("ipykernel_error") or probe.get("probe_error") or "unknown"
        say(f"\nWARNING: ipykernel is NOT importable in the target ({reason}). The kernel cannot start until you install it:")
        say(f"    {path} -m pip install ipykernel")
        if kind == "wrapper":
            say("  (python.sh installs into Kit's Python; keep Isaac Sim's pins, e.g. add  numpy==2.3.1  to the command.)")

    if not args.accept_eula:
        say(f"\nThe licence was NOT accepted for you: Isaac Sim will only start in this kernel after you have read {EULA_URL} "
            f"and either re-run this script with --accept-eula or start Jupyter from a shell with {EULA_VARIABLE}=YES.")
    say("\n" + how_to_select(args.name, display_name, folder, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
