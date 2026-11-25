"""Tests of ncntm.viz.kit (Isaac Sim inside a Jupyter kernel) and scripts/install_kernel.py.

    python -m pytest tests/test_kit.py        or        python tests/test_kit.py

The script form prints the start-up and shutdown times of every kernel at the end.

Three groups
    installer    scripts/install_kernel.py with stand-in interpreters: run everywhere
    decisions    the graphics probe and the tier choice, without Kit: run everywhere
    kernels      short notebooks executed in a REAL kernel through jupyter_client: need
                 jupyter_client in the Python running the tests, an Isaac Sim that
                 install_kernel.py can find (with ipykernel), and the licence accepted in the
                 environment (OMNI_KIT_ACCEPT_EULA=YES; the tests never accept it for you).
                 Otherwise every one of them is skipped with the reason.

The kernel spec is installed into a temporary prefix and found through JUPYTER_PATH; nothing
is written to ~/.local or into the Isaac Sim environment.  Kit's files go to a temporary root.
Every Kit start has the crash reporter and telemetry off (asserted) and no core files.

Environment
    NCNTM_ISAAC_PYTHON     Isaac Sim's interpreter or python.sh (default: what install_kernel.py finds)
    NCNTM_TEST_GPU         GPU index the kernels may use (default 0)
    NCNTM_TEST_OUT         folder for everything the tests write (default: a temporary folder)
    NCNTM_TEST_KIT_FULL    0 skips the one test that starts the stock Isaac Sim application
                           (tier "full", tens of seconds)

What these tests cannot show on a machine without a graphics device: a positive
``renderer_available`` and a picture from ``capture_rtx``.  test_full_tier_rtx_path asserts
the negative path there and the positive path where RTX works; in both cases it runs the
capture sequence itself (render product, annotator, orchestrator step), so that every call
in it has been made against the installed Isaac Sim.

A binary / source build of Isaac Sim (python.sh) is stood in for by a folder of that shape
made from the pip install (test_source_build_layout_through_python_sh); on a machine that
has a real build, NCNTM_ISAAC_PYTHON=<build>/python.sh runs all kernel tests through it.
"""
import atexit
import importlib.util
import json
import os
import queue
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
KIT_PY = REPO / "ncntm" / "viz" / "kit.py"

# Python source that binds ``kit`` to ncntm/viz/kit.py loaded BY PATH, for child processes of the Kit-free tests:
# kit.py imports nothing but the standard library, the package around it needs numpy.
LOAD_KIT_BY_PATH = (
    "import importlib.util, sys\n"
    f"_spec = importlib.util.spec_from_file_location('ncntm_viz_kit_by_path', {str(KIT_PY)!r})\n"
    "kit = importlib.util.module_from_spec(_spec)\n"
    "sys.modules[_spec.name] = kit\n"
    "_spec.loader.exec_module(kit)\n")

try:
    from ncntm.viz import kit  # noqa: E402   (pure Python: imports neither pxr nor warp nor isaacsim)
except ImportError:            # the package imports numpy; without it the tests that need no kernel still run
    exec(LOAD_KIT_BY_PATH)

INSTALLER = REPO / "scripts" / "install_kernel.py"
KERNEL_NAME = "ncntm-isaacsim"
GPU = int(os.environ.get("NCNTM_TEST_GPU", "0"))
GRACE_SECONDS = 5.0            # jupyter_client's shutdown_wait_time: after that the kernel is killed
TIMINGS = []                   # one dict per kernel, printed by the script form


class Skip(Exception):
    """Raised by a test that cannot run here."""


def skip(reason):
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest

        pytest.skip(reason)
    raise Skip(reason)


def raises(error, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error:
        return True
    return False


_CACHE = {}


def once(fn):
    """Run ``fn`` once per test session; later calls get the same result or the same exception."""
    def wrapper():
        if fn.__name__ not in _CACHE:
            try:
                _CACHE[fn.__name__] = (fn(), None)
            except BaseException as error:           # Skip included: every dependent test skips alike
                _CACHE[fn.__name__] = (None, error)
        value, error = _CACHE[fn.__name__]
        if error is not None:
            raise error
        return value
    wrapper.__name__ = fn.__name__
    return wrapper


@once
def out_dir() -> Path:
    given = os.environ.get("NCNTM_TEST_OUT")
    if given:
        path = Path(given).resolve() / f"test_kit_{os.getpid()}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = Path(tempfile.mkdtemp(prefix="ncntm_test_kit_"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def installer_module():
    spec = importlib.util.spec_from_file_location("ncntm_install_kernel", INSTALLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_installer(*args, env=None):
    """Run scripts/install_kernel.py in its own process; returns (exit code, stdout + stderr)."""
    done = subprocess.run([sys.executable, str(INSTALLER), *map(str, args)], capture_output=True, text=True,
                          env=env, timeout=300, stdin=subprocess.DEVNULL)
    return done.returncode, done.stdout + done.stderr


def spec_of(prefix, name=KERNEL_NAME) -> dict:
    with open(Path(prefix) / "share" / "jupyter" / "kernels" / name / "kernel.json") as f:
        return json.load(f)


def make_script(path, text) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


# ---- installer -------------------------------------------------------------------------------
def test_installer_dry_run_writes_nothing_and_explains():
    prefix = out_dir() / "dry"
    code, text = run_installer("--dry-run", "--prefix", prefix, "--python", sys.executable)
    assert code == 0, text
    assert not prefix.exists()                                            # nothing written
    assert '"ISAAC_JUPYTER_KERNEL": "1"' in text and '"CUDA_DEVICE_ORDER": "PCI_BUS_ID"' in text
    assert "OMNI_KIT_ACCEPT_EULA" not in text.split('"env"')[1].split("}")[0]     # not in the env block
    assert "NOT accepted for you" in text and "--accept-eula" in text     # the one sentence
    assert "JupyterLab" in text and "VS Code" in text and "JUPYTER_PATH" in text
    assert KERNEL_NAME in text


def test_installer_writes_the_spec_it_prints_and_uninstalls():
    prefix = out_dir() / "explicit"
    lib_a, lib_b = out_dir() / "lib_a", out_dir() / "lib_b"
    lib_a.mkdir(exist_ok=True)
    lib_b.mkdir(exist_ok=True)
    code, text = run_installer("--prefix", prefix, "--python", sys.executable, "--accept-eula", "--name", "ncntm-test",
                               "--display-name", "A test kernel", "--lib-path", lib_a, "--lib-path", lib_b,
                               "--preload", "/somewhere/libgomp.so.1", "--egl-vendor-dirs", "/etc/egl:/usr/egl", "--gpu", "3")
    assert code == 0, text
    spec = spec_of(prefix, "ncntm-test")
    assert spec["argv"] == [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    assert spec["display_name"] == "A test kernel" and spec["language"] == "python"
    env = spec["env"]
    assert env["ISAAC_JUPYTER_KERNEL"] == "1" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["OMNI_KIT_ACCEPT_EULA"] == "YES"                           # only because of --accept-eula
    assert env[kit.ENV_GPU] == "3"                                        # only because of --gpu: the default of start_kit(gpu=)
    assert env["LD_LIBRARY_PATH"] == f"{lib_a}{os.pathsep}{lib_b}"
    assert env["LD_PRELOAD"] == "/somewhere/libgomp.so.1"
    assert env["__EGL_VENDOR_LIBRARY_DIRS"] == "/etc/egl:/usr/egl"
    assert not any("${" in value for value in env.values())               # ${VAR} stays literal when VAR is unset
    assert json.dumps(spec, indent=1) in text                             # what was printed is what was written
    assert "NOT accepted for you" not in text
    assert "interrupt_mode" not in spec                                   # an interpreter is interrupted by signal, as usual

    # a second run replaces the spec and says so; without --accept-eula the licence line is gone again
    code, text = run_installer("--prefix", prefix, "--python", sys.executable, "--name", "ncntm-test")
    assert code == 0 and "replaced the kernel spec that was there" in text, text
    again = spec_of(prefix, "ncntm-test")
    assert "OMNI_KIT_ACCEPT_EULA" not in again["env"] and "NOT accepted for you" in text
    assert kit.ENV_GPU not in again["env"]                                # no --gpu: the kernel is not tied to a GPU
    assert again["argv"] == spec["argv"]
    assert os.listdir(prefix / "share" / "jupyter" / "kernels") == ["ncntm-test"]      # one spec, not two

    code, text = run_installer("--prefix", prefix, "--name", "ncntm-test", "--uninstall")
    assert code == 0 and "removed" in text
    assert not (prefix / "share" / "jupyter" / "kernels" / "ncntm-test").exists()
    code, text = run_installer("--prefix", prefix, "--name", "ncntm-test", "--uninstall")
    assert code == 0 and "nothing to remove" in text


def test_installer_locations():
    """--user (the default), --prefix and --sys-prefix; nothing of this test touches the real home."""
    home = out_dir() / "home_user"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("JUPYTER_DATA_DIR", "XDG_DATA_HOME", "JUPYTER_PATH")}
    env["HOME"] = str(home)
    code, text = run_installer("--python", sys.executable, env=env)                        # --user is the default
    assert code == 0, text
    assert (home / ".local" / "share" / "jupyter" / "kernels" / KERNEL_NAME / "kernel.json").is_file()
    env["JUPYTER_DATA_DIR"] = str(out_dir() / "jupyter_data")
    code, text = run_installer("--user", "--python", sys.executable, env=env)
    assert code == 0 and (out_dir() / "jupyter_data" / "kernels" / KERNEL_NAME / "kernel.json").is_file()
    code, text = run_installer("--sys-prefix", "--dry-run", "--python", sys.executable, env=env)
    assert code == 0 and os.path.join(sys.prefix, "share", "jupyter", "kernels", KERNEL_NAME, "kernel.json") in text
    code, text = run_installer("--user", "--uninstall", env=env)
    assert code == 0 and not (out_dir() / "jupyter_data" / "kernels" / KERNEL_NAME).exists()


def test_installer_finds_isaac_sim():
    """The search order, with stand-ins for what the candidates say about themselves."""
    module = installer_module()
    build = out_dir() / "found_build"
    wrapper = make_script(build / "python.sh", "#!/bin/bash\n")
    kit_python = build / "kit" / "python" / "bin" / "python3"                # Kit's own Python: a binary, not a script
    kit_python.parent.mkdir(parents=True, exist_ok=True)
    if not kit_python.exists():
        os.symlink(sys.executable, kit_python)
    venv_python = make_script(out_dir() / "found_venv" / "bin" / "python", "#!/bin/bash\n")
    package = str(build / "python_packages" / "isaacsim")
    answers = {str(wrapper): {"isaacsim": package, "ipykernel": "7.4.0"},
               str(kit_python): {"isaacsim": package, "ipykernel": "7.4.0"},          # Kit's own Python inside a build
               str(venv_python): {"isaacsim": "/venv/lib/python3.12/site-packages/isaacsim", "ipykernel": "7.4.0"}}
    saved = module.run_probe, module.sys.executable, dict(os.environ), module.shutil.which
    empty_home = out_dir() / "home_empty"
    empty_home.mkdir(exist_ok=True)
    try:
        module.run_probe = lambda path, timeout=120.0: answers.get(str(path), {"isaacsim": ""})
        module.shutil.which = lambda name: None
        for name in ("ISAACSIM_PYTHON_EXE", "ISAACSIM_PATH", "ISAAC_PATH"):
            os.environ.pop(name, None)
        os.environ["HOME"] = str(empty_home)
        module.sys.executable = str(venv_python)                              # 1. the running interpreter (a pip install)
        path, kind, how, _ = module.find_isaac_python()
        assert (path, kind) == (str(venv_python), "interpreter") and "running this script" in how
        module.sys.executable = str(kit_python)                               # ... which inside a build means: use python.sh
        path, kind, how, _ = module.find_isaac_python()
        assert (path, kind) == (str(wrapper), "wrapper")
        module.sys.executable = "/usr/bin/some-other-python"
        os.environ["ISAACSIM_PYTHON_EXE"] = str(wrapper)                      # 2. the variable of the DGX Spark playbook
        path, kind, how, _ = module.find_isaac_python()
        assert (path, kind, how) == (str(wrapper), "wrapper", "$ISAACSIM_PYTHON_EXE")
        del os.environ["ISAACSIM_PYTHON_EXE"]
        os.environ["ISAACSIM_PATH"] = str(build)
        assert module.find_isaac_python()[:2] == (str(wrapper), "wrapper")
        del os.environ["ISAACSIM_PATH"]
        module.shutil.which = lambda name: str(venv_python) if name == "python" else None      # 3. python on PATH
        assert module.find_isaac_python()[:2] == (str(venv_python), "interpreter")
        module.shutil.which = lambda name: None
        standard = make_script(empty_home / "isaacsim" / "_build" / "linux-aarch64" / "release" / "python.sh", "#!/bin/bash\n")
        assert module.find_isaac_python()[:2] == (str(standard), "wrapper")                    # 4. a usual place
        standard.unlink()
        try:                                                                  # nothing anywhere: say what to do
            module.find_isaac_python()
            raise AssertionError("expected SystemExit")
        except SystemExit as error:
            assert "--python" in str(error) and "python.sh" in str(error)
        path, kind, how, _ = module.find_isaac_python(str(build))            # --python with a folder
        assert (path, kind, how) == (str(wrapper), "wrapper", "given with --python")
        path, kind, how, _ = module.find_isaac_python(str(kit_python))       # --python with Kit's own Python: use python.sh
        assert (path, kind) == (str(wrapper), "wrapper") and "started through" in how

        # A SOURCE build: python_packages is a link into <repo>/source, so python.sh is NOT two folders above
        # the (resolved) package.  Each of the three clues alone must lead to it.
        repo = out_dir() / "found_source" / "IsaacSim"
        release = repo / "_build" / "linux-aarch64" / "release"
        launcher = make_script(release / "python.sh", "#!/bin/bash\n")
        resolved = str(repo / "source" / "python_packages" / "isaacsim")
        clues = {"isaac_path": str(release), "isaacsim_found_at": str(release / "python_packages" / "isaacsim"),
                 "executable": str(release / "kit" / "python" / "bin" / "python3")}
        for name, value in clues.items():
            assert module.wrapper_of_build({"isaacsim": resolved, name: value}) == str(launcher), name
        assert module.find_isaac_python(str(repo))[:2] == (str(launcher), "wrapper")     # --python with the checkout itself
        assert module.wrapper_of_build({"isaacsim": resolved}) == ""         # no clue: not found ...
        assert module.wrapper_of_build({"isaacsim": "/venv/lib/python3.12/site-packages/isaacsim",
                                        "executable": "/venv/bin/python", "isaac_path": str(release)}) == ""     # pip: none
        source_python = release / "kit" / "python" / "bin" / "python3"
        source_python.parent.mkdir(parents=True, exist_ok=True)
        if not source_python.exists():
            os.symlink(saved[1], source_python)
        launcher.unlink()                                                    # ... and a build without python.sh is
        answers[str(source_python)] = {"isaacsim": resolved, "ipykernel": "7.4.0", **clues}     # never registered bare
        module.sys.executable = str(source_python)
        try:
            module.find_isaac_python()
            raise AssertionError("expected SystemExit")
        except SystemExit as error:
            assert "its python.sh was not found" in str(error)
    finally:
        module.run_probe, module.sys.executable, module.shutil.which = saved[0], saved[1], saved[3]
        os.environ.clear()
        os.environ.update(saved[2])


def source_build_tree(name, python, package_files=None, eula=True):
    """A folder shaped like a SOURCE build of Isaac Sim 6.2.0 (premake5.lua, python.sh at the tag):

        <repo>/.eula_accepted                                written by build.sh once the licence is accepted
        <repo>/source/python_packages/isaacsim/__init__.py
        <repo>/_build/<platform>/release/python_packages  -> link to <repo>/source/python_packages
        <repo>/_build/<platform>/release/python.sh           exports ISAAC_PATH, CARB_APP_PATH, EXP_PATH, sources
                                                             setup_python_env.sh, preloads kit/libcarb.so and runs
                                                             the interpreter as a CHILD process
    ``python`` stands in for <release>/kit/python/bin/python3.  Returns (repo, release)."""
    repo = out_dir() / name / "IsaacSim"
    release = repo / "_build" / "linux-x86_64" / "release"
    package = repo / "source" / "python_packages" / "isaacsim"
    package.mkdir(parents=True, exist_ok=True)
    release.mkdir(parents=True, exist_ok=True)
    for source in package_files or ():
        shutil.copy(source, package / os.path.basename(source))
    if not package_files:
        (package / "__init__.py").write_text("")
    if not (release / "python_packages").exists():
        os.symlink(os.path.join(os.pardir, os.pardir, os.pardir, "source", "python_packages"), release / "python_packages")
    if eula:
        (repo / ".eula_accepted").touch()
    (release / "setup_python_env.sh").write_text(
        "export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/python_packages:$SCRIPT_DIR/exts/isaacsim.simulation_app:$SCRIPT_DIR/kit/kernel/py\n"
        "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SCRIPT_DIR/kit:$SCRIPT_DIR/kit/kernel/plugins\n")
    make_script(release / "python.sh",
                "#!/bin/bash\nset -e\n"
                'error_exit()\n{\n    echo "There was an error running python"\n    exit 1\n}\n'
                'SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\n'
                "export CARB_APP_PATH=$SCRIPT_DIR/kit\nexport ISAAC_PATH=$SCRIPT_DIR\nexport EXP_PATH=$SCRIPT_DIR/apps\n"
                "source ${SCRIPT_DIR}/setup_python_env.sh\n"
                f'python_exe=${{PYTHONEXE:-"{python}"}}\n'
                'export RESOURCE_NAME="IsaacSim"\n'
                "if [ -f $SCRIPT_DIR/kit/libcarb.so ]; then\n"
                "    export LD_PRELOAD=$SCRIPT_DIR/kit/libcarb.so${LD_PRELOAD:+:$LD_PRELOAD}\nfi\n"
                '$python_exe "$@" $args || error_exit\n')
    return repo, release


def test_installer_run_through_python_sh_registers_python_sh():
    """"./python.sh scripts/install_kernel.py" in a source build: the interpreter running the script is
    Kit's bare Python, which only works through python.sh.  The kernel must be python.sh."""
    repo, release = source_build_tree("installer_source_build", sys.executable)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "ISAAC_PATH", "ISAACSIM_PATH", "ISAACSIM_PYTHON_EXE")}
    prefix = out_dir() / "installer_source_build" / "prefix"
    done = subprocess.run([str(release / "python.sh"), str(INSTALLER), "--prefix", str(prefix)], capture_output=True,
                          text=True, env=env, timeout=300, stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stdout + done.stderr
    spec = spec_of(prefix)
    assert spec["argv"][0] == str(release / "python.sh"), spec["argv"]    # not sys.executable
    assert spec["interrupt_mode"] == "message"                            # the kernel is the script's child
    assert "launcher script of a binary / source build" in done.stdout
    assert "PYTHONPATH" not in spec["env"]                                # python.sh sets that itself


def test_installer_environment_is_conditional():
    """LD_LIBRARY_PATH, __EGL_VENDOR_LIBRARY_DIRS and LD_PRELOAD appear only when they apply."""
    home = out_dir() / "home_plain"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("__EGL_VENDOR_LIBRARY_DIRS", "LD_LIBRARY_PATH", "JUPYTER_DATA_DIR")}
    env["HOME"] = str(home)                                               # no ~/isaacsim/compat_libs here
    code, text = run_installer("--prefix", out_dir() / "plain", "--python", sys.executable, env=env)
    assert code == 0, text
    plain = spec_of(out_dir() / "plain")["env"]
    arm = installer_module().is_arm()
    expected = {"ISAAC_JUPYTER_KERNEL", "CUDA_DEVICE_ORDER"} | ({"LD_PRELOAD"} if arm and "LD_PRELOAD" in plain else set())
    assert set(plain) == expected, plain

    # a machine with ~/isaacsim/compat_libs and a conda environment that redirects EGL
    home = out_dir() / "home_conda"
    (home / "isaacsim" / "compat_libs").mkdir(parents=True, exist_ok=True)
    conda = out_dir() / "conda"
    (conda / "conda-meta").mkdir(parents=True, exist_ok=True)
    vendor = conda / "share" / "glvnd" / "egl_vendor.d"
    vendor.mkdir(parents=True, exist_ok=True)
    env.update(HOME=str(home), __EGL_VENDOR_LIBRARY_DIRS=str(vendor))
    code, text = run_installer("--prefix", out_dir() / "conda_prefix", "--python", sys.executable, env=env)
    assert code == 0, text
    fixed = spec_of(out_dir() / "conda_prefix")["env"]
    assert fixed["LD_LIBRARY_PATH"] == str(home / "isaacsim" / "compat_libs")
    system = [d for d in installer_module().SYSTEM_EGL_VENDOR_DIRS if os.path.isdir(d)]
    if system:
        assert fixed["__EGL_VENDOR_LIBRARY_DIRS"] == os.pathsep.join(system)
        assert str(conda) in text and "conda" in text                     # and it says why
    else:
        assert "__EGL_VENDOR_LIBRARY_DIRS" not in fixed and "WARNING" in text

    # aarch64 (DGX Spark): libgomp must be preloaded, spelled exactly as Isaac Sim's own check globs it
    module = installer_module()
    import glob
    system_gomp = sorted(glob.glob("/lib/*/libgomp.so.1"))
    torch_gomp = make_script(out_dir() / "torch.libs" / "libgomp-stand-in.so.1", "")
    args = SimpleNamespace(lib_path=None, egl_vendor_dirs=None, preload=None, accept_eula=False)
    saved_arm, saved_home, saved_egl = module.is_arm, os.environ.get("HOME"), os.environ.pop("__EGL_VENDOR_LIBRARY_DIRS", None)
    try:
        os.environ["HOME"] = str(out_dir() / "home_plain")
        module.is_arm = lambda: True
        on_arm, notes = module.kernel_environment(args, {"torch_gomp": [str(torch_gomp), "/no/such/libgomp.so"]})
        assert on_arm["LD_PRELOAD"] == os.pathsep.join(system_gomp + [str(torch_gomp)])
        assert set(on_arm) == {"ISAAC_JUPYTER_KERNEL", "CUDA_DEVICE_ORDER", "LD_PRELOAD"} and any("aarch64" in n for n in notes)
        module.is_arm = lambda: False
        on_x86, _ = module.kernel_environment(args, {"torch_gomp": [str(torch_gomp)]})
        assert "LD_PRELOAD" not in on_x86
    finally:
        module.is_arm = saved_arm
        os.environ["HOME"] = saved_home
        if saved_egl is not None:
            os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = saved_egl

    # the whole command on a machine that says it is aarch64 (platform.machine patched): the printed spec
    import contextlib
    import io
    module = installer_module()
    saved_machine, saved_glob, saved_home = module.platform.machine, module.glob.glob, os.environ.get("HOME")
    saved_egl = os.environ.pop("__EGL_VENDOR_LIBRARY_DIRS", None)
    try:
        os.environ["HOME"] = str(out_dir() / "home_plain")
        module.platform.machine = lambda: "aarch64"
        assert module.is_arm() is (sys.platform == "linux")
        module.glob.glob = lambda pattern, **kw: ["/lib/aarch64-linux-gnu/libgomp.so.1"] if pattern == "/lib/*/libgomp.so.1" else saved_glob(pattern, **kw)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            assert module.main(["--dry-run", "--prefix", str(out_dir() / "arm"), "--python", sys.executable]) == 0
        text = printed.getvalue()
        if sys.platform == "linux":
            assert '"LD_PRELOAD": "/lib/aarch64-linux-gnu/libgomp.so.1"' in text and "aarch64" in text, text
        module.glob.glob = lambda pattern, **kw: [] if pattern == "/lib/*/libgomp.so.1" else saved_glob(pattern, **kw)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            assert module.main(["--dry-run", "--prefix", str(out_dir() / "arm"), "--python", sys.executable]) == 0
        text = printed.getvalue()
        if sys.platform == "linux":                                       # no libgomp: no variable, but a warning
            assert "LD_PRELOAD" not in text.split('"env"')[1].split("}")[0] and "WARNING: aarch64" in text, text
        module.platform.machine = lambda: "x86_64"
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            assert module.main(["--dry-run", "--prefix", str(out_dir() / "arm"), "--python", sys.executable]) == 0
        assert "LD_PRELOAD" not in printed.getvalue()
    finally:
        module.platform.machine, module.glob.glob = saved_machine, saved_glob
        os.environ["HOME"] = saved_home
        if saved_egl is not None:
            os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = saved_egl

    # the same variable pointing somewhere that is NOT a conda environment is left alone
    other = out_dir() / "not_conda" / "egl_vendor.d"
    other.mkdir(parents=True, exist_ok=True)
    env["__EGL_VENDOR_LIBRARY_DIRS"] = str(other)
    code, text = run_installer("--dry-run", "--prefix", out_dir() / "x", "--python", sys.executable, env=env)
    assert code == 0 and "__EGL_VENDOR_LIBRARY_DIRS" not in text.split('"env"')[1].split("}")[0]


def test_installer_uses_a_launcher_script_as_the_executable():
    """A binary / source build is started through python.sh: the wrapper itself is argv[0]."""
    wrapper = make_script(out_dir() / "build" / "python.sh", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    code, text = run_installer("--prefix", out_dir() / "wrap", "--python", wrapper)
    assert code == 0, text
    assert spec_of(out_dir() / "wrap")["argv"][0] == str(wrapper)
    assert "launcher script" in text
    # a folder is accepted too: python.sh inside it is found
    code, text = run_installer("--dry-run", "--prefix", out_dir() / "wrap2", "--python", wrapper.parent)
    assert code == 0 and str(wrapper) in text


def test_installer_says_how_to_install_ipykernel():
    broken = make_script(out_dir() / "broken" / "python.sh", "#!/bin/sh\necho no python here >&2\nexit 3\n")
    code, text = run_installer("--dry-run", "--prefix", out_dir() / "y", "--python", broken)
    assert code == 0
    assert "ipykernel is NOT importable" in text and f"{broken} -m pip install ipykernel" in text


def test_installer_preload_names_torch_as_isaac_sim_will():
    """aarch64, pip install: Isaac Sim refuses to import unless LD_PRELOAD holds, as TEXT, every file of
    glob(torch.__path__[0] + ".libs/libgomp*").  With the environment behind a symbolic link the resolved
    path is another text; the installer must write the one Isaac Sim will compare with."""
    real = out_dir() / "torch_real" / "site-packages"
    (real / "torch").mkdir(parents=True, exist_ok=True)
    (real / "torch" / "__init__.py").write_text("")
    (real / "torch.libs").mkdir(exist_ok=True)
    (real / "torch.libs" / "libgomp-stand-in.so.1").write_text("")
    link = out_dir() / "torch_link"
    if not link.exists():
        os.symlink(real.parent, link)
    module = installer_module()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(link / "site-packages"), PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-c", module.PROBE], capture_output=True, text=True, env=env, timeout=120)
    line = [line for line in done.stdout.splitlines() if line.startswith(module.MARKER)]
    assert line, done.stdout + done.stderr
    probe = json.loads(line[0][len(module.MARKER):])
    expected = str(link / "site-packages" / "torch.libs" / "libgomp-stand-in.so.1")
    assert probe["torch_gomp"] == [expected], probe["torch_gomp"]         # through the link, as torch.__path__[0] is
    # ... which is exactly what Isaac Sim's own check computes in that interpreter
    check = ("import glob, os, torch\n"
             "print('ISAAC ' + ':'.join(glob.glob(os.path.join(f'{torch.__path__[0]}.libs', 'libgomp*'))))")
    done = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, env=env, timeout=120)
    assert f"ISAAC {expected}" in done.stdout, done.stdout + done.stderr
    args = SimpleNamespace(lib_path=None, egl_vendor_dirs=None, preload=None, accept_eula=False)
    saved = module.is_arm
    try:
        module.is_arm = lambda: True
        on_arm, _ = module.kernel_environment(args, probe)
    finally:
        module.is_arm = saved
    assert expected in on_arm["LD_PRELOAD"].split(os.pathsep)


def test_installer_rejects_bad_input():
    code, text = run_installer("--dry-run", "--name", "bad name", "--python", sys.executable)
    assert code == 2 and "kernel name" in text
    code, text = run_installer("--dry-run", "--gpu", "-1", "--python", sys.executable)
    assert code == 2 and "GPU index" in text
    code, text = run_installer("--dry-run", "--python", out_dir() / "does_not_exist")
    assert code != 0 and "does not exist" in text
    code, text = run_installer("--user", "--prefix", out_dir() / "z")       # mutually exclusive
    assert code == 2


# ---- decisions that need no Kit -----------------------------------------------------------------
def test_probe_graphics_is_structured_cheap_and_safe():
    probe = kit.probe_graphics(refresh=True)
    assert isinstance(probe, kit.GraphicsProbe) and isinstance(probe.hardware, bool)
    assert probe.reason and probe.seconds < 10.0
    assert probe.hardware == (probe.initialised > 0)
    if probe.nvidia_devices and not probe.initialised:                    # the development host: enumerate, cannot initialise
        assert not probe.hardware and probe.errors and "eglInitialize failed" in probe.reason
    print(f"probe_graphics: {probe.reason} [{probe.seconds} s]")
    assert kit.probe_graphics() is probe                                  # cached

    original = kit._EGL_PROBE
    try:                                                                  # a driver that crashes, one that hangs, no libEGL
        kit._EGL_PROBE = "import os; os._exit(139)"                       # what a crashing driver leaves: no result
        crashed = kit.probe_graphics(refresh=True)
        assert not crashed.hardware and "exit code" in crashed.reason
        kit._EGL_PROBE = "import time; time.sleep(30)"
        start = time.perf_counter()
        hung = kit.probe_graphics(timeout=1.0, refresh=True)
        assert not hung.hardware and "did not finish" in hung.reason and time.perf_counter() - start < 5.0
        kit._EGL_PROBE = original.replace('"libEGL.so.1"', '"libNoSuchEGL.so.1"').replace('find_library("EGL")', 'find_library("NoSuchEGL")')
        missing = kit.probe_graphics(refresh=True)
        assert not missing.hardware and "libEGL could not be loaded" in missing.reason
    finally:
        kit._EGL_PROBE = original
        kit._state()["probe"] = probe

    # A conda environment points __EGL_VENDOR_LIBRARY_DIRS at its own folder: EGL then lists Mesa only, although
    # the machine has a working NVIDIA device.  The probe asks a second time without the variable.
    def device(nvidia, initialised=False, error=""):
        return {"index": 0, "extensions": "EGL_NV_device_cuda" if nvidia else "EGL_MESA_device_software", "nvidia": nvidia,
                "software": not nvidia, "tried": nvidia, "initialised": initialised, "error": error, "vendor": "", "version": "1.5", "seconds": 0.0}

    def stand_in(timeout, env):
        asked.append(kit.EGL_VENDOR_VARIABLE in env)
        hidden = hides and kit.EGL_VENDOR_VARIABLE in env
        return {"library": "libEGL.so.1", "error": "", "client_extensions": "",
                "devices": [device(False)] if hidden else [device(True, *answer), device(False)]}, ""

    saved_run, saved_value = kit._run_egl_probe, os.environ.get(kit.EGL_VENDOR_VARIABLE)
    try:
        kit._run_egl_probe = stand_in
        hides = True                                                      # the variable points into a conda environment
        os.environ[kit.EGL_VENDOR_VARIABLE] = "/opt/conda/share/glvnd/egl_vendor.d"
        asked, answer = [], (True,)                                       # a DGX Spark inside a conda environment
        hidden = kit.probe_graphics(refresh=True)
        assert asked == [True, False] and hidden.hardware and hidden.nvidia_devices == 1
        assert "ignored" in hidden.reason and "/opt/conda/share/glvnd/egl_vendor.d" in hidden.reason
        asked, answer = [], (False, "EGL_BAD_ACCESS")                     # the development host inside a conda environment
        hidden = kit.probe_graphics(refresh=True)
        assert asked == [True, False] and not hidden.hardware and hidden.errors == ("EGL_BAD_ACCESS",)
        # The kernel spec of install_kernel.py sets the variable to the SYSTEM folders: the NVIDIA devices are listed
        # with it.  Then it hides nothing: no second question, and the reason must not blame the variable.
        hides = False
        os.environ[kit.EGL_VENDOR_VARIABLE] = "/etc/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d"
        asked, answer = [], (False, "EGL_BAD_ACCESS")                     # the development host, kernel from the spec
        plain = kit.probe_graphics(refresh=True)
        assert asked == [True] and not plain.hardware and plain.errors == ("EGL_BAD_ACCESS",)
        assert "ignored" not in plain.reason and "hides" not in plain.reason and kit.EGL_VENDOR_VARIABLE not in plain.reason
        asked, answer = [], (True,)                                       # a DGX Spark, kernel from the spec
        plain = kit.probe_graphics(refresh=True)
        assert asked == [True] and plain.hardware and kit.EGL_VENDOR_VARIABLE not in plain.reason
        del os.environ[kit.EGL_VENDOR_VARIABLE]
        asked, answer = [], (True,)
        assert kit.probe_graphics(refresh=True).hardware and asked == [False]         # no variable: asked once
    finally:
        kit._run_egl_probe = saved_run
        os.environ.pop(kit.EGL_VENDOR_VARIABLE, None)
        if saved_value is not None:
            os.environ[kit.EGL_VENDOR_VARIABLE] = saved_value
        kit._state()["probe"] = probe


def test_choose_tier_rules():
    saved_env = os.environ.pop(kit.ENV_TIER, None)
    saved = kit.isaacsim_install, kit.probe_graphics
    try:
        assert raises(ValueError, kit.choose_tier, "maximum")
        for tier in kit.TIERS:                                            # an argument is obeyed, whatever the machine
            choice = kit.choose_tier(tier)
            assert (choice.tier, choice.source) == (tier, "argument")
        kit.isaacsim_install = lambda: {"found": True, "package": "/x/isaacsim", "layout": "pip"}
        kit.probe_graphics = lambda *a, **k: kit.GraphicsProbe(True, "an NVIDIA EGL device initialised")
        assert kit.choose_tier().tier == "full" and kit.choose_tier().source == "probe"
        kit.probe_graphics = lambda *a, **k: kit.GraphicsProbe(False, "no graphics device")
        choice = kit.choose_tier("auto")
        assert choice.tier == "usd" and choice.reason == "no graphics device" and choice.probe is not None
        os.environ[kit.ENV_TIER] = "core"                                 # the environment beats the probe ...
        assert (kit.choose_tier().tier, kit.choose_tier().source) == ("core", kit.ENV_TIER)
        assert kit.choose_tier("off").tier == "off"                       # ... and the argument beats the environment
        os.environ[kit.ENV_TIER] = "everything"
        assert raises(ValueError, kit.choose_tier)
        os.environ[kit.ENV_TIER] = "auto"
        assert kit.choose_tier().source == "probe"
        del os.environ[kit.ENV_TIER]
        kit.isaacsim_install = lambda: {"found": False, "package": "", "layout": ""}
        choice = kit.choose_tier()
        assert choice.tier == "off" and "isaacsim" in choice.reason      # no Isaac Sim: no Kit, and no probe needed
    finally:
        kit.isaacsim_install, kit.probe_graphics = saved
        os.environ.pop(kit.ENV_TIER, None)
        if saved_env is not None:
            os.environ[kit.ENV_TIER] = saved_env


def test_gpu_selection_sets_the_environment_only_where_there_is_a_choice():
    """gpu=N restricts CUDA through CUDA_VISIBLE_DEVICES, except on a machine with one GPU (DGX Spark), where
    RTX warns against the variable and there is nothing to restrict."""
    saved_env = {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", kit.ENV_GPU)}
    saved = kit.gpu_count, kit._cuda_user
    try:
        for name in saved_env:
            os.environ.pop(name, None)
        kit._cuda_user = lambda: ""
        assert kit.gpu_count() >= 0
        assert kit._select_gpu(None)[:2] == (None, 0) and "CUDA_VISIBLE_DEVICES" not in os.environ
        kit.gpu_count = lambda: 8                                         # the development host
        gpu, index, note = kit._select_gpu(3)
        assert (gpu, index) == (3, 0) and os.environ["CUDA_VISIBLE_DEVICES"] == "3" and os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        gpu, index, note = kit._select_gpu(0)                             # another GPU than the environment says: said in the note
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0" and "'3'" in note
        assert "already set" in kit._select_gpu(0)[2]
        del os.environ["CUDA_VISIBLE_DEVICES"]
        kit.gpu_count = lambda: 1                                         # a DGX Spark
        gpu, index, note = kit._select_gpu(0)
        assert (gpu, index) == (0, 0) and "CUDA_VISIBLE_DEVICES" not in os.environ and "one GPU" in note
        os.environ[kit.ENV_GPU] = "0"                                     # the same through NCNTM_KIT_GPU
        assert kit._select_gpu(None)[0] == 0 and "CUDA_VISIBLE_DEVICES" not in os.environ
        del os.environ[kit.ENV_GPU]
        kit.gpu_count = lambda: 8
        kit._cuda_user = lambda: "torch"                                  # CUDA already initialised: too late, and it says so
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gpu, index, note = kit._select_gpu(2)
        assert (gpu, index) == (2, 2) and "CUDA_VISIBLE_DEVICES" not in os.environ and "NOT restricted" in note
        assert any("already initialised by torch" in str(w.message) for w in caught)
        assert raises(ValueError, kit._select_gpu, -1)
    finally:
        kit.gpu_count, kit._cuda_user = saved
        for name, value in saved_env.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value


def test_renderer_complaint_picks_the_telling_line():
    """The [Error] lines of the stock application's log on the development host (no graphics device)."""
    lines = [
        "2026-11-23T12:30:38Z [8,043ms] [Error] [omni.rtx] The chosen activeGpu index 0 is higher than the available GPUs. An unsupported driver might also prevent it from enumerating your devices properly.",
        "2026-11-23T12:30:38Z [8,043ms] [Error] [omni.rtx] No device could be created. Some known system issues:",
        "2026-11-23T12:30:44Z [14,481ms] [Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU devices, including an attempt with compatibility mode.",
        "2026-11-23T12:30:45Z [14,930ms] [Error] [omni.kit.renderer.plugin] GPU Foundation is not initialized!",
        "2026-11-23T12:30:45Z [14,954ms] [Error] [omni.kit.imgui_renderer.plugin] _prepareDefaultWindow: invalid device attached to the default window!",
    ]
    assert kit.renderer_complaint(lines) == "[omni.rtx] No device could be created. Some known system issues:"
    assert kit.renderer_complaint(lines[:1]).startswith("[omni.rtx] The chosen activeGpu index 0")
    assert kit.renderer_complaint(["... [Error] [omni.usd] something else", "no error here"]) == ""
    assert kit.renderer_complaint([]) == ""
    status = kit.RendererStatus(False, "no Hydra render engine is attached to the USD context")
    assert str(status) == "RTX renderer: not available on this machine because no Hydra render engine is attached to the USD context"
    assert str(kit.RendererStatus(True, "engine rtx, test frame 64 x 64", ("rtx",))) == "RTX renderer: available (engine rtx, test frame 64 x 64)"


def test_experience_files_have_no_paths():
    assert set(kit.EXPERIENCE_TEXT) == {tier for tier, name in kit.EXPERIENCES.items() if name}
    for tier, name in kit.EXPERIENCES.items():
        if not name:
            continue
        text = kit.EXPERIENCE_TEXT[tier]
        code = [line.split("#", 1)[0] for line in text.splitlines()]
        assert not any("/home" in line or "${" in line or "folders" in line for line in code), name
        assert '"omni.usd"' in text and '"omni.usd.schema.usd_particle_field"' in text
        assert "registryEnabled = false" in text                           # never downloads extensions
        packaged = Path(kit.APPS_DIR) / name
        if packaged.is_file():                                             # the file, where it travelled with the package,
            assert packaged.read_text() == text, f"{packaged} and kit.EXPERIENCE_TEXT[{tier!r}] differ: change both"
    for arg in kit.PRIVACY_ARGS:
        assert arg.startswith("--/") and arg.endswith(("=false", "=true"))
    assert kit.excluded_extensions_arg() == '--/app/extensions/excluded=["omni.kit.telemetry"]'


def test_experience_file_is_written_where_the_package_has_none():
    """The repository ignores *.kit, so a clone has no apps/*.kit: start_kit then writes the experience from the
    text kept in kit.py into the Kit folder and starts Kit with that file."""
    root = out_dir() / "experience_root"
    saved = kit.APPS_DIR
    try:
        kit.APPS_DIR = str(out_dir() / "no_apps_here")
        for tier in ("usd", "core"):
            path = kit.experience_file(tier, str(root))
            assert path == str(root / "apps" / kit.EXPERIENCES[tier]) and Path(path).read_text() == kit.EXPERIENCE_TEXT[tier]
            written = os.stat(path).st_mtime_ns
            assert kit.experience_file(tier, str(root)) == path and os.stat(path).st_mtime_ns == written     # not rewritten
            Path(path).write_text("# an older text\n")
            assert Path(kit.experience_file(tier, str(root))).read_text() == kit.EXPERIENCE_TEXT[tier]      # brought up to date
        assert kit.experience_file("full", str(root)) == ""               # the stock application has its own
    finally:
        kit.APPS_DIR = saved
    for tier in ("usd", "core"):                                           # with the package's file there, that file is used
        packaged = Path(kit.APPS_DIR) / kit.EXPERIENCES[tier]
        if packaged.is_file():
            assert kit.experience_file(tier, str(root)) == str(packaged)


def test_every_start_carries_the_privacy_switches():
    """Whatever the tier and the arguments: the configuration start_kit hands to SimulationApp has the crash
    reporter and telemetry switches, and SimulationApp is constructed in ONE place of the repository."""
    import ast
    calls = {}
    for folder in ("ncntm", "scripts", "tests"):
        for path in sorted((REPO / folder).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue                                                   # another module mid-edit: not this test's business
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                    if name == "SimulationApp":
                        calls.setdefault(str(path.relative_to(REPO)), []).append(node.lineno)
    assert list(calls) == [os.path.join("ncntm", "viz", "kit.py")] and len(calls[os.path.join("ncntm", "viz", "kit.py")]) == 1, calls
    notebooks = REPO / "notebooks"
    for path in sorted(notebooks.rglob("*.ipynb")) if notebooks.is_dir() else []:
        if ".ipynb_checkpoints" in path.parts:
            continue
        try:
            cells = json.loads(path.read_text()).get("cells", [])
        except ValueError:
            continue
        for cell in cells:
            source = "".join(cell.get("source", [])) if cell.get("cell_type") == "code" else ""
            assert "SimulationApp(" not in source, f"{path} starts Isaac Sim itself: use ncntm.viz.kit.start_kit()"

    seen = []

    class Stop(Exception):
        pass

    def launch(tier, config, root, quiet):
        seen.append((tier, dict(config), os.environ.get("TMPDIR")))
        raise Stop()

    saved = (kit._launch, kit._refuse_if_impossible, kit._no_core_files, kit._select_gpu, kit.in_jupyter)
    saved_tmpdir, saved_state = os.environ.get("TMPDIR"), dict(kit._state())
    root = out_dir() / "switches_root"
    try:
        kit._launch, kit._refuse_if_impossible, kit._no_core_files = launch, (lambda tier: {}), (lambda *a, **k: None)
        kit._select_gpu, kit.in_jupyter = (lambda gpu: (gpu, 0, "stand-in")), (lambda: False)
        kit._state().update(session=None, tainted="")
        for tier, extra in (("usd", {}), ("core", {"quiet": False}), ("full", {}), ("full", {"livestream": True, "gpu": 0}),
                            ("usd", {"extra_args": ["--/app/something=1"]})):
            assert raises(Stop, kit.start_kit, tier, root=str(root), **extra)
            tier_seen, config, tmpdir = seen[-1]
            assert tier_seen == tier
            for arg in kit.PRIVACY_ARGS + (kit.excluded_extensions_arg(), kit.STDOUT_ARG):
                assert arg in config["extra_args"], (tier, arg)
            assert config["enable_crashreporter"] is False and config["headless"] is True and config["multi_gpu"] is False
            assert tmpdir == str(root / "tmp")                             # Kit's temporary files: its own folder while it starts ...
            assert os.environ.get("TMPDIR") == saved_tmpdir                # ... and the variable is put back afterwards
        assert seen[3][1]["active_gpu"] == 0 and seen[3][1]["hide_ui"] is False and "active_gpu" not in seen[2][1]
        assert seen[4][1]["extra_args"][-1] == "--/app/something=1"
    finally:
        kit._launch, kit._refuse_if_impossible, kit._no_core_files, kit._select_gpu, kit.in_jupyter = saved
        kit._state().clear()
        kit._state().update(saved_state)


def test_describe_says_why_once_and_how_to_override():
    """describe() for a tier that the graphics probe chose: the probe's sentence appears once (in the RTX line),
    with the way to try RTX all the same."""
    probe = kit.GraphicsProbe(False, "no NVIDIA EGL device: the 1 device found is software (Mesa)")
    session = kit.Session("off", kit.TierChoice("off", "probe", probe.reason, probe), gpu_note="stand-in")
    saved = kit.pxr_origin
    try:
        kit.pxr_origin = lambda load=None: kit.PxrOrigin(True, "26.08", "/somewhere/pxr", False)
        text = session.describe()
    finally:
        kit.pxr_origin = saved
    assert text.count(probe.reason) == 1, text
    lines = text.splitlines()
    assert lines[1].startswith("  tier chosen automatically: ") and kit.ENV_TIER in lines[1] and "start_kit(tier=" in lines[1]
    rtx = [line for line in lines if line.strip().startswith("RTX renderer: ")]
    assert len(rtx) == 1 and "not available on this machine because " in rtx[0] and probe.reason in rtx[0]
    assert "start_kit(tier='full')" in rtx[0] and f"{kit.ENV_TIER}=full" in rtx[0]
    found = kit.GraphicsProbe(True, "an NVIDIA EGL device initialised (EGL 1.5, 1 NVIDIA device found)", devices=1, nvidia_devices=1, initialised=1)
    session = kit.Session("off", kit.TierChoice("off", "argument", "start_kit(tier='off') was asked for", found), gpu_note="stand-in")
    assert "start_kit(tier='full') in a new kernel should give RTX" in session.renderer_status().reason


def test_importing_kit_loads_nothing_heavy():
    """``from ncntm.viz import kit`` is the first line of every notebook: it must not load pxr or warp (which have
    to come after start_kit), nor Isaac Sim, nor torch."""
    code = ("import sys\nsys.path.insert(0, sys.argv[1])\n"
            "try:\n    from ncntm.viz import kit\nexcept ImportError as error:\n    print('NO PACKAGE', error); raise SystemExit(0)\n"
            "import ncntm.viz\n"
            "heavy = sorted({name.split('.')[0] for name in sys.modules} & {'pxr', 'warp', 'isaacsim', 'omni', 'carb', 'torch'})\n"
            "print('HEAVY', heavy, kit._pxr_loaded(), kit._loaded('warp'))\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-c", code, str(REPO)], capture_output=True, text=True, env=env, timeout=120)
    if "NO PACKAGE" in done.stdout:
        skip(f"the package ncntm.viz cannot be imported by {sys.executable} ({done.stdout.strip()}): nothing to check")
    assert "HEAVY [] False False" in done.stdout, done.stdout + done.stderr


def test_only_kit_py_imports_isaac_sim():
    """Nothing else in ncntm.viz may import isaacsim, omni or carb."""
    offenders = []
    for path in sorted((REPO / "ncntm" / "viz").glob("*.py")):
        if path.name == "kit.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            words = line.strip().split()
            if len(words) >= 2 and words[0] in ("import", "from") and words[1].split(".")[0] in ("isaacsim", "omni", "carb"):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, offenders


def test_frame_conversion_copies_and_recognises_empty_answers():
    """rgba_from_annotator: everything Replicator's rgb annotator can hand back (read in its 1.13.38 source)."""
    try:
        import numpy as np
    except ImportError as error:
        skip(f"numpy is not installed in {sys.executable}: {error}")
    import warnings
    picture = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    frame = kit.rgba_from_annotator(picture, 3, 2)
    assert frame.shape == (2, 3, 4) and frame.dtype == np.uint8 and (frame == picture).all()
    picture[:] = 0
    assert frame.max() == 23                                              # a copy: it survives the annotator's buffer
    for nothing in (None, {}, {"info": {}}, np.zeros((0,), np.uint8), np.zeros((0, 0, 4), np.uint8), {"data": None}):
        assert kit.rgba_from_annotator(nothing, 3, 2).size == 0, nothing  # no renderer: an empty answer, never an exception
    wrapped = kit.rgba_from_annotator({"data": np.full((2, 3, 4), 7, np.uint8), "info": {"x": 1}}, 3, 2)
    assert wrapped.shape == (2, 3, 4) and int(wrapped.min()) == 7         # the legacy structure with "info"
    assert kit.rgba_from_annotator(np.full(24, 9, np.uint8), 3, 2).shape == (2, 3, 4)       # flat: reshaped

    class OnGpu:                                                          # a Warp array: has .numpy()
        def numpy(self):
            return np.full((2, 3, 4), 5, np.uint8)
    assert int(kit.rgba_from_annotator(OnGpu(), 3, 2).max()) == 5
    with warnings.catch_warnings(record=True) as caught:                  # another size than asked for: returned, with a warning
        warnings.simplefilter("always")
        other = kit.rgba_from_annotator(np.zeros((4, 6, 4), np.uint8), 3, 2)
    assert other.shape == (4, 6, 4) and any("asked" in str(w.message) for w in caught)
    assert raises(RuntimeError, kit.rgba_from_annotator, np.zeros((5, 7), np.uint8), 3, 2)  # not a picture at all


def test_licence_markers_follow_the_layout():
    """Where Isaac Sim itself records the accepted licence: pip install, source build (python_packages is a
    LINK into <repo>/source there), binary build.  Run in a child process whose sys.path holds a stand-in."""
    repo, release = source_build_tree("licence_source_build", sys.executable, eula=False)
    binary = out_dir() / "licence_binary_build" / "isaac-sim"
    (binary / "python_packages" / "isaacsim").mkdir(parents=True, exist_ok=True)
    (binary / "python_packages" / "isaacsim" / "__init__.py").write_text("")
    pip = out_dir() / "licence_pip" / "site-packages"
    (pip / "isaacsim" / "kit").mkdir(parents=True, exist_ok=True)
    (pip / "isaacsim" / "__init__.py").write_text("")
    code = ("import json\n" + LOAD_KIT_BY_PATH +
            "print('RESULT ' + json.dumps({'install': kit.isaacsim_install(), 'status': list(kit.licence_status()), "
            "'markers': kit.licence_markers()}))")

    def ask(python_path, **extra):
        env = {k: v for k, v in os.environ.items() if k not in (kit.EULA_VARIABLE, "ISAAC_PATH", "PYTHONPATH")}
        env.update(PYTHONPATH=str(python_path), PYTHONDONTWRITEBYTECODE="1", **extra)
        done = subprocess.run([sys.executable, "-c", code, str(REPO)], capture_output=True, text=True, env=env, timeout=120)
        lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
        assert lines, done.stdout + done.stderr
        return json.loads(lines[-1][len("RESULT "):])

    marker = str(repo / ".eula_accepted")
    for extra in ({}, {"ISAAC_PATH": str(release)}):                      # with and without what python.sh exports
        answer = ask(release / "python_packages", **extra)
        assert answer["install"]["layout"] == "build"
        assert answer["install"]["package"] == str(repo / "source" / "python_packages" / "isaacsim")      # resolved
        assert answer["install"]["found_at"] == str(release / "python_packages" / "isaacsim")           # as on sys.path
        assert answer["markers"] == [marker], answer["markers"]
        assert answer["status"][0] is False                               # build.sh has not been run: not accepted
    (repo / ".eula_accepted").touch()                                     # what tools/eula_check.sh writes: an empty file
    answer = ask(release / "python_packages", ISAAC_PATH=str(release))
    assert answer["status"] == [True, f"marker file {marker}"]
    assert ask(release / "python_packages", **{kit.EULA_VARIABLE: "YES"})["status"][1] == f"{kit.EULA_VARIABLE} is set"

    answer = ask(binary / "python_packages", ISAAC_PATH=str(binary))      # a binary build keeps no record: the variable decides
    assert answer["install"]["layout"] == "build" and answer["markers"] == [] and answer["status"][0] is False

    answer = ask(pip)
    assert answer["install"]["layout"] == "pip" and answer["markers"] == [str(pip / "isaacsim" / "kit" / "EULA_ACCEPTED")]
    assert answer["status"][0] is False
    (pip / "isaacsim" / "kit" / "EULA_ACCEPTED").write_text("no\n")       # kit_app.check_eula reads the first line
    assert ask(pip)["status"][0] is False
    (pip / "isaacsim" / "kit" / "EULA_ACCEPTED").write_text("yes")
    assert ask(pip)["status"][0] is True


def test_in_jupyter_means_this_process_is_a_kernel():
    """A script started from a notebook inherits JPY_PARENT_PID; it is still a script (exit status, no asyncio patch)."""
    saved = os.environ.get("JPY_PARENT_PID")
    os.environ["JPY_PARENT_PID"] = str(os.getpid())
    try:
        assert kit.in_jupyter() is False
    finally:
        os.environ.pop("JPY_PARENT_PID", None)
        if saved is not None:
            os.environ["JPY_PARENT_PID"] = saved
    saved = os.environ.get(kit.ENV_PROTECT)
    os.environ[kit.ENV_PROTECT] = "0"                                      # the switch that leaves asyncio alone
    try:
        from asyncio import events
        before = events.Handle._run
        assert kit.protect_queued_cells() is False and events.Handle._run is before
    finally:
        os.environ.pop(kit.ENV_PROTECT, None)
        if saved is not None:
            os.environ[kit.ENV_PROTECT] = saved


# ---- a real kernel -------------------------------------------------------------------------------
PRELUDE = "import json, os, sys, time\nsys.path.insert(0, os.environ['NCNTM_TEST_REPO'])\nfrom ncntm.viz import kit\n"
# For the kernels of these tests in which Kit is NOT started (start_kit does this itself when it starts Kit):
# ipykernel 7.4.0 now and then aborts inside libzmq while it shuts down (seen once in 12 such kernels,
# SIGABRT in zmq_msg_send during zmq_ctx_term).  That is not ours to fix; at least leave no core file.
NO_CORE_FILE = "import resource\nresource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"

# Defined in every notebook: the same code builds and measures a stage with and without Kit.
STAGE_CODE = r'''
import numpy as np

def build(stage):
    """A box of 12 x 16 x 8 voxels under a "space" variant set and a Gaussian-splat prim written
    with the generic prim / attribute API (Kit's OpenUSD has no typed ParticleField class)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    T = Sdf.ValueTypeNames
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    data = UsdGeom.Xform.Define(stage, "/World/data")
    voxels = UsdGeom.Xform.Define(stage, "/World/data/voxels")
    scale = voxels.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)            # no value here: the variants hold it
    box = UsdGeom.Mesh.Define(stage, "/World/data/voxels/box")
    corners = [(x, y, z) for z in (0, 8) for y in (0, 16) for x in (0, 12)]
    box.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    box.CreateFaceVertexCountsAttr([4] * 6)
    box.CreateFaceVertexIndicesAttr([0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 1, 3, 7, 5, 0, 4, 6, 2])
    box.CreateExtentAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(12, 16, 8)])
    variants = data.GetPrim().GetVariantSets().AddVariantSet("space")
    for name, size in (("physical", (6.0, 6.0, 30.0)), ("index", (6.0, 6.0, 6.0))):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        with variants.GetVariantEditContext():
            scale.Set(Gf.Vec3d(*size))
    variants.SetVariantSelection("physical")
    splats = stage.DefinePrim("/World/data/splats", "ParticleField3DGaussianSplat")
    means = np.array([[30.0, 40.0, 100.0], [50.0, 60.0, 150.0], [10.0, 80.0, 200.0]], dtype=np.float32)
    sigma = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [3.0, 3.0, 10.0]], dtype=np.float32)
    splats.CreateAttribute("positions", T.Point3fArray).Set(Vt.Vec3fArray.FromNumpy(means))
    splats.CreateAttribute("scales", T.Float3Array).Set(Vt.Vec3fArray.FromNumpy(sigma))
    splats.CreateAttribute("orientations", T.QuatfArray).Set(Vt.QuatfArray([Gf.Quatf(1.0, 0.0, 0.0, 0.0)] * 3))
    splats.CreateAttribute("opacities", T.FloatArray).Set(Vt.FloatArray([0.25, 0.5, 0.75]))
    splats.CreateAttribute("radiance:sphericalHarmonicsDegree", T.Int, False, Sdf.VariabilityUniform).Set(0)
    splats.CreateAttribute("radiance:sphericalHarmonicsCoefficients", T.Float3Array).Set(
        Vt.Vec3fArray([Gf.Vec3f(0.5, 0.0, -0.5)] * 3))
    splats.CreateAttribute("extent", T.Float3Array).Set(
        Vt.Vec3fArray([Gf.Vec3f(*(means - 3 * sigma).min(0).tolist()), Gf.Vec3f(*(means + 3 * sigma).max(0).tolist())]))
    return stage

def measure(stage, root="/World"):
    """Everything composition decides, as plain numbers and strings: prims and types, the world
    box of every prim with a variant set under each variant, array attributes, resolved textures."""
    from pxr import Usd, UsdGeom
    names = ("xformOp:scale", "points", "positions", "scales", "orientations", "opacities", "extent",
             "radiance:sphericalHarmonicsCoefficients", "em:gaussian:covariances", "em:gaussian:means")
    def arrays(top):
        found = {}
        for prim in Usd.PrimRange(top):
            for attribute in prim.GetAuthoredAttributes():
                value = attribute.Get()
                if attribute.GetName() in names and value is not None:
                    a = np.array(value, dtype=np.float64)
                    found[str(attribute.GetPath())] = [list(a.shape), round(float(np.abs(a).sum()), 3), round(float(a.sum()), 3)]
        return found
    top = stage.GetPrimAtPath(root)
    out = {"upAxis": str(UsdGeom.GetStageUpAxis(stage)), "metersPerUnit": float(UsdGeom.GetStageMetersPerUnit(stage)),
           "defaultPrim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else "",
           "prims": [[str(p.GetPath()), str(p.GetTypeName())] for p in Usd.PrimRange(top)], "variants": {}, "textures": []}
    for prim in list(Usd.PrimRange(top)):
        for set_name in prim.GetVariantSets().GetNames():
            variant_set = prim.GetVariantSets().GetVariantSet(set_name)
            per = {}
            for variant in sorted(variant_set.GetVariantNames()):
                with Usd.EditContext(stage, stage.GetSessionLayer()):     # switch without touching the file's layer
                    variant_set.SetVariantSelection(variant)
                cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                per[variant] = {"min": [round(v, 6) for v in box.GetMin()], "max": [round(v, 6) for v in box.GetMax()],
                                "arrays": arrays(prim)}
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                variant_set.ClearVariantSelection()
            out["variants"][f"{prim.GetPath()}:{set_name}"] = {"selection": variant_set.GetVariantSelection(), "per_variant": per}
    for prim in Usd.PrimRange(top):
        attribute = prim.GetAttribute("inputs:file")
        asset = attribute.Get() if attribute else None
        if asset is not None:
            resolved = asset.resolvedPath
            out["textures"].append([asset.path, os.path.basename(resolved), os.path.isfile(resolved),
                                    os.path.getsize(resolved) if os.path.isfile(resolved) else -1])
    out["splat_prims"] = [str(p.GetPath()) for p in Usd.PrimRange(top) if p.GetTypeName() == "ParticleField3DGaussianSplat"]
    out["splat_is_boundable"] = [bool(stage.GetPrimAtPath(p).IsA(UsdGeom.Boundable)) for p in out["splat_prims"]]
    return out
'''

# A dataset authored with ncntm.viz.stage outside Kit: slices with textures next to the file,
# Gaussians as splats, the "space" variant set.
AUTHOR_CODE = r'''
from ncntm.viz import stage as S
from ncntm.viz.grid import VoxelGrid
from ncntm.viz.types import Gaussians
grid = VoxelGrid((8, 16, 12), (30.0, 6.0, 6.0))
rng = np.random.default_rng(0)
raw = rng.integers(0, 255, grid.shape_zyx, dtype=np.uint8)
instance = np.zeros(grid.shape_zyx, np.uint32); instance[2:6, 4:12, 3:9] = 7
authored = os.path.join(os.environ["NCNTM_TEST_OUT_DIR"], "authored", "demo.usda")
st = S.new_stage(authored)
dataset = S.Dataset(st, "demo", grid, raw=raw, instance=instance)
dataset.add_slices("rawvalue", z=3); dataset.add_slices("instance", y=5)
dataset.add_gaussians(Gaussians(means_xyz_nm=np.array([[30.0, 40.0, 100.0], [50.0, 60.0, 150.0]]),
                                covariances_nm2=np.array([np.diag([100.0, 400.0, 900.0]), np.diag([50.0, 60.0, 70.0])]),
                                weights=np.array([0.5, 0.5]), instance_id=7))
S.save_stage(st)
del st, dataset
'''


def need_openusd_without_kit(off):
    start = result(off.start)
    if not start["usd"]:
        skip("pxr cannot be imported without Kit by this interpreter (no usd-core / usd-exchange installed): "
             "tier 'off' has no OpenUSD here")
    return start


def result(cell):
    """The object a cell printed as 'RESULT <json>'."""
    assert cell.error is None, f"the cell failed: {cell.error}\n{cell.text[-2000:]}"
    lines = [line for line in cell.text.splitlines() if line.startswith("RESULT ")]
    assert lines, f"the cell printed no RESULT line:\n{cell.text[-2000:]}"
    return json.loads(lines[-1][len("RESULT "):])


@once
def setup():
    """Find Isaac Sim, install the kernel spec into a temporary prefix, prepare the environment
    of the kernels.  Skips (for every kernel test) when one of the ingredients is missing."""
    try:
        import jupyter_client  # noqa: F401
    except ImportError as error:
        skip(f"jupyter_client is not installed in {sys.executable}: {error}")
    installer = installer_module()
    try:
        path, kind, how, probe = installer.find_isaac_python(os.environ.get("NCNTM_ISAAC_PYTHON"))
    except SystemExit:
        skip("Isaac Sim is not installed here ('isaacsim' is not importable by any interpreter that "
             "scripts/install_kernel.py could find; NCNTM_ISAAC_PYTHON names one): the Kit tests cannot run")
    if not probe.get("isaacsim"):
        skip(f"'isaacsim' is not importable by {path}: the Kit tests cannot run")
    if not probe.get("ipykernel"):
        skip(f"ipykernel is not importable by {path}: {path} -m pip install ipykernel")
    if os.environ.get(kit.EULA_VARIABLE, "N").lower() not in ("y", "yes", "1"):
        skip(f"{kit.EULA_VARIABLE} is not set: the tests do not accept the NVIDIA Omniverse licence for you; "
             f"read {kit.EULA_URL} and run them with {kit.EULA_VARIABLE}=YES")
    prefix = out_dir() / "prefix"
    code, text = run_installer("--prefix", prefix, "--python", path)       # NO --accept-eula: the kernels inherit the variable
    assert code == 0, text
    assert kit.EULA_VARIABLE not in spec_of(prefix)["env"]
    jupyter = out_dir() / "jupyter"
    env = dict(os.environ)
    for name in ("runtime", "config", "data", "ipython", "mpl"):
        (jupyter / name).mkdir(parents=True, exist_ok=True)
    env.update(JUPYTER_PATH=str(prefix / "share" / "jupyter"), JUPYTER_RUNTIME_DIR=str(jupyter / "runtime"),
               JUPYTER_CONFIG_DIR=str(jupyter / "config"), JUPYTER_DATA_DIR=str(jupyter / "data"),
               IPYTHONDIR=str(jupyter / "ipython"), MPLCONFIGDIR=str(jupyter / "mpl"),
               NCNTM_KIT_ROOT=str(out_dir() / "kit_root"), NCNTM_TEST_REPO=str(REPO), NCNTM_TEST_OUT_DIR=str(out_dir()))
    for name in (kit.ENV_TIER, kit.ENV_GPU):
        env.pop(name, None)
    for name in ("JUPYTER_PATH", "JUPYTER_RUNTIME_DIR", "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR"):
        os.environ[name] = env[name]                                       # jupyter_client in this process looks here
    return SimpleNamespace(python=path, kind=kind, how=how, probe=probe, prefix=prefix, env=env,
                           root=out_dir() / "kit_root", installer=installer)


_LIVE = []                     # process groups of kernels that were started and not yet shut down


def _kill_leftovers():
    for pgid in list(_LIVE):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass


atexit.register(_kill_leftovers)


class Kernel:
    """One kernel = one notebook.  ``run`` executes a cell and waits for it, like nbclient."""

    def __init__(self, label, name=KERNEL_NAME, env=None, drop=()):
        base = dict(setup().env)                                           # skips when a kernel cannot be had here
        from jupyter_client.manager import KernelManager
        base.update(env or {})
        for key in drop:
            base.pop(key, None)
        self.label = label
        # everything the kernel leaves behind stays in the test folder: its connection file (jupyter_client would
        # put it into the system's temporary folder) and what it writes to its own stdout / stderr
        slug = "".join(c if c.isalnum() else "_" for c in label)
        number = len(TIMINGS)
        (out_dir() / "kernel_logs").mkdir(exist_ok=True)
        self.log = out_dir() / "kernel_logs" / f"{number:02d}_{slug}.log"
        connection = setup().env["JUPYTER_RUNTIME_DIR"] + f"/kernel-{number:02d}-{slug}.json"
        self.manager = KernelManager(kernel_name=name, connection_file=connection)
        start = time.perf_counter()
        with open(self.log, "w") as log:
            self.manager.start_kernel(env=base, stdout=log, stderr=subprocess.STDOUT)
        self.process = self.manager.provisioner.process
        self.pgid = os.getpgid(self.process.pid)
        _LIVE.append(self.pgid)
        self.client = self.manager.client()
        self.client.start_channels()
        self.client.wait_for_ready(timeout=120)
        self.timing = {"kernel": label, "tier": "", "kernel_ready_s": round(time.perf_counter() - start, 2),
                       "start_kit_s": None, "shutdown_s": None, "exit_code": None}
        TIMINGS.append(self.timing)

    def _collect(self, ids, timeout, interrupt_after=None):
        cells = {i: SimpleNamespace(text="", error=None, seconds=None, done=False) for i in ids}
        order, start, interrupted = [], time.time(), False
        while len(order) < len(ids):
            try:
                message = self.client.get_iopub_msg(timeout=0.2)
            except queue.Empty:
                if interrupt_after is not None and not interrupted and time.time() - start > interrupt_after:
                    self.manager.interrupt_kernel()
                    interrupted = True
                if time.time() - start > timeout or not self.manager.is_alive():
                    break
                continue
            cell = cells.get(message["parent_header"].get("msg_id"))
            if cell is None:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "stream":
                cell.text += content["text"]
            elif kind == "error":
                cell.error = f"{content['ename']}: {content['evalue']}"
            elif kind == "status" and content["execution_state"] == "idle":
                cell.done, cell.seconds = True, round(time.time() - start, 2)
                order.append(ids.index(message["parent_header"]["msg_id"]))
        for cell in cells.values():
            if not cell.done:
                cell.error = "KERNEL DIED" if not self.manager.is_alive() else f"NO ANSWER within {timeout} s"
        return [cells[i] for i in ids], order

    def run(self, code, timeout=180, interrupt_after=None):
        cells, _ = self._collect([self.client.execute(code, allow_stdin=False)], timeout, interrupt_after)
        return cells[0]

    def run_all(self, codes, timeout=180):
        """Send every cell at once (what JupyterLab's "Run All" does); returns (cells, order of completion)."""
        return self._collect([self.client.execute(code, allow_stdin=False) for code in codes], timeout)

    def shutdown(self):
        start = time.perf_counter()
        try:
            self.client.stop_channels()
            self.manager.shutdown_kernel(now=False)                        # shutdown_request, then SIGTERM / SIGKILL after the grace period
        finally:
            self.timing["shutdown_s"] = round(time.perf_counter() - start, 2)
            self.timing["exit_code"] = self.process.poll()
            try:
                os.killpg(self.pgid, 0)
                self.timing["left_running"] = True
                os.killpg(self.pgid, signal.SIGKILL)
            except OSError:
                self.timing["left_running"] = False
            if self.pgid in _LIVE:
                _LIVE.remove(self.pgid)
        return self.timing


def start_cell(tier, extra=""):
    return PRELUDE + f"s = kit.start_kit({tier!r}, gpu={GPU}{extra})\n"


@once
def notebook_off():
    """Tier "off" in the Isaac kernel: author a dataset with ncntm.viz.stage, build the generic
    stage, measure both through the session."""
    kernel = Kernel("off", env={kit.ENV_TIER: "off"})                      # the tier comes from NCNTM_KIT, not from an argument
    r = SimpleNamespace()
    try:
        kernel.timing["tier"] = "off"
        r.start = kernel.run("import resource\ncore_before = list(resource.getrlimit(resource.RLIMIT_CORE))\n"
                             + PRELUDE + f"s = kit.start_kit(gpu={GPU})\n" + "import subprocess\n"
                             "child = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, sys.argv[1]); "
                             "from ncntm.viz import kit; print(kit.in_jupyter())', os.environ['NCNTM_TEST_REPO']], "
                             "capture_output=True, text=True, timeout=120).stdout.strip()\n"
                             "print('RESULT ' + json.dumps({'tier': s.tier, 'app': s.app is None, 'source': s.choice.source, "
                             "'usd': s.usd_version, 'kit_usd': kit.pxr_origin().is_kit, 'isaacsim': 'isaacsim' in sys.modules, "
                             "'available': s.renderer_available, 'describe': s.describe(), 'in_jupyter': kit.in_jupyter(), "
                             "'child_in_jupyter': child, 'child_has_parent_pid': bool(os.environ.get('JPY_PARENT_PID')), "
                             "'core_before': core_before, 'core_after': list(resource.getrlimit(resource.RLIMIT_CORE))}))")
        r.generic = kernel.run(STAGE_CODE + "stage = build(s.new_stage()); s.update(2)\nprint('RESULT ' + json.dumps(measure(stage)))")
        r.saved = kernel.run("path = s.save(os.path.join(os.environ['NCNTM_TEST_OUT_DIR'], 'off_generic.usda'))\n"
                             "print('RESULT ' + json.dumps(measure(s.open_stage(path))))")
        r.author = kernel.run(AUTHOR_CODE + "print('RESULT ' + json.dumps({'file': authored, 'schema': S.splat_schema_available()}))")
        r.authored = kernel.run("print('RESULT ' + json.dumps(measure(s.open_stage(authored))))")
        r.capture = kernel.run("try:\n    s.capture_rtx('/World/Camera', 64, 64)\n    print('RESULT ' + json.dumps('captured'))\n"
                               "except RuntimeError as error:\n    print('RESULT ' + json.dumps(str(error)))")
        r.close = kernel.run("s.close(); s.update(3)\nprint('RESULT ' + json.dumps(s.stage is not None))")
    finally:
        r.timing = kernel.shutdown()
    return r


@once
def notebook_usd():
    """The main notebook: tier "usd" from the first cell to the kernel shutdown."""
    off = notebook_off()                                                   # writes the file this notebook opens
    out = out_dir()
    for name in ("unflushed.txt", "open.h5", "logging.txt"):
        if (out / name).exists():
            (out / name).unlink()
    kernel = Kernel("usd")
    r = SimpleNamespace(off=off)
    try:
        kernel.timing["tier"] = "usd"
        first = (PRELUDE + "import signal, tempfile\n"
                 "if 'before' not in globals():\n"
                 "    before = {'argv': list(sys.argv), 'stdout': id(sys.stdout), 'tmp': tempfile.gettempdir(),\n"
                 "              'sigint': repr(signal.getsignal(signal.SIGINT)), 'origin': str(kit.pxr_origin()),\n"
                 "              'pxr_loaded': 'pxr' in sys.modules, 'session': kit.current_session() is None,\n"
                 "              'tmpdir': os.environ.get('TMPDIR')}\n"
                 "    import resource\n"
                 "    before['core_limit'] = list(resource.getrlimit(resource.RLIMIT_CORE))\n"
                 f"s = kit.start_kit('usd', gpu={GPU})\n"
                 "print(s.describe())\n")
        r.first = kernel.run(first)
        r.facts = kernel.run(
            "import glob, resource\n"
            "print('printed in a later cell')\n"
            "print('RESULT ' + json.dumps({'tier': s.tier, 'start_seconds': s.start_seconds, 'errors': s.kit_errors(),\n"
            "    'extensions': s.extensions, 'usd': s.usd_version, 'kit_usd': kit.pxr_origin().is_kit, 'warp': s.warp_version,\n"
            "    'device': s.device, 'args': s.launch_args, 'args_from_kit': s.launch_args_from_kit, 'privacy': s.privacy(),\n"
            "    'root': s.root, 'log': s.log_file, 'tmpdir': os.environ.get('TMPDIR'), 'python_tmp': tempfile.gettempdir(),\n"
            "    'root_tmp': sorted(os.listdir(os.path.join(s.root, 'tmp'))),\n"
            "    'crash_stubs': glob.glob('/var/tmp/st_*.' + str(os.getpid())),\n"
            "    'before': before, 'argv': list(sys.argv), 'stdout': id(sys.stdout), 'sigint': repr(signal.getsignal(signal.SIGINT)),\n"
            "    'core_limit': list(resource.getrlimit(resource.RLIMIT_CORE)), 'visible': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
            "    'jupyter_flag': os.environ.get('ISAAC_JUPYTER_KERNEL'), 'session_id': id(s), 'pid': os.getpid(),\n"
            "    'start_output': open(s.start_output_file).read() if s.start_output_file else None,\n"
            "    'protected': kit.protect_queued_cells()}))")
        r.build = kernel.run(STAGE_CODE + "stage = build(s.new_stage()); s.update(2)\nm_built = measure(stage)\n"
                             "print('RESULT ' + json.dumps(m_built))")
        r.rerun = kernel.run(first + "print('RESULT ' + json.dumps({'session_id': id(s), 'same': s is kit.current_session(),\n"
                             "    'prims': [str(p.GetPath()) for p in s.stage.Traverse()], 'start_seconds': s.start_seconds,\n"
                             "    'other_tier': kit.start_kit('full') is s, 'auto': kit.start_kit() is s}))")
        r.roundtrip = kernel.run(
            "path = s.save(os.path.join(os.environ['NCNTM_TEST_OUT_DIR'], 'kit_generic.usda'))\n"
            "old = s.stage\n"
            "stage = s.open_stage(path)\n"
            "print('RESULT ' + json.dumps({'measure': measure(stage), 'file': path, 'real_path': s.stage.GetRootLayer().realPath,\n"
            "    'old_handle_alive': bool(old), 'errors': len(s.kit_errors()),\n"
            "    'flattened': 'variantSet \"space\"' not in open(path).read()}))")
        r.authored = kernel.run("stage = s.open_stage(os.path.join(os.environ['NCNTM_TEST_OUT_DIR'], 'authored', 'demo.usda'))\n"
                                "print('RESULT ' + json.dumps({'measure': measure(stage), 'errors': s.kit_errors()}))")
        r.warp = kernel.run(
            "import subprocess\n"
            "try:\n"
            "    import warp as wp\n"
            "except ImportError as error:\n"
            "    print('RESULT ' + json.dumps({'warp': None, 'why': str(error)}))\n"
            "else:\n"
            "    wp.config.quiet = True\n"
            "    @wp.kernel\n"
            "    def saxpy(a: float, x: wp.array(dtype=float), y: wp.array(dtype=float)):\n"
            "        i = wp.tid()\n"
            "        y[i] = a * x[i] + y[i]\n"
            "    x = wp.array(np.arange(1000, dtype=np.float32), device=s.device)\n"
            "    y = wp.array(np.ones(1000, dtype=np.float32), device=s.device)\n"
            "    wp.launch(saxpy, dim=1000, inputs=[2.0, x, y], device=s.device)\n"
            "    total = float(y.numpy().sum())\n"
            "    s.update(3)\n"
            "    held = None\n"
            "    try:\n"
            "        gpus = dict(line.split(', ') for line in subprocess.run(['nvidia-smi', '--query-gpu=uuid,index', '--format=csv,noheader'],\n"
            "                    capture_output=True, text=True, timeout=60).stdout.strip().splitlines())\n"
            "        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'],\n"
            "                              capture_output=True, text=True, timeout=60).stdout.strip().splitlines()\n"
            "        held = sorted(int(gpus[line.split(', ')[1]]) for line in apps if line.split(', ')[0] == str(os.getpid()))\n"
            "    except Exception as error:\n"
            "        held = None\n"
            "    print('RESULT ' + json.dumps({'warp': wp.__version__, 'total': total, 'device': str(x.device), 'is_cuda': bool(x.device.is_cuda),\n"
            "        'cuda_devices': [d.alias for d in wp.get_cuda_devices()], 'held_gpus': held, 'errors': len(s.kit_errors())}))")
        r.rtx = kernel.run(
            "t0 = time.perf_counter()\n"
            "available = s.renderer_available\n"
            "try:\n"
            "    s.capture_rtx('/World/Camera', 64, 64)\n"
            "    message = 'captured'\n"
            "except RuntimeError as error:\n"
            "    message = str(error)\n"
            "try:\n"
            "    s.start_livestream()\n"
            "    stream = 'started'\n"
            "except RuntimeError as error:\n"
            "    stream = str(error)\n"
            "print('RESULT ' + json.dumps({'available': available, 'message': message, 'stream': stream,\n"
            "    'seconds': time.perf_counter() - t0, 'status': str(s.renderer_status()), 'describe': s.describe()}))")
        r.pretend = kernel.run(
            "s.tier, s._renderer = 'full', None        # drive the code of tier 'full' in a Kit that has no renderer\n"
            "t0 = time.perf_counter()\n"
            "try:\n"
            "    status = s.renderer_status()\n"
            "    try:\n"
            "        s.capture_rtx('/World/Camera', 64, 64)\n"
            "        message = 'captured'\n"
            "    except RuntimeError as error:\n"
            "        message = str(error)\n"
            "finally:\n"
            "    s.tier, s._renderer = 'usd', None\n"
            "print('RESULT ' + json.dumps({'available': status.available, 'reason': status.reason, 'engines': list(status.engines),\n"
            "    'message': message, 'seconds': time.perf_counter() - t0, 'tier': s.tier}))")
        r.close = kernel.run("t0 = time.perf_counter()\ns.close(); s.app.close()\ns.update(2)\n"
                             "print('RESULT ' + json.dumps({'seconds': time.perf_counter() - t0, 'stage': s.stage is not None}))")
        r.files = kernel.run(
            "import logging\n"
            "D = os.environ['NCNTM_TEST_OUT_DIR']\n"
            "f_text = open(os.path.join(D, 'unflushed.txt'), 'w'); f_text.write('result line\\n' * 10)\n"
            "handler = logging.FileHandler(os.path.join(D, 'logging.txt')); logging.getLogger('ncntm.test').addHandler(handler)\n"
            "logging.getLogger('ncntm.test').warning('logged line')\n"
            "try:\n"
            "    import h5py\n"
            "    f_h5 = h5py.File(os.path.join(D, 'open.h5'), 'w'); f_h5.create_dataset('x', data=np.arange(1000))\n"
            "    have_h5 = True\n"
            "except ImportError:\n"
            "    have_h5 = False\n"
            "print('RESULT ' + json.dumps({'h5py': have_h5, 'bytes_on_disk_now': os.path.getsize(os.path.join(D, 'unflushed.txt'))}))")
        r.last = kernel.run("print('RESULT ' + json.dumps({'errors': s.kit_errors()}))")
    finally:
        r.timing = kernel.shutdown()
    return r


# ---- tier usd -------------------------------------------------------------------------------------
def test_usd_tier_starts_clean_and_fast():
    r = notebook_usd()
    assert r.first.error is None, r.first.error
    facts = result(r.facts)
    assert facts["tier"] == "usd"
    assert facts["errors"] == [], facts["errors"]                          # 0 [Error] lines in Kit's log
    assert result(r.last)["errors"] == []                                  # ... and still none at the end of the notebook
    assert 0.0 < facts["start_seconds"] < 10.0, facts["start_seconds"]    # measured 0.3 - 0.6 s
    assert facts["extensions"] > 5
    assert facts["kit_usd"] and facts["usd"] != result(r.off.start)["usd"]         # Kit's own OpenUSD, not the interpreter's
    assert facts["jupyter_flag"] == "1" and facts["protected"] is True
    assert facts["core_limit"] == [0, 0]                                   # a teardown crash leaves no core file
    print(f"RLIMIT_CORE before start_kit {facts['before']['core_limit']}, after {facts['core_limit']}")
    # the first cell shows what the notebook printed and nothing of Kit's start-up chatter
    lines = r.first.text.strip().splitlines()
    assert lines[0].startswith("Isaac Sim: Kit ") and "tier 'usd'" in lines[0], r.first.text
    assert not any("[Info]" in line or "[Warning]" in line or "Starting kit application" in line for line in lines)
    assert "RTX renderer: not available on this machine because" in r.first.text
    assert "crash reporter and telemetry: off" in r.first.text
    # the process is as the notebook left it: argv, stdout, the SIGINT handler
    assert facts["argv"] == facts["before"]["argv"] and facts["stdout"] == facts["before"]["stdout"]
    assert facts["sigint"] == facts["before"]["sigint"]
    # asking which OpenUSD is in use before Kit runs must not import pxr (that would make the start impossible)
    assert facts["before"]["origin"] == "OpenUSD: pxr is not imported yet" and facts["before"]["pxr_loaded"] is False
    assert facts["before"]["session"] is True


def test_print_works_in_later_cells():
    r = notebook_usd()
    assert "printed in a later cell" in r.facts.text                       # Kit did not take sys.stdout away
    assert kit.STDOUT_ARG in result(r.facts)["args"]
    for cell in (r.build, r.rerun, r.roundtrip, r.warp, r.rtx, r.pretend, r.close, r.files, r.last):
        assert cell.error is None and "RESULT " in cell.text


def test_crash_reporter_and_telemetry_are_off():
    facts = result(notebook_usd().facts)
    assert facts["args_from_kit"]                                          # the list SimulationApp handed to Kit, not our wish
    for arg in kit.PRIVACY_ARGS:
        assert arg in facts["args"], arg
    assert kit.excluded_extensions_arg() in facts["args"]
    assert not any(facts["privacy"].values()), facts["privacy"]            # read back from the running application
    assert set(facts["privacy"]) >= {"crashreporter_enabled", "telemetry_anonymous_data", "structured_log",
                                     "telemetry_extension_enabled", "crashreporter_plugin_requested"}
    assert uploads_in(setup().root) == []                                  # no transmitter log, no crash dump, no upload line


def uploads_in(root):
    """Traces of data leaving the machine under a Kit root: the telemetry transmitter's log, crash dumps, and
    log lines of the crash reporter or of an upload (what the prototype found after stock starts)."""
    found = []
    for folder, _, files in os.walk(root, onerror=lambda error: None):
        for name in files:
            path = os.path.join(folder, name)
            if "transmitter" in name.lower() or name.endswith((".dmp", ".dmp.zip")):
                found.append(path)
            elif name.startswith("kit_") and name.endswith(".log"):
                with open(path, errors="replace") as f:
                    for line in f:
                        low = line.lower()
                        if "omni.kit.app.plugin] cmd:" in low:               # the command line names the switches themselves
                            continue
                        if "uploaded" in low or "[carb.crashreporter" in low or "telemetry.transmitter" in low:
                            found.append(f"{path}: {line.strip()[:160]}")
    return found


def test_kit_writes_into_its_root_only():
    facts = result(notebook_usd().facts)
    root = str(setup().root)
    args = facts["args"]
    assert "--portable-root" in args and args[args.index("--portable-root") + 1] == root
    assert "--portable" not in args                                        # that would mean: write into the installation
    assert facts["root"] == root and facts["log"].startswith(root) and os.path.isfile(facts["log"])
    assert any(name.startswith("carb.") for name in facts["root_tmp"]), facts["root_tmp"]     # Kit's native temporary files are in its folder
    assert facts["crash_stubs"] == []                                      # ... and OpenUSD's crash file is not in /var/tmp
    assert facts["tmpdir"] == facts["before"]["tmpdir"]                    # TMPDIR itself is as it was (a long one breaks Warp's compiler)
    assert facts["python_tmp"] == facts["before"]["tmp"]                   # Python's tempfile is left alone
    assert facts["start_output"] is not None                               # Kit's console output of the start, kept in a file
    assert not any("-f" == a or a.endswith(".json") for a in args)         # the kernel's own argv did not reach Kit


def test_rerunning_the_first_cell_keeps_the_stage():
    r = notebook_usd()
    facts, again = result(r.facts), result(r.rerun)
    assert again["session_id"] == facts["session_id"] and again["same"]
    assert again["start_seconds"] == facts["start_seconds"]                # no second start
    assert "/World/data/voxels/box" in again["prims"] and "/World/data/splats" in again["prims"]
    assert again["other_tier"] and again["auto"]                           # asking for another tier returns the running session
    assert "needs a kernel restart" in r.rerun.text


def test_variant_set_and_splat_prim_round_trip_through_kit():
    r = notebook_usd()
    built, trip = result(r.build), result(r.roundtrip)
    assert trip["errors"] == 0
    assert trip["measure"] == built                                        # saved by Kit, opened by Kit: nothing changed
    assert os.path.samefile(trip["real_path"], trip["file"])
    assert trip["old_handle_alive"] is False                               # why session.stage is always fetched again
    assert trip["flattened"] is False                                      # the variant set is still in the file
    assert built["splat_prims"] == ["/World/data/splats"] and built["splat_is_boundable"] == [True]
    assert ["/World/data/splats", "ParticleField3DGaussianSplat"] in built["prims"]
    space = built["variants"]["/World/data:space"]
    assert space["selection"] == "physical" and sorted(space["per_variant"]) == ["index", "physical"]
    physical, index = space["per_variant"]["physical"], space["per_variant"]["index"]
    assert physical["arrays"]["/World/data/voxels.xformOp:scale"][1] == 42.0       # 6 + 6 + 30
    assert index["arrays"]["/World/data/voxels.xformOp:scale"][1] == 18.0
    assert physical["min"] == index["min"] == [0.0, 0.0, 0.0]
    assert physical["max"] == [72.0, 96.0, 240.0]                          # 12 x 16 x 8 voxels of 6 x 6 x 30
    assert index["max"] == [72.0, 96.0, 230.0]                             # box 48 high; the splats (200 + 3 * 10) stay where they are
    assert built["upAxis"] == "Z" and built["metersPerUnit"] == 1.0 and built["defaultPrim"] == "/World"


def test_tier_off_gives_identical_stage_measurements():
    r = notebook_usd()
    off = r.off
    start = need_openusd_without_kit(off)
    assert start["tier"] == "off" and start["app"] is True and start["isaacsim"] is False
    assert start["source"] == kit.ENV_TIER and f"tier chosen by {kit.ENV_TIER}" in start["describe"]      # NCNTM_KIT=off decided
    assert start["kit_usd"] is False and start["available"] is False
    assert "not running (tier 'off')" in start["describe"]
    built_in_kit = result(r.build)
    assert result(off.generic) == built_in_kit                             # same code, no Kit: same numbers
    assert result(off.saved) == built_in_kit                               # and after save / open without Kit
    assert result(off.capture).startswith("no renderer")
    assert "nothing to close" in off.close.text and result(off.close) is True       # close() says what is true in this tier
    # "inside Jupyter" means: this process is a kernel.  A script the notebook starts is not, although it inherits
    # JPY_PARENT_PID (isaacsim's own test) from the kernel.
    assert start["in_jupyter"] is True and start["child_in_jupyter"] == "False", start
    # a Kit-free kernel: no core file if ipykernel aborts while it shuts down; the hard limit is left as it was
    assert start["core_after"] == [0, start["core_before"][1]], (start["core_before"], start["core_after"])


def test_stage_authored_by_ncntm_viz_stage_opens_identically_in_kit():
    r = notebook_usd()
    need_openusd_without_kit(r.off)
    assert r.off.author.error is None, f"ncntm.viz.stage could not author the stage: {r.off.author.error}"
    outside = result(r.off.authored)
    inside = result(r.authored)
    assert inside["errors"] == []
    assert inside["measure"] == outside                                    # composed identically
    variants = outside["variants"]["/World/demo:space"]
    assert sorted(variants["per_variant"]) == ["index", "physical"]
    scale = "/World/demo/voxels.xformOp:scale"
    assert variants["per_variant"]["physical"]["arrays"][scale][1] == 42.0 and variants["per_variant"]["index"]["arrays"][scale][1] == 18.0
    assert variants["per_variant"]["physical"]["max"] != variants["per_variant"]["index"]["max"]
    assert len(outside["textures"]) == 2                                   # relative paths, resolved next to the file
    for path, name, exists, size in outside["textures"]:
        assert path.startswith("./textures/") and exists and size > 0 and name.endswith(".png")
    if result(r.off.author)["schema"]:
        assert outside["splat_prims"] and all(outside["splat_is_boundable"])


def test_warp_runs_a_kernel_on_the_session_gpu():
    r = notebook_usd()
    warp = result(r.warp)
    if warp["warp"] is None:
        skip(f"warp is not importable in the kernel: {warp['why']}")
    facts = result(r.facts)
    assert warp["total"] == 2.0 * sum(range(1000)) + 1000.0
    assert warp["is_cuda"] and warp["device"] == facts["device"] == "cuda:0"
    assert facts["visible"] == str(GPU) and warp["cuda_devices"] == ["cuda:0"]      # CUDA sees the one GPU asked for
    assert warp["errors"] == 0
    if warp["held_gpus"] is not None:
        assert warp["held_gpus"] == [GPU], warp["held_gpus"]               # nvidia-smi: the kernel has a context there only


def test_no_renderer_in_tier_usd_is_said_quickly():
    rtx = result(notebook_usd().rtx)
    assert rtx["available"] is False
    assert rtx["message"].startswith("no renderer: ") and "tier 'usd' loads no renderer" in rtx["message"]
    assert rtx["stream"].startswith("no renderer: ")
    assert rtx["seconds"] < 5.0, rtx["seconds"]
    assert rtx["status"].startswith("RTX renderer: not available on this machine because ")


def test_full_tier_code_without_an_engine_says_no_renderer():
    """The branch tier "full" takes when no Hydra engine is attached, driven inside the slim Kit
    (which has none either), so that it is exercised without starting the stock application."""
    pretend = result(notebook_usd().pretend)
    assert pretend["available"] is False and pretend["engines"] == [] and pretend["tier"] == "usd"
    assert pretend["reason"].startswith("no Hydra render engine is attached to the USD context")
    assert pretend["message"] == "no renderer: " + pretend["reason"]
    assert pretend["seconds"] < 5.0


def test_close_is_disabled_and_harmless():
    r = notebook_usd()
    closed = result(r.close)
    assert closed["seconds"] < 2.0 and closed["stage"] is True             # no deadlock; Kit is still there
    assert r.close.text.count("close() is disabled") == 2                  # session.close() and session.app.close()


def test_exit_hook_flushes_files_left_open():
    r = notebook_usd()
    files = result(r.files)
    assert files["bytes_on_disk_now"] == 0                                 # the text was still in Python's buffer ...
    out = out_dir()
    assert (out / "unflushed.txt").read_text() == "result line\n" * 10     # ... and is on disk after the kernel is gone
    assert (out / "logging.txt").read_text().strip() == "logged line"
    if files["h5py"]:
        check = subprocess.run([setup().python, "-c", "import h5py, sys\nwith h5py.File(sys.argv[1], 'r') as f:\n"
                                "    print('H5 OK', f['x'].shape[0], int(f['x'][999]))", str(out / "open.h5")],
                               capture_output=True, text=True, timeout=120)
        assert "H5 OK 1000 999" in check.stdout, check.stdout + check.stderr


def test_kernel_exits_with_code_0_within_the_grace_period():
    timing = notebook_usd().timing
    assert timing["exit_code"] == 0, timing                                # not -15 / -9: nobody had to kill it
    assert timing["shutdown_s"] < GRACE_SECONDS, timing
    assert timing["left_running"] is False


def test_default_root_is_in_the_home_cache_not_in_the_installation():
    """No root given, no NCNTM_KIT_ROOT: ~/.cache/ncntm/kit (with a HOME inside the test folder)."""
    home = out_dir() / "home_default"
    home.mkdir(exist_ok=True)
    kernel = Kernel("default root", env={"HOME": str(home)}, drop=(kit.ENV_ROOT,))
    try:
        kernel.timing["tier"] = "usd"
        started = kernel.run(start_cell("usd") + "print('RESULT ' + json.dumps({'root': s.root, 'default': kit.default_root(), "
                             "'args': s.launch_args, 'log': s.log_file, 'errors': s.kit_errors(), 'start_seconds': s.start_seconds, "
                             "'root_tmp': os.listdir(os.path.join(s.root, 'tmp')), 'isaac': os.environ['ISAAC_PATH']}))")
    finally:
        timing = kernel.shutdown()
    facts = result(started)
    kernel.timing["start_kit_s"] = facts["start_seconds"]
    root = str(home / ".cache" / "ncntm" / "kit")
    assert facts["root"] == facts["default"] == root and facts["errors"] == []
    args = facts["args"]
    assert args[args.index("--portable-root") + 1] == root and "--portable" not in args
    assert facts["log"].startswith(root) and any(name.startswith("carb.") for name in facts["root_tmp"])
    assert not root.startswith(facts["isaac"])                             # not inside the Isaac Sim installation
    assert sorted(os.listdir(root)) == ["cache", "data", "logs", "tmp"]
    assert timing["exit_code"] == 0


# ---- refusals ---------------------------------------------------------------------------------------
def test_pxr_imported_first_is_refused_with_a_readable_error():
    notebook_usd()                                                         # writes kit_generic.usda
    kernel = Kernel("pxr first")
    try:
        first = kernel.run(NO_CORE_FILE + PRELUDE.replace("from ncntm.viz import kit", "from ncntm.viz import kit, stage")
                           + "from pxr import Usd\nprint('RESULT ' + json.dumps(str(kit.pxr_origin())))")       # the mistake
        refused = kernel.run(f"s = kit.start_kit('usd', gpu={GPU})")
        automatic = kernel.run("s = kit.start_kit()")
        after = kernel.run("print('RESULT ' + json.dumps({'isaacsim': 'isaacsim' in sys.modules, 'session': kit.current_session() is None}))")
        off = kernel.run(STAGE_CODE + "s = kit.start_kit('off')\n"
                         "stage = s.open_stage(os.path.join(os.environ['NCNTM_TEST_OUT_DIR'], 'kit_generic.usda'))\n"
                         "print('RESULT ' + json.dumps(measure(stage)))")
    finally:
        kernel.shutdown()
    if first.error is not None and "Error: No module named 'pxr'" in first.error:
        skip("pxr cannot be imported without Kit by this interpreter: there is nothing to guard against")
    assert "interpreter build" in result(first)
    for cell in (refused, automatic):
        assert cell.error is not None and cell.error.startswith("RuntimeError: pxr (OpenUSD "), cell.error
        assert "Restart the kernel" in cell.error and "start_kit(tier='off')" in cell.error
        # it names who imported pxr too early and shows the two lines that must come first
        assert "ncntm.viz.stage" in cell.error and "a cell of this notebook" in cell.error, cell.error
        assert "from ncntm.viz import kit\n" in cell.error and "session = kit.start_kit()" in cell.error
    assert result(after) == {"isaacsim": False, "session": True}           # Kit was not touched
    assert result(off) == result(notebook_usd().build)                     # a file saved by Kit reads the same without Kit


def test_licence_not_accepted_gives_a_readable_error():
    kernel = Kernel("no licence", drop=(kit.EULA_VARIABLE,))
    try:
        status = kernel.run(NO_CORE_FILE + PRELUDE + "print('RESULT ' + json.dumps(list(kit.licence_status())))")
        refused = kernel.run(f"s = kit.start_kit('usd', gpu={GPU})")
        after = kernel.run("print('RESULT ' + json.dumps({'isaacsim': 'isaacsim' in sys.modules, 'variable': os.environ.get(kit.EULA_VARIABLE)}))")
        off = kernel.run("print('RESULT ' + json.dumps(kit.start_kit('off').tier))")
    finally:
        kernel.shutdown()
    accepted, how = result(status)
    if accepted:
        skip(f"the licence is accepted on this machine without the variable ({how}); the refusal cannot be shown")
    assert refused.error is not None and refused.error.startswith("RuntimeError: Isaac Sim was not started"), refused.error
    assert kit.EULA_VARIABLE in refused.error and "--accept-eula" in refused.error and kit.EULA_URL in refused.error
    assert result(after) == {"isaacsim": False, "variable": None}          # not imported (it would ask on stdin), not set by us
    assert result(off) == "off"                                            # working without Kit is still possible


def test_a_start_that_is_cut_short_taints_the_session():
    """KeyboardInterrupt (or anything else) escaping from SimulationApp(): no second attempt."""
    kernel = Kernel("cut short")
    try:
        # a stand-in for the isaacsim module, so that neither Carbonite nor Kit is loaded by this test
        fake = kernel.run(NO_CORE_FILE + PRELUDE + "import builtins, importlib.util, types\n"
                          "def cut_short(*args, **kwargs):\n    raise KeyboardInterrupt()\n"
                          "stand_in = types.ModuleType('isaacsim')\n"
                          "stand_in.__spec__ = importlib.util.find_spec('isaacsim')\n"
                          "stand_in.SimulationApp = cut_short\n"
                          "sys.modules['isaacsim'] = stand_in\n"
                          "builtins.ISAAC_LAUNCHED_FROM_JUPYTER = True\n"
                          "os.environ['ISAAC_PATH'] = os.path.join(os.environ['NCNTM_TEST_OUT_DIR'], 'no_isaac')\n"
                          "os.makedirs(os.path.join(os.environ['ISAAC_PATH'], 'extscache'), exist_ok=True)\n"
                          f"s = kit.start_kit('usd', gpu={GPU})")
        again = kernel.run(f"s = kit.start_kit('usd', gpu={GPU})")
        state = kernel.run("import signal\nprint('RESULT ' + json.dumps({'session': kit.current_session() is None, "
                           "'sigint': signal.getsignal(signal.SIGINT) is signal.default_int_handler}))")
    finally:
        kernel.shutdown()
    assert fake.error is not None and fake.error.startswith("KeyboardInterrupt"), fake.error
    assert again.error is not None and again.error.startswith("RuntimeError: Kit is in an undefined state"), again.error
    assert "Restart the kernel" in again.error and "KeyboardInterrupt" in again.error
    assert result(state) == {"session": True, "sigint": True}


# ---- interrupts and queued cells ----------------------------------------------------------------------
def test_interrupt_while_kit_starts_is_held_back():
    """Tier "core" takes seconds to start: interrupt it after one second."""
    kernel = Kernel("core, interrupted")
    try:
        kernel.timing["tier"] = "core"
        first = kernel.run(start_cell("core") + "print('NOT REACHED')", interrupt_after=1.0)
        again = kernel.run(start_cell("core") + "print('RESULT ' + json.dumps({'tier': s.tier, 'start_seconds': s.start_seconds, "
                           "'errors': s.kit_errors(), 'extensions': s.extensions, 'privacy': s.privacy(), 'warp': s.warp_version, "
                           "'args_ok': all(a in s.launch_args for a in kit.PRIVACY_ARGS) and kit.excluded_extensions_arg() in s.launch_args}))")
        use = kernel.run(STAGE_CODE + "stage = build(s.new_stage()); s.update(3)\nprint('RESULT ' + json.dumps(measure(stage)))")
        pump = kernel.run("n = 0\nfor i in range(500):\n    s.update(); time.sleep(0.02); n += 1\nprint('NOT REACHED')", interrupt_after=1.0)
        after = kernel.run("s.update(2)\nprint('RESULT ' + json.dumps({'n': n, 'errors': s.kit_errors()}))")
    finally:
        timing = kernel.shutdown()
    facts = result(again)
    kernel.timing["start_kit_s"] = facts["start_seconds"]
    assert first.error is not None and first.error.startswith("KeyboardInterrupt"), first.error
    assert "Kit is intact" in first.error and "NOT REACHED" not in first.text
    assert first.seconds >= facts["start_seconds"] > 1.0                   # the cell ended only when the start was complete
    assert facts["tier"] == "core" and facts["errors"] == [], facts["errors"]       # nothing half-initialised
    assert facts["args_ok"] and not any(facts["privacy"].values())
    assert result(use) == result(notebook_usd().build)                     # the same stage numbers in tier core
    assert pump.error is not None and pump.error.startswith("KeyboardInterrupt")    # a later interrupt is an ordinary one
    assert 0 < result(after)["n"] < 500 and result(after)["errors"] == []
    assert timing["exit_code"] == 0 and timing["shutdown_s"] < GRACE_SECONDS, timing


def test_run_all_does_not_lose_queued_cells():
    """JupyterLab's "Run All" sends every cell at once.  With ipykernel 7 the cells queued behind
    a cell that drives Kit were dropped without a trace; kit.protect_queued_cells keeps them."""
    kernel = protected = Kernel("run all")
    try:
        kernel.timing["tier"] = "usd"
        cells, order = kernel.run_all([
            start_cell("usd") + "print('A', s.tier)",
            "for i in range(50):\n    s.update(); time.sleep(0.02)\nprint('B pumped')",
            "from pxr import UsdGeom\nstage = s.new_stage(); UsdGeom.Cube.Define(stage, '/World/Cube'); s.update(3)\n"
            "print('C', [str(p.GetPath()) for p in s.stage.Traverse()])",
            "import asyncio\nawait asyncio.sleep(0.05)\nprint('D awaited')",
            "print('RESULT ' + json.dumps({'errors': s.kit_errors(), 'start_seconds': s.start_seconds}))",
        ], timeout=120)
    finally:
        timing = kernel.shutdown()
    assert [cell.error for cell in cells] == [None] * 5, [cell.error for cell in cells]
    assert order == [0, 1, 2, 3, 4], order                                 # all of them, in order

    # control (reported, not asserted: it describes ipykernel, not this package): the same without the protection
    kernel = Kernel("run all, unprotected")
    try:
        kernel.timing["tier"] = "usd"
        control, _ = kernel.run_all([PRELUDE + "kit.protect_queued_cells = lambda: False\n" + f"s = kit.start_kit('usd', gpu={GPU})\nprint('A')",
                                     "print('B')", "print('C')"], timeout=8)
        alive = kernel.run("print('RESULT ' + json.dumps([l.split('[Error]')[1].strip()[:90] for l in s.kit_errors()][:1]))")
    finally:
        kernel.shutdown()
    lost = sum(1 for cell in control[1:] if not cell.done)
    print(f"control without protect_queued_cells(): {lost} of 2 queued cells never ran"
          + (f"; Kit's log: {result(alive)[0]}" if lost and result(alive) else ""))
    assert "A usd" in cells[0].text and "B pumped" in cells[1].text and "/World/Cube" in cells[2].text and "D awaited" in cells[3].text
    assert result(cells[4])["errors"] == []                                # in particular no "cannot enter context"
    protected.timing["start_kit_s"] = result(cells[4])["start_seconds"]
    assert timing["exit_code"] == 0


# ---- other ways to run ----------------------------------------------------------------------------------
def test_notebook_file_executes_with_nbclient():
    """The route of ``jupyter nbconvert --execute``: an .ipynb, run cell by cell, kernel shut down by nbclient."""
    setup()
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as error:
        skip(f"nbclient / nbformat are not installed: {error}")
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source) for source in (
        start_cell("usd") + "print(s.describe())",
        "from pxr import UsdGeom\nstage = s.new_stage(); UsdGeom.Cube.Define(stage, '/World/Cube'); s.update(2)\nprint('prims', len(list(s.stage.Traverse())))",
        "print('saved', os.path.basename(s.save(os.path.join(os.environ['NCNTM_TEST_OUT_DIR'], 'nbclient.usda'))), len(s.kit_errors()))",
    )]
    path = out_dir() / "executed.ipynb"
    saved = {key: os.environ.get(key) for key in setup().env}
    os.environ.update(setup().env)                                         # nbclient starts the kernel with this process's environment
    saved_tmp = tempfile.tempdir
    (out_dir() / "tmp").mkdir(exist_ok=True)
    tempfile.tempdir = str(out_dir() / "tmp")                              # where nbclient's kernel manager puts the connection file
    start = time.perf_counter()
    try:
        NotebookClient(notebook, kernel_name=KERNEL_NAME, timeout=300, resources={"metadata": {"path": str(out_dir())}}).execute()
    finally:
        tempfile.tempdir = saved_tmp
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    TIMINGS.append({"kernel": "nbclient, 3 cells", "tier": "usd", "kernel_ready_s": None, "start_kit_s": None,
                    "shutdown_s": None, "exit_code": None, "whole_notebook_s": round(time.perf_counter() - start, 2)})
    nbformat.write(notebook, path)
    texts = ["".join(o.get("text", "") for o in cell.outputs if o.output_type == "stream") for cell in notebook.cells]
    assert not any(o.output_type == "error" for cell in notebook.cells for o in cell.outputs)
    assert "tier 'usd'" in texts[0] and "prims 2" in texts[1] and "saved nbclient.usda 0" in texts[2]
    assert (out_dir() / "nbclient.usda").is_file()


def test_nbconvert_command_line_executes_a_notebook():
    """``jupyter nbconvert --to notebook --execute`` end to end, as a command, with the kernel spec by name."""
    env = setup()
    try:
        import nbconvert  # noqa: F401
        import nbformat
    except ImportError as error:
        skip(f"nbconvert / nbformat are not installed in {sys.executable}: {error}")
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source) for source in (
        start_cell("usd") + "print(s.describe())",
        "from ncntm.viz import stage as S              # imports pxr: only after start_kit\nprint('stage module', S.__name__)",
        "from pxr import UsdGeom\nstage = s.new_stage(); UsdGeom.Cube.Define(stage, '/World/Cube'); s.update(2)\nprint('prims', len(list(s.stage.Traverse())))",
        "print('saved', os.path.basename(s.save(os.path.join(os.environ['NCNTM_TEST_OUT_DIR'], 'nbconvert.usda'))), len(s.kit_errors()))",
    )]
    source = out_dir() / "nbconvert_source.ipynb"
    nbformat.write(notebook, source)
    (out_dir() / "tmp").mkdir(exist_ok=True)
    run_env = dict(env.env, TMPDIR=str(out_dir() / "tmp"))                 # nbconvert's connection file goes there
    start = time.perf_counter()
    done = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                           f"--ExecutePreprocessor.kernel_name={KERNEL_NAME}", "--ExecutePreprocessor.timeout=300",
                           "--output", "nbconvert_executed.ipynb", "--output-dir", str(out_dir()), str(source)],
                          capture_output=True, text=True, env=run_env, timeout=600, stdin=subprocess.DEVNULL)
    TIMINGS.append({"kernel": "nbconvert command", "tier": "usd", "kernel_ready_s": None, "start_kit_s": None,
                    "shutdown_s": None, "exit_code": done.returncode, "whole_notebook_s": round(time.perf_counter() - start, 2)})
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    executed = nbformat.read(out_dir() / "nbconvert_executed.ipynb", as_version=4)
    texts = ["".join(o.get("text", "") for o in cell.outputs if o.output_type == "stream") for cell in executed.cells]
    assert not any(o.output_type == "error" for cell in executed.cells for o in cell.outputs)
    assert "tier 'usd'" in texts[0] and "stage module ncntm.viz.stage" in texts[1] and "prims 2" in texts[2]
    assert "saved nbconvert.usda 0" in texts[3] and (out_dir() / "nbconvert.usda").is_file()


def test_plain_script_keeps_its_exit_status():
    """``python script.py`` with start_kit in it: the exit hook leaves with the script's own status and flushes
    files left open.  JPY_PARENT_PID is set as it is for a script started from a notebook ("!python script.py")."""
    env = setup()
    spec = spec_of(env.prefix)
    script = out_dir() / "plain_script.py"
    script.write_text(
        PRELUDE + f"s = kit.start_kit('usd', gpu={GPU})\n"
        "f = open(os.path.join(os.environ['NCNTM_TEST_OUT_DIR'], 'script_' + sys.argv[1] + '.txt'), 'w')\n"
        "f.write('kept ' + sys.argv[1])\n"
        "print('RESULT ' + json.dumps({'in_jupyter': kit.in_jupyter(), 'errors': s.kit_errors(), 'tier': s.tier,\n"
        "    'args_ok': all(a in s.launch_args for a in kit.PRIVACY_ARGS) and kit.excluded_extensions_arg() in s.launch_args}), flush=True)\n"
        "if sys.argv[1] == 'exit3':\n    sys.exit(3)\n"
        "if sys.argv[1] == 'raise':\n    raise ValueError('uncaught')\n")
    run_env = dict(env.env)
    run_env.update(spec["env"])
    run_env.update(JPY_PARENT_PID=str(os.getpid()), PYTHONDONTWRITEBYTECODE="1")
    for mode, expected in (("end", 0), ("exit3", 3), ("raise", 1)):
        start = time.perf_counter()
        done = subprocess.run([spec["argv"][0], str(script), mode], capture_output=True, text=True, env=run_env,
                              timeout=300, stdin=subprocess.DEVNULL)
        lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
        assert lines, done.stdout[-1500:] + done.stderr[-1500:]
        facts = json.loads(lines[-1][len("RESULT "):])
        TIMINGS.append({"kernel": f"script, {mode}", "tier": facts["tier"], "kernel_ready_s": None, "start_kit_s": None,
                        "shutdown_s": None, "exit_code": done.returncode, "whole_notebook_s": round(time.perf_counter() - start, 2)})
        assert facts == {"in_jupyter": False, "errors": [], "tier": "usd", "args_ok": True}, facts
        if env.kind == "wrapper":                                          # python.sh turns every failure into status 1
            assert (done.returncode == 0) == (expected == 0), (mode, done.returncode)
        else:
            assert done.returncode == expected, (mode, done.returncode, done.stderr[-500:])
        assert (out_dir() / f"script_{mode}.txt").read_text() == f"kept {mode}"       # the file left open was flushed


def test_cuda_initialised_first_is_said_and_a_loud_start_is_complete():
    """CUDA initialised before start_kit(gpu=N): too late for CUDA_VISIBLE_DEVICES, and start_kit says so.
    The same kernel starts Kit with quiet=False: the output of the start is in the cell, and the arguments
    Kit really got (the privacy switches among them) are still known."""
    kernel = Kernel("cuda first, loud", drop=("CUDA_VISIBLE_DEVICES",))
    try:
        kernel.timing["tier"] = "usd"
        first = kernel.run(
            PRELUDE + "import ctypes, warnings\n"
            "try:\n    code = ctypes.CDLL('libcuda.so.1').cuInit(0)       # the CUDA driver, without a context on any GPU\n"
            "except OSError as error:\n    code = str(error)\n"
            "with warnings.catch_warnings(record=True) as caught:\n    warnings.simplefilter('always')\n"
            f"    s = kit.start_kit('usd', gpu={GPU}, quiet=False)\n"
            "print('RESULT ' + json.dumps({'cuInit': code, 'warnings': [str(w.message) for w in caught], 'note': s.gpu_note,\n"
            "    'device': s.device, 'visible': os.environ.get('CUDA_VISIBLE_DEVICES'), 'args_from_kit': s.launch_args_from_kit,\n"
            "    'args': s.launch_args, 'start_output_file': s.start_output_file, 'errors': s.kit_errors(),\n"
            "    'start_seconds': s.start_seconds, 'gpus': kit.gpu_count(), 'privacy': s.privacy()}))")
        later = kernel.run("print('printed after a loud start')")
    finally:
        timing = kernel.shutdown()
    facts = result(first)
    kernel.timing["start_kit_s"] = facts["start_seconds"]
    assert facts["errors"] == [] and "printed after a loud start" in later.text
    # quiet=False: SimulationApp's own lines are in the cell, nothing went to a file, and the argument list is Kit's
    assert "Starting kit application with the following args" in first.text
    assert facts["start_output_file"] == "" and facts["args_from_kit"] is True
    for arg in kit.PRIVACY_ARGS + (kit.excluded_extensions_arg(), kit.STDOUT_ARG, "--portable-root"):
        assert arg in facts["args"], arg
    assert not any(arg in facts["args"] for arg in kit.QUIET_ARGS) and not any(facts["privacy"].values())
    if facts["cuInit"] != 0:
        skip(f"the CUDA driver could not be initialised in the kernel (cuInit: {facts['cuInit']}): no warning to show")
    if facts["gpus"] == 1 and GPU == 0:                                    # one GPU: nothing to restrict, nothing to warn about
        assert "one GPU" in facts["note"] and facts["visible"] is None
    else:
        assert any("CUDA was already initialised" in w and "before the first CUDA use" in w for w in facts["warnings"]), facts["warnings"]
        assert facts["visible"] is None and "NOT restricted" in facts["note"]         # the environment was not changed in vain
        assert facts["device"] == f"cuda:{GPU}"                            # all GPUs are visible: the GPU keeps its index
    assert timing["exit_code"] == 0


def test_source_build_layout_through_python_sh():
    """A binary / source build is used through python.sh.  Stand-in: a folder laid out like a SOURCE build of
    Isaac Sim 6.2.0 (see source_build_tree): python_packages is a link into <repo>/source, python.sh has the
    shape of the real one, <repo>/.eula_accepted is what build.sh leaves behind; kit/, exts/, apps/ and
    extscache/ link to the pip install and the pip interpreter stands in for Kit's Python.
    The kernel gets NO OMNI_KIT_ACCEPT_EULA: in a build the licence counts through the build's own record.
    Not shown by this: Kit's own Python, a real setup_python_env.sh, aarch64."""
    env = setup()
    if env.kind == "wrapper":
        skip("the kernels of this run already go through a real python.sh")
    package = env.probe["isaacsim"]
    for name in ("kit", "exts", "apps", "extscache"):
        if not os.path.isdir(os.path.join(package, name)):
            skip(f"{package} has no {name}/ folder: not the pip layout this stand-in is made from")
    repo, release = source_build_tree("build_tree", env.python,
                                      package_files=[os.path.join(package, "__init__.py"), os.path.join(package, "__main__.py")])
    for name in ("kit", "exts", "apps", "extscache"):
        if not (release / name).exists():
            os.symlink(os.path.join(package, name), release / name)
    code, text = run_installer("--prefix", env.prefix, "--python", release / "python.sh", "--name", "ncntm-standin")
    assert code == 0, text
    spec = spec_of(env.prefix, "ncntm-standin")
    assert spec["argv"][0] == str(release / "python.sh") and spec["interrupt_mode"] == "message"
    assert kit.EULA_VARIABLE not in spec["env"]
    kernel = Kernel("source build, python.sh", name="ncntm-standin", drop=(kit.EULA_VARIABLE,))
    try:
        kernel.timing["tier"] = "usd"
        started = kernel.run(start_cell("usd") + "print('RESULT ' + json.dumps({'isaac_path': os.environ['ISAAC_PATH'], 'errors': s.kit_errors(), "
                             "'start_seconds': s.start_seconds, 'args': s.launch_args, 'parent': os.getppid(), 'pid': os.getpid(), "
                             "'folders': kit.isaac_extension_folders(), 'install': kit.isaacsim_install(), "
                             "'licence': list(kit.licence_status()), 'variable': os.environ.get(kit.EULA_VARIABLE), "
                             "'privacy': s.privacy(), 'usd': s.usd_version, 'kit_usd': kit.pxr_origin().is_kit, "
                             "'isaacsim_file': sys.modules['isaacsim'].__file__, 'preload': os.environ.get('LD_PRELOAD', '')}))")
        pump = kernel.run("n = 0\nfor i in range(500):\n    s.update(); time.sleep(0.02); n += 1\nprint('NOT REACHED')", interrupt_after=1.0)
        after = kernel.run("s.new_stage(); s.update(2)\nprint('RESULT ' + json.dumps({'n': n, 'errors': s.kit_errors()}))")
    finally:
        timing = kernel.shutdown()
    facts = result(started)
    kernel.timing["start_kit_s"] = facts["start_seconds"]
    assert facts["parent"] == kernel.process.pid and facts["pid"] != kernel.process.pid      # the kernel is python.sh's child
    assert facts["isaac_path"] == str(release) and facts["errors"] == []
    # the build's own Python package was used, and recognised as a build although it lives behind a link
    assert facts["isaacsim_file"] == str(release / "python_packages" / "isaacsim" / "__init__.py")
    assert facts["install"]["layout"] == "build"
    assert facts["install"]["package"] == str(repo / "source" / "python_packages" / "isaacsim")
    # the licence: no variable, accepted through what build.sh wrote into the repository
    assert facts["variable"] is None and facts["licence"] == [True, f"marker file {repo / '.eula_accepted'}"]
    assert facts["kit_usd"] and str(release / "kit" / "libcarb.so") in facts["preload"]
    assert facts["folders"] == [str(release / "extscache")]                # extension folders follow ISAAC_PATH
    args = facts["args"]
    assert [args[i + 1] for i, a in enumerate(args) if a == "--ext-folder"] == [str(release / "exts"), str(release / "apps"), str(release / "extscache")]
    for arg in kit.PRIVACY_ARGS + (kit.excluded_extensions_arg(),):
        assert arg in args, arg
    assert not any(facts["privacy"].values())
    assert pump.error is not None and pump.error.startswith("KeyboardInterrupt")    # "Interrupt" reaches the child (as a message)
    assert 0 < result(after)["n"] < 500 and result(after)["errors"] == []
    assert timing["exit_code"] == 0 and timing["shutdown_s"] < GRACE_SECONDS and timing["left_running"] is False, timing


# ---- the stock application ----------------------------------------------------------------------------------
def test_full_tier_rtx_path():
    """ONE start of the stock Isaac Sim application (crash reporter and telemetry off).

    Without a graphics device (the development host): renderer_available is False, capture_rtx
    and start_livestream raise "no renderer ..." at once, describe() says why.
    With one: capture_rtx must return a picture.  That branch has never run on the development host."""
    setup()
    if os.environ.get("NCNTM_TEST_KIT_FULL", "1").lower() in ("0", "no", "false", "off"):
        skip("NCNTM_TEST_KIT_FULL=0: the stock Isaac Sim application was not started")
    kernel = Kernel("full")
    try:
        kernel.timing["tier"] = "full"
        started = kernel.run(start_cell("full") + "print('RESULT ' + json.dumps({'tier': s.tier, 'start_seconds': s.start_seconds, "
                             "'extensions': s.extensions, 'errors': s.kit_errors(), 'privacy': s.privacy(), 'usd': s.usd_version, "
                             "'args': s.launch_args, 'args_from_kit': s.launch_args_from_kit, 'root': s.root}))", timeout=600)
        status = kernel.run("t0 = time.perf_counter()\nstatus = s.renderer_status()\n"
                            "import omni.usd\n"
                            "print('RESULT ' + json.dumps({'available': status.available, 'reason': status.reason, 'engines': list(status.engines), "
                            "'attached': list(omni.usd.get_context().get_attached_hydra_engine_names()), "
                            "'seconds': time.perf_counter() - t0, 'describe': s.describe()}))", timeout=600)
        capture = kernel.run(
            "import numpy as np\nfrom pxr import Gf, UsdGeom, UsdLux\n"
            "stage = s.new_stage()\n"
            "UsdGeom.Cube.Define(stage, '/World/Cube')\n"
            "UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1000.0)\n"
            "camera = UsdGeom.Camera.Define(stage, '/World/Camera')\n"
            "UsdGeom.Xformable(camera).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 10.0))        # looks down -Z at the cube\n"
            "s.update(5)\n"
            "out = {}\n"
            "t0 = time.perf_counter()\n"
            "try:\n"
            "    picture = s.capture_rtx('/World/Camera', 96, 64, subframes=8)\n"
            "    out['picture'] = {'shape': list(picture.shape), 'dtype': str(picture.dtype), 'max': int(picture[..., :3].max()),\n"
            "                      'colours': int(len(np.unique(picture.reshape(-1, 4), axis=0)))}\n"
            "except RuntimeError as error:\n"
            "    out['error'] = str(error)\n"
            "out['seconds'] = time.perf_counter() - t0\n"
            "t0 = time.perf_counter()\n"
            "try:\n"
            "    s.start_livestream()\n"
            "    out['stream'] = 'started'\n"
            "except RuntimeError as error:\n"
            "    out['stream'] = str(error)\n"
            "out['stream_seconds'] = time.perf_counter() - t0\n"
            "out['prims'] = [str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith('/World')]\n"
            "def root_layer_prims():\n"
            "    found = []\n"
            "    def visit(path):\n"
            "        found.append(str(path))\n"
            "    s.stage.GetRootLayer().Traverse('/', visit)\n"
            "    return sorted(p for p in found if p != '/' and '.' not in p and not p.startswith('/World'))\n"
            "out['root_before'] = root_layer_prims()\n"
            "print('RESULT ' + json.dumps(out))", timeout=600)
        # The capture sequence itself (render product, annotator, orchestrator step, release), called past the
        # "is there a renderer" gate, in both render modes.  Without a renderer it must come back with an empty
        # frame in bounded time; this is the most of the RTX path that can run on a machine without a graphics device.
        sequence = kernel.run(
            "import carb.settings\n"
            "settings = carb.settings.get_settings()\n"
            "out = {'mode_before': settings.get('/rtx/rendermode'), 'spp_before': settings.get('/rtx/pathtracing/spp'),\n"
            "       'total_before': settings.get('/rtx/pathtracing/totalSpp'), 'persp': bool(s.stage.GetPrimAtPath('/OmniverseKit_Persp'))}\n"
            "for name, arguments in (('test', (None, 64, 64, 2, None, None)),\n"
            "                        ('realtime', ('/World/Camera', 96, 64, 4, 'RealTimePathTracing', None)),\n"
            "                        ('pathtraced', ('/World/Camera', 96, 64, 4, 'PathTracing', 2))):\n"
            "    t0 = time.perf_counter()\n"
            "    try:\n"
            "        frame = s._render(*arguments)\n"
            "        out[name] = {'shape': list(frame.shape), 'size': int(frame.size), 'dtype': str(frame.dtype)}\n"
            "    except Exception as error:\n"
            "        out[name] = {'error': f'{type(error).__name__}: {error}'}\n"
            "    out[name]['seconds'] = round(time.perf_counter() - t0, 2)\n"
            "out.update(mode_after=settings.get('/rtx/rendermode'), spp_after=settings.get('/rtx/pathtracing/spp'),\n"
            "           total_after=settings.get('/rtx/pathtracing/totalSpp'),\n"
            "           capture_on_play=settings.get('/omni/replicator/captureOnPlay'))\n"
            "for name, path in (('missing', '/World/NoSuchCamera'), ('not_a_camera', '/World/Cube')):\n"
            "    try:\n"
            "        s._render(path, 64, 64, 2, None, None)\n"
            "        out[name] = 'rendered'\n"
            "    except ValueError as error:\n"
            "        out[name] = str(error)\n"
            "out['root_after'] = root_layer_prims()\n"
            "out['session_layer_prims'] = [str(spec.path) for spec in s.stage.GetSessionLayer().rootPrims]\n"
            "out['saved'] = s.save(os.path.join(os.environ['NCNTM_TEST_OUT_DIR'], 'full_after_capture.usda'))\n"
            "out['new_errors'] = [line.split('[Error]')[1].strip()[:120] for line in s.kit_errors(200)][-6:]\n"
            "print('RESULT ' + json.dumps(out))", timeout=900)
    finally:
        timing = kernel.shutdown()
    facts = result(started)
    kernel.timing["start_kit_s"] = facts["start_seconds"]
    assert facts["tier"] == "full" and facts["extensions"] > 100
    assert facts["args_from_kit"]
    for arg in kit.PRIVACY_ARGS:                                           # the stock application is the one that would send data
        assert arg in facts["args"], arg
    assert kit.excluded_extensions_arg() in facts["args"]
    assert not any(facts["privacy"].values()), facts["privacy"]
    assert facts["args"][facts["args"].index("--portable-root") + 1] == facts["root"] == str(setup().root)
    assert "--portable" not in facts["args"]
    assert not any("omni.kit.renderer.init" in a for a in facts["args"])   # the renderer is NOT switched off: RTX must get its chance

    assert uploads_in(setup().root) == []                                  # the stock application sent nothing out

    state, shot, steps = result(status), result(capture), result(sequence)
    print(f"tier full: {facts['extensions']} extensions, started in {facts['start_seconds']} s, {len(facts['errors'])} [Error] lines; "
          f"Hydra engines {state['attached']}; renderer available: {state['available']} ({state['reason'][:160]})")
    print("tier full, capture sequence run directly: " + "; ".join(
        f"{name} {steps[name].get('shape', steps[name].get('error'))} in {steps[name]['seconds']} s" for name in ("test", "realtime", "pathtraced")))
    # the capture sequence ran through, whatever the renderer gave: every Replicator call in it exists in this Isaac Sim
    for name in ("test", "realtime", "pathtraced"):
        assert "error" not in steps[name], steps[name]
        assert steps[name]["seconds"] < 120.0, steps[name]                 # bounded: Replicator gives up after 1000 Kit frames
    assert steps["mode_after"] == steps["mode_before"]                     # the render settings it changed are put back
    assert steps["spp_after"] == steps["spp_before"] and steps["total_after"] == steps["total_before"]
    assert steps["capture_on_play"] is False
    assert "no prim /World/NoSuchCamera" in steps["missing"] and "needs a Camera" in steps["not_a_camera"]
    # What ends up in the file the notebook saves, besides /World.  Kit's own viewport keeps a /Render scope in the
    # ROOT layer of every stage of the stock application (reported, not asserted: it is Kit's doing).  The camera
    # of the renderer test and Replicator's graph must not be there.
    added = sorted(set(steps["root_after"]) - set(shot["root_before"]))
    print(f"tier full, prims in the root layer besides /World: {shot['root_before']}; added by the capture sequence: {added}")
    assert not any("ncntm_renderer_test_camera" in path or path.startswith("/Orchestrator") for path in steps["root_after"])
    saved_text = open(steps["saved"]).read()
    assert "ncntm_renderer_test_camera" not in saved_text and 'def Cube "Cube"' in saved_text
    assert "/World/Cube" in shot["prims"] and "/World/Camera" in shot["prims"]            # the stage works either way
    if not state["available"]:
        for name in ("test", "realtime", "pathtraced"):
            assert steps[name]["size"] == 0, steps[name]                   # no renderer: an empty frame, not an exception
        assert state["seconds"] < 5.0, state["seconds"]                    # the answer does not wait for a renderer
        assert "RTX renderer: not available on this machine because " in state["describe"]
        assert shot["error"].startswith("no renderer: ") and shot["seconds"] < 5.0, shot
        assert shot["stream"].startswith("no renderer: ") and shot["stream_seconds"] < 5.0, shot
        if not state["attached"]:
            assert "no Hydra render engine is attached" in state["reason"]
    else:                                                                  # a machine with RTX
        assert steps["test"]["shape"] == [64, 64, 4] and steps["realtime"]["shape"] == [64, 96, 4], steps
        assert steps["pathtraced"]["shape"] == [64, 96, 4] and steps["pathtraced"]["dtype"] == "uint8", steps
        assert state["attached"] and "error" not in shot, shot
        assert shot["picture"]["shape"] == [64, 96, 4] and shot["picture"]["dtype"] == "uint8"
        assert shot["picture"]["max"] > 0 and shot["picture"]["colours"] > 1
        assert "livestream=True" in shot["stream"]                         # the session was started without it
    assert timing["exit_code"] == 0 and timing["shutdown_s"] < GRACE_SECONDS, timing


# ---- script form -----------------------------------------------------------------------------------------------
def report_timings():
    if not TIMINGS:
        return
    print("\nkernel                    tier   kernel ready   start_kit   shutdown   exit code")
    for row in TIMINGS:
        def fmt(value, unit=" s"):
            return "-" if value is None else f"{value}{unit}"
        extra = f"   whole run {row['whole_notebook_s']} s" if "whole_notebook_s" in row else ""
        print(f"{row['kernel']:<25} {row['tier']:<6} {fmt(row['kernel_ready_s']):>12} {fmt(row['start_kit_s']):>11} "
              f"{fmt(row['shutdown_s']):>10} {fmt(row['exit_code'], ''):>11}{extra}")


def _fill_timings():
    """start_kit seconds of the notebooks whose facts were read by tests."""
    for name, key in (("notebook_usd", "facts"),):
        value, error = _CACHE.get(name, (None, None))
        if value is not None:
            try:
                value.timing["start_kit_s"] = result(getattr(value, key))["start_seconds"]
            except Exception:
                pass


if __name__ == "__main__":
    failed = []
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            start = time.perf_counter()
            try:
                test()
                print(f"PASS  {name}  [{time.perf_counter() - start:.1f} s]", flush=True)
            except Skip as reason:
                print(f"SKIP  {name}: {reason}", flush=True)
            except Exception:
                traceback.print_exc()
                print(f"FAIL  {name}", flush=True)
                failed.append(name)
    _fill_timings()
    report_timings()
    print(f"\n{len(failed)} failed" if failed else "\nall tests passed")
    sys.exit(1 if failed else 0)
