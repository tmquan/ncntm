"""Tests of the package as a whole: what ``import ncntm.viz`` loads, the names it offers, and the
portability rules (no absolute path, no hard-coded device, no package outside the listed ones).

    python -m pytest tests/test_package.py        or        python tests/test_package.py

The import tests run in a fresh interpreter each (a module that is already loaded cannot show
when it was loaded).
"""
import ast
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PACKAGE = REPO / "ncntm" / "viz"
SOURCES = sorted(PACKAGE.glob("*.py")) + sorted((REPO / "tests").glob("*.py"))

# what must not be loaded by "import ncntm.viz": the two that are slow or order-sensitive (warp
# compiles kernels, pxr must come after Kit) and every library that is only needed later
HEAVY = ("warp", "pxr", "PIL", "h5py", "scipy", "skimage", "kimimaro", "zmesh", "isaacsim", "omni", "matplotlib")

# third-party packages the package and its tests may import (import name: where it is used)
ALLOWED = {
    "numpy": "everywhere", "warp": "emray", "pxr": "stage, usd_to_emray", "h5py": "samples, usd_to_emray",
    "scipy": "samples, convert", "skimage": "convert", "kimimaro": "convert", "zmesh": "convert", "PIL": "figure",
    "IPython": "figure.show, optional", "pytest": "tests",
    "isaacsim": "kit", "omni": "kit", "carb": "kit",                                       # Isaac Sim itself
    "jupyter_client": "test_kit", "nbclient": "test_kit", "nbconvert": "test_kit", "nbformat": "test_kit",
    "ncntm": "tests",
}


class Skip(Exception):
    """Raised by a test that cannot run here (a package is missing)."""


def fresh_interpreter(code: str) -> dict:
    """Run ``code`` in a new Python process started in another directory; it prints one JSON object."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO.parent),
                          timeout=300, stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


LOADED = f"sorted(m for m in {HEAVY!r} if m in sys.modules)"


def test_import_loads_numpy_and_nothing_heavy():
    out = fresh_interpreter(
        "import json, sys, time\n"
        "import numpy\n"                                            # numpy's own import time is not the package's
        "t0 = time.perf_counter()\n"
        "import ncntm.viz as viz\n"
        "seconds = time.perf_counter() - t0\n"
        f"after_import = {LOADED}\n"
        "from ncntm.viz import *\n"
        f"after_star = {LOADED}\n"
        "names = sorted(n for n in viz.__all__ if n not in globals())\n"
        "kit = viz.kit\n"                                           # must stay possible BEFORE pxr is loaded
        f"after_kit = {LOADED}\n"
        "print(json.dumps(dict(seconds=seconds, after_import=after_import, after_star=after_star, missing=names,\n"
        "                      after_kit=after_kit, kit=kit.__name__)))\n")
    assert out["after_import"] == [], out
    assert out["after_star"] == [] and out["missing"] == [], out    # the star import brings every name of __all__
    assert out["after_kit"] == [] and out["kit"] == "ncntm.viz.kit", out
    assert out["seconds"] < 1.0, out                                # measured: a few milliseconds after numpy
    print(f"\n  import ncntm.viz: {out['seconds'] * 1000:.1f} ms after numpy")


def test_lazy_names_load_their_module_on_first_use():
    out = fresh_interpreter(
        "import json, sys\n"
        "import ncntm.viz as viz\n"
        "steps = {}\n"
        f"steps['start'] = {LOADED}\n"
        "sample_function = viz.load_sample\n"
        f"steps['eager name'] = {LOADED}\n"
        "figure = viz.figure\n"
        f"steps['figure'] = {LOADED}\n"
        "try:\n"
        "    camera = viz.Camera\n"
        f"    steps['Camera'] = {LOADED}\n"
        "    same = camera is viz.emray.Camera and 'Camera' in vars(viz)\n"
        "except ImportError as error:\n"
        "    steps['Camera'], same = str(error), None\n"
        "print(json.dumps(dict(steps=steps, same=same)))\n")
    steps = out["steps"]
    assert steps["start"] == [] and steps["eager name"] == []
    assert steps["figure"] == ["PIL"]
    if out["same"] is None:
        raise_skip(f"warp is not available: {steps['Camera']}")
    assert steps["Camera"] == ["PIL", "warp"] and out["same"] is True        # emray came, pxr did not


def raise_skip(reason):
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest

        pytest.skip(reason)
    raise Skip(reason)


def test_every_offered_name_exists_and_is_the_object_of_its_module():
    import importlib

    import ncntm.viz as viz

    assert len(set(viz.__all__)) == len(viz.__all__)
    for name in viz.__all__:                                        # imported with the package
        assert name in vars(viz), name
    assert set(viz.MODULES) == {path.stem for path in PACKAGE.glob("*.py")} - {"__init__"}
    assert not set(viz.LAZY) & set(viz.__all__) and not set(viz.LAZY) & set(viz.MODULES)
    assert set(dir(viz)) == set(viz.__all__) | set(viz.MODULES) | set(viz.LAZY)
    missing = []
    for name, module_name in viz.LAZY.items():
        try:
            module = importlib.import_module(f"ncntm.viz.{module_name}")
        except ImportError as error:                               # e.g. a plain OpenUSD environment without warp
            missing.append(f"{module_name}: {error}")
            continue
        assert getattr(viz, name) is getattr(module, name), name
    for module_name in viz.MODULES:
        if not any(m.startswith(module_name + ":") for m in missing):
            assert getattr(viz, module_name) is sys.modules[f"ncntm.viz.{module_name}"]
    try:
        viz.no_such_name
    except AttributeError as error:
        assert "no_such_name" in str(error)
    else:
        raise AssertionError("an unknown name must raise AttributeError")
    if missing:
        raise_skip("not importable here: " + "; ".join(sorted(set(missing))))


# ---- portability ---------------------------------------------------------------------------------
def string_constants(tree):
    """(line number, text) of every string in the code that is not a docstring."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    return [(node.lineno, node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings]


def test_no_absolute_path_of_a_user_or_a_machine():
    """Paths are derived from __file__, from environment variables or from tempfile.
    Nowhere, not even in a docstring or a comment (an example there would be copied), is a home
    directory named.  No string of the code names a temporary or mounted directory either; a
    docstring may (kit.py says where Kit keeps its temporary files when left alone)."""
    home = re.compile(r"(?<![\w.~])(/home/|/Users/|/root/)\w|[A-Za-z]:\\\\\w")
    fixed_place = re.compile(r"(?<![\w.])(/tmp/|/mnt/|/media/|/scratch/|/workspace/)\w")
    found = []
    for path in SOURCES + sorted(PACKAGE.glob("apps/*")):
        text = path.read_text()
        for number, line in enumerate(text.splitlines(), start=1):
            if home.search(line):
                found.append(f"{path.relative_to(REPO)}:{number}: {line.strip()[:120]}")
        if path.suffix == ".py":
            for number, value in string_constants(ast.parse(text)):
                if fixed_place.search(value):
                    found.append(f"{path.relative_to(REPO)}:{number}: {value.strip()[:120]!r}")
    assert not found, "\n" + "\n".join(found)


def test_no_hard_coded_device():
    """The Warp device is the ``device`` argument, else NCNTM_DEVICE, else Warp's choice.  No string
    in the package is a device name; the tests name one only where Kit has restricted CUDA to one
    GPU (test_kit.py: that GPU is then cuda:0 by definition) or to say that it must be absent."""
    device = re.compile(r"cuda(:\d+)?")
    found = []
    for path in SOURCES:
        tree = ast.parse(path.read_text())
        absent = {id(node.left) for node in ast.walk(tree)
                  if isinstance(node, ast.Compare) and any(isinstance(op, ast.NotIn) for op in node.ops)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and device.fullmatch(node.value.strip()):
                if id(node) not in absent and path.name != "test_kit.py":
                    found.append(f"{path.relative_to(REPO)}:{node.lineno}: {node.value!r}")
    assert not found, "\n" + "\n".join(found)
    emray = (PACKAGE / "emray.py").read_text()
    assert "NCNTM_DEVICE" in emray and "get_preferred_device" in emray


def test_only_the_listed_packages_are_imported():
    standard = set(sys.stdlib_module_names)
    found = {}
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top not in standard and top not in ALLOWED:
                    found.setdefault(top, []).append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not found, found
    # the package itself (not its tests) needs fewer: Isaac Sim only in kit.py, pytest nowhere
    for path in sorted(PACKAGE.glob("*.py")):
        tops = {(n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0]
                for n in ast.walk(ast.parse(path.read_text()))
                if (isinstance(n, ast.ImportFrom) and n.level == 0 and n.module) or isinstance(n, ast.Import)
                for a in (n.names if isinstance(n, ast.Import) else [None])}
        assert "pytest" not in tops, path.name
        if path.name != "kit.py":
            assert not tops & {"isaacsim", "omni", "carb"}, path.name
        if path.name in ("grid.py", "types.py", "samples.py", "convert.py", "figure.py", "kit.py", "__init__.py"):
            assert "warp" not in tops, path.name                    # these run without a ray caster
        if path.name in ("grid.py", "types.py", "samples.py", "convert.py", "figure.py", "emray.py", "__init__.py"):
            assert "pxr" not in tops, path.name                     # and these without OpenUSD


def test_heavy_imports_of_the_numpy_modules_are_inside_functions():
    """samples.py and convert.py import h5py, scipy, zmesh, kimimaro and skimage where they are
    used, never at the top of the module: that is what keeps ``import ncntm.viz`` cheap."""
    for name in ("grid.py", "types.py", "samples.py", "convert.py", "__init__.py"):
        tree = ast.parse((PACKAGE / name).read_text())
        for node in tree.body:                                      # module level only
            if isinstance(node, ast.Import):
                tops = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                tops = {node.module.split(".")[0]}
            else:
                continue
            assert tops <= set(sys.stdlib_module_names) | {"numpy"}, (name, node.lineno, tops)


if __name__ == "__main__":
    failed = []
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"PASS  {name}")
            except Skip as reason:
                print(f"SKIP  {name}: {reason}")
            except Exception:
                traceback.print_exc()
                print(f"FAIL  {name}")
                failed.append(name)
    print(f"\n{len(failed)} failed" if failed else "\nall tests passed")
    sys.exit(1 if failed else 0)
