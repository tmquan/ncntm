#!/usr/bin/env python
"""Build AC3_cc_padded / AC4_cc_padded: the SNEMI3D stacks with real surrounding EM.

Each padded volume is the same tissue as ``AC{3,4}_cc_inputs.h5`` with a margin of
``--margin`` (default 0.5) of the stack size on both sides of every axis, so the labeled
stack sits in the middle, as in the CREMI ``+_padded`` volumes.  The margin is real
``kasthuri2015/em/cc`` EM (BossDB, 6 x 6 x 30 nm, mip 0), not blank padding, read from
the same layer the unpadded files came from::

    AC3  100 x 1024 x 1024  ->  200 x 2048 x 2048   core at [50:150, 512:1536, 512:1536]
    AC4  100 x 1024 x 1024  ->  200 x 2048 x 2048   core at [50:150, 512:1536, 512:1536]

The padded volume is stored in the same orientation as the unpadded file (AC3: z
reversed; AC4: z, y and x reversed), and the script checks that the core is
byte-identical to ``AC{3,4}_cc_inputs.h5`` before writing.  Image only: the labels stay
in ``AC{3,4}_labels.h5`` and cover the core.

Usage (from the repo root; needs ``cloudvolume`` and ``h5py``)::

    python scripts/make_snemi_padded.py                  # AC3 and AC4
    python scripts/make_snemi_padded.py --names AC4 --margin 0.5

Output goes to ``$NCNTM_DATA_DIR/SNEMI3D`` (default ``./data/SNEMI3D``) as
``AC{3,4}_cc_padded_inputs.h5`` (HDF5 key ``main``, uint8, (z, y, x), (16, 256, 256)
chunks, uncompressed) with the loader's ``.norm.json`` sidecar.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

CC_PATH = "s3://bossdb-open-data/kasthuri2015/em/cc"

# BossDB cc numbering, half-open [start, stop) in x, y, z (docs/DATASETS.md lists the
# z range as first-last slice, inclusive).  ``flip`` is the stored orientation (z, y, x)
# relative to BossDB, from notebooks/data_SNEMI3D.ipynb.
BOXES = {
    "AC3": {"x": (5472, 6496), "y": (8712, 9736), "z": (1155, 1255), "flip": (True, False, False)},
    "AC4": {"x": (4400, 5424), "y": (5440, 6464), "z": (1099, 1199), "flip": (True, True, True)},
}
ORIENTATION = {
    "AC3": "z reversed (matches AC3_em_inputs.h5)",
    "AC4": "z, y, x reversed (matches AC4_em_inputs.h5)",
}
CHUNKS = (16, 256, 256)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--names", nargs="+", default=["AC3", "AC4"], choices=sorted(BOXES))
    p.add_argument("--margin", type=float, default=0.5,
                   help="margin per side as a fraction of the stack size on that axis (default 0.5)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: $NCNTM_DATA_DIR/SNEMI3D, else ./data/SNEMI3D")
    p.add_argument("--parallel", type=int, default=1,
                   help="CloudVolume download processes (1 = threads only; a process pool stalled in testing)")
    p.add_argument("--force", action="store_true", help="overwrite an existing output file")
    return p.parse_args()


def _norm_probe(vol: np.ndarray, probe: int = 512, n_probes: int = 5) -> tuple:
    """Same centred-probe min/max as ``LazyVolDataset._compute_norm_params``."""
    z_dim = vol.shape[0]
    zs = sorted({max(0, min(i, z_dim - 1)) for i in (0, z_dim // 4, z_dim // 2, 3 * z_dim // 4, z_dim - 1)})[:n_probes]
    vmin, vmax = float("inf"), float("-inf")
    for zi in zs:
        sl = [slice(zi, zi + 1)]
        for n in vol.shape[1:]:
            half, centre = min(probe, n) // 2, n // 2
            sl.append(slice(max(0, centre - half), min(n, centre + half)))
        patch = vol[tuple(sl)].astype(np.float32)
        vmin, vmax = min(vmin, float(patch.min())), max(vmax, float(patch.max()))
    return vmin, vmax


def build(name: str, margin: float, out_dir: Path, parallel: int, force: bool) -> None:
    from cloudvolume import CloudVolume

    box = BOXES[name]
    core_path = out_dir / f"{name}_cc_inputs.h5"
    out_path = out_dir / f"{name}_cc_padded_inputs.h5"
    if out_path.exists() and not force:
        print(f"{name}: {out_path.name} exists (use --force to overwrite)")
        return

    with h5py.File(core_path, "r") as f:
        core = f["main"][...]
    core_shape = core.shape                                              # (z, y, x)
    pad = tuple(int(round(margin * n)) for n in core_shape)              # (z, y, x) voxels per side
    lo = {"x": box["x"][0] - pad[2], "y": box["y"][0] - pad[1], "z": box["z"][0] - pad[0]}
    hi = {"x": box["x"][1] + pad[2], "y": box["y"][1] + pad[1], "z": box["z"][1] + pad[0]}
    expect = (core_shape[0] + 2 * pad[0], core_shape[1] + 2 * pad[1], core_shape[2] + 2 * pad[2])
    print(f"{name}: core {core_shape}, margin (z,y,x) {pad} -> padded {expect}")
    print(f"  BossDB cc box  x {lo['x']}-{hi['x']}  y {lo['y']}-{hi['y']}  z {lo['z']}-{hi['z']}")

    vol = CloudVolume(CC_PATH, mip=0, use_https=True, fill_missing=False, bounded=True,
                      parallel=parallel, progress=True)
    size = [int(s) for s in vol.shape[:3]]
    if min(lo.values()) < 0 or hi["x"] > size[0] or hi["y"] > size[1] or hi["z"] > size[2]:
        sys.exit(f"{name}: padded box leaves the BossDB volume {size}; use a smaller --margin")

    t0 = time.time()
    arr = np.asarray(vol[lo["x"]:hi["x"], lo["y"]:hi["y"], lo["z"]:hi["z"]])
    arr = np.transpose(np.squeeze(arr, axis=-1), (2, 1, 0))             # (x, y, z) -> (z, y, x)
    print(f"  downloaded {arr.shape} in {time.time() - t0:.0f} s")
    assert arr.shape == expect, (arr.shape, expect)

    # BossDB orientation -> the orientation of the unpadded file.
    arr = arr[tuple(slice(None, None, -1) if f else slice(None) for f in box["flip"])]

    inner = tuple(slice(p, p + n) for p, n in zip(pad, core_shape))
    if not np.array_equal(arr[inner], core):
        diff = np.abs(arr[inner].astype(np.int16) - core.astype(np.int16))
        sys.exit(f"{name}: core of the padded volume != {core_path.name} "
                 f"(equal on {np.mean(diff == 0):.1%}, mean |diff| {diff.mean():.2f}); nothing written")
    print(f"  core {tuple(s.start for s in inner)}+{core_shape} is byte-identical to {core_path.name}")

    tmp = out_path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        d = f.create_dataset("main", data=arr, chunks=CHUNKS)
        d.attrs["cc_box_xyz"] = (f"x {lo['x']}-{hi['x']}, y {lo['y']}-{hi['y']}, z {lo['z']}-{hi['z'] - 1}")
        d.attrs["core_box_xyz"] = (f"x {box['x'][0]}-{box['x'][1]}, y {box['y'][0]}-{box['y'][1]}, "
                                   f"z {box['z'][0]}-{box['z'][1] - 1}")
        d.attrs["core_offset_zyx"] = np.array([s.start for s in inner])
        d.attrs["core_shape_zyx"] = np.array(core_shape)
        d.attrs["margin_zyx_voxels"] = np.array(pad)
        d.attrs["margin_fraction"] = margin
        d.attrs["mip"] = 0
        d.attrs["orientation"] = ORIENTATION[name] + "; same orientation as the unpadded file"
        d.attrs["source"] = CC_PATH
        d.attrs["voxel_size_nm_zyx"] = np.array([30, 6, 6])
    os.replace(tmp, out_path)

    # Same intensity scaling as the unpadded stack, so one pixel normalises identically
    # in both; fall back to the loader's centred-probe estimate if it has no sidecar.
    core_norm = Path(str(core_path) + ".norm.json")
    if core_norm.exists():
        with open(core_norm) as f:
            norm = json.load(f)
        vmin, vmax = float(norm["min"]), float(norm["max"])
    else:
        vmin, vmax = _norm_probe(arr)
    with open(str(out_path) + ".norm.json", "w") as f:
        json.dump({"min": vmin, "max": vmax}, f)
    nz = float(np.count_nonzero(arr)) / arr.size
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB), non-zero {nz:.2%}, "
          f"min/max {arr.min()}/{arr.max()}, norm {vmin:.0f}/{vmax:.0f}")


def main() -> None:
    a = _args()
    root = Path(os.environ.get("NCNTM_DATA_DIR", Path.cwd() / "data"))
    out_dir = a.out_dir or root / "SNEMI3D"
    for name in a.names:
        build(name, a.margin, out_dir, a.parallel, a.force)


if __name__ == "__main__":
    main()
