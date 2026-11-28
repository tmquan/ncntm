"""CPU probe of the sft crop sampler: baseline vs large-object oversampling.

For every sft training volume of a config this builds the real
``LazyVolDataset`` (same native patch, same gates as the datamodule, no
transform) and draws crops with ``large_object_prob`` = 0 (baseline), 1.0
(every draw targets a large object) and the config's / ``--prob`` mix.  It
reports, per volume, the largest-instance fraction of the accepted crops
(mean and share >= 0.5 / >= 0.8), seconds per crop, and how often the
large-object stage ran out of retries and fell back to a normal draw.

    python scripts/probe_large_object_sampling.py --config-name sft_alldata_r2 \
        --n 48 --prob 0.25 --min-frac 0.5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("NCNTM_DATA_DIR", str(REPO / "data"))
sys.path.insert(0, str(REPO))

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from ncntm.datamodules.joint3d import _scaled  # noqa: E402
from ncntm.datasets.lazy import LazyVolDataset  # noqa: E402

class _Counter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.n = 0

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        if "gate exhausted" in record.getMessage():
            self.n += 1


def _largest_frac(label: np.ndarray) -> float:
    ids, counts = np.unique(label, return_counts=True)
    fg = counts[ids != 0]
    return float(fg.max()) / float(fg.sum()) if fg.size else 1.0


def _probe(ds: LazyVolDataset, n: int, seed0: int) -> dict:
    counter = _Counter()
    logging.getLogger("ncntm.datasets.lazy").addHandler(counter)
    fr, t0 = [], time.time()
    for i in range(n):
        fr.append(_largest_frac(ds.fetch(seed0 + i)["label"]))
    dt = (time.time() - t0) / n
    logging.getLogger("ncntm.datasets.lazy").removeHandler(counter)
    fr = np.asarray(fr)
    return {
        "mean": float(fr.mean()),
        "ge50": float((fr >= 0.5).mean()),
        "ge80": float((fr >= 0.8).mean()),
        "sec": dt,
        "exhausted": counter.n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="sft_alldata")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--prob", type=float, default=0.25)
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--only", default="", help="substring filter on vol name")
    args = ap.parse_args()

    with initialize_config_dir(str(REPO / "configs"), version_base=None):
        cfg = compose(config_name=args.config_name)
    # Configs use ${hydra:runtime.cwd}; outside a hydra run it resolves to the repo.
    OmegaConf.register_new_resolver("hydra", lambda path: str(REPO), replace=True)
    d = cfg.data
    fine_nm = float(d.pixel_size[0])
    fine = tuple(int(x) for x in d.patch_size)
    gates = dict(
        min_foreground=float(d.get("sft_min_foreground", 0.0)),
        image_min_foreground=float(d.get("sft_min_foreground", 0.0)),
        min_instances=int(d.get("sft_min_instances", 0)),
        max_inst_frac=float(d.get("sft_max_inst_frac", 0.0)),
    )
    print(f"gates: {gates}  patch(fine)={fine}  min_frac={args.min_frac}")
    hdr = (f"{'volume':58s} {'mode':8s} {'mean':>5s} {'>=.5':>5s} {'>=.8':>5s} "
           f"{'s/crop':>6s} {'fallbk':>6s}")
    print(hdr)
    for vol in d.branches.sft.volumes:
        if args.only and args.only not in str(vol.vol):
            continue
        native = tuple(float(x) for x in vol.native_resolution)
        patch = _scaled(fine, fine_nm, native)
        spec = {"vol": vol.vol, "seg": vol.seg,
                "root": str(vol.root)}
        for mode, prob in (("base", 0.0), ("large", 1.0), (f"mix{args.prob:g}", args.prob)):
            ds = LazyVolDataset(
                root_dir=spec["root"], volumes=[spec], patch_size=patch,
                transform=None, num_samples=10**6, normalize=False,
                large_object_prob=prob, large_object_min_frac=args.min_frac,
                quiet=True, **gates,
            )
            r = _probe(ds, args.n, seed0=1000)
            print(f"{str(vol.vol)[:58]:58s} {mode:8s} {r['mean']:5.2f} {r['ge50']:5.2f} "
                  f"{r['ge80']:5.2f} {r['sec']:6.2f} {r['exhausted']:6d}", flush=True)


if __name__ == "__main__":
    main()
