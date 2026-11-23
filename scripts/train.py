#!/usr/bin/env python
"""
ncntm pretraining entry point.

Loads a Hydra config, builds the joint datamodule + Cosmos3-Edge Lightning
module + trainer, then calls ``trainer.fit(...)``.

Examples
--------
    python scripts/train.py --config-name pretrain
    python scripts/train.py --config-name pretrain training.devices=1 training.fast_dev_run=true
    python scripts/train.py --config-name pretrain training.resume_from_checkpoint=<run_dir>/checkpoints/last.ckpt
    python scripts/train.py --config-name pretrain +ckpt_path=<run_dir>/checkpoints/last.ckpt
"""

from __future__ import annotations

import collections
import datetime
import os
import subprocess
import sys
import warnings

_COLLECTIVE_TIMEOUT_MIN = 30

# NCCL / torch.distributed env vars must be set before Lightning initialises
# torch.distributed, including in re-imported child processes.
os.environ.setdefault(
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", str(_COLLECTIVE_TIMEOUT_MIN * 60)
)
os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2048")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
# NVLink-SHARP reductions raise Xid 145 on B300 / NVSwitch nodes at the first
# bf16 all-reduce; Ring + Simple avoids that path.
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("NCCL_ALGO", "Ring")
os.environ.setdefault("NCCL_PROTO", "Simple")
# Must be set before ``import torch``: the native Triton JIT bmm kernel fails
# to assemble for sm_103 on this stack.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

from pathlib import Path

# ncntm is not pip-installed; make the repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _prefer_bundled_cudnn() -> None:
    """Re-exec so the cuDNN shipped with the torch wheel wins over
    ``/usr/local/cuda/lib64`` (a mismatched cuDNN fails the first Wan VAE
    ``conv3d`` with ``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED``).  glibc reads
    ``LD_LIBRARY_PATH`` only at process start."""
    lib = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cudnn"
        / "lib"
    )
    if not (lib / "libcudnn_engines_runtime_compiled.so.9").is_file():
        return
    lib_s = str(lib)
    parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if parts and parts[0] == lib_s:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join([lib_s, *(p for p in parts if p != lib_s)])
    print(
        f"Restarting so cuDNN loads from {lib_s} "
        "(the cuDNN on LD_LIBRARY_PATH does not match this torch wheel).",
        file=sys.stderr,
    )
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _py314_hydra_help() -> None:
    """argparse >= 3.14 rejects Hydra 1.3's non-str lazy help objects."""
    if sys.version_info < (3, 14):
        return
    import argparse

    original = argparse.ArgumentParser.add_argument

    def add_argument(self, *args, **kwargs):
        help_text = kwargs.get("help")
        if (
            help_text is not None
            and help_text is not argparse.SUPPRESS
            and not isinstance(help_text, str)
        ):
            kwargs["help"] = str(help_text)
        return original(self, *args, **kwargs)

    argparse.ArgumentParser.add_argument = add_argument


if __name__ == "__main__":
    _prefer_bundled_cudnn()

from typing import Any, Dict, List, Optional

import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata
from omegaconf.nodes import AnyNode, ValueNode
from rich.console import Console

console = Console()


def _install_runtime_patches() -> None:
    """Safe-first ``torch.load``, quieter warnings, TF32, and no cuDNN SDPA."""
    torch.serialization.add_safe_globals([
        Any,
        dict, list, tuple, set,
        collections.defaultdict,
        DictConfig, ListConfig, ContainerMetadata, ValueNode, AnyNode,
    ])

    _orig_torch_load = torch.load

    def _torch_load_safe_first(*args: Any, **kwargs: Any) -> Any:
        if "weights_only" in kwargs:
            return _orig_torch_load(*args, **kwargs)
        try:
            return _orig_torch_load(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on ckpt contents
            warnings.warn(
                "torch.load safe unpickler rejected a global "
                f"({type(exc).__name__}: {exc}); retrying with "
                "weights_only=False. Only load checkpoints you trust.",
                stacklevel=2,
            )
            kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_safe_first  # type: ignore[assignment]

    warnings.filterwarnings("ignore", message=r".*isinstance.*LeafSpec.*is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*AccumulateGrad.*stream.*mismatch.*")
    warnings.filterwarnings("ignore", message=".*lru_cache.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*sync_dist=True.*when logging on epoch level.*")
    warnings.filterwarnings("ignore", message=r".*module.*in eval mode at the start of training.*")

    torch.set_float32_matmul_precision("high")

    # cuDNN Frontend SDPA builds no execution plan on sm_103; use FlashAttention.
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def build_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    from ncntm.datamodules import Joint3DDataModule

    d = cfg.data
    return Joint3DDataModule(
        data_root=d.get("data_root", "data"),
        patch_size=tuple(d.get("patch_size", (256, 256, 256))),
        pixel_size=tuple(d.get("pixel_size", (4.0, 4.0, 4.0))),
        rand_transpose_zyx=bool(d.get("rand_transpose_zyx", False)),
        branches=OmegaConf.to_container(d.get("branches", {}), resolve=True),
        degrade=OmegaConf.to_container(d.get("degrade", {}), resolve=True),
        val_volumes=OmegaConf.to_container(d.get("val_volumes"), resolve=True)
        if d.get("val_volumes") is not None else None,
        num_workers=int(d.get("num_workers", 8)),
        pin_memory=bool(d.get("pin_memory", True)),
        persistent_workers=bool(d.get("persistent_workers", True)),
        prefetch_factor=int(d.get("prefetch_factor", 4)),
        num_samples=int(d.get("num_samples", 8000)),
        val_num_samples=int(d.get("val_num_samples", 16)),
        val_batch_size=int(d.get("val_batch_size", 1)),
        ssl_min_foreground=float(d.get("ssl_min_foreground", 0.0)),
        ssl_min_std=float(d.get("ssl_min_std", 0.0)),
        ssl_min_autocorr=float(d.get("ssl_min_autocorr", 0.0)),
        balance=str(d.get("balance", "resolution")),
        within_dataset=str(d.get("within_dataset", "tissue")),
        val_balance=str(d.get("val_balance", "resolution")),
        dataset_weights=(
            OmegaConf.to_container(d.get("dataset_weights"), resolve=True)
            if d.get("dataset_weights") is not None else {}
        ),
        seed=int(cfg.get("seed", 0)),
    )


def build_module(cfg: DictConfig) -> pl.LightningModule:
    from ncntm.modules import JointEdge3DModule

    model_cfg = dict(cfg.get("model", {}))
    model_cfg.pop("type", None)
    loss_cfg = dict(cfg.get("loss", {}))
    loss_cfg.pop("type", None)
    return JointEdge3DModule(
        model_config=model_cfg,
        optimizer_config=dict(cfg.get("optimizer", {})),
        loss_config=loss_cfg,
        training_config=dict(cfg.get("training", {})),
    )


def setup_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    from ncntm.callbacks import (
        CudaEmptyCacheCallback,
        CudaMemoryLoggerCallback,
        Joint3DImageLogger,
        TrainableWeightsCheckpoint,
    )

    output_dir = cfg.get("output_dir", "outputs")
    callback_cfg = cfg.get("callbacks", {})
    callbacks: List[pl.Callback] = []

    if callback_cfg.get("cuda_empty_cache_before_val", False):
        callbacks.append(CudaEmptyCacheCallback())

    mem_cfg = callback_cfg.get("memory_logger", {})
    if mem_cfg.get("enabled", True):
        mem_every = mem_cfg.get("log_every_n_steps")
        if mem_every is None:
            mem_every = cfg.get("training", {}).get("log_every_n_steps", 50)
        callbacks.append(CudaMemoryLoggerCallback(log_every_n_steps=int(mem_every)))

    ckpt_cfg = callback_cfg.get("checkpoint", {})
    if ckpt_cfg.get("enabled", True):
        callbacks.append(ModelCheckpoint(
            dirpath=ckpt_cfg.get("dirpath") or str(Path(output_dir) / "checkpoints"),
            filename=ckpt_cfg.get("filename", "{epoch:02d}-{val/automatic/loss:.4f}"),
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            monitor=ckpt_cfg.get("monitor", "val/automatic/loss"),
            mode=ckpt_cfg.get("mode", "min"),
            save_last=ckpt_cfg.get("save_last", True),
            verbose=ckpt_cfg.get("verbose", True),
            auto_insert_metric_name=False,
        ))

    wcfg = callback_cfg.get("weights_checkpoint", {}) or {}
    if wcfg.get("enabled", False):
        callbacks.append(TrainableWeightsCheckpoint(
            dirpath=wcfg.get("dirpath") or str(Path(output_dir) / "checkpoints"),
            every_n_epochs=int(wcfg.get("every_n_epochs", 20)),
            save_top_k=int(wcfg.get("save_top_k", -1)),
        ))

    if callback_cfg.get("lr_monitor", {}).get("enabled", True):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    img_cfg = callback_cfg.get("image_logger", {})
    if img_cfg.get("enabled", True):
        callbacks.append(Joint3DImageLogger(
            every_n_epochs=img_cfg.get("every_n_epochs", 1),
            max_images=img_cfg.get("max_images", 4),
            spatial_dims=3,
        ))

    callbacks.append(RichProgressBar())
    callbacks.append(ModelSummary(max_depth=2))
    return callbacks


def setup_strategy(cfg: DictConfig):
    strategy_name = cfg.training.get("strategy", "auto")
    if strategy_name == "ddp":
        # The cold first batch (lazy 3-D reads + augmentation on fresh
        # workers) can delay a rank's first collective by minutes.
        return DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            timeout=datetime.timedelta(minutes=_COLLECTIVE_TIMEOUT_MIN),
        )
    return strategy_name


def build_trainer(cfg: DictConfig, *, callbacks: List[pl.Callback], logger: Any) -> pl.Trainer:
    t = cfg.training
    return pl.Trainer(
        max_epochs=t.get("max_epochs", 100),
        accelerator=t.get("accelerator", "auto"),
        devices=t.get("devices", 1),
        strategy=setup_strategy(cfg),
        precision=t.get("precision", "bf16-true"),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=t.get("log_every_n_steps", 50),
        gradient_clip_val=t.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=t.get("accumulate_grad_batches", 1),
        limit_train_batches=t.get("limit_train_batches", 1.0),
        limit_val_batches=t.get("limit_val_batches", 1.0),
        val_check_interval=t.get("val_check_interval", 1.0),
        check_val_every_n_epoch=t.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps=t.get("num_sanity_val_steps", 2),
        enable_progress_bar=t.get("enable_progress_bar", True),
        enable_model_summary=t.get("enable_model_summary", True),
        deterministic=t.get("deterministic", False),
        benchmark=t.get("benchmark", True),
        fast_dev_run=t.get("fast_dev_run", False),
        # The joint round-robin batch_sampler is not a BatchSampler subclass,
        # so Lightning cannot inject a DistributedSampler.
        use_distributed_sampler=t.get("use_distributed_sampler", False),
    )


def _summarise_keys_by_prefix(keys: List[str], depth: int = 2) -> Dict[str, int]:
    counter: Dict[str, int] = collections.OrderedDict()
    for k in keys:
        prefix = ".".join(k.split(".")[:depth]) + (".*" if k.count(".") >= depth else "")
        counter[prefix] = counter.get(prefix, 0) + 1
    return counter


def _resolve_checkpoint(cfg: DictConfig, module: pl.LightningModule) -> Optional[str]:
    """Full resume (``training.resume_from_checkpoint``) returns the path for
    ``trainer.fit``; a weights-only warm start (``+ckpt_path``, optionally
    filtered by ``ckpt_path_skip_prefixes`` / ``ckpt_path_only_prefixes``)
    loads in place and returns ``None``."""
    resume_ckpt = cfg.training.get("resume_from_checkpoint")
    weights_only_ckpt = cfg.get("ckpt_path")
    if resume_ckpt and weights_only_ckpt:
        raise ValueError(
            "Use either training.resume_from_checkpoint (full Lightning resume) "
            "or +ckpt_path= (weights-only warm start), not both."
        )
    if resume_ckpt:
        console.log(f"Full Lightning resume from: {resume_ckpt}")
        return str(resume_ckpt)
    if not weights_only_ckpt:
        return None

    skip_prefixes = list(cfg.get("ckpt_path_skip_prefixes") or [])
    only_prefixes = list(cfg.get("ckpt_path_only_prefixes") or [])
    if skip_prefixes and only_prefixes:
        raise ValueError("Use either ckpt_path_skip_prefixes or ckpt_path_only_prefixes, not both.")

    console.log(f"Loading model weights from checkpoint: {weights_only_ckpt}")
    ckpt = torch.load(weights_only_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if skip_prefixes:
        dropped = [k for k in state_dict if any(k.startswith(p) for p in skip_prefixes)]
        for k in dropped:
            state_dict.pop(k)
        console.log(f"  Dropped {len(dropped)} keys: {dict(_summarise_keys_by_prefix(dropped))}")
    elif only_prefixes:
        kept = {k for k in state_dict if any(k.startswith(p) for p in only_prefixes)}
        for k in [k for k in state_dict if k not in kept]:
            state_dict.pop(k)
        console.log(f"  Kept {len(kept)} keys: {dict(_summarise_keys_by_prefix(sorted(kept)))}")

    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    n_target = len(module.state_dict())
    n_loaded = n_target - len(missing)
    if n_target and n_loaded < 0.5 * n_target:
        warnings.warn(
            f"Warm start loaded only {n_loaded}/{n_target} of the model's parameters; "
            "check +ckpt_path and the prefix filters.",
            stacklevel=2,
        )
    console.log(
        f"Model weights loaded (optimiser state skipped): {n_loaded} loaded, "
        f"{len(missing)} missing, {len(unexpected)} unexpected."
    )
    return None


def _save_trainer_checkpoint(trainer: pl.Trainer, path: Path) -> None:
    """Every rank must call this: ``save_checkpoint`` ends in a barrier."""
    if trainer.is_global_zero:
        path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(path))


def run_fit_with_recovery(
    trainer: pl.Trainer,
    module: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    *,
    ckpt_path: Optional[str],
    output_dir: Path,
) -> bool:
    try:
        trainer.fit(module, datamodule, ckpt_path=ckpt_path)
        return True
    except KeyboardInterrupt:
        console.log("[yellow]Training interrupted by user.[/yellow]")
        return False
    except Exception as exc:
        console.log(f"[red]Training failed: {exc}[/red]")
        recovery = output_dir / "checkpoints" / "crash_recovery.ckpt"
        try:
            _save_trainer_checkpoint(trainer, recovery)
            if trainer.is_global_zero:
                console.log(f"Recovery checkpoint written to {recovery}")
        except Exception as save_err:  # noqa: BLE001 -- best effort
            if trainer.is_global_zero:
                console.log(f"[red]Could not save recovery checkpoint: {save_err}[/red]")
        raise


def _write_run_provenance(run_dir: Path) -> None:
    """Write ``git_sha.txt`` and ``git_diff.patch`` (local rank 0 only)."""
    if os.environ.get("LOCAL_RANK", "0") not in ("", "0"):
        return
    repo = Path(__file__).resolve().parents[1]

    def _git(*args: str) -> str:
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo), *args],
                check=False, capture_output=True, text=True,
            )
        except OSError as exc:
            return f"git failed: {exc}\n"
        if proc.returncode != 0:
            return f"git {' '.join(args)} failed: {(proc.stderr or proc.stdout).strip()}\n"
        return proc.stdout

    (run_dir / "git_sha.txt").write_text(_git("rev-parse", "HEAD"))
    (run_dir / "git_diff.patch").write_text(_git("diff", "HEAD"))


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    _install_runtime_patches()

    console.rule("ncntm - Cosmos3-Edge pretraining")
    console.print(OmegaConf.to_yaml(cfg))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg.get("output_dir", "outputs")) / f"{timestamp}_{cfg.get('experiment_name', 'run')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_provenance(run_dir)
    OmegaConf.update(cfg, "output_dir", str(run_dir), force_add=True)
    console.log(f"Run directory: {run_dir}")

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    datamodule = build_datamodule(cfg)
    module = build_module(cfg)
    if getattr(getattr(module, "model", None), "_backbone_loaded", None) is False:
        console.log(
            "[yellow]WARNING: pretrained backbone was NOT loaded - training from "
            "random init. Check the HuggingFace cache and network access.[/yellow]"
        )

    callbacks = setup_callbacks(cfg)
    logger = TensorBoardLogger(
        save_dir=str(Path(cfg.output_dir) / "logs"),
        name=cfg.get("experiment_name", "ncntm"),
        version=None,
    )
    trainer = build_trainer(cfg, callbacks=callbacks, logger=logger)

    fit_ckpt_path = _resolve_checkpoint(cfg, module)
    completed = run_fit_with_recovery(
        trainer, module, datamodule,
        ckpt_path=fit_ckpt_path,
        output_dir=Path(cfg.output_dir),
    )

    final_path = Path(cfg.output_dir) / "checkpoints" / "final_model.ckpt"
    if trainer.is_global_zero and not completed:
        console.log("[yellow]WARNING: saving checkpoint from an interrupted run.[/yellow]")
    _save_trainer_checkpoint(trainer, final_path)
    if trainer.is_global_zero:
        console.log(f"Final model saved: {final_path}")


if __name__ == "__main__":
    _py314_hydra_help()
    main()
