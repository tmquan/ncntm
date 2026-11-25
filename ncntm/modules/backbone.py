"""
Lightning-module base for 3-D volumetric backbones.

Specialisation of :class:`~ncntm.modules.base.BaseCircuitModule` for the
VoxelDiT2-3D backbone.  The base ``BaseCircuitModule`` owns training,
evaluation and logging; this module adds:

* HuggingFace token handling (kept out of ``save_hyperparameters``)
* an integer ``freeze_dit_backbone`` warm-up, thawed in
  :meth:`on_train_epoch_start`
* fused-norm NaN/Inf gradient guard (one device->host sync per step
  via :func:`torch._foreach_norm` rather than one per parameter --
  see :meth:`BackboneModule.configure_gradient_clipping`)
* backbone-specific AdamW learning rate and param-group split
  (``model.dit.*`` at ``dit_backbone_lr`` vs everything else at ``lr``)
  in :meth:`configure_optimizers`; empty groups are filtered out
  before AdamW sees them.

Only the **automatic** training mode is supported (the volume is the
only input).

Subclasses define :attr:`_model_cls` / :attr:`_loss_cls` and implement
:meth:`_build_model`.
"""

import logging
import os
import types
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from ncntm.losses import HEAD_CHANNELS
from ncntm.modules.base import BaseCircuitModule
from ncntm.optim.master_adamw import MasterAdamW, master_dtype_from_config

logger = logging.getLogger(__name__)

# Log level for the per-step non-finite-gradient guard in
# ``BackboneModule.configure_gradient_clipping``.  Default DEBUG keeps
# it silent under the usual INFO/WARNING root logger. bf16-true with
# ``gradient_clip_val=1.0`` can still trip the guard, and a raised level
# emits one line per DDP rank. Override via
# ``NCNTM_GRAD_GUARD_LOG=warning`` (or ``info`` / ``error``) when
# investigating loss spikes.
_GRAD_GUARD_LOG_FN = getattr(
    logger,
    os.environ.get("NCNTM_GRAD_GUARD_LOG", "debug").lower(),
    logger.debug,
)


class BackboneModule(BaseCircuitModule):
    """Lightning base for the VoxelDiT2-3D backbone.

    Subclasses **must** define :attr:`_model_cls` and :attr:`_loss_cls`.

    Args:
        model_config: Forwarded to ``_model_cls`` (see
            :class:`ncntm.models.voxeldit2.VoxelDiT2Wrapper3D`).
        optimizer_config: Optimizer / scheduler settings.
        loss_config: Forwarded as ``**loss_config`` to ``_loss_cls``.
        training_config: Training behaviour (mutex_watershed, freeze schedule, ...).
    """

    _SPATIAL_DIMS = 3

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        model_config = dict(model_config or {})
        # ``hf_token`` must never be persisted into the checkpoint. Pop it out
        # BEFORE ``save_hyperparameters`` snapshots ``model_config``. Crucially,
        # ``save_hyperparameters`` keeps a *reference* to this dict, so we must
        # NOT re-insert the token into it afterwards (that leaked the token into
        # ``self.hparams`` and hence every saved checkpoint). Build the model
        # from a separate copy that carries the token instead.
        hf_token = model_config.pop("hf_token", None)
        # NOTE: this snapshots all four config dicts into the checkpoint's
        # ``hyper_parameters``. ``training_config`` therefore also carries
        # environment-specific keys (accelerator / devices / strategy /
        # resume_from_checkpoint / compile); treat those as informational
        # provenance, not a portable spec, when inspecting an old checkpoint.
        self.save_hyperparameters()
        build_config = dict(model_config)
        if hf_token is not None:
            build_config["hf_token"] = hf_token

        super().__init__(
            model_config=build_config,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            training_config=training_config,
            **kwargs,
        )

        logger.info(
            "Validation agglomerator: %s",
            repr(self.agglomerator),
        )

    # ------------------------------------------------------------------
    # Scheduled DiT thaw (``model.freeze_dit_backbone: N`` warm-up)
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """Thaw the DiT backbone when the configured warm-up epoch arrives.

        Wired to ``model.freeze_dit_backbone: N`` (non-negative int):
        the DiT stays frozen for epochs ``0..N-1`` and is unfrozen at
        the start of epoch ``N``.  ``configure_optimizers`` already
        included the DiT params in the optimizer's backbone group up
        front (see ``include_frozen_dit`` there), so the only work
        here is flipping ``requires_grad`` -- AdamW lazily allocates
        moment buffers on the first step that produces gradients, and
        the existing LR scheduler state is preserved verbatim.
        """
        super().on_train_epoch_start()
        thaw_epoch = getattr(self.model, "dit_thaw_epoch", None)
        if thaw_epoch is None:
            return
        if self.current_epoch < thaw_epoch:
            return
        if not self.model._freeze_dit_backbone:
            return
        logger.info(
            "Scheduled DiT thaw at epoch %d (freeze_dit_backbone: %d).",
            self.current_epoch, thaw_epoch,
        )
        self.model.unfreeze_dit_backbone()

    # ------------------------------------------------------------------
    # Optimizer (backbone vs heads split + NaN/Inf gradient zeroing)
    # ------------------------------------------------------------------

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None,
    ) -> None:
        """Zero every gradient when any is non-finite, then clip.

        Hot-path version of the previous per-parameter scan (which
        called ``p.grad.isnan().any()`` and ``.isinf().any()`` on every
        parameter and forced one device->host sync per call -- O(P) in
        the DiT, ~thousands of syncs per step).  This path
        builds the gradient list once, computes a fused stack of
        per-tensor L2 norms with ``torch._foreach_norm`` (single fused
        kernel over all tensors), and bails out only when
        ``torch.isfinite(norm_stack).all()`` is False -- one scalar
        sync per step instead of O(P).

        On a clean step the cost is one ``_foreach_norm`` call plus a
        single ``isfinite`` reduction + bool cast (a few microseconds);
        on a bad step we additionally zero every gradient with
        ``_foreach_zero_``, which is itself a fused kernel.

        Behaviour is otherwise identical to the previous path: on a
        non-finite batch every gradient is zeroed before the clipper
        runs, so the optimiser sees a no-op step rather than NaN
        weights.

        Non-finite-batch events are emitted at DEBUG level (silent by
        default). bf16-true with ``gradient_clip_val=1.0`` can still trip
        the guard, and the log line is emitted on every DDP rank. Raise the
        ``ncntm.modules.backbone`` logger to DEBUG
        (or set ``NCNTM_GRAD_GUARD_LOG=warning`` -- see
        :data:`_GRAD_GUARD_LOG_FN`) to surface them again.

        The DiT group's pre-clip L2 is recorded here for
        ``train/automatic/dit/grad_norm`` and the clip rate. That record
        does not change the clip.
        """
        self._note_dit_grad(optimizer, gradient_clip_val)
        grads = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad)
        if grads:
            norms = torch._foreach_norm(grads)
            norm_stack = torch.stack(norms)
            if not bool(torch.isfinite(norm_stack).all()):
                torch._foreach_zero_(grads)
                _GRAD_GUARD_LOG_FN(
                    "Zeroed all %d gradients at step %d (one or more "
                    "non-finite per-tensor norms).",
                    len(grads), self.global_step,
                )
        if not gradient_clip_val:
            return

        # FSDP path: Lightning's ``FSDPPrecision`` rejects
        # ``self.clip_gradients(..., algorithm='norm')`` -- sharded grads
        # need a cross-rank norm reduction, which the FSDP root module's
        # own ``clip_grad_norm_`` performs.  Route there under FSDP and use
        # the standard ``self.clip_gradients`` everywhere else (DDP / single).
        from pytorch_lightning.strategies import FSDPStrategy

        if isinstance(self.trainer.strategy, FSDPStrategy):
            fsdp_module = self.trainer.model  # FSDP-wrapped root
            clip_fn = getattr(fsdp_module, "clip_grad_norm_", None)
            if callable(clip_fn):
                clip_fn(float(gradient_clip_val))
                return

        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def configure_optimizers(self) -> Any:
        lr = self.optimizer_config.get("lr", 1e-4)
        wd = self.optimizer_config.get("weight_decay", 1e-5)
        # AdamW betas: default to PyTorch's (0.9, 0.999); a lower beta2
        # (e.g. 0.99) adapts the
        # second moment faster when fine-tuning a pretrained backbone.
        betas = tuple(self.optimizer_config.get("betas", (0.9, 0.999)))
        # AdamW epsilon.  PyTorch default 1e-8; 1e-6 is often
        # preferable under bf16 (avoids underflow in the
        # squared-gradient denominator).  Default keeps the historical 1e-8 so
        # configs that don't set it are byte-for-byte unchanged.
        eps = float(self.optimizer_config.get("eps", 1e-8))
        # Use explicit ``is None`` so a deliberate ``dit_backbone_lr: 0``
        # (e.g. to keep the unfrozen DiT weights pinned via gradient-
        # only updates from learned LR schedulers) is honoured rather
        # than silently falling back to ``lr`` via ``or``-truthiness.
        backbone_lr = self.optimizer_config.get("dit_backbone_lr")
        if backbone_lr is None:
            backbone_lr = lr

        # ``include_frozen_dit`` keeps the DiT params in the optimizer
        # even while they're still frozen, so the warm-up schedule
        # (``freeze_dit_backbone: N`` -- frozen for epochs 0..N-1, thawed
        # at epoch N) can simply flip ``requires_grad`` at epoch N
        # without rebuilding the optimizer (and without resetting the
        # LR scheduler).  AdamW skips params whose ``grad is None``,
        # so frozen DiT params sitting in the optimizer are a no-op
        # until they thaw.
        include_frozen_dit = (
            getattr(self.model, "dit_thaw_epoch", None) is not None
        )

        backbone_decay, backbone_no_decay = [], []
        head_decay, head_no_decay = [], []
        for name, param in self.named_parameters():
            is_backbone = name.startswith("model.dit.")
            if not param.requires_grad and not (
                include_frozen_dit and is_backbone
            ):
                continue
            no_decay = param.dim() <= 1 or name.endswith(".bias")
            if is_backbone:
                (backbone_no_decay if no_decay else backbone_decay).append(param)
            else:
                (head_no_decay if no_decay else head_decay).append(param)

        param_groups = [
            {"params": backbone_decay, "lr": backbone_lr, "weight_decay": wd},
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": lr, "weight_decay": wd},
            {"params": head_no_decay, "lr": lr, "weight_decay": 0.0},
        ]
        param_groups = [g for g in param_groups if g["params"]]
        self._dit_param_ids = {id(p) for p in backbone_decay + backbone_no_decay}

        clip_val = self.training_config.get("gradient_clip_val")
        master = master_dtype_from_config(self.optimizer_config)
        # ``fused`` is an AdamW kernel flag. MasterAdamW keeps its own loop.
        use_fused = (
            master is None
            and not clip_val
            and torch.cuda.is_available()
            and all(p.is_cuda for g in param_groups for p in g["params"])
        )
        if master is not None:
            optimizer = MasterAdamW(
                param_groups, lr=lr, betas=betas, eps=eps,
                weight_decay=wd, master_dtype=master,
            )
        else:
            optimizer = torch.optim.AdamW(
                param_groups, lr=lr, betas=betas, eps=eps, weight_decay=wd,
                fused=use_fused,
            )
        optimizer._dit_ids = self._dit_param_ids
        self._attach_update_tracker(optimizer)

        return self._maybe_wrap_scheduler(optimizer)

    def _attach_update_tracker(self, optimizer) -> None:
        """Measure the DiT update ratio inside ``optimizer.step``.

        Lightning calls ``optimizer.step`` either directly or through
        ``LightningModule.optimizer_step``. Wrapping the method covers both.
        The numeric update is the original ``step``.
        """
        module = self
        raw_step = optimizer.step

        def step(self, closure=None, *args, **kwargs):
            # ``self`` is the optimiser. A bound method keeps ``__func__``,
            # which ``LRScheduler`` reads when it wraps ``optimizer.step``.
            every = int(module.training_config.get("log_every_n_steps", 100) or 0)
            track = False
            try:
                step_idx = int(module.global_step) + 1
                track = every > 0 and step_idx % every == 0
            except Exception:
                track = False
            if not module.training_config.get("gradient_clip_val"):
                module._note_dit_grad(self, None)
            before = None
            params: list = []
            if track and isinstance(self, MasterAdamW):
                self._track_update = True
            elif track:
                dit_ids = getattr(module, "_dit_param_ids", ())
                for group in self.param_groups:
                    for p in group["params"]:
                        if id(p) in dit_ids:
                            params.append(p)
                before = [p.detach().clone() for p in params]
            try:
                return raw_step(closure, *args, **kwargs)
            finally:
                if isinstance(self, MasterAdamW):
                    self._track_update = False
                    if track:
                        module._dit_update_ratio = getattr(
                            self, "_last_update_ratio", None,
                        )
                elif before:
                    num = None
                    den = None
                    for prev, p in zip(before, params):
                        diff = p.detach().float() - prev.float()
                        d2 = diff.pow(2).sum()
                        b2 = prev.float().pow(2).sum()
                        num = d2 if num is None else num + d2
                        den = b2 if den is None else den + b2
                    if den is not None:
                        module._dit_update_ratio = float(
                            torch.sqrt(num / den.clamp_min(1e-30))
                        )

        optimizer.step = types.MethodType(step, optimizer)

    def _note_dit_grad(self, optimizer, clip_val) -> None:
        """Record the pre-clip DiT L2 and whether it exceeds the clip.

        The comparison stays on device. The host sync happens once, when
        the window is logged.
        """
        ids = getattr(self, "_dit_param_ids", None)
        if not ids:
            return
        grads = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if id(p) in ids and p.grad is not None:
                    grads.append(p.grad)
        if not grads:
            return
        norms = torch._foreach_norm(grads)
        total = torch.linalg.vector_norm(torch.stack([n.float() for n in norms]))
        self._dit_grad_norm = total.detach()
        one = total.new_ones(())
        seen = getattr(self, "_dit_clip_seen_t", None)
        self._dit_clip_seen_t = one if seen is None else seen + one
        if clip_val:
            hit = (total > float(clip_val)).to(dtype=total.dtype)
            prev = getattr(self, "_dit_clip_hits_t", None)
            self._dit_clip_hits_t = hit if prev is None else prev + hit

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Log DiT grad norm, clip rate, and update ratio on the log cadence."""
        every = int(self.training_config.get("log_every_n_steps", 100) or 0)
        if every <= 0 or int(self.global_step) % every != 0:
            return
        self._log_dit_stats()

    def _log_dit_stats(self) -> None:
        """All-reduce the DiT window and log it on every rank.

        ``grad_norm`` is the pre-clip L2 at this step. ``clip_rate`` is the
        fraction of steps since the previous log whose pre-clip norm
        exceeded ``gradient_clip_val``. ``update_ratio`` is ``||Δw|| / ||w||``
        of the DiT group at this step (fp32 master when ``master_dtype`` is set).
        """
        gnorm = getattr(self, "_dit_grad_norm", None)
        seen = getattr(self, "_dit_clip_seen_t", None)
        if gnorm is None or seen is None:
            return
        hits = getattr(self, "_dit_clip_hits_t", None)
        ratio = getattr(self, "_dit_update_ratio", None)
        zero = gnorm.new_zeros(())
        packed = torch.stack([
            gnorm.detach().float().reshape(()),
            hits.detach().float().reshape(()) if hits is not None else zero,
            seen.detach().float().reshape(()),
            gnorm.new_tensor(float(ratio) if ratio is not None else 0.0),
            gnorm.new_tensor(1.0 if ratio is not None else 0.0),
        ])
        world = int(getattr(self.trainer, "world_size", 1) or 1)
        if world > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            packed[0] = packed[0] / float(world)
        if float(packed[4]) > 0:
            packed[3] = packed[3] / packed[4]
        prefix = f"{self._scalar_prefix('train')}/dit"
        self.log(
            f"{prefix}/grad_norm", packed[0],
            on_step=True, on_epoch=False, sync_dist=False, rank_zero_only=False,
        )
        if float(packed[2]) > 0:
            self.log(
                f"{prefix}/clip_rate", packed[1] / packed[2],
                on_step=True, on_epoch=False, sync_dist=False, rank_zero_only=False,
            )
        if float(packed[4]) > 0:
            self.log(
                f"{prefix}/update_ratio", packed[3],
                on_step=True, on_epoch=False, sync_dist=False, rank_zero_only=False,
            )
        self._dit_grad_norm = None
        self._dit_clip_seen_t = None
        self._dit_clip_hits_t = None
        self._dit_update_ratio = None


__all__ = ["BackboneModule"]
