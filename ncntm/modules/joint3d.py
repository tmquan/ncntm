"""
Joint reconstruction + segmentation Lightning module (the ncntm recipe).

``Joint3DModule`` reuses the **whole** Cosmos-3 Nano training/eval machinery
(model build, freeze schedule, optimiser param-group split, gradient guard,
the epoch-end reduce/log loop) from
:class:`~ncntm.modules.cosmos_3_nano.module.Cosmos3Nano3DModule`, and
swaps in :class:`~ncntm.losses.Joint3DReconSegLoss`.  Only three things
differ from the plain affinity recipe, all task-routing:

* :meth:`_loss_offsets` -- the offset list lives under ``loss.seg.offsets``
  (nested), so the head width is derived from there.
* :meth:`_prepare_targets` -- builds the per-batch ``targets`` for the active
  branch: ``ssl`` carries ``recon_image`` and **no labels**; ``sft`` carries
  native ``labels`` (+ cached affinity target) and optionally ``recon_image``
  (the original large-voxel EM for the ``raw`` data-consistency term).
* :meth:`_accumulate_metrics` -- on ``sft`` the small-voxel head is pooled to
  the native label grid before the foreground / Mutex-Watershed metrics; on
  ``ssl`` (no labels) segmentation metrics are skipped.

``training_step`` / ``validation_step`` / the epoch-end reduce are inherited.
They call ``_prepare_targets`` -> ``self.model(images)`` ->
``self.criterion(head, targets)``, and :class:`Joint3DReconSegLoss` does the
branch routing + small-voxel→native pooling internally.
:meth:`_combine_field` then adds the rectified-flow MSE on ``proj_out``
(``loss/field``) at ``weight_dif`` next to the reconstruction L1, scaled
by the same branch weight.

Batches must be **task-homogeneous** (one branch per step) -- the joint
datamodule's round-robin sampler guarantees this.
"""

from typing import Any, Dict

import torch
from einops import rearrange

from ncntm.losses import SSL, Joint3DReconSegLoss
from ncntm.models.cosmos_3_edge import Cosmos3EdgeWrapper
from ncntm.modules.cosmos_3_nano.module import Cosmos3Nano3DModule


class Joint3DModule(Cosmos3Nano3DModule):
    """Cosmos-3 Nano backbone trained with the joint recon + seg loss."""

    _loss_cls = Joint3DReconSegLoss

    # ------------------------------------------------------------------
    # Config / target plumbing
    # ------------------------------------------------------------------

    def _loss_offsets(self, loss_config: Dict[str, Any]):
        # The joint loss nests the affinity config under ``seg``.
        return (loss_config.get("seg") or {}).get("offsets")

    @staticmethod
    def _batch_task(batch: Dict[str, Any]) -> str:
        """Resolve the (task-homogeneous) batch's branch name to a str."""
        task = batch.get("task")
        if isinstance(task, str):
            return task
        if isinstance(task, (list, tuple)) and task:
            uniq = set(map(str, task))
            if len(uniq) != 1:
                raise ValueError(
                    f"Joint3DModule expects task-homogeneous batches; got mixed "
                    f"{sorted(uniq)}. Use the round-robin multi-task sampler."
                )
            return next(iter(uniq))
        raise KeyError(
            "Joint3DModule batch is missing a 'task' field ('ssl' | 'sft'); "
            "the joint datamodule must tag each batch."
        )

    def _maybe_scale_recon(self, recon: torch.Tensor) -> torch.Tensor:
        """Match the raw head's range: scale [0,1] -> [-1,1] when vae_input_pm1."""
        if getattr(self.model, "vae_input_pm1", False):
            return recon * 2.0 - 1.0
        return recon

    @torch.no_grad()
    def _prepare_targets(
        self, batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        ndim_with_channel = self._SPATIAL_DIMS + 2
        task = self._batch_task(batch)
        targets: Dict[str, Any] = {"task": task}

        # Reconstruction target (clean EM): required on ssl, optional on sft.
        # Resample average-pools the 4 nm head onto the native resolution.
        recon = batch.get("recon_image")
        if recon is not None:
            targets["recon_image"] = self._maybe_scale_recon(recon)
        native_size = batch.get("recon_native_size")
        if native_size is not None:
            targets["recon_native_size"] = native_size

        if task == SSL:
            return targets  # label-free branch

        # ---- sft: native labels + cached affinity target ----
        labels = batch["label"]
        if labels.dim() == ndim_with_channel:
            labels = rearrange(labels, self._SQUEEZE_PATTERN)
        targets["labels"] = labels

        sem_label = batch.get("sem_label")
        if sem_label is not None:
            if sem_label.dim() == ndim_with_channel:
                sem_label = rearrange(sem_label, self._SQUEEZE_PATTERN)
            targets["sem_label"] = sem_label

        targets["_cached_targets"] = self.criterion.build_targets(labels, targets)
        return targets

    def _combine_field(
        self,
        losses: Dict[str, torch.Tensor],
        images: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Add ``weight_dif * MSE(proj_out, v)`` beside the reconstruction L1.

        The one-step head has already run on the degraded volume at timestep
        0. This second forward trains the velocity that few-step integration
        follows. ``recon_image`` stays in the dataset ``[0, 1]`` range; the
        wrapper applies the VAE scale. A missing clean target skips the term
        (an sft batch with no ``recon_image``), and so does every sample whose
        native grid is coarser than the 4 nm cube.
        """
        weight = float(getattr(self.criterion, "weight_dif", 0.0) or 0.0)
        clean = None if weight <= 0 else batch.get("recon_image")
        flow_fn = getattr(self.model, "rectified_flow_loss", None)
        if weight <= 0 or clean is None or not callable(flow_fn):
            return losses
        # Only a sample whose native grid is the 4 nm cube has a clean 4 nm
        # endpoint.  Coarser targets are nearest repeats, and the velocity
        # toward them is a velocity toward slabs.
        native = batch.get("recon_native_size")
        if native is not None:
            native = torch.as_tensor(native, device=images.device)
            if native.dim() == 1:
                native = native.unsqueeze(0).expand(images.shape[0], -1)
            fine = torch.tensor(images.shape[-3:], device=images.device)
            keep = (native == fine).all(dim=1)
            if not bool(keep.any()):
                return losses
            if not bool(keep.all()):
                idx = keep.nonzero().flatten()
                images, clean = images[idx], clean[idx]
        flow = flow_fn(images, clean)
        losses["loss/field"] = flow
        branch = (
            self.criterion.weight_ssl
            if self._batch_task(batch) == SSL
            else self.criterion.weight_sft
        )
        losses["loss"] = losses["loss"] + branch * weight * flow
        return losses

    # ------------------------------------------------------------------
    # Per-dataset validation scalars
    # ------------------------------------------------------------------

    def _val_dataset_names(self) -> list:
        """Sorted dataset names seeded on every rank from the datamodule."""
        trainer = getattr(self, "_trainer", None)
        dm = getattr(trainer, "datamodule", None) if trainer is not None else None
        names = getattr(dm, "val_dataset_names", None) if dm is not None else None
        return list(names) if names else []

    @staticmethod
    def _dataset_names(batch: Dict[str, Any]) -> list:
        ds = batch.get("dataset")
        if ds is None:
            return []
        if isinstance(ds, str):
            return [ds]
        return [str(x) for x in ds]

    def _accum_seeded(self, name: str, value: Any, weight: float) -> None:
        """Accumulate only a key that every rank already seeded."""
        if name not in self._eval_accum:
            return
        self._accum(name, value, weight)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        for name in self._val_dataset_names():
            for key in ("loss/recon", "loss/flow", "loss/recon_identity"):
                _ = self._eval_accum[f"val/{name}/{key}"]
        trainer = getattr(self, "_trainer", None)
        sanity = bool(getattr(trainer, "sanity_checking", False)) if trainer is not None else False
        self._identity_this_epoch = (
            not sanity and not getattr(self, "_identity_done", False)
        )

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if getattr(self, "_identity_this_epoch", False):
            self._identity_done = True

    def _accumulate_dataset_eval(
        self,
        batch: Dict[str, Any],
        head: torch.Tensor,
        images: torch.Tensor,
        targets: Dict[str, Any],
        losses: Dict[str, torch.Tensor],
        stage: str,
    ) -> None:
        """Log per-dataset recon and flow, and the identity L1 once per run.

        The identity baseline is the L1 of the degraded input against the
        target, pooled and scaled the same way as ``loss/recon``. It runs
        on the first real validation epoch, not during the sanity check.
        """
        if stage != "val":
            return
        names = self._dataset_names(batch)
        if not names or "recon_image" not in targets:
            return
        native = targets.get("recon_native_size")
        per = self.criterion.l1_per_sample(head, targets["recon_image"], native)
        for i, name in enumerate(names):
            if i >= per.shape[0]:
                break
            self._accum_seeded(f"val/{name}/loss/recon", per[i], 1.0)
        if len(set(names)) == 1 and "loss/field" in losses:
            self._accum_seeded(
                f"val/{names[0]}/loss/flow", losses["loss/field"], float(len(names)),
            )
        if getattr(self, "_identity_this_epoch", False):
            degraded = self._maybe_scale_recon(images)
            ident = self.criterion.l1_per_sample(
                degraded, targets["recon_image"], native,
            )
            for i, name in enumerate(names):
                if i >= ident.shape[0]:
                    break
                self._accum_seeded(
                    f"val/{name}/loss/recon_identity", ident[i], 1.0,
                )

    # ------------------------------------------------------------------
    # Eval metrics (pool small-voxel head -> native label grid for sft)
    # ------------------------------------------------------------------

    def _accumulate_metrics(
        self,
        head: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        prefix: str,
        bs: float,
    ) -> None:
        # ssl is reconstruction-only (no labels) -> no segmentation metrics.
        if targets.get("task") == SSL or "labels" not in targets:
            return
        # The head is on the fine (small-voxel) grid; the labels are native.
        # Pool the head down to the label grid (a no-op when they match) so the
        # foreground + Mutex-Watershed metrics are scored at the supervised
        # resolution -- exactly the grid the seg loss used.
        labels = targets["labels"]
        pooled = self.criterion._pool_to(head, labels.shape[-3:])
        self._accumulate_semantic_metrics(pooled, targets, prefix, bs)
        self._accumulate_instance_metrics(pooled, targets, prefix, bs)


class JointEdge3DModule(Joint3DModule):
    """Cosmos3-**Edge** (4B) backbone trained with the joint recon + seg loss.

    Identical recipe to :class:`Joint3DModule` (the Nano joint module); the only
    difference is the backbone -- the 4B Cosmos3-Edge omni transformer, loaded
    from the released ``nvidia/Cosmos3-Edge`` checkpoint (see
    :class:`~ncntm.models.cosmos_3_edge.Cosmos3EdgeWrapper`), or random-init
    when ``model.pretrained=false``.  Edge uses the same Wan2.2 VAE family
    (16x spatial / 4x temporal), so the data / loss / pooling recipe is unchanged.
    """

    _model_cls = Cosmos3EdgeWrapper


__all__ = [
    "Joint3DModule",
    "JointEdge3DModule",
]
