"""
Multi-task datamodule for the joint reconstruction + segmentation recipe.

Feeds :class:`~ncntm.modules.Joint3DModule` the batch contract it expects
``task`` (``"ssl"`` | ``"sft"``), the fine-grid
``image``, the native-grid ``label`` (sft), and the ``recon_image`` target.

Geometry.  The network runs on a fixed fine
grid (``patch_size`` @ ``pixel_size`` nm, e.g. [256, 256, 256] @ 4 nm).  Each
volume is read at its **native** voxel size over the same physical field of
view, then :class:`~ncntm.transforms.ToFineGridd` resamples the image onto
the fine grid while the label stays native.  SSL then runs Augment
(one gamma for the whole crop), Degrade (``z`` downsample on a 4 nm crop; on
an anisotropic crop, one random in-plane axis downsampled to the z spacing;
each slab the mean of its planes or one picked plane), Corrupt (per-section jitter and gamma, and noise), and
Missing (whole native sections, every crop) on the input alone.  The target
keeps the native crop, so the degraded fine axis is supervised at its own
resolution.
``ToFineGridd`` nearest-upsamples the image onto the 4 nm cube.  The
augmented target is nearest-upsampled onto that cube after the axis
permutation, and only then collated, so an anisotropic crop still stacks.
``recon_native_size`` records that target's grid.  After the forward pass,
Resample average-pools the 4 nm head onto that native resolution, so each
native voxel is the mean of the 4 nm voxels in its bin, and skips the pool
when the crop is already 4 nm. The comparison
is the augmented value, not the read and not the corrupted input.

``rand_transpose_zyx`` (train only) then draws a uniform permutation of the
three axes, after :class:`~ncntm.transforms.ToFineGridd` and
:class:`~ncntm.transforms.RandResolutionDegraded`, so section degradation
stays on the native section axis.  The fine FOV has to be cubic: ``patch_size``
and ``pixel_size`` must both be cubic or ``__init__`` raises ``ValueError``.
A permutation is then a pure reorientation: the recon grid moves with the
data.  The text prompt (``{imaging} · {tail}``) is the text condition: a frozen
T5 encoder maps it into the DiT class condition.  The finest native axis -- the
highest-resolution dimension -- therefore lands on network z, y, or x equally
often.  Validation stays in the stored axis order.

Branches & batching.  Volumes are bucketed into round-robin groups per the
``balance`` policy -- ``"resolution"`` (one group per
``(task, native_resolution, label_convention)``; the default), ``"subset"``
(one group per ``(task, subset, native_resolution, label_convention)``),
``"volume"`` (one group per volume; each volume scheduled equally), or
``"dataset"`` (each dataset an equal share of batches, then split across
``(dataset, native_resolution, label_convention)`` sub-groups).  Every
group is single-task, single-resolution, and a single ``label_convention``,
so a batch drawn from one group is **task-homogeneous** and, with
``rand_transpose_zyx`` off, **shape-homogeneous**.  That is what
``Joint3DReconSegLoss`` / ``Joint3DModule`` require.  The ssl recon target
is nearest-upsampled onto the cubic fine grid before collation, so an
anisotropic ssl group keeps its branch batch size under
``rand_transpose_zyx``.  A labeled anisotropic group keeps it too, but the
label stays on the native grid, so a permutation changes its shape.  The
sampler therefore draws ONE permutation per batch and every sample of that
batch uses it (an INFO log notes the group); the labels then stack.  A
batch of one is unaffected.  Isotropic groups always keep the branch batch
size.  A round-robin batch sampler draws one group
per batch.  Training seeds ``seed + epoch + 100003 * rank``, so rank 0
keeps the historical sequence and the other ranks draw different groups
in the same step.  Validation seeds ``seed + epoch`` on every rank and
then gives batch ``k`` to rank ``k % world_size``.  The training
sampler's ``__len__`` ignores rank.  The loader passes that object as
``batch_sampler`` and uses the ``forkserver`` context when
``num_workers > 0`` (Lightning ``use_distributed_sampler`` stays false).
``resolution`` / ``subset`` / ``volume`` weight groups by each branch's
``sample_weight`` and (for subset/volume) by ``subset_weights`` /
per-volume ``sample_weight``.  ``dataset`` ignores the group count: a
branch's batches follow ``sample_weight`` only, then each dataset inside
the branch is equal (times ``dataset_weights``).

Config schema (``cfg.data``)::

    patch_size: [256, 256, 256]      # fine grid (z, y, x); cubic if
                                     #   rand_transpose_zyx is on. Pretrain
                                     #   is 256^3; joint 4B is [400, 256, 256]
    pixel_size: [4, 4, 4]            # fine voxel size nm (cubic)
    rand_transpose_zyx: true         # train-only random z/y/x permutation.
                                     #   ssl keeps batch_size (recon is
                                     #   nearest-upsampled before collation).
                                     #   A labeled anisotropic group shares one
                                     #   permutation per batch so labels stack.
    degrade: {zf_range: [...], ...}  # RandResolutionDegraded kwargs (ssl)
    sft_min_foreground: 0.8          # sft gate: BOTH label fg AND image non-zero
    ssl_min_foreground: 0.8          # image non-zero gate (ssl volumes)
    ssl_min_std: 0.05                # ssl contrast gate (reject flat resin crops)
    balance: resolution              # "resolution" (default) | "subset"
                                     #   | "volume" | "dataset"
    within_dataset: tissue           # (balance: dataset) split a dataset's
                                     #   sub-groups by tissue | volume
    dataset_weights: {MICRONS: 1}    # (balance: dataset) per-dataset multiplier
    val_balance: resolution          # "resolution" (default) buckets val by
                                     #   native grid. "dataset" gives each
                                     #   dataset its own val group, so a small
                                     #   volume is not lost inside a bigger
                                     #   one that shares the grid.
    subset_weights: {cremi3d: 50}    # (balance: subset) per-subset schedule weight
    find_boundaries: 1.0             # per-sample boundary-erosion probability
                                     #   (applied to filled / unset sft volumes)
    boundary_target: semantic        # "semantic" (sem_label only) | "both"
    branches:
      ssl: {batch_size, sample_weight, volumes: [{vol, root, native_resolution}]}
      sft:  {batch_size, sample_weight, volumes: [
               {vol, seg, root, native_resolution, label_convention?}]}
      #   label_convention (optional, sft): ``gapped`` | ``filled``
      #     gapped -- membranes / extracellular already label==0; skip
      #               FindBoundariesd (gaps already present).
      #     filled -- space-filling / abutting instances; apply
      #               FindBoundariesd so the sem head sees thin gaps.
      #   per-volume ``sample_weight`` (optional) scales its share when
      #   balance: volume (e.g. overfit one volume).
    val_volumes: [{vol, seg, root, task, native_resolution, label_convention?}]
"""

from __future__ import annotations

import bisect
import logging
import math
import random
import re
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
from monai.transforms import (
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
)
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from ncntm.datamodules._worker import worker_init
from ncntm.datasets.lazy import LazyVolDataset
from ncntm.transforms.rand_transpose_zyx import N_ZYX_PERMS
from ncntm.transforms import (
    FindBoundariesd,
    Labeld,
    RandResolutionDegraded,
    RandTransposeZYXd,
    ReconNearestToFined,
    ToFineGridd,
)

logger = logging.getLogger(__name__)

from ncntm.datamodules.condition_text import (  # noqa: E402
    LABEL_CONVENTIONS,
    prompt_from_volume_spec,
    validate_label_convention,
)

_DEGRADE_KEYS = (
    "zf_range", "prob", "jitter_prob", "max_jitter", "missing_prob",
    "max_missing", "missing_fill", "noise_prob", "noise_std", "contrast_prob",
    "contrast_gamma", "volume_contrast_prob", "volume_contrast_gamma",
    "fine_axis_prob", "slab_mean_prob",
)


def _scaled(fine_patch: Sequence[int], fine_nm: float, res: Sequence[float]) -> Tuple[int, ...]:
    """Native voxel count covering the fine FOV: round(fine * fine_nm / res)."""
    return tuple(max(1, int(round(fine_patch[d] * fine_nm / float(res[d]))))
                 for d in range(len(fine_patch)))


def _largest_remainder(weights: Sequence[float], total: int) -> List[int]:
    """Split ``total`` into integers proportional to ``weights``.

    The parts sum to ``total``. A zero weight gets zero. Negative weights
    raise ``ValueError``.
    """
    n = len(weights)
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if any(float(w) < 0 for w in weights):
        raise ValueError(f"weights must be >= 0, got {list(weights)}")
    if n == 0 or total == 0:
        return [0] * n
    scale = float(sum(float(w) for w in weights))
    if scale <= 0:
        return [0] * n
    raw = [total * float(w) / scale for w in weights]
    floors = [int(math.floor(x)) for x in raw]
    left = total - sum(floors)
    if left < 0:
        order = sorted(range(n), key=lambda i: floors[i], reverse=True)
        for i in order:
            if left == 0:
                break
            take = min(floors[i], -left)
            floors[i] -= take
            left += take
    else:
        order = sorted(
            range(n),
            key=lambda i: (raw[i] - math.floor(raw[i]), float(weights[i])),
            reverse=True,
        )
        for i in order:
            if left <= 0:
                break
            floors[i] += 1
            left -= 1
    return floors


class _RoundRobinBatchSampler(Sampler):
    """Yield one group per batch, round-robin across groups.

    Each group occupies a contiguous index range ``[offset, offset+length)`` in
    the parent :class:`ConcatDataset`.  A group contributes
    ``length // batch_size`` batches per epoch; the per-group batch counts are
    interleaved (shuffled on train, sequential on val).  ``sample_weight`` is
    folded into each group's ``length`` (= num_samples) by the datamodule.

    Training (``shard=False``) seeds ``seed + epoch + 100003 * rank``. Rank 0
    therefore walks the historical sequence. ``__len__`` ignores rank.
    Validation (``shard=True``) builds that same rank-0 schedule, then keeps
    batch ``k`` on rank ``k % world_size``.

    A group may set ``shared_perm`` (4th field).  Its batches then draw ONE
    z/y/x permutation and tag every index with it as ``(index, perm)``, which
    :class:`_PermConcatDataset` forwards to the transform.  That is how a
    labeled anisotropic group, whose label stays on the native grid, keeps a
    batch size above 1 under ``rand_transpose_zyx``: every label in the batch
    has the same permuted shape.
    """

    def __init__(
        self,
        # (offset, length, batch_size[, shared_perm])
        groups: List[Tuple[int, ...]],
        shuffle: bool,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        shard: bool = False,
    ) -> None:
        self.groups = [
            (int(g[0]), int(g[1]), int(g[2]), bool(g[3]) if len(g) > 3 else False)
            for g in groups
        ]
        self.shuffle = shuffle
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.shard = bool(shard)
        self._epoch = 0

    def _full_len(self) -> int:
        return sum(length // bs for _, length, bs, _s in self.groups if bs > 0)

    def __len__(self) -> int:
        n = self._full_len()
        if not self.shard or self.world_size <= 1:
            return n
        if self.rank >= n:
            return 0
        return (n - self.rank + self.world_size - 1) // self.world_size

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1
        if self.shard:
            rng = random.Random(self.seed + epoch)
        else:
            rng = random.Random(self.seed + epoch + 100003 * self.rank)
        schedule: List[int] = []
        for gi, (_off, length, bs, _shared) in enumerate(self.groups):
            schedule += [gi] * (length // bs)
        if self.shuffle:
            rng.shuffle(schedule)
        cursors = [0] * len(self.groups)
        batches: List[List[int]] = []
        for gi in schedule:
            off, length, bs, shared = self.groups[gi]
            if self.shuffle and shared:
                perm = rng.randrange(N_ZYX_PERMS)
                batches.append(
                    [(off + rng.randrange(length), perm) for _ in range(bs)]
                )
            elif self.shuffle:
                batches.append([off + rng.randrange(length) for _ in range(bs)])
            else:
                c = cursors[gi]
                batches.append([off + ((c + j) % length) for j in range(bs)])
                cursors[gi] = (c + bs) % length
        if self.shard and self.world_size > 1:
            batches = [b for k, b in enumerate(batches) if k % self.world_size == self.rank]
        yield from batches


class _PermConcatDataset(ConcatDataset):
    """:class:`ConcatDataset` that also takes ``(index, zyx_perm)`` indices.

    The sampler tags the indices of a ``shared_perm`` batch with one
    permutation id; it is passed to ``LazyVolDataset.fetch`` so the whole batch
    shares an orientation.  A plain integer index behaves as in ``ConcatDataset``.
    """

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            return super().__getitem__(idx)
        idx, perm = idx
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        d = bisect.bisect_right(self.cumulative_sizes, idx)
        sample = idx if d == 0 else idx - self.cumulative_sizes[d - 1]
        return self.datasets[d].fetch(sample, zyx_perm=perm)


class Joint3DDataModule(pl.LightningDataModule):
    """Round-robin multi-task datamodule for the joint recipe.

    Note: unlike the single-dataset datamodules, this one intentionally does
    NOT call ``save_hyperparameters()`` -- the joint data config (branches,
    degrade schema, crop gates) is reproduced from the Hydra config snapshot
    saved under the run directory rather than from checkpoint
    ``datamodule_hyper_parameters``. Model checkpoints are unaffected either way.
    """

    def __init__(
        self,
        data_root: str = "data",
        patch_size: Sequence[int] = (320, 256, 256),
        pixel_size: Sequence[float] = (4.0, 4.0, 4.0),
        rand_transpose_zyx: bool = False,
        branches: Optional[Dict[str, Any]] = None,
        degrade: Optional[Dict[str, Any]] = None,
        val_volumes: Optional[List[Dict[str, Any]]] = None,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        num_samples: int = 8000,
        val_num_samples: int = 16,
        val_batch_size: int = 1,
        min_foreground: float = 0.0,
        sft_min_foreground: float = 0.0,
        sft_min_instances: int = 0,
        sft_max_inst_frac: float = 0.0,
        sft_large_object_prob: float = 0.0,
        sft_large_object_min_frac: float = 0.5,
        ssl_min_foreground: float = 0.0,
        ssl_min_std: float = 0.0,
        ssl_min_autocorr: float = 0.0,
        find_boundaries: float = 0.0,
        boundary_target: str = "semantic",
        balance: str = "resolution",
        subset_weights: Optional[Dict[str, float]] = None,
        within_dataset: str = "tissue",
        dataset_weights: Optional[Dict[str, float]] = None,
        val_balance: str = "resolution",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.fine_patch = tuple(int(s) for s in patch_size)
        pix = tuple(float(p) for p in pixel_size)
        # Cubic fine grid: use the z spacing as the (single) fine voxel size.
        self.fine_nm = float(pix[0])
        # Train-only zyx permutation.  Requires a cubic FOV so each permutation
        # preserves the physical extent on every axis.  See _group_transform.
        self.rand_transpose_zyx = bool(rand_transpose_zyx)
        if self.rand_transpose_zyx:
            if len(self.fine_patch) != 3 or len(set(self.fine_patch)) != 1:
                raise ValueError(
                    "rand_transpose_zyx requires a cubic patch_size so a zyx "
                    "permutation preserves the physical field of view; "
                    f"got patch_size={list(self.fine_patch)}."
                )
            if len(pix) != 3 or len(set(pix)) != 1:
                raise ValueError(
                    "rand_transpose_zyx requires cubic pixel_size (nm); "
                    f"got pixel_size={list(pix)}."
                )
        self.branches = dict(branches or {})
        self.degrade = {k: v for k, v in dict(degrade or {}).items() if k in _DEGRADE_KEYS}
        self.val_volumes = val_volumes
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and num_workers > 0
        self.prefetch_factor = int(prefetch_factor)
        self.num_samples = int(num_samples)
        self.val_num_samples = int(val_num_samples)
        self.val_batch_size = int(val_batch_size)
        self.min_foreground = float(min_foreground)
        # sft crop gate: require BOTH label foreground >= t AND image non-zero >= t
        # (rejects background-heavy / unlabeled crops and zero-padded EM).  Falls
        # back to the legacy label-only ``min_foreground`` when unset.
        self.sft_min_foreground = float(sft_min_foreground) or float(min_foreground)
        # Instance-diversity gate for the sft branch: ``sft_min_foreground`` only
        # checks the label's NON-ZERO fraction, so a crop entirely filled by ONE
        # giant instance (e.g. a single MICrONS dendrite trunk spanning the whole
        # patch) passes trivially yet gives AffinityFGLoss no inter-instance
        # "push" signal -- only "pull".  ``sft_min_instances`` rejects crops with
        # fewer than N distinct nonzero instance ids; ``sft_max_inst_frac``
        # rejects crops where the single largest instance exceeds this fraction
        # of the labeled foreground.  Both 0 = disabled (default).
        self.sft_min_instances = int(sft_min_instances)
        self.sft_max_inst_frac = float(sft_max_inst_frac)
        # Large-object oversampling (sft train groups only; validation stays
        # deterministic and unbiased): with probability ``sft_large_object_prob``
        # a crop is drawn so its largest instance covers at least
        # ``sft_large_object_min_frac`` of the labeled foreground.  Uniform
        # random crops rarely look like that (~10% reach 0.5, ~0% reach 0.8),
        # yet single large objects are what the sem gate and mutex watershed
        # tend to miss.  Keep ``sft_max_inst_frac`` off when enabling it.
        self.sft_large_object_prob = float(sft_large_object_prob)
        self.sft_large_object_min_frac = float(sft_large_object_min_frac)
        # Image-nonzero gate for the label-less ssl branch: rejects mostly-empty /
        # zero-padded crops (e.g. MitoEM2 nnU-Net irregular crops padded to a box,
        # empty COSEM/FLYEM regions) that would otherwise show as black panels.
        self.ssl_min_foreground = float(ssl_min_foreground)
        # SSL contrast gate: local-block std threshold on the per-volume [0, 1]
        # scale (see ``LazyVolDataset._content_fraction``).  A crop passes when
        # the fraction of blocks whose std exceeds this value is at least
        # ``ssl_min_foreground`` (0.5 when that gate is off).  Flat resin fails
        # the block test; a non-zero mid-grey mean does not.  0 = disabled.
        self.ssl_min_std = float(ssl_min_std)
        # Structure gate for the ssl branch: reject NOISE-dominated crops
        # (detector / resin grain) that pass the std/content gate because random
        # noise has high local variance.  Lag-1 spatial autocorrelation on the
        # per-volume [0, 1] scale: real ultrastructure ~0.6-0.8, white noise ~0.
        # 0 = disabled.
        self.ssl_min_autocorr = float(ssl_min_autocorr)
        # sem-head boundary supervision (sft only).  ``find_boundaries`` = per-
        # sample probability of eroding membrane voxels so the sem head targets
        # thin membranes instead of (near-degenerate) full foreground.
        # ``boundary_target``: "semantic" -> a separate eroded ``sem_label`` for
        # the sem head only (instance ``label`` stays pristine for affinity);
        # "both" -> erode the shared ``label`` (sem AND affinity targets).
        self.find_boundaries = float(find_boundaries)
        self.boundary_target = str(boundary_target)
        if self.boundary_target not in ("both", "semantic"):
            raise ValueError(
                f"boundary_target must be 'both' or 'semantic'; got {boundary_target!r}."
            )
        # How volumes are bucketed into round-robin groups (= the unit that is
        # scheduled equally per epoch).  Each batch is drawn from ONE group.
        # ``label_convention`` is part of the key so gapped and filled volumes
        # do not share a FindBoundariesd pipeline.
        #   "resolution" -- one group per (task, native_resolution,
        #                   label_convention) [default].  Within a group,
        #                   volumes are picked voxel-count-weighted, so big
        #                   volumes dominate.
        #   "subset"     -- one group per (task, subset, native_resolution,
        #                   label_convention), weighted by ``subset_weights``.
        #   "volume"     -- one group per volume, weighted by that volume's
        #                   ``sample_weight``.  A batch is drawn from a single
        #                   volume.
        #   "dataset"    -- each dataset gets an equal share of the branch's
        #                   batches (times ``dataset_weights``). A sub-group
        #                   is ``(dataset, native_resolution, label_convention)``.
        #                   ``within_dataset`` splits that share by tissue
        #                   (voxels times voxel volume) or by volume count.
        # ``subset_weights`` / per-volume ``sample_weight`` multiply a group's
        # schedule length for subset/volume balance, so a subset/volume can
        # be deliberately over-sampled (e.g. to overfit CREMI / SNEMI).
        self.balance = str(balance)
        if self.balance not in ("resolution", "subset", "volume", "dataset"):
            raise ValueError(
                "balance must be 'resolution', 'subset', 'volume' or 'dataset'; "
                f"got {balance!r}."
            )
        self.within_dataset = str(within_dataset)
        if self.within_dataset not in ("tissue", "volume"):
            raise ValueError(
                "within_dataset must be 'tissue' or 'volume'; "
                f"got {within_dataset!r}."
            )
        self.subset_weights = {str(k): float(v) for k, v in (subset_weights or {}).items()}
        self.dataset_weights = {
            str(k): float(v) for k, v in (dataset_weights or {}).items()
        }
        self.val_balance = str(val_balance)
        if self.val_balance not in ("resolution", "dataset"):
            raise ValueError(
                "val_balance must be 'resolution' or 'dataset'; "
                f"got {val_balance!r}."
            )
        self.seed = int(seed)

        self._train_groups: List[Tuple[int, ...]] = []
        self._val_groups: List[Tuple[int, ...]] = []
        self._train_group_dataset: List[str] = []
        self._train_group_branch: List[str] = []
        self.balance_rows: List[Dict[str, Any]] = []
        self.val_dataset_names: List[str] = []
        self._val_group_dataset: List[str] = []
        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[ConcatDataset] = None

    # ------------------------------------------------------------------
    # Per-group transform pipelines
    # ------------------------------------------------------------------

    def _append_axis_perm(
        self, tf: List[Any], keys: Sequence[str], native_res: Sequence[float],
    ) -> None:
        """Permute z/y/x after the fine-grid resample (train only).

        The network image is already cubic, so its shape is unchanged.  The
        recon / label grids follow the same permutation.  That places the finest native axis
        on a random network axis.
        """
        tf.append(RandTransposeZYXd(
            keys=list(keys), native_resolution=native_res,
        ))

    def _group_transform(
        self,
        task: str,
        native_res: Sequence[float],
        *,
        label_convention: Optional[str] = None,
        augment: bool = False,
    ) -> Compose:
        # Degrade and nearest-upsample first, then the loader collates.
        # ToFineGridd puts ``image`` on the fine grid.  The clean target
        # stays at the native crop through the optional zyx permutation,
        # then ``ReconNearestToFined`` repeats it onto the fine cube.
        recon_size = _scaled(
            self.fine_patch, self.fine_nm,
            [max(float(r), self.fine_nm) for r in native_res],
        )
        if task == "ssl":
            ssl_keys = ["image", "recon_image"]
            tf: List[Any] = [
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                RandResolutionDegraded(
                    keys=["image"], recon_key="recon_image",
                    downsample=self._is_fine_nm(native_res),
                    fine_axis_factor=self._fine_axis_factor(native_res),
                    **self.degrade,
                ),
                ToFineGridd(
                    image_size=self.fine_patch, recon_size=recon_size,
                    set_recon_from_image=False, task="ssl",
                ),
                EnsureTyped(keys=ssl_keys),
            ]
            if augment:
                self._append_axis_perm(tf, ssl_keys, native_res)
            self._append_recon_on_fine(tf)
            return Compose(tf)
        # sft pipeline.  Optional boundary erosion makes the sem head target thin
        # membranes instead of near-degenerate full foreground.  Erosion runs on
        # the NATIVE label grid using this group's native resolution, so
        # FindBoundariesd's anisotropy guard (xy-only when z is >2x coarser)
        # applies per dataset.  ``gapped`` volumes already have membrane /
        # extracellular as label==0, so FindBoundariesd is skipped for them.
        sft_tf: List[Any] = [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Labeld(keys=["label"], spatial_dims=3),
        ]
        out_keys = ["image", "label", "recon_image"]
        apply_boundaries = (
            self.find_boundaries > 0 and label_convention != "gapped"
        )
        if apply_boundaries:
            if self.boundary_target == "both":
                sft_tf.append(FindBoundariesd(
                    keys=["label"], prob=self.find_boundaries, pixel_size=native_res,
                ))
            else:  # "semantic": eroded sem_label only; instance label stays pristine
                sft_tf += [
                    CopyItemsd(keys=["label"], times=1, names=["sem_label"]),
                    FindBoundariesd(
                        keys=["sem_label"], prob=self.find_boundaries,
                        pixel_size=native_res,
                    ),
                ]
                out_keys.append("sem_label")
        sft_tf += [
            ToFineGridd(
                image_size=self.fine_patch, recon_size=recon_size,
                set_recon_from_image=True, task="sft",
            ),
            EnsureTyped(keys=out_keys),
        ]
        if augment:
            self._append_axis_perm(sft_tf, out_keys, native_res)
        self._append_recon_on_fine(sft_tf)
        return Compose(sft_tf)

    def _is_fine_nm(self, native_res: Sequence[float]) -> bool:
        """True when every native axis is the fine voxel size (the 4 nm crop)."""
        return all(abs(float(r) - self.fine_nm) <= 1e-3 for r in native_res)

    @staticmethod
    def _fine_axis_factor(native_res: Sequence[float]) -> Optional[float]:
        """Native z nm over xy nm when z is coarser than a square xy, else ``None``."""
        z, y, x = (float(r) for r in native_res)
        if abs(y - x) > 1e-3 or z <= y * (1.0 + 1e-3):
            return None
        return z / y

    def _append_recon_on_fine(self, tf: List[Any]) -> None:
        """Nearest-upsample the native clean target onto the fine cube.

        Always runs, after degradation and after the zyx permutation, so
        collation sees one 4 nm shape. ``recon_native_size`` is the permuted
        native resolution Resample pools back to.
        """
        tf.append(ReconNearestToFined(self.fine_patch))

    def _build_group(
        self,
        task: str,
        native_res: Tuple[float, ...],
        volumes: List[Dict[str, Any]],
        num_samples: int,
        deterministic: bool,
        *,
        label_convention: Optional[str] = None,
        augment: bool = False,
        volume_sampling: str = "voxels",
        quiet: bool = False,
    ) -> LazyVolDataset:
        native_patch = _scaled(self.fine_patch, self.fine_nm, native_res)
        # Attach the batch prompt (imaging · tail): the text condition.
        lazy_vols: List[Dict[str, Any]] = []
        for vol in volumes:
            entry = {k: v for k, v in vol.items() if k in ("vol", "seg", "root")}
            entry["dataset"] = self._dataset_name(vol)
            try:
                entry["prompt"] = prompt_from_volume_spec(vol, task=task)
            except ValueError as exc:
                # Volumes that omit ``imaging`` skip the text prompt.
                logger.debug("Skipping prompt for %s: %s", vol.get("vol"), exc)
            lazy_vols.append(entry)
        return LazyVolDataset(
            root_dir=self.data_root,
            volumes=lazy_vols,
            patch_size=native_patch,
            transform=self._group_transform(
                task, native_res, label_convention=label_convention,
                augment=augment,
            ),
            num_samples=num_samples,
            # sft: gate on BOTH label and image at sft_min_foreground.
            # ssl: image-only gate (label-less) at ssl_min_foreground.
            min_foreground=(self.sft_min_foreground if task == "sft" else 0.0),
            image_min_foreground=(
                self.sft_min_foreground if task == "sft"
                else (self.ssl_min_foreground if task == "ssl" else 0.0)
            ),
            # Contrast + structure gates only for the ssl branch (label-less recon).
            image_min_std=(self.ssl_min_std if task == "ssl" else 0.0),
            image_min_autocorr=(self.ssl_min_autocorr if task == "ssl" else 0.0),
            # Instance-diversity gates only for the sft branch (labeled).
            min_instances=(self.sft_min_instances if task == "sft" else 0),
            max_inst_frac=(self.sft_max_inst_frac if task == "sft" else 0.0),
            large_object_prob=(
                self.sft_large_object_prob if task == "sft" else 0.0
            ),
            large_object_min_frac=self.sft_large_object_min_frac,
            deterministic=deterministic,
            volume_sampling=volume_sampling,
            quiet=quiet,
        )

    @staticmethod
    def _group_by_res(
        volumes: List[Dict[str, Any]],
    ) -> Dict[Tuple[float, ...], List[Dict[str, Any]]]:
        groups: Dict[Tuple[float, ...], List[Dict[str, Any]]] = {}
        for vol in volumes:
            res = tuple(float(r) for r in vol["native_resolution"])
            groups.setdefault(res, []).append(vol)
        return groups

    @staticmethod
    def _group_by_res_and_convention(
        volumes: List[Dict[str, Any]],
    ) -> Dict[Tuple[Tuple[float, ...], Optional[str]], List[Dict[str, Any]]]:
        """Bucket by ``(native_resolution, label_convention)``.

        ``label_convention`` is validated when present; missing -> ``None``
        (legacy configs).  Mixing ``gapped`` and ``filled`` at the same
        resolution must not share a transform (FindBoundariesd gating).
        """
        groups: Dict[
            Tuple[Tuple[float, ...], Optional[str]], List[Dict[str, Any]]
        ] = {}
        for vol in volumes:
            res = tuple(float(r) for r in vol["native_resolution"])
            conv = validate_label_convention(
                vol.get("label_convention"), vol=str(vol.get("vol")),
            )
            groups.setdefault((res, conv), []).append(vol)
        return groups

    @staticmethod
    def _derive_subset(vol_name: str) -> str:
        """Best-effort subset (domain) name from a volume stem.

        Covers the on-disk naming conventions in ``docs/DATASETS.md``.  A
        volume may override this with an explicit ``subset:`` field in its
        config spec.
        """
        n = vol_name
        for suf in ("_volume", "_segmentation"):
            if n.endswith(suf):
                n = n[: -len(suf)]
        # SNEMI3D challenge volumes: train_*/test_* stems
        if n.startswith("train_"):
            return "train"
        if n.startswith("test_"):
            return "test"
        # Neurons / neurite cylinder
        if n.startswith("neurons") or n.startswith("neurite"):
            return "neurons"
        # MICrONS
        if n.startswith("minnie65"):
            return "minnie65"
        # FlyWire FAFB v783
        if n.startswith("flywire_"):
            return "flywire"
        # CREMI: cremi3d_sample_A / A+ -> cremi3d (train) / cremi3d_test (image-only)
        m = re.match(r"^(cremi3d)_sample_[A-Za-z]+(\+?)", n)
        if m:
            return "cremi3d_test" if m.group(2) == "+" else "cremi3d"
        # FLYEM3D members (hemibrain / malecns carry the name; rest is FIB-25)
        if n.startswith("flyem3d_hemibrain"):
            return "flyem3d_hemibrain"
        if n.startswith("flyem3d_malecns"):
            return "flyem3d_malecns"
        if re.match(r"^flyem3d_\d+nm_ssl", n):
            return "flyem3d_fib25_ssl"
        if n.startswith("flyem3d"):
            return "flyem3d_fib25"
        # COSEM: jrc_<id>_<rx>x<ry>x<rz>nm_... -> jrc_<id>
        m = re.match(r"^(jrc_[A-Za-z0-9\-]+?)_\d+x\d", n)
        if m:
            return m.group(1)
        # MitoEM2: mitoem2_<subset>_train01 / test01
        m = re.match(r"^(mitoem2_[a-z]+)_(?:train|test)\d+", n)
        if m:
            return m.group(1)
        # Fallback: drop a trailing _x{X}_y{Y}_z{Z} coordinate block.
        m = re.match(r"^(.*?)_x\d+_y\d+_z\d+", n)
        return m.group(1) if m else n

    def _iter_train_groups(
        self, task: str, vols: List[Dict[str, Any]],
    ) -> List[Tuple[str, Tuple[float, ...], List[Dict[str, Any]], float, Optional[str]]]:
        """Bucket a branch's volumes into ``(desc, res, vols, multiplier, convention)``
        round-robin groups per the ``balance`` policy.  ``multiplier`` scales
        the group's schedule length (1.0 = the branch's base share)."""
        out: List[
            Tuple[str, Tuple[float, ...], List[Dict[str, Any]], float, Optional[str]]
        ] = []
        if self.balance == "resolution":
            for (res, conv), gvols in self._group_by_res_and_convention(vols).items():
                tag = f"res={res}" + (f" conv={conv}" if conv else "")
                out.append((tag, res, gvols, 1.0, conv))
        elif self.balance == "subset":
            keyed: Dict[
                Tuple[str, Tuple[float, ...], Optional[str]], List[Dict[str, Any]]
            ] = {}
            for v in vols:
                sub = str(v.get("subset") or self._derive_subset(v["vol"]))
                res = tuple(float(r) for r in v["native_resolution"])
                conv = validate_label_convention(
                    v.get("label_convention"), vol=str(v.get("vol")),
                )
                keyed.setdefault((sub, res, conv), []).append(v)
            for (sub, res, conv), gvols in keyed.items():
                mult = self.subset_weights.get(sub, 1.0)
                tag = f"subset={sub}" + (f" conv={conv}" if conv else "")
                out.append((tag, res, gvols, mult, conv))
        else:  # "volume": one group per volume, equally likely (x sample_weight)
            for v in vols:
                res = tuple(float(r) for r in v["native_resolution"])
                conv = validate_label_convention(
                    v.get("label_convention"), vol=str(v.get("vol")),
                )
                mult = float(v.get("sample_weight", 1.0))
                out.append((f"vol={v['vol']}", res, [v], mult, conv))
        return out

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    @staticmethod
    def _dataset_name(vol: Dict[str, Any]) -> str:
        """Dataset key: the ``dataset`` field, else the root directory's name."""
        name = vol.get("dataset")
        if name:
            return str(name)
        root = str(vol.get("root", "")).rstrip("/\\")
        base = Path(root).name
        return base or "unknown"

    def _dist(self) -> Tuple[int, int]:
        """Return ``(global_rank, world_size)``, or ``(0, 1)`` before fit.

        Lightning sets ``datamodule.trainer`` in ``attach_datamodule``.
        ``_trainer`` is the module slot and stays unset here, so reading
        it made every rank use rank 0.
        """
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 0, 1
        return (
            int(getattr(trainer, "global_rank", 0)),
            int(getattr(trainer, "world_size", 1) or 1),
        )

    @staticmethod
    def _tissue_nm3(ds: LazyVolDataset, res: Sequence[float]) -> Tuple[int, float]:
        """Return ``(spatial voxels, tissue in nm^3)`` for a built group."""
        voxels = 0
        for handle in ds._handles:
            voxels += int(math.prod(int(s) for s in ds._spatial_shape(handle)))
        voxel_nm3 = float(math.prod(float(r) for r in res))
        return voxels, float(voxels) * voxel_nm3

    def _log_balance_table(self, rows: List[Dict[str, Any]], world: int) -> None:
        """Log the dataset-balance table on rank 0."""
        if self._dist()[0] != 0:
            return
        lines = [
            f"Dataset balance (within_dataset={self.within_dataset}, "
            f"world_size={world}). Share is the fraction of batches. "
            "passes/50k counts every rank's crops over the sub-group tissue.",
            f"{'branch':<8} {'dataset':<12} {'subgroup':<24} {'vols':>5} "
            f"{'tissue_um3':>12} {'share':>8} {'passes/50k':>11}",
        ]
        for row in rows:
            lines.append(
                f"{row['branch']:<8} {row['dataset']:<12} {row['subgroup']:<24} "
                f"{row['nvol']:>5d} {row['tissue_um3']:12.3f} "
                f"{100.0 * row['share']:7.2f}% {row['passes']:11.3f}"
            )

        def _iso(res: Sequence[float], nm: float) -> bool:
            return all(abs(float(r) - nm) <= 1e-3 for r in res)

        share_8 = sum(row["share"] for row in rows if _iso(row["res"], 8.0))
        share_16 = sum(row["share"] for row in rows if _iso(row["res"], 16.0))
        lines.append(
            "Isotropic crops that skip the resolution step: "
            f"8 nm is {100.0 * share_8:.2f}% of batches, "
            f"16 nm is {100.0 * share_16:.2f}%."
        )
        logger.info("\n".join(lines))

    def _setup_dataset_balance(self) -> None:
        """Give every dataset an equal share of its branch's batches.

        Groups are built first (``num_samples=1``) so their shapes are
        known, then lengths are set from the allocated batch counts.
        A branch's share depends only on ``sample_weight``.
        """
        pending: List[Dict[str, Any]] = []
        branch_order: List[str] = []
        for task, bcfg in self.branches.items():
            vols = [dict(v) for v in bcfg.get("volumes", [])]
            if not vols:
                continue
            configured_bs = int(bcfg.get("batch_size", 2))
            weight = float(bcfg.get("sample_weight", 1.0))
            keyed: Dict[
                Tuple[str, Tuple[float, ...], Optional[str]], List[Dict[str, Any]]
            ] = {}
            order: List[Tuple[str, Tuple[float, ...], Optional[str]]] = []
            for vol in vols:
                name = self._dataset_name(vol)
                res = tuple(float(r) for r in vol["native_resolution"])
                conv = validate_label_convention(
                    vol.get("label_convention"), vol=str(vol.get("vol")),
                )
                key = (name, res, conv)
                if key not in keyed:
                    order.append(key)
                    keyed[key] = []
                keyed[key].append(vol)
            built_any = False
            sampling = "uniform" if self.within_dataset == "volume" else "voxels"
            for name, res, conv in order:
                gvols = keyed[(name, res, conv)]
                group_bs = configured_bs
                # Labels stay on the native grid, so a permutation changes their
                # shape: one permutation per batch keeps them stackable.
                anisotropic = len({float(r) for r in res}) > 1
                shared_perm = (
                    self.rand_transpose_zyx and group_bs > 1
                    and anisotropic and task != "ssl"
                )
                if shared_perm:
                    logger.info(
                        "rand_transpose_zyx: task=%s dataset=%s native_resolution=%s "
                        "batch_size %d shares one z/y/x permutation per batch",
                        task, name, list(res), group_bs,
                    )
                try:
                    ds = self._build_group(
                        task, res, gvols, 1, deterministic=False,
                        label_convention=conv, augment=self.rand_transpose_zyx,
                        volume_sampling=sampling, quiet=True,
                    )
                except ValueError:
                    logger.warning(
                        "Joint train group skipped (no usable volumes): "
                        "task=%s dataset=%s res=%s", task, name, res,
                    )
                    continue
                voxels, tissue = self._tissue_nm3(ds, res)
                pending.append({
                    "task": task,
                    "dataset": name,
                    "res": res,
                    "conv": conv,
                    "ds": ds,
                    "group_bs": group_bs,
                    "shared_perm": shared_perm,
                    "configured_bs": configured_bs,
                    "nvol": len(ds._handles),
                    "voxels": voxels,
                    "tissue_nm3": tissue,
                    "weight": weight,
                })
                built_any = True
            if built_any:
                branch_order.append(task)
        if not pending:
            raise ValueError(
                "Joint3DDataModule: no train volumes configured under data.branches."
            )

        by_branch: Dict[str, List[Dict[str, Any]]] = {}
        for item in pending:
            by_branch.setdefault(item["task"], []).append(item)
        weights = []
        scales = []
        for task in branch_order:
            items = by_branch[task]
            weights.append(float(items[0]["weight"]))
            bs = max(1, int(items[0]["configured_bs"]))
            scales.append(max(1, int(round(self.num_samples * weights[-1] / bs))))
        branch_batches = _largest_remainder(weights, sum(scales))

        allocated: List[Tuple[Dict[str, Any], int]] = []
        kept = 0
        for task, n_branch in zip(branch_order, branch_batches):
            items = by_branch[task]
            ds_order: List[str] = []
            ds_items: Dict[str, List[Dict[str, Any]]] = {}
            for item in items:
                ds_items.setdefault(item["dataset"], []).append(item)
                if item["dataset"] not in ds_order:
                    ds_order.append(item["dataset"])
            ds_w = [float(self.dataset_weights.get(name, 1.0)) for name in ds_order]
            ds_batches = _largest_remainder(ds_w, int(n_branch))
            for name, n_ds in zip(ds_order, ds_batches):
                subs = ds_items[name]
                if self.within_dataset == "volume":
                    sub_w = [float(item["nvol"]) for item in subs]
                else:
                    sub_w = [float(item["tissue_nm3"]) for item in subs]
                sub_batches = _largest_remainder(sub_w, int(n_ds))
                for item, n_b in zip(subs, sub_batches):
                    allocated.append((item, int(n_b)))
                    kept += int(n_b)
        if kept <= 0:
            raise ValueError(
                "Joint3DDataModule: dataset balance assigned no training batches."
            )

        datasets: List[LazyVolDataset] = []
        group_specs: List[Tuple[int, int, int, bool]] = []
        group_dataset: List[str] = []
        group_branch: List[str] = []
        rows: List[Dict[str, Any]] = []
        offset = 0
        world = self._dist()[1]
        for item, n_b in allocated:
            if n_b <= 0:
                continue
            length = n_b * int(item["group_bs"])
            item["ds"].num_samples = length
            datasets.append(item["ds"])
            group_specs.append(
                (offset, length, int(item["group_bs"]), bool(item["shared_perm"]))
            )
            offset += length
            group_dataset.append(item["dataset"])
            group_branch.append(item["task"])
            share = n_b / float(kept)
            crop = int(math.prod(
                int(s) for s in _scaled(self.fine_patch, self.fine_nm, item["res"])
            ))
            voxels = int(item["voxels"])
            passes = (
                50000.0 * world * int(item["group_bs"]) * share * crop / voxels
                if voxels > 0 else 0.0
            )
            subgroup = "x".join(f"{v:g}" for v in item["res"])
            if item["conv"]:
                subgroup = f"{subgroup} {item['conv']}"
            rows.append({
                "branch": item["task"],
                "dataset": item["dataset"],
                "res": item["res"],
                "conv": item["conv"],
                "subgroup": subgroup,
                "nvol": int(item["nvol"]),
                "tissue_um3": float(item["tissue_nm3"]) / 1e9,
                "tissue_nm3": float(item["tissue_nm3"]),
                "share": share,
                "passes": passes,
                "n_batches": n_b,
            })
            logger.info(
                "Joint train group: task=%s dataset=%s subgroup=%s vols=%d "
                "native_patch=%s len=%d bs=%d share=%.4f transpose_zyx=%s",
                item["task"], item["dataset"], subgroup, item["nvol"],
                _scaled(self.fine_patch, self.fine_nm, item["res"]),
                length, item["group_bs"], share, self.rand_transpose_zyx,
            )
        self.train_dataset = _PermConcatDataset(datasets)
        self._train_groups = group_specs
        self._train_group_dataset = group_dataset
        self._train_group_branch = group_branch
        self.balance_rows = rows
        self._log_balance_table(rows, world)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.balance == "dataset":
            self._setup_dataset_balance()
        else:
            self._setup_legacy_train()
        self._setup_val()

    def _setup_legacy_train(self) -> None:
        """resolution / subset / volume grouping. Lengths follow the old formula."""
        self.balance_rows = []
        self._train_group_dataset = []
        self._train_group_branch = []
        datasets: List[LazyVolDataset] = []
        group_specs: List[Tuple[int, int, int, bool]] = []
        offset = 0
        for task, bcfg in self.branches.items():
            bs = int(bcfg.get("batch_size", 2))
            weight = float(bcfg.get("sample_weight", 1.0))
            vols = [dict(v) for v in bcfg.get("volumes", [])]
            branch_groups = self._iter_train_groups(task, vols)
            # Normalise multipliers by the branch mean so a group's length is
            # ``num_samples * sample_weight`` when all weights are equal -- i.e.
            # weights are RELATIVE ratios, and the branch's total virtual-epoch
            # budget stays fixed regardless of how the weights are set (a high
            # ``subset_weights`` simply reallocates the budget toward that
            # subset rather than inflating the epoch).
            mults = [m for _, _, _, m, _ in branch_groups]
            mean_mult = (sum(mults) / len(mults)) if mults else 1.0
            if mean_mult <= 0:
                mean_mult = 1.0
            for desc, res, gvols, mult, conv in branch_groups:
                n_group = max(bs, int(round(self.num_samples * weight * mult / mean_mult)))
                group_bs = bs
                # The recon target is nearest-upsampled onto the cubic fine
                # grid before collation, so an ssl batch stacks.  Labels stay
                # native: a zyx permutation changes an anisotropic label's
                # shape, so such a batch shares ONE permutation to stack.
                anisotropic = len(set(float(r) for r in res)) > 1
                shared_perm = (
                    self.rand_transpose_zyx and group_bs > 1
                    and anisotropic and task != "ssl"
                )
                if shared_perm:
                    logger.info(
                        "rand_transpose_zyx: task=%s %s native_resolution=%s "
                        "batch_size %d shares one z/y/x permutation per batch",
                        task, desc, list(res), group_bs,
                    )
                try:
                    ds = self._build_group(
                        task, res, gvols, n_group, deterministic=False,
                        label_convention=conv,
                        augment=self.rand_transpose_zyx,
                    )
                except ValueError:
                    # All volumes in the group were skipped (e.g. smaller than
                    # the native patch on some axis).  Skip the empty group
                    # rather than aborting the whole run.
                    logger.warning(
                        "Joint train group skipped (no usable volumes): "
                        "task=%s %s", task, desc,
                    )
                    continue
                datasets.append(ds)
                group_specs.append((offset, len(ds), group_bs, shared_perm))
                offset += len(ds)
                logger.info(
                    "Joint train group: task=%s %s vols=%d native_patch=%s "
                    "len=%d bs=%d mult=%.2f transpose_zyx=%s", task, desc,
                    len(gvols), _scaled(self.fine_patch, self.fine_nm, res),
                    len(ds), group_bs, mult, self.rand_transpose_zyx,
                )
        if not datasets:
            raise ValueError("Joint3DDataModule: no train volumes configured under data.branches.")
        self.train_dataset = _PermConcatDataset(datasets)
        self._train_groups = group_specs

    def _setup_val(self) -> None:
        # Validation: explicit val_volumes (flat, each tagged with task +
        # native_resolution), else the sft branch volumes (labeled -> metrics).
        val_vols = self.val_volumes
        if not val_vols:
            sft = self.branches.get("sft", {})
            val_vols = [dict(v, task="sft") for v in sft.get("volumes", [])]
        self.val_dataset_names = (
            sorted({self._dataset_name(v) for v in val_vols}) if val_vols else []
        )
        v_datasets: List[LazyVolDataset] = []
        v_specs: List[Tuple[int, int, int]] = []
        v_names: List[str] = []
        v_off = 0
        by_task: Dict[str, List[Dict[str, Any]]] = {}
        for v in val_vols:
            by_task.setdefault(v.get("task", "sft"), []).append(v)
        for task, vols in by_task.items():
            for (res, conv), gvols in self._group_by_res_and_convention(vols).items():
                if self.val_balance == "dataset":
                    buckets = {}
                    for vol in gvols:
                        buckets.setdefault(self._dataset_name(vol), []).append(vol)
                    grouped = [(name, buckets[name]) for name in sorted(buckets)]
                else:
                    grouped = [("", gvols)]
                for name, members in grouped:
                    # Uniform: each listed val volume is a representative.
                    # Voxel weighting hides the smaller volume that shares a grid.
                    sampling = "uniform" if self.val_balance == "dataset" else "voxels"
                    ds = self._build_group(
                        task, res, members, self.val_num_samples, deterministic=True,
                        label_convention=conv, volume_sampling=sampling,
                    )
                    v_datasets.append(ds)
                    v_specs.append((v_off, len(ds), self.val_batch_size))
                    v_names.append(name or self._dataset_name(members[0]))
                    v_off += len(ds)
        if v_datasets:
            self.val_dataset = ConcatDataset(v_datasets)
            self._val_groups = v_specs
            self._val_group_dataset = v_names

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def _loader(self, dataset, groups, shuffle, shard: bool = False):
        rank, world = self._dist()
        init = partial(worker_init, rank=rank) if self.num_workers > 0 else None
        return DataLoader(
            dataset,
            batch_sampler=_RoundRobinBatchSampler(
                groups, shuffle=shuffle, seed=self.seed,
                rank=rank, world_size=world, shard=shard,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
            worker_init_fn=init,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self._train_groups, shuffle=True, shard=False)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return self._loader(self.val_dataset, self._val_groups, shuffle=False, shard=True)


__all__ = ["Joint3DDataModule"]
