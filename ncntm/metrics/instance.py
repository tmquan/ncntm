"""
Instance segmentation metrics for connectomics evaluation.

Provides per-point and per-batch variants of:
- ARI  (Adjusted Rand Index)
- AMI  (Adjusted Mutual Information)
- VOI  (Variation of Information, with split/merge decomposition)
- TED  (Tolerant Edit Distance -- min split+merge corrections)
- large-object recall / IoU (best-match, large GT objects only)
"""

from typing import NamedTuple, Tuple

import numpy as np
import torch


# ======================================================================
# Helpers
# ======================================================================

def _prepare_flat_labels(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten and filter label tensors for metric computation."""
    pred_flat = pred_labels.cpu().numpy().ravel()
    true_flat = true_labels.cpu().numpy().ravel()

    if ignore_background:
        fg_mask = (pred_flat > 0) | (true_flat > 0)
        if not fg_mask.any():
            return np.array([]), np.array([])
        pred_flat = pred_flat[fg_mask]
        true_flat = true_flat[fg_mask]

    return pred_flat, true_flat


def _contingency_table(
    pred: np.ndarray,
    true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build contingency matrix and marginals.

    Returns (contingency, pred_counts, true_counts).
    """
    from scipy.sparse import coo_matrix

    n = len(pred)
    cont = coo_matrix(
        (np.ones(n, dtype=np.float64), (true.astype(np.int64), pred.astype(np.int64))),
        shape=(true.max() + 1, pred.max() + 1),
    ).toarray()
    true_counts = cont.sum(axis=1)
    pred_counts = cont.sum(axis=0)
    return cont, pred_counts, true_counts


# ======================================================================
# ARI
# ======================================================================

def compute_per_point_ari(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """Adjusted Rand Index for a single sample.

    Args:
        pred_labels: Predicted instance labels [H, W] or [D, H, W].
        true_labels: Ground truth instance labels, same shape.
        ignore_background: Exclude background (label 0).

    Returns:
        ARI in [0, 1] (clamped).
    """
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return 0.0

    pred_flat, true_flat = _prepare_flat_labels(pred_labels, true_labels, ignore_background)
    if len(pred_flat) == 0:
        return 0.0
    return max(0.0, adjusted_rand_score(true_flat, pred_flat))


def compute_per_batch_ari(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """ARI averaged over a batch [B, ...]."""
    total, count = 0.0, 0
    for b in range(pred_labels.shape[0]):
        total += compute_per_point_ari(pred_labels[b], true_labels[b], ignore_background)
        count += 1
    return total / count if count > 0 else 0.0


# ======================================================================
# Large-object recall
# ======================================================================

def compute_large_object_recall(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    min_frac: float = 0.5,
) -> Tuple[float, float, int]:
    """Recall / IoU of the best-matching predicted segment for large GT objects.

    A GT object is *large* when its voxels are at least ``min_frac`` of the
    labelled foreground of its crop -- the same definition the sampler's
    large-object draw uses (``LazyVolDataset.large_object_min_frac``).  For each
    large object we take the predicted segment that overlaps it most and report
    ``recall = |g & p| / |g|`` and ``IoU = |g & p| / |g | p|``.  Recall says
    whether the object is kept as one piece; IoU also penalises merging it with
    neighbours.  Background (0) is never matched.

    Args:
        pred_labels: Predicted instance labels [B, D, H, W] (MWS, 0 = bg).
        true_labels: GT instance labels, same shape (0 = bg).
        min_frac: Largest-object share of the crop foreground to count as large.

    Returns:
        ``(recall_sum, iou_sum, n_objects)`` pooled over the batch, so the caller
        can weight the epoch mean by the number of objects.  ``n_objects`` is 0
        when no crop has a large object.
    """
    rec_sum, iou_sum, n_obj = 0.0, 0.0, 0
    for b in range(pred_labels.shape[0]):
        gt = true_labels[b].cpu().numpy().ravel()
        pr = pred_labels[b].cpu().numpy().ravel()
        ids, counts = np.unique(gt[gt > 0], return_counts=True)
        if ids.size == 0:
            continue
        large = ids[counts >= min_frac * counts.sum()]
        for g, g_size in zip(large, counts[counts >= min_frac * counts.sum()]):
            p_vals = pr[gt == g]
            p_ids, p_cnt = np.unique(p_vals[p_vals > 0], return_counts=True)
            rec, iou = 0.0, 0.0
            if p_ids.size:
                j = int(np.argmax(p_cnt))
                inter = float(p_cnt[j])
                p_size = float(np.count_nonzero(pr == p_ids[j]))
                rec = inter / float(g_size)
                iou = inter / (float(g_size) + p_size - inter)
            rec_sum += rec
            iou_sum += iou
            n_obj += 1
    return rec_sum, iou_sum, n_obj


# ======================================================================
# AMI
# ======================================================================

def compute_per_point_ami(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """Adjusted Mutual Information for a single sample.

    Args:
        pred_labels: Predicted instance labels [H, W] or [D, H, W].
        true_labels: Ground truth instance labels, same shape.
        ignore_background: Exclude background (label 0).

    Returns:
        AMI in [0, 1] (clamped).
    """
    try:
        from sklearn.metrics import adjusted_mutual_info_score
    except ImportError:
        return 0.0

    pred_flat, true_flat = _prepare_flat_labels(pred_labels, true_labels, ignore_background)
    if len(pred_flat) == 0:
        return 0.0
    return max(0.0, adjusted_mutual_info_score(true_flat, pred_flat))


def compute_per_batch_ami(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """AMI averaged over a batch [B, ...]."""
    total, count = 0.0, 0
    for b in range(pred_labels.shape[0]):
        total += compute_per_point_ami(pred_labels[b], true_labels[b], ignore_background)
        count += 1
    return total / count if count > 0 else 0.0


# ======================================================================
# VOI  (Variation of Information)
# ======================================================================

class VOIResult(NamedTuple):
    """Variation of Information decomposed into split and merge errors."""
    split: float
    merge: float
    total: float


def compute_per_point_voi(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> VOIResult:
    """Variation of Information for a single sample.

    VOI = H(pred|true) + H(true|pred)  (standard Meila convention)
      - H(pred|true) = *split* error  (over-segmentation: one GT object
        cut into several predicted segments)
      - H(true|pred) = *merge* error  (under-segmentation: one predicted
        segment covering several GT objects)

    Args:
        pred_labels: Predicted instance labels [H, W] or [D, H, W].
        true_labels: Ground truth instance labels, same shape.
        ignore_background: Exclude background (label 0).

    Returns:
        VOIResult(split, merge, total).  Lower is better; 0 = perfect.
    """
    pred_flat, true_flat = _prepare_flat_labels(pred_labels, true_labels, ignore_background)
    if len(pred_flat) == 0:
        return VOIResult(0.0, 0.0, 0.0)

    cont, pred_counts, true_counts = _contingency_table(pred_flat, true_flat)
    n = float(len(pred_flat))

    nz = cont > 0

    # H(true|pred) -- MERGE error (under-segmentation): a predicted segment
    # spanning several GT objects leaves the GT label uncertain given pred.
    pred_counts_broadcast = np.broadcast_to(pred_counts[np.newaxis, :], cont.shape)
    h_true_given_pred = -np.sum((cont[nz] / n) * np.log2(cont[nz] / pred_counts_broadcast[nz]))

    # H(pred|true) -- SPLIT error (over-segmentation): a GT object cut into
    # several predicted segments leaves the pred label uncertain given GT.
    true_counts_broadcast = np.broadcast_to(true_counts[:, np.newaxis], cont.shape)
    h_pred_given_true = -np.sum((cont[nz] / n) * np.log2(cont[nz] / true_counts_broadcast[nz]))

    return VOIResult(
        split=float(h_pred_given_true),
        merge=float(h_true_given_pred),
        total=float(h_true_given_pred + h_pred_given_true),
    )


def compute_per_batch_voi(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> VOIResult:
    """VOI averaged over a batch [B, ...]."""
    splits, merges = [], []
    for b in range(pred_labels.shape[0]):
        r = compute_per_point_voi(pred_labels[b], true_labels[b], ignore_background)
        splits.append(r.split)
        merges.append(r.merge)
    s = float(np.mean(splits)) if splits else 0.0
    m = float(np.mean(merges)) if merges else 0.0
    return VOIResult(split=s, merge=m, total=s + m)


# ======================================================================
# TED  (Tolerant Edit Distance)
# ======================================================================

def compute_per_point_ted(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """Tolerant Edit Distance for a single sample.

    Counts the minimum number of split and merge operations needed to
    transform *pred* into *true*.  A GT segment mapped to K > 1 predicted
    segments costs (K - 1) splits; a predicted segment covering M > 1 GT
    segments costs (M - 1) merges.

    Args:
        pred_labels: Predicted instance labels [H, W] or [D, H, W].
        true_labels: Ground truth instance labels, same shape.
        ignore_background: Exclude background (label 0).

    Returns:
        TED score: a non-negative float (an integer-valued edit count
        returned as ``float``).  0 = perfect match.
    """
    pred_flat, true_flat = _prepare_flat_labels(pred_labels, true_labels, ignore_background)
    if len(pred_flat) == 0:
        return 0.0

    cont, _, _ = _contingency_table(pred_flat, true_flat)

    n_splits = 0
    for i in range(cont.shape[0]):
        row = cont[i]
        overlapping = (row > 0).sum()
        if overlapping > 1:
            n_splits += int(overlapping - 1)

    n_merges = 0
    for j in range(cont.shape[1]):
        col = cont[:, j]
        overlapping = (col > 0).sum()
        if overlapping > 1:
            n_merges += int(overlapping - 1)

    return float(n_splits + n_merges)


def compute_per_batch_ted(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """TED averaged over a batch [B, ...]."""
    total, count = 0.0, 0
    for b in range(pred_labels.shape[0]):
        total += compute_per_point_ted(pred_labels[b], true_labels[b], ignore_background)
        count += 1
    return total / count if count > 0 else 0.0
