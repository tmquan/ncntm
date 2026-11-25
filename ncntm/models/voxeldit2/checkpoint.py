"""Load a released 2-D PixelDiT2 checkpoint into :class:`VoxelDiT2_3D`.

The transformer blocks, timestep MLP, in-context position embedding, final
adaLN and the grounding projector (``dino_conditioner.dino_proj``) do not
depend on the patch geometry and are copied as they are. Three tensors are
inflated:

* ``s_embedder.proj.weight`` ``[hidden, 3 * p * p]``: RGB is summed into the
  single EM channel (grey replicated to RGB sees the same response), the
  in-plane kernel is resized when the 3-D patch differs from ``p``, and the
  kernel is spread over ``pz`` sections and divided by ``pz``, so a volume
  that is constant along z embeds exactly like its 2-D section.
* ``final_layer.linear`` ``[p * p * 3, hidden]``: the RGB outputs are
  averaged and every section of the patch gets the in-plane prediction.
* ``y_embedder.embedding_table`` ``[1001, hidden]``: every EM class row
  starts from the ImageNet null-class row.

The DINOv3 encoder is not in the checkpoint; timm loads it. The 2-D rotary
layout is not a parameter; see :func:`ncntm.models.voxeldit2.layers.rope_axes`
for how the 3-D layout keeps most of its channels.
"""

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ncntm.models.voxeldit2.network import VoxelDiT2_3D

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_SUBDIR = "pixeldit2"
_ENCODER_PREFIX = "dino_conditioner.encoder."


def download_checkpoint(
    repo_id: Optional[str],
    filename: str,
    revision: str = "main",
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Local path of ``filename``: an existing file as is, else a rank-aware HF download.

    Rank 0 downloads while the other ranks wait at a barrier (when
    ``torch.distributed`` is initialised). The cache is ``cache_dir`` >
    ``$NCNTM_CACHE_DIR`` > ``~/.cache/ncntm/pixeldit2``.
    """
    if os.path.isfile(filename):
        return Path(filename)
    if not repo_id:
        raise FileNotFoundError(f"{filename} is not a file and no HuggingFace repo is set.")
    from huggingface_hub import hf_hub_download
    import torch.distributed as dist

    cache_dir = cache_dir or os.environ.get("NCNTM_CACHE_DIR") or str(
        Path.home() / ".cache" / "ncntm" / _DEFAULT_CACHE_SUBDIR
    )
    token = token or os.environ.get("HF_TOKEN")
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    kwargs = dict(repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir, token=token)
    if rank == 0:
        try:
            path = hf_hub_download(**kwargs)
        finally:
            if distributed:
                dist.barrier()
    else:
        dist.barrier()
        path = hf_hub_download(**kwargs, local_files_only=True)
    logger.info("PixelDiT2 checkpoint %s/%s -> %s", repo_id, filename, path)
    return Path(path)


def read_2d_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """The denoiser weights of a PixelDiT2 Lightning checkpoint (EMA when present)."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    for prefix in ("ema_denoiser.", "denoiser.", ""):
        picked = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if picked:
            return picked
    raise ValueError(f"{path} holds no denoiser weights.")


def _resize_kernel(w: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Bilinear resize of the last two dims of ``w``."""
    if tuple(w.shape[-2:]) == tuple(size):
        return w
    lead = w.shape[:-2]
    out = F.interpolate(
        w.reshape(-1, 1, *w.shape[-2:]).float(), size=size, mode="bilinear", align_corners=False,
    )
    return out.reshape(*lead, *size)


def _inflate_patch_embed(w2d: torch.Tensor, net: VoxelDiT2_3D) -> torch.Tensor:
    hidden, n_in = w2d.shape
    p2 = int(round(math.sqrt(n_in / 3)))
    if 3 * p2 * p2 != n_in:
        raise ValueError(f"s_embedder.proj.weight {tuple(w2d.shape)} is not a 3-channel square patch.")
    pz, py, px = net.patch_size
    w = w2d.float().reshape(hidden, 3, p2, p2)
    if net.in_channels == 1:
        w = w.sum(dim=1, keepdim=True)
    elif net.in_channels != 3:
        raise ValueError(f"cannot inflate a 3-channel patch embedding to {net.in_channels} channels.")
    w = _resize_kernel(w, (py, px)) * (p2 * p2) / (py * px)
    w = w.unsqueeze(2).expand(-1, -1, pz, -1, -1) / pz
    return w.reshape(hidden, -1)


def _inflate_final(t2d: torch.Tensor, net: VoxelDiT2_3D) -> torch.Tensor:
    """``[p * p * 3, ...]`` (features ``(ky, kx, c)``) -> ``[pz * py * px * C, ...]``."""
    trailing = t2d.shape[1:]
    p2 = int(round(math.sqrt(t2d.shape[0] / 3)))
    pz, py, px = net.patch_size
    t = t2d.float().reshape(p2, p2, 3, -1)
    if net.out_channels == 1:
        t = t.mean(dim=2, keepdim=True)
    t = t.permute(2, 3, 0, 1)
    t = _resize_kernel(t, (py, px)).permute(2, 3, 0, 1)
    t = t.unsqueeze(0).expand(pz, -1, -1, -1, -1)
    return t.reshape(pz * py * px * net.out_channels, *trailing)


def inflate_state_dict(
    state_2d: Dict[str, torch.Tensor], net: VoxelDiT2_3D,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """3-D state dict for ``net`` built from ``state_2d``, plus the 2-D keys that were not used."""
    target = net.state_dict()
    out: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in state_2d.items():
        if key.startswith(_ENCODER_PREFIX):
            continue
        if key == "s_embedder.proj.weight":
            new = _inflate_patch_embed(value, net)
        elif key in ("final_layer.linear.weight", "final_layer.linear.bias"):
            new = _inflate_final(value, net)
        elif key == "y_embedder.embedding_table.weight":
            rows = target[key].shape[0]
            new = value[-1:].expand(rows, -1)
        else:
            new = value
        if key not in target or tuple(new.shape) != tuple(target[key].shape):
            skipped.append(key)
            continue
        out[key] = new.to(dtype=target[key].dtype).contiguous()
    return out, skipped


def load_pretrained_2d(net: VoxelDiT2_3D, path: Path) -> Dict[str, List[str]]:
    """Inflate the checkpoint at ``path`` into ``net`` in place.

    Returns ``missing`` (3-D parameters left at their init, the frozen
    encoder excluded) and ``skipped`` (2-D tensors with no 3-D counterpart,
    e.g. blocks beyond ``net.depth``).
    """
    state, skipped = inflate_state_dict(read_2d_state_dict(path), net)
    result = net.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(_ENCODER_PREFIX)]
    logger.info(
        "VoxelDiT2-3D: inflated %d tensors from %s; %d left at init, %d 2-D tensors unused.",
        len(state), path, len(missing), len(skipped),
    )
    if missing:
        logger.warning("VoxelDiT2-3D parameters not in the checkpoint: %s", missing[:8])
    if skipped:
        logger.warning("PixelDiT2 checkpoint tensors without a 3-D match: %s", skipped[:8])
    return {"missing": missing, "skipped": skipped}


__all__ = [
    "download_checkpoint",
    "inflate_state_dict",
    "load_pretrained_2d",
    "read_2d_state_dict",
]
