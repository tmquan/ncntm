"""Frozen T5 encoder for the batch prompt (``{imaging} · {tail}``).

The encoder is not an ``nn.Module`` of the network, so its weights stay out
of the Lightning ``state_dict`` and checkpoints.  The prompt set is small
(one string per imaging type and label convention), so each unique string is
encoded once and its pooled feature vector is cached in memory.

Weights are fetched from the Hugging Face Hub on first use and kept in
``cache_dir`` (default: the Hugging Face cache).  Pooled features are also
written to ``<cache root>/features/<model>.pt``, so a restart or resume reads
them from disk and never loads or runs T5 for prompts it has already seen.
The cache root is ``cache_dir`` or ``~/.cache/ncntm/t5``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

_ENCODERS: Dict[tuple, "PromptEncoder"] = {}


class PromptEncoder:
    """Mean-pooled T5 encoder output for a list of prompt strings."""

    def __init__(self, name: str, cache_dir: Optional[str] = None,
                 hf_token: Optional[str] = None) -> None:
        from transformers import T5Config

        self.name = str(name)
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.dim = int(
            T5Config.from_pretrained(self.name, cache_dir=cache_dir, token=hf_token).d_model
        )
        self._model = None
        self._tokenizer = None
        self._cache: Dict[str, torch.Tensor] = {}
        self._feature_file = self._feature_path(self.name, cache_dir)
        if os.path.isfile(self._feature_file):
            self._cache.update(torch.load(self._feature_file, map_location="cpu"))
            logger.info("Loaded %d prompt features from %s",
                        len(self._cache), self._feature_file)

    @staticmethod
    def _feature_path(name: str, cache_dir: Optional[str]) -> str:
        root = cache_dir or os.path.expanduser("~/.cache/ncntm/t5")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return os.path.join(root, "features", f"{safe}.pt")

    def _save(self) -> None:
        """Atomic write, so concurrent ranks never leave a partial file."""
        os.makedirs(os.path.dirname(self._feature_file), exist_ok=True)
        tmp = f"{self._feature_file}.{os.getpid()}.tmp"
        torch.save(self._cache, tmp)
        os.replace(tmp, self._feature_file)

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoTokenizer, T5EncoderModel

        logger.info("Loading text encoder %s (cache_dir=%s)", self.name, self.cache_dir)
        kw = dict(cache_dir=self.cache_dir, token=self.hf_token)
        self._tokenizer = AutoTokenizer.from_pretrained(self.name, **kw)
        model = T5EncoderModel.from_pretrained(self.name, **kw)
        model.eval().requires_grad_(False)
        self._model = model

    @torch.no_grad()
    def _encode(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        self._load()
        batch = self._tokenizer(list(prompts), return_tensors="pt", padding=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        hidden = self._model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def __call__(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        """``[B, dim]`` float features, one row per prompt.

        Known prompts come from memory (or the feature file).  T5 runs only
        for prompts that are new, and those are then written to disk.
        """
        missing = [p for p in dict.fromkeys(prompts) if p not in self._cache]
        if missing:
            feats = self._encode(missing, device).cpu()
            for p, f in zip(missing, feats):
                self._cache[p] = f
            self._save()
        return torch.stack([self._cache[p] for p in prompts]).to(device)


def get_prompt_encoder(name: str, cache_dir: Optional[str] = None,
                       hf_token: Optional[str] = None) -> PromptEncoder:
    """Shared encoder per ``(name, cache_dir)`` so the weights load once."""
    key = (str(name), cache_dir)
    if key not in _ENCODERS:
        _ENCODERS[key] = PromptEncoder(name, cache_dir=cache_dir, hf_token=hf_token)
    return _ENCODERS[key]


__all__ = ["PromptEncoder", "get_prompt_encoder"]
