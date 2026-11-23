"""AdamW that keeps fp32 master weights when the parameters are bf16.

``torch.optim.AdamW`` stores the parameter and both moments in the
parameter dtype. Under ``bf16-true`` an update smaller than the bf16
spacing is rounded away, so large weights stop moving. This class
applies AdamW to an fp32 copy and writes the rounded value back.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch


def as_dtype(value: Any) -> torch.dtype:
    """Map a config string (``fp32``, ``bf16``, ...) to a torch dtype."""
    if isinstance(value, torch.dtype):
        return value
    name = str(value).lower().replace("torch.", "")
    table = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if name not in table:
        raise ValueError(
            f"master_dtype must be fp32, bf16, or fp16; got {value!r}."
        )
    return table[name]


def master_dtype_from_config(config: dict) -> Optional[torch.dtype]:
    """Return the master dtype, or ``None`` to keep plain AdamW.

    ``null`` / ``none`` / a missing key keep the historical optimiser.
    """
    if "master_dtype" not in config:
        return None
    value = config.get("master_dtype")
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
        return None
    return as_dtype(value)


class MasterAdamW(torch.optim.Optimizer):
    """AdamW that keeps an fp32 master copy of every parameter and fp32 moments.

    The parameter itself may be bf16: after each step the master is written
    back into it (rounded). Updates smaller than the bf16 spacing therefore
    accumulate in the master instead of being rounded away.

    ``fused`` does not apply. ``configure_gradient_clipping`` still walks
    ``param_groups``. After an external weight load that runs once a step
    has already created masters, call :meth:`resync` so the masters match
    the loaded parameters. A weights-only warm start that loads before the
    first step does not need it: the masters are created from those weights.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        master_dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(master_dtype, str):
            master_dtype = as_dtype(master_dtype)
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )
        self.master_dtype = master_dtype

    @torch.no_grad()
    def resync(self) -> None:
        """Re-read masters from the parameters (call after an external weight load)."""
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st and "master" in st:
                    st["master"].copy_(p.detach().to(self.master_dtype))

    def load_state_dict(self, state_dict):
        """torch casts optimizer state to the PARAMETER dtype on load (bf16 here).

        Restore the master and the moments from the raw dict in master_dtype.
        """
        super().load_state_dict(state_dict)
        raw = state_dict["state"]
        saved_ids = [i for g in state_dict["param_groups"] for i in g["params"]]
        params = [p for g in self.param_groups for p in g["params"]]
        for sid, p in zip(saved_ids, params):
            if sid not in raw:
                continue
            for k in ("master", "exp_avg", "exp_avg_sq"):
                if k in raw[sid]:
                    self.state[p][k] = (
                        raw[sid][k]
                        .detach()
                        .to(device=p.device, dtype=self.master_dtype)
                        .clone()
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Set by the Lightning module on log steps. Off, the update below
        # is the appendix algorithm with no extra ops on the master.
        track_ids = (
            getattr(self, "_dit_ids", None)
            if getattr(self, "_track_update", False)
            else None
        )
        sq_delta = None
        sq_base = None
        self._last_update_ratio = None
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = (
                group["lr"], group["betas"], group["eps"], group["weight_decay"],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "master" not in st:  # also covers a legacy AdamW state_dict
                    st["master"] = p.detach().to(self.master_dtype).clone()
                    st.setdefault("step", 0)
                    st["exp_avg"] = (
                        st["exp_avg"].to(self.master_dtype)
                        if "exp_avg" in st
                        else torch.zeros_like(st["master"])
                    )
                    st["exp_avg_sq"] = (
                        st["exp_avg_sq"].to(self.master_dtype)
                        if "exp_avg_sq" in st
                        else torch.zeros_like(st["master"])
                    )
                g = p.grad.to(self.master_dtype)
                st["step"] = int(st["step"]) + 1
                t = st["step"]
                m, v, w = st["exp_avg"], st["exp_avg_sq"], st["master"]
                track = track_ids is not None and id(p) in track_ids
                before = w.detach().clone() if track else None
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if wd:
                    w.mul_(1 - lr * wd)
                denom = (v.sqrt() / math.sqrt(1 - b2 ** t)).add_(eps)
                w.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))
                p.copy_(w)  # rounds to the parameter dtype
                if before is not None:
                    diff = w.float() - before.float()
                    d2 = diff.pow(2).sum()
                    b2s = before.float().pow(2).sum()
                    sq_delta = d2 if sq_delta is None else sq_delta + d2
                    sq_base = b2s if sq_base is None else sq_base + b2s
        if sq_delta is not None and sq_base is not None:
            self._last_update_ratio = float(
                torch.sqrt(sq_delta / sq_base.clamp_min(1e-30))
            )
        return loss


__all__ = ["MasterAdamW", "as_dtype", "master_dtype_from_config"]
