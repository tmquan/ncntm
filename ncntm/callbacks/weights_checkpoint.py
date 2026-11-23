"""Save trainable weights on a fixed epoch interval.

The full Lightning checkpoint also stores the frozen VAE, the frozen
reasoner, and the optimiser state. This callback writes only parameters
that require grad, in their parameter dtype, so a bf16 DiT is about
2 bytes per weight.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch


class TrainableWeightsCheckpoint(pl.Callback):
    """Write ``weights-epochXXX.pt`` every ``every_n_epochs`` epochs.

    ``save_top_k=-1`` keeps every file. A positive ``save_top_k`` keeps
    the newest files and deletes the older ones. Rank 0 writes the file.
    """

    def __init__(
        self,
        dirpath: str,
        every_n_epochs: int = 20,
        save_top_k: int = -1,
    ) -> None:
        if int(every_n_epochs) < 1:
            raise ValueError(
                f"every_n_epochs must be >= 1, got {every_n_epochs}."
            )
        self.dirpath = str(dirpath)
        self.every_n_epochs = int(every_n_epochs)
        self.save_top_k = int(save_top_k)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        if self.save_top_k == 0:
            return
        directory = Path(self.dirpath)
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            name: param.detach().to("cpu").clone()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }
        path = directory / f"weights-epoch{epoch:03d}.pt"
        torch.save({"epoch": epoch, "state_dict": state}, path)
        if self.save_top_k > 0:
            files = sorted(directory.glob("weights-epoch*.pt"))
            for old in files[: max(0, len(files) - self.save_top_k)]:
                old.unlink()


__all__ = ["TrainableWeightsCheckpoint"]
