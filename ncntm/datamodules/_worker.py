"""DataLoader worker initialisation shared by the datamodules."""


def worker_init(worker_id: int, rank: int = 0) -> None:
    """Reseed torch, numpy, random, and each dataset transform for this worker.

    Lightning adds its own worker init only when the loader has none, so
    this calls ``pl_worker_init_function`` itself. ``rank`` is the process
    rank; joint loaders bind it with ``functools.partial``.
    """
    from torch.utils.data import get_worker_info

    try:
        from lightning_fabric.utilities.seed import pl_worker_init_function
    except Exception:
        from pytorch_lightning.utilities.seed import pl_worker_init_function

    pl_worker_init_function(worker_id, rank=rank)
    info = get_worker_info()
    if info is None or info.dataset is None:
        return
    subs = getattr(info.dataset, "datasets", [info.dataset])
    for i, ds in enumerate(subs):
        tf = getattr(ds, "transform", None)
        if hasattr(tf, "set_random_state"):
            tf.set_random_state(seed=(int(info.seed) + 7919 * i) % (2 ** 32))
