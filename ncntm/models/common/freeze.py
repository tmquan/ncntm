"""``freeze_dit_backbone`` config parsing."""

from typing import Optional, Tuple, Union


def resolve_freeze_dit_backbone(
    value: Union[bool, int, None],
) -> Tuple[bool, Optional[int]]:
    """Parse ``freeze_dit_backbone`` into ``(initial_frozen, thaw_epoch)``.

    Accepted forms:

    * ``True`` / ``False`` -- permanent state.  ``thaw_epoch`` is ``None``.
    * non-negative ``int`` ``N`` (and not a ``bool``) -- DiT is frozen
      for epochs ``0..N-1`` and unfrozen at the start of epoch ``N``.
      ``N == 0`` is equivalent to ``False`` (never frozen).

    Raises:
        TypeError: ``value`` is neither ``bool`` nor ``int`` (and not ``None``).
        ValueError: ``value`` is a negative integer.
    """
    if value is None or isinstance(value, bool):
        return bool(value), None
    if isinstance(value, int):
        if value < 0:
            raise ValueError(
                "freeze_dit_backbone integer schedule must be non-negative "
                f"(got {value}).  Use ``N >= 0`` for 'freeze epochs 0..N-1, "
                "thaw at epoch N', or pass a bool for permanent state."
            )
        if value == 0:
            return False, None
        return True, value
    raise TypeError(
        "freeze_dit_backbone must be a bool or non-negative int "
        f"(got {type(value).__name__}: {value!r})."
    )
