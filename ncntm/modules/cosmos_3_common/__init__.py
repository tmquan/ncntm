"""Shared Cosmos 3 family Lightning-module package.

Re-exports :class:`BaseCosmosModule` (the shared base, which lives in
:mod:`ncntm.modules.cosmos_2_5_common`).  The concrete Lightning modules
(:mod:`ncntm.modules.cosmos_3_nano`, :mod:`ncntm.modules.joint3d`) are short
declarations of which ``model_cls`` to wire to the shared loop.
"""

from ncntm.modules.cosmos_3_common.base import BaseCosmosModule

__all__ = ["BaseCosmosModule"]
