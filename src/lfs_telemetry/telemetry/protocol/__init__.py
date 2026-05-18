"""LFS wire protocol packets (OutSim, OutSim2, OutGauge, InSim).

Consumers should import from the submodules directly
(``from .protocol.packets import …`` / ``from .protocol.insim import …``);
this package re-exports the submodules but does not flatten their names.
"""

from . import insim, packets

__all__ = ["insim", "packets"]
