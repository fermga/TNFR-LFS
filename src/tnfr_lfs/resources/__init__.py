"""Bundled resources distributed with TNFR × LFS."""

from __future__ import annotations

from tnfr_lfs.resources.paths import (
    data_root,
    modifiers_root,
    pack_root,
    set_pack_root_override,
)
from tnfr_lfs.resources.tyre_compounds import (
    CAR_COMPOUND_COMPATIBILITY,
    get_allowed_compounds,
    normalise_car_model,
    normalise_compound_identifier,
)

__all__ = [
    "pack_root",
    "data_root",
    "modifiers_root",
    "set_pack_root_override",
    "CAR_COMPOUND_COMPATIBILITY",
    "get_allowed_compounds",
    "normalise_car_model",
    "normalise_compound_identifier",
]
