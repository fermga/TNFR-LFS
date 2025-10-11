"""Helpers for constructing EPI-related test fixtures."""

from tnfr_lfs.core.epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)

from .constants import BASE_NU_F


def build_epi_nodes(delta_nfr: float, sense_index: float):
    """Create a mapping of EPI nodes using shared baseline constants."""
    share = delta_nfr / 7
    return {
        "tyres": TyresNode(delta_nfr=share, sense_index=sense_index, nu_f=BASE_NU_F["tyres"]),
        "suspension": SuspensionNode(
            delta_nfr=share,
            sense_index=sense_index,
            nu_f=BASE_NU_F["suspension"],
        ),
        "chassis": ChassisNode(delta_nfr=share, sense_index=sense_index, nu_f=BASE_NU_F["chassis"]),
        "brakes": BrakesNode(delta_nfr=share, sense_index=sense_index, nu_f=BASE_NU_F["brakes"]),
        "transmission": TransmissionNode(
            delta_nfr=share,
            sense_index=sense_index,
            nu_f=BASE_NU_F["transmission"],
        ),
        "track": TrackNode(delta_nfr=share, sense_index=sense_index, nu_f=BASE_NU_F["track"]),
        "driver": DriverNode(delta_nfr=share, sense_index=sense_index, nu_f=BASE_NU_F["driver"]),
    }


__all__ = ["build_epi_nodes"]
