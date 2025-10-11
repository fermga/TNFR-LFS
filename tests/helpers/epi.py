"""Helpers for constructing EPI-related test fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from tnfr_lfs.core.epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    EPIBundle,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)

from .constants import BASE_NU_F

NodeFactory = Callable[..., Any]

NODE_FACTORIES: dict[str, NodeFactory] = {
    "tyres": TyresNode,
    "suspension": SuspensionNode,
    "chassis": ChassisNode,
    "brakes": BrakesNode,
    "transmission": TransmissionNode,
    "track": TrackNode,
    "driver": DriverNode,
}


def _coerce_node(
    name: str,
    sense_index: float,
    overrides: Mapping[str, Any] | TyresNode | SuspensionNode | ChassisNode | BrakesNode | TransmissionNode | TrackNode | DriverNode | None,
) -> Any:
    factory = NODE_FACTORIES[name]
    if isinstance(overrides, factory):
        return overrides

    base_kwargs: dict[str, Any] = {
        "delta_nfr": 0.0,
        "sense_index": sense_index,
        "nu_f": BASE_NU_F.get(name, 0.0),
    }

    if isinstance(overrides, Mapping):
        base_kwargs.update(overrides)
    elif overrides is not None:
        raise TypeError(f"Unsupported overrides for node '{name}': {type(overrides)!r}")

    return factory(**base_kwargs)


def build_epi_bundle(
    *,
    timestamp: float,
    epi: float = 0.0,
    delta_nfr: float | None = None,
    sense_index: float = 0.8,
    structural_timestamp: float | None = None,
    delta_nfr_proj_longitudinal: float = 0.0,
    delta_nfr_proj_lateral: float = 0.0,
    tyres: Mapping[str, Any] | TyresNode | None = None,
    suspension: Mapping[str, Any] | SuspensionNode | None = None,
    chassis: Mapping[str, Any] | ChassisNode | None = None,
    brakes: Mapping[str, Any] | BrakesNode | None = None,
    transmission: Mapping[str, Any] | TransmissionNode | None = None,
    track: Mapping[str, Any] | TrackNode | None = None,
    driver: Mapping[str, Any] | DriverNode | None = None,
    **bundle_overrides: Any,
) -> EPIBundle:
    """Construct an :class:`EPIBundle` with convenient defaults for tests."""

    node_overrides = {
        "tyres": tyres,
        "suspension": suspension,
        "chassis": chassis,
        "brakes": brakes,
        "transmission": transmission,
        "track": track,
        "driver": driver,
    }

    nodes = {
        name: _coerce_node(name, sense_index, overrides)
        for name, overrides in node_overrides.items()
    }

    bundle_kwargs: dict[str, Any] = {
        "timestamp": timestamp,
        "epi": epi,
        "sense_index": sense_index,
        "delta_nfr": delta_nfr
        if delta_nfr is not None
        else nodes["tyres"].delta_nfr + nodes["suspension"].delta_nfr,
        "structural_timestamp": structural_timestamp,
        "delta_nfr_proj_longitudinal": delta_nfr_proj_longitudinal,
        "delta_nfr_proj_lateral": delta_nfr_proj_lateral,
        **nodes,
    }

    bundle_kwargs.update(bundle_overrides)

    return EPIBundle(**bundle_kwargs)


def build_balanced_bundle(timestamp: float, delta_nfr: float, si: float) -> EPIBundle:
    """Construct a bundle that evenly distributes delta_nfr across nodes."""

    share = delta_nfr / 6
    return build_epi_bundle(
        timestamp=timestamp,
        delta_nfr=delta_nfr,
        sense_index=si,
        tyres={"delta_nfr": share},
        suspension={"delta_nfr": share},
        chassis={"delta_nfr": share},
        brakes={"delta_nfr": share},
        transmission={"delta_nfr": share},
        track={"delta_nfr": share},
        driver={"delta_nfr": share},
    )


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

__all__ = ["build_epi_bundle", "build_balanced_bundle", "build_epi_nodes"]
