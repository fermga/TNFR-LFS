"""Helpers for constructing EPI-related test fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

from tests.helpers.constants import BASE_NU_F

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


def build_node_bundle(
    *,
    timestamp: float,
    delta_nfr_by_node: Mapping[str, float] | None = None,
    overrides: Mapping[
        str,
        Mapping[str, Any]
        | TyresNode
        | SuspensionNode
        | ChassisNode
        | BrakesNode
        | TransmissionNode
        | TrackNode
        | DriverNode,
    ]
    | None = None,
    epi: float = 0.0,
    delta_nfr: float | None = None,
    sense_index: float = 0.8,
    structural_timestamp: float | None = None,
    delta_nfr_proj_longitudinal: float = 0.0,
    delta_nfr_proj_lateral: float = 0.0,
    **bundle_overrides: Any,
) -> EPIBundle:
    """Build an :class:`EPIBundle` using explicit per-node ΔNFR values."""

    node_delta_map = dict(delta_nfr_by_node or {})
    node_overrides = dict(overrides or {})

    unknown_deltas = set(node_delta_map) - NODE_FACTORIES.keys()
    if unknown_deltas:
        raise KeyError(f"Unknown node names in delta map: {sorted(unknown_deltas)!r}")

    unknown_overrides = set(node_overrides) - NODE_FACTORIES.keys()
    if unknown_overrides:
        raise KeyError(f"Unknown node names in overrides: {sorted(unknown_overrides)!r}")

    nodes: dict[str, Any] = {}
    for name in NODE_FACTORIES:
        override_value = node_overrides.get(name)
        delta_value = node_delta_map.get(name)

        if override_value is None:
            if delta_value is not None:
                nodes[name] = {"delta_nfr": delta_value}
            continue

        if isinstance(override_value, Mapping):
            data = dict(override_value)
            if delta_value is not None:
                data["delta_nfr"] = delta_value
            nodes[name] = data
            continue

        if delta_value is not None:
            raise ValueError(
                "Cannot provide both a pre-constructed node and a delta override "
                f"for '{name}'"
            )

        nodes[name] = override_value

    bundle_kwargs: dict[str, Any] = {
        "timestamp": timestamp,
        "epi": epi,
        "delta_nfr": delta_nfr,
        "sense_index": sense_index,
        "structural_timestamp": structural_timestamp,
        "delta_nfr_proj_longitudinal": delta_nfr_proj_longitudinal,
        "delta_nfr_proj_lateral": delta_nfr_proj_lateral,
        **{name: value for name, value in nodes.items()},
    }

    bundle_kwargs.update(bundle_overrides)

    return build_epi_bundle(**bundle_kwargs)


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

def build_support_bundle(
    *,
    timestamp: float,
    structural_timestamp: float,
    tyre_delta: float,
    suspension_delta: float,
    longitudinal_delta: float,
    lateral_delta: float,
    yaw_rate: float,
    sense_index: float = 0.8,
) -> EPIBundle:
    """Bundle tailored for support metric calculations."""

    return build_epi_bundle(
        timestamp=timestamp,
        delta_nfr=tyre_delta + suspension_delta,
        sense_index=sense_index,
        structural_timestamp=structural_timestamp,
        delta_nfr_proj_longitudinal=longitudinal_delta,
        delta_nfr_proj_lateral=lateral_delta,
        tyres={"delta_nfr": tyre_delta},
        suspension={"delta_nfr": suspension_delta},
        chassis={"delta_nfr": 0.0, "yaw_rate": yaw_rate},
    )


def build_operator_bundle(
    *,
    timestamp: float,
    tyre_delta: float,
    delta_nfr: float | None = None,
    yaw_rate: float = 0.0,
    sense_index: float = 0.9,
) -> EPIBundle:
    """Bundle used in operator-specific tests with convenient defaults."""

    delta_value = tyre_delta if delta_nfr is None else delta_nfr
    return build_epi_bundle(
        timestamp=timestamp,
        delta_nfr=delta_value,
        sense_index=sense_index,
        tyres={"delta_nfr": tyre_delta},
        suspension={"delta_nfr": delta_value},
        chassis={"delta_nfr": delta_value, "yaw_rate": yaw_rate},
    )


def build_axis_bundle(
    *,
    delta_nfr: float,
    long_component: float,
    lat_component: float,
    sense_index: float = 0.8,
    gradient: float = 0.0,
) -> EPIBundle:
    """Bundle distributing delta evenly across nodes with axis projections."""

    share = delta_nfr / 7.0
    return build_epi_bundle(
        timestamp=0.0,
        delta_nfr=delta_nfr,
        sense_index=sense_index,
        delta_nfr_proj_longitudinal=long_component,
        delta_nfr_proj_lateral=lat_component,
        tyres={"delta_nfr": share},
        suspension={"delta_nfr": share},
        chassis={"delta_nfr": share},
        brakes={"delta_nfr": share},
        transmission={"delta_nfr": share},
        track={"delta_nfr": share, "gradient": gradient},
        driver={"delta_nfr": share},
    )


def build_udr_bundle_series(
    values: Sequence[float],
    *,
    start_timestamp: float = 0.0,
    step: float = 0.1,
    sense_index: float = 0.8,
) -> list[EPIBundle]:
    """Produce a sequence of bundles mirroring uniform delta ratio samples."""

    bundles: list[EPIBundle] = []
    timestamp = start_timestamp
    for value in values:
        bundles.append(
            build_epi_bundle(
                timestamp=timestamp,
                delta_nfr=value,
                sense_index=sense_index,
                delta_nfr_proj_longitudinal=value,
                tyres={"delta_nfr": value},
                suspension={"delta_nfr": value},
                chassis={"delta_nfr": value},
                brakes={"delta_nfr": value},
                transmission={"delta_nfr": value},
                track={"delta_nfr": value},
                driver={"delta_nfr": value},
            )
        )
        timestamp += step
    return bundles


def build_rich_bundle(
    *,
    timestamp: float,
    delta_nfr: float = 2.0,
    sense_index: float = 0.85,
    yaw_rate: float = 0.1,
    travel_front: float = 0.04,
    travel_rear: float = 0.04,
    temps: tuple[float, float, float, float] = (82.0, 81.5, 79.5, 79.0),
    mu_front: tuple[float, float] = (1.25, 1.05),
    mu_rear: tuple[float, float] = (1.15, 0.95),
) -> EPIBundle:
    """Bundle capturing a rich snapshot of telemetry context."""

    share = delta_nfr / 6
    return build_epi_bundle(
        timestamp=timestamp,
        delta_nfr=delta_nfr,
        sense_index=sense_index,
        tyres={
            "delta_nfr": share,
            "mu_eff_front": mu_front[0],
            "mu_eff_rear": mu_rear[0],
            "mu_eff_front_lateral": mu_front[0],
            "mu_eff_front_longitudinal": mu_front[1],
            "mu_eff_rear_lateral": mu_rear[0],
            "mu_eff_rear_longitudinal": mu_rear[1],
            "tyre_temp_fl": temps[0],
            "tyre_temp_fr": temps[1],
            "tyre_temp_rl": temps[2],
            "tyre_temp_rr": temps[3],
        },
        suspension={
            "delta_nfr": share,
            "travel_front": travel_front,
            "travel_rear": travel_rear,
        },
        chassis={"delta_nfr": share, "yaw_rate": yaw_rate},
        brakes={"delta_nfr": share},
        transmission={"delta_nfr": share, "throttle": 0.4, "gear": 4, "speed": 140.0},
        track={"delta_nfr": share, "yaw": 0.0},
        driver={"delta_nfr": share, "steer": 0.1, "throttle": 0.5, "style_index": sense_index},
    )


__all__ = [
    "build_epi_bundle",
    "build_node_bundle",
    "build_balanced_bundle",
    "build_epi_nodes",
    "build_support_bundle",
    "build_operator_bundle",
    "build_axis_bundle",
    "build_udr_bundle_series",
    "build_rich_bundle",
]
