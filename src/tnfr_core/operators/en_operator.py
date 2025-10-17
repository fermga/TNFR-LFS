"""ΔNFR emission and reception operators."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple, Type, TypeVar

from tnfr_core.equations.epi import EPIExtractor, TelemetryRecord
from tnfr_core.equations.epi_models import (
    EPIBundle,
    BrakesNode,
    ChassisNode,
    DriverNode,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)
from tnfr_core.runtime.shared import SupportsEPIBundle


NodeType = TypeVar("NodeType")


def _coerce_node(node: object, node_type: Type[NodeType]) -> NodeType:
    """Return ``node`` as ``node_type`` by copying dataclass fields if needed."""

    if isinstance(node, node_type):
        return node
    kwargs: Dict[str, object] = {}
    for entry in fields(node_type):
        kwargs[entry.name] = getattr(node, entry.name)
    return node_type(**kwargs)


def _normalise_delta_breakdown(
    payload: Mapping[str, Mapping[str, float]] | None,
) -> Dict[str, Dict[str, float]]:
    """Return a concrete mapping for delta breakdown payloads."""

    if not payload:
        return {}
    normalised: Dict[str, Dict[str, float]] = {}
    for system, entries in payload.items():
        if not isinstance(entries, Mapping):
            continue
        normalised[str(system)] = {
            str(component): float(value)
            for component, value in entries.items()
            if isinstance(value, (int, float))
        }
    return normalised


def _normalise_node_evolution(
    payload: Mapping[str, tuple[float, float]] | None,
) -> Dict[str, tuple[float, float]]:
    """Convert node evolution payloads into plain dictionaries."""

    if not payload:
        return {}
    evolution: Dict[str, tuple[float, float]] = {}
    for node, values in payload.items():
        if not isinstance(values, Sequence) or len(values) != 2:
            continue
        evolution[str(node)] = (float(values[0]), float(values[1]))
    return evolution


def _ensure_bundle(bundle: SupportsEPIBundle) -> EPIBundle:
    """Return a concrete :class:`EPIBundle` for ``bundle``."""

    if isinstance(bundle, EPIBundle):
        return bundle
    tyres = _coerce_node(bundle.tyres, TyresNode)
    suspension = _coerce_node(bundle.suspension, SuspensionNode)
    chassis = _coerce_node(bundle.chassis, ChassisNode)
    brakes = _coerce_node(bundle.brakes, BrakesNode)
    transmission = _coerce_node(bundle.transmission, TransmissionNode)
    track = _coerce_node(bundle.track, TrackNode)
    driver = _coerce_node(bundle.driver, DriverNode)
    return EPIBundle(
        timestamp=float(bundle.timestamp),
        epi=float(bundle.epi),
        delta_nfr=float(bundle.delta_nfr),
        sense_index=float(bundle.sense_index),
        tyres=tyres,
        suspension=suspension,
        chassis=chassis,
        brakes=brakes,
        transmission=transmission,
        track=track,
        driver=driver,
        structural_timestamp=(
            None
            if bundle.structural_timestamp is None
            else float(bundle.structural_timestamp)
        ),
        delta_breakdown=_normalise_delta_breakdown(bundle.delta_breakdown),
        dEPI_dt=float(bundle.dEPI_dt),
        integrated_epi=float(bundle.integrated_epi),
        node_evolution=_normalise_node_evolution(bundle.node_evolution),
        delta_nfr_proj_longitudinal=float(bundle.delta_nfr_proj_longitudinal),
        delta_nfr_proj_lateral=float(bundle.delta_nfr_proj_lateral),
        nu_f_classification=str(bundle.nu_f_classification),
        nu_f_category=str(bundle.nu_f_category),
        nu_f_label=str(bundle.nu_f_label),
        nu_f_dominant=float(bundle.nu_f_dominant),
        coherence_index=float(bundle.coherence_index),
        ackermann_parallel_index=float(bundle.ackermann_parallel_index),
    )


def _clone_bundle(
    bundle: SupportsEPIBundle, *, delta_nfr: float, sense_index: float
) -> EPIBundle:
    """Return a concrete copy of ``bundle`` with updated ΔNFR and Si values."""

    concrete = _ensure_bundle(bundle)
    data: Dict[str, object] = {}
    for entry in fields(EPIBundle):
        data[entry.name] = getattr(concrete, entry.name)
    data["delta_nfr"] = float(delta_nfr)
    data["sense_index"] = float(sense_index)
    return EPIBundle(**data)


def emission_operator(target_delta_nfr: float, target_sense_index: float) -> Dict[str, float]:
    """Return normalised objectives for ΔNFR and sense index targets."""

    target_si = max(0.0, min(1.0, target_sense_index))
    return {"delta_nfr": float(target_delta_nfr), "sense_index": target_si}


def reception_operator(
    records: Sequence[TelemetryRecord], extractor: EPIExtractor | None = None
) -> List[EPIBundle]:
    """Convert raw telemetry records into EPI bundles."""

    if not records:
        return []
    extractor = extractor or EPIExtractor()
    return extractor.extract(records)


def _update_bundles(
    bundles: Sequence[SupportsEPIBundle],
    delta_series: Sequence[float],
    si_series: Sequence[float],
) -> List[SupportsEPIBundle]:
    """Return copies of ``bundles`` with updated ΔNFR and sense index values."""

    updated: List[SupportsEPIBundle] = []
    for bundle, delta_value, si_value in zip(bundles, delta_series, si_series):
        updated_bundle = _clone_bundle(
            bundle,
            delta_nfr=delta_value,
            sense_index=max(0.0, min(1.0, si_value)),
        )
        updated.append(updated_bundle)
    return updated


__all__ = [
    "_clone_bundle",
    "_coerce_node",
    "_ensure_bundle",
    "_normalise_delta_breakdown",
    "_normalise_node_evolution",
    "_update_bundles",
    "emission_operator",
    "reception_operator",
]

