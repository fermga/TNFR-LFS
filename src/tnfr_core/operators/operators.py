"""High-level TNFR × LFS operators for telemetry analytics pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import partial
import math
from importlib import import_module
from importlib.util import find_spec
import warnings
from statistics import mean
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
    TypedDict,
    Type,
    TypeVar,
    cast,
)

import numpy as np


def _is_module_available(module_name: str) -> bool:
    """Return ``True`` if ``module_name`` can be imported."""

    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


_HAS_JAX = _is_module_available("jax.numpy")
if _HAS_JAX:
    import jax.numpy as jnp  # type: ignore[import-not-found]
else:  # pragma: no cover - exercised when JAX is unavailable
    jnp = None

from tnfr_core.equations.constants import WHEEL_SUFFIXES

from tnfr_core.equations.contextual_delta import (
    apply_contextual_delta,
    load_context_matrix,
    resolve_context_from_bundle,
    resolve_series_context,
)
from tnfr_core.equations.dissonance import compute_useful_dissonance_stats
from tnfr_core.equations.epi import (
    EPIExtractor,
    NaturalFrequencyAnalyzer,
    TelemetryRecord,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
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
from tnfr_core.equations.phases import LEGACY_PHASE_MAP
from tnfr_core.equations.archetypes import ARCHETYPE_MEDIUM
from tnfr_core.operators.operator_detection import (
    normalize_structural_operator_identifier,
    silence_event_payloads,
)
from tnfr_core.operators.structural.coherence_il import coherence_operator_il
from tnfr_core.operators.interfaces import (
    SupportsChassisNode,
    SupportsEPIBundle,
    SupportsMicrosector,
    SupportsSuspensionNode,
    SupportsTelemetrySample,
    SupportsTyresNode,
)
from tnfr_core.operators.entry.recursivity import (
    RecursivityMicroState,
    RecursivityMicroStateSnapshot,
    RecursivityNetworkHistoryEntry,
    RecursivityNetworkMemory,
    RecursivityNetworkSession,
    RecursivityOperatorResult,
    RecursivitySessionState,
    RecursivityStateRoot,
    extract_network_memory,
    recursivity_operator,
)

_extract_network_memory = extract_network_memory
from tnfr_core._canonical import CANONICAL_REQUESTED, import_tnfr
from tnfr_core.operators.structural.epi_evolution import (
    NodalEvolution,
    evolve_epi,
)
from tnfr_core.operators.pipeline import (
    orchestrate_delta_metrics as pipeline_orchestrate_delta_metrics,
)
from tnfr_core.operators.pipeline.coherence import (
    _stage_coherence as pipeline_stage_coherence,
)
from tnfr_core.operators.pipeline.epi import (
    _stage_epi_evolution as pipeline_stage_epi,
)
from tnfr_core.operators.pipeline.events import (
    _aggregate_operator_events as pipeline_aggregate_operator_events,
)
from tnfr_core.operators.pipeline.nodal import (
    StructuralDeltaComponent,
    _stage_nodal_metrics as pipeline_stage_nodal,
)
from tnfr_core.operators.pipeline.reception import (
    _stage_reception as pipeline_stage_reception,
)
from tnfr_core.operators.pipeline.sense import (
    _stage_sense as pipeline_stage_sense,
)
from tnfr_core.operators.pipeline.variability import (
    _microsector_variability as pipeline_microsector_variability,
    _phase_context_from_microsectors,
    compute_window_metrics as pipeline_compute_window_metrics,
)


_APEX_PHASE_CANDIDATES: Tuple[str, ...] = LEGACY_PHASE_MAP.get("apex", tuple())
if "apex" not in _APEX_PHASE_CANDIDATES:
    _APEX_PHASE_CANDIDATES = _APEX_PHASE_CANDIDATES + ("apex",)


@dataclass(frozen=True)
class DissonanceBreakdown:
    """Breakdown of dissonance events by usefulness."""

    value: float
    useful_magnitude: float
    parasitic_magnitude: float
    useful_ratio: float
    parasitic_ratio: float
    useful_percentage: float
    parasitic_percentage: float
    total_events: int
    useful_events: int
    parasitic_events: int
    useful_dissonance_ratio: float
    useful_dissonance_percentage: float
    high_yaw_acc_samples: int
    useful_dissonance_samples: int


def _zero_dissonance_breakdown() -> DissonanceBreakdown:
    """Return a :class:`DissonanceBreakdown` filled with zeros."""

    return DissonanceBreakdown(
        value=0.0,
        useful_magnitude=0.0,
        parasitic_magnitude=0.0,
        useful_ratio=0.0,
        parasitic_ratio=0.0,
        useful_percentage=0.0,
        parasitic_percentage=0.0,
        total_events=0,
        useful_events=0,
        parasitic_events=0,
        useful_dissonance_ratio=0.0,
        useful_dissonance_percentage=0.0,
        high_yaw_acc_samples=0,
        useful_dissonance_samples=0,
    )


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


def coherence_operator(series: Sequence[float], window: int = 3) -> List[float]:
    """Smooth a numeric series while preserving its average value."""

    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if not series:
        return []
    xp = jnp if jnp is not None else np
    array = xp.asarray(series, dtype=xp.float64)
    kernel = xp.ones(window, dtype=array.dtype)
    numerator = xp.convolve(array, kernel, mode="same")
    counts = xp.convolve(xp.ones_like(array), kernel, mode="same")
    smoothed = numerator / counts
    bias = float(array.mean() - smoothed.mean())
    if abs(bias) >= 1e-12:
        smoothed = smoothed + bias
    return np.asarray(smoothed, dtype=float).tolist()


def dissonance_operator(series: Sequence[float], target: float) -> float:
    """Compute the mean absolute deviation relative to a target value."""

    if not series:
        return 0.0
    xp = jnp if jnp is not None else np
    array = xp.asarray(series, dtype=xp.float64)
    differences = xp.abs(array - target)
    return float(xp.mean(differences))


def dissonance_breakdown_operator(
    series: Sequence[float],
    target: float,
    *,
    microsectors: Sequence[SupportsMicrosector] | None = None,
    bundles: Sequence[SupportsEPIBundle] | None = None,
) -> DissonanceBreakdown:
    """Classify support events into useful (positive) and parasitic dissonance."""

    base_value = dissonance_operator(series, target)
    useful_events = 0
    parasitic_events = 0
    useful_magnitude = 0.0
    parasitic_magnitude = 0.0
    useful_dissonance_samples = 0
    high_yaw_acc_samples = 0

    context_matrix = load_context_matrix()
    bundle_context = (
        resolve_series_context(bundles, matrix=context_matrix) if bundles else []
    )

    if microsectors and bundles:
        bundle_count = len(bundles)
        tyre_nodes: Sequence[SupportsTyresNode] = [bundle.tyres for bundle in bundles]
        for microsector in microsectors:
            if not microsector.support_event:
                continue
            apex_goal = None
            for apex_phase in _APEX_PHASE_CANDIDATES:
                apex_goal = next(
                    (goal for goal in microsector.goals if goal.phase == apex_phase),
                    None,
                )
                if apex_goal is not None:
                    break
            if apex_goal is None:
                continue
            apex_indices: List[int] = []
            for apex_phase in _APEX_PHASE_CANDIDATES:
                indices = microsector.phase_samples.get(apex_phase) or ()
                apex_indices.extend(idx for idx in indices if 0 <= idx < bundle_count)
            if not apex_indices:
                continue
            tyre_delta = []
            for idx in apex_indices:
                multiplier = 1.0
                if 0 <= idx < len(bundle_context):
                    multiplier = max(
                        context_matrix.min_multiplier,
                        min(
                            context_matrix.max_multiplier,
                            bundle_context[idx].multiplier,
                        ),
                    )
                tyre_delta.append(tyre_nodes[idx].delta_nfr * multiplier)
            if not tyre_delta:
                continue
            deviation = mean(tyre_delta) - apex_goal.target_delta_nfr
            contribution = abs(deviation)
            if contribution <= 1e-12:
                continue
            if deviation >= 0.0:
                useful_events += 1
                useful_magnitude += contribution
            else:
                parasitic_events += 1
                parasitic_magnitude += contribution

    useful_dissonance_ratio = 0.0
    if bundles:
        timestamps = [bundle.timestamp for bundle in bundles]
        delta_series = [
            apply_contextual_delta(
                bundle.delta_nfr,
                bundle_context[idx],
                context_matrix=context_matrix,
            )
            for idx, bundle in enumerate(bundles)
        ]
        chassis_nodes: Sequence[SupportsChassisNode] = [
            bundle.chassis for bundle in bundles
        ]
        yaw_rates = [node.yaw_rate for node in chassis_nodes]
        (
            useful_dissonance_samples,
            high_yaw_acc_samples,
            useful_dissonance_ratio,
        ) = compute_useful_dissonance_stats(timestamps, delta_series, yaw_rates)

    total_events = useful_events + parasitic_events
    total_magnitude = useful_magnitude + parasitic_magnitude
    if total_magnitude > 1e-12:
        useful_ratio = useful_magnitude / total_magnitude
        parasitic_ratio = parasitic_magnitude / total_magnitude
    elif total_events > 0:
        useful_ratio = useful_events / total_events
        parasitic_ratio = parasitic_events / total_events
    else:
        useful_ratio = 0.0
        parasitic_ratio = 0.0

    return DissonanceBreakdown(
        value=base_value,
        useful_magnitude=useful_magnitude,
        parasitic_magnitude=parasitic_magnitude,
        useful_ratio=useful_ratio,
        parasitic_ratio=parasitic_ratio,
        useful_percentage=useful_ratio * 100.0,
        parasitic_percentage=parasitic_ratio * 100.0,
        total_events=total_events,
        useful_events=useful_events,
        parasitic_events=parasitic_events,
        useful_dissonance_ratio=useful_dissonance_ratio,
        useful_dissonance_percentage=useful_dissonance_ratio * 100.0,
        high_yaw_acc_samples=high_yaw_acc_samples,
        useful_dissonance_samples=useful_dissonance_samples,
    )


def _prepare_series_pair(
    series_a: Sequence[float] | np.ndarray,
    series_b: Sequence[float] | np.ndarray,
    *,
    strict_length: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return truncated NumPy arrays for ``series_a`` and ``series_b``."""

    array_a = np.asarray(series_a, dtype=float).ravel()
    array_b = np.asarray(series_b, dtype=float).ravel()
    if strict_length and array_a.shape[0] != array_b.shape[0]:
        raise ValueError("series must have the same length")

    length = min(array_a.shape[0], array_b.shape[0])
    return array_a[:length], array_b[:length]


def _batch_coupling(
    values: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Compute coupling values for stacked series using vectorised operations."""

    xp = jnp if jnp is not None else np
    data = xp.asarray(values, dtype=xp.float64)
    if mask is None:
        mask_xp = xp.ones(data.shape[:-1], dtype=bool)
    else:
        mask_xp = xp.asarray(mask, dtype=bool)

    counts = mask_xp.sum(axis=1, keepdims=True)
    counts_float = counts.astype(data.dtype)
    safe_counts = xp.where(counts > 0, counts_float, xp.ones_like(counts_float))
    sums = xp.sum(xp.where(mask_xp[..., None], data, 0.0), axis=1, keepdims=True)
    means = xp.where(
        (counts > 0)[..., None],
        sums / safe_counts[..., None],
        0.0,
    )
    centered = xp.where(mask_xp[..., None], data - means, 0.0)
    covariance = xp.sum(centered[..., 0] * centered[..., 1], axis=1)
    variance_a = xp.sum(centered[..., 0] ** 2, axis=1)
    variance_b = xp.sum(centered[..., 1] ** 2, axis=1)
    denominator = xp.sqrt(variance_a * variance_b)

    if xp is np:
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(denominator > 0, covariance / denominator, 0.0)
    else:  # pragma: no cover - exercised only when JAX is available
        result = xp.where(denominator > 0, covariance / denominator, 0.0)
    return np.asarray(result, dtype=float)


def coupling_operator(
    series_a: Sequence[float], series_b: Sequence[float], *, strict_length: bool = True
) -> float:
    """Return the normalised coupling (correlation) between two series."""

    values_a, values_b = _prepare_series_pair(
        series_a, series_b, strict_length=strict_length
    )
    if values_a.size == 0:
        return 0.0

    stacked = np.stack((values_a, values_b), axis=-1)[np.newaxis, ...]
    return float(_batch_coupling(stacked)[0])


def acoplamiento_operator(
    series_a: Sequence[float], series_b: Sequence[float], *, strict_length: bool = True
) -> float:
    """Compatibility wrapper for :func:`coupling_operator`."""

    warnings.warn(
        "acoplamiento_operator has been renamed to coupling_operator; "
        "please update imports before the legacy name is removed.",
        DeprecationWarning,
        stacklevel=2,
    )
    return coupling_operator(series_a, series_b, strict_length=strict_length)


def pairwise_coupling_operator(
    series_by_node: Mapping[str, Sequence[float]],
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> Dict[str, float]:
    """Compute coupling metrics for each node pair using :func:`coupling_operator`."""

    if pairs is None:
        ordered_nodes = list(series_by_node.keys())
        pairs = [(a, b) for idx, a in enumerate(ordered_nodes) for b in ordered_nodes[idx + 1 :]]

    pairs = list(pairs)
    if not pairs:
        return {}

    prepared = [
        _prepare_series_pair(
            series_by_node.get(first, ()),
            series_by_node.get(second, ()),
            strict_length=False,
        )
        for first, second in pairs
    ]

    lengths = [values_a.shape[0] for values_a, _ in prepared]
    max_length = max(lengths, default=0)
    stacked = np.zeros((len(prepared), max_length, 2), dtype=float)
    mask = np.zeros((len(prepared), max_length), dtype=bool)

    for idx, ((values_a, values_b), length) in enumerate(zip(prepared, lengths)):
        if length == 0:
            continue
        stacked[idx, :length, 0] = values_a
        stacked[idx, :length, 1] = values_b
        mask[idx, :length] = True

    results = _batch_coupling(stacked, mask)
    return {
        f"{first}↔{second}": float(results[idx])
        for idx, (first, second) in enumerate(pairs)
    }


def resonance_operator(series: Sequence[float]) -> float:
    """Compute the root-mean-square (RMS) resonance of a series."""

    numpy_series = np.asarray(series, dtype=float).ravel()
    if numpy_series.size == 0:
        return 0.0
    xp = jnp if jnp is not None else np
    array = xp.asarray(numpy_series, dtype=xp.float64)
    rms = xp.sqrt(xp.mean(array ** 2))
    return float(rms)




@dataclass(frozen=True)
class TyreBalanceControlOutput:
    """Aggregated ΔP/Δcamber recommendations for a stint."""

    pressure_delta_front: float
    pressure_delta_rear: float
    camber_delta_front: float
    camber_delta_rear: float
    per_wheel_pressure: Mapping[str, float] = field(default_factory=dict)


def tyre_balance_controller(
    filtered_metrics: Mapping[str, float],
    *,
    delta_nfr_flat: float | None = None,
    target_front: float = 0.82,
    target_rear: float = 0.80,
    pressure_gain: float = 0.25,
    nfr_gain: float = 0.2,
    pressure_max_step: float = 0.16,
    camber_gain: float = 0.18,
    camber_max_step: float = 0.25,
    bias_gain: float = 0.04,
    offsets: Mapping[str, float] | None = None,
) -> TyreBalanceControlOutput:
    """Compute ΔP and camber tweaks from CPHI-derived tyre metrics."""

    def _safe_value(key: str) -> float | None:
        value = filtered_metrics.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    xp = jnp if jnp is not None else np

    def _value_or_nan(key: str) -> float:
        value = _safe_value(key)
        return value if value is not None else float("nan")

    def _to_bool(value: object) -> bool:
        if hasattr(value, "item"):
            return bool(value.item())  # type: ignore[attr-defined]
        return bool(value)

    def _to_float(value: object) -> float:
        if isinstance(value, (float, int)):
            return float(value)
        if hasattr(value, "item"):
            return float(value.item())  # type: ignore[attr-defined]
        return float(value)  # pragma: no cover - defensive fallback

    cphi_values = xp.asarray(
        [_value_or_nan(f"cphi_{suffix}") for suffix in WHEEL_SUFFIXES], dtype=xp.float64
    )

    all_missing = _to_bool(xp.all(xp.isnan(cphi_values)))
    if all_missing:
        zero_map = {suffix: 0.0 for suffix in WHEEL_SUFFIXES}
        return TyreBalanceControlOutput(
            pressure_delta_front=0.0,
            pressure_delta_rear=0.0,
            camber_delta_front=0.0,
            camber_delta_rear=0.0,
            per_wheel_pressure=zero_map,
        )

    delta_flat = (
        float(delta_nfr_flat)
        if delta_nfr_flat is not None
        else float(filtered_metrics.get("d_nfr_flat", 0.0))
    )

    def _min_finite(values: Sequence[float] | object) -> float | None:
        array = xp.asarray(values, dtype=xp.float64)
        mask = xp.isfinite(array)
        if not _to_bool(xp.any(mask)):
            return None
        minimum = xp.nanmin(array)
        return _to_float(minimum)

    front_health = _min_finite(cphi_values[:2])
    rear_health = _min_finite(cphi_values[2:])

    pressure_front = nfr_gain * delta_flat
    pressure_rear = nfr_gain * delta_flat
    if front_health is not None:
        pressure_front += pressure_gain * (target_front - front_health)
    if rear_health is not None:
        pressure_rear += pressure_gain * (target_rear - rear_health)

    if offsets:
        pressure_front += float(offsets.get("pressure_front", 0.0))
        pressure_rear += float(offsets.get("pressure_rear", 0.0))

    pressure_front = _to_float(
        xp.clip(xp.asarray(pressure_front, dtype=xp.float64), -pressure_max_step, pressure_max_step)
    )
    pressure_rear = _to_float(
        xp.clip(xp.asarray(pressure_rear, dtype=xp.float64), -pressure_max_step, pressure_max_step)
    )

    def _component_average(suffixes: Sequence[str], key: str) -> float:
        values = xp.asarray(
            [_value_or_nan(f"cphi_{suffix}_{key}") for suffix in suffixes], dtype=xp.float64
        )
        mask = xp.isfinite(values)
        counts = xp.sum(mask)
        sums = xp.sum(xp.where(mask, values, 0.0))
        counts_float = counts.astype(values.dtype)
        safe_counts = xp.where(counts > 0, counts_float, xp.ones_like(counts_float))
        mean_value = xp.where(counts > 0, sums / safe_counts, 0.0)
        return _to_float(mean_value)

    front_gradient_component = _component_average(["fl", "fr"], "gradient")
    rear_gradient_component = _component_average(["rl", "rr"], "gradient")

    camber_front = _to_float(
        xp.clip(
            xp.asarray(-front_gradient_component * camber_gain, dtype=xp.float64),
            -camber_max_step,
            camber_max_step,
        )
    )
    camber_rear = _to_float(
        xp.clip(
            xp.asarray(-rear_gradient_component * camber_gain, dtype=xp.float64),
            -camber_max_step,
            camber_max_step,
        )
    )

    if offsets:
        camber_front += float(offsets.get("camber_front", 0.0))
        camber_rear += float(offsets.get("camber_rear", 0.0))

    bias_values = xp.asarray(
        [_value_or_nan(f"cphi_{suffix}_temp_delta") for suffix in WHEEL_SUFFIXES], dtype=xp.float64
    )
    bias_safe = xp.where(xp.isfinite(bias_values), bias_values, 0.0)

    base_pressures = xp.asarray(
        [pressure_front, pressure_front, pressure_rear, pressure_rear], dtype=xp.float64
    )
    per_wheel_array = xp.clip(
        base_pressures + bias_gain * bias_safe,
        -pressure_max_step,
        pressure_max_step,
    )
    per_wheel_numpy = np.asarray(per_wheel_array, dtype=float)
    per_wheel = {suffix: float(value) for suffix, value in zip(WHEEL_SUFFIXES, per_wheel_numpy)}

    return TyreBalanceControlOutput(
        pressure_delta_front=pressure_front,
        pressure_delta_rear=pressure_rear,
        camber_delta_front=camber_front,
        camber_delta_rear=camber_rear,
        per_wheel_pressure=per_wheel,
    )


def mutation_operator(
    state: MutableMapping[str, Dict[str, object]],
    triggers: Mapping[str, object],
    *,
    entropy_threshold: float = 0.65,
    entropy_increase: float = 0.08,
    style_threshold: float = 0.12,
) -> Dict[str, object]:
    """Update the target archetype when entropy or style shifts are detected.

    The operator keeps per-microsector memory of the previous entropy, style
    index and active phase to detect meaningful regime changes.  When the
    entropy rises sharply or the driving style drifts outside the configured
    window the archetype mutates to the provided candidate or fallback.

    Parameters
    ----------
    state:
        Mutable mapping storing the mutation state per microsector.
    triggers:
        Mapping providing the measurements required to evaluate the mutation
        rules.  Expected keys include ``"microsector_id"``,
        ``"current_archetype"``, ``"candidate_archetype"``,
        ``"fallback_archetype"``, ``"entropy"``, ``"style_index"``,
        ``"style_reference"`` and ``"phase"``.
    entropy_threshold:
        Absolute entropy level that must be reached to trigger a fallback
        archetype.
    entropy_increase:
        Minimum entropy delta compared to the stored baseline to trigger the
        fallback archetype.
    style_threshold:
        Allowed absolute deviation between the filtered style index and the
        reference target before mutating towards the candidate archetype.

    Returns
    -------
    dict
        A dictionary with the selected archetype, whether a mutation happened
        and diagnostic information.
    """

    microsector_id = triggers.get("microsector_id")
    if not isinstance(microsector_id, str) or not microsector_id:
        raise ValueError("microsector_id trigger must be a non-empty string")

    current_archetype = str(triggers.get("current_archetype", ARCHETYPE_MEDIUM))
    candidate_archetype = str(triggers.get("candidate_archetype", current_archetype))
    fallback_archetype = str(triggers.get("fallback_archetype", ARCHETYPE_MEDIUM))

    entropy = float(triggers.get("entropy", 0.0))
    style_index = float(triggers.get("style_index", 1.0))
    style_reference = float(triggers.get("style_reference", style_index))
    phase = triggers.get("phase") if isinstance(triggers.get("phase"), str) else None
    dynamic_flag = bool(triggers.get("dynamic_conditions", False))

    micro_state = state.setdefault(
        microsector_id,
        {
            "archetype": current_archetype,
            "entropy": entropy,
            "style_index": style_index,
            "phase": phase,
        },
    )

    previous_entropy = float(micro_state.get("entropy", entropy))
    previous_style = float(micro_state.get("style_index", style_index))
    previous_phase = micro_state.get("phase") if isinstance(micro_state.get("phase"), str) else None

    entropy_delta = entropy - previous_entropy
    style_delta = abs(style_index - style_reference)

    mutated = False
    selected_archetype = current_archetype

    if entropy >= entropy_threshold and entropy_delta >= entropy_increase:
        selected_archetype = fallback_archetype
        mutated = selected_archetype != current_archetype
    elif style_delta >= style_threshold or (dynamic_flag and style_delta >= style_threshold * 0.5):
        selected_archetype = candidate_archetype
        mutated = selected_archetype != current_archetype
    elif phase is not None and previous_phase is not None and phase != previous_phase:
        # When the phase changes we allow the system to adopt the candidate
        # archetype if the style trend keeps diverging from the stored value.
        secondary_delta = abs(style_index - previous_style)
        if secondary_delta >= style_threshold * 0.5:
            selected_archetype = candidate_archetype
            mutated = selected_archetype != current_archetype

    micro_state.update(
        {
            "archetype": selected_archetype,
            "entropy": entropy,
            "style_index": style_index,
            "phase": phase if phase is not None else previous_phase,
        }
    )

    return {
        "microsector_id": microsector_id,
        "archetype": selected_archetype,
        "mutated": mutated,
        "entropy": entropy,
        "entropy_delta": entropy_delta,
        "style_delta": style_delta,
        "phase": micro_state["phase"],
    }


def recursive_filter_operator(
    series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5
) -> List[float]:
    """Apply a recursive filter to a series to capture hysteresis effects."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in the [0, 1) interval")
    if not series:
        return []
    xp = jnp if jnp is not None else np
    array = xp.asarray(series, dtype=float)
    length = array.shape[0]
    dtype = array.dtype
    kernel = xp.power(decay, xp.arange(length, dtype=dtype)) * (1.0 - decay)
    filtered = xp.convolve(array, kernel, mode="full")[:length]
    seed_powers = xp.power(decay, xp.arange(1, length + 1, dtype=dtype)) * seed
    trace = filtered + seed_powers
    return np.asarray(trace, dtype=float).tolist()


def recursividad_operator(
    series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5
) -> List[float]:
    """Compatibility wrapper for :func:`recursive_filter_operator`."""

    warnings.warn(
        "recursividad_operator has been renamed to recursive_filter_operator; "
        "please update imports before the legacy name is removed.",
        DeprecationWarning,
        stacklevel=2,
    )
    return recursive_filter_operator(series, seed=seed, decay=decay)


def _update_bundles(
    bundles: Sequence[SupportsEPIBundle],
    delta_series: Sequence[float],
    si_series: Sequence[float],
) -> List[SupportsEPIBundle]:
    updated: List[SupportsEPIBundle] = []
    for bundle, delta_value, si_value in zip(bundles, delta_series, si_series):
        updated_bundle = _clone_bundle(
            bundle,
            delta_nfr=delta_value,
            sense_index=max(0.0, min(1.0, si_value)),
        )
        updated.append(updated_bundle)
    return updated


def _microsector_sample_indices(microsector: SupportsMicrosector) -> List[int]:
    indices: set[int] = set()
    for samples in getattr(microsector, "phase_samples", {}).values():
        if samples is None:
            continue
        for idx in samples:
            indices.add(int(idx))
    if not indices:
        spans = [tuple(bounds) for bounds in getattr(microsector, "phase_boundaries", {}).values()]
        if spans:
            start = min(span[0] for span in spans)
            end = max(span[1] for span in spans)
            indices.update(range(start, end))
    return sorted(indices)




_STABILITY_COV_THRESHOLD = 0.15


def _variance_payload(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "stdev": 0.0,
            "coefficient_of_variation": 0.0,
            "stability_score": 1.0,
        }
    xp = jnp if _HAS_JAX else np
    array = xp.asarray(values, dtype=float)
    average = float(xp.mean(array))
    variance = float(xp.var(array, ddof=0))
    if variance < 0.0 and abs(variance) < 1e-12:
        variance = 0.0
    stdev = float(xp.sqrt(variance)) if variance > 0.0 else 0.0
    baseline = max(abs(average), 1e-9)
    coefficient = stdev / baseline
    stability = 1.0 - min(1.0, coefficient / _STABILITY_COV_THRESHOLD)
    if stability < 0.0:
        stability = 0.0
    return {
        "mean": average,
        "variance": variance,
        "stdev": stdev,
        "coefficient_of_variation": coefficient,
        "stability_score": stability,
    }


def _delta_integral_series(
    bundles: Sequence[SupportsEPIBundle],
    sample_indices: Sequence[int],
    *,
    delta_series: Any | None = None,
    timestamp_series: Any | None = None,
) -> List[float]:
    if not bundles or not sample_indices:
        return []

    xp = jnp if _HAS_JAX else np
    if delta_series is None or timestamp_series is None:
        timestamps = xp.asarray(
            [float(bundles[idx].timestamp) for idx in sample_indices], dtype=float
        )
        delta_nfr = xp.asarray(
            [float(bundles[idx].delta_nfr) for idx in sample_indices], dtype=float
        )
    else:
        indices = xp.asarray(sample_indices, dtype=int)
        timestamps = xp.take(timestamp_series, indices)
        delta_nfr = xp.take(delta_series, indices)

    forward_dt = xp.diff(timestamps)
    forward_dt = xp.concatenate(
        (forward_dt, xp.zeros((1,), dtype=timestamps.dtype)),
        axis=0,
    )
    backward_dt = xp.diff(timestamps)
    backward_dt = xp.concatenate(
        (xp.zeros((1,), dtype=timestamps.dtype), backward_dt),
        axis=0,
    )

    forward_dt = xp.maximum(forward_dt, 0.0)
    backward_dt = xp.maximum(backward_dt, 0.0)

    positions = xp.arange(len(sample_indices))
    use_forward = positions < len(sample_indices) - 1
    dt = xp.where(use_forward, forward_dt, backward_dt)
    dt = xp.where(dt <= 0.0, 1.0, dt)

    integrals = xp.abs(delta_nfr) * dt
    return np.asarray(integrals, dtype=float).tolist()






def orchestrate_delta_metrics(
    telemetry_segments: Sequence[Sequence[TelemetryRecord]],
    target_delta_nfr: float,
    target_sense_index: float,
    *,
    coherence_window: int = 3,
    recursion_decay: float = 0.4,
    microsectors: Sequence[SupportsMicrosector] | None = None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None = None,
    operator_state: Mapping[str, Dict[str, object]] | None = None,
) -> Mapping[str, object]:
    """Pipeline orchestration producing aggregated ΔNFR and Si metrics."""

    from tnfr_core.operators.pipeline.delta_workflow import (
        build_delta_metrics_dependencies,
    )

    xp_module = jnp if _HAS_JAX else np
    structural_component = StructuralDeltaComponent()

    dependencies = build_delta_metrics_dependencies(
        coherence_window=coherence_window,
        recursion_decay=recursion_decay,
        microsectors=microsectors,
        xp_module=xp_module,
        has_jax=_HAS_JAX,
        structural_component=structural_component,
        emission_operator=emission_operator,
        reception_operator=reception_operator,
        dissonance_breakdown_operator=dissonance_breakdown_operator,
        coherence_operator=coherence_operator,
        coupling_operator=coupling_operator,
        resonance_operator=resonance_operator,
        pairwise_coupling_operator=pairwise_coupling_operator,
        recursive_filter_operator=recursive_filter_operator,
        stage_reception=pipeline_stage_reception,
        stage_coherence=pipeline_stage_coherence,
        stage_nodal=pipeline_stage_nodal,
        stage_epi=pipeline_stage_epi,
        stage_sense=pipeline_stage_sense,
        stage_variability=pipeline_microsector_variability,
        aggregate_events=pipeline_aggregate_operator_events,
        compute_window_metrics=pipeline_compute_window_metrics,
        phase_context_resolver=_phase_context_from_microsectors,
        network_memory_extractor=_extract_network_memory,
        zero_breakdown_factory=_zero_dissonance_breakdown,
        load_context_matrix=load_context_matrix,
        resolve_context_from_bundle=resolve_context_from_bundle,
        apply_contextual_delta=apply_contextual_delta,
        update_bundles=_update_bundles,
        ensure_bundle=_ensure_bundle,
        normalise_node_evolution=_normalise_node_evolution,
        delta_integral=_delta_integral_series,
        variance_payload=_variance_payload,
    )

    return pipeline_orchestrate_delta_metrics(
        telemetry_segments,
        target_delta_nfr,
        target_sense_index,
        coherence_window=coherence_window,
        recursion_decay=recursion_decay,
        microsectors=microsectors,
        phase_weights=phase_weights,
        operator_state=operator_state,
        dependencies=dependencies,
    )


__all__ = [
    "emission_operator",
    "reception_operator",
    "coherence_operator",
    "coherence_operator_il",
    "dissonance_operator",
    "dissonance_breakdown_operator",
    "DissonanceBreakdown",
    "coupling_operator",
    "acoplamiento_operator",
    "pairwise_coupling_operator",
    "resonance_operator",
    "recursivity_operator",
    "mutation_operator",
    "recursive_filter_operator",
    "recursividad_operator",
    "orchestrate_delta_metrics",
    "evolve_epi",
]


if CANONICAL_REQUESTED:  # pragma: no cover - depends on optional package
    tnfr = import_tnfr()
    canonical_ops = import_module(f"{tnfr.__name__}.operators.operators")

    canonical_exports = getattr(canonical_ops, "__all__", None)
    if canonical_exports is not None:
        __all__ = list(dict.fromkeys([*canonical_exports, *__all__]))

    for name in dir(canonical_ops):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(canonical_ops, name)

