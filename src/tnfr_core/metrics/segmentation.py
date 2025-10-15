"""Microsector segmentation utilities.

This module analyses the stream of telemetry samples together with the
corresponding :class:`~tnfr_core.operators.interfaces.SupportsEPIBundle` instances to
derive microsectors and their tactical goals.  The segmentation is
performed using three simple heuristics inspired by motorsport
engineering practice:

* **Curvature detection** – sustained lateral acceleration indicates a
  cornering event which is the base unit for microsectors.
* **Support events** – sharp vertical load increases signal the moment
  where the chassis "leans" on the tyres and generates grip.
* **ΔNFR signatures** – the average ΔNFR within the microsector is used
  to classify the underlying archetype which in turn drives the goals
  for the entry, apex and exit phases.
"""

from __future__ import annotations

import math
from collections import defaultdict
from heapq import nlargest
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Tuple,
    cast,
)

from tnfr_core.equations.epi import (
    DEFAULT_PHASE_WEIGHTS,
    DeltaCalculator,
    NaturalFrequencyAnalyzer,
    NaturalFrequencySnapshot,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
from tnfr_core.equations.contextual_delta import (
    ContextFactors,
    ContextMatrix,
    load_context_matrix,
    resolve_context_from_record,
    resolve_microsector_context,
    resolve_series_context,
)
from tnfr_core.operators.interfaces import (
    SupportsContextBundle,
    SupportsContextRecord,
    SupportsEPIBundle,
    SupportsTelemetrySample,
)
from tnfr_core.metrics.metrics import (
    compute_window_metrics,
    phase_synchrony_index,
)
from tnfr_core.operators.operator_detection import (
    canonical_operator_label,
    detect_al,
    detect_en,
    detect_il,
    detect_nul,
    detect_oz,
    detect_ra,
    detect_remesh,
    detect_silence,
    detect_thol,
    detect_um,
    detect_val,
    detect_zhir,
    silence_event_payloads,
)
from tnfr_core.operators.operators import (
    RecursivityMicroStateSnapshot,
    RecursivityOperatorResult,
    RecursivityStateRoot,
    mutation_operator,
    recursivity_operator,
)
from tnfr_core.equations.phases import (
    LEGACY_PHASE_MAP,
    PHASE_SEQUENCE,
    expand_phase_alias,
    phase_family,
    replicate_phase_aliases,
)
from tnfr_core.equations.archetypes import (
    ARCHETYPE_CHICANE,
    ARCHETYPE_HAIRPIN,
    ARCHETYPE_FAST,
    ARCHETYPE_MEDIUM,
    archetype_phase_targets,
)
from tnfr_core.metrics.resonance import estimate_excitation_frequency
from tnfr_core.metrics.spectrum import estimate_sample_rate, phase_alignment

# Public API: core.__init__ re-exports segmentation artefacts via these names.
__all__ = [
    "Goal",
    "Microsector",
    "detect_quiet_microsector_streaks",
    "microsector_stability_metrics",
    "segment_microsectors",
]

# Thresholds derived from typical race car dynamics.  They can be tuned in
# the future without affecting the public API of the segmentation module.
CURVATURE_THRESHOLD = 1.2  # g units of lateral acceleration
MIN_SEGMENT_LENGTH = 3
BRAKE_THRESHOLD = -0.35  # g units of longitudinal deceleration
SUPPORT_THRESHOLD = 350.0  # Newtons of vertical load delta


PhaseLiteral = str


def _extract_signal(records: Sequence[SupportsTelemetrySample], attr: str) -> List[float]:
    values: List[float] = []
    for record in records:
        try:
            value = float(getattr(record, attr))
        except (AttributeError, TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        values.append(value)
    return values


def _normalise_signal(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mean_value = fmean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    if variance <= 1e-12:
        return [0.0 for _ in values]
    scale = math.sqrt(variance)
    return [(value - mean_value) / scale for value in values]


def _resolve_session_components(
    records: Sequence[SupportsTelemetrySample],
    baseline: SupportsTelemetrySample,
) -> tuple[str, str, str]:
    """Return the (car, track, compound) tuple used to scope recursivity state."""

    candidate = records[0] if records else baseline
    car = getattr(candidate, "car_model", None) or getattr(baseline, "car_model", None)
    track = (
        getattr(candidate, "track_name", None)
        or getattr(candidate, "track", None)
        or getattr(baseline, "track_name", None)
    )
    compound = (
        getattr(candidate, "tyre_compound", None)
        or getattr(baseline, "tyre_compound", None)
    )

    def _normalise(value: object, fallback: str) -> str:
        if value is None:
            return fallback
        token = str(value).strip()
        return token or fallback

    return (
        _normalise(car, "generic"),
        _normalise(track, "unknown"),
        _normalise(compound, "default"),
    )


def _resolve_context_multiplier(
    factors: ContextFactors | Mapping[str, float] | None,
    *,
    context_matrix: ContextMatrix,
) -> float:
    """Clamp the contextual multiplier for ``factors`` to the matrix bounds."""

    if isinstance(factors, ContextFactors):
        multiplier = factors.multiplier
    elif isinstance(factors, Mapping):
        curve = float(factors.get("curve", 1.0))
        surface = float(factors.get("surface", 1.0))
        traffic = float(factors.get("traffic", 1.0))
        multiplier = curve * surface * traffic
    else:
        multiplier = 1.0
    return max(
        context_matrix.min_multiplier,
        min(context_matrix.max_multiplier, multiplier),
    )


@dataclass(frozen=True)
class _SegmentSummary:
    curvature: float
    brake_event: bool
    support_event: bool
    avg_vertical_load: float
    duration: float
    speed_drop: float
    direction_changes: int
    brake_temperatures: Dict[str, float]
    brake_temperature_std: Dict[str, float]
    end_timestamp: float


@dataclass(frozen=True)
class _AnalyzerState:
    """Minimal snapshot required to resume the ν_f analyzer."""

    history_start: int | None
    history_end: int | None
    smoothed: Tuple[Tuple[str, float], ...]
    last_car: str | None
    last_snapshot: NaturalFrequencySnapshot | None


@dataclass(frozen=True)
class _BundleRecomputeResult:
    """Return value for bundle recomputation together with ν_f state."""

    bundles: MutableSequence[SupportsEPIBundle]
    analyzer_states: List[_AnalyzerState | None]


def _capture_analyzer_state(
    analyzer: NaturalFrequencyAnalyzer, current_index: int
) -> _AnalyzerState:
    """Serialise the incremental analyzer state for later reuse."""

    history_length = len(analyzer._history)
    if history_length <= 0:
        history_start: int | None = None
        history_end: int | None = None
    else:
        history_start = max(0, current_index - history_length + 1)
        history_end = current_index
    return _AnalyzerState(
        history_start=history_start,
        history_end=history_end,
        smoothed=tuple(analyzer._smoothed.items()),
        last_car=analyzer._last_car,
        last_snapshot=analyzer._last_snapshot,
    )


def _bundle_node_delta(bundle: SupportsEPIBundle) -> Mapping[str, float]:
    """Extract the ΔNFR distribution from the structured bundle."""

    node_deltas: Dict[str, float] = {}
    for node_name in (
        "tyres",
        "suspension",
        "chassis",
        "brakes",
        "transmission",
        "track",
        "driver",
    ):
        node = getattr(bundle, node_name, None)
        if node is None:
            continue
        try:
            delta = float(getattr(node, "delta_nfr", 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(delta):
            continue
        node_deltas[node_name] = delta
    return node_deltas


def _refresh_node_delta_cache(
    cache: List[Mapping[str, float]] | None,
    bundles: Sequence[SupportsEPIBundle],
    *,
    start_index: int = 0,
) -> List[Mapping[str, float]]:
    """Update ``cache`` with ΔNFR snapshots from ``bundles``."""

    total = len(bundles)
    effective_start = min(max(start_index, 0), total)
    if cache is None or len(cache) != total or effective_start <= 0:
        return [_bundle_node_delta(bundle) for bundle in bundles]
    cache[effective_start:] = [
        _bundle_node_delta(bundles[idx]) for idx in range(effective_start, total)
    ]
    return cache


def _bundle_track_gradient(bundle: SupportsEPIBundle) -> float:
    """Extract the track gradient from ``bundle`` guarding against NaNs."""

    track = getattr(bundle, "track", None)
    if track is None:
        return math.nan
    try:
        gradient_value = float(getattr(track, "gradient", 0.0))
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(gradient_value):
        return math.nan
    return gradient_value


def _refresh_track_gradient_cache(
    cache: List[float] | None,
    bundles: Sequence[SupportsEPIBundle],
    *,
    start_index: int = 0,
) -> List[float]:
    """Update ``cache`` with track gradient samples from ``bundles``."""

    total = len(bundles)
    effective_start = min(max(start_index, 0), total)
    if cache is None or len(cache) != total or effective_start <= 0:
        return [_bundle_track_gradient(bundle) for bundle in bundles]
    for idx in range(effective_start, total):
        cache[idx] = _bundle_track_gradient(bundles[idx])
    return cache


def _restore_analyzer_state(
    analyzer: NaturalFrequencyAnalyzer,
    state: _AnalyzerState,
    records: Sequence[SupportsTelemetrySample],
) -> int | None:
    """Restore a previously captured analyzer state."""

    analyzer.reset()
    history_end: int | None = None
    history_start = state.history_start
    history_stop = state.history_end
    if history_start is not None and history_stop is not None and records:
        start_index = max(0, history_start)
        stop_index = min(len(records) - 1, history_stop)
        if stop_index >= start_index:
            for index in range(start_index, stop_index + 1):
                analyzer._append_record(records[index])
            history_end = stop_index
    analyzer._smoothed = {key: value for key, value in state.smoothed}
    analyzer._last_car = state.last_car
    analyzer._last_snapshot = state.last_snapshot
    return history_end


def _accumulate_segment_metrics(
    records: Sequence[SupportsTelemetrySample], start: int, end: int
) -> _SegmentSummary:
    if start > end:
        return _SegmentSummary(
            curvature=0.0,
            brake_event=False,
            support_event=False,
            avg_vertical_load=0.0,
            duration=0.0,
            speed_drop=0.0,
            direction_changes=0,
            brake_temperatures={},
            brake_temperature_std={},
            end_timestamp=0.0,
        )

    brake_keys = tuple(f"brake_temp_{suffix}" for suffix in ("fl", "fr", "rl", "rr"))
    brake_sums = {key: 0.0 for key in brake_keys}
    brake_sums_sq = {key: 0.0 for key in brake_keys}
    brake_counts = {key: 0 for key in brake_keys}

    samples = end - start + 1
    brake_event = False
    sum_abs_lateral = 0.0
    vertical_sum = 0.0
    vertical_min = math.inf
    vertical_max = -math.inf
    direction_changes = 0
    last_sign = 0

    first_record = records[start]
    entry_speed = float(first_record.speed)
    min_speed = entry_speed
    start_timestamp = float(first_record.timestamp)
    end_timestamp = start_timestamp

    for idx in range(start, end + 1):
        record = records[idx]
        lateral_value = float(record.lateral_accel)
        sum_abs_lateral += abs(lateral_value)
        if abs(lateral_value) >= 0.25:
            sign = 1 if lateral_value > 0 else -1
            if last_sign and sign != last_sign:
                direction_changes += 1
            last_sign = sign

        longitudinal_value = float(record.longitudinal_accel)
        if longitudinal_value <= BRAKE_THRESHOLD:
            brake_event = True

        vertical_value = float(record.vertical_load)
        vertical_sum += vertical_value
        vertical_min = min(vertical_min, vertical_value)
        vertical_max = max(vertical_max, vertical_value)

        speed_value = float(record.speed)
        if speed_value < min_speed:
            min_speed = speed_value

        end_timestamp = float(record.timestamp)

        for key in brake_keys:
            raw_value = getattr(record, key, None)
            if raw_value is None:
                continue
            try:
                temperature_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(temperature_value):
                continue
            brake_counts[key] += 1
            brake_sums[key] += temperature_value
            brake_sums_sq[key] += temperature_value * temperature_value

    curvature = sum_abs_lateral / samples if samples else 0.0
    if math.isfinite(vertical_min) and math.isfinite(vertical_max):
        support_event = (vertical_max - vertical_min) >= SUPPORT_THRESHOLD
    else:
        support_event = False
    avg_vertical_load = vertical_sum / samples if samples else 0.0
    duration = max(0.0, end_timestamp - start_timestamp)
    speed_drop = max(0.0, entry_speed - min_speed)

    brake_temperatures: Dict[str, float] = {}
    brake_temperature_std: Dict[str, float] = {}
    for key, count in brake_counts.items():
        if count <= 0:
            brake_temperatures[key] = 0.0
            brake_temperature_std[f"{key}_std"] = 0.0
            continue
        mean_value = brake_sums[key] / count
        variance = (brake_sums_sq[key] / count) - (mean_value * mean_value)
        if variance < 0.0:
            variance = 0.0
        brake_temperatures[key] = mean_value
        brake_temperature_std[f"{key}_std"] = math.sqrt(variance) if variance > 0.0 else 0.0

    return _SegmentSummary(
        curvature=curvature,
        brake_event=brake_event,
        support_event=support_event,
        avg_vertical_load=avg_vertical_load,
        duration=duration,
        speed_drop=speed_drop,
        direction_changes=direction_changes,
        brake_temperatures=brake_temperatures,
        brake_temperature_std=brake_temperature_std,
        end_timestamp=end_timestamp,
    )


@dataclass(frozen=True)
class Goal:
    """Operational goal associated with a microsector phase."""

    phase: PhaseLiteral
    archetype: str
    description: str
    target_delta_nfr: float
    target_sense_index: float
    nu_f_target: float
    nu_exc_target: float
    rho_target: float
    target_phase_lag: float
    target_phase_alignment: float
    measured_phase_lag: float
    measured_phase_alignment: float
    target_phase_synchrony: float = field(init=False)
    measured_phase_synchrony: float = field(init=False)
    slip_lat_window: Tuple[float, float]
    slip_long_window: Tuple[float, float]
    yaw_rate_window: Tuple[float, float]
    dominant_nodes: Tuple[str, ...]
    target_delta_nfr_long: float = 0.0
    target_delta_nfr_lat: float = 0.0
    delta_axis_weights: Mapping[str, float] = field(
        default_factory=lambda: {"longitudinal": 0.5, "lateral": 0.5}
    )
    archetype_delta_nfr_long_target: float = 0.0
    archetype_delta_nfr_lat_target: float = 0.0
    archetype_nu_f_target: float = 0.0
    archetype_si_phi_target: float = 0.0
    detune_ratio_weights: Mapping[str, float] = field(
        default_factory=lambda: {"longitudinal": 0.5, "lateral": 0.5}
    )
    track_gradient: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_phase_synchrony",
            phase_synchrony_index(self.target_phase_lag, self.target_phase_alignment),
        )
        object.__setattr__(
            self,
            "measured_phase_synchrony",
            phase_synchrony_index(self.measured_phase_lag, self.measured_phase_alignment),
        )


@dataclass(frozen=True)
class Microsector:
    """Segment of the lap grouped by curvature and ΔNFR behaviour."""

    index: int
    start_time: float
    end_time: float
    curvature: float
    brake_event: bool
    support_event: bool
    delta_nfr_signature: float
    goals: Tuple[Goal, ...]
    phase_boundaries: Mapping[PhaseLiteral, Tuple[int, int]]
    phase_samples: Mapping[PhaseLiteral, Tuple[int, ...]]
    active_phase: PhaseLiteral
    dominant_nodes: Mapping[PhaseLiteral, Tuple[str, ...]]
    phase_weights: Mapping[PhaseLiteral, Mapping[str, float] | float]
    grip_rel: float
    phase_lag: Mapping[PhaseLiteral, float]
    phase_alignment: Mapping[PhaseLiteral, float]
    phase_synchrony: Mapping[PhaseLiteral, float] = field(default_factory=dict)
    phase_motor_latency: Mapping[PhaseLiteral, float] = field(default_factory=dict)
    motor_latency_ms: float = 0.0
    filtered_measures: Mapping[str, object] = field(default_factory=dict)
    recursivity_trace: Tuple[Mapping[str, float | str | None], ...] = ()
    last_mutation: Mapping[str, object] | None = None
    window_occupancy: Mapping[PhaseLiteral, Mapping[str, float]] = field(
        default_factory=dict
    )
    delta_nfr_std: float = 0.0
    nodal_delta_nfr_std: float = 0.0
    phase_delta_nfr_std: Mapping[PhaseLiteral, float] = field(default_factory=dict)
    phase_nodal_delta_nfr_std: Mapping[PhaseLiteral, float] = field(
        default_factory=dict
    )
    delta_nfr_entropy: float = 0.0
    node_entropy: float = 0.0
    phase_delta_nfr_entropy: Mapping[PhaseLiteral, float] = field(
        default_factory=dict
    )
    phase_node_entropy: Mapping[PhaseLiteral, float] = field(default_factory=dict)
    phase_axis_targets: Mapping[PhaseLiteral, Mapping[str, float]] = field(
        default_factory=dict
    )
    phase_axis_weights: Mapping[PhaseLiteral, Mapping[str, float]] = field(
        default_factory=dict
    )
    context_factors: Mapping[str, float] = field(default_factory=dict)
    sample_context_factors: Mapping[int, Mapping[str, float]] = field(
        default_factory=dict
    )
    operator_events: Mapping[str, Tuple[Mapping[str, object], ...]] = field(
        default_factory=dict
    )

    def phase_indices(self, phase: PhaseLiteral) -> range:
        """Return the range of sample indices assigned to ``phase``."""

        if phase in self.phase_boundaries:
            start, stop = self.phase_boundaries[phase]
        else:
            ranges = [
                self.phase_boundaries[candidate]
                for candidate in expand_phase_alias(phase)
                if candidate in self.phase_boundaries
            ]
            if not ranges:
                raise KeyError(phase)
            start = min(entry[0] for entry in ranges)
            stop = max(entry[1] for entry in ranges)
        return range(start, stop)


def _merge_phase_indices(sequences: Sequence[Iterable[int]]) -> Tuple[int, ...]:
    merged: list[int] = []
    for sequence in sequences:
        for value in sequence:
            index = int(value)
            if not merged or merged[-1] != index:
                merged.append(index)
    return tuple(merged)


def _segment_index_range(metadata: object, start: int, end: int) -> range:
    """Return the inclusive range of indices covered by a segment payload."""

    if isinstance(metadata, range):
        return metadata
    if (
        isinstance(metadata, tuple)
        and len(metadata) == 2
        and all(isinstance(value, int) for value in metadata)
    ):
        first, last = metadata
        if last < first:
            first, last = last, first
        return range(first, last + 1)
    return range(start, end + 1)


def _serialise_sample_context(
    metadata: object,
    start: int,
    end: int,
    sample_context: Sequence[ContextFactors],
) -> Dict[int, Mapping[str, float]]:
    """Materialise contextual factors for the indices referenced by ``metadata``."""

    if not sample_context:
        return {}

    resolved: Dict[int, Mapping[str, float]] = {}
    if isinstance(metadata, Mapping):
        source = metadata.items()
    else:
        indices = _segment_index_range(metadata, start, end)
        source = (
            (idx, sample_context[idx])
            for idx in indices
            if 0 <= idx < len(sample_context)
        )

    for idx, factors in source:
        if not isinstance(idx, int):
            continue
        if 0 <= idx < len(sample_context):
            candidate = sample_context[idx]
        else:
            candidate = factors
        if hasattr(candidate, "as_mapping"):
            resolved[idx] = candidate.as_mapping()
        else:
            resolved[idx] = dict(candidate)
    return resolved


def segment_microsectors(
    records: Sequence[SupportsTelemetrySample],
    bundles: Sequence[SupportsEPIBundle],
    *,
    operator_state: MutableMapping[str, Mapping[str, object]] | None = None,
    recursion_decay: float = 0.4,
    mutation_thresholds: Mapping[str, float] | None = None,
    phase_weight_overrides: Mapping[str, Mapping[str, float] | float] | None = None,
    baseline: SupportsTelemetrySample | None = None,
) -> List[Microsector]:
    """Derive microsectors from telemetry and ΔNFR signatures.

    Parameters
    ----------
    records:
        Telemetry samples in chronological order. Each instance must implement
        :class:`~tnfr_core.operators.interfaces.SupportsTelemetrySample` and satisfy
        :class:`~tnfr_core.operators.interfaces.SupportsContextRecord` so the
        contextual weighting heuristics can access the required signals.
    bundles:
        Computed :class:`~tnfr_core.operators.interfaces.SupportsEPIBundle` entries for
        the same timestamps as ``records``. Every bundle must also implement the
        :class:`~tnfr_core.operators.interfaces.SupportsContextBundle` contract.
    baseline:
        Optional baseline sample obtained during EPI extraction. When omitted the
        function derives a baseline from ``records`` before computing the
        microsectors.

    Returns
    -------
    list of :class:`Microsector`
        Each microsector contains phase objectives bound to an archetype. When
        ``phase_weight_overrides`` is provided the heuristically derived
        weighting profiles for every phase are blended with the supplied
        multipliers before recomputing ΔNFR/Sense Index bundles.
    """

    if len(records) != len(bundles):
        raise ValueError("records and bundles must have the same length")
    if recursion_decay < 0.0 or recursion_decay >= 1.0:
        raise ValueError("recursion_decay must be in the [0, 1) interval")

    if records and not isinstance(records[0], SupportsContextRecord):
        raise TypeError("records must provide lateral/vertical/longitudinal signals")
    if bundles and not isinstance(bundles[0], SupportsContextBundle):
        raise TypeError("bundles must expose chassis, tyres and transmission nodes")

    baseline_record = baseline or DeltaCalculator.derive_baseline(records)
    context_matrix = load_context_matrix()
    bundle_list = list(bundles)
    microsectors: List[Microsector] = []
    rec_state_root: RecursivityStateRoot | None = None
    mutation_state: MutableMapping[str, Dict[str, object]] | None = None
    if operator_state is not None:
        rec_state_root = cast(
            RecursivityStateRoot,
            operator_state.setdefault(
                "recursivity", cast(RecursivityStateRoot, {})
            ),
        )
        mutation_state = operator_state.setdefault("mutation", {})
    thresholds = mutation_thresholds or {}
    total_samples = len(records)
    default_phase = PHASE_SEQUENCE[0] if PHASE_SEQUENCE else "entry1"
    assigned_phases: List[PhaseLiteral] = [default_phase] * total_samples
    assigned_weights: List[Mapping[str, Mapping[str, float] | float]] = [
        DEFAULT_PHASE_WEIGHTS
    ] * total_samples
    specs: List[Dict[str, object]] = []
    baseline_vertical = float(getattr(baseline_record, "vertical_load", 0.0))
    sample_rate = estimate_sample_rate(records)
    steer_series = _extract_signal(records, "steer")
    lat_series = _extract_signal(records, "lateral_accel")
    yaw_rate_cache = [
        _resolve_yaw_rate(records, idx) for idx in range(len(records))
    ]
    steer_norm = _normalise_signal(steer_series)
    yaw_norm = _normalise_signal(yaw_rate_cache)
    lat_norm = _normalise_signal(lat_series)

    segments = _identify_corner_segments(records)
    if not segments:
        return []

    sample_context: List[ContextFactors] = []
    sample_multipliers: List[float] = []
    for record in records:
        factors = resolve_context_from_record(
            context_matrix,
            record,
            baseline_vertical_load=baseline_vertical,
        )
        sample_context.append(factors)
        sample_multipliers.append(
            _resolve_context_multiplier(
                factors,
                context_matrix=context_matrix,
            )
        )
    session_components = _resolve_session_components(records, baseline_record)
    node_delta_cache: List[Mapping[str, float]] | None = None
    track_gradient_cache: List[float] | None = None

    for index, (start, end) in enumerate(segments):
        phase_boundaries = _compute_phase_boundaries(records, start, end)
        phase_samples = {
            phase: range(bounds[0], bounds[1])
            for phase, bounds in phase_boundaries.items()
        }
        segment_summary = _accumulate_segment_metrics(records, start, end)
        curvature = segment_summary.curvature
        brake_event = segment_summary.brake_event
        support_event = segment_summary.support_event
        avg_vertical_load = segment_summary.avg_vertical_load
        grip_rel = (
            avg_vertical_load / baseline_vertical if baseline_vertical > 1e-9 else 0.0
        )
        duration = segment_summary.duration
        speed_drop = segment_summary.speed_drop
        direction_changes = segment_summary.direction_changes
        brake_temperatures = segment_summary.brake_temperatures
        brake_temperature_dispersion = segment_summary.brake_temperature_std
        phase_weight_map = _initial_phase_weight_map(
            records, phase_samples, yaw_rate_cache
        )
        if phase_weight_overrides:
            phase_weight_map = _blend_phase_weight_map(
                phase_weight_map, phase_weight_overrides
            )
        context_factors = resolve_microsector_context(
            context_matrix,
            curvature=curvature,
            grip_rel=grip_rel if grip_rel > 0.0 else 1.0,
            speed_drop=speed_drop,
            direction_changes=float(direction_changes),
        )
        segment_span = (start, end)
        specs.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "curvature": curvature,
                "brake_event": brake_event,
                "support_event": support_event,
                "avg_vertical_load": avg_vertical_load,
                "grip_rel": grip_rel,
                "phase_boundaries": phase_boundaries,
                "phase_samples": phase_samples,
                "phase_weights": phase_weight_map,
                "brake_temperatures": brake_temperatures,
                "brake_temperature_std": brake_temperature_dispersion,
                "duration": duration,
                "speed_drop": speed_drop,
                "direction_changes": direction_changes,
                "end_timestamp": segment_summary.end_timestamp,
                "context_factors": context_factors,
                "sample_context": segment_span,
                "context_multipliers": segment_span,
                "goals": (),
                "dominant_nodes": {},
                "phase_axis_targets": {},
                "phase_axis_weights": {},
                "goal_cache_valid": False,
                "goal_archetype": None,
                "goal_gradient_signature": None,
                "goal_phase_cache": None,
            }
        )
        for phase, indices in phase_samples.items():
            for sample_index in indices:
                if 0 <= sample_index < total_samples:
                    assigned_phases[sample_index] = phase
                    assigned_weights[sample_index] = phase_weight_map

    analyzer_states: List[_AnalyzerState | None] | None = None

    recompute_result = _recompute_bundles(
        records,
        bundle_list,
        baseline_record,
        assigned_phases,
        assigned_weights,
        analyzer_states=analyzer_states,
    )
    recomputed_bundles = recompute_result.bundles
    analyzer_states = recompute_result.analyzer_states
    node_delta_cache = _refresh_node_delta_cache(
        node_delta_cache, recomputed_bundles
    )
    track_gradient_cache = _refresh_track_gradient_cache(
        track_gradient_cache, recomputed_bundles
    )

    weights_adjusted, weight_start_index = _adjust_phase_weights_with_dominance(
        specs,
        recomputed_bundles,
        records,
        context_matrix=context_matrix,
        sample_context=sample_context,
        sample_multipliers=sample_multipliers,
        yaw_rates=yaw_rate_cache,
        node_delta_cache=node_delta_cache,
        sample_rate=sample_rate,
        steer_norm=steer_norm,
        yaw_norm=yaw_norm,
        lat_norm=lat_norm,
    )

    if weights_adjusted:
        recompute_start = weight_start_index if weight_start_index is not None else 0
        recompute_result = _recompute_bundles(
            records,
            recomputed_bundles,
            baseline_record,
            assigned_phases,
            assigned_weights,
            start_index=recompute_start,
            analyzer_states=analyzer_states,
        )
        recomputed_bundles = recompute_result.bundles
        analyzer_states = recompute_result.analyzer_states
        node_delta_cache = _refresh_node_delta_cache(
            node_delta_cache,
            recomputed_bundles,
            start_index=recompute_start,
        )
        track_gradient_cache = _refresh_track_gradient_cache(
            track_gradient_cache,
            recomputed_bundles,
            start_index=recompute_start,
        )
        _invalidate_goal_cache(specs, recompute_start)

    goal_nu_f_lookup: Dict[int, float] = {}
    goal_start_index: int | None = None
    for spec in specs:
        start = spec["start"]
        end = spec["end"]
        multiplier_meta = spec.get("context_multipliers")
        multiplier_indices = _segment_index_range(multiplier_meta, start, end)
        adjusted_deltas = []
        for offset, bundle in zip(multiplier_indices, recomputed_bundles[start : end + 1]):
            multiplier = None
            if isinstance(multiplier_meta, Mapping):
                multiplier = multiplier_meta.get(offset)
            if multiplier is None and 0 <= offset < len(sample_multipliers):
                multiplier = sample_multipliers[offset]
            if multiplier is None:
                multiplier = 1.0
            adjusted_deltas.append(bundle.delta_nfr * multiplier)
        spec["adjusted_deltas"] = tuple(adjusted_deltas)
        delta_signature = fmean(adjusted_deltas)
        spec["delta_signature"] = float(delta_signature)
        avg_si = fmean(b.sense_index for b in recomputed_bundles[start : end + 1])
        spec["avg_si"] = float(avg_si)
        archetype = _classify_archetype(
            spec["curvature"],
            spec.get("duration", 0.0),
            spec.get("speed_drop", 0.0),
            spec.get("direction_changes", 0),
        )
        (
            phase_dominant_nodes,
            phase_nu_f_targets,
            phase_dominance_weights,
            sample_nu_f_lookup,
        ) = _phase_nu_f_targets(
            recomputed_bundles,
            records,
            spec["phase_boundaries"],
            spec.get("phase_samples"),
            yaw_rates=yaw_rate_cache,
            context_matrix=context_matrix,
            sample_context=sample_context,
            node_delta_cache=node_delta_cache,
        )
        _store_phase_goal_cache(
            spec,
            phase_dominant_nodes,
            phase_nu_f_targets,
            phase_dominance_weights,
            sample_nu_f_lookup,
        )
        for sample_index, nu_f_target in sample_nu_f_lookup.items():
            goal_nu_f_lookup[sample_index] = nu_f_target
            start_candidate = spec.get("start")
            if not isinstance(start_candidate, int):
                start_candidate = sample_index
            if goal_start_index is None:
                goal_start_index = start_candidate
            else:
                goal_start_index = min(goal_start_index, start_candidate)

    if goal_nu_f_lookup:
        goal_recompute_start = goal_start_index if goal_start_index is not None else 0
        recompute_result = _recompute_bundles(
            records,
            recomputed_bundles,
            baseline_record,
            assigned_phases,
            assigned_weights,
            goal_nu_f_lookup=goal_nu_f_lookup,
            start_index=goal_recompute_start,
            analyzer_states=analyzer_states,
        )
        recomputed_bundles = recompute_result.bundles
        analyzer_states = recompute_result.analyzer_states
        node_delta_cache = _refresh_node_delta_cache(
            node_delta_cache,
            recomputed_bundles,
            start_index=goal_recompute_start,
        )
        track_gradient_cache = _refresh_track_gradient_cache(
            track_gradient_cache,
            recomputed_bundles,
            start_index=goal_recompute_start,
        )
        _invalidate_goal_cache(specs, goal_recompute_start)

    for spec in specs:
        start = spec["start"]
        end = spec["end"]
        record_window = records[start : end + 1]
        bundle_window = recomputed_bundles[start : end + 1]
        phase_boundaries = spec["phase_boundaries"]
        phase_samples = spec["phase_samples"]
        phase_weights = {
            phase: dict(profile)
            for phase, profile in spec["phase_weights"].items()
        }
        if spec.get("goal_phase_cache") is None:
            (
                phase_dominant_nodes,
                phase_nu_f_targets,
                phase_dominance_weights,
                sample_nu_f_lookup,
            ) = _phase_nu_f_targets(
                recomputed_bundles,
                records,
                phase_boundaries,
                phase_samples,
                yaw_rates=yaw_rate_cache,
                context_matrix=context_matrix,
                sample_context=sample_context,
                node_delta_cache=node_delta_cache,
            )
            _store_phase_goal_cache(
                spec,
                phase_dominant_nodes,
                phase_nu_f_targets,
                phase_dominance_weights,
                sample_nu_f_lookup,
            )
        curvature = spec["curvature"]
        brake_event = spec["brake_event"]
        support_event = spec["support_event"]
        avg_vertical_load = float(spec.get("avg_vertical_load", 0.0))
        grip_rel = float(spec.get("grip_rel", 0.0))
        duration = float(spec.get("duration", 0.0))
        speed_drop = float(spec.get("speed_drop", 0.0))
        direction_changes = spec.get("direction_changes", 0)
        stored_deltas = spec.get("adjusted_deltas")
        if stored_deltas is None:
            multiplier_meta = spec.get("context_multipliers")
            multiplier_indices = _segment_index_range(multiplier_meta, start, end)
            adjusted_deltas = []
            for offset, bundle in zip(multiplier_indices, bundle_window):
                multiplier = None
                if isinstance(multiplier_meta, Mapping):
                    multiplier = multiplier_meta.get(offset)
                if multiplier is None and 0 <= offset < len(sample_multipliers):
                    multiplier = sample_multipliers[offset]
                if multiplier is None:
                    multiplier = 1.0
                adjusted_deltas.append(bundle.delta_nfr * multiplier)
            stored_deltas = tuple(adjusted_deltas)
            spec["adjusted_deltas"] = stored_deltas
        delta_signature = spec.get("delta_signature")
        if delta_signature is None:
            delta_signature = fmean(stored_deltas)
            spec["delta_signature"] = float(delta_signature)
        avg_si = spec.get("avg_si")
        if avg_si is None:
            avg_si = fmean(b.sense_index for b in bundle_window)
            spec["avg_si"] = float(avg_si)
        archetype = _classify_archetype(
            curvature,
            duration,
            speed_drop,
            direction_changes,
        )
        phase_gradient_map: Dict[PhaseLiteral, float] = {}
        gradient_samples: List[float] = []
        for phase, (phase_start, phase_stop) in phase_boundaries.items():
            phase_values: List[float] = []
            for idx in range(phase_start, min(phase_stop, len(recomputed_bundles))):
                if not (0 <= idx < len(recomputed_bundles)):
                    continue
                if track_gradient_cache is None:
                    continue
                if not (0 <= idx < len(track_gradient_cache)):
                    continue
                gradient_value = track_gradient_cache[idx]
                if not math.isfinite(gradient_value):
                    continue
                phase_values.append(gradient_value)
                gradient_samples.append(gradient_value)
            phase_gradient_map[phase] = fmean(phase_values) if phase_values else 0.0
        microsector_gradient = fmean(gradient_samples) if gradient_samples else 0.0

        gradient_signature = _phase_gradient_signature(phase_gradient_map)
        cache_valid = bool(spec.get("goal_cache_valid"))
        cached_archetype = spec.get("goal_archetype")
        cached_signature = spec.get("goal_gradient_signature")
        (
            cached_dominant_nodes,
            cached_nu_f_targets,
            cached_dominance_weights,
            _,
        ) = _extract_phase_goal_cache(spec)
        if (
            not cache_valid
            or cached_archetype != archetype
            or cached_signature != gradient_signature
        ):
            (
                goals_tuple,
                dominant_nodes_map,
                axis_targets_map,
                axis_weights_map,
            ) = _build_goals(
                archetype,
                recomputed_bundles,
                records,
                phase_boundaries,
                yaw_rates=yaw_rate_cache,
                context_matrix=context_matrix,
                sample_context=sample_context,
                sample_multipliers=sample_multipliers,
                node_delta_cache=node_delta_cache,
                phase_gradients=phase_gradient_map,
                phase_dominant_nodes=cached_dominant_nodes,
                phase_nu_f_targets=cached_nu_f_targets,
                phase_dominance_weights=cached_dominance_weights,
                sample_rate=sample_rate,
                steer_norm=steer_norm,
                yaw_norm=yaw_norm,
                lat_norm=lat_norm,
            )
            spec["goals"] = goals_tuple
            spec["dominant_nodes"] = dominant_nodes_map
            spec["phase_axis_targets"] = axis_targets_map
            spec["phase_axis_weights"] = axis_weights_map
            spec["goal_cache_valid"] = True
            spec["goal_archetype"] = archetype
            spec["goal_gradient_signature"] = gradient_signature
        goals = cast(Tuple[Goal, ...], spec.get("goals", ()))
        dominant_nodes = cast(
            Mapping[PhaseLiteral, Tuple[str, ...]],
            spec.get("dominant_nodes", {}),
        )
        axis_targets = cast(
            Mapping[PhaseLiteral, Mapping[str, float]],
            spec.get("phase_axis_targets", {}),
        )
        axis_weights = cast(
            Mapping[PhaseLiteral, Mapping[str, float]],
            spec.get("phase_axis_weights", {}),
        )
        active_goal = max(
            goals,
            key=lambda goal: abs(goal.target_delta_nfr) + goal.nu_f_target,
        )
        spec["active_phase"] = active_goal.phase
        context_factors_obj = spec.get("context_factors")
        if context_factors_obj is None:
            context_factors_obj = resolve_microsector_context(
                context_matrix,
                curvature=curvature,
                grip_rel=grip_rel if grip_rel > 0.0 else 1.0,
                speed_drop=speed_drop,
                direction_changes=float(direction_changes),
            )
        surface_band = context_matrix.surface_band(max(grip_rel, 1e-9))
        surface_label = surface_band[3] or "neutral"
        surface_factor = float(getattr(context_factors_obj, "surface", 1.0))
        entry_goals = [goal for goal in goals if phase_family(goal.phase) == "entry"]
        if entry_goals:
            base_delta_threshold = fmean(abs(goal.target_delta_nfr) for goal in entry_goals)
        else:
            base_delta_threshold = abs(delta_signature)
        delta_threshold = max(0.05, base_delta_threshold * surface_factor)
        filtered_measures: Dict[str, object] = {
            "thermal_load": avg_vertical_load,
            "style_index": avg_si,
            "grip_rel": grip_rel,
        }
        filtered_measures["gradient"] = microsector_gradient
        brake_temperatures = {
            key: float(value)
            for key, value in spec.get("brake_temperatures", {}).items()
        }
        brake_temperature_std = {
            key: float(value)
            for key, value in spec.get("brake_temperature_std", {}).items()
        }
        filtered_measures.update(brake_temperatures)
        filtered_measures.update(brake_temperature_std)
        local_phase_indices = {
            phase: tuple(
                index - start
                for index in indices
                if start <= index <= end
            )
            for phase, indices in phase_samples.items()
        }
        window_yaw_rates = yaw_rate_cache[start : end + 1]

        window_metrics = compute_window_metrics(
            record_window,
            bundles=bundle_window,
            phase_indices=local_phase_indices,
        )
        front_velocity = window_metrics.suspension_velocity_front
        rear_velocity = window_metrics.suspension_velocity_rear
        filtered_measures.update(
            {
                "motor_latency_ms": window_metrics.motor_latency_ms,
                "d_nfr_couple": window_metrics.d_nfr_couple,
                "d_nfr_res": window_metrics.d_nfr_res,
                "d_nfr_flat": window_metrics.d_nfr_flat,
                "nu_f": window_metrics.nu_f,
                "nu_exc": window_metrics.nu_exc,
                "rho": window_metrics.rho,
                "udr": window_metrics.useful_dissonance_ratio,
                "phase_lag_window": window_metrics.phase_lag,
                "phase_alignment_window": window_metrics.phase_alignment,
                "phase_synchrony_window": window_metrics.phase_synchrony_index,
                "coherence_index": window_metrics.coherence_index,
                "ackermann_parallel_index": window_metrics.ackermann_parallel_index,
                "slide_catch_budget": window_metrics.slide_catch_budget.value,
                "slide_catch_budget_yaw": window_metrics.slide_catch_budget.yaw_acceleration_ratio,
                "slide_catch_budget_steer": window_metrics.slide_catch_budget.steer_velocity_ratio,
                "slide_catch_budget_overshoot": window_metrics.slide_catch_budget.overshoot_ratio,
                "locking_window_score": window_metrics.locking_window_score.value,
                "locking_window_score_on": window_metrics.locking_window_score.on_throttle,
                "locking_window_score_off": window_metrics.locking_window_score.off_throttle,
                "locking_window_transitions": float(
                    window_metrics.locking_window_score.transition_samples
                ),
                "support_effective": window_metrics.support_effective,
                "load_support_ratio": window_metrics.load_support_ratio,
                "structural_expansion_longitudinal": window_metrics.structural_expansion_longitudinal,
                "structural_contraction_longitudinal": window_metrics.structural_contraction_longitudinal,
                "structural_expansion_lateral": window_metrics.structural_expansion_lateral,
                "structural_contraction_lateral": window_metrics.structural_contraction_lateral,
                "bottoming_ratio_front": window_metrics.bottoming_ratio_front,
                "bottoming_ratio_rear": window_metrics.bottoming_ratio_rear,
                "bumpstop_front_density": window_metrics.bumpstop_histogram.front_total_density,
                "bumpstop_rear_density": window_metrics.bumpstop_histogram.rear_total_density,
                "bumpstop_front_energy": window_metrics.bumpstop_histogram.front_total_energy,
                "bumpstop_rear_energy": window_metrics.bumpstop_histogram.rear_total_energy,
                "mu_usage_front_ratio": window_metrics.mu_usage_front_ratio,
                "mu_usage_rear_ratio": window_metrics.mu_usage_rear_ratio,
                "mu_balance": window_metrics.mu_balance,
                "brake_longitudinal_correlation": window_metrics.brake_longitudinal_correlation,
                "throttle_longitudinal_correlation": window_metrics.throttle_longitudinal_correlation,
                "delta_nfr_std": window_metrics.delta_nfr_std,
                "nodal_delta_nfr_std": window_metrics.nodal_delta_nfr_std,
                "exit_gear_match": window_metrics.exit_gear_match,
                "shift_stability": window_metrics.shift_stability,
                "suspension_velocity_front_compression_low_ratio": front_velocity.compression_low_ratio,
                "suspension_velocity_front_compression_medium_ratio": front_velocity.compression_medium_ratio,
                "suspension_velocity_front_compression_high_ratio": front_velocity.compression_high_ratio,
                "suspension_velocity_front_rebound_low_ratio": front_velocity.rebound_low_ratio,
                "suspension_velocity_front_rebound_medium_ratio": front_velocity.rebound_medium_ratio,
                "suspension_velocity_front_rebound_high_ratio": front_velocity.rebound_high_ratio,
                "suspension_velocity_front_high_speed_pct": front_velocity.compression_high_speed_percentage,
                "suspension_velocity_front_high_speed_rebound_pct": front_velocity.rebound_high_speed_percentage,
                "suspension_velocity_front_ar_index": front_velocity.ar_index,
                "suspension_velocity_rear_compression_low_ratio": rear_velocity.compression_low_ratio,
                "suspension_velocity_rear_compression_medium_ratio": rear_velocity.compression_medium_ratio,
                "suspension_velocity_rear_compression_high_ratio": rear_velocity.compression_high_ratio,
                "suspension_velocity_rear_rebound_low_ratio": rear_velocity.rebound_low_ratio,
                "suspension_velocity_rear_rebound_medium_ratio": rear_velocity.rebound_medium_ratio,
                "suspension_velocity_rear_rebound_high_ratio": rear_velocity.rebound_high_ratio,
                "suspension_velocity_rear_high_speed_pct": rear_velocity.compression_high_speed_percentage,
                "suspension_velocity_rear_high_speed_rebound_pct": rear_velocity.rebound_high_speed_percentage,
                "suspension_velocity_rear_ar_index": rear_velocity.ar_index,
                "aero_low_imbalance": window_metrics.aero_coherence.low_speed_imbalance,
                "aero_medium_imbalance": window_metrics.aero_coherence.medium_speed_imbalance,
                "aero_high_imbalance": window_metrics.aero_coherence.high_speed_imbalance,
                "aero_low_samples": float(window_metrics.aero_coherence.low_speed_samples),
                "aero_medium_samples": float(
                    window_metrics.aero_coherence.medium_speed_samples
                ),
                "aero_high_samples": float(window_metrics.aero_coherence.high_speed_samples),
                "aero_low_front_total": window_metrics.aero_coherence.low_speed.total.front,
                "aero_low_rear_total": window_metrics.aero_coherence.low_speed.total.rear,
                "aero_medium_front_total": window_metrics.aero_coherence.medium_speed.total.front,
                "aero_medium_rear_total": window_metrics.aero_coherence.medium_speed.total.rear,
                "aero_high_front_total": window_metrics.aero_coherence.high_speed.total.front,
                "aero_high_rear_total": window_metrics.aero_coherence.high_speed.total.rear,
                "aero_high_front_lateral": window_metrics.aero_coherence.high_speed.lateral.front,
                "aero_high_rear_lateral": window_metrics.aero_coherence.high_speed.lateral.rear,
                "aero_high_front_longitudinal": window_metrics.aero_coherence.high_speed.longitudinal.front,
                "aero_high_rear_longitudinal": window_metrics.aero_coherence.high_speed.longitudinal.rear,
                "aero_mechanical_coherence": window_metrics.aero_mechanical_coherence,
                "aero_drift_guidance": window_metrics.aero_balance_drift.guidance,
                "aero_drift_mu_tolerance": window_metrics.aero_balance_drift.mu_tolerance,
                "aero_drift_low_mu_delta": window_metrics.aero_balance_drift.low_speed.mu_delta,
                "aero_drift_low_mu_ratio": window_metrics.aero_balance_drift.low_speed.mu_ratio,
                "aero_drift_low_rake": window_metrics.aero_balance_drift.low_speed.rake_mean,
                "aero_drift_low_rake_deg": window_metrics.aero_balance_drift.low_speed.rake_deg,
                "aero_drift_low_samples": float(window_metrics.aero_balance_drift.low_speed.samples),
                "aero_drift_low_mu_balance_slope": window_metrics.aero_balance_drift.low_speed.mu_balance_slope,
                "aero_drift_low_mu_balance_sign_change": (
                    window_metrics.aero_balance_drift.low_speed.mu_balance_sign_change
                ),
                "aero_drift_medium_mu_delta": window_metrics.aero_balance_drift.medium_speed.mu_delta,
                "aero_drift_medium_mu_ratio": window_metrics.aero_balance_drift.medium_speed.mu_ratio,
                "aero_drift_medium_rake": window_metrics.aero_balance_drift.medium_speed.rake_mean,
                "aero_drift_medium_rake_deg": window_metrics.aero_balance_drift.medium_speed.rake_deg,
                "aero_drift_medium_samples": float(window_metrics.aero_balance_drift.medium_speed.samples),
                "aero_drift_medium_mu_balance_slope": window_metrics.aero_balance_drift.medium_speed.mu_balance_slope,
                "aero_drift_medium_mu_balance_sign_change": (
                    window_metrics.aero_balance_drift.medium_speed.mu_balance_sign_change
                ),
                "aero_drift_high_mu_delta": window_metrics.aero_balance_drift.high_speed.mu_delta,
                "aero_drift_high_mu_ratio": window_metrics.aero_balance_drift.high_speed.mu_ratio,
                "aero_drift_high_rake": window_metrics.aero_balance_drift.high_speed.rake_mean,
                "aero_drift_high_rake_deg": window_metrics.aero_balance_drift.high_speed.rake_deg,
                "aero_drift_high_samples": float(window_metrics.aero_balance_drift.high_speed.samples),
                "aero_drift_high_mu_balance_slope": window_metrics.aero_balance_drift.high_speed.mu_balance_slope,
                "aero_drift_high_mu_balance_sign_change": (
                    window_metrics.aero_balance_drift.high_speed.mu_balance_sign_change
                ),
                "si_variance": window_metrics.si_variance,
                "epi_derivative_abs": window_metrics.epi_derivative_abs,
                "brake_headroom": window_metrics.brake_headroom.value,
                "brake_headroom_peak_decel": window_metrics.brake_headroom.peak_decel,
                "brake_headroom_abs_activation": window_metrics.brake_headroom.abs_activation_ratio,
                "brake_headroom_partial_locking": window_metrics.brake_headroom.partial_locking_ratio,
                "brake_headroom_sustained_locking": window_metrics.brake_headroom.sustained_locking_ratio,
            }
        )
        phase_motor_latency_payload = {
            str(label): float(value)
            for label, value in window_metrics.phase_motor_latency_ms.items()
        }
        if phase_motor_latency_payload:
            filtered_measures["phase_motor_latency_ms"] = phase_motor_latency_payload
        headroom = window_metrics.brake_headroom
        temperature_available = getattr(headroom, "temperature_available", True)
        fade_available = getattr(headroom, "fade_available", True)
        symmetry_map = getattr(window_metrics, "mu_symmetry", {}) or {}
        window_symmetry = symmetry_map.get("window")
        if isinstance(window_symmetry, Mapping):
            try:
                filtered_measures["mu_symmetry_front"] = float(window_symmetry.get("front", 0.0))
            except (TypeError, ValueError):
                filtered_measures["mu_symmetry_front"] = 0.0
            try:
                filtered_measures["mu_symmetry_rear"] = float(window_symmetry.get("rear", 0.0))
            except (TypeError, ValueError):
                filtered_measures["mu_symmetry_rear"] = 0.0
        else:
            filtered_measures.setdefault("mu_symmetry_front", 0.0)
            filtered_measures.setdefault("mu_symmetry_rear", 0.0)
        for phase_label, components in symmetry_map.items():
            if phase_label == "window" or not isinstance(components, Mapping):
                continue
            try:
                front_value = float(components.get("front", 0.0))
            except (TypeError, ValueError):
                front_value = 0.0
            try:
                rear_value = float(components.get("rear", 0.0))
            except (TypeError, ValueError):
                rear_value = 0.0
            filtered_measures[f"mu_symmetry_{phase_label}_front"] = front_value
            filtered_measures[f"mu_symmetry_{phase_label}_rear"] = rear_value
        if fade_available and math.isfinite(headroom.fade_slope):
            filtered_measures["brake_headroom_fade_slope"] = headroom.fade_slope
        else:
            filtered_measures["brake_headroom_fade_slope"] = None
        if fade_available and math.isfinite(headroom.fade_ratio):
            filtered_measures["brake_headroom_fade_ratio"] = headroom.fade_ratio
        else:
            filtered_measures["brake_headroom_fade_ratio"] = None
        if temperature_available and math.isfinite(headroom.temperature_peak):
            filtered_measures["brake_headroom_temperature_peak"] = headroom.temperature_peak
        else:
            filtered_measures["brake_headroom_temperature_peak"] = None
        if temperature_available and math.isfinite(headroom.temperature_mean):
            filtered_measures["brake_headroom_temperature_mean"] = headroom.temperature_mean
        else:
            filtered_measures["brake_headroom_temperature_mean"] = None
        if temperature_available and math.isfinite(headroom.ventilation_index):
            filtered_measures["brake_headroom_ventilation_index"] = headroom.ventilation_index
        else:
            filtered_measures["brake_headroom_ventilation_index"] = None
        filtered_measures["brake_headroom_temperature_available"] = temperature_available
        filtered_measures["brake_headroom_fade_available"] = fade_available
        if window_metrics.cphi:
            for suffix, wheel in window_metrics.cphi.items():
                value = float(wheel.value)
                filtered_measures[f"cphi_{suffix}"] = (
                    value if math.isfinite(value) else None
                )
                temp_component = float(wheel.temperature_component)
                filtered_measures[f"cphi_{suffix}_temperature"] = (
                    temp_component if math.isfinite(temp_component) else None
                )
                gradient_component = float(wheel.gradient_component)
                filtered_measures[f"cphi_{suffix}_gradient"] = (
                    gradient_component if math.isfinite(gradient_component) else None
                )
                mu_component = float(wheel.mu_component)
                filtered_measures[f"cphi_{suffix}_mu"] = (
                    mu_component if math.isfinite(mu_component) else None
                )
                temp_delta = float(wheel.temperature_delta)
                filtered_measures[f"cphi_{suffix}_temp_delta"] = (
                    temp_delta if math.isfinite(temp_delta) else None
                )
                gradient_rate = float(wheel.gradient_rate)
                filtered_measures[f"cphi_{suffix}_gradient_rate"] = (
                    gradient_rate if math.isfinite(gradient_rate) else None
                )
        filtered_measures["cphi"] = window_metrics.cphi.as_dict()
        ventilation_alert = window_metrics.brake_headroom.ventilation_alert
        if ventilation_alert:
            filtered_measures["brake_headroom_ventilation_alert"] = ventilation_alert
        for phase_label, value in (
            window_metrics.phase_delta_nfr_std or {}
        ).items():
            filtered_measures[f"delta_nfr_std_{phase_label}"] = float(value)
        for phase_label, value in (
            window_metrics.phase_nodal_delta_nfr_std or {}
        ).items():
            filtered_measures[f"nodal_delta_nfr_std_{phase_label}"] = float(value)
        filtered_measures["delta_nfr_entropy"] = float(
            window_metrics.delta_nfr_entropy
        )
        filtered_measures["node_entropy"] = float(window_metrics.node_entropy)
        for phase_label, value in (
            window_metrics.phase_delta_nfr_entropy or {}
        ).items():
            filtered_measures[f"delta_nfr_entropy_{phase_label}"] = float(value)
        for phase_label, value in (
            window_metrics.phase_node_entropy or {}
        ).items():
            filtered_measures[f"node_entropy_{phase_label}"] = float(value)
        for phase_label, value in (
            window_metrics.phase_brake_longitudinal_correlation or {}
        ).items():
            filtered_measures[f"brake_longitudinal_correlation_{phase_label}"] = float(value)
        for phase_label, value in (
            window_metrics.phase_throttle_longitudinal_correlation or {}
        ).items():
            filtered_measures[
                f"throttle_longitudinal_correlation_{phase_label}"
            ] = float(value)
        histogram = window_metrics.bumpstop_histogram
        for index, _ in enumerate(histogram.depth_bins):
            filtered_measures[f"bumpstop_front_density_bin_{index}"] = histogram.front_density[index]
            filtered_measures[f"bumpstop_rear_density_bin_{index}"] = histogram.rear_density[index]
            filtered_measures[f"bumpstop_front_energy_bin_{index}"] = histogram.front_energy[index]
            filtered_measures[f"bumpstop_rear_energy_bin_{index}"] = histogram.rear_energy[index]
        rec_trace: Tuple[Mapping[str, float | str | None], ...] = ()
        mutation_details: Mapping[str, object] | None = None
        if rec_state_root is not None:
            measures = {
                "thermal_load": avg_vertical_load,
                "style_index": avg_si,
                "phase": active_goal.phase,
                "grip_rel": grip_rel,
                "d_nfr_flat": window_metrics.d_nfr_flat,
                "si_variance": window_metrics.si_variance,
                "epi_derivative_abs": window_metrics.epi_derivative_abs,
                "timestamp": float(spec.get("end_timestamp", records[spec["end"]].timestamp)),
            }
            measures.update(brake_temperatures)
            measures.update(brake_temperature_std)
            # The recursivity operator returns a structured payload describing
            # the live session and the current microsector snapshot.
            rec_info: RecursivityOperatorResult = recursivity_operator(
                rec_state_root,
                session_components,
                str(index),
                measures,
                decay=recursion_decay,
            )
            rec_phase = rec_info["phase"] or active_goal.phase
            entropy_value = _estimate_entropy(
                records,
                start,
                end,
                bundles=recomputed_bundles,
                node_delta_cache=node_delta_cache,
                cache_offset=0,
            )
            triggers = {
                "microsector_id": str(index),
                "current_archetype": archetype,
                "candidate_archetype": archetype,
                "fallback_archetype": "recuperacion",
                "entropy": entropy_value,
                "style_index": rec_info["filtered"].get("style_index", avg_si),
                "style_reference": avg_si,
                "phase": rec_phase,
                "dynamic_conditions": brake_event or support_event,
            }
            mutation_info = mutation_operator(
                mutation_state if mutation_state is not None else {},
                triggers,
                entropy_threshold=thresholds.get("entropy_threshold", 0.65),
                entropy_increase=thresholds.get("entropy_increase", 0.08),
                style_threshold=thresholds.get("style_threshold", 0.12),
            )
            final_archetype = mutation_info["archetype"]
            if final_archetype != archetype:
                archetype = final_archetype
                (
                    goals,
                    dominant_nodes,
                    axis_targets,
                    axis_weights,
                ) = _build_goals(
                    archetype,
                    recomputed_bundles,
                    records,
                    phase_boundaries,
                    yaw_rates=yaw_rate_cache,
                    context_matrix=context_matrix,
                    sample_context=sample_context,
                    sample_multipliers=sample_multipliers,
                    node_delta_cache=node_delta_cache,
                    phase_gradients=phase_gradient_map,
                    phase_dominant_nodes=cached_dominant_nodes,
                    phase_nu_f_targets=cached_nu_f_targets,
                    phase_dominance_weights=cached_dominance_weights,
                    sample_rate=sample_rate,
                    steer_norm=steer_norm,
                    yaw_norm=yaw_norm,
                    lat_norm=lat_norm,
                )
                spec["goals"] = goals
                spec["dominant_nodes"] = dominant_nodes
                spec["phase_axis_targets"] = axis_targets
                spec["phase_axis_weights"] = axis_weights
                spec["goal_cache_valid"] = True
                spec["goal_archetype"] = archetype
                spec["goal_gradient_signature"] = gradient_signature
                active_goal = max(
                    goals,
                    key=lambda goal: abs(goal.target_delta_nfr) + goal.nu_f_target,
                )
                spec["active_phase"] = active_goal.phase
                recursivity_operator(
                    rec_state_root,
                    session_components,
                    str(index),
                    {
                        "thermal_load": measures["thermal_load"],
                        "style_index": measures["style_index"],
                        "grip_rel": measures["grip_rel"],
                        "phase": active_goal.phase,
                    },
                    decay=recursion_decay,
                )
            micro_state_entry: RecursivityMicroStateSnapshot = rec_info["state"]
            filtered_snapshot = {
                key: float(value)
                for key, value in micro_state_entry["filtered"].items()
                if isinstance(value, (int, float))
            }
            filtered_measures.update(filtered_snapshot)
            defaults = {
                "thermal_load": avg_vertical_load,
                "style_index": avg_si,
                "grip_rel": grip_rel,
                "d_nfr_couple": window_metrics.d_nfr_couple,
                "d_nfr_res": window_metrics.d_nfr_res,
                "d_nfr_flat": window_metrics.d_nfr_flat,
                "nu_f": window_metrics.nu_f,
                "nu_exc": window_metrics.nu_exc,
                "rho": window_metrics.rho,
                "udr": window_metrics.useful_dissonance_ratio,
                "phase_lag_window": window_metrics.phase_lag,
                "phase_alignment_window": window_metrics.phase_alignment,
                "coherence_index": window_metrics.coherence_index,
                "locking_window_score": window_metrics.locking_window_score.value,
                "locking_window_score_on": window_metrics.locking_window_score.on_throttle,
                "locking_window_score_off": window_metrics.locking_window_score.off_throttle,
                "locking_window_transitions": float(
                    window_metrics.locking_window_score.transition_samples
                ),
                "support_effective": window_metrics.support_effective,
                "load_support_ratio": window_metrics.load_support_ratio,
                "structural_expansion_longitudinal": window_metrics.structural_expansion_longitudinal,
                "structural_contraction_longitudinal": window_metrics.structural_contraction_longitudinal,
                "structural_expansion_lateral": window_metrics.structural_expansion_lateral,
                "structural_contraction_lateral": window_metrics.structural_contraction_lateral,
                "mu_usage_front_ratio": window_metrics.mu_usage_front_ratio,
                "mu_usage_rear_ratio": window_metrics.mu_usage_rear_ratio,
                "si_variance": window_metrics.si_variance,
                "epi_derivative_abs": window_metrics.epi_derivative_abs,
                "phase_synchrony_window": window_metrics.phase_synchrony_index,
            }
            defaults.update(brake_temperatures)
            defaults.update(brake_temperature_std)
            for key, value in defaults.items():
                filtered_measures.setdefault(key, value)
            rec_trace = tuple(
                {
                    trace_key: (
                        float(trace_value)
                        if isinstance(trace_value, (int, float))
                        else trace_value
                    )
                    for trace_key, trace_value in trace_entry.items()
                }
                for trace_entry in micro_state_entry["trace"]
            )
            mutation_details = {
                key: (
                    float(value)
                    if isinstance(value, (int, float))
                    else value
                )
                for key, value in (mutation_info or {}).items()
            }
        occupancy = _compute_window_occupancy(
            goals, local_phase_indices, record_window, window_yaw_rates
        )
        phase_lag_map = {
            goal.phase: float(goal.measured_phase_lag) for goal in goals
        }
        phase_alignment_map = {
            goal.phase: float(goal.measured_phase_alignment) for goal in goals
        }
        phase_synchrony_map = {
            goal.phase: float(goal.measured_phase_synchrony) for goal in goals
        }
        for legacy, phases in LEGACY_PHASE_MAP.items():
            lag_values = [phase_lag_map.get(candidate) for candidate in phases if candidate in phase_lag_map]
            align_values = [
                phase_alignment_map.get(candidate)
                for candidate in phases
                if candidate in phase_alignment_map
            ]
            synchrony_values = [
                phase_synchrony_map.get(candidate)
                for candidate in phases
                if candidate in phase_synchrony_map
            ]
            if lag_values:
                phase_lag_map[legacy] = float(
                    fmean(value for value in lag_values if value is not None)
                )
            if align_values:
                phase_alignment_map[legacy] = float(
                    fmean(value for value in align_values if value is not None)
                )
            if synchrony_values:
                phase_synchrony_map[legacy] = float(
                    fmean(value for value in synchrony_values if value is not None)
                )
        operator_events: Dict[str, Tuple[Mapping[str, object], ...]] = {}
        silence_window = max(6, min(len(record_window), 25))
        silence_struct_window = max(5, min(silence_window, 21))
        micro_duration = (
            float(record_window[-1].timestamp - record_window[0].timestamp)
            if len(record_window) >= 2
            else 0.0
        )
        detector_specs = (
            ("AL", detect_al, {}),
            ("EN", detect_en, {}),
            ("IL", detect_il, {}),
            ("NUL", detect_nul, {}),
            ("OZ", detect_oz, {}),
            ("RA", detect_ra, {}),
            ("REMESH", detect_remesh, {}),
            ("THOL", detect_thol, {}),
            ("UM", detect_um, {}),
            ("VAL", detect_val, {}),
            ("ZHIR", detect_zhir, {}),
            (
                "SILENCE",
                detect_silence,
                {
                    "window": silence_window,
                    "structural_window": silence_struct_window,
                    "min_duration": max(0.6, micro_duration * 0.25),
                },
            ),
        )
        for name, detector, detector_kwargs in detector_specs:
            events: List[Mapping[str, object]] = []
            for event in detector(record_window, **detector_kwargs):
                payload: Dict[str, object] = {**event, "microsector": spec["index"]}
                local_start = int(event.get("start_index", 0))
                local_end = int(event.get("end_index", local_start))
                global_start = max(start, min(end, start + max(0, local_start)))
                global_end = max(global_start, min(end, start + max(0, local_end)))
                payload.setdefault("global_start_index", global_start)
                payload.setdefault("global_end_index", global_end)
                payload.setdefault("si_variance", float(window_metrics.si_variance))
                payload.setdefault(
                    "epi_derivative_abs", float(window_metrics.epi_derivative_abs)
                )
                if name in {"OZ", "IL"}:
                    delta_values: List[float] = []
                    for idx in range(global_start, global_end + 1):
                        if 0 <= idx < len(recomputed_bundles):
                            delta_values.append(float(recomputed_bundles[idx].delta_nfr))
                    if delta_values:
                        avg_delta = fmean(delta_values)
                        peak_delta = max(abs(value) for value in delta_values)
                    else:
                        avg_delta = 0.0
                        peak_delta = 0.0
                    ratio = peak_delta / delta_threshold if delta_threshold > 1e-9 else 0.0
                    surface_payload = {
                        "label": surface_label,
                        "factor": surface_factor,
                        "lower": surface_band[0],
                        "upper": surface_band[1],
                    }
                    payload.update(
                        {
                            "global_start_index": global_start,
                            "global_end_index": global_end,
                            "delta_nfr_avg": avg_delta,
                            "delta_nfr_peak": peak_delta,
                            "delta_nfr_threshold": delta_threshold,
                            "delta_nfr_ratio": ratio,
                            "surface": surface_payload,
                            "surface_label": surface_label,
                            "surface_factor": surface_factor,
                        }
                    )
                elif name == "SILENCE":
                    payload.setdefault(
                        "structural_start",
                        float(
                            getattr(recomputed_bundles[global_start], "structural_timestamp", 0.0)
                            if 0 <= global_start < len(recomputed_bundles)
                            else event.get("structural_start", 0.0)
                        ),
                    )
                    payload.setdefault(
                        "structural_end",
                        float(
                            getattr(recomputed_bundles[global_end], "structural_timestamp", 0.0)
                            if 0 <= global_end < len(recomputed_bundles)
                            else event.get("structural_end", 0.0)
                        ),
                    )
                phase_votes: Dict[str, int] = {}
                for idx in range(global_start, global_end + 1):
                    if 0 <= idx < len(assigned_phases):
                        phase_candidate = assigned_phases[idx]
                    else:
                        phase_candidate = None
                    if not phase_candidate:
                        continue
                    phase_key = phase_family(str(phase_candidate))
                    phase_votes[phase_key] = phase_votes.get(phase_key, 0) + 1
                if phase_votes:
                    dominant_phase = max(phase_votes.items(), key=lambda item: item[1])[0]
                else:
                    dominant_phase = phase_family(active_goal.phase)
                payload.setdefault("phase", dominant_phase)
                payload.setdefault("operator_id", name)
                operator_label = payload.get("name")
                if not isinstance(operator_label, str) or not operator_label:
                    operator_label = canonical_operator_label(name)
                payload.setdefault("operator_label", operator_label)
                payload.setdefault(
                    "operator",
                    f"{payload['operator_id']} · {payload['operator_label']}",
                )
                payload.setdefault("start_time", float(event.get("start_time", records[global_start].timestamp)))
                payload.setdefault("end_time", float(event.get("end_time", records[global_end].timestamp)))
                payload.setdefault(
                    "severity",
                    float(event.get("severity", event.get("peak_value", 0.0))),
                )
                events.append(payload)
            if events:
                operator_events[name] = tuple(events)
        serialised_phase_samples = {
            str(phase): tuple(int(index) for index in indices)
            for phase, indices in phase_samples.items()
        }
        serialised_phase_samples = replicate_phase_aliases(
            serialised_phase_samples,
            combine=_merge_phase_indices,
        )
        serialised_axis_targets = replicate_phase_aliases(
            {
                str(phase): {
                    str(axis): float(value)
                    for axis, value in (payload or {}).items()
                }
                for phase, payload in axis_targets.items()
            }
        )
        serialised_axis_weights = replicate_phase_aliases(
            {
                str(phase): {
                    str(axis): float(value)
                    for axis, value in (payload or {}).items()
                }
                for phase, payload in axis_weights.items()
            }
        )
        phase_delta_payload = replicate_phase_aliases(
            {
                str(label): float(value)
                for label, value in (
                    (window_metrics.phase_delta_nfr_std or {}).items()
                )
            }
        )
        phase_nodal_payload = replicate_phase_aliases(
            {
                str(label): float(value)
                for label, value in (
                    (window_metrics.phase_nodal_delta_nfr_std or {}).items()
                )
            }
        )
        phase_entropy_payload = replicate_phase_aliases(
            {
                str(label): float(value)
                for label, value in (
                    (window_metrics.phase_delta_nfr_entropy or {}).items()
                )
            }
        )
        phase_node_entropy_payload = replicate_phase_aliases(
            {
                str(label): float(value)
                for label, value in (
                    (window_metrics.phase_node_entropy or {}).items()
                )
            }
        )

        microsectors.append(
            Microsector(
                index=spec["index"],
                start_time=records[start].timestamp,
                end_time=records[end].timestamp,
                curvature=curvature,
                brake_event=brake_event,
                support_event=support_event,
                delta_nfr_signature=delta_signature,
                goals=goals,
                phase_boundaries=phase_boundaries,
                phase_samples=serialised_phase_samples,
                active_phase=active_goal.phase,
                dominant_nodes=dict(dominant_nodes),
                phase_weights=phase_weights,
                grip_rel=float(filtered_measures.get("grip_rel", grip_rel)),
                phase_lag=dict(phase_lag_map),
                phase_alignment=dict(phase_alignment_map),
                phase_synchrony=dict(phase_synchrony_map),
                phase_motor_latency=dict(phase_motor_latency_payload),
                motor_latency_ms=float(window_metrics.motor_latency_ms),
                delta_nfr_std=float(window_metrics.delta_nfr_std),
                nodal_delta_nfr_std=float(window_metrics.nodal_delta_nfr_std),
                phase_delta_nfr_std=phase_delta_payload,
                phase_nodal_delta_nfr_std=phase_nodal_payload,
                delta_nfr_entropy=float(window_metrics.delta_nfr_entropy),
                node_entropy=float(window_metrics.node_entropy),
                phase_delta_nfr_entropy=phase_entropy_payload,
                phase_node_entropy=phase_node_entropy_payload,
                filtered_measures=dict(filtered_measures),
                recursivity_trace=rec_trace,
                last_mutation=dict(mutation_details) if mutation_details is not None else None,
                window_occupancy=occupancy,
                phase_axis_targets=serialised_axis_targets,
                phase_axis_weights=serialised_axis_weights,
                context_factors=(
                    spec.get("context_factors").as_mapping()
                    if spec.get("context_factors")
                    else {}
                ),
                sample_context_factors=_serialise_sample_context(
                    spec.get("sample_context"),
                    start,
                    end,
                    sample_context,
                ),
                operator_events=operator_events,
            )
        )

    if isinstance(bundles, list):
        bundles[:] = recomputed_bundles

    return microsectors


def _recompute_bundles(
    records: Sequence[SupportsTelemetrySample],
    bundles: Sequence[SupportsEPIBundle],
    baseline: SupportsTelemetrySample,
    phase_assignments: Sequence[PhaseLiteral],
    weight_lookup: Sequence[Mapping[str, Mapping[str, float] | float]],
    goal_nu_f_lookup: Mapping[int, Mapping[str, float] | float] | None = None,
    *,
    start_index: int = 0,
    analyzer_states: Sequence[_AnalyzerState | None] | None = None,
) -> _BundleRecomputeResult:
    def _ensure_mutable_bundles(
        source: Sequence[SupportsEPIBundle],
    ) -> MutableSequence[SupportsEPIBundle]:
        if isinstance(source, MutableSequence):
            return cast(MutableSequence[SupportsEPIBundle], source)
        return cast(MutableSequence[SupportsEPIBundle], list(source))

    if not records:
        cached_states = list(analyzer_states) if analyzer_states is not None else []
        return _BundleRecomputeResult(_ensure_mutable_bundles(bundles), cached_states)

    total_samples = len(records)
    default_phase = PHASE_SEQUENCE[0] if PHASE_SEQUENCE else "entry1"
    assignment_count = len(phase_assignments)
    weight_count = len(weight_lookup)

    def _phase_for(index: int) -> PhaseLiteral:
        if 0 <= index < assignment_count:
            candidate = phase_assignments[index]
            if isinstance(candidate, str) and candidate:
                return cast(PhaseLiteral, candidate)
        return default_phase

    def _weights_for(index: int) -> Mapping[str, Mapping[str, float] | float]:
        if 0 <= index < weight_count:
            candidate = weight_lookup[index]
            if candidate is not None:
                return candidate
        return DEFAULT_PHASE_WEIGHTS

    if start_index <= 0:
        recompute_start = 0
    elif start_index >= total_samples:
        cached_states = list(analyzer_states) if analyzer_states is not None else [None] * total_samples
        if len(cached_states) < total_samples:
            cached_states.extend([None] * (total_samples - len(cached_states)))
        elif len(cached_states) > total_samples:
            cached_states = cached_states[:total_samples]
        mutable_bundles = _ensure_mutable_bundles(bundles)
        if len(mutable_bundles) > total_samples:
            del mutable_bundles[total_samples:]
        return _BundleRecomputeResult(mutable_bundles, cached_states)
    else:
        recompute_start = start_index

    analyzer = NaturalFrequencyAnalyzer()
    result = _ensure_mutable_bundles(bundles)

    if analyzer_states is None:
        state_cache: List[_AnalyzerState | None] = [None for _ in range(total_samples)]
    else:
        state_cache = list(analyzer_states)
        if len(state_cache) < total_samples:
            state_cache.extend([None for _ in range(total_samples - len(state_cache))])
        elif len(state_cache) > total_samples:
            del state_cache[total_samples:]

    restore_index = -1
    if recompute_start > 0 and state_cache:
        search_limit = min(recompute_start - 1, len(state_cache) - 1)
        for candidate in range(search_limit, -1, -1):
            cached_state = state_cache[candidate]
            if cached_state is not None:
                restored = _restore_analyzer_state(analyzer, cached_state, records)
                if restored is not None:
                    restore_index = restored
                    break

    prime_start = max(restore_index + 1, 0)
    for idx in range(prime_start, recompute_start):
        record = records[idx]
        phase = _phase_for(idx)
        phase_weights = _weights_for(idx)
        resolve_nu_f_by_node(
            record,
            phase=phase,
            phase_weights=phase_weights,
            analyzer=analyzer,
        )
        if idx < len(state_cache):
            state_cache[idx] = _capture_analyzer_state(analyzer, idx)

    prev_integrated: float | None = None
    if recompute_start > 0 and recompute_start - 1 < len(result):
        prev_integrated = result[recompute_start - 1].integrated_epi

    for idx in range(recompute_start, total_samples):
        record = records[idx]
        structural_ts = getattr(record, "structural_timestamp", None)
        if idx == 0:
            dt = 0.0
            prev_timestamp = record.timestamp
            prev_structural = structural_ts
        else:
            prev_record = records[idx - 1]
            prev_timestamp = prev_record.timestamp
            prev_structural = getattr(prev_record, "structural_timestamp", None)
            if (
                structural_ts is not None
                and prev_structural is not None
                and math.isfinite(structural_ts)
                and math.isfinite(prev_structural)
            ):
                dt = max(0.0, structural_ts - prev_structural)
            else:
                dt = max(0.0, record.timestamp - prev_timestamp)
        phase = _phase_for(idx)
        phase_weights = _weights_for(idx)
        target_nu_f = goal_nu_f_lookup.get(idx) if goal_nu_f_lookup else None
        nu_snapshot = resolve_nu_f_by_node(
            record,
            phase=phase,
            phase_weights=phase_weights,
            analyzer=analyzer,
        )
        epi_value = bundles[idx].epi if idx < len(bundles) else 0.0
        recomputed_bundle = DeltaCalculator.compute_bundle(
            record,
            baseline,
            epi_value,
            prev_integrated_epi=prev_integrated,
            dt=dt,
            nu_f_by_node=nu_snapshot.by_node,
            nu_f_snapshot=nu_snapshot,
            phase=phase,
            phase_weights=phase_weights,
            phase_target_nu_f=target_nu_f,
        )
        if idx < len(result):
            result[idx] = recomputed_bundle
        else:
            result.append(recomputed_bundle)
        prev_integrated = recomputed_bundle.integrated_epi
        if idx < len(state_cache):
            state_cache[idx] = _capture_analyzer_state(analyzer, idx)
    return _BundleRecomputeResult(result, state_cache)


def _estimate_entropy(
    records: Sequence[SupportsTelemetrySample],
    start: int,
    end: int,
    *,
    bundles: Sequence[SupportsEPIBundle] | None = None,
    node_delta_cache: (
        Sequence[Mapping[str, float]]
        | MutableSequence[Mapping[str, float]]
        | None
    ) = None,
    cache_offset: int = 0,
) -> float:
    node_weights: Dict[str, float] = defaultdict(float)
    for index in range(start, end + 1):
        distribution: Mapping[str, float] | None = None
        if node_delta_cache is not None:
            cache_index = index - cache_offset
            if 0 <= cache_index < len(node_delta_cache):
                distribution = node_delta_cache[cache_index]
        if distribution is None and bundles is not None and 0 <= index < len(bundles):
            distribution = _bundle_node_delta(bundles[index])
        if distribution is None:
            distribution = delta_nfr_by_node(records[index])
        for node, delta in distribution.items():
            weight = abs(delta)
            if weight > 0.0:
                node_weights[node] += weight
    total = sum(node_weights.values())
    if total <= 0.0:
        return 0.0
    probabilities = [value / total for value in node_weights.values() if value > 0.0]
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(prob * math.log(prob) for prob in probabilities)
    max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    if max_entropy <= 0.0:
        return 0.0
    return min(1.0, entropy / max_entropy)


def _identify_corner_segments(records: Sequence[SupportsTelemetrySample]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start_index: int | None = None
    for idx, record in enumerate(records):
        if abs(record.lateral_accel) >= CURVATURE_THRESHOLD:
            if start_index is None:
                start_index = idx
        else:
            if start_index is not None and idx - start_index >= MIN_SEGMENT_LENGTH:
                segments.append((start_index, idx - 1))
            start_index = None
    if start_index is not None and len(records) - start_index >= MIN_SEGMENT_LENGTH:
        segments.append((start_index, len(records) - 1))
    return segments


def _compute_phase_boundaries(
    records: Sequence[SupportsTelemetrySample], start: int, end: int
) -> Dict[PhaseLiteral, Tuple[int, int]]:
    idx_range = range(start, end + 1)
    entry_idx = min(idx_range, key=lambda i: records[i].longitudinal_accel)
    apex_idx = max(idx_range, key=lambda i: abs(records[i].lateral_accel))
    exit_idx = max(idx_range, key=lambda i: records[i].longitudinal_accel)

    entry_idx = max(start, min(entry_idx, end))
    apex_idx = max(start, min(apex_idx, end))
    exit_idx = max(apex_idx, min(exit_idx, end))

    total_stop = end + 1
    targets: Dict[str, int] = {
        "entry1": min(entry_idx + 1, total_stop),
        "entry2": min(max(entry_idx + 1, apex_idx), total_stop),
        "apex3a": min(apex_idx + 1, total_stop),
        "apex3b": min(max(apex_idx + 1, exit_idx), total_stop),
        "exit4": total_stop,
    }

    spans: Dict[str, Tuple[int, int]] = {}
    cursor = start
    total_phases = len(PHASE_SEQUENCE)
    for index, phase in enumerate(PHASE_SEQUENCE):
        remaining = total_phases - index
        available = total_stop - cursor
        if remaining == 1:
            stop = total_stop
        elif available <= remaining - 1:
            stop = cursor
        else:
            max_stop = total_stop - (remaining - 1)
            target = max(cursor + 1, targets.get(phase, max_stop))
            stop = min(max_stop, max(cursor + 1, target))
        spans[phase] = (cursor, stop)
        cursor = stop

    ordered: Dict[PhaseLiteral, Tuple[int, int]] = {}
    previous_stop = start
    for phase in PHASE_SEQUENCE:
        phase_start, phase_stop = spans[phase]
        phase_start = max(previous_stop, min(phase_start, total_stop))
        phase_stop = max(phase_start, min(phase_stop, total_stop))
        ordered[phase] = (phase_start, phase_stop)
        previous_stop = phase_stop
    last_phase = PHASE_SEQUENCE[-1]
    start_last, _ = ordered[last_phase]
    ordered[last_phase] = (start_last, total_stop)
    for legacy, phases in LEGACY_PHASE_MAP.items():
        relevant = [ordered[phase] for phase in phases if phase in ordered]
        if not relevant:
            continue
        ordered[legacy] = (relevant[0][0], relevant[-1][1])
    return ordered


def _detect_support_event(records: Sequence[SupportsTelemetrySample]) -> bool:
    loads = [record.vertical_load for record in records]
    return max(loads) - min(loads) >= SUPPORT_THRESHOLD


def _direction_changes(records: Sequence[SupportsTelemetrySample]) -> int:
    last_sign = 0
    changes = 0
    for record in records:
        value = float(record.lateral_accel)
        if abs(value) < 0.25:
            continue
        sign = 1 if value > 0 else -1
        if last_sign and sign != last_sign:
            changes += 1
        last_sign = sign
    return changes


def _classify_archetype(
    curvature: float,
    duration: float,
    speed_drop: float,
    direction_changes: int,
) -> str:
    if direction_changes >= 1:
        return ARCHETYPE_CHICANE
    if curvature >= 2.2 and duration >= 2.4 and speed_drop >= 10.0:
        return ARCHETYPE_HAIRPIN
    if curvature <= 1.6 and duration <= 2.4 and speed_drop <= 7.5:
        return ARCHETYPE_FAST
    return ARCHETYPE_MEDIUM


def _resolve_yaw_rate(
    records: Sequence[SupportsTelemetrySample], index: int
) -> float:
    if not (0 <= index < len(records)):
        return 0.0
    record = records[index]
    try:
        yaw_rate = float(getattr(record, "yaw_rate"))
    except (AttributeError, TypeError, ValueError):
        yaw_rate = math.nan
    if math.isfinite(yaw_rate):
        return yaw_rate
    return _compute_yaw_rate(records, index, record=record, yaw_rate=yaw_rate)


def _compute_yaw_rate(
    records: Sequence[SupportsTelemetrySample],
    index: int,
    *,
    record: SupportsTelemetrySample | None = None,
    yaw_rate: float | None = None,
) -> float:
    if yaw_rate is not None and math.isfinite(yaw_rate):
        return float(yaw_rate)
    if record is not None:
        candidate = getattr(record, "yaw_rate", math.nan)
        if math.isfinite(candidate):
            return float(candidate)
    if index <= 0 or index >= len(records):
        return 0.0
    if record is None:
        record = records[index]
    previous = records[index - 1]
    dt = record.timestamp - previous.timestamp
    if dt <= 1e-9:
        return 0.0
    delta = record.yaw - previous.yaw
    wrapped = (delta + math.pi) % (2.0 * math.pi) - math.pi
    return wrapped / dt


def _cached_yaw_rate(cache: Sequence[float], index: int) -> float:
    if 0 <= index < len(cache):
        try:
            return float(cache[index])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return fmean(values) if values else default


def _window(values: Sequence[float], scale: float, minimum: float = 0.01) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    centre = _safe_mean(values)
    if len(values) > 1:
        deviation = pstdev(values)
    else:
        deviation = 0.0
    if not math.isfinite(deviation) or deviation < minimum:
        deviation = max(minimum, abs(centre) * 0.05)
    span = max(minimum, deviation * max(1.0, scale))
    lower = centre - span
    upper = centre + span
    if lower > upper:
        lower, upper = upper, lower
    return (lower, upper)


def _phase_nu_f_targets(
    bundles: Sequence[SupportsEPIBundle],
    records: Sequence[SupportsTelemetrySample],
    boundaries: Mapping[PhaseLiteral, Tuple[int, int]],
    phase_samples: Mapping[PhaseLiteral, Sequence[int]] | None,
    yaw_rates: Sequence[float],
    *,
    context_matrix: ContextMatrix | None = None,
    sample_context: Sequence[ContextFactors] | None = None,
    node_delta_cache: (
        Sequence[Mapping[str, float]]
        | MutableSequence[Mapping[str, float]]
        | None
    ) = None,
    cache_window: Tuple[int, int] | None = None,
) -> Tuple[
    Dict[PhaseLiteral, Tuple[str, ...]],
    Dict[PhaseLiteral, float],
    Dict[PhaseLiteral, float],
    Dict[int, float],
]:
    """Resolve dominant nodes and ν_f targets for each phase."""

    # The helper focuses on the dominance/ν_f calculations so consumers can
    # reuse the resulting maps without having to build full goal payloads.
    phase_sample_map: Dict[PhaseLiteral, Sequence[int]] = {}
    if phase_samples is None:
        phase_samples = {}
    bundle_count = len(bundles)
    window_start = 0
    window_stop = len(records)
    if cache_window is not None:
        start_candidate, stop_candidate = cache_window
        window_start = max(0, min(start_candidate, stop_candidate))
        window_stop = max(window_start, max(start_candidate, stop_candidate))

    cache_offset = 0
    if node_delta_cache is None:
        effective_stop = min(window_stop, len(bundles))
        effective_start = min(max(0, window_start), effective_stop)
        node_delta_cache = [
            _bundle_node_delta(bundles[idx]) for idx in range(effective_start, effective_stop)
        ]
        cache_offset = effective_start
    else:
        expected_length = max(0, window_stop - window_start)
        if cache_window is not None and len(node_delta_cache) == expected_length:
            cache_offset = window_start
    for phase in PHASE_SEQUENCE:
        start, stop = boundaries.get(phase, (0, 0))
        phase_range = range(start, min(stop, bundle_count))
        provided = phase_samples.get(phase)
        if provided is not None:
            phase_sample_map[phase] = tuple(int(idx) for idx in provided)
        else:
            phase_sample_map[phase] = phase_range

    dominant_nodes: Dict[PhaseLiteral, Tuple[str, ...]] = {}
    nu_f_targets: Dict[PhaseLiteral, float] = {}
    dominance_weights: Dict[PhaseLiteral, float] = {}
    sample_lookup: Dict[int, float] = {}

    for phase in PHASE_SEQUENCE:
        start, stop = boundaries.get(phase, (0, 0))
        upper_bound = min(stop, bundle_count)

        node_metrics: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"abs_delta": 0.0, "nu_f_weight": 0.0}
        )
        for idx in range(start, upper_bound):
            bundle = bundles[idx]
            cache_index = idx - cache_offset
            if 0 <= cache_index < len(node_delta_cache):
                node_deltas = node_delta_cache[cache_index]
            elif 0 <= idx < len(bundles):
                node_deltas = _bundle_node_delta(bundle)
            elif 0 <= idx < len(records):
                node_deltas = delta_nfr_by_node(records[idx])
            else:
                node_deltas = {}
            for node, delta in node_deltas.items():
                weight = abs(delta)
                node_metrics[node]["abs_delta"] += weight
                nu_f = getattr(bundle, node).nu_f if hasattr(bundle, node) else 0.0
                node_metrics[node]["nu_f_weight"] += weight * nu_f

        significant_nodes = (
            (
                node,
                metrics,
            )
            for node, metrics in node_metrics.items()
            if metrics["abs_delta"] > 0.0
        )
        top_nodes = nlargest(3, significant_nodes, key=lambda item: item[1]["abs_delta"])
        phase_nodes = tuple(node for node, _ in top_nodes)
        if not phase_nodes and node_metrics:
            phase_nodes = tuple(list(node_metrics.keys())[:3])
        dominant_nodes[phase] = phase_nodes

        total_weight = sum(node_metrics[node]["abs_delta"] for node in phase_nodes)
        if total_weight > 0.0:
            weighted_nu_f = sum(
                node_metrics[node]["nu_f_weight"] for node in phase_nodes
            )
            nu_f_target = weighted_nu_f / total_weight
        else:
            nu_f_target = 0.0

        nu_f_targets[phase] = float(nu_f_target)
        dominance_weights[phase] = float(total_weight)

        for sample_index in phase_sample_map.get(phase, ()):  # pragma: no branch
            if isinstance(sample_index, int) and 0 <= sample_index < len(bundles):
                sample_lookup[sample_index] = float(nu_f_target)

    return dominant_nodes, nu_f_targets, dominance_weights, sample_lookup


def _store_phase_goal_cache(
    spec: MutableMapping[str, object],
    dominant_nodes: Mapping[PhaseLiteral, Tuple[str, ...]],
    nu_f_targets: Mapping[PhaseLiteral, float],
    dominance_weights: Mapping[PhaseLiteral, float],
    sample_lookup: Mapping[int, float],
) -> None:
    spec["goal_phase_cache"] = {
        "dominant_nodes": {
            str(phase): tuple(nodes)
            for phase, nodes in dominant_nodes.items()
        },
        "nu_f_targets": {
            str(phase): float(value)
            for phase, value in nu_f_targets.items()
        },
        "dominance_weights": {
            str(phase): float(value)
            for phase, value in dominance_weights.items()
        },
        "sample_lookup": {
            int(idx): float(value)
            for idx, value in sample_lookup.items()
        },
    }


def _extract_phase_goal_cache(
    spec: Mapping[str, object]
) -> Tuple[
    Mapping[PhaseLiteral, Tuple[str, ...]] | None,
    Mapping[PhaseLiteral, float] | None,
    Mapping[PhaseLiteral, float] | None,
    Mapping[int, float] | None,
]:
    payload = spec.get("goal_phase_cache")
    if isinstance(payload, Mapping):
        dominant_nodes = cast(
            Mapping[PhaseLiteral, Tuple[str, ...]] | None,
            payload.get("dominant_nodes"),
        )
        nu_f_targets = cast(
            Mapping[PhaseLiteral, float] | None,
            payload.get("nu_f_targets"),
        )
        dominance_weights = cast(
            Mapping[PhaseLiteral, float] | None,
            payload.get("dominance_weights"),
        )
        sample_lookup = cast(
            Mapping[int, float] | None,
            payload.get("sample_lookup"),
        )
        return dominant_nodes, nu_f_targets, dominance_weights, sample_lookup
    return None, None, None, None


def _build_goals(
    archetype: str,
    bundles: Sequence[SupportsEPIBundle],
    records: Sequence[SupportsTelemetrySample],
    boundaries: Mapping[PhaseLiteral, Tuple[int, int]],
    yaw_rates: Sequence[float],
    *,
    context_matrix: ContextMatrix | None = None,
    sample_context: Sequence[ContextFactors] | None = None,
    sample_multipliers: Sequence[float] | None = None,
    node_delta_cache: (
        Sequence[Mapping[str, float]]
        | MutableSequence[Mapping[str, float]]
        | None
    ) = None,
    phase_gradients: Mapping[PhaseLiteral, float] | None = None,
    phase_dominant_nodes: Mapping[PhaseLiteral, Tuple[str, ...]] | None = None,
    phase_nu_f_targets: Mapping[PhaseLiteral, float] | None = None,
    phase_dominance_weights: Mapping[PhaseLiteral, float] | None = None,
    sample_rate: float | None = None,
    steer_norm: Sequence[float] | None = None,
    yaw_norm: Sequence[float] | None = None,
    lat_norm: Sequence[float] | None = None,
) -> Tuple[
    Tuple[Goal, ...],
    Mapping[PhaseLiteral, Tuple[str, ...]],
    Mapping[PhaseLiteral, Mapping[str, float]],
    Mapping[PhaseLiteral, Mapping[str, float]],
]:
    descriptions = _goal_descriptions(archetype)
    alignment_targets = _phase_alignment_targets(archetype)
    archetype_targets = archetype_phase_targets(archetype)
    default_targets = archetype_phase_targets(ARCHETYPE_MEDIUM)
    goals: List[Goal] = []
    axis_targets: Dict[PhaseLiteral, Dict[str, float]] = {}
    axis_weights: Dict[PhaseLiteral, Dict[str, float]] = {}
    context_matrix = context_matrix or load_context_matrix()

    if (
        phase_dominant_nodes is None
        or phase_nu_f_targets is None
        or phase_dominance_weights is None
    ):
        (
            base_dominant_nodes,
            resolved_nu_f_targets,
            resolved_dominance_weights,
            _,
        ) = _phase_nu_f_targets(
            bundles,
            records,
            boundaries,
            None,
            yaw_rates,
            context_matrix=context_matrix,
            sample_context=sample_context,
            node_delta_cache=node_delta_cache,
        )
    else:
        base_dominant_nodes = dict(phase_dominant_nodes)
        resolved_nu_f_targets = dict(phase_nu_f_targets)
        resolved_dominance_weights = dict(phase_dominance_weights)
    phase_nu_f_targets = resolved_nu_f_targets
    dominance_weights = resolved_dominance_weights
    dominant_nodes: Dict[PhaseLiteral, Tuple[str, ...]] = dict(base_dominant_nodes)

    for phase in PHASE_SEQUENCE:
        start, stop = boundaries[phase]
        stop = max(start, min(stop, len(bundles), len(records)))
        indices = range(start, stop)
        segment = list(bundles[start:stop])
        phase_records = list(records[start:stop])
        yaw_slice = list(yaw_rates[start:stop]) if yaw_rates else []
        if len(yaw_slice) < len(segment):
            yaw_slice.extend(
                _cached_yaw_rate(yaw_rates, idx)
                for idx in range(start + len(yaw_slice), start + len(segment))
            )
        phase_yaw_rates = yaw_slice
        if segment:
            multipliers: List[float] = []

            def _clamp(value: float) -> float:
                return max(
                    context_matrix.min_multiplier,
                    min(context_matrix.max_multiplier, value),
                )

            use_direct_multipliers = False
            if sample_multipliers is not None and indices:
                if all(0 <= idx < len(sample_multipliers) for idx in indices):
                    direct_values: List[float] = []
                    for idx in indices:
                        try:
                            numeric = float(sample_multipliers[idx])
                        except (TypeError, ValueError):
                            break
                        direct_values.append(_clamp(numeric))
                    else:
                        multipliers = direct_values
                        use_direct_multipliers = True

            if not use_direct_multipliers:
                if sample_context:
                    resolved_context: List[ContextFactors] = []
                    if indices:
                        first_idx = indices[0]
                        last_idx = indices[-1]
                        all_cached = 0 <= first_idx and last_idx < len(sample_context)
                    else:
                        all_cached = True
                    if all_cached:
                        resolved_context = [sample_context[idx] for idx in indices]
                    else:
                        fallback_context = resolve_series_context(
                            segment, matrix=context_matrix
                        )
                        for local_index, idx in enumerate(indices):
                            if 0 <= idx < len(sample_context):
                                resolved_context.append(sample_context[idx])
                            else:
                                resolved_context.append(fallback_context[local_index])
                else:
                    resolved_context = resolve_series_context(
                        segment, matrix=context_matrix
                    )
                multipliers = [
                    _clamp(ctx.multiplier)
                    for ctx in resolved_context
                ]
            adjusted_delta = [
                bundle.delta_nfr * multipliers[idx]
                for idx, bundle in enumerate(segment)
            ]
            avg_delta = fmean(adjusted_delta)
            avg_si = fmean(bundle.sense_index for bundle in segment)
            avg_long = fmean(
                bundle.delta_nfr_proj_longitudinal * multipliers[idx]
                for idx, bundle in enumerate(segment)
            )
            avg_lat = fmean(
                bundle.delta_nfr_proj_lateral * multipliers[idx]
                for idx, bundle in enumerate(segment)
            )
            abs_long = fmean(
                abs(bundle.delta_nfr_proj_longitudinal) * multipliers[idx]
                for idx, bundle in enumerate(segment)
            )
            abs_lat = fmean(
                abs(bundle.delta_nfr_proj_lateral) * multipliers[idx]
                for idx, bundle in enumerate(segment)
            )
        else:
            avg_delta = 0.0
            avg_si = 1.0
            avg_long = 0.0
            avg_lat = 0.0
            abs_long = 0.0
            abs_lat = 0.0

        def _slice_series(series: Sequence[float] | None) -> List[float] | None:
            if series is None:
                return None
            window: List[float] = []
            limit = len(series)
            for idx in indices:
                if 0 <= idx < limit:
                    try:
                        window.append(float(series[idx]))
                    except (TypeError, ValueError):
                        window.append(0.0)
                else:
                    window.append(0.0)
            return window

        steer_slice = _slice_series(steer_norm)
        yaw_slice = _slice_series(yaw_norm)
        lat_slice = _slice_series(lat_norm)

        _, measured_lag, measured_alignment = phase_alignment(
            phase_records,
            sample_rate=sample_rate,
            steer_norm=steer_slice,
            yaw_norm=yaw_slice,
            lat_norm=lat_slice,
        )
        family = phase_family(phase)
        phase_target = archetype_targets.get(family)
        if phase_target is None:
            phase_target = default_targets.get(family)
        target_lag, target_alignment = alignment_targets.get(
            phase, (phase_target.lag if phase_target else 0.0, phase_target.si_phi if phase_target else 0.9)
        )

        phase_nodes = dominant_nodes.get(phase, ())
        total_weight = dominance_weights.get(phase, 0.0)
        nu_f_target = phase_nu_f_targets.get(phase, 0.0)

        nu_exc_target = estimate_excitation_frequency(
            phase_records, sample_rate=sample_rate
        )
        rho_target = nu_exc_target / nu_f_target if nu_f_target > 1e-9 else 0.0

        sample_count = max(1, len(indices))
        dominant_intensity = total_weight / sample_count
        influence_factor = 1.0 + min(2.0, nu_f_target) + min(1.5, dominant_intensity / 5.0)

        slip_values = [record.slip_ratio for record in phase_records]
        lat_values = [record.lateral_accel for record in phase_records]
        long_values = [record.longitudinal_accel for record in phase_records]
        lat_scale = influence_factor * (1.0 + min(1.5, abs(_safe_mean(lat_values)) / 5.0))
        long_scale = influence_factor * (1.0 + min(1.5, abs(_safe_mean(long_values)) / 5.0))
        yaw_scale = influence_factor * (
            1.0 + min(1.5, abs(_safe_mean(phase_yaw_rates)) / 2.0)
        )

        slip_lat_window = _window(slip_values, lat_scale)
        slip_long_window = _window(slip_values, long_scale)
        yaw_rate_window = _window(phase_yaw_rates, yaw_scale, minimum=0.005)

        total_axis = abs_long + abs_lat
        if total_axis > 1e-9:
            weight_map = {
                "longitudinal": abs_long / total_axis,
                "lateral": abs_lat / total_axis,
            }
        else:
            weight_map = {"longitudinal": 0.5, "lateral": 0.5}
        axis_targets[phase] = {
            "longitudinal": float(avg_long),
            "lateral": float(avg_lat),
        }
        axis_weights[phase] = {
            "longitudinal": float(weight_map["longitudinal"]),
            "lateral": float(weight_map["lateral"]),
        }

        detune_weights = (
            dict(phase_target.detune_weights) if phase_target is not None else weight_map
        )

        gradient_value = 0.0
        if phase_gradients and phase in phase_gradients:
            try:
                gradient_value = float(phase_gradients[phase])
            except (TypeError, ValueError):
                gradient_value = 0.0

        goals.append(
            Goal(
                phase=phase,
                archetype=archetype,
                description=descriptions[phase],
                target_delta_nfr=avg_delta,
                target_sense_index=avg_si,
                nu_f_target=nu_f_target,
                nu_exc_target=nu_exc_target,
                rho_target=rho_target,
                target_phase_lag=target_lag,
                target_phase_alignment=target_alignment,
                measured_phase_lag=measured_lag,
                measured_phase_alignment=measured_alignment,
                slip_lat_window=slip_lat_window,
                slip_long_window=slip_long_window,
                yaw_rate_window=yaw_rate_window,
                dominant_nodes=phase_nodes,
                target_delta_nfr_long=avg_long,
                target_delta_nfr_lat=avg_lat,
                delta_axis_weights=weight_map,
                archetype_delta_nfr_long_target=(
                    phase_target.delta_nfr_long if phase_target is not None else avg_long
                ),
                archetype_delta_nfr_lat_target=(
                    phase_target.delta_nfr_lat if phase_target is not None else avg_lat
                ),
                archetype_nu_f_target=(
                    phase_target.nu_f if phase_target is not None else nu_f_target
                ),
                archetype_si_phi_target=(
                    phase_target.si_phi if phase_target is not None else target_alignment
                ),
                detune_ratio_weights=detune_weights,
                track_gradient=gradient_value,
            )
        )
    for legacy, phases in LEGACY_PHASE_MAP.items():
        for candidate in reversed(phases):
            nodes = dominant_nodes.get(candidate)
            if nodes:
                dominant_nodes[legacy] = nodes
                break
        else:
            if phases:
                dominant_nodes[legacy] = dominant_nodes.get(phases[-1], ())
        weight_values = [axis_weights.get(candidate) for candidate in phases if candidate in axis_weights]
        if weight_values:
            axis_weights[legacy] = {
                "longitudinal": float(
                    fmean(entry["longitudinal"] for entry in weight_values if entry is not None)
                ),
                "lateral": float(
                    fmean(entry["lateral"] for entry in weight_values if entry is not None)
                ),
            }
        target_values = [axis_targets.get(candidate) for candidate in phases if candidate in axis_targets]
        if target_values:
            axis_targets[legacy] = {
                "longitudinal": float(
                    fmean(entry["longitudinal"] for entry in target_values if entry is not None)
                ),
                "lateral": float(
                    fmean(entry["lateral"] for entry in target_values if entry is not None)
                ),
            }
    return tuple(goals), dominant_nodes, axis_targets, axis_weights


def _initial_phase_weight_map(
    records: Sequence[SupportsTelemetrySample],
    phase_samples: Mapping[PhaseLiteral, Iterable[int]],
    yaw_rates: Sequence[float],
) -> Dict[PhaseLiteral, Dict[str, float]]:
    weights: Dict[PhaseLiteral, Dict[str, float]] = {}

    def _tightness(values: Sequence[float], reference: float, cap: float) -> float:
        if not values:
            return 1.0
        span = max(values) - min(values)
        return 1.0 + min(cap, reference / (span + 1e-6))

    for phase, indices in phase_samples.items():
        phase_records = [records[i] for i in indices]
        slip_values = [record.slip_ratio for record in phase_records]
        lat_values = [record.lateral_accel for record in phase_records]
        long_values = [record.longitudinal_accel for record in phase_records]
        phase_yaw_rates = [_cached_yaw_rate(yaw_rates, idx) for idx in indices]

        slip_factor = _tightness(slip_values, 0.25, 1.6)
        long_factor = _tightness(long_values, 0.8, 1.4)
        lat_factor = _tightness(lat_values, 1.0, 1.4)
        yaw_factor = _tightness(phase_yaw_rates, 0.5, 1.5)

        profile = {
            "__default__": 1.0,
            "tyres": slip_factor,
            "transmission": (slip_factor + long_factor) / 2.0,
            "brakes": long_factor,
            "suspension": max(1.0, lat_factor),
            "chassis": max(1.0, (lat_factor + yaw_factor) / 2.0),
            "track": max(1.0, yaw_factor),
            "driver": max(1.0, yaw_factor),
        }

        weights[phase] = profile

    return weights


def _apply_phase_override(
    base: Mapping[str, float],
    override: Mapping[str, float] | float | None,
) -> Dict[str, float]:
    profile = {str(node): float(value) for node, value in base.items()}
    if override is None:
        return profile
    if isinstance(override, Mapping):
        default_scale = override.get("__default__")
        for node, value in list(profile.items()):
            scale = override.get(node, default_scale)
            if scale is None:
                continue
            try:
                profile[node] = float(value) * float(scale)
            except (TypeError, ValueError):
                continue
        for node, scale in override.items():
            if node in {"__default__"}:
                continue
            if node in profile:
                continue
            try:
                numeric = float(scale)
            except (TypeError, ValueError):
                continue
            base_default = (
                float(default_scale)
                if isinstance(default_scale, (int, float))
                else 1.0
            )
            profile[str(node)] = base_default * numeric
        return profile
    try:
        factor = float(override)
    except (TypeError, ValueError):
        return profile
    return {node: float(value) * factor for node, value in profile.items()}


def _blend_phase_weight_map(
    baseline: Mapping[PhaseLiteral, Mapping[str, float]],
    overrides: Mapping[str, Mapping[str, float] | float],
) -> Dict[PhaseLiteral, Dict[str, float]]:
    default_override = overrides.get("__default__")
    blended: Dict[PhaseLiteral, Dict[str, float]] = {}
    for phase, profile in baseline.items():
        override = overrides.get(phase)
        if override is None:
            for legacy, phases in LEGACY_PHASE_MAP.items():
                if phase in phases and legacy in overrides:
                    override = overrides[legacy]
                    break
        if override is None:
            override = default_override
        blended[phase] = _apply_phase_override(profile, override)
    for phase, override in overrides.items():
        if phase == "__default__":
            continue
        targets = expand_phase_alias(phase)
        if any(target in blended for target in targets):
            continue
        blended[str(phase)] = _apply_phase_override({"__default__": 1.0}, override)
    return blended


def _compute_window_occupancy(
    goals: Sequence[Goal],
    phase_samples: Mapping[PhaseLiteral, Iterable[int]],
    records: Sequence[SupportsTelemetrySample],
    yaw_rates: Sequence[float],
) -> Dict[PhaseLiteral, Dict[str, float]]:
    def _percentage(values: Sequence[float], window: Tuple[float, float]) -> float:
        if not values:
            return 0.0
        lower, upper = window
        if lower > upper:
            lower, upper = upper, lower
        total = len(values)
        if total == 0:
            return 0.0
        count = sum(1 for value in values if lower <= value <= upper)
        return 100.0 * count / total

    occupancy: Dict[PhaseLiteral, Dict[str, float]] = {}
    for goal in goals:
        indices = phase_samples.get(goal.phase, ())
        slip_values = [
            records[i].slip_ratio
            for i in indices
            if 0 <= i < len(records)
        ]
        phase_yaw_rates = [
            _cached_yaw_rate(yaw_rates, idx)
            for idx in indices
            if 0 <= idx < len(yaw_rates)
        ]
        occupancy[goal.phase] = {
            "slip_lat": _percentage(slip_values, goal.slip_lat_window),
            "slip_long": _percentage(slip_values, goal.slip_long_window),
            "yaw_rate": _percentage(phase_yaw_rates, goal.yaw_rate_window),
        }
    for legacy, phases in LEGACY_PHASE_MAP.items():
        values = [occupancy.get(phase) for phase in phases if phase in occupancy]
        if not values:
            continue
        aggregated: Dict[str, float] = {}
        keys = {key for entry in values for key in entry}
        for key in keys:
            aggregated[key] = fmean(entry.get(key, 0.0) for entry in values)
        occupancy[legacy] = aggregated
    return occupancy


def microsector_stability_metrics(
    microsector: Microsector,
) -> Tuple[float, float, float, float]:
    """Return structural silence coverage and variance metrics for ``microsector``."""

    events = getattr(microsector, "operator_events", {}) or {}
    payloads = [
        payload
        for payload in silence_event_payloads(events)  # type: ignore[assignment]
        if isinstance(payload, Mapping)
    ]
    start_time = float(getattr(microsector, "start_time", 0.0))
    end_time = float(getattr(microsector, "end_time", 0.0))
    duration = max(0.0, end_time - start_time)
    quiet_duration = sum(
        max(0.0, float(payload.get("duration", 0.0))) for payload in payloads
    )
    coverage = 0.0
    if duration > 1e-9:
        coverage = min(1.0, quiet_duration / duration)
    slack = max((float(payload.get("slack", 0.0)) for payload in payloads), default=0.0)
    filtered = getattr(microsector, "filtered_measures", {}) or {}
    try:
        si_variance = abs(float(filtered.get("si_variance", 0.0)))
    except (TypeError, ValueError):
        si_variance = 0.0
    try:
        epi_abs = abs(float(filtered.get("epi_derivative_abs", 0.0)))
    except (TypeError, ValueError):
        epi_abs = 0.0
    return coverage, slack, si_variance, epi_abs


def detect_quiet_microsector_streaks(
    microsectors: Sequence[Microsector],
    *,
    min_length: int = 3,
    coverage_threshold: float = 0.65,
    slack_threshold: float = 0.25,
    si_variance_threshold: float = 0.0025,
    epi_derivative_threshold: float = 0.18,
) -> List[Tuple[int, ...]]:
    """Return index streaks where consecutive microsectors remain quiet."""

    if min_length <= 0:
        min_length = 1
    quiet_flags: List[bool] = []
    for microsector in microsectors:
        coverage, slack, si_variance, epi_abs = microsector_stability_metrics(microsector)
        quiet_flags.append(
            coverage >= coverage_threshold
            and slack >= slack_threshold
            and si_variance <= si_variance_threshold
            and epi_abs <= epi_derivative_threshold
        )

    sequences: List[Tuple[int, ...]] = []
    start: int | None = None
    for index, quiet in enumerate(quiet_flags):
        if quiet:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= min_length:
                sequences.append(tuple(range(start, index)))
            start = None
    if start is not None and len(quiet_flags) - start >= min_length:
        sequences.append(tuple(range(start, len(quiet_flags))))
    return sequences


def _adjust_phase_weights_with_dominance(
    specs: Sequence[Dict[str, object]],
    bundles: Sequence[SupportsEPIBundle],
    records: Sequence[SupportsTelemetrySample],
    *,
    context_matrix: ContextMatrix,
    sample_context: Sequence[ContextFactors],
    sample_multipliers: Sequence[float] | None = None,
    yaw_rates: Sequence[float],
    node_delta_cache: (
        Sequence[Mapping[str, float]]
        | MutableSequence[Mapping[str, float]]
        | None
    ) = None,
    sample_rate: float | None = None,
    steer_norm: Sequence[float] | None = None,
    yaw_norm: Sequence[float] | None = None,
    lat_norm: Sequence[float] | None = None,
) -> tuple[bool, int | None]:
    adjusted = False
    first_index: int | None = None
    for spec in specs:
        phase_boundaries = spec["phase_boundaries"]
        archetype = _classify_archetype(
            spec["curvature"],
            spec.get("duration", 0.0),
            spec.get("speed_drop", 0.0),
            spec.get("direction_changes", 0),
        )
        goals, dominant_nodes, axis_targets, axis_weights = _build_goals(
            archetype,
            bundles,
            records,
            phase_boundaries,
            yaw_rates=yaw_rates,
            context_matrix=context_matrix,
            sample_context=sample_context,
            sample_multipliers=sample_multipliers,
            node_delta_cache=node_delta_cache,
            sample_rate=sample_rate,
            steer_norm=steer_norm,
            yaw_norm=yaw_norm,
            lat_norm=lat_norm,
        )
        spec["goals"] = goals
        spec["dominant_nodes"] = dominant_nodes
        spec["phase_axis_targets"] = axis_targets
        spec["phase_axis_weights"] = axis_weights
        spec["goal_cache_valid"] = True
        spec["goal_archetype"] = archetype
        spec["goal_gradient_signature"] = None
        active_goal = max(
            goals,
            key=lambda goal: abs(goal.target_delta_nfr) + goal.nu_f_target,
        )
        spec["active_phase"] = active_goal.phase
        profile = spec["phase_weights"]
        for goal in goals:
            phase_profile = dict(profile.get(goal.phase, {"__default__": 1.0}))
            boost = 1.0 + min(0.6, max(0.0, goal.nu_f_target))
            changed = False
            for node in goal.dominant_nodes:
                current = phase_profile.get(node, phase_profile.get("__default__", 1.0))
                boosted = max(0.5, current * boost)
                if abs(boosted - current) > 1e-9:
                    phase_profile[node] = boosted
                    changed = True
            if changed:
                profile[goal.phase] = phase_profile
                adjusted = True
                candidate = spec.get("start")
                if not isinstance(candidate, int):
                    indices = spec.get("phase_samples", {}).get(goal.phase, ())
                    if indices:
                        candidate = min(indices)
                if isinstance(candidate, int):
                    first_index = candidate if first_index is None else min(first_index, candidate)
        spec["phase_weights"] = profile
    return adjusted, first_index


def _phase_gradient_signature(
    phase_gradients: Mapping[PhaseLiteral, float] | None,
) -> Tuple[Tuple[str, float], ...] | None:
    if not phase_gradients:
        return None
    signature: List[Tuple[str, float]] = []
    for phase, value in phase_gradients.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        signature.append((str(phase), numeric))
    if not signature:
        return None
    signature.sort(key=lambda entry: entry[0])
    return tuple(signature)


_BUNDLE_DERIVED_FIELDS = (
    "goal_phase_cache",
    "adjusted_deltas",
    "delta_signature",
    "avg_si",
    "phase_axis_targets",
    "phase_axis_weights",
    "dominant_nodes",
    "goals",
    "active_phase",
)


def _invalidate_goal_cache(specs: Sequence[Dict[str, object]], start_index: int | None) -> None:
    if start_index is None:
        return

    def _coerce_index(value: object, *, default: int | None = None) -> int | None:
        if isinstance(value, bool):  # pragma: no cover - defensive guard
            return int(value)
        if isinstance(value, int):
            return value
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    if start_index <= 0:
        affected = specs
    else:
        affected = []
        for spec in specs:
            end_index = _coerce_index(spec.get("end"), default=-1)
            if end_index is not None and end_index >= start_index:
                affected.append(spec)

    for spec in affected:
        spec["goal_cache_valid"] = False
        spec["goal_gradient_signature"] = None
        spec["goal_archetype"] = None
        for field in _BUNDLE_DERIVED_FIELDS:
            spec.pop(field, None)


def _phase_alignment_targets(archetype: str) -> Mapping[str, Tuple[float, float]]:
    table = archetype_phase_targets(archetype)
    result: Dict[str, Tuple[float, float]] = {}
    for phase in PHASE_SEQUENCE:
        family = phase_family(phase)
        target = table.get(family)
        if target is None:
            fallback = archetype_phase_targets(ARCHETYPE_MEDIUM).get(family)
        else:
            fallback = target
        if fallback is None:
            result[phase] = (0.0, 0.9)
        else:
            result[phase] = (fallback.lag, fallback.si_phi)
    return result


def _goal_descriptions(archetype: str) -> Mapping[str, str]:
    base = {
        "entry1": "Modulate the initial transfer to consolidate the {archetype} archetype.",
        "entry2": "Deepen brake preparation in line with the {archetype} archetype.",
        "apex3a": "Align the approach to the apex with the {archetype} pattern.",
        "apex3b": "Hold the apex per the {archetype} archetype while keeping ΔNFR steady.",
        "exit4": "Release energy on exit following the {archetype} archetype.",
    }
    if archetype == ARCHETYPE_HAIRPIN:
        base.update(
            {
                "entry1": "Extend braking to nail the hairpin with control.",
                "entry2": "Settle the car into maximum support before the apex.",
                "apex3a": "Pivot patiently to close the hairpin.",
                "apex3b": "Manage rotation while keeping the support point.",
                "exit4": "Open the steering while prioritising traction.",
            }
        )
    elif archetype == ARCHETYPE_CHICANE:
        base.update(
            {
                "entry1": "Prepare the first direction change with neutral balance.",
                "entry2": "Synchronise the weight transfer into the second support.",
                "apex3a": "Link the apexes while keeping the transition fluid.",
                "apex3b": "Maintain lightness to avoid saturating the second support.",
                "exit4": "Complete the chicane by stabilising the car on exit.",
            }
        )
    elif archetype == ARCHETYPE_FAST:
        base.update(
            {
                "entry1": "Trace the fast approach without compromising stability.",
                "entry2": "Refine the line while minimising corrections.",
                "apex3a": "Cross the apex with smooth, constant load.",
                "apex3b": "Sustain speed through continuous support.",
                "exit4": "Project the exit while maintaining high pace.",
            }
        )
    elif archetype == ARCHETYPE_MEDIUM:
        base.update(
            {
                "entry1": "Settle braking while balancing the car.",
                "entry2": "Prepare the apex with progressive transfer.",
                "apex3a": "Link the turn while keeping load even.",
                "apex3b": "Control the drift to hold the line.",
                "exit4": "Reapply power without breaking stability.",
            }
        )
    return {phase: message.format(archetype=archetype) for phase, message in base.items()}

