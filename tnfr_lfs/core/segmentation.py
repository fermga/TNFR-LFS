"""Microsector segmentation utilities.

This module analyses the stream of telemetry samples together with the
corresponding :class:`~tnfr_lfs.core.epi_models.EPIBundle` instances to
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
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .epi import (
    DEFAULT_PHASE_WEIGHTS,
    DeltaCalculator,
    TelemetryRecord,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
from .epi_models import EPIBundle
from .operators import mutation_operator, recursivity_operator
from .phases import LEGACY_PHASE_MAP, PHASE_SEQUENCE, expand_phase_alias
from .spectrum import phase_alignment

# Thresholds derived from typical race car dynamics.  They can be tuned in
# the future without affecting the public API of the segmentation module.
CURVATURE_THRESHOLD = 1.2  # g units of lateral acceleration
MIN_SEGMENT_LENGTH = 3
BRAKE_THRESHOLD = -0.35  # g units of longitudinal deceleration
SUPPORT_THRESHOLD = 350.0  # Newtons of vertical load delta


PhaseLiteral = str


@dataclass(frozen=True)
class Goal:
    """Operational goal associated with a microsector phase."""

    phase: PhaseLiteral
    archetype: str
    description: str
    target_delta_nfr: float
    target_sense_index: float
    nu_f_target: float
    target_phase_lag: float
    target_phase_alignment: float
    measured_phase_lag: float
    measured_phase_alignment: float
    slip_lat_window: Tuple[float, float]
    slip_long_window: Tuple[float, float]
    yaw_rate_window: Tuple[float, float]
    dominant_nodes: Tuple[str, ...]


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
    filtered_measures: Mapping[str, float]
    recursivity_trace: Tuple[Mapping[str, float | str | None], ...]
    last_mutation: Mapping[str, object] | None
    window_occupancy: Mapping[PhaseLiteral, Mapping[str, float]]

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


def segment_microsectors(
    records: Sequence[TelemetryRecord],
    bundles: Sequence[EPIBundle],
    *,
    operator_state: MutableMapping[str, Dict[str, Dict[str, object]]] | None = None,
    recursion_decay: float = 0.4,
    mutation_thresholds: Mapping[str, float] | None = None,
    phase_weight_overrides: Mapping[str, Mapping[str, float] | float] | None = None,
) -> List[Microsector]:
    """Derive microsectors from telemetry and ΔNFR signatures.

    Parameters
    ----------
    records:
        Telemetry samples in chronological order.
    bundles:
        Computed :class:`~tnfr_lfs.core.epi_models.EPIBundle` for the same
        timestamps as ``records``.

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

    segments = _identify_corner_segments(records)
    if not segments:
        return []

    baseline = DeltaCalculator.derive_baseline(records)
    bundle_list = list(bundles)
    microsectors: List[Microsector] = []
    rec_state: MutableMapping[str, Dict[str, object]] | None = None
    mutation_state: MutableMapping[str, Dict[str, object]] | None = None
    if operator_state is not None:
        rec_state = operator_state.setdefault("recursivity", {})
        mutation_state = operator_state.setdefault("mutation", {})
    thresholds = mutation_thresholds or {}
    phase_assignments: Dict[int, PhaseLiteral] = {}
    weight_lookup: Dict[int, Mapping[str, Mapping[str, float] | float]] = {}
    specs: List[Dict[str, object]] = []
    for index, (start, end) in enumerate(segments):
        phase_boundaries = _compute_phase_boundaries(records, start, end)
        phase_samples = {
            phase: tuple(range(bounds[0], bounds[1]))
            for phase, bounds in phase_boundaries.items()
        }
        curvature = mean(abs(records[i].lateral_accel) for i in range(start, end + 1))
        brake_event = any(records[i].longitudinal_accel <= BRAKE_THRESHOLD for i in range(start, end + 1))
        support_event = _detect_support_event(records[start : end + 1])
        avg_vertical_load = mean(record.vertical_load for record in records[start : end + 1])
        baseline_vertical = getattr(baseline, "vertical_load", 0.0)
        grip_rel = (
            avg_vertical_load / baseline_vertical if baseline_vertical > 1e-9 else 0.0
        )
        phase_weight_map = _initial_phase_weight_map(records, phase_samples)
        if phase_weight_overrides:
            phase_weight_map = _blend_phase_weight_map(
                phase_weight_map, phase_weight_overrides
            )
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
            }
        )
        for phase, indices in phase_samples.items():
            for sample_index in indices:
                phase_assignments[sample_index] = phase
                weight_lookup[sample_index] = phase_weight_map

    recomputed_bundles = _recompute_bundles(
        records,
        bundle_list,
        baseline,
        phase_assignments,
        weight_lookup,
    )

    weights_adjusted = _adjust_phase_weights_with_dominance(
        specs,
        recomputed_bundles,
        records,
    )

    if weights_adjusted:
        recomputed_bundles = _recompute_bundles(
            records,
            recomputed_bundles,
            baseline,
            phase_assignments,
            weight_lookup,
        )

    goal_nu_f_lookup: Dict[int, float] = {}
    for spec in specs:
        start = spec["start"]
        end = spec["end"]
        delta_signature = mean(b.delta_nfr for b in recomputed_bundles[start : end + 1])
        avg_si = mean(b.sense_index for b in recomputed_bundles[start : end + 1])
        archetype = _classify_archetype(
            delta_signature,
            avg_si,
            spec["brake_event"],
            spec["support_event"],
        )
        goals, _ = _build_goals(
            archetype,
            recomputed_bundles,
            records,
            spec["phase_boundaries"],
        )
        for goal in goals:
            indices = spec["phase_samples"].get(goal.phase, ())
            for sample_index in indices:
                goal_nu_f_lookup[sample_index] = goal.nu_f_target

    if goal_nu_f_lookup:
        recomputed_bundles = _recompute_bundles(
            records,
            recomputed_bundles,
            baseline,
            phase_assignments,
            weight_lookup,
            goal_nu_f_lookup=goal_nu_f_lookup,
        )

    for spec in specs:
        start = spec["start"]
        end = spec["end"]
        phase_boundaries = spec["phase_boundaries"]
        phase_samples = spec["phase_samples"]
        phase_weights = {
            phase: dict(profile)
            for phase, profile in spec["phase_weights"].items()
        }
        curvature = spec["curvature"]
        brake_event = spec["brake_event"]
        support_event = spec["support_event"]
        avg_vertical_load = float(spec.get("avg_vertical_load", 0.0))
        grip_rel = float(spec.get("grip_rel", 0.0))
        delta_signature = mean(b.delta_nfr for b in recomputed_bundles[start : end + 1])
        avg_si = mean(b.sense_index for b in recomputed_bundles[start : end + 1])
        archetype = _classify_archetype(delta_signature, avg_si, brake_event, support_event)
        goals, dominant_nodes = _build_goals(
            archetype,
            recomputed_bundles,
            records,
            phase_boundaries,
        )
        active_goal = max(
            goals,
            key=lambda goal: abs(goal.target_delta_nfr) + goal.nu_f_target,
        )
        filtered_measures: Dict[str, float] = {
            "thermal_load": avg_vertical_load,
            "style_index": avg_si,
            "grip_rel": grip_rel,
        }
        rec_trace: Tuple[Mapping[str, float | str | None], ...] = ()
        mutation_details: Mapping[str, object] | None = None
        if rec_state is not None:
            measures = {
                "thermal_load": avg_vertical_load,
                "style_index": avg_si,
                "phase": active_goal.phase,
                "grip_rel": grip_rel,
            }
            rec_info = recursivity_operator(
                rec_state,
                str(index),
                measures,
                decay=recursion_decay,
            )
            entropy_value = _estimate_entropy(records, start, end)
            triggers = {
                "microsector_id": str(index),
                "current_archetype": archetype,
                "candidate_archetype": archetype,
                "fallback_archetype": "recuperacion",
                "entropy": entropy_value,
                "style_index": rec_info["filtered"].get("style_index", avg_si),
                "style_reference": avg_si,
                "phase": rec_info.get("phase", active_goal.phase),
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
                goals, dominant_nodes = _build_goals(
                    archetype,
                    bundles,
                    records,
                    phase_boundaries,
                )
                active_goal = max(
                    goals,
                    key=lambda goal: abs(goal.target_delta_nfr) + goal.nu_f_target,
                )
                recursivity_operator(
                    rec_state,
                    str(index),
                    {
                        "thermal_load": measures["thermal_load"],
                        "style_index": measures["style_index"],
                        "grip_rel": measures["grip_rel"],
                        "phase": active_goal.phase,
                    },
                    decay=recursion_decay,
                )
            micro_state_entry = rec_state.get(str(index), {})
            filtered_measures = {
                key: float(value)
                for key, value in micro_state_entry.get("filtered", {}).items()
            }
            if "thermal_load" not in filtered_measures:
                filtered_measures["thermal_load"] = avg_vertical_load
            if "style_index" not in filtered_measures:
                filtered_measures["style_index"] = avg_si
            if "grip_rel" not in filtered_measures:
                filtered_measures["grip_rel"] = grip_rel
            rec_trace = tuple(
                {
                    trace_key: (
                        float(trace_value)
                        if isinstance(trace_value, (int, float))
                        else trace_value
                    )
                    for trace_key, trace_value in trace_entry.items()
                }
                for trace_entry in micro_state_entry.get("trace", [])
            )
            mutation_details = {
                key: (
                    float(value)
                    if isinstance(value, (int, float))
                    else value
                )
                for key, value in (mutation_info or {}).items()
            }
        occupancy = _compute_window_occupancy(goals, phase_samples, records)
        phase_lag_map = {
            goal.phase: float(goal.measured_phase_lag) for goal in goals
        }
        phase_alignment_map = {
            goal.phase: float(goal.measured_phase_alignment) for goal in goals
        }
        for legacy, phases in LEGACY_PHASE_MAP.items():
            lag_values = [phase_lag_map.get(candidate) for candidate in phases if candidate in phase_lag_map]
            align_values = [
                phase_alignment_map.get(candidate)
                for candidate in phases
                if candidate in phase_alignment_map
            ]
            if lag_values:
                phase_lag_map[legacy] = float(mean(value for value in lag_values if value is not None))
            if align_values:
                phase_alignment_map[legacy] = float(
                    mean(value for value in align_values if value is not None)
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
                phase_samples=dict(phase_samples),
                active_phase=active_goal.phase,
                dominant_nodes=dict(dominant_nodes),
                phase_weights=phase_weights,
                grip_rel=float(filtered_measures.get("grip_rel", grip_rel)),
                phase_lag=dict(phase_lag_map),
                phase_alignment=dict(phase_alignment_map),
                filtered_measures=dict(filtered_measures),
                recursivity_trace=rec_trace,
                last_mutation=dict(mutation_details) if mutation_details is not None else None,
                window_occupancy=occupancy,
            )
        )

    if isinstance(bundles, list):
        bundles[:] = recomputed_bundles

    return microsectors


def _recompute_bundles(
    records: Sequence[TelemetryRecord],
    bundles: Sequence[EPIBundle],
    baseline: TelemetryRecord,
    phase_assignments: Mapping[int, PhaseLiteral],
    weight_lookup: Mapping[int, Mapping[str, Mapping[str, float] | float]],
    goal_nu_f_lookup: Mapping[int, Mapping[str, float] | float] | None = None,
) -> List[EPIBundle]:
    recomputed: List[EPIBundle] = []
    prev_integrated: float | None = None
    prev_timestamp = records[0].timestamp if records else 0.0
    for idx, record in enumerate(records):
        dt = 0.0 if idx == 0 else max(0.0, record.timestamp - prev_timestamp)
        phase = phase_assignments.get(idx, PHASE_SEQUENCE[0])
        phase_weights = weight_lookup.get(idx, DEFAULT_PHASE_WEIGHTS)
        target_nu_f = goal_nu_f_lookup.get(idx) if goal_nu_f_lookup else None
        nu_f_map = resolve_nu_f_by_node(
            record,
            phase=phase,
            phase_weights=phase_weights,
        )
        epi_value = bundles[idx].epi if idx < len(bundles) else 0.0
        recomputed_bundle = DeltaCalculator.compute_bundle(
            record,
            baseline,
            epi_value,
            prev_integrated_epi=prev_integrated,
            dt=dt,
            nu_f_by_node=nu_f_map,
            phase=phase,
            phase_weights=phase_weights,
            phase_target_nu_f=target_nu_f,
        )
        recomputed.append(recomputed_bundle)
        prev_integrated = recomputed_bundle.integrated_epi
        prev_timestamp = record.timestamp
    return recomputed


def _estimate_entropy(
    records: Sequence[TelemetryRecord], start: int, end: int
) -> float:
    node_weights: Dict[str, float] = defaultdict(float)
    for index in range(start, end + 1):
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


def _identify_corner_segments(records: Sequence[TelemetryRecord]) -> List[Tuple[int, int]]:
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
    records: Sequence[TelemetryRecord], start: int, end: int
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


def _detect_support_event(records: Sequence[TelemetryRecord]) -> bool:
    loads = [record.vertical_load for record in records]
    return max(loads) - min(loads) >= SUPPORT_THRESHOLD


def _classify_archetype(
    delta_signature: float,
    sense_index_value: float,
    brake_event: bool,
    support_event: bool,
) -> str:
    if sense_index_value < 0.55:
        return "recuperacion"
    if delta_signature > 5.0 and support_event:
        return "apoyo"
    if delta_signature < -5.0 and brake_event:
        return "liberacion"
    return "equilibrio"


def _compute_yaw_rate(records: Sequence[TelemetryRecord], index: int) -> float:
    if index <= 0 or index >= len(records):
        return 0.0
    current = records[index]
    previous = records[index - 1]
    dt = current.timestamp - previous.timestamp
    if dt <= 1e-9:
        return 0.0
    delta = current.yaw - previous.yaw
    wrapped = (delta + math.pi) % (2.0 * math.pi) - math.pi
    return wrapped / dt


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return mean(values) if values else default


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


def _build_goals(
    archetype: str,
    bundles: Sequence[EPIBundle],
    records: Sequence[TelemetryRecord],
    boundaries: Mapping[PhaseLiteral, Tuple[int, int]],
) -> Tuple[Tuple[Goal, ...], Mapping[PhaseLiteral, Tuple[str, ...]]]:
    descriptions = _goal_descriptions(archetype)
    alignment_targets = _phase_alignment_targets(archetype)
    goals: List[Goal] = []
    dominant_nodes: Dict[PhaseLiteral, Tuple[str, ...]] = {}
    for phase in PHASE_SEQUENCE:
        start, stop = boundaries[phase]
        indices = [idx for idx in range(start, min(stop, len(bundles)))]
        segment = [bundles[idx] for idx in indices]
        phase_records = [records[idx] for idx in indices]
        if segment:
            avg_delta = mean(bundle.delta_nfr for bundle in segment)
            avg_si = mean(bundle.sense_index for bundle in segment)
        else:
            avg_delta = 0.0
            avg_si = 1.0

        _, measured_lag, measured_alignment = phase_alignment(phase_records)
        target_lag, target_alignment = alignment_targets.get(
            phase, (0.0, 0.9)
        )

        node_metrics: Dict[str, Dict[str, float]] = defaultdict(lambda: {"abs_delta": 0.0, "nu_f_weight": 0.0})
        for local_index, idx in enumerate(indices):
            record = phase_records[local_index]
            bundle = segment[local_index]
            node_deltas = delta_nfr_by_node(record)
            for node, delta in node_deltas.items():
                weight = abs(delta)
                node_metrics[node]["abs_delta"] += weight
                nu_f = getattr(bundle, node).nu_f if hasattr(bundle, node) else 0.0
                node_metrics[node]["nu_f_weight"] += weight * nu_f

        sorted_nodes = sorted(
            (node, metrics)
            for node, metrics in node_metrics.items()
            if metrics["abs_delta"] > 0.0
        )
        sorted_nodes.sort(key=lambda item: item[1]["abs_delta"], reverse=True)
        phase_nodes = tuple(node for node, _ in sorted_nodes[:3])
        if not phase_nodes and node_metrics:
            phase_nodes = tuple(list(node_metrics.keys())[:3])
        dominant_nodes[phase] = phase_nodes

        total_weight = sum(node_metrics[node]["abs_delta"] for node in phase_nodes)
        if total_weight > 0.0:
            weighted_nu_f = sum(node_metrics[node]["nu_f_weight"] for node in phase_nodes)
            nu_f_target = weighted_nu_f / total_weight
        else:
            nu_f_target = 0.0

        sample_count = max(1, len(indices))
        dominant_intensity = total_weight / sample_count
        influence_factor = 1.0 + min(2.0, nu_f_target) + min(1.5, dominant_intensity / 5.0)

        slip_values = [record.slip_ratio for record in phase_records]
        lat_values = [record.lateral_accel for record in phase_records]
        long_values = [record.longitudinal_accel for record in phase_records]
        yaw_rates = [_compute_yaw_rate(records, idx) for idx in indices]

        lat_scale = influence_factor * (1.0 + min(1.5, abs(_safe_mean(lat_values)) / 5.0))
        long_scale = influence_factor * (1.0 + min(1.5, abs(_safe_mean(long_values)) / 5.0))
        yaw_scale = influence_factor * (1.0 + min(1.5, abs(_safe_mean(yaw_rates)) / 2.0))

        slip_lat_window = _window(slip_values, lat_scale)
        slip_long_window = _window(slip_values, long_scale)
        yaw_rate_window = _window(yaw_rates, yaw_scale, minimum=0.005)

        goals.append(
            Goal(
                phase=phase,
                archetype=archetype,
                description=descriptions[phase],
                target_delta_nfr=avg_delta,
                target_sense_index=avg_si,
                nu_f_target=nu_f_target,
                target_phase_lag=target_lag,
                target_phase_alignment=target_alignment,
                measured_phase_lag=measured_lag,
                measured_phase_alignment=measured_alignment,
                slip_lat_window=slip_lat_window,
                slip_long_window=slip_long_window,
                yaw_rate_window=yaw_rate_window,
                dominant_nodes=phase_nodes,
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
    return tuple(goals), dominant_nodes


def _initial_phase_weight_map(
    records: Sequence[TelemetryRecord],
    phase_samples: Mapping[PhaseLiteral, Tuple[int, ...]],
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
        yaw_rates = [_compute_yaw_rate(records, idx) for idx in indices]

        slip_factor = _tightness(slip_values, 0.25, 1.6)
        long_factor = _tightness(long_values, 0.8, 1.4)
        lat_factor = _tightness(lat_values, 1.0, 1.4)
        yaw_factor = _tightness(yaw_rates, 0.5, 1.5)

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
    phase_samples: Mapping[PhaseLiteral, Tuple[int, ...]],
    records: Sequence[TelemetryRecord],
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
        slip_values = [records[i].slip_ratio for i in indices]
        yaw_rates = [_compute_yaw_rate(records, idx) for idx in indices]
        occupancy[goal.phase] = {
            "slip_lat": _percentage(slip_values, goal.slip_lat_window),
            "slip_long": _percentage(slip_values, goal.slip_long_window),
            "yaw_rate": _percentage(yaw_rates, goal.yaw_rate_window),
        }
    for legacy, phases in LEGACY_PHASE_MAP.items():
        values = [occupancy.get(phase) for phase in phases if phase in occupancy]
        if not values:
            continue
        aggregated: Dict[str, float] = {}
        keys = {key for entry in values for key in entry}
        for key in keys:
            aggregated[key] = mean(entry.get(key, 0.0) for entry in values)
        occupancy[legacy] = aggregated
    return occupancy


def _adjust_phase_weights_with_dominance(
    specs: Sequence[Dict[str, object]],
    bundles: Sequence[EPIBundle],
    records: Sequence[TelemetryRecord],
) -> bool:
    adjusted = False
    for spec in specs:
        phase_boundaries = spec["phase_boundaries"]
        archetype = _classify_archetype(
            mean(b.delta_nfr for b in bundles[spec["start"] : spec["end"] + 1]),
            mean(b.sense_index for b in bundles[spec["start"] : spec["end"] + 1]),
            spec["brake_event"],
            spec["support_event"],
        )
        goals, dominant_nodes = _build_goals(
            archetype,
            bundles,
            records,
            phase_boundaries,
        )
        spec["goals"] = goals
        spec["dominant_nodes"] = dominant_nodes
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
        spec["phase_weights"] = profile
    return adjusted


def _phase_alignment_targets(archetype: str) -> Mapping[str, Tuple[float, float]]:
    base: Dict[str, Tuple[float, float]] = {
        "entry1": (0.0, 0.9),
        "entry2": (0.0, 0.9),
        "apex3a": (0.0, 0.92),
        "apex3b": (0.0, 0.92),
        "exit4": (0.0, 0.88),
    }
    if archetype == "apoyo":
        adjustments = {
            "entry1": (-0.08, 0.9),
            "entry2": (-0.05, 0.9),
            "apex3a": (-0.02, 0.93),
            "apex3b": (0.0, 0.94),
            "exit4": (0.04, 0.9),
        }
    elif archetype == "liberacion":
        adjustments = {
            "entry1": (0.05, 0.85),
            "entry2": (0.08, 0.82),
            "apex3a": (0.12, 0.8),
            "apex3b": (0.14, 0.78),
            "exit4": (0.18, 0.75),
        }
    elif archetype == "recuperacion":
        adjustments = {
            "entry1": (0.0, 0.88),
            "entry2": (0.0, 0.9),
            "apex3a": (0.0, 0.9),
            "apex3b": (0.0, 0.92),
            "exit4": (0.02, 0.9),
        }
    else:
        adjustments = {
            "entry1": (0.0, 0.92),
            "entry2": (0.0, 0.92),
            "apex3a": (0.0, 0.94),
            "apex3b": (0.0, 0.94),
            "exit4": (0.0, 0.9),
        }
    result: Dict[str, Tuple[float, float]] = {}
    for phase, defaults in base.items():
        if phase in adjustments:
            result[phase] = adjustments[phase]
        else:
            result[phase] = defaults
    return result


def _goal_descriptions(archetype: str) -> Mapping[str, str]:
    base = {
        "entry1": "Modular la transferencia inicial para consolidar el arquetipo de {archetype}.",
        "entry2": "Profundizar la preparación de frenada siguiendo el arquetipo de {archetype}.",
        "apex3a": "Alinear la aproximación al vértice con el patrón de {archetype}.",
        "apex3b": "Sostener el vértice según el arquetipo de {archetype} manteniendo ΔNFR estable.",
        "exit4": "Liberar energía siguiendo el arquetipo de {archetype} hacia la salida.",
    }
    if archetype == "apoyo":
        base.update(
            {
                "entry1": "Generar deceleración progresiva para preparar el apoyo.",
                "entry2": "Consolidar la transferencia previa reforzando el apoyo.",
                "apex3a": "Sostener la carga máxima al llegar al vértice.",
                "apex3b": "Mantener el apoyo al pivotar sobre el vértice.",
                "exit4": "Transferir carga manteniendo el apoyo en la tracción inicial.",
            }
        )
    elif archetype == "liberacion":
        base.update(
            {
                "entry1": "Extender la frenada inicial para inducir liberación controlada.",
                "entry2": "Dosificar la liberación antes del vértice.",
                "apex3a": "Permitir la rotación asociada al arquetipo de liberación.",
                "apex3b": "Dirigir la liberación hacia la reaplicación de par.",
                "exit4": "Reequilibrar al salir para cerrar la fase de liberación.",
            }
        )
    elif archetype == "recuperacion":
        base.update(
            {
                "entry1": "Recuperar estabilidad en la aproximación inicial.",
                "entry2": "Afirmar la estabilidad antes del vértice.",
                "apex3a": "Maximizar el índice de sentido al tomar el vértice.",
                "apex3b": "Garantizar la recuperación durante la transición del vértice.",
                "exit4": "Asegurar la tracción evitando pérdidas posteriores.",
            }
        )
    return {phase: message.format(archetype=archetype) for phase, message in base.items()}

