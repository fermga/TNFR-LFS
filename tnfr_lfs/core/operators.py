"""High-level TNFR × LFS operators for telemetry analytics pipelines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from statistics import mean, pvariance
from typing import Dict, List, Mapping, MutableMapping, Sequence, TYPE_CHECKING

from .dissonance import compute_useful_dissonance_stats
from .epi import (
    EPIExtractor,
    TelemetryRecord,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
from .epi_models import EPIBundle
from .phases import expand_phase_alias, phase_family

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .segmentation import Microsector


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


def evolve_epi(
    prev_epi: float,
    delta_map: Mapping[str, float],
    dt: float,
    nu_f_by_node: Mapping[str, float],
) -> tuple[float, float, Dict[str, tuple[float, float]]]:
    """Integrate the Event Performance Index using explicit Euler steps.

    The integrator now returns the global derivative/integral together with a
    per-node breakdown.  The nodal contribution dictionary maps the node name
    to a ``(integral, derivative)`` tuple representing the instantaneous
    change produced during ``dt``.
    """

    if dt < 0.0:
        raise ValueError("dt must be non-negative")

    nodal_evolution: Dict[str, tuple[float, float]] = {}
    derivative = 0.0
    nodes = set(delta_map) | set(nu_f_by_node)
    for node in nodes:
        weight = nu_f_by_node.get(node, 0.0)
        node_delta = delta_map.get(node, 0.0)
        node_derivative = weight * node_delta
        node_integral = node_derivative * dt
        derivative += node_derivative
        nodal_evolution[node] = (node_integral, node_derivative)
    new_epi = prev_epi + (derivative * dt)
    return new_epi, derivative, nodal_evolution


def emission_operator(target_delta_nfr: float, target_sense_index: float) -> Dict[str, float]:
    """Return normalised objectives for ΔNFR and sense index targets."""

    target_si = max(0.0, min(1.0, target_sense_index))
    return {"delta_nfr": float(target_delta_nfr), "sense_index": target_si}


def recepcion_operator(
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
    half_window = window // 2
    smoothed: List[float] = []
    for index in range(len(series)):
        start = max(0, index - half_window)
        end = min(len(series), index + half_window + 1)
        smoothed.append(mean(series[start:end]))
    original_mean = mean(series)
    smoothed_mean = mean(smoothed)
    bias = original_mean - smoothed_mean
    if abs(bias) < 1e-12:
        return smoothed
    return [value + bias for value in smoothed]


def dissonance_operator(series: Sequence[float], target: float) -> float:
    """Compute the mean absolute deviation relative to a target value."""

    if not series:
        return 0.0
    return mean(abs(value - target) for value in series)


def dissonance_breakdown_operator(
    series: Sequence[float],
    target: float,
    *,
    microsectors: Sequence["Microsector"] | None = None,
    bundles: Sequence[EPIBundle] | None = None,
) -> DissonanceBreakdown:
    """Classify support events into useful (positive) and parasitic dissonance."""

    base_value = dissonance_operator(series, target)
    useful_events = 0
    parasitic_events = 0
    useful_magnitude = 0.0
    parasitic_magnitude = 0.0
    useful_dissonance_samples = 0
    high_yaw_acc_samples = 0

    if microsectors and bundles:
        bundle_count = len(bundles)
        for microsector in microsectors:
            if not microsector.support_event:
                continue
            apex_goal = None
            for apex_phase in expand_phase_alias("apex"):
                apex_goal = next(
                    (goal for goal in microsector.goals if goal.phase == apex_phase),
                    None,
                )
                if apex_goal is not None:
                    break
            if apex_goal is None:
                continue
            apex_indices: List[int] = []
            for apex_phase in expand_phase_alias("apex"):
                indices = microsector.phase_samples.get(apex_phase) or ()
                apex_indices.extend(idx for idx in indices if 0 <= idx < bundle_count)
            if not apex_indices:
                continue
            tyre_delta = [bundles[idx].tyres.delta_nfr for idx in apex_indices]
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
        delta_series = [bundle.delta_nfr for bundle in bundles]
        yaw_rates = [bundle.chassis.yaw_rate for bundle in bundles]
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


def acoplamiento_operator(
    series_a: Sequence[float], series_b: Sequence[float], *, strict_length: bool = True
) -> float:
    """Return the normalised coupling (correlation) between two series."""

    values_a = list(series_a)
    values_b = list(series_b)
    if strict_length and len(values_a) != len(values_b):
        raise ValueError("series must have the same length")

    length = min(len(values_a), len(values_b))
    if length == 0:
        return 0.0

    if len(values_a) != length:
        values_a = values_a[:length]
    if len(values_b) != length:
        values_b = values_b[:length]

    mean_a = mean(values_a)
    mean_b = mean(values_b)
    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
    variance_a = sum((a - mean_a) ** 2 for a in values_a)
    variance_b = sum((b - mean_b) ** 2 for b in values_b)
    if variance_a == 0 or variance_b == 0:
        return 0.0
    return covariance / sqrt(variance_a * variance_b)


def pairwise_coupling_operator(
    series_by_node: Mapping[str, Sequence[float]],
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> Dict[str, float]:
    """Compute coupling metrics for each node pair using ``acoplamiento``."""

    if pairs is None:
        ordered_nodes = list(series_by_node.keys())
        pairs = [(a, b) for idx, a in enumerate(ordered_nodes) for b in ordered_nodes[idx + 1 :]]

    coupling: Dict[str, float] = {}
    for first, second in pairs:
        series_a = series_by_node.get(first, ())
        series_b = series_by_node.get(second, ())
        label = f"{first}↔{second}"
        if not series_a or not series_b:
            coupling[label] = 0.0
            continue
        coupling[label] = acoplamiento_operator(series_a, series_b, strict_length=False)
    return coupling


def resonance_operator(series: Sequence[float]) -> float:
    """Compute the root-mean-square (RMS) resonance of a series."""

    if not series:
        return 0.0
    return sqrt(mean(value * value for value in series))


def recursivity_operator(
    state: MutableMapping[str, Dict[str, object]],
    microsector_id: str,
    measures: Mapping[str, float | str],
    *,
    decay: float = 0.4,
    history: int = 20,
) -> Dict[str, object]:
    """Maintain a recursive state per microsector for thermal/style metrics.

    Parameters
    ----------
    state:
        Mutable mapping that stores the per-microsector internal state.  The
        mapping is updated in-place.
    microsector_id:
        Identifier of the microsector being updated.  Typically the ordinal
        position in the segmentation output.
    measures:
        Mapping with the instantaneous measurements for the microsector.  The
        canonical keys are ``"thermal_load"`` and ``"style_index"`` together
        with an optional ``"phase"`` literal describing the active phase
        (entry/apex/exit).  Additional numeric keys are filtered using the same
        exponential decay.
    decay:
        Exponential decay factor in the ``[0, 1)`` interval.  Values closer to
        one keep a longer memory while values near zero prioritise the most
        recent measure.
    history:
        Maximum number of filtered samples to retain in the per-microsector
        history trace.

    Returns
    -------
    dict
        A dictionary containing the filtered metrics, the phase and whether a
        phase change was detected.
    """

    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in the [0, 1) interval")
    if not microsector_id:
        raise ValueError("microsector_id must be a non-empty string")
    if history <= 0:
        raise ValueError("history must be a positive integer")

    micro_state = state.setdefault(
        microsector_id,
        {
            "filtered": {},
            "phase": None,
            "samples": 0,
            "trace": [],
        },
    )

    phase = measures.get("phase") if isinstance(measures.get("phase"), str) else None
    previous_phase = micro_state.get("phase")
    phase_changed = phase is not None and previous_phase is not None and phase != previous_phase
    if phase is not None:
        micro_state["phase"] = phase

    filtered_values: Dict[str, float] = {}
    numeric_keys = [
        key
        for key, value in measures.items()
        if key != "phase" and isinstance(value, (int, float))
    ]
    for key in numeric_keys:
        value = float(measures[key])
        previous = micro_state["filtered"].get(key)
        if previous is None or phase_changed:
            filtered = value
        else:
            filtered = (decay * float(previous)) + ((1.0 - decay) * value)
        micro_state["filtered"][key] = filtered
        filtered_values[key] = filtered

    micro_state["samples"] = int(micro_state.get("samples", 0)) + 1
    trace_entry = {"phase": micro_state.get("phase"), **filtered_values}
    micro_state.setdefault("trace", []).append(trace_entry)
    if len(micro_state["trace"]) > history:
        overflow = len(micro_state["trace"]) - history
        del micro_state["trace"][:overflow]

    micro_state["last_measures"] = dict(measures)

    return {
        "microsector_id": microsector_id,
        "phase": micro_state.get("phase"),
        "filtered": filtered_values,
        "samples": micro_state["samples"],
        "phase_changed": phase_changed,
    }


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

    current_archetype = str(triggers.get("current_archetype", "equilibrio"))
    candidate_archetype = str(triggers.get("candidate_archetype", current_archetype))
    fallback_archetype = str(triggers.get("fallback_archetype", "recuperacion"))

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


def recursividad_operator(
    series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5
) -> List[float]:
    """Apply a recursive filter to a series to capture hysteresis effects."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in the [0, 1) interval")
    if not series:
        return []
    state = seed
    trace: List[float] = []
    for value in series:
        state = (decay * state) + ((1.0 - decay) * value)
        trace.append(state)
    return trace


def _stage_recepcion(
    telemetry_segments: Sequence[Sequence[TelemetryRecord]],
) -> tuple[Dict[str, object], List[TelemetryRecord]]:
    bundles: List[EPIBundle] = []
    lap_indices: List[int] = []
    lap_metadata: List[Dict[str, object]] = []
    flattened_records: List[TelemetryRecord] = []

    for lap_index, segment in enumerate(telemetry_segments):
        segment_records = list(segment)
        flattened_records.extend(segment_records)
        label_value = next(
            (record.lap for record in segment_records if getattr(record, "lap", None) is not None),
            None,
        )
        explicit = label_value is not None
        label = str(label_value) if explicit else f"Vuelta {lap_index + 1}"
        lap_metadata.append(
            {
                "index": lap_index,
                "label": label,
                "value": label_value,
                "explicit": explicit,
            }
        )
        segment_bundles = recepcion_operator(segment_records)
        lap_indices.extend([lap_index] * len(segment_bundles))
        bundles.extend(segment_bundles)

    stage_payload = {
        "bundles": bundles,
        "lap_indices": lap_indices,
        "lap_sequence": lap_metadata,
        "sample_count": len(bundles),
    }
    return stage_payload, flattened_records


def _stage_coherence(
    bundles: Sequence[EPIBundle],
    objectives: Mapping[str, float],
    *,
    coherence_window: int,
    microsectors: Sequence["Microsector"] | None = None,
) -> Dict[str, object]:
    if not bundles:
        empty_breakdown = DissonanceBreakdown(
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
        return {
            "raw_delta": [],
            "raw_sense_index": [],
            "smoothed_delta": [],
            "smoothed_sense_index": [],
            "bundles": [],
            "dissonance": 0.0,
            "dissonance_breakdown": empty_breakdown,
            "coupling": 0.0,
            "resonance": 0.0,
        }

    delta_series = [bundle.delta_nfr for bundle in bundles]
    si_series = [bundle.sense_index for bundle in bundles]
    smoothed_delta = coherence_operator(delta_series, window=coherence_window)
    smoothed_si = coherence_operator(si_series, window=coherence_window)
    clamped_si = [max(0.0, min(1.0, value)) for value in smoothed_si]
    updated_bundles = _update_bundles(bundles, smoothed_delta, clamped_si)
    breakdown = dissonance_breakdown_operator(
        smoothed_delta,
        objectives["delta_nfr"],
        microsectors=microsectors,
        bundles=updated_bundles,
    )
    dissonance = breakdown.value
    coupling = acoplamiento_operator(smoothed_delta, clamped_si)
    resonance = resonance_operator(clamped_si)

    return {
        "raw_delta": delta_series,
        "raw_sense_index": si_series,
        "smoothed_delta": smoothed_delta,
        "smoothed_sense_index": clamped_si,
        "bundles": updated_bundles,
        "dissonance": dissonance,
        "dissonance_breakdown": breakdown,
        "coupling": coupling,
        "resonance": resonance,
    }


def _stage_nodal_metrics(bundles: Sequence[EPIBundle]) -> Dict[str, object]:
    node_pairs = (
        ("tyres", "suspension"),
        ("tyres", "chassis"),
        ("suspension", "chassis"),
    )
    delta_by_node = {
        "tyres": [bundle.tyres.delta_nfr for bundle in bundles],
        "suspension": [bundle.suspension.delta_nfr for bundle in bundles],
        "chassis": [bundle.chassis.delta_nfr for bundle in bundles],
    }
    si_by_node = {
        "tyres": [bundle.tyres.sense_index for bundle in bundles],
        "suspension": [bundle.suspension.sense_index for bundle in bundles],
        "chassis": [bundle.chassis.sense_index for bundle in bundles],
    }
    pairwise_delta = pairwise_coupling_operator(delta_by_node, pairs=node_pairs)
    pairwise_si = pairwise_coupling_operator(si_by_node, pairs=node_pairs)
    return {
        "delta_by_node": delta_by_node,
        "sense_index_by_node": si_by_node,
        "pairwise_coupling": {
            "delta_nfr": pairwise_delta,
            "sense_index": pairwise_si,
        },
    }


def _stage_epi_evolution(
    records: Sequence[TelemetryRecord],
    *,
    phase_assignments: Mapping[int, str] | None = None,
    phase_weight_lookup: Mapping[int, Mapping[str, Mapping[str, float] | float]] | None = None,
    global_phase_weights: Mapping[str, Mapping[str, float] | float] | None = None,
) -> Dict[str, object]:
    if not records:
        return {
            "integrated": [],
            "derivative": [],
            "per_node_integrated": {},
            "per_node_derivative": {},
        }

    integrated_series: List[float] = []
    derivative_series: List[float] = []
    per_node_integrated: Dict[str, List[float]] = {}
    per_node_derivative: Dict[str, List[float]] = {}
    cumulative_by_node: Dict[str, float] = {}

    prev_epi = 0.0
    prev_timestamp = records[0].timestamp

    for index, record in enumerate(records):
        delta_map = delta_nfr_by_node(record)
        phase = phase_assignments.get(index) if phase_assignments else None
        weights = None
        if phase_weight_lookup and index in phase_weight_lookup:
            weights = phase_weight_lookup[index]
        elif global_phase_weights:
            weights = global_phase_weights
        nu_map = resolve_nu_f_by_node(
            record,
            phase=phase,
            phase_weights=weights,
        )
        dt = 0.0 if index == 0 else max(0.0, record.timestamp - prev_timestamp)
        new_epi, derivative, nodal = evolve_epi(prev_epi, delta_map, dt, nu_map)
        integrated_series.append(new_epi)
        derivative_series.append(derivative)
        nodes = set(per_node_integrated) | set(nodal)
        for node in nodes:
            node_integral, node_derivative = nodal.get(node, (0.0, 0.0))
            cumulative = cumulative_by_node.get(node, 0.0) + node_integral
            cumulative_by_node[node] = cumulative
            per_node_integrated.setdefault(node, []).append(cumulative)
            per_node_derivative.setdefault(node, []).append(node_derivative)
        prev_epi = new_epi
        prev_timestamp = record.timestamp

    return {
        "integrated": integrated_series,
        "derivative": derivative_series,
        "per_node_integrated": per_node_integrated,
        "per_node_derivative": per_node_derivative,
    }


def _stage_sense(
    series: Sequence[float], *, recursion_decay: float
) -> Dict[str, object]:
    if not series:
        return {
            "series": [],
            "memory": [],
            "average": 0.0,
            "decay": recursion_decay,
        }

    recursive_trace = recursividad_operator(series, seed=series[0], decay=recursion_decay)
    return {
        "series": list(series),
        "memory": recursive_trace,
        "average": mean(series),
        "decay": recursion_decay,
    }


def _update_bundles(
    bundles: Sequence[EPIBundle],
    delta_series: Sequence[float],
    si_series: Sequence[float],
) -> List[EPIBundle]:
    updated: List[EPIBundle] = []
    for bundle, delta_value, si_value in zip(bundles, delta_series, si_series):
        updated.append(
            replace(
                bundle,
                delta_nfr=delta_value,
                sense_index=max(0.0, min(1.0, si_value)),
            )
        )
    return updated


def _microsector_sample_indices(microsector: "Microsector") -> List[int]:
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


def _phase_context_from_microsectors(
    microsectors: Sequence["Microsector"] | None,
) -> tuple[Dict[int, str], Dict[int, Mapping[str, Mapping[str, float] | float]]]:
    assignments: Dict[int, str] = {}
    weight_lookup: Dict[int, Mapping[str, Mapping[str, float] | float]] = {}
    if not microsectors:
        return assignments, weight_lookup
    for microsector in microsectors:
        raw_weights = getattr(microsector, "phase_weights", {}) or {}
        weight_profile: Dict[str, Mapping[str, float] | float] = {}
        for phase, profile in raw_weights.items():
            if isinstance(profile, Mapping):
                weight_profile[str(phase)] = dict(profile)
            else:
                weight_profile[str(phase)] = float(profile)
        for phase, samples in getattr(microsector, "phase_samples", {}).items():
            if not samples:
                continue
            for sample in samples:
                index = int(sample)
                if index < 0:
                    continue
                assignments[index] = str(phase)
                weight_lookup[index] = weight_profile
    return assignments, weight_lookup


def _variance_payload(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"variance": 0.0, "stdev": 0.0}
    variance = float(pvariance(values))
    if variance < 0.0 and abs(variance) < 1e-12:
        variance = 0.0
    stdev = sqrt(variance) if variance > 0.0 else 0.0
    return {"variance": variance, "stdev": stdev}


def _microsector_variability(
    microsectors: Sequence["Microsector"] | None,
    bundles: Sequence[EPIBundle],
    lap_indices: Sequence[int],
    lap_metadata: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    if not microsectors:
        return []
    bundle_count = len(bundles)
    include_laps = len(lap_metadata) > 1
    variability: List[Dict[str, object]] = []
    for microsector in microsectors:
        sample_indices = [
            idx for idx in _microsector_sample_indices(microsector) if 0 <= idx < bundle_count
        ]
        delta_values = [bundles[idx].delta_nfr for idx in sample_indices]
        si_values = [bundles[idx].sense_index for idx in sample_indices]
        entry: Dict[str, object] = {
            "microsector": microsector.index,
            "label": f"Curva {microsector.index + 1}",
            "overall": {
                "samples": len(sample_indices),
                "delta_nfr": _variance_payload(delta_values),
                "sense_index": _variance_payload(si_values),
            },
        }
        if include_laps and lap_indices:
            lap_payload: Dict[str, Dict[str, object]] = {}
            for lap_entry in lap_metadata:
                lap_index = int(lap_entry.get("index", 0))
                lap_label = str(lap_entry.get("label", lap_index))
                lap_specific_indices = [
                    idx
                    for idx in sample_indices
                    if idx < len(lap_indices) and lap_indices[idx] == lap_index
                ]
                if not lap_specific_indices:
                    continue
                lap_payload[lap_label] = {
                    "samples": len(lap_specific_indices),
                    "delta_nfr": _variance_payload(
                        [bundles[idx].delta_nfr for idx in lap_specific_indices]
                    ),
                    "sense_index": _variance_payload(
                        [bundles[idx].sense_index for idx in lap_specific_indices]
                    ),
                }
            if lap_payload:
                entry["laps"] = lap_payload
        variability.append(entry)
    return variability


def orchestrate_delta_metrics(
    telemetry_segments: Sequence[Sequence[TelemetryRecord]],
    target_delta_nfr: float,
    target_sense_index: float,
    *,
    coherence_window: int = 3,
    recursion_decay: float = 0.4,
    microsectors: Sequence["Microsector"] | None = None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None = None,
) -> Mapping[str, object]:
    """Pipeline orchestration producing aggregated ΔNFR and Si metrics."""

    objectives = emission_operator(target_delta_nfr, target_sense_index)
    reception_stage, flattened_records = _stage_recepcion(telemetry_segments)
    phase_assignments, weight_lookup = _phase_context_from_microsectors(
        microsectors
    )

    if not reception_stage["bundles"]:
        empty_breakdown = DissonanceBreakdown(
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
        stages = {
            "recepcion": reception_stage,
            "coherence": {
                "raw_delta": [],
                "raw_sense_index": [],
                "smoothed_delta": [],
                "smoothed_sense_index": [],
                "bundles": [],
                "dissonance": 0.0,
                "dissonance_breakdown": empty_breakdown,
                "coupling": 0.0,
                "resonance": 0.0,
            },
            "nodal": {
                "delta_by_node": {},
                "sense_index_by_node": {},
                "pairwise_coupling": {"delta_nfr": {}, "sense_index": {}},
            },
            "epi": {
                "integrated": [],
                "derivative": [],
                "per_node_integrated": {},
                "per_node_derivative": {},
            },
            "sense": {
                "series": [],
                "memory": [],
                "average": 0.0,
                "decay": recursion_decay,
            },
        }
        return {
            "objectives": objectives,
            "bundles": [],
            "delta_nfr_series": [],
            "sense_index_series": [],
            "delta_nfr": 0.0,
            "sense_index": 0.0,
            "dissonance": 0.0,
            "dissonance_breakdown": empty_breakdown,
            "coupling": 0.0,
            "resonance": 0.0,
            "recursive_trace": [],
            "lap_sequence": reception_stage["lap_sequence"],
            "microsector_variability": [],
            "pairwise_coupling": {"delta_nfr": {}, "sense_index": {}},
            "nodal_metrics": stages["nodal"],
            "epi_evolution": stages["epi"],
            "sense_memory": stages["sense"],
            "stages": stages,
        }

    coherence_stage = _stage_coherence(
        reception_stage["bundles"],
        objectives,
        coherence_window=coherence_window,
        microsectors=microsectors,
    )
    nodal_stage = _stage_nodal_metrics(coherence_stage["bundles"])
    epi_stage = _stage_epi_evolution(
        flattened_records,
        phase_assignments=phase_assignments,
        phase_weight_lookup=weight_lookup,
        global_phase_weights=phase_weights,
    )
    sense_stage = _stage_sense(
        coherence_stage["smoothed_sense_index"], recursion_decay=recursion_decay
    )
    variability = _microsector_variability(
        microsectors,
        coherence_stage["bundles"],
        reception_stage["lap_indices"],
        reception_stage["lap_sequence"],
    )

    stages = {
        "recepcion": reception_stage,
        "coherence": coherence_stage,
        "nodal": nodal_stage,
        "epi": epi_stage,
        "sense": sense_stage,
    }

    return {
        "objectives": objectives,
        "bundles": coherence_stage["bundles"],
        "delta_nfr_series": coherence_stage["smoothed_delta"],
        "sense_index_series": coherence_stage["smoothed_sense_index"],
        "delta_nfr": mean(coherence_stage["smoothed_delta"])
        if coherence_stage["smoothed_delta"]
        else 0.0,
        "sense_index": mean(coherence_stage["smoothed_sense_index"])
        if coherence_stage["smoothed_sense_index"]
        else 0.0,
        "dissonance": coherence_stage["dissonance"],
        "dissonance_breakdown": coherence_stage["dissonance_breakdown"],
        "coupling": coherence_stage["coupling"],
        "resonance": coherence_stage["resonance"],
        "recursive_trace": sense_stage["memory"],
        "lap_sequence": reception_stage["lap_sequence"],
        "microsector_variability": variability,
        "pairwise_coupling": nodal_stage["pairwise_coupling"],
        "nodal_metrics": nodal_stage,
        "epi_evolution": epi_stage,
        "sense_memory": sense_stage,
        "stages": stages,
    }


__all__ = [
    "emission_operator",
    "recepcion_operator",
    "coherence_operator",
    "dissonance_operator",
    "dissonance_breakdown_operator",
    "DissonanceBreakdown",
    "acoplamiento_operator",
    "pairwise_coupling_operator",
    "resonance_operator",
    "recursivity_operator",
    "mutation_operator",
    "recursividad_operator",
    "orchestrate_delta_metrics",
    "evolve_epi",
]

