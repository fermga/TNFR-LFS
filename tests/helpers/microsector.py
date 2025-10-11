"""Factories for building goal and microsector fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tnfr_lfs.core import Goal, Microsector
from tnfr_lfs.core.constants import WHEEL_SUFFIXES
from tnfr_lfs.core.phases import PHASE_SEQUENCE, expand_phase_alias


def build_goal(
    phase: str,
    target_delta_nfr: float = 0.0,
    *,
    archetype: str = "medium",
    description: str | None = None,
    target_sense_index: float = 0.9,
    nu_f_target: float = 0.0,
    nu_exc_target: float = 0.0,
    rho_target: float = 0.0,
    target_phase_lag: float = 0.0,
    target_phase_alignment: float = 0.9,
    measured_phase_lag: float = 0.0,
    measured_phase_alignment: float = 1.0,
    slip_lat_window: tuple[float, float] = (-0.5, 0.5),
    slip_long_window: tuple[float, float] = (-0.5, 0.5),
    yaw_rate_window: tuple[float, float] = (-0.5, 0.5),
    dominant_nodes: Sequence[str] = ("tyres",),
    **overrides: Any,
) -> Goal:
    """Construct a :class:`~tnfr_lfs.core.Goal` with convenient defaults."""

    aliases = expand_phase_alias(phase)
    canonical_phase = aliases[-1] if aliases else phase
    payload: dict[str, Any] = {
        "phase": canonical_phase,
        "archetype": archetype,
        "description": description or f"Synthetic goal for {canonical_phase}",
        "target_delta_nfr": target_delta_nfr,
        "target_sense_index": target_sense_index,
        "nu_f_target": nu_f_target,
        "nu_exc_target": nu_exc_target,
        "rho_target": rho_target,
        "target_phase_lag": target_phase_lag,
        "target_phase_alignment": target_phase_alignment,
        "measured_phase_lag": measured_phase_lag,
        "measured_phase_alignment": measured_phase_alignment,
        "slip_lat_window": slip_lat_window,
        "slip_long_window": slip_long_window,
        "yaw_rate_window": yaw_rate_window,
        "dominant_nodes": tuple(dominant_nodes),
    }
    payload.update(overrides)
    return Goal(**payload)


def build_microsector(
    *,
    index: int = 0,
    entry_index: int = 0,
    apex_index: int = 0,
    exit_index: int = 0,
    start_time: float | None = None,
    end_time: float | None = None,
    curvature: float = 1.0,
    brake_event: bool = False,
    support_event: bool = True,
    delta_nfr_signature: float = 0.0,
    phases: Sequence[str] | None = None,
    goals: Sequence[Goal] | None = None,
    target_map: Mapping[str, float] | None = None,
    archetype: str = "hairpin",
    dominant_nodes: Mapping[str, Sequence[str]] | None = None,
    phase_weights: Mapping[str, Mapping[str, float] | float] | None = None,
    phase_samples: Mapping[str, Sequence[int]] | None = None,
    phase_boundaries: Mapping[str, tuple[int, int]] | None = None,
    phase_lag: Mapping[str, float] | None = None,
    phase_alignment: Mapping[str, float] | None = None,
    phase_synchrony: Mapping[str, float] | None = None,
    filtered_measures: Mapping[str, Any] | None = None,
    window_occupancy: Mapping[str, Mapping[str, float]] | None = None,
    operator_events: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    apex_target: float = 0.0,
    cphi_values: Mapping[str, float] | None = None,
    active_phase: str = "apex",
    recursivity_trace: Sequence[Mapping[str, float | str | None]] = (),
    last_mutation: Mapping[str, Any] | None = None,
    include_cphi: bool = True,
    **overrides: Any,
) -> Microsector:
    """Construct a :class:`~tnfr_lfs.core.Microsector` with reusable defaults."""

    selected_phases: tuple[str, ...]
    if phases is None:
        selected_phases = tuple(PHASE_SEQUENCE)
    else:
        selected_phases = tuple(phases)

    if goals is None:
        computed_target_map: dict[str, float] = {phase: 0.0 for phase in selected_phases}
        if "apex3b" in PHASE_SEQUENCE and "apex3b" not in computed_target_map:
            computed_target_map["apex3b"] = apex_target
        elif "apex3b" in computed_target_map:
            computed_target_map["apex3b"] = apex_target
        if target_map:
            computed_target_map.update({str(k): float(v) for k, v in target_map.items()})
        goals = tuple(
            build_goal(phase, computed_target_map.get(phase, 0.0), archetype=archetype)
            for phase in selected_phases
        )
    else:
        goals = tuple(goals)

    if phase_boundaries is None:
        if selected_phases and set(selected_phases).issubset(set(PHASE_SEQUENCE)):
            boundary_start = {
                "entry1": entry_index,
                "entry2": entry_index + 1,
                "apex3a": apex_index,
                "apex3b": apex_index + 1,
                "exit4": exit_index,
            }
            phase_boundaries = {
                phase: (
                    int(boundary_start.get(phase, entry_index)),
                    int(boundary_start.get(phase, entry_index)) + 1,
                )
                for phase in selected_phases
            }
        else:
            phase_boundaries = {
                phase: (int(entry_index + offset), int(entry_index + offset + 1))
                for offset, phase in enumerate(selected_phases)
            }
    else:
        phase_boundaries = {
            str(phase): (int(bounds[0]), int(bounds[1]))
            for phase, bounds in phase_boundaries.items()
        }

    if phase_samples is None:
        phase_samples = {
            phase: (phase_boundaries[phase][0],)
            for phase in selected_phases
            if phase in phase_boundaries
        }
    else:
        phase_samples = {
            str(phase): tuple(int(sample) for sample in samples)
            for phase, samples in phase_samples.items()
        }

    if dominant_nodes is None:
        dominant_nodes_payload = {
            phase: ("tyres",) for phase in selected_phases
        }
    else:
        dominant_nodes_payload = {
            str(phase): tuple(nodes) for phase, nodes in dominant_nodes.items()
        }

    if phase_weights is None:
        phase_weights_payload: dict[str, Mapping[str, float] | float] = {
            phase: {} for phase in selected_phases
        }
    else:
        phase_weights_payload = {str(phase): weights for phase, weights in phase_weights.items()}

    if phase_lag is None:
        phase_lag_payload = {phase: 0.0 for phase in selected_phases}
    else:
        phase_lag_payload = {str(phase): float(value) for phase, value in phase_lag.items()}

    if phase_alignment is None:
        phase_alignment_payload = {phase: 1.0 for phase in selected_phases}
    else:
        phase_alignment_payload = {
            str(phase): float(value) for phase, value in phase_alignment.items()
        }

    phase_synchrony_payload = {phase: 1.0 for phase in selected_phases}
    if phase_synchrony:
        phase_synchrony_payload.update({str(k): float(v) for k, v in phase_synchrony.items()})

    filtered_measures_payload: dict[str, Any] = {
        "thermal_load": 5000.0,
        "style_index": 0.9,
        "grip_rel": 1.0,
    }
    if include_cphi:
        cphi_mapping = {suffix: 0.7 for suffix in WHEEL_SUFFIXES}
        if cphi_values:
            cphi_mapping.update({str(k): float(v) for k, v in cphi_values.items()})
        filtered_measures_payload.update(
            {
                "cphi": {
                    "wheels": {
                        suffix: {"value": value, "components": {}}
                        for suffix, value in cphi_mapping.items()
                    },
                    "thresholds": {"red": 0.4, "amber": 0.6, "green": 0.8},
                }
            }
        )
        for suffix, value in cphi_mapping.items():
            filtered_measures_payload[f"cphi_{suffix}"] = value
    if filtered_measures:
        filtered_measures_payload.update(filtered_measures)

    if window_occupancy is None:
        window_occupancy_payload = {
            phase: {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0}
            for phase in selected_phases
        }
    else:
        window_occupancy_payload = {
            str(phase): {str(k): float(v) for k, v in measures.items()}
            for phase, measures in window_occupancy.items()
        }

    if operator_events is None:
        operator_events_payload: dict[str, tuple[Mapping[str, Any], ...]] = {}
    else:
        operator_events_payload = {
            str(name): tuple(event for event in events)
            for name, events in operator_events.items()
        }

    payload: dict[str, Any] = {
        "index": index,
        "start_time": float(entry_index if start_time is None else start_time),
        "end_time": float(exit_index if end_time is None else end_time),
        "curvature": curvature,
        "brake_event": brake_event,
        "support_event": support_event,
        "delta_nfr_signature": delta_nfr_signature,
        "goals": tuple(goals),
        "phase_boundaries": phase_boundaries,
        "phase_samples": phase_samples,
        "active_phase": active_phase,
        "dominant_nodes": dominant_nodes_payload,
        "phase_weights": phase_weights_payload,
        "grip_rel": 1.0,
        "phase_lag": phase_lag_payload,
        "phase_alignment": phase_alignment_payload,
        "phase_synchrony": phase_synchrony_payload,
        "filtered_measures": filtered_measures_payload,
        "recursivity_trace": tuple(recursivity_trace),
        "last_mutation": last_mutation,
        "window_occupancy": window_occupancy_payload,
        "operator_events": operator_events_payload,
    }
    payload.update(overrides)
    return Microsector(**payload)
