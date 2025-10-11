"""Helper builders for recommender tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Tuple

from tnfr_lfs.core.epi_models import EPIBundle
from tnfr_lfs.core.segmentation import Goal, Microsector

from .epi import build_axis_bundle
from .microsector import build_goal, build_microsector


__all__ = (
    "_brake_headroom_microsector",
    "_udr_goal",
    "_udr_microsector",
    "_entry_goal_with_gradient",
    "_entry_microsector_with_gradient",
    "_entry_results_with_gradient",
)


def _brake_headroom_microsector(
    index: int,
    headroom: float,
    *,
    abs_activation: float,
    partial: float,
    sustained: float,
    peak: float,
    brake_event: bool = True,
) -> Microsector:
    return build_microsector(
        index=index,
        start_time=0.0,
        end_time=0.4,
        curvature=1.0,
        brake_event=brake_event,
        support_event=False,
        delta_nfr_signature=0.0,
        goals=(),
        phases=(),
        phase_boundaries={},
        phase_samples={},
        active_phase="entry",
        dominant_nodes={},
        phase_weights={},
        phase_lag={},
        phase_alignment={},
        phase_synchrony={},
        filtered_measures={
            "brake_headroom": headroom,
            "brake_headroom_peak_decel": peak,
            "brake_headroom_abs_activation": abs_activation,
            "brake_headroom_partial_locking": partial,
            "brake_headroom_sustained_locking": sustained,
        },
        window_occupancy={},
        include_cphi=False,
        context_factors={},
        sample_context_factors={},
    )


def _udr_goal(phase: str = "apex", target_delta: float = 0.2) -> Goal:
    return build_goal(
        phase,
        target_delta,
        archetype="hairpin",
        description="",
        target_sense_index=0.85,
        nu_f_target=0.3,
        nu_exc_target=0.25,
        rho_target=0.8,
        target_phase_alignment=0.9,
        measured_phase_alignment=0.9,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.3, 0.3),
        dominant_nodes=("suspension", "chassis"),
        detune_ratio_weights={"longitudinal": 0.7, "lateral": 0.3},
    )


def _udr_microsector(
    goal: Goal,
    *,
    udr: float,
    sample_count: int,
    filtered_measures: Mapping[str, float] | None = None,
    operator_events: Mapping[str, Tuple[Mapping[str, object], ...]] | None = None,
    index: int = 7,
) -> Microsector:
    samples = tuple(range(sample_count))
    boundary = (0, sample_count)
    measures = {"udr": udr}
    if filtered_measures:
        measures.update({k: float(v) for k, v in filtered_measures.items()})
    phase = goal.phase
    return build_microsector(
        index=index,
        start_time=0.0,
        end_time=sample_count * 0.1,
        curvature=1.5,
        brake_event=False,
        support_event=True,
        delta_nfr_signature=goal.target_delta_nfr + 0.4,
        goals=(goal,),
        phases=(phase,),
        phase_boundaries={phase: boundary},
        phase_samples={phase: samples},
        active_phase=phase,
        dominant_nodes={phase: goal.dominant_nodes},
        phase_weights={phase: {"__default__": 1.0}},
        phase_lag={phase: goal.measured_phase_lag},
        phase_alignment={phase: goal.measured_phase_alignment},
        phase_synchrony={phase: goal.measured_phase_synchrony},
        filtered_measures=measures,
        window_occupancy={phase: {}},
        operator_events=operator_events or {},
        include_cphi=False,
    )


def _entry_goal_with_gradient(gradient: float) -> Goal:
    return build_goal(
        "entry1",
        0.2,
        archetype="braking",
        description="",
        nu_f_target=0.25,
        nu_exc_target=0.2,
        rho_target=1.0,
        measured_phase_lag=0.25,
        measured_phase_alignment=0.86,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.4, 0.4),
        yaw_rate_window=(-0.5, 0.5),
        dominant_nodes=("brakes",),
        target_delta_nfr_long=0.5,
        target_delta_nfr_lat=0.1,
        delta_axis_weights={"longitudinal": 0.75, "lateral": 0.25},
        track_gradient=gradient,
    )


def _entry_microsector_with_gradient(goal: Goal, gradient: float) -> Microsector:
    return build_microsector(
        index=3,
        start_time=0.0,
        end_time=0.45,
        curvature=1.2,
        brake_event=True,
        support_event=False,
        delta_nfr_signature=0.6,
        phases=(goal.phase,),
        goals=(goal,),
        phase_boundaries={goal.phase: (0, 3)},
        phase_samples={goal.phase: (0, 1, 2)},
        active_phase=goal.phase,
        dominant_nodes={goal.phase: goal.dominant_nodes},
        phase_weights={goal.phase: {"__default__": 1.0}},
        phase_lag={goal.phase: goal.measured_phase_lag},
        phase_alignment={goal.phase: goal.measured_phase_alignment},
        filtered_measures={"gradient": gradient},
        window_occupancy={goal.phase: {}},
        operator_events={},
        include_cphi=False,
    )


def _entry_results_with_gradient(gradient: float) -> Sequence[EPIBundle]:
    return [
        build_axis_bundle(
            delta_nfr=0.62,
            long_component=0.5,
            lat_component=0.1,
            gradient=gradient,
        ),
        build_axis_bundle(
            delta_nfr=0.6,
            long_component=0.5,
            lat_component=0.1,
            gradient=gradient,
        ),
        build_axis_bundle(
            delta_nfr=0.58,
            long_component=0.5,
            lat_component=0.1,
            gradient=gradient,
        ),
    ]

