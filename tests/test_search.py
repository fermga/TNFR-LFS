from __future__ import annotations

from typing import Mapping, Sequence

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
from tnfr_lfs.core.segmentation import Goal, Microsector
from tnfr_lfs.recommender.search import DEFAULT_DECISION_LIBRARY, SetupPlanner, objective_score


BASE_NU_F = {
    "tyres": 0.18,
    "suspension": 0.14,
    "chassis": 0.12,
    "brakes": 0.16,
    "transmission": 0.11,
    "track": 0.08,
    "driver": 0.05,
}


def _build_bundle(timestamp: float, delta_nfr: float, si: float) -> EPIBundle:
    share = delta_nfr / 6
    tyre_node = TyresNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["tyres"])
    return EPIBundle(
        timestamp=timestamp,
        epi=0.0,
        delta_nfr=delta_nfr,
        sense_index=si,
        tyres=tyre_node,
        suspension=SuspensionNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["suspension"]),
        chassis=ChassisNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["chassis"]),
        brakes=BrakesNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["brakes"]),
        transmission=TransmissionNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["transmission"]),
        track=TrackNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["track"]),
        driver=DriverNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["driver"]),
    )


def _microsector() -> Microsector:
    window = (-0.2, 0.2)
    yaw_window = (-0.5, 0.5)
    nodes = ("tyres", "suspension")
    phase_samples = {
        "entry": (0, 1),
        "apex": (2, 3),
        "exit": (4, 5),
    }
    phase_weights = {
        "entry": {"__default__": 1.0},
        "apex": {"__default__": 1.0},
        "exit": {"__default__": 1.0},
    }
    window_occupancy = {
        "entry": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
        "apex": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
        "exit": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
    }
    filtered_measures = {
        "thermal_load": 5000.0,
        "style_index": 0.9,
        "grip_rel": 1.0,
    }
    return Microsector(
        index=0,
        start_time=0.0,
        end_time=0.3,
        curvature=1.5,
        brake_event=True,
        support_event=True,
        delta_nfr_signature=2.0,
        goals=(
            Goal(
                phase="entry",
                archetype="apoyo",
                description="",
                target_delta_nfr=0.0,
                target_sense_index=0.9,
                nu_f_target=0.2,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=nodes,
            ),
            Goal(
                phase="apex",
                archetype="apoyo",
                description="",
                target_delta_nfr=0.0,
                target_sense_index=0.9,
                nu_f_target=0.2,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=nodes,
            ),
            Goal(
                phase="exit",
                archetype="apoyo",
                description="",
                target_delta_nfr=0.0,
                target_sense_index=0.9,
                nu_f_target=0.2,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=nodes,
            ),
        ),
        phase_boundaries={"entry": (0, 2), "apex": (2, 4), "exit": (4, 6)},
        phase_samples=phase_samples,
        active_phase="entry",
        dominant_nodes={
            "entry": nodes,
            "apex": nodes,
            "exit": nodes,
        },
        phase_weights=phase_weights,
        grip_rel=1.0,
        filtered_measures=filtered_measures,
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy=window_occupancy,
    )


def test_objective_penalises_delta_nfr_integral():
    results = [
        _build_bundle(0.0, delta_nfr=8.0, si=0.6),
        _build_bundle(0.1, delta_nfr=6.0, si=0.62),
        _build_bundle(0.2, delta_nfr=-5.0, si=0.65),
        _build_bundle(0.3, delta_nfr=-4.0, si=0.66),
        _build_bundle(0.4, delta_nfr=3.0, si=0.68),
        _build_bundle(0.5, delta_nfr=2.0, si=0.69),
    ]
    microsector = _microsector()
    score_with_micro = objective_score(results, [microsector])
    score_without_micro = objective_score(results, [])
    assert score_with_micro < score_without_micro


def test_setup_planner_converges_and_respects_bounds():
    baseline = [
        _build_bundle(0.0, delta_nfr=10.0, si=0.55),
        _build_bundle(0.1, delta_nfr=8.0, si=0.56),
        _build_bundle(0.2, delta_nfr=6.0, si=0.58),
        _build_bundle(0.3, delta_nfr=5.5, si=0.59),
        _build_bundle(0.4, delta_nfr=5.0, si=0.60),
        _build_bundle(0.5, delta_nfr=4.5, si=0.61),
    ]
    microsector = _microsector()

    def simulator(vector: Mapping[str, float], _: Sequence[EPIBundle]) -> Sequence[EPIBundle]:
        rear = vector["rear_ride_height"]
        front = vector["front_ride_height"]
        wing = vector["rear_wing_angle"]
        scale = 1.0 - 0.05 * abs(rear - front)
        si_gain = 0.015 * (rear + wing)
        adjusted: list[EPIBundle] = []
        for bundle in baseline:
            delta = bundle.delta_nfr - 1.2 * rear - 0.5 * wing
            sense = min(1.0, bundle.sense_index + si_gain)
            adjusted.append(
                EPIBundle(
                    timestamp=bundle.timestamp,
                    epi=bundle.epi,
                    delta_nfr=delta * scale,
                    sense_index=sense,
                    tyres=bundle.tyres,
                    suspension=bundle.suspension,
                    chassis=bundle.chassis,
                    brakes=bundle.brakes,
                    transmission=bundle.transmission,
                    track=bundle.track,
                    driver=bundle.driver,
                )
            )
        return adjusted

    planner = SetupPlanner()
    plan = planner.plan(baseline, [microsector], car_model="generic_gt", simulator=simulator)

    space = DEFAULT_DECISION_LIBRARY["generic_gt"]
    for variable in space.variables:
        value = plan.decision_vector[variable.name]
        assert variable.lower <= value <= variable.upper

    baseline_score = objective_score(baseline, [microsector])
    assert plan.objective_value > baseline_score
    assert plan.recommendations  # ensure explainable rules are still available
