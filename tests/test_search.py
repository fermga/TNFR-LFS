from __future__ import annotations

from typing import Mapping, Sequence

from pathlib import Path

import pytest

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
from tnfr_lfs.io.profiles import ProfileManager
from tnfr_lfs.recommender import RecommendationEngine
import tnfr_lfs.recommender.search as search_module
from tnfr_lfs.recommender.search import DEFAULT_DECISION_LIBRARY, SetupPlanner, objective_score


SUPPORTED_CAR_MODELS = [
    "XFG",
    "XRG",
    "RB4",
    "FXO",
    "FXR",
    "XRR",
    "FZR",
    "FO8",
    "BF1",
]


def test_decision_space_includes_parallel_steer_controls() -> None:
    space = DEFAULT_DECISION_LIBRARY["XFG"]
    variables = {variable.name for variable in space.variables}
    assert {"parallel_steer", "steering_lock_deg"}.issubset(variables)


def test_front_wing_limits_defined_per_model() -> None:
    for model, limits in search_module._GTR_FRONT_WING_LIMITS.items():
        space = DEFAULT_DECISION_LIBRARY[model]
        variable = next(var for var in space.variables if var.name == "front_wing_angle")
        assert variable.lower == pytest.approx(limits[0])
        assert variable.upper == pytest.approx(limits[1])
        assert variable.step == pytest.approx(limits[2])

    for model, limits in search_module._FORMULA_FRONT_WING_LIMITS.items():
        space = DEFAULT_DECISION_LIBRARY[model]
        variable = next(var for var in space.variables if var.name == "front_wing_angle")
        assert variable.lower == pytest.approx(limits[0])
        assert variable.upper == pytest.approx(limits[1])
        assert variable.step == pytest.approx(limits[2])


BASE_NU_F = {
    "tyres": 0.18,
    "suspension": 0.14,
    "chassis": 0.12,
    "brakes": 0.16,
    "transmission": 0.11,
    "track": 0.08,
    "driver": 0.05,
}


def _timestamp_delta(results, index: int) -> float:
    if index + 1 < len(results):
        return max(1e-3, results[index + 1].timestamp - results[index].timestamp)
    if index > 0:
        return max(1e-3, results[index].timestamp - results[index - 1].timestamp)
    return 1e-3


def _absolute_integral(results) -> float:
    total = 0.0
    for idx, bundle in enumerate(results):
        total += abs(bundle.delta_nfr) * _timestamp_delta(results, idx)
    return total


def _phase_integrals(results, microsectors) -> dict[str, float]:
    totals: dict[str, float] = {}
    if not microsectors:
        return totals
    for microsector in microsectors:
        for phase, (start, stop) in microsector.phase_boundaries.items():
            subtotal = 0.0
            for idx in range(start, min(stop, len(results))):
                subtotal += abs(results[idx].delta_nfr) * _timestamp_delta(results, idx)
            if subtotal:
                totals[phase] = totals.get(phase, 0.0) + subtotal
    return totals


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


def _rich_bundle(
    timestamp: float,
    *,
    delta_nfr: float = 2.0,
    si: float = 0.85,
    yaw_rate: float = 0.1,
    travel_front: float = 0.04,
    travel_rear: float = 0.04,
    temps: tuple[float, float, float, float] = (82.0, 81.5, 79.5, 79.0),
    mu_front: tuple[float, float] = (1.25, 1.05),
    mu_rear: tuple[float, float] = (1.15, 0.95),
) -> EPIBundle:
    share = delta_nfr / 6
    tyres = TyresNode(
        delta_nfr=share,
        sense_index=si,
        nu_f=BASE_NU_F["tyres"],
        mu_eff_front=mu_front[0],
        mu_eff_rear=mu_rear[0],
        mu_eff_front_lateral=mu_front[0],
        mu_eff_front_longitudinal=mu_front[1],
        mu_eff_rear_lateral=mu_rear[0],
        mu_eff_rear_longitudinal=mu_rear[1],
        tyre_temp_fl=temps[0],
        tyre_temp_fr=temps[1],
        tyre_temp_rl=temps[2],
        tyre_temp_rr=temps[3],
    )
    suspension = SuspensionNode(
        delta_nfr=share,
        sense_index=si,
        nu_f=BASE_NU_F["suspension"],
        travel_front=travel_front,
        travel_rear=travel_rear,
    )
    chassis = ChassisNode(
        delta_nfr=share,
        sense_index=si,
        nu_f=BASE_NU_F["chassis"],
        yaw_rate=yaw_rate,
    )
    brakes = BrakesNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["brakes"])
    transmission = TransmissionNode(
        delta_nfr=share,
        sense_index=si,
        nu_f=BASE_NU_F["transmission"],
        throttle=0.4,
        gear=4,
        speed=140.0,
    )
    track = TrackNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["track"], yaw=0.0)
    driver = DriverNode(
        delta_nfr=share,
        sense_index=si,
        nu_f=BASE_NU_F["driver"],
        steer=0.1,
        throttle=0.5,
        style_index=si,
    )
    return EPIBundle(
        timestamp=timestamp,
        epi=0.0,
        delta_nfr=delta_nfr,
        sense_index=si,
        tyres=tyres,
        suspension=suspension,
        chassis=chassis,
        brakes=brakes,
        transmission=transmission,
        track=track,
        driver=driver,
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
    phase_lag = {"entry": 0.0, "apex": 0.0, "exit": 0.0}
    phase_alignment = {"entry": 1.0, "apex": 1.0, "exit": 1.0}
    phase_synchrony = {"entry": 1.0, "apex": 1.0, "exit": 1.0}
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
                archetype="hairpin",
                description="",
                target_delta_nfr=0.0,
                target_sense_index=0.9,
                nu_f_target=0.2,
                nu_exc_target=0.2,
                rho_target=1.0,
                target_phase_lag=0.0,
                target_phase_alignment=0.9,
                measured_phase_lag=0.0,
                measured_phase_alignment=1.0,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=nodes,
            ),
            Goal(
                phase="apex",
                archetype="hairpin",
                description="",
                target_delta_nfr=0.0,
                target_sense_index=0.9,
                nu_f_target=0.2,
                nu_exc_target=0.2,
                rho_target=1.0,
                target_phase_lag=0.0,
                target_phase_alignment=0.9,
                measured_phase_lag=0.0,
                measured_phase_alignment=1.0,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=nodes,
            ),
            Goal(
                phase="exit",
                archetype="hairpin",
                description="",
                target_delta_nfr=0.0,
                target_sense_index=0.9,
                nu_f_target=0.2,
                nu_exc_target=0.2,
                rho_target=1.0,
                target_phase_lag=0.0,
                target_phase_alignment=0.9,
                measured_phase_lag=0.0,
                measured_phase_alignment=1.0,
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
        phase_lag=phase_lag,
        phase_alignment=phase_alignment,
        phase_synchrony=phase_synchrony,
        filtered_measures=filtered_measures,
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy=window_occupancy,
        operator_events={},
    )


def test_sci_breakdown_matches_score() -> None:
    results = [
        _rich_bundle(0.0),
        _rich_bundle(0.2),
        _rich_bundle(0.4),
        _rich_bundle(0.6),
    ]
    breakdown: dict[str, float] = {}
    score = objective_score(results, breakdown=breakdown)
    assert set(breakdown) == {"sense", "delta", "udr", "bottoming", "aero"}
    assert score == pytest.approx(sum(breakdown.values()), rel=1e-6)
    assert all(value >= 0.0 for value in breakdown.values())


def test_objective_responds_to_component_variations(monkeypatch: pytest.MonkeyPatch) -> None:
    base_series = [
        _rich_bundle(index * 0.2) for index in range(4)
    ]
    score_base = objective_score(base_series)

    low_si_series = [_rich_bundle(index * 0.2, si=0.6) for index in range(4)]
    assert objective_score(low_si_series) < score_base

    high_delta_series = [_rich_bundle(index * 0.2, delta_nfr=12.0) for index in range(4)]
    assert objective_score(high_delta_series) < score_base

    original_udr = search_module.compute_useful_dissonance_stats
    monkeypatch.setattr(
        search_module,
        "compute_useful_dissonance_stats",
        lambda *args, **kwargs: (0, 0, 0.9),
    )
    score_high_udr = objective_score(base_series)
    monkeypatch.setattr(
        search_module,
        "compute_useful_dissonance_stats",
        lambda *args, **kwargs: (0, 0, 0.1),
    )
    score_low_udr = objective_score(base_series)
    monkeypatch.setattr(search_module, "compute_useful_dissonance_stats", original_udr)
    assert score_low_udr < score_high_udr

    bottoming_series = [
        _rich_bundle(index * 0.2, travel_front=0.0, travel_rear=0.0) for index in range(4)
    ]
    assert objective_score(bottoming_series) < score_base

    aero_imbalanced_series = [
        _rich_bundle(index * 0.2, mu_front=(1.7, 1.3), mu_rear=(0.45, 0.35))
        for index in range(4)
    ]
    assert objective_score(aero_imbalanced_series) < score_base

    overheated_series = [
        _rich_bundle(index * 0.2, temps=(110.0, 109.5, 105.0, 104.5))
        for index in range(4)
    ]
    assert objective_score(overheated_series) == score_base


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
    assert score_with_micro == pytest.approx(score_without_micro, rel=1e-6)


@pytest.mark.parametrize("car_model", SUPPORTED_CAR_MODELS)
def test_setup_planner_converges_and_respects_bounds(car_model: str):
    baseline = [
        _build_bundle(0.0, delta_nfr=10.0, si=0.55),
        _build_bundle(0.1, delta_nfr=8.0, si=0.56),
        _build_bundle(0.2, delta_nfr=6.0, si=0.58),
        _build_bundle(0.3, delta_nfr=5.5, si=0.59),
        _build_bundle(0.4, delta_nfr=5.0, si=0.60),
        _build_bundle(0.5, delta_nfr=4.5, si=0.61),
    ]
    microsector = _microsector()

    space = DEFAULT_DECISION_LIBRARY[car_model]
    targets = {
        var.name: var.lower + 0.6 * (var.upper - var.lower)
        for var in space.variables
        if var.upper > var.lower
    }

    def _closeness(vector: Mapping[str, float]) -> float:
        total = 0.0
        count = 0
        for variable in space.variables:
            span = variable.upper - variable.lower
            if span <= 0:
                continue
            target_value = targets.get(variable.name)
            if target_value is None:
                continue
            distance = abs(vector.get(variable.name, 0.0) - target_value)
            normaliser = max(span * 0.5, variable.step)
            closeness = max(0.0, 1.0 - distance / normaliser)
            total += closeness
            count += 1
        return total / max(count, 1)

    def simulator(vector: Mapping[str, float], _: Sequence[EPIBundle]) -> Sequence[EPIBundle]:
        closeness = _closeness(vector)
        balance = vector.get("rear_ride_height", 0.0) - vector.get("front_ride_height", 0.0)
        aero = vector.get("rear_wing_angle", 0.0)
        diff_bias = vector.get("diff_power_lock", 0.0) * 0.01
        scale = max(0.2, 1.0 - 0.05 * abs(balance))
        si_gain = 0.03 * closeness + 0.01 * (aero + diff_bias)
        adjusted: list[EPIBundle] = []
        for bundle in baseline:
            delta = bundle.delta_nfr * (1.0 - 0.35 * closeness) - 0.25 * aero
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
    plan = planner.plan(baseline, [microsector], car_model=car_model, simulator=simulator)

    for variable in space.variables:
        value = plan.decision_vector[variable.name]
        assert variable.lower <= value <= variable.upper

    initial_vector = space.initial_guess()
    assert any(
        abs(plan.decision_vector[variable.name] - initial_vector[variable.name])
        >= variable.step - 1e-9
        for variable in space.variables
    )
    assert _closeness(plan.decision_vector) > _closeness(initial_vector)

    baseline_score = objective_score(baseline, [microsector])
    assert plan.sci > baseline_score
    assert plan.recommendations  # ensure explainable rules are still available
    assert plan.sensitivities  # gradients are reported
    assert "delta_nfr_integral" in plan.sensitivities
    assert plan.phase_sensitivities


def test_setup_planner_reports_consistent_sensitivities():
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

    mean_si = sum(bundle.sense_index for bundle in plan.telemetry) / len(plan.telemetry)
    sci_score = objective_score(plan.telemetry, [microsector])
    base_integral = _absolute_integral(plan.telemetry)
    base_phase = _phase_integrals(plan.telemetry, [microsector])

    space = DEFAULT_DECISION_LIBRARY["generic_gt"]
    for variable in space.variables:
        base_value = plan.decision_vector[variable.name]
        raw_step = max(variable.step * 0.25, 1e-3)
        forward_room = variable.upper - base_value
        backward_room = base_value - variable.lower
        central_step = min(raw_step, forward_room, backward_room)

        if central_step > 1e-9:
            plus_value = base_value + central_step
            minus_value = base_value - central_step
            plus_vector = dict(plan.decision_vector)
            minus_vector = dict(plan.decision_vector)
            plus_vector[variable.name] = plus_value
            minus_vector[variable.name] = minus_value
            plus_results = simulator(space.clamp(plus_vector), baseline)
            minus_results = simulator(space.clamp(minus_vector), baseline)
            si_plus = sum(bundle.sense_index for bundle in plus_results) / len(plus_results)
            si_minus = sum(bundle.sense_index for bundle in minus_results) / len(minus_results)
            sci_plus = objective_score(plus_results, [microsector])
            sci_minus = objective_score(minus_results, [microsector])
            integral_plus = _absolute_integral(plus_results)
            integral_minus = _absolute_integral(minus_results)
            phase_plus = _phase_integrals(plus_results, [microsector])
            phase_minus = _phase_integrals(minus_results, [microsector])
            denom = 2.0 * central_step
            expected_si = (si_plus - si_minus) / denom
            expected_sci = (sci_plus - sci_minus) / denom
            expected_integral = (integral_plus - integral_minus) / denom
            expected_phase = {
                phase: (phase_plus.get(phase, 0.0) - phase_minus.get(phase, 0.0)) / denom
                for phase in set(phase_plus) | set(phase_minus)
            }
        elif forward_room > 1e-9:
            step = min(raw_step, forward_room)
            plus_vector = dict(plan.decision_vector)
            plus_vector[variable.name] = base_value + step
            plus_results = simulator(space.clamp(plus_vector), baseline)
            si_plus = sum(bundle.sense_index for bundle in plus_results) / len(plus_results)
            sci_plus = objective_score(plus_results, [microsector])
            expected_si = (si_plus - mean_si) / step
            expected_sci = (sci_plus - sci_score) / step
            integral_plus = _absolute_integral(plus_results)
            phase_plus = _phase_integrals(plus_results, [microsector])
            expected_integral = (integral_plus - base_integral) / step
            expected_phase = {
                phase: (phase_plus.get(phase, 0.0) - base_phase.get(phase, 0.0)) / step
                for phase in set(phase_plus) | set(base_phase)
            }
        elif backward_room > 1e-9:
            step = min(raw_step, backward_room)
            minus_vector = dict(plan.decision_vector)
            minus_vector[variable.name] = base_value - step
            minus_results = simulator(space.clamp(minus_vector), baseline)
            si_minus = sum(bundle.sense_index for bundle in minus_results) / len(minus_results)
            sci_minus = objective_score(minus_results, [microsector])
            expected_si = (mean_si - si_minus) / step
            expected_sci = (sci_score - sci_minus) / step
            integral_minus = _absolute_integral(minus_results)
            phase_minus = _phase_integrals(minus_results, [microsector])
            expected_integral = (base_integral - integral_minus) / step
            expected_phase = {
                phase: (base_phase.get(phase, 0.0) - phase_minus.get(phase, 0.0)) / step
                for phase in set(phase_minus) | set(base_phase)
            }
        else:
            expected_si = 0.0
            expected_sci = 0.0
            expected_integral = 0.0
            expected_phase = {}

        reported_si = plan.sensitivities["sense_index"][variable.name]
        reported_sci = plan.sensitivities["sci"][variable.name]
        reported_integral = plan.sensitivities["delta_nfr_integral"][variable.name]
        assert reported_si == pytest.approx(expected_si, rel=1e-2, abs=1e-4)
        assert reported_sci == pytest.approx(expected_sci, rel=1e-2, abs=1e-4)
        assert reported_integral == pytest.approx(expected_integral, rel=1e-2, abs=1e-4)
        for phase, gradient in expected_phase.items():
            reported_phase = (
                plan.phase_sensitivities.get(phase, {})
                .get("delta_nfr_integral", {})
                .get(variable.name, 0.0)
            )
            assert reported_phase == pytest.approx(gradient, rel=1e-2, abs=1e-4)


def test_setup_planner_rejects_unknown_car_model():
    baseline = [
        _build_bundle(0.0, delta_nfr=5.0, si=0.6),
        _build_bundle(0.1, delta_nfr=4.8, si=0.61),
    ]
    planner = SetupPlanner()
    with pytest.raises(ValueError):
        planner.plan(baseline, (), car_model="UNKNOWN")


def test_setup_planner_consults_profile_jacobian(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.toml"
    manager = ProfileManager(profiles_path)
    car_model = "generic_gt"
    track = "generic"
    manager.resolve(car_model, track)
    manager.register_plan(
        car_model,
        track,
        {"entry": 1.0},
        baseline_metrics=(0.6, 4.0),
        jacobian={
            "sense_index": {
                "rear_wing_angle": 1.6,
                "front_camber_deg": 0.1,
            },
            "delta_nfr_integral": {"rear_wing_angle": -0.9},
        },
        phase_jacobian={"entry": {"delta_nfr_integral": {"rear_wing_angle": -0.5}}},
    )
    manager.register_result(car_model, track, sense_index=0.7, delta_nfr=3.2)
    stored_overall, _ = manager.gradient_history(car_model, track)
    assert stored_overall["sense_index"]["rear_wing_angle"] == pytest.approx(1.6)
    assert stored_overall["delta_nfr_integral"]["rear_wing_angle"] == pytest.approx(-0.9)

    engine = RecommendationEngine(
        car_model=car_model, track_name=track, profile_manager=manager
    )

    class RecordingOptimiser:
        def __init__(self) -> None:
            self.seen_order: tuple[str, ...] = ()
            self.step_by_param: dict[str, float] = {}

        def optimise(self, objective, space, initial_vector=None):
            self.seen_order = tuple(var.name for var in space.variables)
            self.step_by_param = {var.name: var.step for var in space.variables}
            vector = space.clamp(initial_vector or space.initial_guess())
            return vector, objective(vector), 0, 1

    optimiser = RecordingOptimiser()
    planner = SetupPlanner(recommendation_engine=engine, optimiser=optimiser)
    baseline = [
        _build_bundle(0.0, delta_nfr=5.0, si=0.6),
        _build_bundle(0.1, delta_nfr=4.9, si=0.61),
        _build_bundle(0.2, delta_nfr=4.7, si=0.62),
    ]

    planner.plan(baseline, (), car_model=car_model, track_name=track)

    assert optimiser.seen_order[0] == "rear_wing_angle"
    original_space = DEFAULT_DECISION_LIBRARY[car_model]
    original_step = next(
        var.step for var in original_space.variables if var.name == "rear_wing_angle"
    )
    assert optimiser.step_by_param["rear_wing_angle"] < original_step


def test_decision_spaces_include_gearing_sliders():
    for car_model in SUPPORTED_CAR_MODELS:
        space = DEFAULT_DECISION_LIBRARY[car_model]
        variable_names = {variable.name for variable in space.variables}
        assert "final_drive_ratio" in variable_names
        for gear in range(1, 8):
            assert f"gear_{gear}_ratio" in variable_names
