from pathlib import Path

import pytest

from typing import Sequence, Tuple
from dataclasses import replace
from types import SimpleNamespace

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
from collections.abc import Mapping

from tnfr_lfs.core.segmentation import Goal, Microsector
from tnfr_lfs.io.profiles import AeroProfile, ProfileManager
from tnfr_lfs.recommender.rules import (
    AeroCoherenceRule,
    BottomingPriorityRule,
    DetuneRatioRule,
    MANUAL_REFERENCES,
    PhaseDeltaDeviationRule,
    PhaseNodeOperatorRule,
    ParallelSteerRule,
    Recommendation,
    RecommendationEngine,
    RuleContext,
    TyreBalanceRule,
    UsefulDissonanceRule,
    ThresholdProfile,
)


BASE_NU_F = {
    "tyres": 0.18,
    "suspension": 0.14,
    "chassis": 0.12,
    "brakes": 0.16,
    "transmission": 0.11,
    "track": 0.08,
    "driver": 0.05,
}


def _udr_bundle_series(values: Sequence[float], *, si: float = 0.8) -> Sequence[EPIBundle]:
    bundles: list[EPIBundle] = []
    for index, value in enumerate(values):
        nodes = dict(
            tyres=TyresNode(delta_nfr=value, sense_index=si, nu_f=BASE_NU_F["tyres"]),
            suspension=SuspensionNode(
                delta_nfr=value,
                sense_index=si,
                nu_f=BASE_NU_F["suspension"],
            ),
            chassis=ChassisNode(
                delta_nfr=value,
                sense_index=si,
                nu_f=BASE_NU_F["chassis"],
                yaw_rate=0.0,
            ),
            brakes=BrakesNode(delta_nfr=value, sense_index=si, nu_f=BASE_NU_F["brakes"]),
            transmission=TransmissionNode(
                delta_nfr=value,
                sense_index=si,
                nu_f=BASE_NU_F["transmission"],
            ),
            track=TrackNode(delta_nfr=value, sense_index=si, nu_f=BASE_NU_F["track"]),
            driver=DriverNode(delta_nfr=value, sense_index=si, nu_f=BASE_NU_F["driver"]),
        )
        bundles.append(
            EPIBundle(
                timestamp=index * 0.1,
                epi=0.0,
                delta_nfr=value,
                delta_nfr_longitudinal=value,
                delta_nfr_lateral=0.0,
                sense_index=si,
                **nodes,
            )
        )
    return bundles


def _axis_bundle(
    delta_nfr: float,
    long_component: float,
    lat_component: float,
    *,
    si: float = 0.8,
    gradient: float = 0.0,
) -> EPIBundle:
    share = delta_nfr / 7.0
    nodes = dict(
        tyres=TyresNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["tyres"]),
        suspension=SuspensionNode(
            delta_nfr=share,
            sense_index=si,
            nu_f=BASE_NU_F["suspension"],
        ),
        chassis=ChassisNode(
            delta_nfr=share,
            sense_index=si,
            nu_f=BASE_NU_F["chassis"],
        ),
        brakes=BrakesNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["brakes"]),
        transmission=TransmissionNode(
            delta_nfr=share,
            sense_index=si,
            nu_f=BASE_NU_F["transmission"],
        ),
        track=TrackNode(
            delta_nfr=share,
            sense_index=si,
            nu_f=BASE_NU_F["track"],
            gradient=gradient,
        ),
        driver=DriverNode(delta_nfr=share, sense_index=si, nu_f=BASE_NU_F["driver"]),
    )
    return EPIBundle(
        timestamp=0.0,
        epi=0.0,
        delta_nfr=delta_nfr,
        delta_nfr_longitudinal=long_component,
        delta_nfr_lateral=lat_component,
        sense_index=si,
        **nodes,
    )


def _udr_goal(phase: str = "apex", target_delta: float = 0.2) -> Goal:
    return Goal(
        phase=phase,
        archetype="hairpin",
        description="",
        target_delta_nfr=target_delta,
        target_sense_index=0.85,
        nu_f_target=0.3,
        nu_exc_target=0.25,
        rho_target=0.8,
        target_phase_lag=0.0,
        target_phase_alignment=0.9,
        measured_phase_lag=0.0,
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
    return Microsector(
        index=index,
        start_time=0.0,
        end_time=sample_count * 0.1,
        curvature=1.5,
        brake_event=False,
        support_event=True,
        delta_nfr_signature=goal.target_delta_nfr + 0.4,
        goals=(goal,),
        phase_boundaries={goal.phase: boundary},
        phase_samples={goal.phase: samples},
        active_phase=goal.phase,
        dominant_nodes={goal.phase: goal.dominant_nodes},
        phase_weights={goal.phase: {"__default__": 1.0}},
        grip_rel=1.0,
        phase_lag={goal.phase: goal.measured_phase_lag},
        phase_alignment={goal.phase: goal.measured_phase_alignment},
        filtered_measures=measures,
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={goal.phase: {}},
        operator_events=operator_events or {},
    )


@pytest.fixture
def bottoming_microsectors() -> Tuple[Microsector, Microsector]:
    goal = _udr_goal("entry", 0.25)
    smooth = _udr_microsector(
        goal,
        udr=0.32,
        sample_count=5,
        filtered_measures={
            "bottoming_ratio_front": 0.62,
            "bottoming_ratio_rear": 0.14,
        },
        index=2,
    )
    rough = _udr_microsector(
        goal,
        udr=0.34,
        sample_count=5,
        filtered_measures={
            "bottoming_ratio_front": 0.18,
            "bottoming_ratio_rear": 0.71,
        },
        index=3,
    )
    smooth = replace(smooth, context_factors={"surface": 0.94})
    rough = replace(rough, context_factors={"surface": 1.28})
    return smooth, rough


def test_tyre_balance_rule_generates_guidance():
    goal = Goal(
        phase="apex",
        archetype="medium",
        description="",
        target_delta_nfr=0.4,
        target_sense_index=0.9,
        nu_f_target=0.25,
        nu_exc_target=0.25,
        rho_target=1.0,
        target_phase_lag=0.0,
        target_phase_alignment=0.9,
        measured_phase_lag=0.0,
        measured_phase_alignment=0.88,
        slip_lat_window=(-0.4, 0.4),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.5, 0.5),
        dominant_nodes=("tyres",),
    )
    microsector = Microsector(
        index=5,
        start_time=0.0,
        end_time=0.4,
        curvature=1.2,
        brake_event=False,
        support_event=False,
        delta_nfr_signature=0.5,
        goals=(goal,),
        phase_boundaries={"apex": (0, 4)},
        phase_samples={"apex": (0, 1, 2, 3)},
        active_phase="apex",
        dominant_nodes={"apex": ("tyres",)},
        phase_weights={"apex": {"__default__": 1.0}},
        grip_rel=1.0,
        phase_lag={"apex": 0.0},
        phase_alignment={"apex": 0.9},
        filtered_measures={
            "thermal_load": 5150.0,
            "style_index": 0.82,
            "grip_rel": 1.0,
            "d_nfr_flat": -0.32,
            "tyre_temp_fl": 84.0,
            "tyre_temp_fr": 83.5,
            "tyre_temp_rl": 79.2,
            "tyre_temp_rr": 78.8,
            "tyre_temp_fl_dt": 1.2,
            "tyre_temp_fr_dt": 1.0,
            "tyre_temp_rl_dt": 0.7,
            "tyre_temp_rr_dt": 0.6,
        },
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={"apex": {}},
        operator_events={},
    )
    thresholds = ThresholdProfile(
        entry_delta_tolerance=0.6,
        apex_delta_tolerance=0.6,
        exit_delta_tolerance=0.6,
        piano_delta_tolerance=0.5,
        rho_detune_threshold=0.4,
    )
    context = RuleContext(
        car_model="generic_gt",
        track_name="valencia",
        thresholds=thresholds,
        tyre_offsets={"pressure_front": -0.02},
    )
    rule = TyreBalanceRule(priority=18)

    recommendations = list(rule.evaluate([], [microsector], context))
    assert recommendations
    pressure_rec = next(rec for rec in recommendations if "ΔPfront" in rec.message)
    camber_rec = next(rec for rec in recommendations if "camber" in rec.message)
    assert pressure_rec.priority == 18
    assert pressure_rec.delta is not None and pressure_rec.delta < 0
    assert camber_rec.priority == 19
    assert MANUAL_REFERENCES["tyre_balance"].split()[0] in camber_rec.rationale


def test_parallel_steer_rule_recommends_open_toe_on_negative_delta() -> None:
    rule = ParallelSteerRule(priority=16, threshold=0.05, delta_step=0.2)
    microsector = SimpleNamespace(
        index=7,
        filtered_measures={"ackermann_parallel_index": -0.12},
    )
    recommendations = list(rule.evaluate([], [microsector], None))
    assert recommendations
    recommendation = recommendations[0]
    assert "parallel steer" in recommendation.message.lower()
    assert recommendation.parameter == "front_toe_deg"
    assert recommendation.delta == pytest.approx(0.2)


def test_parallel_steer_rule_recommends_closing_toe_on_positive_delta() -> None:
    rule = ParallelSteerRule(priority=16, threshold=0.05, delta_step=0.15)
    microsector = SimpleNamespace(
        index=5,
        filtered_measures={"ackermann_parallel_index": 0.11},
    )
    recommendations = list(rule.evaluate([], [microsector], None))
    assert recommendations
    recommendation = recommendations[0]
    assert recommendation.delta == pytest.approx(-0.15)
    assert "parallel steer" in recommendation.message.lower()


def test_recommendation_engine_suppresses_when_quiet_sequence():
    goal = _udr_goal()
    quiet_payload = (
        {
            "duration": 0.9,
            "slack": 0.55,
            "structural_density_mean": 0.04,
        },
    )
    microsectors = [
        _udr_microsector(
            goal,
            udr=0.05,
            sample_count=6,
            filtered_measures={
                "udr": 0.05,
                "si_variance": 0.0003,
                "epi_derivative_abs": 0.04,
            },
            operator_events={"SILENCIO": quiet_payload},
            index=i,
        )
        for i in range(3)
    ]
    bundles = _udr_bundle_series([0.05, 0.04, 0.03, 0.02, 0.01])
    engine = RecommendationEngine(rules=[])
    recommendations = engine.generate(bundles, microsectors)
    assert len(recommendations) == 1
    message = recommendations[0].message.lower()
    rationale = recommendations[0].rationale.lower()
    assert "no tocar" in message
    assert "silencio" in rationale


def test_aero_coherence_rule_flags_high_speed_bias() -> None:
    thresholds = ThresholdProfile(
        entry_delta_tolerance=0.6,
        apex_delta_tolerance=0.6,
        exit_delta_tolerance=0.6,
        piano_delta_tolerance=0.5,
        rho_detune_threshold=0.4,
    )
    context = RuleContext(
        car_model="generic_gt",
        track_name="valencia",
        thresholds=thresholds,
        tyre_offsets={},
        aero_profiles={"race": AeroProfile(low_speed_target=0.0, high_speed_target=0.0)},
    )
    microsector = SimpleNamespace(
        index=3,
        filtered_measures={
            "aero_high_imbalance": 0.28,
            "aero_low_imbalance": 0.0,
            "aero_high_samples": 6,
            "aero_low_samples": 6,
            "aero_mechanical_coherence": 0.45,
        },
    )
    rule = AeroCoherenceRule(high_speed_threshold=0.1, low_speed_tolerance=0.05, min_high_samples=2, delta_step=0.75)

    recommendations = list(rule.evaluate([], [microsector], context))

    assert recommendations
    rec = recommendations[0]
    assert rec.parameter == "rear_wing_angle"
    assert rec.delta and rec.delta > 0
    assert "Alta velocidad" in rec.message
    assert "carga trasera" in rec.rationale
    assert "C(a/m)" in rec.rationale


def test_aero_coherence_rule_respects_low_speed_window() -> None:
    thresholds = ThresholdProfile(
        entry_delta_tolerance=0.6,
        apex_delta_tolerance=0.6,
        exit_delta_tolerance=0.6,
        piano_delta_tolerance=0.5,
        rho_detune_threshold=0.4,
    )
    context = RuleContext(
        car_model="generic_gt",
        track_name="valencia",
        thresholds=thresholds,
        tyre_offsets={},
        aero_profiles={"race": AeroProfile(low_speed_target=0.0, high_speed_target=0.0)},
    )
    microsector = SimpleNamespace(
        index=5,
        filtered_measures={
            "aero_high_imbalance": -0.3,
            "aero_low_imbalance": 0.2,
            "aero_high_samples": 8,
            "aero_low_samples": 8,
            "aero_mechanical_coherence": 0.4,
        },
    )
    rule = AeroCoherenceRule(high_speed_threshold=0.1, low_speed_tolerance=0.05, min_high_samples=2)

    recommendations = list(rule.evaluate([], [microsector], context))

    assert not recommendations


def test_recommendation_engine_detects_anomalies(car_track_thresholds):
    def build_nodes(delta_nfr: float, sense_index: float):
        return dict(
            tyres=TyresNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["tyres"]),
            suspension=SuspensionNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["suspension"]),
            chassis=ChassisNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["chassis"]),
            brakes=BrakesNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["brakes"]),
            transmission=TransmissionNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["transmission"]),
            track=TrackNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["track"]),
            driver=DriverNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["driver"]),
        )

    results = [
        EPIBundle(timestamp=0.0, epi=0.4, delta_nfr=15.0, sense_index=0.55, **build_nodes(15.0, 0.55)),
        EPIBundle(timestamp=0.1, epi=0.5, delta_nfr=-12.0, sense_index=0.50, **build_nodes(-12.0, 0.50)),
        EPIBundle(timestamp=0.2, epi=0.6, delta_nfr=2.0, sense_index=0.90, **build_nodes(2.0, 0.90)),
    ]
    engine = RecommendationEngine(threshold_library=car_track_thresholds)
    recommendations = engine.generate(results)
    categories = {recommendation.category for recommendation in recommendations}
    assert {"suspension", "aero", "driver"} <= categories
    messages = [recommendation.message for recommendation in recommendations]
    assert any("Ride" in message for message in messages)
    assert any("sense index" in message.lower() for message in messages)


def test_phase_specific_rules_triggered_with_microsectors(car_track_thresholds):
    def build_bundle(timestamp: float, delta_nfr: float, sense_index: float, tyre_delta: float) -> EPIBundle:
        tyre_node = TyresNode(delta_nfr=tyre_delta, sense_index=sense_index, nu_f=BASE_NU_F["tyres"])
        return EPIBundle(
            timestamp=timestamp,
            epi=0.5,
            delta_nfr=delta_nfr,
            sense_index=sense_index,
            tyres=tyre_node,
            suspension=SuspensionNode(delta_nfr=delta_nfr / 2, sense_index=sense_index, nu_f=BASE_NU_F["suspension"]),
            chassis=ChassisNode(delta_nfr=delta_nfr / 2, sense_index=sense_index, nu_f=BASE_NU_F["chassis"]),
            brakes=BrakesNode(delta_nfr=delta_nfr / 2, sense_index=sense_index, nu_f=BASE_NU_F["brakes"]),
            transmission=TransmissionNode(delta_nfr=delta_nfr / 2, sense_index=sense_index, nu_f=BASE_NU_F["transmission"]),
            track=TrackNode(delta_nfr=delta_nfr / 2, sense_index=sense_index, nu_f=BASE_NU_F["track"]),
            driver=DriverNode(delta_nfr=delta_nfr / 2, sense_index=sense_index, nu_f=BASE_NU_F["driver"]),
        )

    entry_target = 1.0
    apex_target = 0.5
    exit_target = -0.2
    window = (-0.3, 0.3)
    yaw_window = (-0.6, 0.6)
    entry_nodes = ("tyres", "brakes")
    apex_nodes = ("suspension", "tyres")
    exit_nodes = ("transmission", "tyres")
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
        "thermal_load": 5200.0,
        "style_index": 0.9,
        "grip_rel": 1.0,
    }
    phase_lag = {"entry": 0.0, "apex": 0.0, "exit": 0.0}
    phase_alignment = {"entry": 1.0, "apex": 1.0, "exit": 1.0}
    microsector = Microsector(
        index=3,
        start_time=0.0,
        end_time=0.6,
        curvature=1.8,
        brake_event=True,
        support_event=True,
        delta_nfr_signature=0.0,
        goals=(
            Goal(
                phase="entry",
                archetype="hairpin",
                description="",
                target_delta_nfr=entry_target,
                target_sense_index=0.9,
                nu_f_target=0.25,
                nu_exc_target=0.25,
                rho_target=1.0,
                target_phase_lag=0.0,
                target_phase_alignment=0.9,
                measured_phase_lag=0.0,
                measured_phase_alignment=1.0,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=entry_nodes,
            ),
            Goal(
                phase="apex",
                archetype="hairpin",
                description="",
                target_delta_nfr=apex_target,
                target_sense_index=0.9,
                nu_f_target=0.25,
                nu_exc_target=0.25,
                rho_target=1.0,
                target_phase_lag=0.0,
                target_phase_alignment=0.9,
                measured_phase_lag=0.0,
                measured_phase_alignment=1.0,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=apex_nodes,
            ),
            Goal(
                phase="exit",
                archetype="hairpin",
                description="",
                target_delta_nfr=exit_target,
                target_sense_index=0.9,
                nu_f_target=0.25,
                nu_exc_target=0.25,
                rho_target=1.0,
                target_phase_lag=0.0,
                target_phase_alignment=0.9,
                measured_phase_lag=0.0,
                measured_phase_alignment=1.0,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=exit_nodes,
            ),
        ),
        phase_boundaries={
            "entry": (0, 2),
            "apex": (2, 4),
            "exit": (4, 6),
        },
        phase_samples=phase_samples,
        active_phase="entry",
        dominant_nodes={
            "entry": entry_nodes,
            "apex": apex_nodes,
            "exit": exit_nodes,
        },
        phase_weights=phase_weights,
        grip_rel=1.0,
        phase_lag=phase_lag,
        phase_alignment=phase_alignment,
        filtered_measures=filtered_measures,
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy=window_occupancy,
        operator_events={},
    )

    results = [
        build_bundle(0.0, 4.0, 0.92, tyre_delta=6.0),
        build_bundle(0.1, 4.2, 0.91, tyre_delta=6.2),
        build_bundle(0.2, 2.8, 0.93, tyre_delta=6.6),
        build_bundle(0.3, 2.6, 0.94, tyre_delta=6.5),
        build_bundle(0.4, -3.0, 0.95, tyre_delta=6.0),
        build_bundle(0.5, -3.2, 0.95, tyre_delta=5.8),
    ]

    engine = RecommendationEngine(
        car_model="generic_gt",
        track_name="valencia",
        threshold_library=car_track_thresholds,
    )
    recommendations = engine.generate(results, [microsector])

    assert len(recommendations) >= 6
    categories = {recommendation.category for recommendation in recommendations}
    assert {"entry", "apex", "pianos", "exit"} <= categories

    entry_messages = [rec.message for rec in recommendations if rec.category == "entry"]
    assert any("bias" in message.lower() for message in entry_messages)
    assert any("click" in message.lower() or "psi" in message.lower() for message in entry_messages)

    apex_messages = [rec.message for rec in recommendations if rec.category == "apex"]
    assert any("barra" in message.lower() for message in apex_messages)
    assert any("psi" in message.lower() for message in apex_messages)

    exit_messages = [rec.message for rec in recommendations if rec.category == "exit"]
    assert any("lsd" in message.lower() or "%" in message.lower() for message in exit_messages)

    entry_rationales = [
        recommendation.rationale for recommendation in recommendations if recommendation.category == "entry"
    ]
    assert any("ν_f" in rationale for rationale in entry_rationales)
    assert any(f"{entry_target:.2f}" in rationale for rationale in entry_rationales)


def test_recommendation_engine_updates_persistent_profile(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.toml"
    manager = ProfileManager(profiles_path)
    engine = RecommendationEngine(
        car_model="generic_gt",
        track_name="valencia",
        profile_manager=manager,
    )

    baseline_context = engine._resolve_context("generic_gt", "valencia")
    before_weights = baseline_context.thresholds.weights_for_phase("entry")
    if isinstance(before_weights, Mapping):
        entry_before = float(before_weights.get("__default__", 1.0))
    else:
        entry_before = float(before_weights)

    engine.register_stint_result(
        sense_index=0.7,
        delta_nfr=2.5,
        car_model="generic_gt",
        track_name="valencia",
    )

    recommendations = [
        Recommendation(category="entry", message="", rationale=""),
        Recommendation(category="apex", message="", rationale=""),
    ]

    engine.register_plan(
        recommendations,
        car_model="generic_gt",
        track_name="valencia",
        baseline_sense_index=0.7,
        baseline_delta_nfr=2.5,
    )

    engine.register_stint_result(
        sense_index=0.78,
        delta_nfr=1.4,
        car_model="generic_gt",
        track_name="valencia",
    )

    updated_context = engine._resolve_context("generic_gt", "valencia")
    after_weights = updated_context.thresholds.weights_for_phase("entry")
    if isinstance(after_weights, Mapping):
        entry_after = float(after_weights.get("__default__", entry_before))
    else:
        entry_after = float(after_weights)

    assert entry_after > entry_before
    assert profiles_path.exists()


def test_profile_manager_rehydrates_track_weights(
    tmp_path: Path, car_track_thresholds
) -> None:
    profiles_path = tmp_path / "profiles.toml"
    profiles_path.write_text(
        """
[profiles.generic_gt.valencia.objectives]
target_delta_nfr = 0.4
target_sense_index = 0.82

[profiles.generic_gt.valencia.tolerances]
entry = 0.9
apex = 0.6
exit = 1.1
piano = 1.5
""".strip()
        + "\n",
        encoding="utf8",
    )

    manager = ProfileManager(
        profiles_path, threshold_library=car_track_thresholds
    )

    snapshot = manager.resolve("generic_gt", "valencia")
    entry_snapshot = snapshot.phase_weights.get("entry", {})
    assert entry_snapshot.get("tyres") == pytest.approx(1.2)

    manager.save()

    reloaded = ProfileManager(
        profiles_path, threshold_library=car_track_thresholds
    )
    context_engine = RecommendationEngine(
        car_model="generic_gt",
        track_name="valencia",
        threshold_library=car_track_thresholds,
        profile_manager=reloaded,
    )._resolve_context("generic_gt", "valencia")
    entry_weights = context_engine.thresholds.weights_for_phase("entry")
    assert isinstance(entry_weights, Mapping)
    assert entry_weights.get("tyres") == pytest.approx(1.2)

def test_threshold_profile_exposes_phase_weights():
    engine = RecommendationEngine(car_model="generic_gt", track_name="valencia")
    profile = engine._resolve_context("generic_gt", "valencia").thresholds  # type: ignore[attr-defined]
    entry_weights = profile.weights_for_phase("entry")
    assert isinstance(entry_weights, Mapping)
    assert entry_weights.get("tyres", 0.0) > 1.0
    apex_weights = profile.weights_for_phase("apex")
    assert isinstance(apex_weights, Mapping)
    assert apex_weights.get("suspension", 0.0) >= 1.0


def test_track_specific_profile_tightens_entry_threshold():
    def build_bundle(index: int, delta_nfr: float, sense_index: float = 0.9) -> EPIBundle:
        tyre_node = TyresNode(delta_nfr=delta_nfr / 2, sense_index=sense_index, nu_f=BASE_NU_F["tyres"])
        return EPIBundle(
            timestamp=index * 0.1,
            epi=0.5,
            delta_nfr=delta_nfr,
            sense_index=sense_index,
            tyres=tyre_node,
            suspension=SuspensionNode(
                delta_nfr=delta_nfr / 4,
                sense_index=sense_index,
                nu_f=BASE_NU_F["suspension"],
            ),
            chassis=ChassisNode(
                delta_nfr=delta_nfr / 4,
                sense_index=sense_index,
                nu_f=BASE_NU_F["chassis"],
            ),
            brakes=BrakesNode(
                delta_nfr=delta_nfr / 4,
                sense_index=sense_index,
                nu_f=BASE_NU_F["brakes"],
            ),
            transmission=TransmissionNode(
                delta_nfr=delta_nfr / 4,
                sense_index=sense_index,
                nu_f=BASE_NU_F["transmission"],
            ),
            track=TrackNode(
                delta_nfr=delta_nfr / 4,
                sense_index=sense_index,
                nu_f=BASE_NU_F["track"],
            ),
            driver=DriverNode(
                delta_nfr=delta_nfr / 4,
                sense_index=sense_index,
                nu_f=BASE_NU_F["driver"],
            ),
        )

    window = (-0.05, 0.05)
    yaw_window = (-0.3, 0.3)
    nodes = ("tyres", "brakes")
    filtered_measures = {
        "thermal_load": 5100.0,
        "style_index": 0.9,
        "grip_rel": 1.0,
    }
    phase_lag = {"entry": 0.0, "apex": 0.0, "exit": 0.0}
    phase_alignment = {"entry": 1.0, "apex": 1.0, "exit": 1.0}
    window_occupancy = {
        "entry": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
        "apex": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
        "exit": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
    }
    microsector = Microsector(
        index=4,
        start_time=0.0,
        end_time=0.6,
        curvature=1.5,
        brake_event=True,
        support_event=True,
        delta_nfr_signature=0.0,
        goals=(
            Goal(
                phase="entry",
                archetype="hairpin",
                description="",
                target_delta_nfr=0.0,
                target_sense_index=0.9,
                nu_f_target=0.25,
                nu_exc_target=0.25,
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
                target_delta_nfr=0.2,
                target_sense_index=0.9,
                nu_f_target=0.25,
                nu_exc_target=0.25,
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
                target_delta_nfr=-0.1,
                target_sense_index=0.9,
                nu_f_target=0.25,
                nu_exc_target=0.25,
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
        phase_boundaries={
            "entry": (0, 2),
            "apex": (2, 4),
            "exit": (4, 6),
        },
        phase_samples={
            "entry": (0, 1),
            "apex": (2, 3),
            "exit": (4, 5),
        },
        active_phase="entry",
        dominant_nodes={
            "entry": nodes,
            "apex": nodes,
            "exit": nodes,
        },
        phase_weights={
            "entry": {"__default__": 1.0},
            "apex": {"__default__": 1.0},
            "exit": {"__default__": 1.0},
        },
        grip_rel=1.0,
        phase_lag=phase_lag,
        phase_alignment=phase_alignment,
        filtered_measures=filtered_measures,
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy=window_occupancy,
        operator_events={},
    )

    results = [
        build_bundle(0, 1.0),
        build_bundle(1, 1.1),
        build_bundle(2, 0.2),
        build_bundle(3, 0.2),
        build_bundle(4, -0.05),
        build_bundle(5, -0.05),
    ]

    generic_engine = RecommendationEngine(car_model="generic", track_name="generic")
    generic_recs = generic_engine.generate(results, [microsector])
    valencia_engine = RecommendationEngine(car_model="generic_gt", track_name="valencia")
    valencia_recs = valencia_engine.generate(results, [microsector])

    generic_entry_global = [
        rec for rec in generic_recs if rec.category == "entry" and "ΔNFR global" in rec.message
    ]
    valencia_entry_global = [
        rec for rec in valencia_recs if rec.category == "entry" and "ΔNFR global" in rec.message
    ]

    assert not generic_entry_global
    assert valencia_entry_global


def test_node_operator_rule_responds_to_nu_f_excess(car_track_thresholds):
    def build_bundle(timestamp: float, transmission_nu_f: float) -> EPIBundle:
        return EPIBundle(
            timestamp=timestamp,
            epi=0.5,
            delta_nfr=0.3,
            sense_index=0.92,
            tyres=TyresNode(delta_nfr=0.15, sense_index=0.92, nu_f=BASE_NU_F["tyres"]),
            suspension=SuspensionNode(
                delta_nfr=0.15, sense_index=0.92, nu_f=BASE_NU_F["suspension"]
            ),
            chassis=ChassisNode(delta_nfr=0.15, sense_index=0.92, nu_f=BASE_NU_F["chassis"]),
            brakes=BrakesNode(delta_nfr=0.15, sense_index=0.92, nu_f=BASE_NU_F["brakes"]),
            transmission=TransmissionNode(
                delta_nfr=0.15, sense_index=0.92, nu_f=transmission_nu_f
            ),
            track=TrackNode(delta_nfr=0.15, sense_index=0.92, nu_f=BASE_NU_F["track"]),
            driver=DriverNode(delta_nfr=0.15, sense_index=0.92, nu_f=BASE_NU_F["driver"]),
        )

    window = (-0.05, 0.05)
    yaw_window = (-0.3, 0.3)
    exit_nodes = ("transmission",)
    phase_lag = {"entry": 0.0, "apex": 0.0, "exit": 0.0}
    phase_alignment = {"entry": 1.0, "apex": 1.0, "exit": 1.0}
    microsector = Microsector(
        index=7,
        start_time=0.0,
        end_time=0.6,
        curvature=1.2,
        brake_event=False,
        support_event=False,
        delta_nfr_signature=0.0,
        goals=(
            Goal(
                phase="entry",
                archetype="transición",
                description="",
                target_delta_nfr=0.0,
                target_sense_index=0.9,
                nu_f_target=0.1,
                nu_exc_target=0.1,
                rho_target=1.0,
                target_phase_lag=0.0,
                target_phase_alignment=0.9,
                measured_phase_lag=0.0,
                measured_phase_alignment=1.0,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=("tyres",),
            ),
            Goal(
                phase="apex",
                archetype="transición",
                description="",
                target_delta_nfr=0.1,
                target_sense_index=0.9,
                nu_f_target=0.15,
                nu_exc_target=0.15,
                rho_target=1.0,
                target_phase_lag=0.0,
                target_phase_alignment=0.9,
                measured_phase_lag=0.0,
                measured_phase_alignment=1.0,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=("suspension",),
            ),
            Goal(
                phase="exit",
                archetype="tracción",
                description="",
                target_delta_nfr=-0.05,
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
                dominant_nodes=exit_nodes,
            ),
        ),
        phase_boundaries={
            "entry": (0, 2),
            "apex": (2, 4),
            "exit": (4, 6),
        },
        phase_samples={
            "entry": (0, 1),
            "apex": (2, 3),
            "exit": (4, 5),
        },
        active_phase="exit",
        dominant_nodes={
            "entry": ("tyres",),
            "apex": ("suspension",),
            "exit": exit_nodes,
        },
        phase_weights={
            "entry": {"__default__": 1.0},
            "apex": {"__default__": 1.0},
            "exit": {"__default__": 1.0},
        },
        grip_rel=1.0,
        phase_lag=phase_lag,
        phase_alignment=phase_alignment,
        filtered_measures={
            "thermal_load": 5000.0,
            "style_index": 0.9,
            "grip_rel": 1.0,
        },
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={
            "entry": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
            "apex": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
            "exit": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
        },
        operator_events={},
    )

    results = [
        build_bundle(0.0, BASE_NU_F["transmission"]),
        build_bundle(0.1, BASE_NU_F["transmission"]),
        build_bundle(0.2, BASE_NU_F["transmission"]),
        build_bundle(0.3, BASE_NU_F["transmission"]),
        build_bundle(0.4, 0.35),
        build_bundle(0.5, 0.34),
    ]

    engine = RecommendationEngine(
        car_model="generic_gt",
        track_name="valencia",
        threshold_library=car_track_thresholds,
    )
    recommendations = engine.generate(results, [microsector])

    exit_messages = [
        rec
        for rec in recommendations
        if rec.category == "exit" and "diferencial" in rec.message.lower()
    ]
    assert exit_messages
    assert any("abrir" in rec.message.lower() for rec in exit_messages)

    rationale = exit_messages[0].rationale
    assert "transmisión" in rationale
    assert "ν_f medio" in rationale
    assert "generic_gt/valencia" in rationale


def test_phase_node_rule_flips_with_phase_misalignment(car_track_thresholds) -> None:
    engine = RecommendationEngine(
        car_model="generic_gt", track_name="valencia", threshold_library=car_track_thresholds
    )
    context = engine._resolve_context("generic_gt", "valencia")
    rule = PhaseNodeOperatorRule(
        phase="apex",
        operator_label="Operador",
        category="apex",
        priority=25,
        reference_key="antiroll",
    )

    def _bundle(nu_f: float, timestamp: float) -> EPIBundle:
        return EPIBundle(
            timestamp=timestamp,
            epi=0.0,
            delta_nfr=0.8,
            sense_index=0.82,
            tyres=TyresNode(delta_nfr=0.2, sense_index=0.82, nu_f=BASE_NU_F["tyres"]),
            suspension=SuspensionNode(delta_nfr=0.4, sense_index=0.82, nu_f=nu_f),
            chassis=ChassisNode(delta_nfr=0.2, sense_index=0.82, nu_f=BASE_NU_F["chassis"]),
            brakes=BrakesNode(delta_nfr=0.05, sense_index=0.82, nu_f=BASE_NU_F["brakes"]),
            transmission=TransmissionNode(
                delta_nfr=0.05, sense_index=0.82, nu_f=BASE_NU_F["transmission"]
            ),
            track=TrackNode(delta_nfr=0.05, sense_index=0.82, nu_f=BASE_NU_F["track"]),
            driver=DriverNode(delta_nfr=0.05, sense_index=0.82, nu_f=BASE_NU_F["driver"]),
        )

    results = [_bundle(1.5, 0.0), _bundle(1.55, 0.1)]
    goal = Goal(
        phase="apex",
        archetype="hairpin",
        description="",
        target_delta_nfr=0.2,
        target_sense_index=0.88,
        nu_f_target=0.25,
                nu_exc_target=0.25,
                rho_target=1.0,
        target_phase_lag=0.0,
        target_phase_alignment=0.9,
        measured_phase_lag=0.6,
        measured_phase_alignment=-0.4,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.3, 0.3),
        dominant_nodes=("suspension",),
    )
    microsector = Microsector(
        index=12,
        start_time=0.0,
        end_time=0.2,
        curvature=1.4,
        brake_event=False,
        support_event=False,
        delta_nfr_signature=0.4,
        goals=(goal,),
        phase_boundaries={"apex": (0, 2)},
        phase_samples={"apex": (0, 1)},
        active_phase="apex",
        dominant_nodes={"apex": ("suspension",)},
        phase_weights={"apex": {"__default__": 1.0}},
        grip_rel=1.0,
        phase_lag={"apex": 0.6},
        phase_alignment={"apex": -0.4},
        filtered_measures={"thermal_load": 5000.0, "style_index": 0.8},
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={"apex": {"slip_lat": 75.0, "slip_long": 70.0, "yaw_rate": 68.0}},
        operator_events={},
    )

    recommendations = list(rule.evaluate(results, [microsector], context=context))

    rebound_actions = [rec for rec in recommendations if rec.parameter == "rear_rebound_clicks"]
    assert rebound_actions, "expected rebound click recommendation"
    assert rebound_actions[0].delta > 0
    assert "θ" in rebound_actions[0].rationale
    assert "Siφ" in rebound_actions[0].rationale
    assert "invierte el sentido" in rebound_actions[0].rationale.lower()


def test_detune_ratio_rule_emits_modal_guidance() -> None:
    goal = Goal(
        phase="apex",
        archetype="hairpin",
        description="",
        target_delta_nfr=0.3,
        target_sense_index=0.85,
        nu_f_target=0.28,
        nu_exc_target=0.22,
        rho_target=0.78,
        target_phase_lag=0.0,
        target_phase_alignment=0.9,
        measured_phase_lag=0.0,
        measured_phase_alignment=0.9,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.3, 0.3),
        dominant_nodes=("suspension", "chassis"),
        detune_ratio_weights={"longitudinal": 0.7, "lateral": 0.3},
    )
    microsector = Microsector(
        index=4,
        start_time=0.0,
        end_time=0.4,
        curvature=1.6,
        brake_event=False,
        support_event=True,
        delta_nfr_signature=0.4,
        goals=(goal,),
        phase_boundaries={"apex": (0, 3)},
        phase_samples={"apex": (0, 1, 2)},
        active_phase="apex",
        dominant_nodes={"apex": ("suspension",)},
        phase_weights={"apex": {"__default__": 1.0}},
        grip_rel=1.1,
        phase_lag={"apex": 0.0},
        phase_alignment={"apex": 0.9},
        filtered_measures={
            "thermal_load": 5200.0,
            "style_index": 0.82,
            "grip_rel": 1.1,
            "nu_f": 0.9,
            "nu_exc": 0.5,
            "rho": 0.55,
            "d_nfr_res": 0.9,
        },
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={"apex": {"slip_lat": 80.0, "slip_long": 78.0, "yaw_rate": 72.0}},
        operator_events={},
    )

    engine = RecommendationEngine(rules=[DetuneRatioRule(priority=42)])
    recommendations = engine.generate([], [microsector])

    assert recommendations, "expected detune ratio warning"
    rationale = recommendations[0].rationale.lower()
    assert "ρ=" in recommendations[0].rationale
    assert "barras" in rationale
    assert "detune" in rationale
    assert "longitudinal" in rationale


def test_useful_dissonance_rule_reinforces_rear_when_udr_high(car_track_thresholds):
    values = [1.4, 1.5, 1.3]
    results = _udr_bundle_series(values)
    goal = _udr_goal("apex", target_delta=0.2)
    microsector = _udr_microsector(goal, udr=0.8, sample_count=len(results))

    engine = RecommendationEngine(
        rules=[UsefulDissonanceRule(priority=40)],
        car_model="generic_gt",
        track_name="valencia",
        threshold_library=car_track_thresholds,
    )

    recommendations = engine.generate(results, [microsector])

    assert recommendations, "expected UDR escalation recommendation"
    messages = [rec.message.lower() for rec in recommendations]
    assert any("reforzar" in message for message in messages)
    rationales = " ".join(rec.rationale.lower() for rec in recommendations)
    assert "udr" in rationales


def test_useful_dissonance_rule_softens_axle_when_udr_low(car_track_thresholds):
    values = [1.1, 1.2, 1.05]
    results = _udr_bundle_series(values)
    goal = _udr_goal("apex", target_delta=0.2)
    microsector = _udr_microsector(goal, udr=0.1, sample_count=len(results))

    engine = RecommendationEngine(
        rules=[UsefulDissonanceRule(priority=38)],
        car_model="generic_gt",
        track_name="valencia",
        threshold_library=car_track_thresholds,
    )

    recommendations = engine.generate(results, [microsector])

    assert recommendations, "expected UDR softening recommendation"
    messages = [rec.message.lower() for rec in recommendations]
    assert any("ablandar" in message and "delanter" in message for message in messages)

    # Oversteer scenario should target the rear axle.
    oversteer_results = _udr_bundle_series([-1.2, -1.3, -1.1])
    oversteer_goal = _udr_goal("apex", target_delta=0.0)
    oversteer_microsector = _udr_microsector(
        oversteer_goal,
        udr=0.15,
        sample_count=len(oversteer_results),
    )

    oversteer_recs = engine.generate(oversteer_results, [oversteer_microsector])

    assert oversteer_recs, "expected UDR rear softening recommendation"
    oversteer_messages = [rec.message.lower() for rec in oversteer_recs]
    assert any("traser" in message for message in oversteer_messages)

def test_phase_delta_rule_prioritises_brake_bias_for_longitudinal_axis() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="entry",
        operator_label="Operador de frenado",
        category="entry",
        phase_label="entrada",
        priority=10,
        reference_key="braking",
    )
    goal = Goal(
        phase="entry1",
        archetype="frenada",
        description="",
        target_delta_nfr=0.1,
        target_sense_index=0.9,
        nu_f_target=0.25,
        nu_exc_target=0.2,
        rho_target=1.0,
        target_phase_lag=0.0,
        target_phase_alignment=0.9,
        measured_phase_lag=0.25,
        measured_phase_alignment=0.85,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.4, 0.4),
        yaw_rate_window=(-0.5, 0.5),
        dominant_nodes=("brakes",),
        target_delta_nfr_long=0.08,
        target_delta_nfr_lat=0.02,
        delta_axis_weights={"longitudinal": 0.75, "lateral": 0.25},
    )
    microsector = Microsector(
        index=1,
        start_time=0.0,
        end_time=0.3,
        curvature=1.1,
        brake_event=True,
        support_event=False,
        delta_nfr_signature=0.4,
        goals=(goal,),
        phase_boundaries={"entry1": (0, 3)},
        phase_samples={"entry1": (0, 1, 2)},
        active_phase="entry1",
        dominant_nodes={"entry1": ("brakes",)},
        phase_weights={"entry1": {"__default__": 1.0}},
        grip_rel=1.0,
        phase_lag={"entry1": 0.25},
        phase_alignment={"entry1": 0.82},
        filtered_measures={},
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={"entry1": {}},
        operator_events={},
    )
    results = [
        _axis_bundle(0.6, 0.5, 0.1),
        _axis_bundle(0.58, 0.46, 0.12),
        _axis_bundle(0.62, 0.52, 0.1),
    ]
    thresholds = ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    messages = [rec.message for rec in recommendations]
    assert any("bias de frenos" in message for message in messages)
    targeted = [rec for rec in recommendations if rec.parameter == "brake_bias_pct"]
    assert targeted
    assert all(rec.priority <= rule.priority - 1 for rec in targeted)


def _entry_goal_with_gradient(gradient: float) -> Goal:
    return Goal(
        phase="entry1",
        archetype="frenada",
        description="",
        target_delta_nfr=0.2,
        target_sense_index=0.9,
        nu_f_target=0.25,
        nu_exc_target=0.2,
        rho_target=1.0,
        target_phase_lag=0.0,
        target_phase_alignment=0.9,
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
    return Microsector(
        index=3,
        start_time=0.0,
        end_time=0.45,
        curvature=1.2,
        brake_event=True,
        support_event=False,
        delta_nfr_signature=0.6,
        goals=(goal,),
        phase_boundaries={goal.phase: (0, 3)},
        phase_samples={goal.phase: (0, 1, 2)},
        active_phase=goal.phase,
        dominant_nodes={goal.phase: goal.dominant_nodes},
        phase_weights={goal.phase: {"__default__": 1.0}},
        grip_rel=1.0,
        phase_lag={goal.phase: goal.measured_phase_lag},
        phase_alignment={goal.phase: goal.measured_phase_alignment},
        filtered_measures={"gradient": gradient},
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={goal.phase: {}},
        operator_events={},
    )


def _entry_results_with_gradient(gradient: float) -> Sequence[EPIBundle]:
    return [
        _axis_bundle(0.62, 0.5, 0.1, gradient=gradient),
        _axis_bundle(0.6, 0.5, 0.1, gradient=gradient),
        _axis_bundle(0.58, 0.5, 0.1, gradient=gradient),
    ]


def test_phase_delta_rule_offsets_brake_bias_with_downhill_gradient() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="entry",
        operator_label="Operador de frenado",
        category="entry",
        phase_label="entrada",
        priority=10,
        reference_key="braking",
    )
    gradient = -0.015
    goal = _entry_goal_with_gradient(gradient)
    microsector = _entry_microsector_with_gradient(goal, gradient)
    results = list(_entry_results_with_gradient(gradient))
    thresholds = ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    brake_targets = [rec for rec in recommendations if rec.parameter == "brake_bias_pct"]
    assert brake_targets
    downhill_delta = brake_targets[0].delta
    assert downhill_delta is not None
    assert downhill_delta < 0
    assert downhill_delta <= -1.0


def test_phase_delta_rule_offsets_brake_bias_with_uphill_gradient() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="entry",
        operator_label="Operador de frenado",
        category="entry",
        phase_label="entrada",
        priority=10,
        reference_key="braking",
    )
    gradient = 0.015
    goal = _entry_goal_with_gradient(gradient)
    microsector = _entry_microsector_with_gradient(goal, gradient)
    results = list(_entry_results_with_gradient(gradient))
    thresholds = ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    brake_targets = [rec for rec in recommendations if rec.parameter == "brake_bias_pct"]
    assert brake_targets
    uphill_delta = brake_targets[0].delta
    assert uphill_delta is not None
    assert uphill_delta > 0
    assert uphill_delta >= 0.5


def test_bottoming_priority_rule_switches_focus(bottoming_microsectors) -> None:
    smooth, rough = bottoming_microsectors
    rule = BottomingPriorityRule(
        priority=18,
        ratio_threshold=0.5,
        smooth_surface_cutoff=1.1,
        ride_height_delta=1.5,
        bump_delta=3.0,
    )
    thresholds = ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate([], [smooth, rough], context))
    assert recommendations
    height_targets = [rec for rec in recommendations if rec.parameter == "front_ride_height"]
    bump_targets = [rec for rec in recommendations if rec.parameter == "rear_compression_clicks"]
    assert height_targets
    assert bump_targets
    assert "altura" in height_targets[0].message.lower()
    assert "compresión" in bump_targets[0].message.lower()


def test_phase_delta_rule_brake_bias_uses_operator_events() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="entry",
        operator_label="Operador de frenado",
        category="entry",
        phase_label="entrada",
        priority=12,
        reference_key="braking",
    )
    goal = Goal(
        phase="entry1",
        archetype="frenada",
        description="",
        target_delta_nfr=0.3,
        target_sense_index=0.85,
        nu_f_target=0.28,
        nu_exc_target=0.22,
        rho_target=1.0,
        target_phase_lag=0.0,
        target_phase_alignment=0.9,
        measured_phase_lag=0.1,
        measured_phase_alignment=0.82,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.4, 0.4),
        yaw_rate_window=(-0.5, 0.5),
        dominant_nodes=("brakes", "tyres"),
        target_delta_nfr_long=0.18,
        target_delta_nfr_lat=0.06,
        delta_axis_weights={"longitudinal": 0.65, "lateral": 0.35},
    )
    operator_events = {
        "OZ": (
            {
                "name": "OZ",
                "start_index": 0,
                "end_index": 2,
                "microsector": 4,
                "delta_nfr_threshold": 0.28,
                "delta_nfr_peak": 0.35,
                "delta_nfr_avg": 0.31,
                "delta_nfr_ratio": 1.25,
                "surface_label": "low_grip",
                "surface_factor": 0.94,
            },
        ),
    }
    microsector = Microsector(
        index=4,
        start_time=0.0,
        end_time=0.4,
        curvature=1.3,
        brake_event=True,
        support_event=True,
        delta_nfr_signature=0.42,
        goals=(goal,),
        phase_boundaries={"entry1": (0, 3)},
        phase_samples={"entry1": (0, 1, 2)},
        active_phase="entry1",
        dominant_nodes={"entry1": ("brakes", "tyres")},
        phase_weights={"entry1": {"__default__": 1.0}},
        grip_rel=0.95,
        phase_lag={"entry1": 0.1},
        phase_alignment={"entry1": 0.8},
        filtered_measures={},
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={"entry1": {}},
        operator_events=operator_events,
    )
    results = [
        _axis_bundle(-0.32, -0.28, -0.07),
        _axis_bundle(-0.34, -0.29, -0.06),
        _axis_bundle(-0.31, -0.3, -0.05),
    ]
    thresholds = ThresholdProfile(0.12, 0.12, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    brake_recs = [rec for rec in recommendations if rec.parameter == "brake_bias_pct"]
    assert brake_recs, "expected brake bias recommendation"
    assert all(rec.delta is not None and rec.delta > 0 for rec in brake_recs)
    assert all(rec.priority <= rule.priority - 2 for rec in brake_recs)
    rationale_blob = " ".join(rec.rationale for rec in brake_recs)
    assert "OZ×1" in rationale_blob
    assert "low_grip" in rationale_blob
    assert "ΔNFR" in rationale_blob


def test_phase_delta_rule_prioritises_sway_bar_for_lateral_axis() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="apex",
        operator_label="Operador de vértice",
        category="apex",
        phase_label="vértice",
        priority=20,
        reference_key="antiroll",
    )
    goal = Goal(
        phase="apex3a",
        archetype="hairpin",
        description="",
        target_delta_nfr=0.05,
        target_sense_index=0.88,
        nu_f_target=0.3,
        nu_exc_target=0.25,
        rho_target=0.9,
        target_phase_lag=0.0,
        target_phase_alignment=0.92,
        measured_phase_lag=-0.22,
        measured_phase_alignment=0.75,
        slip_lat_window=(-0.25, 0.25),
        slip_long_window=(-0.25, 0.25),
        yaw_rate_window=(-0.4, 0.4),
        dominant_nodes=("suspension", "chassis"),
        target_delta_nfr_long=0.01,
        target_delta_nfr_lat=0.06,
        delta_axis_weights={"longitudinal": 0.2, "lateral": 0.8},
    )
    microsector = Microsector(
        index=2,
        start_time=0.0,
        end_time=0.3,
        curvature=1.6,
        brake_event=False,
        support_event=True,
        delta_nfr_signature=0.3,
        goals=(goal,),
        phase_boundaries={"apex3a": (0, 4)},
        phase_samples={"apex3a": (0, 1, 2, 3)},
        active_phase="apex3a",
        dominant_nodes={"apex3a": ("suspension", "chassis")},
        phase_weights={"apex3a": {"__default__": 1.0}},
        grip_rel=1.0,
        phase_lag={"apex3a": -0.22},
        phase_alignment={"apex3a": 0.74},
        filtered_measures={},
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={"apex3a": {}},
        operator_events={},
    )
    results = [
        _axis_bundle(1.0, 0.12, 0.88),
        _axis_bundle(0.95, 0.1, 0.85),
        _axis_bundle(1.05, 0.14, 0.92),
        _axis_bundle(1.0, 0.11, 0.87),
    ]
    thresholds = ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    messages = [rec.message for rec in recommendations]
    assert any("barras estabilizadoras" in message for message in messages)
    sway_recs = [
        rec
        for rec in recommendations
        if rec.parameter in {"front_arb_steps", "rear_arb_steps"}
    ]
    assert sway_recs
    assert all(rec.priority <= rule.priority - 1 for rec in sway_recs)


def test_phase_delta_rule_emits_geometry_actions_for_coherence_gap() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="entry",
        operator_label="Operador de frenado",
        category="entry",
        phase_label="entrada",
        priority=14,
        reference_key="braking",
    )
    goal = Goal(
        phase="entry",
        archetype="tight",
        description="",
        target_delta_nfr=0.2,
        target_sense_index=0.9,
        nu_f_target=0.28,
        nu_exc_target=0.22,
        rho_target=1.0,
        target_phase_lag=0.0,
        target_phase_alignment=0.9,
        measured_phase_lag=0.18,
        measured_phase_alignment=0.62,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.4, 0.4),
        dominant_nodes=("suspension", "tyres"),
        target_delta_nfr_long=0.15,
        target_delta_nfr_lat=0.05,
    )
    microsector = Microsector(
        index=9,
        start_time=0.0,
        end_time=0.3,
        curvature=1.3,
        brake_event=True,
        support_event=False,
        delta_nfr_signature=0.25,
        goals=(goal,),
        phase_boundaries={"entry": (0, 3)},
        phase_samples={"entry": (0, 1, 2)},
        active_phase="entry",
        dominant_nodes={"entry": ("suspension", "tyres")},
        phase_weights={"entry": {"__default__": 1.0}},
        grip_rel=1.0,
        phase_lag={"entry": 0.18},
        phase_alignment={"entry": 0.62},
        filtered_measures={"coherence_index": 0.32},
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={"entry_a": {}},
        operator_events={},
    )
    results = []
    for value in (0.46, 0.48, 0.44):
        bundle = _axis_bundle(value, 0.36, 0.1)
        results.append(replace(bundle, coherence_index=0.32))
    thresholds = ThresholdProfile(0.2, 0.5, 0.5, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    geometry_params = {"front_camber_deg", "front_toe_deg", "caster_deg"}
    geometry_recs = [rec for rec in recommendations if rec.parameter in geometry_params]
    assert geometry_recs, "expected geometry recommendations triggered by coherence gap"
    assert any("camber" in rec.message.lower() or "toe" in rec.message.lower() for rec in geometry_recs)
    assert all(rec.priority <= rule.priority - 1 for rec in geometry_recs)


def test_phase_delta_rule_prioritises_front_spring_with_lateral_bias() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="apex",
        operator_label="Operador de vértice",
        category="apex",
        phase_label="vértice",
        priority=22,
        reference_key="antiroll",
    )
    goal = Goal(
        phase="apex",
        archetype="medium",
        description="",
        target_delta_nfr=0.25,
        target_sense_index=0.9,
        nu_f_target=0.28,
        nu_exc_target=0.23,
        rho_target=0.9,
        target_phase_lag=0.0,
        target_phase_alignment=0.88,
        measured_phase_lag=0.12,
        measured_phase_alignment=0.8,
        slip_lat_window=(-0.4, 0.4),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.5, 0.5),
        dominant_nodes=("suspension",),
        target_delta_nfr_long=0.06,
        target_delta_nfr_lat=0.18,
    )
    samples = tuple(range(4))
    microsector = Microsector(
        index=12,
        start_time=0.0,
        end_time=0.4,
        curvature=1.6,
        brake_event=False,
        support_event=True,
        delta_nfr_signature=0.32,
        goals=(goal,),
        phase_boundaries={goal.phase: (0, len(samples))},
        phase_samples={goal.phase: samples},
        active_phase=goal.phase,
        dominant_nodes={goal.phase: goal.dominant_nodes},
        phase_weights={goal.phase: {"__default__": 1.0}},
        grip_rel=1.0,
        phase_lag={goal.phase: goal.measured_phase_lag},
        phase_alignment={goal.phase: goal.measured_phase_alignment},
        filtered_measures={},
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={goal.phase: {}},
        operator_events={},
    )
    raw_results: list[EPIBundle] = []
    for lat_component in (0.3, 0.32, 0.31, 0.29):
        bundle = _axis_bundle(0.38, 0.07, lat_component)
        suspension = replace(bundle.suspension, nu_f=0.36)
        raw_results.append(replace(bundle, suspension=suspension))
    thresholds = ThresholdProfile(0.1, 0.05, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="GT3", track_name="VAL", thresholds=thresholds)
    recommendations = list(rule.evaluate(raw_results, [microsector], context))
    spring_recs = [rec for rec in recommendations if rec.parameter == "front_spring_stiffness"]
    assert spring_recs, "expected front spring recommendation under lateral dominance"
    assert all(rec.delta is not None and rec.delta < 0 for rec in spring_recs)
    assert all("νf_susp" in rec.message for rec in spring_recs)
    assert all("ΔNFR⊥" in rec.rationale for rec in spring_recs)


def test_phase_delta_rule_scales_rear_spring_with_lateral_bias_and_low_frequency() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="exit",
        operator_label="Operador de salida",
        category="exit",
        phase_label="salida",
        priority=24,
        reference_key="differential",
    )
    goal = Goal(
        phase="exit",
        archetype="medium",
        description="",
        target_delta_nfr=0.3,
        target_sense_index=0.88,
        nu_f_target=0.26,
        nu_exc_target=0.2,
        rho_target=0.85,
        target_phase_lag=0.0,
        target_phase_alignment=0.86,
        measured_phase_lag=0.08,
        measured_phase_alignment=0.81,
        slip_lat_window=(-0.4, 0.4),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.5, 0.5),
        dominant_nodes=("suspension",),
        target_delta_nfr_long=0.08,
        target_delta_nfr_lat=0.14,
    )
    sample_indices = tuple(range(3))
    microsector = Microsector(
        index=14,
        start_time=0.0,
        end_time=0.3,
        curvature=1.2,
        brake_event=False,
        support_event=False,
        delta_nfr_signature=0.34,
        goals=(goal,),
        phase_boundaries={goal.phase: (0, len(sample_indices))},
        phase_samples={goal.phase: sample_indices},
        active_phase=goal.phase,
        dominant_nodes={goal.phase: goal.dominant_nodes},
        phase_weights={goal.phase: {"__default__": 1.0}},
        grip_rel=1.0,
        phase_lag={goal.phase: goal.measured_phase_lag},
        phase_alignment={goal.phase: goal.measured_phase_alignment},
        filtered_measures={},
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={goal.phase: {}},
        operator_events={},
    )
    bundles: list[EPIBundle] = []
    for lat_component in (0.22, 0.24, 0.23):
        bundle = _axis_bundle(0.36, 0.05, lat_component)
        suspension = replace(bundle.suspension, nu_f=0.18)
        bundles.append(replace(bundle, suspension=suspension))
    thresholds = ThresholdProfile(0.1, 0.05, 0.08, 0.2, 0.5)
    context = RuleContext(car_model="GT3", track_name="VAL", thresholds=thresholds)
    recommendations = list(rule.evaluate(bundles, [microsector], context))
    rear_spring_recs = [rec for rec in recommendations if rec.parameter == "rear_spring_stiffness"]
    assert rear_spring_recs, "expected rear spring recommendation for exit phase"
    assert all(rec.delta is not None and rec.delta > 0 for rec in rear_spring_recs)
    assert all("νf_susp" in rec.message for rec in rear_spring_recs)
    assert all("ΔNFR⊥" in rec.rationale for rec in rear_spring_recs)


def test_phase_node_rule_prioritises_geometry_with_alignment_gap() -> None:
    rule = PhaseNodeOperatorRule(
        phase="apex",
        operator_label="Operador de vértice",
        category="apex",
        priority=24,
        reference_key="antiroll",
    )
    goal = Goal(
        phase="apex",
        archetype="hairpin",
        description="",
        target_delta_nfr=0.15,
        target_sense_index=0.88,
        nu_f_target=0.32,
        nu_exc_target=0.27,
        rho_target=0.9,
        target_phase_lag=0.0,
        target_phase_alignment=0.92,
        measured_phase_lag=0.21,
        measured_phase_alignment=0.66,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.4, 0.4),
        dominant_nodes=("suspension", "tyres"),
    )
    microsector = Microsector(
        index=6,
        start_time=0.0,
        end_time=0.4,
        curvature=1.5,
        brake_event=False,
        support_event=True,
        delta_nfr_signature=0.3,
        goals=(goal,),
        phase_boundaries={"apex": (0, 4)},
        phase_samples={"apex": (0, 1, 2, 3)},
        active_phase="apex",
        dominant_nodes={"apex": ("suspension", "tyres")},
        phase_weights={"apex": {"__default__": 1.0}},
        grip_rel=1.0,
        phase_lag={"apex": 0.21},
        phase_alignment={"apex": 0.66},
        filtered_measures={"coherence_index": 0.35},
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy={"apex_r": {}},
        operator_events={},
    )
    results = []
    for value in (0.35, 0.33, 0.36, 0.34):
        bundle = _axis_bundle(value, 0.14, 0.21)
        results.append(replace(bundle, coherence_index=0.35))
    thresholds = ThresholdProfile(0.4, 0.3, 0.4, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    geometry_params = {"front_camber_deg", "rear_camber_deg", "front_toe_deg", "rear_toe_deg"}
    geometry_recs = [rec for rec in recommendations if rec.parameter in geometry_params]
    assert geometry_recs, "expected geometry reinforcement recommendations"
    assert any("camber" in rec.message.lower() or "toe" in rec.message.lower() for rec in geometry_recs)
    assert any("C(t)" in rec.rationale for rec in geometry_recs)
    assert all(rec.priority <= rule.priority - 1 for rec in geometry_recs)
