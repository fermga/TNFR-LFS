from pathlib import Path

import pytest

from typing import Any, Callable, Iterable, Sequence, Tuple
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from statistics import mean

from tnfr_core.epi_models import EPIBundle
from collections.abc import Mapping

from tnfr_core.segmentation import Goal, Microsector
from tnfr_core.operator_detection import canonical_operator_label
from tnfr_lfs.ingestion.offline import AeroProfile, ProfileManager
from tnfr_lfs.recommender.rules import (
    AeroCoherenceRule,
    FrontWingBalanceRule,
    BrakeHeadroomRule,
    BottomingPriorityRule,
    DetuneRatioRule,
    LockingWindowRule,
    MANUAL_REFERENCES,
    PhaseDeltaDeviationRule,
    PhaseNodeOperatorRule,
    ParallelSteerRule,
    Recommendation,
    RecommendationEngine,
    RuleProfileObjectives,
    RuleContext,
    FootprintEfficiencyRule,
    TyreBalanceRule,
    UsefulDissonanceRule,
    ThresholdProfile,
)
from tnfr_core.operators import tyre_balance_controller

from tests.helpers import (
    BASE_NU_F,
    _brake_headroom_microsector,
    _entry_goal_with_gradient,
    _entry_microsector_with_gradient,
    _entry_results_with_gradient,
    _udr_goal,
    _udr_microsector,
    build_axis_bundle,
    build_goal,
    build_microsector,
    build_epi_nodes,
    build_node_bundle,
    build_parallel_window_metrics,
    build_steering_bundle,
    build_steering_record,
    build_udr_bundle_series,
)


@dataclass(frozen=True)
class RuleCase:
    """Container describing a scenario for rule-based recommendations."""

    description: str
    goal_overrides: Mapping[str, Any] = field(default_factory=dict)
    microsector_overrides: Mapping[str, Any] = field(default_factory=dict)
    context_overrides: Mapping[str, Any] = field(default_factory=dict)
    rule_kwargs: Mapping[str, Any] = field(default_factory=dict)
    expected_parameters: tuple[str, ...] = ()
    expected_deltas: Mapping[str, float] = field(default_factory=dict)
    message_contains: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    forbidden_parameters: tuple[str, ...] = ()
    expected_count: int | None = None


def _assert_rule_outcome(
    recommendations: Sequence[Recommendation],
    case: RuleCase,
) -> None:
    """Validate rule recommendations against the declarative case definition."""

    if not case.expected_parameters:
        assert not recommendations
    else:
        assert recommendations
        if case.expected_count is not None:
            assert len(recommendations) == case.expected_count
        for parameter in case.expected_parameters:
            match = next((rec for rec in recommendations if rec.parameter == parameter), None)
            assert match is not None
            if parameter in case.expected_deltas:
                assert match.delta == pytest.approx(case.expected_deltas[parameter])
            for needle in case.message_contains.get(parameter, ()):  # pragma: no branch - simple loop
                assert needle.lower() in match.message.lower()
    for forbidden in case.forbidden_parameters:
        assert all(rec.parameter != forbidden for rec in recommendations)

TYRE_BALANCE_CASES = [
    RuleCase(
        description="balanced-microsector-no-guidance",
    ),
    RuleCase(
        description="suppresses-low-dispersion",
        microsector_overrides={
            "support_event": False,
            "goals": (),
            "phase_boundaries": {"apex": (0, 3)},
            "phase_samples": {"apex": (0, 1, 2)},
            "dominant_nodes": {"apex": ()},
            "phase_weights": {"apex": {}},
            "phase_lag": {},
            "phase_alignment": {},
            "filtered_measures": {
                "thermal_load": 5150.0,
                "style_index": 0.82,
                "grip_rel": 1.0,
                "d_nfr_flat": -0.32,
            },
            "window_occupancy": {"apex": {}},
            "operator_events": {},
        },
    ),
    RuleCase(
        description="neutral-when-cphi-missing",
        microsector_overrides={
            "index": 4,
            "filtered_measures": {
                "thermal_load": 5200.0,
                "style_index": 0.83,
                "grip_rel": 1.0,
                "d_nfr_flat": -0.28,
                "cphi_fl": float("nan"),
                "cphi_fr": None,
                "cphi_rl": float("nan"),
                "cphi_rr": None,
                "cphi_fl_temperature": None,
                "cphi_fr_temperature": float("nan"),
                "cphi_rl_temperature": None,
                "cphi_rr_temperature": float("nan"),
                "cphi_fl_gradient": None,
                "cphi_fr_gradient": float("nan"),
                "cphi_rl_gradient": None,
                "cphi_rr_gradient": float("nan"),
                "cphi_fl_mu": None,
                "cphi_fr_mu": float("nan"),
                "cphi_rl_mu": None,
                "cphi_rr_mu": float("nan"),
                "cphi_fl_temp_delta": None,
                "cphi_fr_temp_delta": float("nan"),
                "cphi_rl_temp_delta": None,
                "cphi_rr_temp_delta": float("nan"),
                "cphi_fl_gradient_rate": None,
                "cphi_fr_gradient_rate": float("nan"),
                "cphi_rl_gradient_rate": None,
                "cphi_rr_gradient_rate": float("nan"),
            },
        },
        context_overrides={"tyre_offsets": {}},
    ),
    RuleCase(
        description="skips-when-cphi-healthy",
        microsector_overrides={
            "index": 6,
            "filtered_measures": {
                "thermal_load": 5100.0,
                "style_index": 0.84,
                "grip_rel": 1.0,
                "d_nfr_flat": -0.22,
            },
        },
    ),
]

TYRE_BALANCE_IDS = [case.description for case in TYRE_BALANCE_CASES]


_parallel_negative_metrics = build_parallel_window_metrics(
    [(0.04, 0.015), (0.034, 0.018), (0.03, 0.02)],
    yaw_sign=1.0,
)
_parallel_positive_metrics = build_parallel_window_metrics(
    [(0.082, 0.012), (0.078, 0.015), (0.074, 0.018)],
    yaw_sign=1.0,
)
_parallel_budget_metrics = build_parallel_window_metrics(
    [(0.02, 0.019), (0.022, 0.0215), (0.18, 0.01)],
    yaw_sign=1.0,
    yaw_rates=[0.0, 1.5, -1.5],
    steer_series=[0.0, 2.5, -2.5],
)

PARALLEL_STEER_CASES = [
    RuleCase(
        description="increase-parallel-steer-on-negative-index",
        microsector_overrides={
            "index": 7,
            "filtered_measures": {
                "ackermann_parallel_index": -0.12,
                "slide_catch_budget": _parallel_negative_metrics.slide_catch_budget.value,
                "slide_catch_budget_yaw": _parallel_negative_metrics.slide_catch_budget.yaw_acceleration_ratio,
                "slide_catch_budget_steer": _parallel_negative_metrics.slide_catch_budget.steer_velocity_ratio,
                "slide_catch_budget_overshoot": _parallel_negative_metrics.slide_catch_budget.overshoot_ratio,
            },
        },
        rule_kwargs={"priority": 16, "threshold": 0.05, "delta_step": 0.2},
        expected_parameters=("parallel_steer",),
        expected_deltas={"parallel_steer": 0.2},
        message_contains={"parallel_steer": ("parallel steer",)},
        forbidden_parameters=("steering_lock_deg",),
        expected_count=1,
    ),
    RuleCase(
        description="reduce-parallel-steer-on-positive-index",
        microsector_overrides={
            "index": 5,
            "filtered_measures": {
                "ackermann_parallel_index": 0.11,
                "slide_catch_budget": _parallel_positive_metrics.slide_catch_budget.value,
                "slide_catch_budget_yaw": _parallel_positive_metrics.slide_catch_budget.yaw_acceleration_ratio,
                "slide_catch_budget_steer": _parallel_positive_metrics.slide_catch_budget.steer_velocity_ratio,
                "slide_catch_budget_overshoot": _parallel_positive_metrics.slide_catch_budget.overshoot_ratio,
            },
        },
        rule_kwargs={"priority": 16, "threshold": 0.05, "delta_step": 0.15},
        expected_parameters=("parallel_steer",),
        expected_deltas={"parallel_steer": -0.15},
        message_contains={"parallel_steer": ("parallel steer",)},
        forbidden_parameters=("steering_lock_deg",),
        expected_count=1,
    ),
    RuleCase(
        description="add-lock-when-budget-limited",
        microsector_overrides={
            "index": 3,
            "filtered_measures": {
                "ackermann_parallel_index": -0.16,
                "slide_catch_budget": _parallel_budget_metrics.slide_catch_budget.value,
                "slide_catch_budget_yaw": _parallel_budget_metrics.slide_catch_budget.yaw_acceleration_ratio,
                "slide_catch_budget_steer": _parallel_budget_metrics.slide_catch_budget.steer_velocity_ratio,
                "slide_catch_budget_overshoot": _parallel_budget_metrics.slide_catch_budget.overshoot_ratio,
            },
        },
        rule_kwargs={
            "priority": 16,
            "threshold": 0.05,
            "delta_step": 0.1,
            "lock_step": 0.75,
        },
        expected_parameters=("parallel_steer", "steering_lock_deg"),
        expected_deltas={"parallel_steer": 0.1, "steering_lock_deg": 0.75},
        message_contains={
            "parallel_steer": ("parallel steer",),
            "steering_lock_deg": ("steering lock",),
        },
        expected_count=2,
    ),
]

PARALLEL_STEER_IDS = [case.description for case in PARALLEL_STEER_CASES]


LOCKING_WINDOW_CASES = [
    RuleCase(
        description="open-power-lock-when-on-threshold-crossed",
        microsector_overrides={
            "index": 4,
            "active_phase": "exit",
            "filtered_measures": {
                "locking_window_score": 0.35,
                "locking_window_score_on": 0.4,
                "locking_window_score_off": 0.82,
                "locking_window_transitions": 3,
            },
        },
        rule_kwargs={
            "priority": 27,
            "on_threshold": 0.7,
            "off_threshold": 0.6,
            "min_transitions": 1,
            "power_lock_step": 6.0,
        },
        expected_parameters=("diff_power_lock",),
        expected_deltas={"diff_power_lock": -6.0},
        message_contains={"diff_power_lock": ("lsd",)},
        expected_count=1,
    ),
    RuleCase(
        description="reduce-preload-when-locking-high",
        microsector_overrides={
            "index": 6,
            "active_phase": "exit",
            "filtered_measures": {
                "locking_window_score": 0.42,
                "locking_window_score_on": 0.72,
                "locking_window_score_off": 0.55,
                "locking_window_transitions": 4,
            },
        },
        rule_kwargs={
            "priority": 29,
            "on_threshold": 0.5,
            "off_threshold": 0.7,
            "min_transitions": 2,
            "preload_step": 50.0,
        },
        expected_parameters=("diff_preload_nm",),
        expected_deltas={"diff_preload_nm": -50.0},
        message_contains={"diff_preload_nm": ("reduce preload",)},
        expected_count=1,
    ),
    RuleCase(
        description="skip-when-transitions-low",
        microsector_overrides={
            "index": 2,
            "active_phase": "exit",
            "filtered_measures": {
                "locking_window_score": 0.3,
                "locking_window_score_on": 0.4,
                "locking_window_score_off": 0.45,
                "locking_window_transitions": 1,
            },
        },
        rule_kwargs={"min_transitions": 3},
    ),
]

LOCKING_WINDOW_IDS = [case.description for case in LOCKING_WINDOW_CASES]


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
            "bumpstop_front_density": 0.12,
            "bumpstop_front_energy": 0.18,
            "bumpstop_front_density_bin_0": 0.12,
            "bumpstop_front_energy_bin_0": 0.09,
            "bumpstop_rear_density": 0.05,
            "bumpstop_rear_energy": 0.04,
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
            "bumpstop_front_density": 0.08,
            "bumpstop_front_energy": 0.12,
            "bumpstop_rear_density": 0.32,
            "bumpstop_rear_energy": 1.2,
            "bumpstop_rear_density_bin_2": 0.21,
            "bumpstop_rear_energy_bin_2": 0.8,
        },
        index=3,
    )
    smooth = replace(smooth, context_factors={"surface": 0.94})
    rough = replace(rough, context_factors={"surface": 1.28})
    return smooth, rough


def test_recommendation_engine_uses_session_weights() -> None:
    captured_contexts: list[RuleContext] = []

    class CaptureRule:
        def evaluate(
            self,
            results: Sequence[EPIBundle],
            microsectors: Sequence[Microsector] | None = None,
            context: RuleContext | None = None,
        ) -> Iterable[Recommendation]:
            if context is not None:
                captured_contexts.append(context)
            return []

    engine = RecommendationEngine(
        rules=[CaptureRule()],
        car_model="generic",
        track_name="generic",
    )

    engine.session = {
        "weights": {
            "entry": {"__default__": 1.0, "brakes": "1.25"},
            "exit": {"__default__": 0.9},
        },
        "hints": {"slip_ratio_bias": "aggressive"},
    }

    result = engine.generate([])

    assert result == []
    assert captured_contexts
    context = captured_contexts[0]
    assert context.session_weights["entry"]["brakes"] == pytest.approx(1.25)
    assert context.session_weights["exit"]["__default__"] == pytest.approx(0.9)
    assert context.session_hints["slip_ratio_bias"] == "aggressive"


@pytest.mark.parametrize("case", TYRE_BALANCE_CASES, ids=TYRE_BALANCE_IDS)
def test_tyre_balance_rule_scenarios(
    rule_scenario_factory: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], tuple[RuleContext, Goal, Microsector]],
    case: RuleCase,
) -> None:
    context, _, microsector = rule_scenario_factory(
        goal_overrides=dict(case.goal_overrides),
        microsector_overrides=dict(case.microsector_overrides),
        context_overrides=dict(case.context_overrides),
    )
    rule_kwargs = {"priority": 18}
    rule_kwargs.update(dict(case.rule_kwargs))
    rule = TyreBalanceRule(**rule_kwargs)

    recommendations = list(rule.evaluate([], [microsector], context))
    _assert_rule_outcome(recommendations, case)

@pytest.mark.parametrize("case", PARALLEL_STEER_CASES, ids=PARALLEL_STEER_IDS)
def test_parallel_steer_rule_scenarios(
    rule_scenario_factory: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], tuple[RuleContext, Goal, Microsector]],
    case: RuleCase,
) -> None:
    context, _, microsector = rule_scenario_factory(
        goal_overrides=dict(case.goal_overrides),
        microsector_overrides=dict(case.microsector_overrides),
        context_overrides=dict(case.context_overrides),
    )
    rule_kwargs = {"priority": 16, "threshold": 0.05, "delta_step": 0.2}
    rule_kwargs.update(dict(case.rule_kwargs))
    rule = ParallelSteerRule(**rule_kwargs)

    recommendations = list(rule.evaluate([], [microsector], context))
    _assert_rule_outcome(recommendations, case)


@pytest.mark.parametrize("case", LOCKING_WINDOW_CASES, ids=LOCKING_WINDOW_IDS)
def test_locking_window_rule_scenarios(
    rule_scenario_factory: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], tuple[RuleContext, Goal, Microsector]],
    case: RuleCase,
) -> None:
    context, _, microsector = rule_scenario_factory(
        goal_overrides=dict(case.goal_overrides),
        microsector_overrides=dict(case.microsector_overrides),
        context_overrides=dict(case.context_overrides),
    )
    rule = LockingWindowRule(**dict(case.rule_kwargs))

    recommendations = list(rule.evaluate([], [microsector], context))
    _assert_rule_outcome(recommendations, case)


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
            operator_events={"SILENCE": quiet_payload},
            index=i,
        )
        for i in range(3)
    ]
    bundles = build_udr_bundle_series([0.05, 0.04, 0.03, 0.02, 0.01])
    engine = RecommendationEngine(rules=[])
    recommendations = engine.generate(bundles, microsectors)
    assert len(recommendations) == 1
    message = recommendations[0].message.lower()
    rationale = recommendations[0].rationale.lower()
    assert "do not adjust" in message
    assert "silence" in rationale or "quiet" in rationale


def test_aero_coherence_rule_flags_high_speed_bias() -> None:
    thresholds = ThresholdProfile(
        entry_delta_tolerance=0.6,
        apex_delta_tolerance=0.6,
        exit_delta_tolerance=0.6,
        piano_delta_tolerance=0.5,
        rho_detune_threshold=0.4,
    )
    context = RuleContext(
        car_model="FZR",
        track_name="AS5",
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
    assert "High-speed microsector" in rec.message
    assert "rear axle load" in rec.rationale.lower()
    assert "C(c/d/a)" in rec.rationale


def test_front_wing_balance_rule_targets_front_limited_bias() -> None:
    thresholds = ThresholdProfile(
        entry_delta_tolerance=0.6,
        apex_delta_tolerance=0.6,
        exit_delta_tolerance=0.6,
        piano_delta_tolerance=0.5,
        rho_detune_threshold=0.4,
    )
    context = RuleContext(
        car_model="FZR",
        track_name="AS5",
        thresholds=thresholds,
        tyre_offsets={},
        aero_profiles={"race": AeroProfile(low_speed_target=0.0, high_speed_target=0.0)},
    )
    microsector = SimpleNamespace(
        index=7,
        filtered_measures={
            "aero_high_imbalance": -0.32,
            "aero_high_samples": 6,
            "aero_mechanical_coherence": 0.52,
            "aero_high_front_total": 0.12,
            "aero_high_rear_total": 0.36,
            "aero_high_front_lateral": 0.08,
            "aero_high_rear_lateral": 0.22,
            "aero_high_front_longitudinal": 0.04,
            "aero_high_rear_longitudinal": 0.14,
            "aero_drift_high_mu_delta": -0.18,
            "aero_drift_mu_tolerance": 0.05,
        },
    )

    rule = FrontWingBalanceRule(high_speed_threshold=0.1, min_high_samples=2, max_aero_mechanical=0.6, delta_step=0.6)

    recommendations = list(rule.evaluate([], [microsector], context))

    assert recommendations
    rec = recommendations[0]
    assert rec.parameter == "front_wing_angle"
    assert rec.delta and rec.delta > 0
    assert "increase front wing angle" in rec.message.lower()
    assert "front axle load" in rec.rationale.lower()


def test_aero_coherence_rule_respects_low_speed_window() -> None:
    thresholds = ThresholdProfile(
        entry_delta_tolerance=0.6,
        apex_delta_tolerance=0.6,
        exit_delta_tolerance=0.6,
        piano_delta_tolerance=0.5,
        rho_detune_threshold=0.4,
    )
    context = RuleContext(
        car_model="FZR",
        track_name="AS5",
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
    results = [
        EPIBundle(timestamp=0.0, epi=0.4, delta_nfr=15.0, sense_index=0.55, **build_epi_nodes(15.0, 0.55)),
        EPIBundle(timestamp=0.1, epi=0.5, delta_nfr=-12.0, sense_index=0.50, **build_epi_nodes(-12.0, 0.50)),
        EPIBundle(timestamp=0.2, epi=0.6, delta_nfr=2.0, sense_index=0.90, **build_epi_nodes(2.0, 0.90)),
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
        node_deltas = {
            "tyres": tyre_delta,
            "suspension": delta_nfr / 2,
            "chassis": delta_nfr / 2,
            "brakes": delta_nfr / 2,
            "transmission": delta_nfr / 2,
            "track": delta_nfr / 2,
            "driver": delta_nfr / 2,
        }
        return build_node_bundle(
            timestamp=timestamp,
            epi=0.5,
            delta_nfr=delta_nfr,
            sense_index=sense_index,
            delta_nfr_by_node=node_deltas,
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
    goals = (
        build_goal(
            "entry",
            entry_target,
            archetype="hairpin",
            description="",
            nu_f_target=0.25,
            nu_exc_target=0.25,
            rho_target=1.0,
            slip_lat_window=window,
            slip_long_window=window,
            yaw_rate_window=yaw_window,
            dominant_nodes=entry_nodes,
        ),
        build_goal(
            "apex",
            apex_target,
            archetype="hairpin",
            description="",
            nu_f_target=0.25,
            nu_exc_target=0.25,
            rho_target=1.0,
            slip_lat_window=window,
            slip_long_window=window,
            yaw_rate_window=yaw_window,
            dominant_nodes=apex_nodes,
        ),
        build_goal(
            "exit",
            exit_target,
            archetype="hairpin",
            description="",
            nu_f_target=0.25,
            nu_exc_target=0.25,
            rho_target=1.0,
            slip_lat_window=window,
            slip_long_window=window,
            yaw_rate_window=yaw_window,
            dominant_nodes=exit_nodes,
        ),
    )
    microsector = build_microsector(
        index=3,
        start_time=0.0,
        end_time=0.6,
        curvature=1.8,
        brake_event=True,
        support_event=True,
        delta_nfr_signature=0.0,
        phases=("entry", "apex", "exit"),
        goals=goals,
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
        phase_lag=phase_lag,
        phase_alignment=phase_alignment,
        filtered_measures=filtered_measures,
        window_occupancy=window_occupancy,
        operator_events={},
        include_cphi=False,
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
        car_model="FZR",
        track_name="AS5",
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
    assert any("anti-roll" in message.lower() for message in apex_messages)
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
        car_model="FZR",
        track_name="AS5",
        profile_manager=manager,
    )

    baseline_context = engine._resolve_context("FZR", "AS5")
    before_weights = baseline_context.thresholds.weights_for_phase("entry")
    if isinstance(before_weights, Mapping):
        entry_before = float(before_weights.get("__default__", 1.0))
    else:
        entry_before = float(before_weights)

    engine.register_stint_result(
        sense_index=0.7,
        delta_nfr=2.5,
        car_model="FZR",
        track_name="AS5",
    )

    recommendations = [
        Recommendation(category="entry", message="", rationale=""),
        Recommendation(category="apex", message="", rationale=""),
    ]

    engine.register_plan(
        recommendations,
        car_model="FZR",
        track_name="AS5",
        baseline_sense_index=0.7,
        baseline_delta_nfr=2.5,
    )

    engine.register_stint_result(
        sense_index=0.78,
        delta_nfr=1.4,
        car_model="FZR",
        track_name="AS5",
    )

    updated_context = engine._resolve_context("FZR", "AS5")
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
[profiles.FZR.AS5.objectives]
target_delta_nfr = 0.4
target_sense_index = 0.82

[profiles.FZR.AS5.tolerances]
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

    snapshot = manager.resolve("FZR", "AS5")
    entry_snapshot = snapshot.phase_weights.get("entry", {})
    assert entry_snapshot.get("tyres") == pytest.approx(1.2)

    manager.save()

    reloaded = ProfileManager(
        profiles_path, threshold_library=car_track_thresholds
    )
    context_engine = RecommendationEngine(
        car_model="FZR",
        track_name="AS5",
        threshold_library=car_track_thresholds,
        profile_manager=reloaded,
    )._resolve_context("FZR", "AS5")
    entry_weights = context_engine.thresholds.weights_for_phase("entry")
    assert isinstance(entry_weights, Mapping)
    assert entry_weights.get("tyres") == pytest.approx(1.2)

def test_threshold_profile_exposes_phase_weights():
    engine = RecommendationEngine(car_model="FZR", track_name="AS5")
    profile = engine._resolve_context("FZR", "AS5").thresholds  # type: ignore[attr-defined]
    entry_weights = profile.weights_for_phase("entry")
    assert isinstance(entry_weights, Mapping)
    assert entry_weights.get("tyres", 0.0) > 1.0
    apex_weights = profile.weights_for_phase("apex")
    assert isinstance(apex_weights, Mapping)
    assert apex_weights.get("suspension", 0.0) >= 1.0


def test_track_specific_profile_tightens_entry_threshold():
    def build_bundle(index: int, delta_nfr: float, sense_index: float = 0.9) -> EPIBundle:
        quarter = delta_nfr / 4
        node_deltas = {
            "tyres": delta_nfr / 2,
            "suspension": quarter,
            "chassis": quarter,
            "brakes": quarter,
            "transmission": quarter,
            "track": quarter,
            "driver": quarter,
        }
        return build_node_bundle(
            timestamp=index * 0.1,
            epi=0.5,
            delta_nfr=delta_nfr,
            sense_index=sense_index,
            delta_nfr_by_node=node_deltas,
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
    goals = (
        build_goal(
            "entry",
            0.0,
            archetype="hairpin",
            description="",
            nu_f_target=0.25,
            nu_exc_target=0.25,
            rho_target=1.0,
            slip_lat_window=window,
            slip_long_window=window,
            yaw_rate_window=yaw_window,
            dominant_nodes=nodes,
        ),
        build_goal(
            "apex",
            0.2,
            archetype="hairpin",
            description="",
            nu_f_target=0.25,
            nu_exc_target=0.25,
            rho_target=1.0,
            slip_lat_window=window,
            slip_long_window=window,
            yaw_rate_window=yaw_window,
            dominant_nodes=nodes,
        ),
        build_goal(
            "exit",
            -0.1,
            archetype="hairpin",
            description="",
            nu_f_target=0.25,
            nu_exc_target=0.25,
            rho_target=1.0,
            slip_lat_window=window,
            slip_long_window=window,
            yaw_rate_window=yaw_window,
            dominant_nodes=nodes,
        ),
    )
    microsector = build_microsector(
        index=4,
        start_time=0.0,
        end_time=0.6,
        curvature=1.5,
        brake_event=True,
        support_event=True,
        delta_nfr_signature=0.0,
        phases=("entry", "apex", "exit"),
        goals=goals,
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
        phase_lag=phase_lag,
        phase_alignment=phase_alignment,
        filtered_measures=filtered_measures,
        window_occupancy=window_occupancy,
        operator_events={},
        include_cphi=False,
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
    AS5_engine = RecommendationEngine(car_model="FZR", track_name="AS5")
    AS5_recs = AS5_engine.generate(results, [microsector])

    generic_entry_global = [
        rec for rec in generic_recs if rec.category == "entry" and "global ΔNFR" in rec.message
    ]
    AS5_entry_global = [
        rec for rec in AS5_recs if rec.category == "entry" and "global ΔNFR" in rec.message
    ]

    assert not generic_entry_global
    assert AS5_entry_global


def test_node_operator_rule_responds_to_nu_f_excess(car_track_thresholds):
    def build_bundle(timestamp: float, transmission_nu_f: float) -> EPIBundle:
        node_deltas = {
            "tyres": 0.15,
            "suspension": 0.15,
            "chassis": 0.15,
            "brakes": 0.15,
            "transmission": 0.15,
            "track": 0.15,
            "driver": 0.15,
        }
        return build_node_bundle(
            timestamp=timestamp,
            epi=0.5,
            delta_nfr=0.3,
            sense_index=0.92,
            delta_nfr_by_node=node_deltas,
            overrides={"transmission": {"nu_f": transmission_nu_f}},
        )

    window = (-0.05, 0.05)
    yaw_window = (-0.3, 0.3)
    exit_nodes = ("transmission",)
    phase_lag = {"entry": 0.0, "apex": 0.0, "exit": 0.0}
    phase_alignment = {"entry": 1.0, "apex": 1.0, "exit": 1.0}
    goals = (
        build_goal(
            "entry",
            0.0,
            archetype="transition",
            description="",
            nu_f_target=0.1,
            nu_exc_target=0.1,
            rho_target=1.0,
            slip_lat_window=window,
            slip_long_window=window,
            yaw_rate_window=yaw_window,
            dominant_nodes=("tyres",),
        ),
        build_goal(
            "apex",
            0.1,
            archetype="transition",
            description="",
            nu_f_target=0.15,
            nu_exc_target=0.15,
            rho_target=1.0,
            slip_lat_window=window,
            slip_long_window=window,
            yaw_rate_window=yaw_window,
            dominant_nodes=("suspension",),
        ),
        build_goal(
            "exit",
            -0.05,
            archetype="traction",
            description="",
            nu_f_target=0.2,
            nu_exc_target=0.2,
            rho_target=1.0,
            slip_lat_window=window,
            slip_long_window=window,
            yaw_rate_window=yaw_window,
            dominant_nodes=exit_nodes,
        ),
    )
    microsector = build_microsector(
        index=7,
        start_time=0.0,
        end_time=0.6,
        curvature=1.2,
        brake_event=False,
        support_event=False,
        delta_nfr_signature=0.0,
        phases=("entry", "apex", "exit"),
        goals=goals,
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
        phase_lag=phase_lag,
        phase_alignment=phase_alignment,
        filtered_measures={
            "thermal_load": 5000.0,
            "style_index": 0.9,
            "grip_rel": 1.0,
        },
        window_occupancy={
            "entry": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
            "apex": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
            "exit": {"slip_lat": 100.0, "slip_long": 100.0, "yaw_rate": 100.0},
        },
        operator_events={},
        include_cphi=False,
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
        car_model="FZR",
        track_name="AS5",
        threshold_library=car_track_thresholds,
    )
    recommendations = engine.generate(results, [microsector])

    exit_messages = [
        rec
        for rec in recommendations
        if rec.category == "exit" and "traction operator" in rec.message.lower()
    ]
    assert exit_messages
    assert any("power locking" in rec.message.lower() for rec in exit_messages)

    rationale = exit_messages[0].rationale
    assert "differential" in rationale.lower()
    assert "mean ν_f" in rationale
    assert "FZR/AS5" in rationale


def test_phase_node_rule_flips_with_phase_misalignment(car_track_thresholds) -> None:
    engine = RecommendationEngine(
        car_model="FZR", track_name="AS5", threshold_library=car_track_thresholds
    )
    context = engine._resolve_context("FZR", "AS5")
    rule = PhaseNodeOperatorRule(
        phase="apex",
        operator_label="Apex operator",
        category="apex",
        priority=25,
        reference_key="antiroll",
    )

    def _bundle(nu_f: float, timestamp: float) -> EPIBundle:
        node_deltas = {
            "tyres": 0.2,
            "suspension": 0.4,
            "chassis": 0.2,
            "brakes": 0.05,
            "transmission": 0.05,
            "track": 0.05,
            "driver": 0.05,
        }
        return build_node_bundle(
            timestamp=timestamp,
            epi=0.0,
            delta_nfr=0.8,
            sense_index=0.82,
            delta_nfr_by_node=node_deltas,
            overrides={"suspension": {"nu_f": nu_f}},
        )

    results = [_bundle(1.5, 0.0), _bundle(1.55, 0.1)]
    goal = build_goal(
        "apex",
        0.2,
        archetype="hairpin",
        description="",
        target_sense_index=0.88,
        nu_f_target=0.25,
        nu_exc_target=0.25,
        rho_target=1.0,
        measured_phase_lag=0.6,
        measured_phase_alignment=-0.4,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.3, 0.3),
        dominant_nodes=("suspension",),
    )
    microsector = build_microsector(
        index=12,
        start_time=0.0,
        end_time=0.2,
        curvature=1.4,
        brake_event=False,
        support_event=False,
        delta_nfr_signature=0.4,
        phases=("apex",),
        goals=(goal,),
        phase_boundaries={"apex": (0, 2)},
        phase_samples={"apex": (0, 1)},
        active_phase="apex",
        dominant_nodes={"apex": ("suspension",)},
        phase_weights={"apex": {"__default__": 1.0}},
        phase_lag={"apex": 0.6},
        phase_alignment={"apex": -0.4},
        filtered_measures={"thermal_load": 5000.0, "style_index": 0.8},
        window_occupancy={"apex": {"slip_lat": 75.0, "slip_long": 70.0, "yaw_rate": 68.0}},
        operator_events={},
        include_cphi=False,
    )

    recommendations = list(rule.evaluate(results, [microsector], context=context))

    rebound_actions = [rec for rec in recommendations if rec.parameter == "rear_rebound_clicks"]
    assert rebound_actions, "expected rebound click recommendation"
    assert rebound_actions[0].delta > 0
    assert "θ" in rebound_actions[0].rationale
    assert "Siφ" in rebound_actions[0].rationale
    assert "direction is inverted" in rebound_actions[0].rationale.lower()


def test_detune_ratio_rule_emits_modal_guidance() -> None:
    goal = build_goal(
        "apex",
        0.3,
        archetype="hairpin",
        description="",
        target_sense_index=0.85,
        nu_f_target=0.28,
        nu_exc_target=0.22,
        rho_target=0.78,
        measured_phase_alignment=0.9,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.3, 0.3),
        dominant_nodes=("suspension", "chassis"),
        detune_ratio_weights={"longitudinal": 0.7, "lateral": 0.3},
    )
    microsector = build_microsector(
        index=4,
        start_time=0.0,
        end_time=0.4,
        curvature=1.6,
        brake_event=False,
        support_event=True,
        delta_nfr_signature=0.4,
        phases=("apex",),
        goals=(goal,),
        phase_boundaries={"apex": (0, 3)},
        phase_samples={"apex": (0, 1, 2)},
        active_phase="apex",
        dominant_nodes={"apex": ("suspension",)},
        phase_weights={"apex": {"__default__": 1.0}},
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
        window_occupancy={"apex": {"slip_lat": 80.0, "slip_long": 78.0, "yaw_rate": 72.0}},
        operator_events={},
        include_cphi=False,
    )

    engine = RecommendationEngine(rules=[DetuneRatioRule(priority=42)])
    recommendations = engine.generate([], [microsector])

    assert recommendations, "expected detune ratio warning"
    rationale = recommendations[0].rationale.lower()
    assert "ρ=" in recommendations[0].rationale
    assert "anti-roll" in rationale
    assert "detune" in rationale
    assert "longitudinal" in rationale


def test_useful_dissonance_rule_reinforces_rear_when_udr_high(car_track_thresholds):
    values = [1.4, 1.5, 1.3]
    results = build_udr_bundle_series(values)
    goal = _udr_goal("apex", target_delta=0.2)
    microsector = _udr_microsector(goal, udr=0.8, sample_count=len(results))

    engine = RecommendationEngine(
        rules=[UsefulDissonanceRule(priority=40)],
        car_model="FZR",
        track_name="AS5",
        threshold_library=car_track_thresholds,
    )

    recommendations = engine.generate(results, [microsector])

    assert recommendations, "expected UDR escalation recommendation"
    messages = [rec.message.lower() for rec in recommendations]
    assert any("reinforce" in message and "rear" in message for message in messages)
    rationales = " ".join(rec.rationale.lower() for rec in recommendations)
    assert "udr" in rationales


def test_useful_dissonance_rule_softens_axle_when_udr_low(car_track_thresholds):
    values = [1.1, 1.2, 1.05]
    results = build_udr_bundle_series(values)
    goal = _udr_goal("apex", target_delta=0.2)
    microsector = _udr_microsector(goal, udr=0.1, sample_count=len(results))

    engine = RecommendationEngine(
        rules=[UsefulDissonanceRule(priority=38)],
        car_model="FZR",
        track_name="AS5",
        threshold_library=car_track_thresholds,
    )

    recommendations = engine.generate(results, [microsector])

    assert recommendations, "expected UDR softening recommendation"
    messages = [rec.message.lower() for rec in recommendations]
    assert any("soften front" in message for message in messages)

    # Oversteer scenario should target the rear axle.
    oversteer_results = build_udr_bundle_series([-1.2, -1.3, -1.1])
    oversteer_goal = _udr_goal("apex", target_delta=0.0)
    oversteer_microsector = _udr_microsector(
        oversteer_goal,
        udr=0.15,
        sample_count=len(oversteer_results),
    )

    oversteer_recs = engine.generate(oversteer_results, [oversteer_microsector])

    assert oversteer_recs, "expected UDR rear softening recommendation"
    oversteer_messages = [rec.message.lower() for rec in oversteer_recs]
    assert any("rear" in message for message in oversteer_messages)

def test_phase_delta_rule_prioritises_brake_bias_for_longitudinal_axis() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="entry",
        operator_label="Braking operator",
        category="entry",
        phase_label="entry",
        priority=10,
        reference_key="braking",
    )
    goal = build_goal(
        "entry1",
        0.1,
        archetype="braking",
        description="",
        nu_f_target=0.25,
        nu_exc_target=0.2,
        rho_target=1.0,
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
    microsector = build_microsector(
        index=1,
        start_time=0.0,
        end_time=0.3,
        curvature=1.1,
        brake_event=True,
        support_event=False,
        delta_nfr_signature=0.4,
        phases=("entry1",),
        goals=(goal,),
        phase_boundaries={"entry1": (0, 3)},
        phase_samples={"entry1": (0, 1, 2)},
        active_phase="entry1",
        dominant_nodes={"entry1": ("brakes",)},
        phase_weights={"entry1": {"__default__": 1.0}},
        phase_lag={"entry1": 0.25},
        phase_alignment={"entry1": 0.82},
        filtered_measures={},
        window_occupancy={"entry1": {}},
        operator_events={},
        include_cphi=False,
    )
    results = [
        build_axis_bundle(delta_nfr=0.6, long_component=0.5, lat_component=0.1),
        build_axis_bundle(delta_nfr=0.58, long_component=0.46, lat_component=0.12),
        build_axis_bundle(delta_nfr=0.62, long_component=0.52, lat_component=0.1),
    ]
    thresholds = ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    messages = [rec.message.lower() for rec in recommendations]
    assert any("brake bias" in message for message in messages)
    targeted = [rec for rec in recommendations if rec.parameter == "brake_bias_pct"]
    assert targeted
    assert all(rec.priority <= rule.priority - 1 for rec in targeted)


def test_phase_delta_rule_offsets_brake_bias_with_downhill_gradient() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="entry",
        operator_label="Braking operator",
        category="entry",
        phase_label="entry",
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
        operator_label="Braking operator",
        category="entry",
        phase_label="entry",
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
    assert "ride height" in height_targets[0].message.lower()
    assert "increase bump damping" in bump_targets[0].message.lower()


def test_bottoming_priority_rule_prefers_springs_on_apex_energy() -> None:
    goal = _udr_goal("apex", 0.3)
    apex_sector = _udr_microsector(
        goal,
        udr=0.28,
        sample_count=6,
        filtered_measures={
            "bottoming_ratio_front": 0.64,
            "bumpstop_front_density": 0.34,
            "bumpstop_front_energy": 1.05,
        },
        index=4,
    )
    apex_sector = replace(apex_sector, context_factors={"surface": 0.98})
    rule = BottomingPriorityRule(ratio_threshold=0.5)
    recommendations = list(rule.evaluate([], [apex_sector]))
    assert any(rec.parameter == "front_spring_stiffness" for rec in recommendations)


def test_brake_headroom_rule_increases_force_when_surplus() -> None:
    rule = BrakeHeadroomRule(priority=14, margin=0.05, increase_step=0.03, decrease_step=0.02)
    thresholds = ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5)
    objectives = RuleProfileObjectives(target_brake_headroom=0.3)
    context = RuleContext(
        car_model="XFG",
        track_name="BL1",
        thresholds=thresholds,
        objectives=objectives,
    )
    microsector = _brake_headroom_microsector(
        5,
        0.45,
        abs_activation=0.2,
        partial=0.1,
        sustained=0.1,
        peak=6.5,
    )

    recommendations = list(rule.evaluate([], [microsector], context))

    assert recommendations
    recommendation = recommendations[0]
    assert recommendation.parameter == "brake_max_per_wheel"
    assert recommendation.delta == pytest.approx(0.03)
    assert "raise per-wheel maximum force" in recommendation.message.lower()
    assert "brake margin" in recommendation.rationale.lower()


def test_brake_headroom_rule_reduces_force_on_sustained_locking() -> None:
    rule = BrakeHeadroomRule(priority=16, decrease_step=0.04, sustained_lock_threshold=0.4)
    thresholds = ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5)
    objectives = RuleProfileObjectives(target_brake_headroom=0.35)
    context = RuleContext(
        car_model="XFG",
        track_name="BL1",
        thresholds=thresholds,
        objectives=objectives,
    )
    microsector = _brake_headroom_microsector(
        7,
        0.12,
        abs_activation=0.85,
        partial=0.7,
        sustained=0.65,
        peak=9.2,
    )

    recommendations = list(rule.evaluate([], [microsector], context))

    assert recommendations
    recommendation = recommendations[0]
    assert recommendation.delta == pytest.approx(-0.04)
    assert "sustained locking" in recommendation.message.lower()
    assert "sustained locking" in recommendation.rationale.lower()


def test_footprint_efficiency_rule_relaxes_delta_when_usage_high() -> None:
    goal = _udr_goal("apex", target_delta=0.2)
    results = build_udr_bundle_series([0.2, 0.21, 0.19])
    microsector = _udr_microsector(
        goal,
        udr=0.3,
        sample_count=len(results),
        filtered_measures={
            "mu_usage_front_ratio": 0.92,
            "mu_usage_rear_ratio": 0.82,
        },
    )
    context = RuleContext(
        car_model="XFG",
        track_name="BL1",
        thresholds=ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5),
    )
    rule = FootprintEfficiencyRule(priority=18, threshold=0.9)

    recommendations = list(rule.evaluate(results, [microsector], context))

    assert recommendations
    messages = [rec.message.lower() for rec in recommendations]
    assert any("footprint" in message for message in messages)
    assert any("front" in message for message in messages)
    assert not any("rear" in message for message in messages)
    rationales = " ".join(rec.rationale.lower() for rec in recommendations)
    assert "0.92" in rationales


def test_phase_delta_rule_brake_bias_uses_operator_events() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="entry",
        operator_label="Braking operator",
        category="entry",
        phase_label="entry",
        priority=12,
        reference_key="braking",
    )
    goal = build_goal(
        "entry1",
        0.3,
        archetype="braking",
        description="",
        target_sense_index=0.85,
        nu_f_target=0.28,
        nu_exc_target=0.22,
        rho_target=1.0,
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
                "name": canonical_operator_label("OZ"),
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
    microsector = build_microsector(
        index=4,
        start_time=0.0,
        end_time=0.4,
        curvature=1.3,
        brake_event=True,
        support_event=True,
        delta_nfr_signature=0.42,
        phases=("entry1",),
        goals=(goal,),
        phase_boundaries={"entry1": (0, 3)},
        phase_samples={"entry1": (0, 1, 2)},
        active_phase="entry1",
        dominant_nodes={"entry1": ("brakes", "tyres")},
        phase_weights={"entry1": {"__default__": 1.0}},
        phase_lag={"entry1": 0.1},
        phase_alignment={"entry1": 0.8},
        filtered_measures={},
        window_occupancy={"entry1": {}},
        operator_events=operator_events,
        include_cphi=False,
    )
    results = [
        build_axis_bundle(delta_nfr=-0.32, long_component=-0.28, lat_component=-0.07),
        build_axis_bundle(delta_nfr=-0.34, long_component=-0.29, lat_component=-0.06),
        build_axis_bundle(delta_nfr=-0.31, long_component=-0.3, lat_component=-0.05),
    ]
    thresholds = ThresholdProfile(0.12, 0.12, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    brake_recs = [rec for rec in recommendations if rec.parameter == "brake_bias_pct"]
    assert brake_recs, "expected brake bias recommendation"
    assert all(rec.delta is not None and rec.delta > 0 for rec in brake_recs)
    assert all(rec.priority <= rule.priority - 2 for rec in brake_recs)
    rationale_blob = " ".join(rec.rationale for rec in brake_recs)
    assert f"{canonical_operator_label('OZ')}×1" in rationale_blob
    assert "low_grip" in rationale_blob
    assert "ΔNFR" in rationale_blob


def test_phase_delta_rule_prioritises_sway_bar_for_lateral_axis() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="apex",
        operator_label="Apex operator",
        category="apex",
        phase_label="apex",
        priority=20,
        reference_key="antiroll",
    )
    goal = build_goal(
        "apex3a",
        0.05,
        archetype="hairpin",
        description="",
        target_sense_index=0.88,
        nu_f_target=0.3,
        nu_exc_target=0.25,
        rho_target=0.9,
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
    microsector = build_microsector(
        index=2,
        start_time=0.0,
        end_time=0.3,
        curvature=1.6,
        brake_event=False,
        support_event=True,
        delta_nfr_signature=0.3,
        phases=("apex3a",),
        goals=(goal,),
        phase_boundaries={"apex3a": (0, 4)},
        phase_samples={"apex3a": (0, 1, 2, 3)},
        active_phase="apex3a",
        dominant_nodes={"apex3a": ("suspension", "chassis")},
        phase_weights={"apex3a": {"__default__": 1.0}},
        phase_lag={"apex3a": -0.22},
        phase_alignment={"apex3a": 0.74},
        filtered_measures={},
        window_occupancy={"apex3a": {}},
        operator_events={},
        include_cphi=False,
    )
    results = [
        build_axis_bundle(delta_nfr=1.0, long_component=0.12, lat_component=0.88),
        build_axis_bundle(delta_nfr=0.95, long_component=0.1, lat_component=0.85),
        build_axis_bundle(delta_nfr=1.05, long_component=0.14, lat_component=0.92),
        build_axis_bundle(delta_nfr=1.0, long_component=0.11, lat_component=0.87),
    ]
    thresholds = ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    messages = [rec.message.lower() for rec in recommendations]
    assert any("anti-roll bar" in message for message in messages)
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
        operator_label="Braking operator",
        category="entry",
        phase_label="entry",
        priority=14,
        reference_key="braking",
    )
    goal = build_goal(
        "entry",
        0.2,
        archetype="tight",
        description="",
        nu_f_target=0.28,
        nu_exc_target=0.22,
        rho_target=1.0,
        measured_phase_lag=0.18,
        measured_phase_alignment=0.62,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.4, 0.4),
        dominant_nodes=("suspension", "tyres"),
        target_delta_nfr_long=0.15,
        target_delta_nfr_lat=0.05,
    )
    microsector = build_microsector(
        index=9,
        start_time=0.0,
        end_time=0.3,
        curvature=1.3,
        brake_event=True,
        support_event=False,
        delta_nfr_signature=0.25,
        phases=("entry",),
        goals=(goal,),
        phase_boundaries={"entry": (0, 3)},
        phase_samples={"entry": (0, 1, 2)},
        active_phase="entry",
        dominant_nodes={"entry": ("suspension", "tyres")},
        phase_weights={"entry": {"__default__": 1.0}},
        phase_lag={"entry": 0.18},
        phase_alignment={"entry": 0.62},
        filtered_measures={"coherence_index": 0.32},
        window_occupancy={"entry_a": {}},
        operator_events={},
        include_cphi=False,
    )
    results = []
    for value in (0.46, 0.48, 0.44):
        bundle = build_axis_bundle(delta_nfr=value, long_component=0.36, lat_component=0.1)
        results.append(replace(bundle, coherence_index=0.32))
    thresholds = ThresholdProfile(0.2, 0.5, 0.5, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    geometry_params = {"front_camber_deg", "front_toe_deg", "caster_deg"}
    geometry_recs = [rec for rec in recommendations if rec.parameter in geometry_params]
    assert geometry_recs, "expected geometry recommendations triggered by coherence gap"
    assert any("camber" in rec.message.lower() or "toe" in rec.message.lower() for rec in geometry_recs)
    assert all(rec.priority <= rule.priority - 1 for rec in geometry_recs)


def test_phase_delta_rule_targets_front_toe_for_entry_synchrony_gap() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="entry",
        operator_label="Braking operator",
        category="entry",
        phase_label="entry",
        priority=12,
        reference_key="braking",
    )
    goal = build_goal(
        "entry1",
        0.3,
        archetype="braking",
        description="",
        target_sense_index=0.9,
        nu_f_target=0.28,
        nu_exc_target=0.22,
        rho_target=1.0,
        measured_phase_lag=0.25,
        measured_phase_alignment=0.6,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.3, 0.3),
        dominant_nodes=("tyres", "suspension"),
        target_delta_nfr_long=0.18,
        target_delta_nfr_lat=0.12,
    )
    samples = tuple(range(3))
    microsector = build_microsector(
        index=6,
        start_time=0.0,
        end_time=0.3,
        curvature=1.2,
        brake_event=True,
        support_event=False,
        delta_nfr_signature=0.3,
        phases=(goal.phase,),
        goals=(goal,),
        phase_boundaries={goal.phase: (0, 3)},
        phase_samples={goal.phase: samples},
        active_phase=goal.phase,
        dominant_nodes={goal.phase: goal.dominant_nodes},
        phase_weights={goal.phase: {"__default__": 1.0}},
        phase_lag={goal.phase: goal.measured_phase_lag},
        phase_alignment={goal.phase: goal.measured_phase_alignment},
        phase_synchrony={goal.phase: goal.measured_phase_synchrony},
        filtered_measures={},
        window_occupancy={goal.phase: {}},
        operator_events={},
        include_cphi=False,
    )
    results = [
        build_axis_bundle(delta_nfr=0.3, long_component=0.18, lat_component=0.12),
        build_axis_bundle(delta_nfr=0.3, long_component=0.18, lat_component=0.12),
        build_axis_bundle(delta_nfr=0.3, long_component=0.18, lat_component=0.12),
    ]
    thresholds = ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    front_toe = [rec for rec in recommendations if rec.parameter == "front_toe_deg"]
    assert front_toe, "expected front toe recommendations for synchrony gap"
    assert all(
        rec.delta is not None and rec.delta > 0 for rec in front_toe
    ), "synchrony gap should trigger toe-out"


def test_phase_delta_rule_prioritises_front_spring_with_lateral_bias() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="apex",
        operator_label="Apex operator",
        category="apex",
        phase_label="apex",
        priority=22,
        reference_key="antiroll",
    )
    goal = build_goal(
        "apex",
        0.25,
        archetype="medium",
        description="",
        nu_f_target=0.28,
        nu_exc_target=0.23,
        rho_target=0.9,
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
    microsector = build_microsector(
        index=12,
        start_time=0.0,
        end_time=0.4,
        curvature=1.6,
        brake_event=False,
        support_event=True,
        delta_nfr_signature=0.32,
        phases=(goal.phase,),
        goals=(goal,),
        phase_boundaries={goal.phase: (0, len(samples))},
        phase_samples={goal.phase: samples},
        active_phase=goal.phase,
        dominant_nodes={goal.phase: goal.dominant_nodes},
        phase_weights={goal.phase: {"__default__": 1.0}},
        phase_lag={goal.phase: goal.measured_phase_lag},
        phase_alignment={goal.phase: goal.measured_phase_alignment},
        filtered_measures={},
        window_occupancy={goal.phase: {}},
        operator_events={},
        include_cphi=False,
    )
    raw_results: list[EPIBundle] = []
    for lat_component in (0.3, 0.32, 0.31, 0.29):
        bundle = build_axis_bundle(
            delta_nfr=0.38,
            long_component=0.07,
            lat_component=lat_component,
        )
        suspension = replace(bundle.suspension, nu_f=0.36)
        raw_results.append(replace(bundle, suspension=suspension))
    thresholds = ThresholdProfile(0.1, 0.05, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="GT3", track_name="VAL", thresholds=thresholds)
    recommendations = list(rule.evaluate(raw_results, [microsector], context))
    spring_recs = [rec for rec in recommendations if rec.parameter == "front_spring_stiffness"]
    assert spring_recs, "expected front spring recommendation under lateral dominance"
    assert all(rec.delta is not None and rec.delta < 0 for rec in spring_recs)
    assert all("νf_susp" in rec.message for rec in spring_recs)
    assert all("∇NFR⊥" in rec.rationale for rec in spring_recs)


def test_phase_delta_rule_scales_rear_spring_with_lateral_bias_and_low_frequency() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="exit",
        operator_label="Traction operator",
        category="exit",
        phase_label="exit",
        priority=24,
        reference_key="differential",
    )
    goal = build_goal(
        "exit",
        0.3,
        archetype="medium",
        description="",
        target_sense_index=0.88,
        nu_f_target=0.26,
        nu_exc_target=0.2,
        rho_target=0.85,
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
    microsector = build_microsector(
        index=14,
        start_time=0.0,
        end_time=0.3,
        curvature=1.2,
        brake_event=False,
        support_event=False,
        delta_nfr_signature=0.34,
        phases=(goal.phase,),
        goals=(goal,),
        phase_boundaries={goal.phase: (0, len(sample_indices))},
        phase_samples={goal.phase: sample_indices},
        active_phase=goal.phase,
        dominant_nodes={goal.phase: goal.dominant_nodes},
        phase_weights={goal.phase: {"__default__": 1.0}},
        phase_lag={goal.phase: goal.measured_phase_lag},
        phase_alignment={goal.phase: goal.measured_phase_alignment},
        filtered_measures={},
        window_occupancy={goal.phase: {}},
        operator_events={},
        include_cphi=False,
    )
    bundles: list[EPIBundle] = []
    for lat_component in (0.22, 0.24, 0.23):
        bundle = build_axis_bundle(
            delta_nfr=0.36,
            long_component=0.05,
            lat_component=lat_component,
        )
        suspension = replace(bundle.suspension, nu_f=0.18)
        bundles.append(replace(bundle, suspension=suspension))
    thresholds = ThresholdProfile(0.1, 0.05, 0.08, 0.2, 0.5)
    context = RuleContext(car_model="GT3", track_name="VAL", thresholds=thresholds)
    recommendations = list(rule.evaluate(bundles, [microsector], context))
    rear_spring_recs = [rec for rec in recommendations if rec.parameter == "rear_spring_stiffness"]
    assert rear_spring_recs, "expected rear spring recommendation for exit phase"
    assert all(rec.delta is not None and rec.delta > 0 for rec in rear_spring_recs)
    assert all("νf_susp" in rec.message for rec in rear_spring_recs)
    assert all("∇NFR⊥" in rec.rationale for rec in rear_spring_recs)


def test_phase_node_rule_prioritises_geometry_with_alignment_gap() -> None:
    rule = PhaseNodeOperatorRule(
        phase="apex",
        operator_label="Apex operator",
        category="apex",
        priority=24,
        reference_key="antiroll",
    )
    goal = build_goal(
        "apex",
        0.15,
        archetype="hairpin",
        description="",
        target_sense_index=0.88,
        nu_f_target=0.32,
        nu_exc_target=0.27,
        rho_target=0.9,
        measured_phase_lag=0.21,
        measured_phase_alignment=0.66,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.4, 0.4),
        dominant_nodes=("suspension", "tyres"),
    )
    microsector = build_microsector(
        index=6,
        start_time=0.0,
        end_time=0.4,
        curvature=1.5,
        brake_event=False,
        support_event=True,
        delta_nfr_signature=0.3,
        phases=("apex",),
        goals=(goal,),
        phase_boundaries={"apex": (0, 4)},
        phase_samples={"apex": (0, 1, 2, 3)},
        active_phase="apex",
        dominant_nodes={"apex": ("suspension", "tyres")},
        phase_weights={"apex": {"__default__": 1.0}},
        phase_lag={"apex": 0.21},
        phase_alignment={"apex": 0.66},
        filtered_measures={"coherence_index": 0.35},
        window_occupancy={"apex_r": {}},
        operator_events={},
        include_cphi=False,
    )
    results = []
    for value in (0.35, 0.33, 0.36, 0.34):
        bundle = build_axis_bundle(
            delta_nfr=value,
            long_component=0.14,
            lat_component=0.21,
        )
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


def test_phase_delta_rule_targets_rear_toe_for_exit_synchrony_gap() -> None:
    rule = PhaseDeltaDeviationRule(
        phase="exit",
        operator_label="Traction operator",
        category="exit",
        phase_label="exit",
        priority=16,
        reference_key="differential",
    )
    goal = build_goal(
        "exit",
        0.28,
        archetype="medium",
        description="",
        target_sense_index=0.88,
        nu_f_target=0.3,
        nu_exc_target=0.24,
        rho_target=0.95,
        target_phase_lag=0.25,
        target_phase_alignment=0.8,
        measured_phase_lag=-0.05,
        measured_phase_alignment=1.0,
        slip_lat_window=(-0.3, 0.3),
        slip_long_window=(-0.3, 0.3),
        yaw_rate_window=(-0.4, 0.4),
        dominant_nodes=("tyres", "transmission"),
        target_delta_nfr_long=0.14,
        target_delta_nfr_lat=0.14,
    )
    samples = tuple(range(4))
    microsector = build_microsector(
        index=11,
        start_time=0.0,
        end_time=0.4,
        curvature=1.4,
        brake_event=False,
        support_event=True,
        delta_nfr_signature=0.28,
        phases=(goal.phase,),
        goals=(goal,),
        phase_boundaries={goal.phase: (0, 4)},
        phase_samples={goal.phase: samples},
        active_phase=goal.phase,
        dominant_nodes={goal.phase: goal.dominant_nodes},
        phase_weights={goal.phase: {"__default__": 1.0}},
        phase_lag={goal.phase: goal.measured_phase_lag},
        phase_alignment={goal.phase: goal.measured_phase_alignment},
        phase_synchrony={goal.phase: goal.measured_phase_synchrony},
        filtered_measures={},
        window_occupancy={goal.phase: {}},
        operator_events={},
        include_cphi=False,
    )
    results = [
        build_axis_bundle(delta_nfr=0.28, long_component=0.14, lat_component=0.14),
        build_axis_bundle(delta_nfr=0.28, long_component=0.14, lat_component=0.14),
        build_axis_bundle(delta_nfr=0.28, long_component=0.14, lat_component=0.14),
        build_axis_bundle(delta_nfr=0.28, long_component=0.14, lat_component=0.14),
    ]
    thresholds = ThresholdProfile(0.1, 0.1, 0.1, 0.2, 0.5)
    context = RuleContext(car_model="XFG", track_name="BL1", thresholds=thresholds)
    recommendations = list(rule.evaluate(results, [microsector], context))
    rear_toe = [rec for rec in recommendations if rec.parameter == "rear_toe_deg"]
    assert rear_toe, "expected rear toe recommendations for synchrony spike"
    assert all(
        rec.delta is not None and rec.delta > 0 for rec in rear_toe
    ), "synchrony spike should trigger toe-in"
