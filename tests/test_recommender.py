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
from collections.abc import Mapping

from tnfr_lfs.core.segmentation import Goal, Microsector
from tnfr_lfs.io.profiles import ProfileManager
from tnfr_lfs.recommender.rules import Recommendation, RecommendationEngine


BASE_NU_F = {
    "tyres": 0.18,
    "suspension": 0.14,
    "chassis": 0.12,
    "brakes": 0.16,
    "transmission": 0.11,
    "track": 0.08,
    "driver": 0.05,
}


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
                archetype="apoyo",
                description="",
                target_delta_nfr=entry_target,
                target_sense_index=0.9,
                nu_f_target=0.25,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=entry_nodes,
            ),
            Goal(
                phase="apex",
                archetype="apoyo",
                description="",
                target_delta_nfr=apex_target,
                target_sense_index=0.9,
                nu_f_target=0.25,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=apex_nodes,
            ),
            Goal(
                phase="exit",
                archetype="apoyo",
                description="",
                target_delta_nfr=exit_target,
                target_sense_index=0.9,
                nu_f_target=0.25,
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
        filtered_measures=filtered_measures,
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy=window_occupancy,
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
                archetype="apoyo",
                description="",
                target_delta_nfr=0.0,
                target_sense_index=0.9,
                nu_f_target=0.25,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=nodes,
            ),
            Goal(
                phase="apex",
                archetype="apoyo",
                description="",
                target_delta_nfr=0.2,
                target_sense_index=0.9,
                nu_f_target=0.25,
                slip_lat_window=window,
                slip_long_window=window,
                yaw_rate_window=yaw_window,
                dominant_nodes=nodes,
            ),
            Goal(
                phase="exit",
                archetype="apoyo",
                description="",
                target_delta_nfr=-0.1,
                target_sense_index=0.9,
                nu_f_target=0.25,
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
        filtered_measures=filtered_measures,
        recursivity_trace=(),
        last_mutation=None,
        window_occupancy=window_occupancy,
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
