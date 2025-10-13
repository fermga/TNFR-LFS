from pathlib import Path

import pytest

from tnfr_lfs.telemetry.offline import ProfileManager


CAR_MODEL = "FZR"
TRACK = "generic"


@pytest.fixture
def profile_env(tmp_path: Path) -> tuple[ProfileManager, Path]:
    profiles_path = tmp_path / "profiles.toml"
    manager = ProfileManager(profiles_path)
    manager.resolve(CAR_MODEL, TRACK)
    return manager, profiles_path


def test_profile_manager_persists_jacobian_history(profile_env: tuple[ProfileManager, Path]) -> None:
    manager, profiles_path = profile_env
    manager.register_plan(
        CAR_MODEL,
        TRACK,
        {"entry": 1.0, "apex": 0.5},
        baseline_metrics=(0.5, 5.0),
        jacobian={
            "sense_index": {"rear_wing_angle": 1.4},
            "delta_nfr_integral": {"rear_wing_angle": -0.6},
        },
        phase_jacobian={"entry": {"delta_nfr_integral": {"rear_wing_angle": -0.4}}},
    )
    manager.register_result(CAR_MODEL, TRACK, sense_index=0.7, delta_nfr=3.0)

    contents = profiles_path.read_text(encoding="utf8")
    assert "[profiles.\"FZR\".generic.jacobian.overall.sense_index]" in contents
    assert "rear_wing_angle = 1.4" in contents

    reloaded = ProfileManager(profiles_path)
    overall, phases = reloaded.gradient_history(CAR_MODEL, TRACK)
    assert overall["sense_index"]["rear_wing_angle"] == pytest.approx(1.4)
    assert overall["delta_nfr_integral"]["rear_wing_angle"] == pytest.approx(-0.6)
    assert phases["entry"]["delta_nfr_integral"]["rear_wing_angle"] == pytest.approx(-0.4)


def test_profile_manager_updates_tyre_offsets(profile_env: tuple[ProfileManager, Path]) -> None:
    manager, profiles_path = profile_env
    manager.update_tyre_offsets(
        CAR_MODEL,
        TRACK,
        {"pressure_front": -0.04, "camber_rear": 0.08},
    )

    reloaded = ProfileManager(profiles_path)
    snapshot = reloaded.resolve(CAR_MODEL, TRACK)
    assert snapshot.tyre_offsets["pressure_front"] == pytest.approx(-0.04)
    assert snapshot.tyre_offsets["camber_rear"] == pytest.approx(0.08)
    assert "hairpin" in snapshot.archetype_targets
    assert "entry" in snapshot.archetype_targets["hairpin"]


def test_profile_manager_updates_aero_profiles(profile_env: tuple[ProfileManager, Path]) -> None:
    manager, profiles_path = profile_env

    manager.update_aero_profile(CAR_MODEL, TRACK, "race", high_speed_target=0.32)
    manager.update_aero_profile(
        CAR_MODEL,
        TRACK,
        "stint_save",
        low_speed_target=-0.12,
        high_speed_target=0.08,
    )

    snapshot = manager.resolve(CAR_MODEL, TRACK)
    assert snapshot.aero_profiles["race"].high_speed_target == pytest.approx(0.32)
    assert snapshot.aero_profiles["stint_save"].low_speed_target == pytest.approx(-0.12)

    manager.save()
    persisted = profiles_path.read_text(encoding="utf8")
    assert "[profiles.\"FZR\".generic.aero_profiles.race]" in persisted
    assert "high_speed_target = 0.32" in persisted

    reloaded = ProfileManager(profiles_path)
    re_snapshot = reloaded.resolve(CAR_MODEL, TRACK)
    assert re_snapshot.aero_profiles["stint_save"].high_speed_target == pytest.approx(0.08)


def test_profile_manager_merges_session_weights(profile_env: tuple[ProfileManager, Path]) -> None:
    manager, _ = profile_env
    session_one = {
        "weights": {"entry": {"__default__": 1.25, "brakes": 1.4}},
        "hints": {"rho_detune_threshold": 0.63, "surface": "mixed"},
    }
    snapshot = manager.resolve(CAR_MODEL, TRACK, session=session_one)
    assert snapshot.session_weights["entry"]["__default__"] == pytest.approx(1.25)
    assert snapshot.thresholds.phase_weights["entry"]["__default__"] == pytest.approx(1.25)
    assert snapshot.session_hints["rho_detune_threshold"] == pytest.approx(0.63)
    assert snapshot.thresholds.rho_detune_threshold == pytest.approx(0.63)

    session_two = {
        "weights": {"entry": {"__default__": 1.1}},
        "hints": {"rho_detune_threshold": 0.7},
    }
    snapshot_two = manager.resolve(CAR_MODEL, TRACK, session=session_two)
    assert snapshot_two.session_weights["entry"]["__default__"] == pytest.approx(1.1)
    assert snapshot_two.thresholds.phase_weights["entry"]["__default__"] == pytest.approx(1.1)
    assert snapshot_two.thresholds.rho_detune_threshold == pytest.approx(0.7)
