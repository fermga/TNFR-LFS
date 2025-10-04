from pathlib import Path

import pytest

from tnfr_lfs.io.profiles import ProfileManager


def test_profile_manager_persists_jacobian_history(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.toml"
    manager = ProfileManager(profiles_path)
    car_model = "generic_gt"
    track = "generic"
    manager.resolve(car_model, track)
    manager.register_plan(
        car_model,
        track,
        {"entry": 1.0, "apex": 0.5},
        baseline_metrics=(0.5, 5.0),
        jacobian={
            "sense_index": {"rear_wing_angle": 1.4},
            "delta_nfr_integral": {"rear_wing_angle": -0.6},
        },
        phase_jacobian={"entry": {"delta_nfr_integral": {"rear_wing_angle": -0.4}}},
    )
    manager.register_result(car_model, track, sense_index=0.7, delta_nfr=3.0)

    contents = profiles_path.read_text(encoding="utf8")
    assert "[profiles.generic_gt.generic.jacobian.overall.sense_index]" in contents
    assert "rear_wing_angle = 1.4" in contents

    reloaded = ProfileManager(profiles_path)
    overall, phases = reloaded.gradient_history(car_model, track)
    assert overall["sense_index"]["rear_wing_angle"] == pytest.approx(1.4)
    assert overall["delta_nfr_integral"]["rear_wing_angle"] == pytest.approx(-0.6)
    assert phases["entry"]["delta_nfr_integral"]["rear_wing_angle"] == pytest.approx(-0.4)


def test_profile_manager_updates_tyre_offsets(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.toml"
    manager = ProfileManager(profiles_path)
    car_model = "generic_gt"
    track = "generic"
    manager.resolve(car_model, track)
    manager.update_tyre_offsets(car_model, track, {"pressure_front": -0.04, "camber_rear": 0.08})

    reloaded = ProfileManager(profiles_path)
    snapshot = reloaded.resolve(car_model, track)
    assert snapshot.tyre_offsets["pressure_front"] == pytest.approx(-0.04)
    assert snapshot.tyre_offsets["camber_rear"] == pytest.approx(0.08)
    assert "hairpin" in snapshot.archetype_targets
    assert "entry" in snapshot.archetype_targets["hairpin"]
