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
