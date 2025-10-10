from __future__ import annotations

from pathlib import Path

from tnfr_lfs.ingestion.offline import ProfileManager


def preloaded_profile_manager(
    base_path: Path,
    *,
    car_model: str = "FZR",
    track: str = "generic",
) -> ProfileManager:
    """Return a ``ProfileManager`` that already stores gradient history data."""
    profiles_path = base_path / "profiles.toml"
    manager = ProfileManager(profiles_path)
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
    return ProfileManager(profiles_path)
