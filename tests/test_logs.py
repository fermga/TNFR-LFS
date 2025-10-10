from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tnfr_lfs.ingestion.offline import DeterministicReplayer, iter_run, write_run

from tests.helpers import build_telemetry_record


def _sample_payload(seed: int) -> dict[str, float]:
    base = 1000.0 + seed
    return {
        "timestamp": float(seed),
        "vertical_load": base + 10.0,
        "slip_ratio": 0.01 * seed,
        "lateral_accel": 0.1 * seed,
        "longitudinal_accel": 0.2 * seed,
        "yaw": 0.05 * seed,
        "pitch": 0.03 * seed,
        "roll": 0.02 * seed,
        "brake_pressure": 0.4 * seed,
        "locking": 0.01 * seed,
        "nfr": 0.5 * seed,
        "si": 0.25 * seed,
        "speed": 40.0 + seed,
        "yaw_rate": 0.06 * seed,
        "slip_angle": 0.04 * seed,
        "steer": 0.02 * seed,
        "throttle": 0.8,
        "gear": seed % 6,
        "vertical_load_front": base + 5.0,
        "vertical_load_rear": base + 7.5,
        "mu_eff_front": 1.0 + 0.01 * seed,
        "mu_eff_rear": 0.95 + 0.01 * seed,
        "mu_eff_front_lateral": 1.0 + 0.01 * seed,
        "mu_eff_front_longitudinal": 0.92 + 0.01 * seed,
        "mu_eff_rear_lateral": 0.95 + 0.01 * seed,
        "mu_eff_rear_longitudinal": 0.9 + 0.01 * seed,
        "suspension_travel_front": 0.1 * seed,
        "suspension_travel_rear": 0.12 * seed,
        "suspension_velocity_front": 0.14 * seed,
        "suspension_velocity_rear": 0.16 * seed,
    }


def test_write_and_iter_run_roundtrip_with_compression(tmp_path: Path) -> None:
    baseline = build_telemetry_record(**_sample_payload(1))
    sample = replace(build_telemetry_record(**_sample_payload(2)), reference=baseline)
    destination = tmp_path / "telemetry.jsonl.gz"

    write_run([baseline, sample], destination)
    restored = list(iter_run(destination))

    assert restored == [baseline, sample]


def test_deterministic_replayer_is_repeatable() -> None:
    records = [build_telemetry_record(**_sample_payload(index)) for index in range(3)]
    replayer = DeterministicReplayer(records)

    first = list(replayer)
    second = list(replayer)
    explicit_iter = list(replayer.iter())

    assert first == records
    assert second == records
    assert explicit_iter == records
