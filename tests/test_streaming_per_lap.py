"""Streaming per-lap captures: ``skip_lap_indices`` contract.

During a live capture the CLI now writes each completed flying lap to
disk atomically as it is detected, so the user can open laps in Studio
without stopping the recorder. At session end, ``_flush_capture`` calls
:func:`write_per_lap_files` again as a safety-net flush, but it must
*not* rewrite the laps already streamed: it passes the indices it has
already persisted in ``skip_lap_indices``.

These tests pin that contract:

* ``skip_lap_indices`` skips writing the named lap files but still
  reports them in the result with ``rows=0``.
* Laps not in the skip set still produce real CSVs.
* The skip set never affects which lap indices the slicer assigns.
"""
from __future__ import annotations

from pathlib import Path

from lfs_telemetry.telemetry.lap_slicer import write_per_lap_files
from lfs_telemetry.telemetry.live import TelemetrySample
from lfs_telemetry.telemetry.protocol.packets import (
    OutGaugePacket,
    OutSimPack2,
    OutSimPacket,
)


def _mk(time_ms: int, d: float) -> TelemetrySample:
    """Build a *complete* synthetic sample (outsim+outgauge+outsim2).

    ``write_csv_replay`` skips samples where ``is_complete`` is False
    (i.e. missing OutSim or OutGauge), so the lap-distance-only samples
    used by other slicer tests would write zero rows here.
    """
    return TelemetrySample(
        time_ms=time_ms,
        outsim=OutSimPacket(
            time_ms=time_ms,
            ang_vel=(0.0, 0.0, 0.0),
            heading=0.0, pitch=0.0, roll=0.0,
            accel=(0.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
        outgauge=OutGaugePacket(
            time_ms=time_ms, car="FBM", flags=0, gear=2, player_id=1,
            speed_ms=30.0, rpm=7000.0, turbo_bar=0.0, eng_temp_c=80.0,
            fuel=0.5, oil_pressure_bar=4.0, oil_temp_c=90.0,
            dash_lights=0, show_lights=0,
            throttle=0.5, brake=0.0, clutch=0.0,
            display1="", display2="",
        ),
        outsim2=OutSimPack2(opts=0, current_lap_dist_m=float(d)),
    )


def _three_lap_buffer(track_len_m: float = 2000.0,
                      samples_per_lap: int = 100) -> list[TelemetrySample]:
    """Three flying laps preceded by a partial pre-lap. Distances march
    linearly from 0 to ``track_len_m`` then wrap, exactly like real
    OutSim2 ``current_lap_dist_m``.
    """
    out: list[TelemetrySample] = []
    t = 0
    dt = 10
    # short pre-lap so canonical slicer marks lap0 as out-lap
    for i in range(20):
        out.append(_mk(t, 1500.0 + i * 20.0))
        t += dt
    for _ in range(3):
        for i in range(samples_per_lap):
            out.append(_mk(t, i * track_len_m / samples_per_lap))
            t += dt
    # short partial post-lap so lap3 has a closing crossing
    for i in range(10):
        out.append(_mk(t, i * 50.0))
        t += dt
    return out


def test_skip_lap_indices_skips_existing_files(tmp_path: Path) -> None:
    samples = _three_lap_buffer()
    # First "streaming" pass: pretend the live capture already wrote
    # laps 1 and 2 to disk by calling write_per_lap_files normally and
    # then deleting lap2 so we can prove the second pass leaves it
    # alone (skip wins) rather than overwriting it.
    first = write_per_lap_files(
        samples, out_dir=tmp_path, stem="cap", session_tag="t",
    )
    lap_paths = {lap.lap_index: path for path, lap, _ in first}
    assert {1, 2, 3}.issubset(lap_paths)

    # Tamper with lap2 so any rewrite would change its content.
    tampered = lap_paths[2]
    tampered.write_text("SENTINEL\n", encoding="utf-8")

    # Second pass: tell the slicer laps 1 & 2 are already on disk.
    second = write_per_lap_files(
        samples, out_dir=tmp_path, stem="cap", session_tag="t",
        skip_lap_indices=[1, 2],
    )

    by_index = {lap.lap_index: (path, rows) for path, lap, rows in second}
    # Skipped laps reported with rows=0 and untouched on disk.
    assert by_index[1][1] == 0
    assert by_index[2][1] == 0
    assert tampered.read_text(encoding="utf-8") == "SENTINEL\n"
    # Lap 3 was not in the skip set, so it must have been (re)written.
    assert by_index[3][1] > 0
    assert by_index[3][0].read_text(encoding="utf-8") != "SENTINEL\n"


def test_skip_lap_indices_does_not_shift_lap_numbering(
    tmp_path: Path,
) -> None:
    samples = _three_lap_buffer()
    result = write_per_lap_files(
        samples, out_dir=tmp_path, stem="cap",
        skip_lap_indices=[1],
    )
    indices = sorted(lap.lap_index for _, lap, _ in result)
    # Canonical slicer still numbers laps 1..3 regardless of skip set.
    assert indices == [1, 2, 3]


def test_empty_skip_set_writes_everything(tmp_path: Path) -> None:
    samples = _three_lap_buffer()
    result = write_per_lap_files(
        samples, out_dir=tmp_path, stem="cap",
    )
    rows_per_lap = {lap.lap_index: rows for _, lap, rows in result}
    assert all(r > 0 for r in rows_per_lap.values())
    # All files exist on disk.
    for path, _, rows in result:
        assert path.exists()
        assert rows > 0
