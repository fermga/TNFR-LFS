"""Canonical line-to-line lap slicer tests.

These tests use synthetic :class:`TelemetrySample` buffers to verify
that ``find_line_crossings`` and ``slice_into_laps`` behave as
documented, then validate the helpers against a real BL1 FBM stint
capture if it is available on disk.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lfs_telemetry.telemetry.live import TelemetrySample
from lfs_telemetry.telemetry.protocol.packets import OutSimPack2
from lfs_telemetry.telemetry.lap_slicer import (
    find_line_crossings,
    reslice_csv,
    slice_into_laps,
)


def _mk(time_ms: int, d: float | None) -> TelemetrySample:
    pkt2 = None
    if d is not None:
        pkt2 = OutSimPack2(opts=0, current_lap_dist_m=float(d))
    return TelemetrySample(time_ms=time_ms, outsim2=pkt2)


def _build_buffer(track_len_m: float, lap_dt_ms: int, samples_per_lap: int,
                  laps: int, *, leading_offset_m: float = 1500.0,
                  trailing_offset_m: float = 0.0) -> list[TelemetrySample]:
    """Build a synthetic buffer of ``laps`` full laps preceded by a
    partial pre-lap and (optionally) followed by a partial post-lap.
    Sample distances march linearly from 0 to ``track_len_m`` then wrap.
    """
    samples: list[TelemetrySample] = []
    t = 0
    # partial pre-lap (out-lap), starts mid-lap
    pre_n = max(1, int(samples_per_lap * leading_offset_m / track_len_m))
    for i in range(pre_n):
        d = leading_offset_m + i * (track_len_m - leading_offset_m) / pre_n
        samples.append(_mk(t, d))
        t += lap_dt_ms // samples_per_lap
    # full laps
    for _ in range(laps):
        for i in range(samples_per_lap):
            d = i * track_len_m / samples_per_lap
            samples.append(_mk(t, d))
            t += lap_dt_ms // samples_per_lap
    # partial post-lap
    if trailing_offset_m > 0:
        post_n = max(1, int(samples_per_lap * trailing_offset_m / track_len_m))
        for i in range(post_n):
            d = i * trailing_offset_m / post_n
            samples.append(_mk(t, d))
            t += lap_dt_ms // samples_per_lap
    return samples


def test_find_line_crossings_basic():
    s = _build_buffer(3000.0, 75000, 200, laps=3,
                      leading_offset_m=1800.0, trailing_offset_m=600.0)
    xs = find_line_crossings(s)
    # 1 from pre→lap1, 2 between full laps, 1 from lap3→post = 4 total
    assert len(xs) == 4
    # each crossing index must have a strictly smaller d than the
    # previous sample
    for i in xs:
        d_prev = s[i - 1].outsim2.current_lap_dist_m
        d_here = s[i].outsim2.current_lap_dist_m
        assert d_here < d_prev


def test_find_line_crossings_ignores_small_jitter():
    # tiny noise (< min_drop_m) must NOT register as a crossing
    s = [_mk(i * 10, 100.0 + (i % 2) * -5.0) for i in range(20)]
    assert find_line_crossings(s) == []


def test_find_line_crossings_skips_missing():
    # missing OutSimPack2 samples must be transparently skipped
    s: list[TelemetrySample] = []
    s.append(_mk(0, 1000.0))
    s.append(_mk(10, None))     # gap
    s.append(_mk(20, 1500.0))
    s.append(_mk(30, None))
    s.append(_mk(40, 50.0))     # crossing
    s.append(_mk(50, 200.0))
    xs = find_line_crossings(s)
    assert xs == [4]


def test_slice_into_laps_drops_partial_tails():
    track = 3285.6   # BL1 length
    s = _build_buffer(track, 75000, 200, laps=3,
                      leading_offset_m=2350.0, trailing_offset_m=500.0)
    laps = slice_into_laps(s)
    assert len(laps) == 3
    for n, lap in enumerate(laps, start=1):
        assert lap.lap_index == n
        # canonical: starts at d≈0 and ends just before next wrap
        assert lap.samples[0].outsim2.current_lap_dist_m < 50.0
        assert lap.samples[-1].outsim2.current_lap_dist_m > track * 0.9
        # distance_m matches synthetic track length within sample step
        assert lap.distance_m >= track * 0.99


def test_slice_into_laps_zero_laps_when_no_crossings():
    s = [_mk(i * 10, float(i)) for i in range(50)]   # monotonic, no wrap
    assert slice_into_laps(s) == []


def test_slice_into_laps_zero_laps_when_only_one_crossing():
    # one crossing = pre-lap + partial lap1 → no full lap recoverable
    s = _build_buffer(3000.0, 75000, 100, laps=1,
                      leading_offset_m=1500.0)
    # remove the trailing partial → still has 1 crossing only
    crossings = find_line_crossings(s)
    if len(crossings) == 1:
        assert slice_into_laps(s) == []


# ---------------------------------------------------------------------------
# Real-world validation against the BL1 FBM stint, if present.
# ---------------------------------------------------------------------------

REAL_AGGREGATE = Path("captures") / "stint.csv"


@pytest.mark.skipif(not REAL_AGGREGATE.exists(),
                    reason="real BL1 FBM stint not on disk")
def test_real_stint_yields_three_canonical_laps(tmp_path: Path):
    written = reslice_csv(REAL_AGGREGATE, out_dir=tmp_path,
                          stem="stint_canonical")
    # The BL1 FBM stint has 4 line crossings → 3 full laps.
    assert len(written) == 3
    for path, lap, n in written:
        assert n > 1000
        d0 = lap.samples[0].outsim2.current_lap_dist_m
        d1 = lap.samples[-1].outsim2.current_lap_dist_m
        # canonical bounds: starts within 50 m of the line, ends near it
        assert d0 < 50.0, f"{path.name} starts at d={d0:.1f} (not at line)"
        # BL1 ≈ 3285 m → last sample should be in the last 5 % of the lap
        assert d1 > 3000.0, f"{path.name} ends at d={d1:.1f} (truncated?)"
