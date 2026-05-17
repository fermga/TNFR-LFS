"""Phase 9.A — multi_stint comparison tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from lfs_telemetry.telemetry.lap import LapTelemetry
from lfs_telemetry.tnfr_racing.multi_stint import (
    StintComparison, compare_stints,
)
from lfs_telemetry.tnfr_racing.serialize import (
    comparison_to_json, comparison_to_markdown,
)

from .test_advisor import _synthetic_baseline

ASSETS = Path(__file__).resolve().parents[2] / "assets"

FORBIDDEN_TERMS = (
    "EPI", "νf", "ΔNFR", "Φ_s", " Si", "operador",
    " AL ", " EN ", " IL ", " OZ ", " UM ", " RA ",
    " SHA ", " VAL ", " NUL ", "THOL", "ZHIR", "NAV", "REMESH",
    "U1", "U2", "U3", "U4", "U5", "U6", "tétrada",
)


@pytest.fixture(scope="module")
def bl1_laps_and_baseline():
    paths = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))
    laps = [LapTelemetry.from_csv(p) for p in paths]
    return laps, _synthetic_baseline(), laps[0].car


def test_compare_same_stint_is_zero(bl1_laps_and_baseline) -> None:
    """Comparing a stint against itself yields a zero realised change."""
    laps, baseline, car = bl1_laps_and_baseline
    cmp = compare_stints(
        before_laps=laps, after_laps=laps,
        baseline_before=baseline, baseline_after=baseline,
        car=car, track_code="BL1",
    )
    assert isinstance(cmp, StintComparison)
    assert cmp.coherence_change == pytest.approx(0.0, abs=1e-9)
    if cmp.lap_time_change_ms is not None:
        assert cmp.lap_time_change_ms == pytest.approx(0.0, abs=1e-6)
    # Same stint = same proposal again = no actions resolved
    assert cmp.proposed_actions_validated == 0


def test_compare_is_deterministic(bl1_laps_and_baseline) -> None:
    laps, baseline, car = bl1_laps_and_baseline
    a = compare_stints(
        before_laps=laps, after_laps=laps,
        baseline_before=baseline, baseline_after=baseline,
        car=car, track_code="BL1", seed=20260516,
    )
    b = compare_stints(
        before_laps=laps, after_laps=laps,
        baseline_before=baseline, baseline_after=baseline,
        car=car, track_code="BL1", seed=20260516,
    )
    assert a.coherence_before == b.coherence_before
    assert a.coherence_after == b.coherence_after
    assert a.coherence_change == b.coherence_change
    assert a.headline == b.headline
    assert comparison_to_json(a) == comparison_to_json(b)


def test_short_before_stint_yields_incomplete(bl1_laps_and_baseline) -> None:
    laps, baseline, car = bl1_laps_and_baseline
    cmp = compare_stints(
        before_laps=laps[:2], after_laps=laps,
        baseline_before=baseline, baseline_after=baseline,
        car=car, track_code="BL1",
    )
    # 'before' is a refusal (insufficient stint) and emits no coherence
    # diagnostic, so the comparison is documented as incomplete.
    assert isinstance(cmp, StintComparison)
    assert any(d.key == "comparison_incomplete" for d in cmp.diagnostics)


def test_serializers_are_jargon_free(bl1_laps_and_baseline) -> None:
    laps, baseline, car = bl1_laps_and_baseline
    cmp = compare_stints(
        before_laps=laps, after_laps=laps,
        baseline_before=baseline, baseline_after=baseline,
        car=car, track_code="BL1",
    )
    md = comparison_to_markdown(cmp, car_key="FBM", track_code="BL1")
    js = comparison_to_json(cmp)
    for blob in (md, js):
        for term in FORBIDDEN_TERMS:
            assert term not in blob, (
                f"forbidden jargon {term!r} leaked into report"
            )


def test_json_roundtrip(bl1_laps_and_baseline) -> None:
    import json
    laps, baseline, car = bl1_laps_and_baseline
    cmp = compare_stints(
        before_laps=laps, after_laps=laps,
        baseline_before=baseline, baseline_after=baseline,
        car=car, track_code="BL1",
    )
    payload = json.loads(comparison_to_json(cmp))
    assert payload["coherence_change"] == pytest.approx(cmp.coherence_change)
    assert "before" in payload and "after" in payload
    # The nested advisor payloads must also be valid.
    assert payload["before"]["status"] in ("proposal", "refusal")
    assert payload["after"]["status"] in ("proposal", "refusal")
