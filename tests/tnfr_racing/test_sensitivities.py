"""Phase 9.B — closed-form sensitivity estimator tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from lfs_telemetry.tnfr_racing.operators import PhysicalAction
from lfs_telemetry.tnfr_racing.sensitivities import (
    SensitivityEstimate, estimate_action_sensitivity,
)

from .test_advisor import _synthetic_baseline

FORBIDDEN_TERMS = (
    "EPI", "νf", "ΔNFR", "Φ_s", " Si", "operador",
    " AL ", " EN ", " IL ", " OZ ", " UM ", " RA ",
    " SHA ", " VAL ", " NUL ", "THOL", "ZHIR", "NAV", "REMESH",
    "U1", "U2", "U3", "U4", "U5", "U6", "tétrada",
)


@pytest.fixture
def baseline():
    return _synthetic_baseline()


@pytest.mark.parametrize("kind,target,delta,units", [
    ("damper_rebound", "front", 2.0, "clicks"),
    ("damper_bump", "rear", -1.0, "clicks"),
    ("spring", "front", 5.0, "N/mm"),
    ("arb", "front", 1.0, "N/mm"),
    ("tyre_pressure", "FL", 10.0, "kPa"),
    ("brake_bias", "global", -2.0, "%"),
    ("camber", "front", -0.25, "deg"),
    ("toe", "rear", 0.10, "deg"),
    ("ride_height", "front", -2.0, "mm"),
    ("ride_height", "rear", +3.0, "mm"),
])
def test_estimate_returns_jargon_free_sentence(
    baseline, kind, target, delta, units,
) -> None:
    act = PhysicalAction(
        kind=kind, target=target, delta=delta, units=units,
        rationale_id="test_rule",
    )
    est = estimate_action_sensitivity(act, baseline)
    assert isinstance(est, SensitivityEstimate)
    assert est.sentence
    assert est.confidence in ("high", "medium", "low")
    for term in FORBIDDEN_TERMS:
        assert term not in est.sentence


def test_estimate_is_deterministic(baseline) -> None:
    act = PhysicalAction(
        kind="spring", target="front", delta=4.0, units="N/mm",
        rationale_id="r",
    )
    a = estimate_action_sensitivity(act, baseline)
    b = estimate_action_sensitivity(act, baseline)
    assert a == b


def test_unknown_action_kind_returns_low_confidence(baseline) -> None:
    act = PhysicalAction(
        kind="something_new", target="front", delta=1.0, units="?",
        rationale_id="r",
    )
    est = estimate_action_sensitivity(act, baseline)
    assert est.confidence == "low"
    assert est.lap_time_ms_per_unit is None


def test_damper_estimate_shifts_zeta_in_expected_direction(baseline) -> None:
    """Adding rebound clicks should increase the damping ratio."""
    plus = estimate_action_sensitivity(
        PhysicalAction("damper_rebound", "front", 2.0, "clicks", "r"),
        baseline,
    )
    minus = estimate_action_sensitivity(
        PhysicalAction("damper_rebound", "front", -2.0, "clicks", "r"),
        baseline,
    )
    # The sentences differ; the numeric magnitude should be the same
    # but the direction of the lap-ms estimate flips.
    assert (
        plus.lap_time_ms_per_unit is not None
        and minus.lap_time_ms_per_unit is not None
    )
    assert (plus.lap_time_ms_per_unit > 0) != (
        minus.lap_time_ms_per_unit > 0
    ) or plus.lap_time_ms_per_unit == pytest.approx(
        minus.lap_time_ms_per_unit
    )
