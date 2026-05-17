"""Tests for the Qt-free serializers in ``tnfr_racing.serialize``.

These cover the public surface that the Studio UI and the CLI both
render through, so they pin down: jargon-free output, deterministic
JSON, refusal-path rendering and that the consolidated synthesis
section reaches the user.
"""
from __future__ import annotations

import json

import pytest

from lfs_telemetry.tnfr_racing.advisor import (
    AdvisorResult, Diagnostic, SetupAdvisor,
)
from lfs_telemetry.tnfr_racing.serialize import (
    format_refusal,
    humanize_action,
    result_to_json,
    result_to_markdown,
)

from .test_advisor import bl1_inputs  # noqa: F401 — reuse fixture
from .test_sensitivities import FORBIDDEN_TERMS


# ---------------------------------------------------------------------------
# Refusal path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", [
    "insufficient_stint:got 3 laps",
    "no_rule_fired",
    "grammar_U_violation",
    "no_coherence_positive_rule",
    "sector_decomposition_failed",
    "unknown_reason_xyz",
])
def test_format_refusal_is_jargon_free(reason) -> None:
    text = format_refusal(reason)
    assert text
    for term in FORBIDDEN_TERMS:
        assert term not in text


def test_refusal_result_renders_in_json_and_markdown() -> None:
    res = AdvisorResult.no_safe_recommendation(
        reason="insufficient_stint:got 3 laps",
        diagnostics=(Diagnostic(key="laps", message="Only 3 laps received.",
                                value=3.0),),
    )
    js = json.loads(result_to_json(res))
    assert js["status"] == "refusal"
    assert js["refusal_reason"].startswith("insufficient_stint")
    assert "Need" in js["refusal_message"]
    md = result_to_markdown(res, car_key="FBM", track_code="BL1", n_laps=3)
    assert "no safe recommendation" in md.lower()
    assert "Only 3 laps received." in md


# ---------------------------------------------------------------------------
# Proposal path (uses the synthetic BL1/FBM stint)
# ---------------------------------------------------------------------------


def test_proposal_json_is_valid_deterministic_and_jargon_free(
    bl1_inputs,  # noqa: F811
) -> None:
    laps, baseline, car = bl1_inputs
    res = SetupAdvisor(seed=17).advise(laps, baseline, car, "BL1")
    if res.proposed is None:
        pytest.skip("no proposal on this stint")

    a = result_to_json(res)
    b = result_to_json(res)
    assert a == b, "serializer must be deterministic"

    payload = json.loads(a)
    assert payload["status"] == "proposal"
    # Mandatory keys downstream consumers (CLI, UI, history) read.
    for key in (
        "actions", "consolidated_setup", "conflict_groups",
        "expected_coherence_delta", "grammar_passed",
        "baseline_hash", "stint_signature", "coherence_before",
        "coherence_after", "diagnostics",
    ):
        assert key in payload, f"missing key: {key}"

    # Every consolidated entry must correspond to a real action channel.
    action_channels = {(a["kind"], a["target"]) for a in payload["actions"]}
    synth_channels = {(c["kind"], c["target"])
                       for c in payload["consolidated_setup"]}
    assert action_channels == synth_channels

    # Jargon scan over the whole JSON blob.
    for term in FORBIDDEN_TERMS:
        assert term not in a, f"jargon leaked into JSON: {term!r}"


def test_proposal_markdown_is_jargon_free_and_contains_synthesis(
    bl1_inputs,  # noqa: F811
) -> None:
    laps, baseline, car = bl1_inputs
    res = SetupAdvisor(seed=17).advise(laps, baseline, car, "BL1")
    if res.proposed is None:
        pytest.skip("no proposal on this stint")
    md = result_to_markdown(
        res, car_key="FBM", track_code="BL1",
        n_laps=len(laps), baseline=baseline,
    )
    assert "## Proposed changes" in md
    assert "## Consolidated optimal setup" in md
    assert "## Diagnostics" in md
    for term in FORBIDDEN_TERMS:
        assert term not in md, f"jargon leaked into markdown: {term!r}"


def test_proposal_markdown_omits_synthesis_when_no_baseline(
    bl1_inputs,  # noqa: F811
) -> None:
    """Synthesis section must still render even without a baseline,
    because it depends only on the proposed actions."""
    laps, baseline, car = bl1_inputs
    res = SetupAdvisor(seed=17).advise(laps, baseline, car, "BL1")
    if res.proposed is None:
        pytest.skip("no proposal on this stint")
    md = result_to_markdown(res)  # no baseline argument
    assert "## Consolidated optimal setup" in md


# ---------------------------------------------------------------------------
# humanize_action low-level contract
# ---------------------------------------------------------------------------


def test_humanize_action_sign_and_units() -> None:
    from lfs_telemetry.tnfr_racing.operators import PhysicalAction
    act = PhysicalAction(
        kind="spring", target="front", delta=-3.0, units="N/mm",
        rationale_id="example_rule",
    )
    sub, change, rationale = humanize_action(act)
    assert sub == "Spring rate (front axle)"
    assert change == "-3 N/mm"
    assert "example rule" in rationale  # underscores stripped
    for term in FORBIDDEN_TERMS:
        assert term not in (sub + change + rationale)
