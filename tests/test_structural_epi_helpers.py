"""Unit tests for structural EPI helper utilities."""

from __future__ import annotations

import pytest

from tnfr_core.operators.structural.epi import (
    PhaseContext,
    compute_nodal_contributions,
    extract_phase_context,
    resolve_nu_targets,
)


def test_extract_phase_context_prefers_attributes() -> None:
    class Delta:
        __theta__ = "apex"
        __w_phase__ = {
            "apex": {"__default__": 0.75, "tyres": 2.0},
            "__default__": {"__default__": 1.0},
        }

    context = extract_phase_context(Delta())

    assert isinstance(context, PhaseContext)
    assert context.identifier == "apex"
    assert context.weights is Delta.__w_phase__


def test_resolve_nu_targets_filters_invalid_values() -> None:
    delta = {
        "nu_f_objectives": {"tyres": "0.5", "driver": object(), "chassis": None}
    }

    targets = resolve_nu_targets(delta)

    assert targets is not None
    assert set(targets) == {"tyres"}
    assert targets["tyres"] == pytest.approx(0.5)


def test_compute_nodal_contributions_respects_phase_and_objectives() -> None:
    deltas = {
        "tyres": 1.0,
        "suspension": -0.4,
        "__theta__": "apex",
        "__w_phase__": {
            "apex": {"__default__": 0.5, "tyres": 2.0},
            "__default__": {"__default__": 1.0},
        },
        "nu_f_objectives": {"suspension": "0.2"},
    }
    nu_map = {"tyres": 0.2, "suspension": 0.15}
    phase_context = extract_phase_context(deltas)
    nu_targets = resolve_nu_targets(deltas)

    contributions, theta_effects, derivative = compute_nodal_contributions(
        deltas, nu_map, nu_targets, phase_context, 0.2
    )

    assert pytest.approx(0.36, rel=1e-9) == derivative
    assert contributions["tyres"][1] == pytest.approx(0.4, rel=1e-9)
    assert contributions["tyres"][0] == pytest.approx(0.08, rel=1e-9)
    assert contributions["suspension"][1] == pytest.approx(-0.04, rel=1e-9)
    assert contributions["suspension"][0] == pytest.approx(-0.008, rel=1e-9)
    assert set(theta_effects) == {"tyres", "suspension"}
    assert theta_effects["tyres"] == pytest.approx(2.0, rel=1e-9)
    assert theta_effects["suspension"] == pytest.approx(0.5, rel=1e-9)
