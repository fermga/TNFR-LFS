"""Tests for entropy-based ΔNFR distribution and Si metrics."""

from __future__ import annotations

import math

import pytest

from tests.helpers import BASE_NU_F
from tnfr_lfs.core.coherence import compute_node_delta_nfr, sense_index
from tnfr_lfs.core.utils import normalised_entropy


PHASE_WEIGHTS = {
    "__default__": 1.0,
    "entry": {"__default__": 1.0},
    "apex": {"tyres": 1.5, "__default__": 1.3},
    "exit": 0.8,
}


def test_compute_node_delta_nfr_respects_relative_weights():
    delta = 12.0
    features = {"slip": 2.0, "load": 1.0, "camber": 1.0}

    distribution = compute_node_delta_nfr("tyres", delta, features)

    assert pytest.approx(sum(distribution.values()), rel=1e-6) == delta
    assert distribution["tyres.slip"] == pytest.approx(6.0)
    assert distribution["tyres.load"] == distribution["tyres.camber"]


def test_compute_node_delta_nfr_uniform_when_no_signal():
    delta = -9.0
    features = {"slip": 0.0, "locking": 0.0, "temperature": 0.0}

    distribution = compute_node_delta_nfr("tyres", delta, features)

    assert pytest.approx(sum(distribution.values()), rel=1e-6) == delta
    assert all(value == pytest.approx(-3.0) for value in distribution.values())


def test_compute_node_delta_nfr_allows_non_prefixed_keys():
    delta = 4.0
    features = {"inner": 1.0, "outer": 1.0}

    distribution = compute_node_delta_nfr("brakes", delta, features, prefix=False)

    assert set(distribution.keys()) == {"inner", "outer"}
    assert pytest.approx(sum(distribution.values()), rel=1e-6) == delta


def test_compute_node_delta_nfr_returns_empty_for_missing_features():
    assert compute_node_delta_nfr("track", 5.0, {}) == {}


def test_sense_index_penalises_large_delta_with_entropy():
    node_deltas = {"tyres": 6.0, "suspension": 3.0, "driver": 3.0}
    baseline = 520.0

    high_delta = sense_index(
        12.0,
        node_deltas,
        baseline,
        nu_f_by_node=BASE_NU_F,
        active_phase="entry",
        w_phase=PHASE_WEIGHTS,
    )
    scaled_deltas = {node: value * (2.0 / 12.0) for node, value in node_deltas.items()}
    small_delta = sense_index(
        2.0,
        scaled_deltas,
        baseline,
        nu_f_by_node=BASE_NU_F,
        active_phase="entry",
        w_phase=PHASE_WEIGHTS,
    )

    assert 0.0 <= high_delta <= 1.0
    assert small_delta > high_delta

    concentrated = {"tyres": 12.0, "suspension": 0.0, "driver": 0.0}
    concentrated_index = sense_index(
        12.0,
        concentrated,
        baseline,
        nu_f_by_node=BASE_NU_F,
        active_phase="entry",
        w_phase=PHASE_WEIGHTS,
    )
    assert concentrated_index >= high_delta

    assert not math.isnan(concentrated_index)


def test_sense_index_reflects_phase_weights():
    node_deltas = {"tyres": 4.0, "suspension": 2.0}
    baseline = 500.0

    entry_value = sense_index(
        6.0,
        node_deltas,
        baseline,
        nu_f_by_node=BASE_NU_F,
        active_phase="entry",
        w_phase=PHASE_WEIGHTS,
    )
    apex_value = sense_index(
        6.0,
        node_deltas,
        baseline,
        nu_f_by_node=BASE_NU_F,
        active_phase="apex",
        w_phase=PHASE_WEIGHTS,
    )
    exit_value = sense_index(
        6.0,
        node_deltas,
        baseline,
        nu_f_by_node=BASE_NU_F,
        active_phase="exit",
        w_phase=PHASE_WEIGHTS,
    )

    assert exit_value > entry_value > apex_value


def test_sense_index_penalises_fast_natural_frequencies():
    node_deltas = {"tyres": 5.0}
    baseline = 480.0

    slow_nu = {"tyres": 0.1}
    fast_nu = {"tyres": 0.4}

    slow_index = sense_index(
        5.0,
        node_deltas,
        baseline,
        nu_f_by_node=slow_nu,
        active_phase="entry",
        w_phase={"entry": 1.0},
    )
    fast_index = sense_index(
        5.0,
        node_deltas,
        baseline,
        nu_f_by_node=fast_nu,
        active_phase="entry",
        w_phase={"entry": 1.0},
    )

    assert slow_index > fast_index


def test_sense_index_penalises_goal_frequency_targets():
    node_deltas = {"tyres": 4.0, "suspension": 2.0}
    baseline = 500.0

    nu_f_map = {"tyres": 0.18, "suspension": 0.12}
    neutral_targets = {"entry": {"__default__": 0.0}}
    aggressive_targets = {"entry": {"tyres": 0.6, "__default__": 0.4}}

    neutral_index = sense_index(
        6.0,
        node_deltas,
        baseline,
        nu_f_by_node=nu_f_map,
        active_phase="entry",
        w_phase={"entry": 1.0},
        nu_f_targets=neutral_targets,
    )
    aggressive_index = sense_index(
        6.0,
        node_deltas,
        baseline,
        nu_f_by_node=nu_f_map,
        active_phase="entry",
        w_phase={"entry": 1.0},
        nu_f_targets=aggressive_targets,
    )

    assert aggressive_index < neutral_index


def test_sense_index_entropy_matches_helper():
    node_deltas = {"tyres": 3.0, "suspension": 3.0}
    nu_f = {"tyres": 0.2, "suspension": 0.1}
    entropy_lambda = 0.05

    index_value = sense_index(
        6.0,
        node_deltas,
        500.0,
        nu_f_by_node=nu_f,
        active_phase="entry",
        w_phase={"entry": 1.0},
        entropy_lambda=entropy_lambda,
    )

    weighted_sum = sum(
        (1.0 + nu_f[node]) * abs(node_deltas[node])
        for node in node_deltas
    )
    base_index = 1.0 / (1.0 + weighted_sum)
    entropy_penalty = entropy_lambda * normalised_entropy([abs(v) for v in node_deltas.values()])
    expected = max(0.0, min(1.0, base_index - entropy_penalty))

    assert index_value == pytest.approx(expected)
