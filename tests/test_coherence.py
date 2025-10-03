"""Tests for entropy-based ΔNFR distribution and Si metrics."""

from __future__ import annotations

import math

import pytest

from tnfr_lfs.core.coherence import compute_node_delta_nfr, sense_index


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


def test_sense_index_penalises_large_delta_with_entropy():
    node_deltas = {"tyres": 6.0, "suspension": 3.0, "driver": 3.0}
    baseline = 520.0

    high_delta = sense_index(12.0, node_deltas, baseline)
    small_delta = sense_index(2.0, node_deltas, baseline)

    assert 0.0 <= high_delta <= 1.0
    assert small_delta > high_delta

    concentrated = {"tyres": 12.0, "suspension": 0.0, "driver": 0.0}
    concentrated_index = sense_index(12.0, concentrated, baseline)
    assert concentrated_index >= high_delta

    assert not math.isnan(concentrated_index)
