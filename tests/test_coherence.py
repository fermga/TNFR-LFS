"""Tests for entropy-based ΔNFR distribution and Si metrics."""

from __future__ import annotations

import math

import pytest

from tnfr_lfs.core.coherence import compute_node_delta_nfr, sense_index


def test_compute_node_delta_nfr_respects_relative_weights():
    delta = 12.0
    features = {"tyres": 2.0, "suspension": 1.0, "driver": 1.0}

    distribution = compute_node_delta_nfr(delta, features)

    assert pytest.approx(sum(distribution.values()), rel=1e-6) == delta
    assert distribution["tyres"] == pytest.approx(6.0)
    assert distribution["suspension"] == distribution["driver"]


def test_compute_node_delta_nfr_uniform_when_no_signal():
    delta = -9.0
    features = {"tyres": 0.0, "suspension": 0.0, "brakes": 0.0}

    distribution = compute_node_delta_nfr(delta, features)

    assert pytest.approx(sum(distribution.values()), rel=1e-6) == delta
    assert all(value == pytest.approx(-3.0) for value in distribution.values())


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
