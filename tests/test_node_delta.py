"""Unit tests for :mod:`lfs_telemetry.telemetry.node_delta`."""

from __future__ import annotations

from lfs_telemetry.telemetry.node_delta import NodeDeltaTracker


def test_no_delta_before_first_complete_lap() -> None:
    tracker = NodeDeltaTracker()
    tracker.record(node=10, elapsed_ms=1_000)
    tracker.record(node=20, elapsed_ms=2_000)
    assert tracker.delta_ms(node=15, elapsed_ms=1_500) is None


def test_delta_zero_on_pb_lap_replay() -> None:
    tracker = NodeDeltaTracker()
    for node, t in [(0, 0), (50, 5_000), (100, 10_000), (150, 15_000)]:
        tracker.record(node=node, elapsed_ms=t)
    tracker.complete_lap(15_000)
    # Re-driving at exactly the same pace → zero delta at known nodes
    # and at interpolated nodes.
    assert tracker.delta_ms(node=50, elapsed_ms=5_000) == 0
    # Halfway between node 50 (5s) and node 100 (10s): expect 7.5s.
    assert tracker.delta_ms(node=75, elapsed_ms=7_500) == 0


def test_positive_delta_when_slower() -> None:
    tracker = NodeDeltaTracker()
    for node, t in [(0, 0), (100, 10_000)]:
        tracker.record(node=node, elapsed_ms=t)
    tracker.complete_lap(20_000)
    # On lap 2 we are 300 ms slower at node 50 (best interpolated 5s).
    assert tracker.delta_ms(node=50, elapsed_ms=5_300) == 300


def test_negative_delta_when_faster() -> None:
    tracker = NodeDeltaTracker()
    for node, t in [(0, 0), (100, 10_000)]:
        tracker.record(node=node, elapsed_ms=t)
    tracker.complete_lap(20_000)
    assert tracker.delta_ms(node=50, elapsed_ms=4_700) == -300


def test_pb_only_replaced_when_faster() -> None:
    tracker = NodeDeltaTracker()
    for node, t in [(0, 0), (100, 10_000)]:
        tracker.record(node=node, elapsed_ms=t)
    tracker.complete_lap(20_000)
    # Slower lap: PB stays.
    for node, t in [(0, 0), (100, 11_000)]:
        tracker.record(node=node, elapsed_ms=t)
    tracker.complete_lap(22_000)
    assert tracker.best_lap_ms == 20_000
    assert tracker.best_node_to_ms[100] == 10_000


def test_invalid_lap_just_resets_buffer() -> None:
    tracker = NodeDeltaTracker()
    tracker.record(node=10, elapsed_ms=500)
    tracker.complete_lap(0)  # invalid
    assert tracker.best_lap_ms is None
    assert tracker._cur_node_to_ms == {}


def test_record_keeps_minimum_elapsed_per_node() -> None:
    tracker = NodeDeltaTracker()
    tracker.record(node=42, elapsed_ms=1_000)
    tracker.record(node=42, elapsed_ms=1_200)  # later tick same node
    tracker.complete_lap(2_000)
    assert tracker.best_node_to_ms[42] == 1_000


def test_delta_extrapolates_with_endpoints() -> None:
    tracker = NodeDeltaTracker()
    for node, t in [(10, 1_000), (90, 9_000)]:
        tracker.record(node=node, elapsed_ms=t)
    tracker.complete_lap(10_000)
    # Node 5 < first known key 10 → use endpoint 1_000.
    assert tracker.delta_ms(node=5, elapsed_ms=1_500) == 500
    # Node 95 > last known key 90 → use endpoint 9_000.
    assert tracker.delta_ms(node=95, elapsed_ms=8_500) == -500


def test_ghost_node_returns_none_before_first_lap() -> None:
    tracker = NodeDeltaTracker()
    assert tracker.ghost_node_at(elapsed_ms=1_000) is None


def test_ghost_node_finds_closest_pb_node() -> None:
    tracker = NodeDeltaTracker()
    for node, t in [(0, 0), (50, 5_000), (100, 10_000)]:
        tracker.record(node=node, elapsed_ms=t)
    tracker.complete_lap(10_000)
    # Exact match.
    assert tracker.ghost_node_at(elapsed_ms=5_000) == 50
    # Closer to node 50 than to node 100.
    assert tracker.ghost_node_at(elapsed_ms=6_000) == 50
    # Closer to node 100.
    assert tracker.ghost_node_at(elapsed_ms=9_500) == 100


def test_ghost_node_negative_elapsed_returns_none() -> None:
    tracker = NodeDeltaTracker()
    for node, t in [(0, 0), (10, 1_000)]:
        tracker.record(node=node, elapsed_ms=t)
    tracker.complete_lap(1_000)
    assert tracker.ghost_node_at(elapsed_ms=-1) is None
