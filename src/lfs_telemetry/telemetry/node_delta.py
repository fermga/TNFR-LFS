"""Continuous lap-delta tracker keyed on LFS track *node* index.

LFS publishes a track-node index (``CompCar.node`` / ``NodeLap.node``,
0..``num_nodes-1``) for every car at every IS_MCI tick. By recording
the *elapsed time since lap start* the first time the view car crossed
each node during its personal-best lap, we can later compute a
continuous "delta vs PB" by linearly interpolating the best-lap clock
at the driver's current node and subtracting it from the live elapsed
time.

This is the same idea as the in-game LFS simple-delta and the
RaceDeparture / ProDelta plugins: a per-node reference table replaces
the much sparser per-split table, so the bar can oscillate every tick
instead of jumping only at split lines.

Pure logic (no I/O, no asyncio); easy to unit-test.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field


@dataclass
class NodeDeltaTracker:
    """Per-node best-lap clock + live delta interpolation.

    Usage from the capture loop::

        tracker = NodeDeltaTracker()
        # On each IS_MCI tick, once we know the view car's node and the
        # elapsed time since the last IS_LAP:
        tracker.record(node=view.node, elapsed_ms=cur_lap_ms)
        delta = tracker.delta_ms(node=view.node, elapsed_ms=cur_lap_ms)
        # On every IS_LAP for the view car:
        tracker.complete_lap(last_lap_ms)
    """

    best_lap_ms: int | None = None
    best_node_to_ms: dict[int, int] = field(default_factory=dict)
    # Parallel per-node *speed* table (m/s) for the same lap that
    # produced ``best_node_to_ms``. Lets us expose a "speed delta vs PB
    # at the same track point" gauge, mirroring Detect&Monitor's
    # km/h-delta bar that complements the time delta.
    best_node_to_speed_ms: dict[int, float] = field(default_factory=dict)
    # Last *completed valid* lap, kept as a transient reference so the
    # delta bar has something to show on lap 2 even when lap 1 was the
    # PB itself (typical case) AND when the first attempted lap was
    # invalid (cut, /restart, lag) so no PB has been promoted yet.
    # Detect&Monitor exposes the same fallback so the gauge never
    # sits at "--.---" once the driver has done at least one lap.
    last_lap_ms: int | None = None
    _last_node_to_ms: dict[int, int] = field(default_factory=dict)
    _last_node_to_speed_ms: dict[int, float] = field(default_factory=dict)
    _last_keys_sorted: list[int] = field(default_factory=list)
    _cur_node_to_ms: dict[int, int] = field(default_factory=dict)
    _cur_node_to_speed_ms: dict[int, float] = field(default_factory=dict)
    # Cached sorted keys of ``best_node_to_ms`` for O(log n) interpolation.
    _best_keys_sorted: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def reset_lap(self) -> None:
        """Drop the in-progress lap buffer (e.g. on /restart, pit-out)."""
        self._cur_node_to_ms.clear()
        self._cur_node_to_speed_ms.clear()

    def record(
        self, *, node: int, elapsed_ms: int, speed_ms: float | None = None,
    ) -> None:
        """Note that the view car is at ``node`` after ``elapsed_ms``.

        Only the *first* (i.e. minimum) elapsed time per node within
        the current lap is kept; subsequent ticks at the same node
        (the car is moving slowly, or the publisher fires multiple
        times before the node advances) are ignored. When ``speed_ms``
        is supplied, the same first-sample-per-node policy is applied
        to the parallel speed table so both gauges share a consistent
        reference for the lap.
        """
        if node < 0 or elapsed_ms < 0:
            return
        prev = self._cur_node_to_ms.get(node)
        if prev is None or elapsed_ms < prev:
            self._cur_node_to_ms[node] = int(elapsed_ms)
            if speed_ms is not None:
                self._cur_node_to_speed_ms[node] = float(speed_ms)

    def complete_lap(self, lap_ms: int) -> None:
        """Close the current lap; promote it to PB if faster.

        ``lap_ms`` is the value LFS reports in IS_LAP. Non-positive
        values are treated as "invalid lap" and only clear the buffer.
        Valid laps are *always* stored as the "last lap" reference so
        the delta gauge has a fallback before any PB is set.
        """
        if lap_ms > 0:
            self.last_lap_ms = int(lap_ms)
            self._last_node_to_ms = dict(self._cur_node_to_ms)
            self._last_node_to_speed_ms = dict(self._cur_node_to_speed_ms)
            self._last_keys_sorted = sorted(self._last_node_to_ms)
            if self.best_lap_ms is None or lap_ms < self.best_lap_ms:
                self.best_lap_ms = int(lap_ms)
                self.best_node_to_ms = dict(self._cur_node_to_ms)
                self.best_node_to_speed_ms = dict(self._cur_node_to_speed_ms)
                self._best_keys_sorted = sorted(self.best_node_to_ms)
        self._cur_node_to_ms.clear()
        self._cur_node_to_speed_ms.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def delta_ms(
        self, *, node: int, elapsed_ms: int
    ) -> int | None:
        """Live delta (ms) vs PB at the same track node.

        Positive = slower than PB, negative = faster. Returns ``None``
        until at least one full valid lap has been recorded; falls back
        to the *last completed valid lap* when no PB exists yet so the
        gauge becomes useful from lap 2 onwards even after invalid
        first laps (Detect&Monitor parity).
        """
        if elapsed_ms < 0 or node < 0:
            return None
        if self._best_keys_sorted:
            keys = self._best_keys_sorted
            table = self.best_node_to_ms
        elif self._last_keys_sorted:
            keys = self._last_keys_sorted
            table = self._last_node_to_ms
        else:
            return None
        i = bisect_left(keys, node)
        if i < len(keys) and keys[i] == node:
            best_at = table[node]
        elif i == 0:
            best_at = table[keys[0]]
        elif i == len(keys):
            best_at = table[keys[-1]]
        else:
            lo, hi = keys[i - 1], keys[i]
            t_lo = table[lo]
            t_hi = table[hi]
            span = hi - lo
            if span <= 0:
                best_at = t_lo
            else:
                frac = (node - lo) / span
                best_at = int(t_lo + frac * (t_hi - t_lo))
        return int(elapsed_ms) - int(best_at)

    def speed_delta_ms(
        self, *, node: int, speed_ms: float,
    ) -> float | None:
        """Live speed delta (m/s) vs PB through the same track node.

        Positive = faster than PB at the same point, negative = slower.
        Mirrors :meth:`delta_ms` but on the parallel per-node speed
        table: lets the overlay show a Detect&Monitor-style "are you
        carrying more or less speed than your best lap right here?"
        bar. Falls back to the last completed valid lap when no PB
        speed table exists yet.
        """
        if node < 0:
            return None
        if self._best_keys_sorted and self.best_node_to_speed_ms:
            keys = self._best_keys_sorted
            table = self.best_node_to_speed_ms
        elif self._last_keys_sorted and self._last_node_to_speed_ms:
            keys = self._last_keys_sorted
            table = self._last_node_to_speed_ms
        else:
            return None
        i = bisect_left(keys, node)
        if i < len(keys) and keys[i] == node and node in table:
            best_at = table[node]
        elif i == 0:
            first = keys[0]
            if first not in table:
                return None
            best_at = table[first]
        elif i == len(keys):
            last = keys[-1]
            if last not in table:
                return None
            best_at = table[last]
        else:
            lo, hi = keys[i - 1], keys[i]
            if lo not in table or hi not in table:
                return None
            v_lo = table[lo]
            v_hi = table[hi]
            span = hi - lo
            if span <= 0:
                best_at = v_lo
            else:
                frac = (node - lo) / span
                best_at = v_lo + frac * (v_hi - v_lo)
        return float(speed_ms) - float(best_at)

    def ghost_node_at(self, *, elapsed_ms: int) -> int | None:
        """Find the PB node whose recorded clock matches ``elapsed_ms``.

        Used by the track-map replay ghost dots: where on the
        racing line you would be right now if you were on PB pace.
        Falls back to the last completed valid lap when no PB exists
        yet, mirroring :meth:`delta_ms`. Returns ``None`` until at
        least one full lap has been recorded.
        """
        if elapsed_ms < 0:
            return None
        if self._best_keys_sorted:
            keys = self._best_keys_sorted
            table = self.best_node_to_ms
        elif self._last_keys_sorted:
            keys = self._last_keys_sorted
            table = self._last_node_to_ms
        else:
            return None
        # Pairs are (best_time_ms, node), sorted by node. We want the
        # node whose best_time is closest to elapsed_ms; do a linear
        # scan because n_nodes is small (~few hundred for any LFS
        # track) and ticks are 10 Hz.
        best_diff = None
        best_node = None
        for node in keys:
            t = table[node]
            diff = abs(int(elapsed_ms) - t)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_node = node
        return best_node


__all__ = ["NodeDeltaTracker"]
