"""Live lap-time predictor: Sum-of-Personal-Best splits (SPB).

This is a pure-logic helper (no I/O, no asyncio) that ingests the split
events LFS sends via InSim:

* :class:`lfs_telemetry.telemetry.protocol.packets.InSimSplitX` — fired at
  each split with the *cumulative* time since lap start;
* :class:`lfs_telemetry.telemetry.protocol.packets.InSimLap` — fired at the
  start/finish line with the full lap time.

From those, it maintains:

* :attr:`best_lap_ms` — fastest full lap seen so far,
* :attr:`best_segments_ms` — fastest *segment* (split N − split N−1)
  ever seen, per segment index (the final segment goes from the last
  split to the line),
* a transient view of the lap currently under way.

Two metrics that an overlay/dashboard typically wants:

* :meth:`spb_ms` — Sum-of-Personal-Best segment times (a hypothetical
  perfect lap with the personal best of every segment);
* :meth:`predicted_lap_ms` — projected time of the lap currently under
  way, computed as ``elapsed + best_remaining``.

Both round-trip cleanly through :meth:`to_dict` / :meth:`from_dict` so
they can be persisted between sessions (``laps/<track>_<car>.json``).

Example::

    pred = SplitPredictor(n_splits=2)         # 2 splits → 3 segments (BL1)
    pred.observe_split(1, 28_500)             # IS_SPX split=1 split_time=28.5s
    pred.observe_split(2, 58_200)             # IS_SPX split=2
    pred.observe_lap(86_500)                  # IS_LAP last_lap_ms=86.5s
    pred.spb_ms                               # → e.g. 86_100
    pred.predicted_lap_ms(elapsed_ms=30_000, last_split_idx=1)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SplitPredictor:
    """Personal-best segment store + live lap projection.

    ``n_splits`` is the number of *intermediate* splits per lap (LFS
    typically emits IS_SPX with ``split=1..3``). Total segments per
    lap is ``n_splits + 1`` (the last segment runs from the final
    split to the start/finish line).

    ``best_segments_ms[i]`` (1-based segment index) is the fastest
    time ever observed for that segment in any lap.
    """

    n_splits: int
    best_lap_ms: int | None = None
    best_segments_ms: dict[int, int] = field(default_factory=dict)
    # Transient buffer for the lap underway: cumulative split times.
    _current_splits_ms: dict[int, int] = field(default_factory=dict)

    @property
    def n_segments(self) -> int:
        return self.n_splits + 1

    # ------------------------------------------------------------------
    # Event observers
    # ------------------------------------------------------------------

    def observe_split(self, split_idx: int, cumulative_ms: int) -> None:
        """Ingest one IS_SPX event (cumulative time since lap start)."""
        if split_idx < 1 or split_idx > self.n_splits:
            return
        if cumulative_ms < 0:
            return
        self._current_splits_ms[split_idx] = int(cumulative_ms)
        # Update best for this segment if the previous split is known
        # (or this is the first segment).
        prev = (self._current_splits_ms.get(split_idx - 1, 0)
                if split_idx > 1 else 0)
        seg_ms = int(cumulative_ms) - prev
        if seg_ms > 0:
            cur_best = self.best_segments_ms.get(split_idx)
            if cur_best is None or seg_ms < cur_best:
                self.best_segments_ms[split_idx] = seg_ms

    def observe_lap(self, lap_ms: int) -> None:
        """Ingest one IS_LAP event (full lap time)."""
        if lap_ms <= 0:
            return
        # Update the last segment best, then reset transient state.
        if self._current_splits_ms:
            last_split = max(self._current_splits_ms)
            last_cum = self._current_splits_ms[last_split]
            tail_ms = int(lap_ms) - last_cum
            if tail_ms > 0:
                cur_best = self.best_segments_ms.get(self.n_segments)
                if cur_best is None or tail_ms < cur_best:
                    self.best_segments_ms[self.n_segments] = tail_ms
        if self.best_lap_ms is None or lap_ms < self.best_lap_ms:
            self.best_lap_ms = int(lap_ms)
        self._current_splits_ms.clear()

    def reset_lap(self) -> None:
        """Drop the transient buffer (e.g. on /restart, pit-out)."""
        self._current_splits_ms.clear()

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def spb_ms(self) -> int | None:
        """Sum of personal-best segments. ``None`` if any is missing."""
        if len(self.best_segments_ms) != self.n_segments:
            return None
        return int(sum(self.best_segments_ms.values()))

    def best_cumulative_at_split(self, split_idx: int) -> int | None:
        """SPB-cumulative time at the end of segment ``split_idx``."""
        if split_idx < 1 or split_idx > self.n_segments:
            return None
        total = 0
        for i in range(1, split_idx + 1):
            v = self.best_segments_ms.get(i)
            if v is None:
                return None
            total += v
        return total

    def predicted_lap_ms(
        self,
        *,
        elapsed_ms: int,
        last_split_idx: int,
    ) -> int | None:
        """Project the lap currently underway.

        ``last_split_idx`` is the index of the most recently completed
        split (``0`` if no split crossed yet). ``elapsed_ms`` is the
        time since lap start at the moment of the query.

        Result is ``elapsed_ms + sum(best_segments[last_split_idx+1:])``
        — i.e. assume the remaining segments are run at the personal
        best. Returns ``None`` if any remaining segment best is unknown
        and no fallback (``best_lap_ms``) is available.
        """
        if elapsed_ms < 0:
            return None
        remaining = 0
        missing = False
        for i in range(last_split_idx + 1, self.n_segments + 1):
            v = self.best_segments_ms.get(i)
            if v is None:
                missing = True
                break
            remaining += v
        if not missing:
            return int(elapsed_ms + remaining)
        # Fallback: subtract best cumulative at last_split_idx from
        # best_lap_ms.
        if self.best_lap_ms is None:
            return None
        if last_split_idx == 0:
            return int(elapsed_ms + self.best_lap_ms)
        cum_best = self.best_cumulative_at_split(last_split_idx)
        if cum_best is None:
            return None
        return int(elapsed_ms + max(0, self.best_lap_ms - cum_best))

    def delta_to_best_ms(
        self,
        *,
        elapsed_ms: int,
        last_split_idx: int,
    ) -> int | None:
        """Live delta to the personal best at the same point in lap.

        Positive = slower than PB, negative = faster.
        """
        cum_best = (
            self.best_cumulative_at_split(last_split_idx)
            if last_split_idx > 0 else 0
        )
        if cum_best is None:
            return None
        # Add the partial elapsed since the last split to the cumulative
        # best at the last split → that's the comparable time.
        # However ``elapsed_ms`` is total since lap start, not since
        # the last split, so the comparison is direct: actual_cum vs
        # best_cum at the same split crossing point.
        if last_split_idx == 0:
            # No split yet — nothing to compare against.
            return None
        # We compare actual cumulative at the last completed split
        # against the personal-best cumulative at that same split.
        actual_at_split = self._current_splits_ms.get(last_split_idx)
        if actual_at_split is None:
            return None
        return int(actual_at_split - cum_best)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_splits": self.n_splits,
            "best_lap_ms": self.best_lap_ms,
            "best_segments_ms": {str(k): int(v)
                                 for k, v in self.best_segments_ms.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> SplitPredictor:
        return cls(
            n_splits=int(data["n_splits"]),
            best_lap_ms=(int(data["best_lap_ms"])
                         if data.get("best_lap_ms") is not None else None),
            best_segments_ms={int(k): int(v)
                              for k, v in (data.get("best_segments_ms")
                                           or {}).items()},
        )


__all__ = ["SplitPredictor"]
