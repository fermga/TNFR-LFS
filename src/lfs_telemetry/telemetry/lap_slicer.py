"""Canonical line-to-line lap slicing.

The cleanest, frame-accurate way to know that the car crossed the
start/finish line is the OutSim ``current_lap_dist_m`` channel: when the
car crosses the line it resets from ``~track_length`` to ``0``. This is
geometrically canonical and does not depend on the (possibly delayed)
arrival of an InSim ``IS_LAP`` packet.

This module exposes the helpers used by the CLI to:

* find the indices of all line crossings inside a buffer of
  :class:`~lfs_telemetry.telemetry.live.TelemetrySample`,
* slice a buffer into *full* laps (everything between two consecutive
  line crossings; partial pre/post laps are dropped),
* re-slice a previously captured aggregate CSV into clean per-lap CSVs.

The IS_LAP-derived ``last_lap_ms`` from LFS, when available on the first
sample of the next lap, is exposed as :pyattr:`LapSlice.lap_ms` so the
official LFS lap time stays attached to each canonical slice.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .live import TelemetrySample
from .replay import read_csv_replay, write_csv_replay


@dataclass(frozen=True, slots=True)
class LapSlice:
    """A canonical line-to-line lap slice.

    ``lap_index`` is 1-based and counts only the *full* laps recovered
    from the buffer. ``samples`` are the raw fused samples between the
    two line crossings ``[start_idx, end_idx)`` of the source buffer.
    ``lap_ms`` carries the InSim-reported lap time (``last_lap_ms`` on
    the first sample of the *next* lap) when available, else ``None``.
    """

    lap_index: int
    samples: list[TelemetrySample]
    start_idx: int
    end_idx: int
    duration_s: float
    distance_m: float
    lap_ms: int | None = None


def _lap_distance(sample: TelemetrySample) -> float | None:
    """Return ``current_lap_dist_m`` from OutSimPack2 if present."""
    pkt2 = sample.outsim2
    if pkt2 is None:
        return None
    return pkt2.current_lap_dist_m


def find_line_crossings(
    samples: Sequence[TelemetrySample],
    *,
    min_drop_m: float = 100.0,
) -> list[int]:
    """Return indices ``i`` where ``samples[i]`` is the first sample of a
    new lap, i.e. ``d[i] < d[prev] - min_drop_m`` for the previous
    sample with a defined ``current_lap_dist_m``.

    The default ``min_drop_m`` (100 m) is well above any realistic
    intra-lap regression noise (numerical jitter, brief reversal during
    spins) yet far below any realistic lap length, so it cleanly
    separates true line crossings from artefacts.
    """
    crossings: list[int] = []
    prev: float | None = None
    for i, sample in enumerate(samples):
        d = _lap_distance(sample)
        if d is None:
            continue
        if prev is not None and d < prev - min_drop_m:
            crossings.append(i)
        prev = d
    return crossings


def _lap_ms_at(sample: TelemetrySample) -> int | None:
    """Best-effort: read the view-player's ``last_lap_ms`` from the
    InSim race context attached to this sample.
    """
    rc = sample.race_context
    if rc is None:
        return None
    plid = rc.view_player_id
    if plid is None:
        # Fallback: the player with the highest lap_count is most likely
        # the local driver (matches the CLI's discovery heuristic).
        if not rc.lap_count:
            return None
        plid = max(rc.lap_count, key=rc.lap_count.get)  # type: ignore[arg-type]
    return rc.last_lap_ms.get(plid)


def slice_into_laps(
    samples: Sequence[TelemetrySample],
    *,
    min_drop_m: float = 100.0,
    include_out_lap: bool = False,
) -> list[LapSlice]:
    """Slice ``samples`` into canonical full laps.

    A full lap is the segment between two consecutive line crossings.
    Anything after the last crossing (the truncated post-stop lap) is
    always *dropped*. The pre-first-crossing segment (the out-lap from
    pit/grid exit to the first start/finish crossing) is dropped by
    default; pass ``include_out_lap=True`` to keep it as ``lap_index=0``.
    """
    crossings = find_line_crossings(samples, min_drop_m=min_drop_m)
    laps: list[LapSlice] = []
    if include_out_lap and crossings and crossings[0] > 0:
        out_samples = list(samples[: crossings[0]])
        if out_samples:
            t0 = out_samples[0].time_ms / 1000.0
            t1 = out_samples[-1].time_ms / 1000.0
            dmax = 0.0
            for s in out_samples:
                d = _lap_distance(s)
                if d is not None and d > dmax:
                    dmax = d
            # The IS_LAP for the out-lap (if any) lands on samples[crossings[0]].
            lap_ms = _lap_ms_at(samples[crossings[0]])
            laps.append(
                LapSlice(
                    lap_index=0,
                    samples=out_samples,
                    start_idx=0,
                    end_idx=crossings[0],
                    duration_s=t1 - t0,
                    distance_m=dmax,
                    lap_ms=lap_ms,
                )
            )
    for n, (a, b) in enumerate(zip(crossings, crossings[1:], strict=False), start=1):
        slice_samples = list(samples[a:b])
        if not slice_samples:
            continue
        t0 = slice_samples[0].time_ms / 1000.0
        t1 = slice_samples[-1].time_ms / 1000.0
        dmax = 0.0
        for s in slice_samples:
            d = _lap_distance(s)
            if d is not None and d > dmax:
                dmax = d
        # The IS_LAP packet for lap N typically updates the race context
        # *just before* sample[b] (the first sample of lap N+1), so the
        # canonical place to look up the lap time is samples[b].
        lap_ms: int | None = None
        if b < len(samples):
            lap_ms = _lap_ms_at(samples[b])
        laps.append(
            LapSlice(
                lap_index=n,
                samples=slice_samples,
                start_idx=a,
                end_idx=b,
                duration_s=t1 - t0,
                distance_m=dmax,
                lap_ms=lap_ms,
            )
        )
    return laps


def write_per_lap_files(
    samples: Sequence[TelemetrySample],
    *,
    out_dir: Path,
    stem: str,
    suffix: str = ".csv",
    session_tag: str = "",
    min_drop_m: float = 100.0,
    include_out_lap: bool = False,
    skip_lap_indices: Iterable[int] = (),
) -> list[tuple[Path, LapSlice, int]]:
    """Slice ``samples`` canonically and write one CSV per full lap.

    Returns the list of ``(path, LapSlice, rows_written)`` for every lap
    that was actually written. The naming follows the existing capture
    convention: ``{stem}_{session_tag}_lapNN{suffix}`` (the underscore
    before ``session_tag`` is omitted when ``session_tag`` is empty).
    With ``include_out_lap=True`` the out-lap is written as ``_lap00``.

    ``skip_lap_indices`` lets callers skip laps that have already been
    written incrementally during a live capture (streaming per-lap
    mode). Skipped laps are still returned in the result list with
    their ``LapSlice`` and ``rows_written=0`` so the caller can log
    them, but no file is (re)written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, LapSlice, int]] = []
    skip = frozenset(int(i) for i in skip_lap_indices)
    laps = slice_into_laps(
        samples, min_drop_m=min_drop_m, include_out_lap=include_out_lap,
    )
    for lap in laps:
        tag = f"_{session_tag}" if session_tag else ""
        path = out_dir / f"{stem}{tag}_lap{lap.lap_index:02d}{suffix}"
        if lap.lap_index in skip:
            written.append((path, lap, 0))
            continue
        rows = write_csv_replay(path, lap.samples)
        written.append((path, lap, rows))
    return written


def reslice_csv(
    src_csv: Path,
    *,
    out_dir: Path | None = None,
    stem: str | None = None,
    suffix: str = ".csv",
    session_tag: str = "",
    min_drop_m: float = 100.0,
    include_out_lap: bool = False,
) -> list[tuple[Path, LapSlice, int]]:
    """Re-slice a previously captured aggregate CSV into clean per-lap
    files using the canonical ``current_lap_dist_m`` wraparound.

    ``out_dir`` defaults to ``src_csv.parent``; ``stem`` defaults to
    ``src_csv.stem``. The original file is left untouched.
    """
    src_csv = Path(src_csv)
    if out_dir is None:
        out_dir = src_csv.parent
    if stem is None:
        stem = src_csv.stem
    samples: list[TelemetrySample] = list(read_csv_replay(src_csv))
    return write_per_lap_files(
        samples,
        out_dir=out_dir,
        stem=stem,
        suffix=suffix,
        session_tag=session_tag,
        min_drop_m=min_drop_m,
        include_out_lap=include_out_lap,
    )


__all__ = [
    "LapSlice",
    "find_line_crossings",
    "slice_into_laps",
    "write_per_lap_files",
    "reslice_csv",
]
