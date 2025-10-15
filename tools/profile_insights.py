"""Profile :func:`tnfr_lfs.analysis.insights.compute_insights` using cProfile."""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import pstats
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, MutableMapping
from typing import get_args, get_origin

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from tnfr_lfs import TelemetryRecord


def _contains_type(annotation: Any, target: type) -> bool:
    """Return ``True`` when ``annotation`` includes ``target`` in its union."""

    if annotation is Any:
        return True
    if isinstance(annotation, str):
        normalised = annotation.strip().lower()
        if target is float:
            return normalised in {"float", "builtins.float"}
        if target is int:
            return normalised in {"int", "builtins.int"}
        if target is type(None):
            return normalised in {"none", "nonetype", "types.nonetype"}
        return False
    origin = get_origin(annotation)
    if origin is None:
        return annotation is target
    return any(_contains_type(arg, target) for arg in get_args(annotation))


def _coerce_value(token: str, annotation: Any) -> Any:
    token = token.strip()
    if token == "":
        return None
    try:
        if _contains_type(annotation, int) and not _contains_type(annotation, float):
            # Preserve round-trip behaviour for integer-like annotations.
            return int(float(token))
        if _contains_type(annotation, float):
            return float(token)
    except ValueError:
        return token
    return token


def load_records(path: Path) -> list["TelemetryRecord"]:
    """Load telemetry samples from ``path`` into :class:`TelemetryRecord` objects."""

    from tnfr_lfs import TelemetryRecord

    with path.open("r", encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        available = {column.strip(): column for column in reader.fieldnames}
        telemetry_fields = tuple(fields(TelemetryRecord))
        records: list[TelemetryRecord] = []
        for row in reader:
            payload: MutableMapping[str, Any] = {}
            for field in telemetry_fields:
                column = available.get(field.name)
                if not column:
                    continue
                token = row.get(column)
                if token is None or token == "":
                    continue
                value = _coerce_value(token, field.type)
                if value is None and not _contains_type(field.type, type(None)):
                    continue
                payload[field.name] = value
            records.append(TelemetryRecord(**payload))
    return records


def _resolve_sort_key(label: str) -> pstats.SortKey:
    mapping: Mapping[str, pstats.SortKey] = {
        "cumulative": pstats.SortKey.CUMULATIVE,
        "time": pstats.SortKey.TIME,
        "calls": pstats.SortKey.CALLS,
        "pcalls": pstats.SortKey.PCALLS,
        "filename": pstats.SortKey.FILENAME,
        "line": pstats.SortKey.LINE,
        "name": pstats.SortKey.NAME,
    }
    try:
        return mapping[label]
    except KeyError as error:  # pragma: no cover - defensive branch
        raise argparse.ArgumentTypeError(f"Unsupported sort key: {label}") from error


def profile_insights(
    *,
    source: Path,
    car_model: str,
    track_name: str,
    limit: int,
    sort_key: pstats.SortKey,
    profile_output: Path | None,
) -> None:
    """Run :func:`compute_insights` under :mod:`cProfile` and print hot paths."""

    from tnfr_lfs.analysis.insights import compute_insights

    records = load_records(source)
    profiler = cProfile.Profile()
    profiler.enable()
    compute_insights(records, car_model=car_model, track_name=track_name)
    profiler.disable()

    if profile_output is not None:
        profile_output.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(profile_output))

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).strip_dirs().sort_stats(sort_key)
    stats.print_stats(limit)

    report = buffer.getvalue().strip()
    if report:
        print("Top hot paths (sorted by", sort_key.name.lower(), f", top {limit}):")
        print(report)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile tnfr_lfs.analysis.insights.compute_insights using telemetry samples."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the telemetry CSV file (e.g., tests/data/synthetic_stint.csv)",
    )
    parser.add_argument(
        "--car-model",
        default="FZR",
        help="Car model used to seed the recommendation engine (default: FZR)",
    )
    parser.add_argument(
        "--track-name",
        default="AS5",
        help="Track identifier used to seed the recommendation engine (default: AS5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of rows to display from the pstats report (default: 25)",
    )
    parser.add_argument(
        "--sort",
        default="cumulative",
        type=lambda value: _resolve_sort_key(value.lower()),
        help=(
            "Sort key for the pstats output. Options: cumulative, time, calls, pcalls, "
            "filename, line, name (default: cumulative)"
        ),
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="Optional path to dump the raw cProfile stats for offline inspection.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    profile_insights(
        source=args.source,
        car_model=args.car_model,
        track_name=args.track_name,
        limit=args.limit,
        sort_key=args.sort,
        profile_output=args.profile_output,
    )


if __name__ == "__main__":  # pragma: no cover - manual tool
    main()
