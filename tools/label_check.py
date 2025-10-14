"""Validate labelled microsector artefacts for calibration workflows.

The command line interface exposed by this module reuses the label loaders
available in :mod:`tools.calibrate_detectors` to avoid duplicating the parsing
logic across utilities.  It focuses on static checks for the artefacts that are
typically supplied to the calibration pipeline:

* highlight *overlaps* (duplicate microsector indices per capture),
* detect *gaps* across the labelled microsector ranges, and
* provide a per-lap coverage summary derived from contiguous microsector spans.

Additionally, the validator inspects operator interval payloads to flag invalid
windows and cross-checks capture level metadata to surface inconsistencies that
could derail the calibration run.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from tools.calibrate_detectors import (
    LabelledMicrosector,
    _load_labels,
    _normalise_interval_bounds,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LapCoverage:
    """Contiguous microsector span treated as a labelled lap."""

    start_index: int
    end_index: int
    count: int


@dataclass(frozen=True)
class CaptureReport:
    """Summary of the validation results for a single capture."""

    capture_id: str
    raf_path: Path | None
    label_count: int
    coverage: tuple[LapCoverage, ...]
    overlaps: Mapping[int, int]
    gaps: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ValidationIssue:
    """Problem surfaced while inspecting the label artefact."""

    severity: Literal["warning", "error"]
    capture_id: str | None
    message: str


def _format_gap(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _compute_lap_coverage(indices: Sequence[int]) -> tuple[LapCoverage, ...]:
    if not indices:
        return ()
    segments: list[LapCoverage] = []
    start = prev = indices[0]
    for index in indices[1:]:
        if index == prev + 1:
            prev = index
            continue
        segments.append(LapCoverage(start, prev, prev - start + 1))
        start = prev = index
    segments.append(LapCoverage(start, prev, prev - start + 1))
    return tuple(segments)


def _detect_overlaps(indices: Sequence[int]) -> Mapping[int, int]:
    counts = Counter(indices)
    return {index: count for index, count in counts.items() if count > 1}


def _detect_gaps(sorted_unique_indices: Sequence[int]) -> tuple[tuple[int, int], ...]:
    gaps: list[tuple[int, int]] = []
    if not sorted_unique_indices:
        return ()
    previous = sorted_unique_indices[0]
    for index in sorted_unique_indices[1:]:
        if index - previous > 1:
            gaps.append((previous + 1, index - 1))
        previous = index
    return tuple(gaps)


def _check_metadata(
    capture_id: str,
    labels: Sequence[LabelledMicrosector],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    field_labels = {
        "track": "track",
        "car": "car",
        "car_class": "car class",
        "compound": "compound",
    }
    for field, description in field_labels.items():
        values = {
            getattr(label, field)
            for label in labels
            if getattr(label, field)
        }
        if len(values) > 1:
            value_list = ", ".join(sorted(str(value) for value in values))
            issues.append(
                ValidationIssue(
                    "warning",
                    capture_id,
                    f"Capture '{capture_id}' has inconsistent {description} metadata: {value_list}",
                )
            )
    raf_paths = {label.raf_path for label in labels if label.raf_path}
    if len(raf_paths) > 1:
        path_list = ", ".join(str(path) for path in sorted(raf_paths))
        issues.append(
            ValidationIssue(
                "warning",
                capture_id,
                f"Capture '{capture_id}' references multiple RAF paths: {path_list}",
            )
        )
    return issues


def _validate_intervals(
    capture_id: str,
    labels: Sequence[LabelledMicrosector],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for label in labels:
        for operator_id, raw_intervals in label.operator_intervals.items():
            intervals = list(raw_intervals)
            if not intervals:
                continue
            resolved = _normalise_interval_bounds(
                intervals,
                default_start=0.0,
                default_end=1.0,
            )
            if len(resolved) < len(intervals):
                issues.append(
                    ValidationIssue(
                        "error",
                        capture_id,
                        (
                            f"Microsector {label.microsector_index} operator {operator_id} "
                            "defines invalid intervals that collapse after normalisation."
                        ),
                    )
                )
                # Skip overlap detection when intervals are already invalid.
                continue
            previous_end: float | None = None
            for start, end in resolved:
                if previous_end is not None and start < previous_end - 1e-9:
                    issues.append(
                        ValidationIssue(
                            "error",
                            capture_id,
                            (
                                f"Microsector {label.microsector_index} operator {operator_id} "
                                "defines overlapping intervals."
                            ),
                        )
                    )
                    break
                previous_end = end
    return issues


def inspect_labels(
    labels: Sequence[LabelledMicrosector],
) -> tuple[list[CaptureReport], list[ValidationIssue]]:
    grouped: dict[str, list[LabelledMicrosector]] = defaultdict(list)
    for label in labels:
        grouped[label.capture_id].append(label)

    reports: list[CaptureReport] = []
    issues: list[ValidationIssue] = []

    for capture_id in sorted(grouped):
        capture_labels = grouped[capture_id]
        indices = [label.microsector_index for label in capture_labels]
        overlaps = _detect_overlaps(indices)
        if overlaps:
            detail = ", ".join(
                f"{index} ({count} entries)" for index, count in sorted(overlaps.items())
            )
            issues.append(
                ValidationIssue(
                    "error",
                    capture_id,
                    f"Capture '{capture_id}' has duplicate microsector indices: {detail}",
                )
            )
        unique_indices = sorted(set(indices))
        gaps = _detect_gaps(unique_indices)
        if gaps:
            gap_desc = ", ".join(_format_gap(start, end) for start, end in gaps)
            issues.append(
                ValidationIssue(
                    "warning",
                    capture_id,
                    f"Capture '{capture_id}' is missing microsector indices: {gap_desc}",
                )
            )
        metadata_issues = _check_metadata(capture_id, capture_labels)
        issues.extend(metadata_issues)
        interval_issues = _validate_intervals(capture_id, capture_labels)
        issues.extend(interval_issues)
        raf_paths = {label.raf_path for label in capture_labels if label.raf_path}
        raf_path = next(iter(sorted(raf_paths))) if raf_paths else None
        reports.append(
            CaptureReport(
                capture_id=capture_id,
                raf_path=raf_path,
                label_count=len(capture_labels),
                coverage=_compute_lap_coverage(unique_indices),
                overlaps=overlaps,
                gaps=gaps,
            )
        )

    return reports, issues


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate labelled microsector artefacts for calibration",
    )
    parser.add_argument("labels", help="Path to the label artefact (JSON, TOML, YAML, CSV, JSONL)")
    parser.add_argument(
        "--raf-root",
        default=".",
        help="Base directory used to resolve RAF references inside the artefact",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit debug logs for troubleshooting",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    labels_path = Path(args.labels).expanduser().resolve()
    raf_root = Path(args.raf_root).expanduser().resolve()

    LOGGER.info("Loading labelled microsectors from %s", labels_path)
    try:
        labels = _load_labels(labels_path, raf_root=raf_root)
    except Exception as exc:  # pragma: no cover - surfacing loader errors
        LOGGER.error("Failed to load labels: %s", exc)
        return 1

    if not labels:
        LOGGER.warning("No labelled microsectors were found in the artefact.")
        return 0

    reports, issues = inspect_labels(labels)

    for report in reports:
        header = f"Capture {report.capture_id}"
        if report.raf_path:
            header += f" ({report.raf_path})"
        print(header)
        print(f"  labelled microsectors: {report.label_count}")
        if report.coverage:
            print("  coverage segments:")
            for idx, segment in enumerate(report.coverage, start=1):
                if segment.start_index == segment.end_index:
                    span = str(segment.start_index)
                else:
                    span = f"{segment.start_index}-{segment.end_index}"
                print(
                    f"    - Lap {idx}: indices {span} ({segment.count} microsectors)"
                )
        else:
            print("  coverage segments: none")
        if report.gaps:
            gap_desc = ", ".join(_format_gap(start, end) for start, end in report.gaps)
            print(f"  gaps: {gap_desc}")
        else:
            print("  gaps: none")
        if report.overlaps:
            overlap_desc = ", ".join(
                f"{index} ({count} entries)" for index, count in sorted(report.overlaps.items())
            )
            print(f"  overlaps: {overlap_desc}")
        else:
            print("  overlaps: none")

    exit_code = 0
    for issue in issues:
        if issue.severity == "error":
            exit_code = 1
            LOGGER.error(issue.message)
        else:
            LOGGER.warning(issue.message)

    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
