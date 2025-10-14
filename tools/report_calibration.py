"""Generate a calibration acceptance report from detector sweep artefacts.

The command line interface provided by this module aggregates the calibration
results produced by :mod:`tools.calibrate_detectors`.  It consumes the
``best_params.yaml`` summary, the ``metrics/best_selection.csv`` export (or
derives the relevant metrics from the confusion CSVs when the consolidated file
is unavailable) and optional acceptance thresholds.  The collected data is
collapsed across three levels – operator, operator/car and
operator/car/compound – and evaluated against the acceptance criteria defined
by the thresholds.  The final outcome is written as a Markdown document that is
suitable for hand-off to stakeholders.

Command line usage::

    python -m tools.report_calibration \
        --best-params calibration/summary/best_params.yaml \
        --metrics calibration/metrics \
        --thresholds calibration/summary/thresholds.yaml \
        --output calibration/summary/calibration_report.md

Arguments
---------
``--best-params``
    Path to the YAML document describing the selected parameters per
    operator/car/compound combination.
``--metrics``
    Directory containing CSV files with at least ``operator``, ``car`` and
    ``compound`` identifiers as well as ``f1`` and ``fp_per_min`` metrics.
``--thresholds``
    Optional YAML file describing the acceptance thresholds.  When omitted the
    defaults of ``F1 >= 0.72`` and ``FP/min <= 0.8`` are used.
``--output``
    Destination path for the generated Markdown report.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Constraints applied to the aggregated metrics."""

    f1_min: float = 0.72
    fp_per_min_max: float = 0.8

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "AcceptanceThresholds":
        if not data:
            return cls()
        normalised: MutableMapping[str, float] = {}
        for key, value in data.items():
            normalised_key = key.strip().lower().replace("-", "_").replace(" ", "_")
            if normalised_key in {"f1", "f1_min", "minimum_f1"}:
                normalised["f1_min"] = float(value)
            elif normalised_key in {"fp_per_min", "fp_per_min_max", "maximum_fp_per_min"}:
                normalised["fp_per_min_max"] = float(value)
        return cls(
            f1_min=normalised.get("f1_min", cls.f1_min),
            fp_per_min_max=normalised.get("fp_per_min_max", cls.fp_per_min_max),
        )

    @classmethod
    def from_yaml(cls, path: Path | None) -> "AcceptanceThresholds":
        if path is None:
            return cls()
        with path.open("r", encoding="utf8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, Mapping):  # pragma: no cover - defensive branch
            raise TypeError("Thresholds YAML must define a mapping")
        return cls.from_mapping(payload)

    def describe(self) -> str:
        return (
            f"F1 ≥ {self.f1_min:.2f} and false positives per minute ≤ {self.fp_per_min_max:.2f}"
        )


def load_best_params(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("best_params.yaml must contain a mapping")
    return payload


def _normalise_columns(columns: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in columns:
        normalised = column.strip().lower().replace("/", "_").replace(" ", "_")
        if normalised in {"operator", "operator_id"}:
            mapping[column] = "operator"
        elif normalised in {"car", "car_model"}:
            mapping[column] = "car"
        elif normalised in {"compound", "tyre_compound", "tyre"}:
            mapping[column] = "compound"
        elif normalised in {"f1", "f1_score"}:
            mapping[column] = "f1"
        elif normalised in {
            "fp_per_min",
            "false_positives_per_minute",
            "fp_min",
            "fp_per_minute",
        }:
            mapping[column] = "fp_per_min"
    return mapping


def _load_best_selection(metrics_dir: Path) -> pd.DataFrame | None:
    best_path = metrics_dir / "best_selection.csv"
    if not best_path.exists():
        return None
    frame = pd.read_csv(best_path)
    mapping = _normalise_columns(frame.columns)
    frame = frame.rename(columns=mapping)
    required = ["operator", "car", "compound", "f1", "fp_per_min"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(
            "best_selection.csv is missing required columns: " + ", ".join(sorted(missing))
        )
    result = frame[required].copy()
    result["f1"] = result["f1"].astype(float)
    result["fp_per_min"] = result["fp_per_min"].astype(float)
    return result


def _load_confusion_metrics(metrics_dir: Path) -> pd.DataFrame | None:
    if metrics_dir.is_file():
        csv_paths = [metrics_dir]
    else:
        confusion_dir = metrics_dir / "confusion"
        candidates = []
        if metrics_dir.is_dir():
            candidates.extend(sorted(metrics_dir.glob("*_confusion.csv")))
            candidates.extend(sorted(metrics_dir.glob("*.csv")))
        if confusion_dir.is_dir():
            candidates.extend(sorted(confusion_dir.glob("*.csv")))
        # Preserve order but remove duplicates
        seen: set[Path] = set()
        csv_paths = []
        for path in candidates:
            if path.suffix.lower() != ".csv":
                continue
            if path in seen:
                continue
            seen.add(path)
            csv_paths.append(path)
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - I/O safeguard
            LOGGER.warning("Failed to read metrics CSV %s: %s", csv_path, exc)
            continue
        mapping = _normalise_columns(frame.columns)
        frame = frame.rename(columns=mapping)
        expected = {"operator", "car", "compound", "tp", "fp", "fn", "duration_minutes"}
        if not expected.issubset(frame.columns):
            LOGGER.debug("Skipping %s due to missing confusion columns", csv_path)
            continue
        subset = frame[list(expected)].copy()
        for column in ["tp", "fp", "fn", "duration_minutes"]:
            subset[column] = subset[column].astype(float)
        # Compute metrics using the same formulae as in tools.calibrate_detectors
        denom_precision = subset["tp"] + subset["fp"]
        precision = subset["tp"] / denom_precision.where(denom_precision != 0, pd.NA)
        denom_recall = subset["tp"] + subset["fn"]
        recall = subset["tp"] / denom_recall.where(denom_recall != 0, pd.NA)
        f1 = 2 * precision * recall / (precision + recall)
        f1 = f1.fillna(0.0)
        duration = subset["duration_minutes"]
        fp_per_min = subset["fp"] / duration.where(duration > 0, pd.NA)
        fallback = subset["fp"].apply(lambda value: float("inf") if value > 0 else 0.0)
        fp_per_min = fp_per_min.fillna(fallback)
        frames.append(
            pd.DataFrame(
                {
                    "operator": subset["operator"],
                    "car": subset["car"],
                    "compound": subset["compound"],
                    "f1": f1.astype(float),
                    "fp_per_min": fp_per_min.astype(float),
                }
            )
        )
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics path does not exist: {metrics_dir}")

    best_frame = _load_best_selection(metrics_dir)
    if best_frame is not None:
        return best_frame

    confusion_frame = _load_confusion_metrics(metrics_dir)
    if confusion_frame is not None:
        return confusion_frame

    raise FileNotFoundError(
        "No usable metrics found – supply metrics/best_selection.csv or confusion CSV exports"
    )


def aggregate_metrics(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    aggregations = {
        "operator": ["operator"],
        "operator_car": ["operator", "car"],
        "operator_car_compound": ["operator", "car", "compound"],
    }
    grouped: dict[str, pd.DataFrame] = {}
    for key, columns in aggregations.items():
        grouped_frame = (
            frame.groupby(columns)[["f1", "fp_per_min"]]
            .mean()
            .reset_index()
            .sort_values(columns)
        )
        grouped[key] = grouped_frame
    return grouped


def evaluate_acceptance(frame: pd.DataFrame, thresholds: AcceptanceThresholds) -> pd.DataFrame:
    evaluated = frame.copy()
    evaluated["f1_pass"] = evaluated["f1"] >= thresholds.f1_min
    evaluated["fp_per_min_pass"] = evaluated["fp_per_min"] <= thresholds.fp_per_min_max
    evaluated["accepted"] = evaluated["f1_pass"] & evaluated["fp_per_min_pass"]
    return evaluated


def _format_best_params(best_params: Mapping[str, object]) -> str:
    lines: list[str] = []
    for operator, payload in sorted(best_params.items()):
        if isinstance(payload, Mapping):
            lines.append(f"- **{operator}**")
            for car, car_payload in sorted(payload.items()):
                prefix = "    - "
                if isinstance(car_payload, Mapping):
                    lines.append(f"{prefix}**{car}**")
                    for compound, params in sorted(car_payload.items()):
                        formatted = yaml.safe_dump(params, sort_keys=True).strip()
                        indented = textwrap.indent(formatted, "        ")
                        lines.append(f"        - *{compound}*\n{indented}")
                else:
                    formatted = yaml.safe_dump(car_payload, sort_keys=True).strip()
                    indented = textwrap.indent(formatted, "      ")
                    lines.append(f"    - {car}\n{indented}")
        else:
            formatted = yaml.safe_dump(payload, sort_keys=True).strip()
            indented = textwrap.indent(formatted, "  ")
            lines.append(f"- **{operator}**\n{indented}")
    return "\n".join(lines) if lines else "_No calibration parameters were found._"


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(no data)_"
    try:
        return frame.to_markdown(index=False, floatfmt=".3f")
    except ImportError:  # pragma: no cover - fallback when tabulate is missing
        # ``pandas.DataFrame.to_markdown`` depends on ``tabulate`` which may not be
        # available in lightweight environments (for example during CI runs).
        # Provide a simple pipe-delimited table as a graceful fallback.
        header = " | ".join(frame.columns)
        separator = " | ".join(["---"] * len(frame.columns))
        rows = []
        for row in frame.itertuples(index=False):
            cells = []
            for value in row:
                if isinstance(value, Real):
                    cells.append(f"{float(value):.3f}")
                else:
                    cells.append(str(value))
            rows.append(" | ".join(cells))
        return "\n".join([header, separator, *rows])


def render_report(
    best_params: Mapping[str, object],
    aggregated: Mapping[str, pd.DataFrame],
    thresholds: AcceptanceThresholds,
) -> str:
    sections = ["# Detector calibration summary", ""]
    sections.append("## Acceptance thresholds")
    sections.append(f"- {thresholds.describe()}")
    sections.append("")

    sections.append("## Best parameter selection")
    sections.append(_format_best_params(best_params))
    sections.append("")

    for key, title in (
        ("operator", "Operator overview"),
        ("operator_car", "Operator by car"),
        ("operator_car_compound", "Operator by car and compound"),
    ):
        sections.append(f"## {title}")
        frame = aggregated.get(key, pd.DataFrame())
        if not frame.empty:
            evaluated = evaluate_acceptance(frame, thresholds)
            sections.append(_frame_to_markdown(evaluated))
        else:
            sections.append("_(no data)_")
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best-params",
        type=Path,
        default=Path("calibration/summary/best_params.yaml"),
        help="Path to the best_params.yaml artefact (default: calibration/summary/best_params.yaml)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("calibration"),
        help="Directory containing the per-combination metrics CSV files",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="Optional YAML file describing the acceptance thresholds",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration/summary/calibration_report.md"),
        help="Destination path for the Markdown report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    best_params = load_best_params(args.best_params)
    thresholds = AcceptanceThresholds.from_yaml(args.thresholds)
    metrics_frame = load_metrics(args.metrics)
    aggregated = aggregate_metrics(metrics_frame)
    report = render_report(best_params, aggregated, thresholds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf8")
    LOGGER.info("Calibration report written to %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
