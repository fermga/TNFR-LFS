"""Tests for the calibration reporting utility."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.report_calibration import (
    AcceptanceThresholds,
    aggregate_metrics,
    evaluate_acceptance,
    load_best_params,
    load_metrics,
    render_report,
)


def test_thresholds_from_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "thresholds.yaml"
    yaml_path.write_text("f1: 0.75\nfp_per_min: 0.5\n", encoding="utf8")

    thresholds = AcceptanceThresholds.from_yaml(yaml_path)

    assert thresholds.f1_min == 0.75
    assert thresholds.fp_per_min_max == 0.5


def test_evaluate_acceptance_flags() -> None:
    metrics = pd.DataFrame(
        [
            {"operator": "AL", "car": "XFG", "compound": "R1", "f1": 0.80, "fp_per_min": 0.50},
            {"operator": "AL", "car": "XFG", "compound": "R2", "f1": 0.70, "fp_per_min": 0.90},
        ]
    )
    aggregated = aggregate_metrics(metrics)
    thresholds = AcceptanceThresholds()

    evaluated = evaluate_acceptance(aggregated["operator_car_compound"], thresholds)

    assert bool(evaluated.loc[0, "accepted"])  # first combination satisfies the thresholds
    assert not bool(evaluated.loc[1, "accepted"])  # second combination violates the thresholds

    report = render_report({"AL": {"XFG": {"R1": {"window": 5}}}}, aggregated, thresholds)

    assert "Detector calibration summary" in report
    assert "Operator overview" in report


def test_load_best_params_requires_mapping(tmp_path: Path) -> None:
    yaml_path = tmp_path / "best_params.yaml"
    yaml_path.write_text("- not-a-mapping\n", encoding="utf8")

    with pytest.raises(TypeError):
        load_best_params(yaml_path)


def test_load_metrics_prefers_best_selection(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "best_selection.csv").write_text(
        "operator,car,compound,f1,fp_per_min\n"
        "AL,XFG,R1,0.81,0.42\n"
        "EN,XRG,R2,0.65,0.30\n",
        encoding="utf8",
    )
    # Additional CSVs should be ignored once best_selection is available
    (metrics_dir / "other.csv").write_text(
        "operator,car,compound,f1,fp_per_min\nAL,XFG,R1,0.10,5.0\n",
        encoding="utf8",
    )

    frame = load_metrics(metrics_dir)

    assert list(frame["operator"]) == ["AL", "EN"]
    assert frame.loc[0, "f1"] == pytest.approx(0.81)
    assert frame.loc[0, "fp_per_min"] == pytest.approx(0.42)


def test_load_metrics_falls_back_to_confusion(tmp_path: Path) -> None:
    confusion_dir = tmp_path / "confusion"
    confusion_dir.mkdir()
    (confusion_dir / "al_confusion.csv").write_text(
        "operator,car,compound,tp,fp,fn,duration_minutes\n"
        "AL,XFG,R1,4,1,1,2.5\n",
        encoding="utf8",
    )

    frame = load_metrics(tmp_path)

    assert list(frame["operator"]) == ["AL"]
    # Precision=4/5, Recall=4/5 -> F1=0.8
    assert frame.loc[0, "f1"] == pytest.approx(0.8)
    # FP/minute = 1 / 2.5 = 0.4
    assert frame.loc[0, "fp_per_min"] == pytest.approx(0.4)


def test_acceptance_table_uses_best_selection(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "best_selection.csv").write_text(
        "operator,car,compound,f1,fp_per_min\nAL,XFG,R1,0.82,0.30\n",
        encoding="utf8",
    )
    # Noise that would have skewed the aggregation prior to the regression fix
    (metrics_dir / "candidates.csv").write_text(
        "operator,car,compound,f1,fp_per_min\nAL,XFG,R1,0.10,5.0\n",
        encoding="utf8",
    )

    metrics_frame = load_metrics(metrics_dir)
    aggregated = aggregate_metrics(metrics_frame)
    thresholds = AcceptanceThresholds(f1_min=0.75, fp_per_min_max=0.5)

    evaluated = evaluate_acceptance(aggregated["operator_car_compound"], thresholds)

    assert len(evaluated) == 1
    assert evaluated.loc[0, "f1"] == pytest.approx(0.82)
    assert bool(evaluated.loc[0, "accepted"])  # Selected configuration passes the thresholds
