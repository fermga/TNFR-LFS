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
