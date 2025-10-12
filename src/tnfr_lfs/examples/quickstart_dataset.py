"""Helpers that expose the bundled quickstart dataset."""

from __future__ import annotations

from pathlib import Path

from tnfr_lfs.resources import data_root

_DATASET_NAME = "BL1_XFG_baseline.csv"


def dataset_path(root: Path | None = None) -> Path:
    """Return the path to the bundled quickstart dataset."""

    base = root if root is not None else data_root()
    dataset = base / _DATASET_NAME
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    return dataset


def dataset_columns() -> tuple[str, ...]:
    """Column names expected by the CLI quickstart pipeline."""

    return (
        "timestamp",
        "vertical_load",
        "slip_ratio",
        "lateral_accel",
        "longitudinal_accel",
        "yaw",
        "pitch",
        "roll",
        "brake_pressure",
        "locking",
        "nfr",
        "si",
    )


def dataset_sample_count(root: Path | None = None) -> int:
    """Number of telemetry samples available in the dataset."""

    dataset = dataset_path(root)
    lines = dataset.read_text(encoding="utf8").splitlines()
    return max(len(lines) - 1, 0)
