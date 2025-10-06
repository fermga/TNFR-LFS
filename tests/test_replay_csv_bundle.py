from __future__ import annotations

import csv
import math
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from tnfr_lfs.io import ReplayCSVBundleReader


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
BUNDLE_PATH = DATA_DIR / "test1.zip"
REFERENCE_CSV_DIR = DATA_DIR / "csv"


def _read_reference_rows(name: str, limit: int = 5) -> list[tuple[float, float]]:
    path = REFERENCE_CSV_DIR / name
    rows: list[tuple[float, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((float(row["d"]), float(row["test1"])))
            if len(rows) >= limit:
                break
    return rows


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def test_replay_csv_dataframe_contains_key_columns() -> None:
    reader = ReplayCSVBundleReader(BUNDLE_PATH)
    frame = reader.to_dataframe()

    required_columns = {
        "timestamp",
        "speed_kmh",
        "speed",
        "wheel_load_fl",
        "wheel_load_fr",
        "wheel_load_rl",
        "wheel_load_rr",
        "wheel_lateral_force_fl",
        "wheel_lateral_force_fr",
        "wheel_lateral_force_rl",
        "wheel_longitudinal_force_fl",
        "wheel_longitudinal_force_fr",
        "wheel_longitudinal_force_rl",
        "wheel_longitudinal_force_rr",
    }

    available_columns = set(frame.columns)
    assert required_columns <= available_columns

    # Some bundles contain a shortened ``latwfRR`` entry that should still be parsed.
    lateral_force_rr_column = (
        "wheel_lateral_force_rr"
        if "wheel_lateral_force_rr" in available_columns
        else "latw_rr"
    )
    assert lateral_force_rr_column in available_columns

    for column in required_columns | {lateral_force_rr_column}:
        assert _is_numeric_series(frame[column])


def test_replay_csv_aliases_normalised_with_expected_values() -> None:
    reader = ReplayCSVBundleReader(BUNDLE_PATH)
    frame = reader.to_dataframe()

    alias_expectations = {
        "timestamp": ("time.csv", lambda value: value),
        "speed_kmh": ("speed.csv", lambda value: value),
        "speed": ("speed.csv", lambda value: value * (1000.0 / 3600.0)),
        "steer_input": ("steer.csv", lambda value: value),
        "throttle_input": ("throtle.csv", lambda value: value),
        "wheel_lateral_force_rl": (
            "latwheelforcefRL.csv",
            lambda value: value,
        ),
        "wheel_longitudinal_force_rl": (
            "longwheelforcefRL.csv",
            lambda value: value,
        ),
    }

    for column, (csv_name, transform) in alias_expectations.items():
        assert column in frame.columns
        reference_rows = _read_reference_rows(csv_name)
        actual_rows = list(
            frame[["distance", column]].head(len(reference_rows)).itertuples(index=False, name=None)
        )
        for (actual_distance, actual_value), (reference_distance, expected_value) in zip(
            actual_rows, reference_rows
        ):
            assert math.isclose(actual_distance, reference_distance, rel_tol=1e-9)
            assert math.isclose(actual_value, transform(expected_value), rel_tol=1e-6)


def _is_numeric(value: object) -> bool:
    if isinstance(value, bool):  # Guard against bool being subclass of int.
        return False
    if isinstance(value, (int, float)):
        numeric = float(value)
        return math.isfinite(numeric) or math.isnan(numeric)
    return False


def test_replay_csv_records_align_with_dataframe() -> None:
    reader = ReplayCSVBundleReader(BUNDLE_PATH)
    frame = reader.to_dataframe()
    records = reader.to_records()

    assert len(records) == len(frame)

    required_fields = [
        "timestamp",
        "speed",
        "steer",
        "throttle",
        "gear",
        "rpm",
        "wheel_load_fl",
        "wheel_load_fr",
        "wheel_load_rl",
        "wheel_load_rr",
        "wheel_lateral_force_fl",
        "wheel_lateral_force_fr",
        "wheel_lateral_force_rl",
        "wheel_lateral_force_rr",
        "wheel_longitudinal_force_fl",
        "wheel_longitudinal_force_fr",
        "wheel_longitudinal_force_rl",
        "wheel_longitudinal_force_rr",
    ]

    for record in records:
        for field in required_fields:
            value = getattr(record, field)
            if field == "gear":
                assert isinstance(value, int)
            else:
                assert _is_numeric(value)


def test_replay_csv_bundle_without_time_entry_raises(tmp_path: Path) -> None:
    bundle = tmp_path / "missing_time.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("speed.csv", "d,value\n0,100\n")

    reader = ReplayCSVBundleReader(bundle)
    with pytest.raises(ValueError, match="must contain a time.csv entry"):
        reader.to_dataframe()


def test_replay_csv_bundle_missing_distance_column_raises(tmp_path: Path) -> None:
    bundle = tmp_path / "missing_distance.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("time.csv", "d,value\n0,0.0\n")
        archive.writestr("speed.csv", "distance,value\n0,100\n")

    reader = ReplayCSVBundleReader(bundle)
    with pytest.raises(ValueError, match="does not contain a distance column"):
        reader.to_dataframe()
