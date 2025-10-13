from __future__ import annotations

from collections.abc import Callable
import math
import time
import importlib
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tnfr_lfs.telemetry.offline import ReplayCSVBundleReader
from tnfr_lfs.resources import data_root
from tests.helpers import (
    is_numeric_series,
    is_numeric_value,
    monkeypatch_row_to_record_counter,
    read_reference_rows,
)


replay_csv_bundle = importlib.import_module(ReplayCSVBundleReader.__module__)


DATA_DIR = data_root()
BUNDLE_PATH = DATA_DIR / "test1.zip"
REFERENCE_CSV_DIR = DATA_DIR / "csv"


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
        assert is_numeric_series(frame[column])


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
        reference_rows = read_reference_rows(REFERENCE_CSV_DIR / csv_name)
        actual_rows = list(
            frame[["distance", column]].head(len(reference_rows)).itertuples(index=False, name=None)
        )
        for (actual_distance, actual_value), (reference_distance, expected_value) in zip(
            actual_rows, reference_rows
        ):
            assert math.isclose(actual_distance, reference_distance, rel_tol=1e-9)
            assert math.isclose(actual_value, transform(expected_value), rel_tol=1e-6)


def test_replay_csv_dataframe_matches_legacy_coercion() -> None:
    def _build_legacy_frame() -> pd.DataFrame:
        legacy_reader = ReplayCSVBundleReader(BUNDLE_PATH)
        pd_module = replay_csv_bundle._get_pandas()
        frames: list[pd.DataFrame] = []
        timestamp_present = False
        for name, frame in legacy_reader._iter_entries():
            if "d" not in frame.columns:
                raise AssertionError("Legacy reconstruction expected a distance column")
            value_column = replay_csv_bundle._extract_value_column(frame)
            signal_name = replay_csv_bundle._normalise_signal_name(name)
            if signal_name == "timestamp":
                timestamp_present = True
            cleaned = frame.rename(columns={"d": "distance", value_column: signal_name})[
                ["distance", signal_name]
            ]
            frames.append(cleaned.set_index("distance"))

        if not frames or not timestamp_present:
            raise AssertionError("Legacy reconstruction requires telemetry and timestamp data")

        merged = pd_module.concat(frames, axis=1).sort_index().reset_index()
        merged.rename(columns={"index": "distance"}, inplace=True)

        def _legacy_coerce(columns: list[str]) -> None:
            existing = [column for column in columns if column in merged.columns]
            if not existing:
                return

            data = merged[existing].to_numpy(copy=True)
            numeric = pd_module.to_numeric(data.reshape(-1), errors="coerce")
            values = np.asarray(numeric).reshape(data.shape)
            values[~np.isfinite(values)] = np.nan
            coerced = pd_module.DataFrame(values, columns=existing, index=merged.index)
            merged[existing] = coerced

        _legacy_coerce(["timestamp"])

        if "speed_kmh" in merged.columns:
            _legacy_coerce(["speed_kmh"])
            merged["speed"] = merged["speed_kmh"] * replay_csv_bundle._KMH_TO_MS

        _legacy_coerce(["lateral_accel_g", "longitudinal_accel_g", "drift_angle_deg"])
        _legacy_coerce(["distance"])

        telemetry_columns = {
            column
            for column in merged.columns
            if column.startswith("wheel_")
            or column.startswith("suspension_")
            or column.endswith(("_input", "_force", "_load", "_ratio"))
        }
        _legacy_coerce(sorted(telemetry_columns))

        return merged

    legacy_start = time.perf_counter()
    legacy_frame = _build_legacy_frame()
    legacy_time = time.perf_counter() - legacy_start

    new_reader = ReplayCSVBundleReader(BUNDLE_PATH)
    new_start = time.perf_counter()
    new_frame = new_reader.to_dataframe()
    new_time = time.perf_counter() - new_start

    pd.testing.assert_frame_equal(new_frame, legacy_frame)

    if legacy_time > 0:
        # Allow for some noise while still ensuring the refactor is not slower.
        assert new_time <= legacy_time * 2.0

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
                assert is_numeric_value(value)


def test_replay_csv_to_records_reuses_cached_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = ReplayCSVBundleReader(BUNDLE_PATH)
    pd_module = replay_csv_bundle._get_pandas()

    copy_calls = 0
    original_copy = pd_module.DataFrame.copy

    def _counting_copy(self: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
        nonlocal copy_calls
        copy_calls += 1
        return original_copy(self, deep=deep)

    monkeypatch.setattr(pd_module.DataFrame, "copy", _counting_copy)

    reader.to_dataframe()
    initial_copy_calls = copy_calls

    reader.to_records()

    assert copy_calls == initial_copy_calls


def test_replay_csv_to_records_uses_cached_results(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = ReplayCSVBundleReader(BUNDLE_PATH)

    with monkeypatch_row_to_record_counter(monkeypatch) as counter:
        first_records = reader.to_records()
        first_call_count = counter.count

        second_records = reader.to_records()

    assert counter.count == first_call_count
    assert first_records == second_records
    assert first_records is not second_records


def test_replay_csv_to_records_disable_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = ReplayCSVBundleReader(BUNDLE_PATH, cache_size=0)

    with monkeypatch_row_to_record_counter(monkeypatch) as counter:
        reader.to_records()
        baseline_calls = counter.count

        reader.to_records()

        assert counter.count > baseline_calls


def test_replay_csv_clear_cache_refreshes_records(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = ReplayCSVBundleReader(BUNDLE_PATH)

    with monkeypatch_row_to_record_counter(monkeypatch) as counter:
        reader.to_records()
        baseline_calls = counter.count

        reader.to_records()
        assert counter.count == baseline_calls

        reader.clear_cache()
        reader.to_records()
        assert counter.count > baseline_calls

        mutation_calls = counter.count
        reader.to_records(copy=False)
        assert counter.count > mutation_calls


@pytest.mark.parametrize(
    ("bundle_kwargs", "expected_message"),
    (
        ({"missing": "time"}, "must contain a time.csv entry"),
        ({"with_distance": False}, "does not contain a distance column"),
    ),
)
def test_replay_csv_bundle_validation_errors(
    csv_bundle: Callable[..., Path], bundle_kwargs: dict[str, object], expected_message: str
) -> None:
    bundle = csv_bundle(**bundle_kwargs)

    reader = ReplayCSVBundleReader(bundle)
    with pytest.raises(ValueError, match=expected_message):
        reader.to_dataframe()


def test_streaming_statistics_ignore_non_finite_values() -> None:
    values = (value for value in [1.0, math.nan, float("inf"), -float("inf"), 3.0])
    assert replay_csv_bundle._mean(values) == pytest.approx(2.0)

    values_for_sum = (value for value in [1.0, math.nan, float("inf"), -float("inf"), 3.0])
    assert replay_csv_bundle._sum(values_for_sum) == pytest.approx(4.0)


def test_streaming_statistics_use_bounded_memory() -> None:
    count = 1_000_000
    expected_sum = sum(range(100)) * (count // 100)
    expected_mean = expected_sum / count

    def _measure_peak_bytes(func: Callable[[], float]) -> tuple[float, int]:
        tracemalloc.start()
        try:
            result = func()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return result, peak

    mean_result, mean_peak = _measure_peak_bytes(
        lambda: replay_csv_bundle._mean(float(i % 100) for i in range(count))
    )
    sum_result, sum_peak = _measure_peak_bytes(
        lambda: replay_csv_bundle._sum(float(i % 100) for i in range(count))
    )

    assert mean_result == pytest.approx(expected_mean)
    assert sum_result == pytest.approx(expected_sum)

    # A list of ``count`` float objects would require tens of megabytes; the
    # streaming implementations should stay comfortably below this range.
    limit_bytes = 5 * 1024 * 1024
    assert mean_peak < limit_bytes
    assert sum_peak < limit_bytes
