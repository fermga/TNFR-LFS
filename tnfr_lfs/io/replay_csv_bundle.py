"""Utilities for reading CSV bundles exported from Live for Speed replays."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
import math
import zipfile

from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.epi import TelemetryRecord

_KMH_TO_MS = 1000.0 / 3600.0
_G_TO_MS2 = 9.80665

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only.
    import pandas as pd

_PANDAS: Any | None = None


def _get_pandas() -> Any:
    global _PANDAS
    if _PANDAS is None:
        import pandas as _pd

        _PANDAS = _pd
    return _PANDAS


def _is_finite(value: float) -> bool:
    """Return ``True`` if ``value`` is a finite float."""

    return math.isfinite(value)


def _mean(values: Iterator[float]) -> float:
    finite = [value for value in values if _is_finite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def _sum(values: Iterator[float]) -> float:
    finite = [value for value in values if _is_finite(value)]
    if not finite:
        return math.nan
    return sum(finite)


_WHEEL_SUFFIXES: Mapping[str, str] = {
    "fl": "fl",
    "lf": "fl",
    "frontleft": "fl",
    "fr": "fr",
    "rf": "fr",
    "frontright": "fr",
    "rl": "rl",
    "lr": "rl",
    "rearleft": "rl",
    "rr": "rr",
    "rearright": "rr",
}


_BASE_ALIASES: Mapping[str, str] = {
    "time": "timestamp",
    "timestamp": "timestamp",
    "speed": "speed_kmh",
    "latg": "lateral_accel_g",
    "longg": "longitudinal_accel_g",
    "totalg": "total_g",
    "driftangle": "drift_angle_deg",
    "steer": "steer_input",
    "steerinput": "steer_input",
    "throttle": "throttle_input",
    "throtle": "throttle_input",
    "brake": "brake_input",
    "hb": "handbrake_input",
    "clutch": "clutch_input",
    "rpm": "rpm",
    "gear": "gear",
    "elevation": "elevation",
    "latwheelforce": "wheel_lateral_force",
    "latwf": "wheel_lateral_force",
    "longwheelforce": "wheel_longitudinal_force",
    "longwf": "wheel_longitudinal_force",
    "tyreload": "wheel_load",
    "suspensiontravelremaining": "suspension_deflection",
    "verticalwheelspeed": "suspension_velocity",
    "slipratio": "slip_ratio",
    "whang": "wheel_angle",
    "rotwhsp": "wheel_rotational_speed",
    "power": "wheel_power",
    "camber": "camber",
}


def _normalise_signal_name(name: str) -> str:
    stem = name.lower().replace(".csv", "").strip()
    stem = stem.replace(" ", "").replace("-", "").replace("_", "")
    for suffix, canonical in _WHEEL_SUFFIXES.items():
        if stem.endswith("f" + suffix):
            base = stem[: -(len(suffix) + 1)]
            break
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            break
    else:
        base = stem
        canonical = ""

    base = base.replace("forcef", "force")
    base = base.replace("wheelwheel", "wheel")
    base = base.replace("wheelrotwh", "rotwh")

    alias = _BASE_ALIASES.get(base, base)
    if canonical:
        return f"{alias}_{canonical}"
    return alias


def _extract_value_column(frame: Any) -> str:
    candidates = [column for column in frame.columns if column.lower() != "d"]
    if len(candidates) != 1:
        raise ValueError("Expected a single data column in CSV bundle entry")
    return candidates[0]


def _to_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(numeric):
        return math.nan
    return numeric


@dataclass(slots=True)
class ReplayCSVBundleReader:
    """Reader for CSV bundles exported by the LFS replay analyser."""

    source: Path | str
    _path: Path = field(init=False)
    _frame_cache: Any | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if isinstance(self.source, Path):
            self._path = self.source
        else:
            self._path = Path(self.source)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the bundle contents as a merged :class:`~pandas.DataFrame`."""

        if self._frame_cache is not None:
            return self._frame_cache.copy()

        pd = _get_pandas()

        frames: list[Any] = []
        timestamp_present = False
        for name, frame in self._iter_entries():
            if "d" not in frame.columns:
                raise ValueError(f"Bundle entry {name!r} does not contain a distance column")
            value_column = _extract_value_column(frame)
            signal_name = _normalise_signal_name(name)
            if signal_name == "timestamp":
                timestamp_present = True
            cleaned = frame.rename(columns={"d": "distance", value_column: signal_name})[
                ["distance", signal_name]
            ]
            frames.append(cleaned.set_index("distance"))

        if not frames:
            raise ValueError("Bundle does not contain any CSV telemetry signals")

        if not timestamp_present:
            raise ValueError(
                f"Replay CSV bundle {self._path} must contain a time.csv entry"
            )

        merged = pd.concat(frames, axis=1).sort_index().reset_index()
        merged.rename(columns={"index": "distance"}, inplace=True)

        def _coerce_numeric_columns(columns: Sequence[str]) -> None:
            existing = [column for column in columns if column in merged.columns]
            if not existing:
                return

            data = merged[existing].to_numpy(copy=True)
            numeric = pd.to_numeric(data.reshape(-1), errors="coerce")
            values = np.asarray(numeric).reshape(data.shape)
            values[~np.isfinite(values)] = np.nan

            coerced = pd.DataFrame(values, columns=existing, index=merged.index)
            merged[existing] = coerced

        _coerce_numeric_columns(["timestamp"])

        if "speed_kmh" in merged.columns:
            _coerce_numeric_columns(["speed_kmh"])
            merged["speed"] = merged["speed_kmh"] * _KMH_TO_MS

        _coerce_numeric_columns(["lateral_accel_g", "longitudinal_accel_g", "drift_angle_deg"])
        _coerce_numeric_columns(["distance"])

        telemetry_columns = {
            column
            for column in merged.columns
            if column.startswith("wheel_")
            or column.startswith("suspension_")
            or column.endswith(("_input", "_force", "_load", "_ratio"))
        }
        _coerce_numeric_columns(sorted(telemetry_columns))

        self._frame_cache = merged
        return merged.copy()

    def to_records(self) -> list[TelemetryRecord]:
        """Convert the bundle contents into :class:`TelemetryRecord` samples."""

        frame = self.to_dataframe()
        column_index = {name: position for position, name in enumerate(frame.columns)}

        return [
            self._row_to_record(row, column_index)
            for row in frame.itertuples(index=False, name=None)
        ]

    def _iter_entries(self) -> Iterator[tuple[str, pd.DataFrame]]:
        pd = _get_pandas()
        path = self._path
        if path.is_dir():
            for csv_path in sorted(path.glob("*.csv")):
                yield csv_path.stem, pd.read_csv(csv_path)
            return

        if path.is_file() and path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for info in sorted(archive.infolist(), key=lambda item: item.filename):
                    if info.is_dir() or not info.filename.lower().endswith(".csv"):
                        continue
                    with archive.open(info) as handle:
                        yield Path(info.filename).stem, pd.read_csv(handle)
            return

        raise ValueError(f"Unsupported bundle path: {path}")

    def _row_to_record(
        self, row: Sequence[object], column_index: Mapping[str, int]
    ) -> TelemetryRecord:
        def _has_column(key: str) -> bool:
            return key in column_index

        def _get_value(key: str) -> object:
            position = column_index.get(key)
            if position is None:
                return None
            return row[position]

        def _get_float(key: str) -> float:
            return _to_float(_get_value(key))

        def _sum_columns(columns: tuple[str, ...]) -> float:
            return _sum(_get_float(column) for column in columns if _has_column(column))

        def _mean_columns(columns: tuple[str, ...]) -> float:
            return _mean(
                _get_float(column) for column in columns if _has_column(column)
            )

        lateral_g = _get_float("lateral_accel_g")
        longitudinal_g = _get_float("longitudinal_accel_g")
        slip_angle = _get_float("drift_angle_deg")
        wheel_load_columns = (
            "wheel_load_fl",
            "wheel_load_fr",
            "wheel_load_rl",
            "wheel_load_rr",
        )

        vertical_load_front = _sum_columns(("wheel_load_fl", "wheel_load_fr"))
        vertical_load_rear = _sum_columns(("wheel_load_rl", "wheel_load_rr"))

        slip_ratio = _mean_columns(
            (
                "slip_ratio_fl",
                "slip_ratio_fr",
                "slip_ratio_rl",
                "slip_ratio_rr",
            )
        )

        suspension_travel_front = _mean_columns(
            ("suspension_deflection_fl", "suspension_deflection_fr")
        )
        suspension_travel_rear = _mean_columns(
            ("suspension_deflection_rl", "suspension_deflection_rr")
        )
        suspension_velocity_front = _mean_columns(
            ("suspension_velocity_fl", "suspension_velocity_fr")
        )
        suspension_velocity_rear = _mean_columns(
            ("suspension_velocity_rl", "suspension_velocity_rr")
        )

        vertical_load_total = _sum_columns(wheel_load_columns)

        gear_value = _get_float("gear")

        record = TelemetryRecord(
            timestamp=_get_float("timestamp"),
            vertical_load=vertical_load_total,
            slip_ratio=slip_ratio,
            lateral_accel=lateral_g * _G_TO_MS2 if _is_finite(lateral_g) else math.nan,
            longitudinal_accel=longitudinal_g * _G_TO_MS2 if _is_finite(longitudinal_g) else math.nan,
            yaw=math.nan,
            pitch=math.nan,
            roll=math.nan,
            brake_pressure=math.nan,
            locking=math.nan,
            nfr=math.nan,
            si=math.nan,
            speed=_get_float("speed"),
            yaw_rate=math.nan,
            slip_angle=math.radians(slip_angle) if _is_finite(slip_angle) else math.nan,
            steer=_get_float("steer_input"),
            throttle=_get_float("throttle_input"),
            gear=int(round(gear_value)) if _is_finite(gear_value) else 0,
            vertical_load_front=vertical_load_front,
            vertical_load_rear=vertical_load_rear,
            mu_eff_front=math.nan,
            mu_eff_rear=math.nan,
            mu_eff_front_lateral=math.nan,
            mu_eff_front_longitudinal=math.nan,
            mu_eff_rear_lateral=math.nan,
            mu_eff_rear_longitudinal=math.nan,
            suspension_travel_front=suspension_travel_front,
            suspension_travel_rear=suspension_travel_rear,
            suspension_velocity_front=suspension_velocity_front,
            suspension_velocity_rear=suspension_velocity_rear,
        )

        optional_fields: dict[str, float] = {}
        for column in (
            "brake_input",
            "clutch_input",
            "handbrake_input",
            "steer_input",
            "throttle_input",
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
            "suspension_deflection_fl",
            "suspension_deflection_fr",
            "suspension_deflection_rl",
            "suspension_deflection_rr",
        ):
            if _has_column(column):
                optional_fields[column] = _get_float(column)

        slip_ratio_columns = {
            "slip_ratio_fl",
            "slip_ratio_fr",
            "slip_ratio_rl",
            "slip_ratio_rr",
        }
        for column in slip_ratio_columns:
            if _has_column(column):
                optional_fields[column] = _get_float(column)

        return replace(record, **optional_fields)
