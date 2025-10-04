"""Simplified OutSim telemetry ingestion client.

The original TNFR project ingests data from an OutSim UDP stream that
encodes suspension loads, slip angles, and wheel data.  For the
purposes of this library we implement a light-weight client that reads
CSV-formatted telemetry into :class:`~tnfr_lfs.core.epi.TelemetryRecord`
instances.  The goal is to provide deterministic behaviour that can be
unit-tested while staying faithful to the OutSim schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, TextIO

from ..core.epi import TelemetryRecord


@dataclass
class TelemetrySchema:
    """Schema configuration for a telemetry dataset.

    Attributes
    ----------
    columns:
        Ordered list of expected column names.  The default schema maps
        closely to the OutSim telemetry stream.
    delimiter:
        Delimiter used when parsing the telemetry file.
    """

    columns: Sequence[str]
    delimiter: str = ","


DEFAULT_SCHEMA = TelemetrySchema(
    columns=(
        "timestamp",
        "structural_timestamp",
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
        "speed",
        "yaw_rate",
        "slip_angle",
        "steer",
        "throttle",
        "gear",
        "vertical_load_front",
        "vertical_load_rear",
        "mu_eff_front",
        "mu_eff_rear",
        "mu_eff_front_lateral",
        "mu_eff_front_longitudinal",
        "mu_eff_rear_lateral",
        "mu_eff_rear_longitudinal",
        "suspension_travel_front",
        "suspension_travel_rear",
        "suspension_velocity_front",
        "suspension_velocity_rear",
        "tyre_temp_fl",
        "tyre_temp_fr",
        "tyre_temp_rl",
        "tyre_temp_rr",
        "tyre_pressure_fl",
        "tyre_pressure_fr",
        "tyre_pressure_rl",
        "tyre_pressure_rr",
    ),
)

OPTIONAL_SCHEMA_COLUMNS = {
    "structural_timestamp",
    "tyre_temp_fl",
    "tyre_temp_fr",
    "tyre_temp_rl",
    "tyre_temp_rr",
    "tyre_pressure_fl",
    "tyre_pressure_fr",
    "tyre_pressure_rl",
    "tyre_pressure_rr",
}

LEGACY_COLUMNS = (
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

LEGACY_DEFAULTS = {
    "speed": 0.0,
    "yaw_rate": 0.0,
    "slip_angle": 0.0,
    "steer": 0.0,
    "throttle": 0.0,
    "gear": 0,
    "vertical_load_front": 0.0,
    "vertical_load_rear": 0.0,
    "mu_eff_front": 0.0,
    "mu_eff_rear": 0.0,
    "mu_eff_front_lateral": 0.0,
    "mu_eff_front_longitudinal": 0.0,
    "mu_eff_rear_lateral": 0.0,
    "mu_eff_rear_longitudinal": 0.0,
    "suspension_travel_front": 0.0,
    "suspension_travel_rear": 0.0,
    "suspension_velocity_front": 0.0,
    "suspension_velocity_rear": 0.0,
    "tyre_temp_fl": 0.0,
    "tyre_temp_fr": 0.0,
    "tyre_temp_rl": 0.0,
    "tyre_temp_rr": 0.0,
    "tyre_pressure_fl": 0.0,
    "tyre_pressure_fr": 0.0,
    "tyre_pressure_rl": 0.0,
    "tyre_pressure_rr": 0.0,
    "rpm": 0.0,
    "line_deviation": 0.0,
}


class TelemetryFormatError(ValueError):
    """Raised when the incoming telemetry cannot be parsed."""


class OutSimClient:
    """Client responsible for ingesting telemetry from different sources.

    The client accepts either iterables of strings (such as file handles
    or in-memory lists) or filesystem paths.  The parsing logic is kept
    intentionally small to make it easy to extend with real OutSim
    decoding if needed in the future.
    """

    def __init__(self, schema: TelemetrySchema | None = None) -> None:
        self.schema = schema or DEFAULT_SCHEMA

    def ingest(self, source: str | Path | TextIO | Iterable[str]) -> List[TelemetryRecord]:
        """Return a list of :class:`TelemetryRecord` objects.

        Parameters
        ----------
        source:
            Either a path to a CSV file or any iterable that yields lines
            of text.
        """

        iterator = self._open_source(source)
        header = next(iterator, None)
        if header is None:
            return []

        header_columns = [column.strip() for column in header.split(self.schema.delimiter)]
        is_legacy = tuple(header_columns) == LEGACY_COLUMNS
        lap_column_present = False
        value_columns: Sequence[str] = header_columns
        column_lookup: dict[str, int] = {}
        if not is_legacy:
            schema_columns = tuple(self.schema.columns)
            normalised_schema = tuple(column.lower() for column in schema_columns)
            normalised_header = tuple(column.lower() for column in header_columns)
            if normalised_header and normalised_header[-1] == "lap":
                lap_column_present = True
                header_core = normalised_header[:-1]
                value_columns = header_columns[:-1]
            else:
                header_core = normalised_header
                value_columns = header_columns
            header_without_optional = tuple(
                column for column in header_core if column not in OPTIONAL_SCHEMA_COLUMNS
            )
            schema_without_optional = tuple(
                column for column in normalised_schema if column not in OPTIONAL_SCHEMA_COLUMNS
            )
            if header_without_optional != schema_without_optional:
                raise TelemetryFormatError(
                    f"Unexpected header {header_columns!r}. Expected {self.schema.columns!r}"
                )
            column_lookup = {
                column.lower(): index for index, column in enumerate(value_columns)
            }
        else:
            value_columns = header_columns
            column_lookup = {column.lower(): index for index, column in enumerate(value_columns)}

        value_columns = tuple(value_columns)

        records: List[TelemetryRecord] = []
        for line in iterator:
            if not line.strip():
                continue
            values = [value.strip() for value in line.split(self.schema.delimiter)]
            expected_len = len(LEGACY_COLUMNS)
            if not is_legacy:
                expected_len = len(value_columns) + (1 if lap_column_present else 0)
            if len(values) != expected_len:
                raise TelemetryFormatError(
                    f"Expected {expected_len} columns, got {len(values)}: {values!r}"
                )
            try:
                lap_value = None
                if lap_column_present:
                    lap_token = values[-1].strip()
                    if lap_token:
                        try:
                            int_token = int(float(lap_token))
                        except ValueError:
                            lap_value = lap_token
                        else:
                            lap_value = int_token
                if is_legacy:
                    record = TelemetryRecord(
                        timestamp=float(values[0]),
                        vertical_load=float(values[1]),
                        slip_ratio=float(values[2]),
                        lateral_accel=float(values[3]),
                        longitudinal_accel=float(values[4]),
                        yaw=float(values[5]),
                        pitch=float(values[6]),
                        roll=float(values[7]),
                        brake_pressure=float(values[8]),
                        locking=float(values[9]),
                        nfr=float(values[10]),
                        si=float(values[11]),
                        speed=LEGACY_DEFAULTS["speed"],
                        yaw_rate=LEGACY_DEFAULTS["yaw_rate"],
                        slip_angle=LEGACY_DEFAULTS["slip_angle"],
                        steer=LEGACY_DEFAULTS["steer"],
                        throttle=LEGACY_DEFAULTS["throttle"],
                        gear=LEGACY_DEFAULTS["gear"],
                        vertical_load_front=LEGACY_DEFAULTS["vertical_load_front"],
                        vertical_load_rear=LEGACY_DEFAULTS["vertical_load_rear"],
                        mu_eff_front=LEGACY_DEFAULTS["mu_eff_front"],
                        mu_eff_rear=LEGACY_DEFAULTS["mu_eff_rear"],
                        mu_eff_front_lateral=LEGACY_DEFAULTS["mu_eff_front_lateral"],
                        mu_eff_front_longitudinal=LEGACY_DEFAULTS[
                            "mu_eff_front_longitudinal"
                        ],
                        mu_eff_rear_lateral=LEGACY_DEFAULTS["mu_eff_rear_lateral"],
                        mu_eff_rear_longitudinal=LEGACY_DEFAULTS[
                            "mu_eff_rear_longitudinal"
                        ],
                        suspension_travel_front=LEGACY_DEFAULTS["suspension_travel_front"],
                        suspension_travel_rear=LEGACY_DEFAULTS["suspension_travel_rear"],
                        suspension_velocity_front=LEGACY_DEFAULTS["suspension_velocity_front"],
                        suspension_velocity_rear=LEGACY_DEFAULTS["suspension_velocity_rear"],
                        tyre_temp_fl=LEGACY_DEFAULTS["tyre_temp_fl"],
                        tyre_temp_fr=LEGACY_DEFAULTS["tyre_temp_fr"],
                        tyre_temp_rl=LEGACY_DEFAULTS["tyre_temp_rl"],
                        tyre_temp_rr=LEGACY_DEFAULTS["tyre_temp_rr"],
                        tyre_pressure_fl=LEGACY_DEFAULTS["tyre_pressure_fl"],
                        tyre_pressure_fr=LEGACY_DEFAULTS["tyre_pressure_fr"],
                        tyre_pressure_rl=LEGACY_DEFAULTS["tyre_pressure_rl"],
                        tyre_pressure_rr=LEGACY_DEFAULTS["tyre_pressure_rr"],
                        rpm=LEGACY_DEFAULTS["rpm"],
                        line_deviation=LEGACY_DEFAULTS["line_deviation"],
                        lap=None,
                    )
                else:
                    data_values = values[:-1] if lap_column_present else values
                    value_map = {
                        column: data_values[index]
                        for column, index in column_lookup.items()
                    }
                    structural_value = value_map.get("structural_timestamp")
                    record = TelemetryRecord(
                        timestamp=float(value_map["timestamp"]),
                        structural_timestamp=(
                            float(structural_value)
                            if structural_value not in (None, "")
                            else None
                        ),
                        vertical_load=float(value_map["vertical_load"]),
                        slip_ratio=float(value_map["slip_ratio"]),
                        lateral_accel=float(value_map["lateral_accel"]),
                        longitudinal_accel=float(value_map["longitudinal_accel"]),
                        yaw=float(value_map["yaw"]),
                        pitch=float(value_map["pitch"]),
                        roll=float(value_map["roll"]),
                        brake_pressure=float(value_map["brake_pressure"]),
                        locking=float(value_map["locking"]),
                        nfr=float(value_map["nfr"]),
                        si=float(value_map["si"]),
                        speed=float(value_map["speed"]),
                        yaw_rate=float(value_map["yaw_rate"]),
                        slip_angle=float(value_map["slip_angle"]),
                        steer=float(value_map["steer"]),
                        throttle=float(value_map["throttle"]),
                        gear=int(float(value_map["gear"])),
                        vertical_load_front=float(value_map["vertical_load_front"]),
                        vertical_load_rear=float(value_map["vertical_load_rear"]),
                        mu_eff_front=float(value_map["mu_eff_front"]),
                        mu_eff_rear=float(value_map["mu_eff_rear"]),
                        mu_eff_front_lateral=float(value_map["mu_eff_front_lateral"]),
                        mu_eff_front_longitudinal=float(value_map["mu_eff_front_longitudinal"]),
                        mu_eff_rear_lateral=float(value_map["mu_eff_rear_lateral"]),
                        mu_eff_rear_longitudinal=float(value_map["mu_eff_rear_longitudinal"]),
                        suspension_travel_front=float(value_map["suspension_travel_front"]),
                        suspension_travel_rear=float(value_map["suspension_travel_rear"]),
                        suspension_velocity_front=float(value_map["suspension_velocity_front"]),
                        suspension_velocity_rear=float(value_map["suspension_velocity_rear"]),
                        tyre_temp_fl=float(value_map.get("tyre_temp_fl", 0.0)),
                        tyre_temp_fr=float(value_map.get("tyre_temp_fr", 0.0)),
                        tyre_temp_rl=float(value_map.get("tyre_temp_rl", 0.0)),
                        tyre_temp_rr=float(value_map.get("tyre_temp_rr", 0.0)),
                        tyre_pressure_fl=float(value_map.get("tyre_pressure_fl", 0.0)),
                        tyre_pressure_fr=float(value_map.get("tyre_pressure_fr", 0.0)),
                        tyre_pressure_rl=float(value_map.get("tyre_pressure_rl", 0.0)),
                        tyre_pressure_rr=float(value_map.get("tyre_pressure_rr", 0.0)),
                        rpm=float(value_map.get("rpm", 0.0)),
                        line_deviation=float(value_map.get("line_deviation", 0.0)),
                        lap=lap_value,
                    )
            except ValueError as exc:  # pragma: no cover - defensive branch
                raise TelemetryFormatError(f"Cannot parse telemetry values: {values!r}") from exc
            records.append(record)
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_source(self, source: str | Path | TextIO | Iterable[str]) -> Iterator[str]:
        if isinstance(source, (str, Path)):
            with open(source, "r", encoding="utf8") as handle:
                yield from handle
            return
        if hasattr(source, "read"):
            yield from source  # type: ignore[misc]
            return
        yield from source
