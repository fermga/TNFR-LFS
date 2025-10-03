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
        "suspension_travel_front",
        "suspension_travel_rear",
        "suspension_velocity_front",
        "suspension_velocity_rear",
    ),
)

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
    "suspension_travel_front": 0.0,
    "suspension_travel_rear": 0.0,
    "suspension_velocity_front": 0.0,
    "suspension_velocity_rear": 0.0,
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
        if not is_legacy and tuple(header_columns) != tuple(self.schema.columns):
            raise TelemetryFormatError(
                f"Unexpected header {header_columns!r}. Expected {self.schema.columns!r}"
            )

        records: List[TelemetryRecord] = []
        for line in iterator:
            if not line.strip():
                continue
            values = [value.strip() for value in line.split(self.schema.delimiter)]
            expected_len = len(LEGACY_COLUMNS) if is_legacy else len(self.schema.columns)
            if len(values) != expected_len:
                raise TelemetryFormatError(
                    f"Expected {expected_len} columns, got {len(values)}: {values!r}"
                )
            try:
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
                        suspension_travel_front=LEGACY_DEFAULTS["suspension_travel_front"],
                        suspension_travel_rear=LEGACY_DEFAULTS["suspension_travel_rear"],
                        suspension_velocity_front=LEGACY_DEFAULTS["suspension_velocity_front"],
                        suspension_velocity_rear=LEGACY_DEFAULTS["suspension_velocity_rear"],
                    )
                else:
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
                        speed=float(values[12]),
                        yaw_rate=float(values[13]),
                        slip_angle=float(values[14]),
                        steer=float(values[15]),
                        throttle=float(values[16]),
                        gear=int(float(values[17])),
                        vertical_load_front=float(values[18]),
                        vertical_load_rear=float(values[19]),
                        mu_eff_front=float(values[20]),
                        mu_eff_rear=float(values[21]),
                        suspension_travel_front=float(values[22]),
                        suspension_travel_rear=float(values[23]),
                        suspension_velocity_front=float(values[24]),
                        suspension_velocity_rear=float(values[25]),
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
