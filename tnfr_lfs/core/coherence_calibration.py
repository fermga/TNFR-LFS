"""Coherence calibration utilities for persistent ΔNFR baselines."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from math import isfinite
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence, Tuple, TypeAlias

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from .epi import TelemetryRecord

TelemetryBaselineValue: TypeAlias = float | int | str | None

__all__ = ["CoherenceCalibrationStore", "CalibrationSnapshot"]


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and isfinite(float(value))


def _record_to_numeric_map(record: TelemetryRecord) -> dict[str, float]:
    payload: dict[str, float] = {}
    for field_name, value in record.__dict__.items():
        if _is_numeric(value):
            payload[field_name] = float(value)
    return payload


_TELEMETRY_FIELDS = {field.name: field for field in fields(TelemetryRecord) if field.init}


def _coerce_field_value(name: str, value: float) -> TelemetryBaselineValue:
    field = _TELEMETRY_FIELDS.get(name)
    if field is None:
        return value
    annotation = field.type
    if annotation is int:
        return int(round(value))
    return float(value)


def _default_template() -> dict[str, TelemetryBaselineValue]:
    template: dict[str, TelemetryBaselineValue] = {}
    for name, field in _TELEMETRY_FIELDS.items():
        if field.default is not MISSING:
            template[name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            template[name] = field.default_factory()  # type: ignore[misc]
        else:
            template[name] = 0.0
    return template


@dataclass
class CalibrationMetric:
    """Exponentially smoothed statistic for a telemetry attribute."""

    mean: float
    deviation: float = 0.0

    def update(self, value: float, decay: float) -> None:
        if not _is_numeric(value):
            return
        numeric = float(value)
        self.mean = (1.0 - decay) * self.mean + decay * numeric
        self.deviation = (1.0 - decay) * self.deviation + decay * abs(numeric - self.mean)

    @property
    def norm_range(self) -> Tuple[float, float]:
        width = max(0.0, self.deviation)
        return self.mean - width, self.mean + width


@dataclass
class CalibrationEntry:
    """Calibration data accumulated for a player/car combination."""

    player_name: str
    car_model: str
    laps: int = 0
    metrics: dict[str, CalibrationMetric] = field(default_factory=dict)
    template: dict[str, TelemetryBaselineValue] = field(default_factory=_default_template)

    def update(self, record: TelemetryRecord, decay: float) -> None:
        payload = _record_to_numeric_map(record)
        if not self.metrics:
            for name, value in payload.items():
                self.metrics[name] = CalibrationMetric(mean=float(value))
        else:
            for name, value in payload.items():
                metric = self.metrics.get(name)
                if metric is None:
                    self.metrics[name] = CalibrationMetric(mean=float(value))
                else:
                    metric.update(float(value), decay)
        updates: dict[str, TelemetryBaselineValue] = {
            name: payload[name] for name in payload
        }
        self.template.update(updates)

    def build_baseline(self) -> TelemetryRecord:
        data: dict[str, TelemetryBaselineValue] = dict(self.template)
        for name, metric in self.metrics.items():
            data[name] = _coerce_field_value(name, metric.mean)
        return TelemetryRecord(**data)

    def ranges(self) -> dict[str, Tuple[float, float]]:
        return {name: metric.norm_range for name, metric in self.metrics.items()}


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Read-only view of a calibration entry."""

    player_name: str
    car_model: str
    laps: int
    baseline: TelemetryRecord
    ranges: Mapping[str, Tuple[float, float]]


class CoherenceCalibrationStore:
    """Manage ΔNFR baseline calibrations for player/car combinations."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        decay: float = 0.2,
        min_laps: int = 10,
        max_laps: int = 20,
    ) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in the (0, 1] range")
        if min_laps <= 0:
            raise ValueError("min_laps must be positive")
        if max_laps < min_laps:
            raise ValueError("max_laps must be >= min_laps")
        self.path = Path(path or "coherence_calibration.toml")
        self.decay = float(decay)
        self.min_laps = int(min_laps)
        self.max_laps = int(max_laps)
        self._entries: dict[Tuple[str, str], CalibrationEntry] = {}
        self._load()

    def register_lap(self, player_name: str, car_model: str, records: Sequence[TelemetryRecord]) -> None:
        if not records:
            return
        from .epi import DeltaCalculator  # Local import to avoid circular dependency

        baseline = DeltaCalculator.derive_baseline(records)
        self.observe_baseline(player_name, car_model, baseline)

    def observe_baseline(self, player_name: str, car_model: str, baseline: TelemetryRecord) -> None:
        if not player_name or not car_model:
            return
        key = (player_name, car_model)
        entry = self._entries.get(key)
        if entry is None:
            entry = CalibrationEntry(player_name=player_name, car_model=car_model)
            self._entries[key] = entry
        entry.update(baseline, self.decay)
        entry.laps = min(self.max_laps, entry.laps + 1)

    def baseline_for(
        self,
        player_name: str,
        car_model: str,
        fallback: TelemetryRecord,
    ) -> TelemetryRecord:
        entry = self._entries.get((player_name, car_model))
        if entry is None or entry.laps < self.min_laps:
            return fallback
        return entry.build_baseline()

    def snapshot(self, player_name: str, car_model: str) -> CalibrationSnapshot | None:
        entry = self._entries.get((player_name, car_model))
        if entry is None:
            return None
        return CalibrationSnapshot(
            player_name=player_name,
            car_model=car_model,
            laps=entry.laps,
            baseline=entry.build_baseline(),
            ranges=entry.ranges(),
        )

    def save(self) -> None:
        if not self._entries:
            if self.path.exists():
                self.path.unlink()
            return
        root: dict[str, MutableMapping[str, object]] = {}
        for (player, car), entry in self._entries.items():
            player_node = root.setdefault(str(player), {})
            car_node: MutableMapping[str, object] = {
                "laps": entry.laps,
                "metrics": {
                    name: {"mean": metric.mean, "deviation": metric.deviation}
                    for name, metric in entry.metrics.items()
                },
            }
            player_node[str(car)] = car_node
        lines: list[str] = []
        for player_key in sorted(root):
            player = str(player_key)
            lines.append(f'[players."{player}"]')
            cars = root[player_key]
            for car_key in sorted(cars):
                car = str(car_key)
                spec = cars[car_key]
                lines.append(f'[players."{player}"."{car}"]')
                lines.append(f'laps = {int(spec.get("laps", 0))}')
                metrics = spec.get("metrics", {})
                if isinstance(metrics, Mapping):
                    for name_key in sorted(metrics):
                        metric = metrics[name_key]
                        if not isinstance(metric, Mapping):
                            continue
                        mean = metric.get("mean")
                        if not isinstance(mean, (int, float)):
                            continue
                        deviation = metric.get("deviation", 0.0)
                        deviation_value = (
                            float(deviation) if isinstance(deviation, (int, float)) else 0.0
                        )
                        metric_name = str(name_key)
                        lines.append(f'[players."{player}"."{car}".metrics."{metric_name}"]')
                        lines.append(f'mean = {float(mean)}')
                        lines.append(f'deviation = {deviation_value}')
                lines.append("")
            if lines and lines[-1] != "":
                lines.append("")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            payload = tomllib.load(handle)
        root = payload.get("players", payload)
        if not isinstance(root, Mapping):
            return
        for player, cars in root.items():
            if not isinstance(cars, Mapping):
                continue
            for car, spec in cars.items():
                if not isinstance(spec, Mapping):
                    continue
                entry = CalibrationEntry(player_name=str(player), car_model=str(car))
                laps = spec.get("laps")
                if isinstance(laps, int):
                    entry.laps = max(0, int(laps))
                metrics = spec.get("metrics")
                if isinstance(metrics, Mapping):
                    for name, values in metrics.items():
                        if not isinstance(values, Mapping):
                            continue
                        mean = values.get("mean")
                        deviation = values.get("deviation", 0.0)
                        if isinstance(mean, (int, float)):
                            entry.metrics[str(name)] = CalibrationMetric(
                                mean=float(mean),
                                deviation=float(deviation) if isinstance(deviation, (int, float)) else 0.0,
                            )
                self._entries[(entry.player_name, entry.car_model)] = entry

