"""Calibration workflow for structural operator detectors.

The command line interface exposed by this module orchestrates the end-to-end
calibration process used by TNFR × LFS deployments.  It consumes labelled RAF
captures, derives TNFR telemetry features, performs parameter sweeps for the
detectors under ``tnfr_core.operators.operator_detection`` and materialises the
resulting overrides together with human-readable reports.

The workflow intentionally relies on the existing telemetry helpers so it
remains aligned with the runtime pipeline: ``read_raf`` and
``raf_to_telemetry_records`` extract :class:`~tnfr_core.equations.epi.TelemetryRecord`
sequences, :class:`~tnfr_core.equations.epi.EPIExtractor` reconstructs ΔNFR
bundles and :func:`tnfr_core.metrics.segmentation.segment_microsectors` provides
microsector level aggregates that the detectors inspect.  Labels supplied by
the caller are aligned against the generated microsectors to evaluate the
detector outcomes and determine the best configuration per operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
import argparse
import csv
import json
import logging
import math
import statistics
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

import yaml

from tnfr_core.equations.epi import EPIExtractor, TelemetryRecord
from tnfr_core.metrics.segmentation import Microsector, segment_microsectors
from tnfr_core.operators import operator_detection
from tnfr_core.operators.operator_detection import (
    canonical_operator_label,
    normalize_structural_operator_identifier,
)

from tnfr_lfs.resources import data_root
from tnfr_lfs.resources.tyre_compounds import (
    CAR_COMPOUND_COMPATIBILITY,
    normalise_car_model,
    normalise_compound_identifier,
)
from tnfr_lfs.telemetry.offline.raf import raf_to_telemetry_records, read_raf


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelledMicrosector:
    """Label payload resolved from the ``--labels`` artefact."""

    capture_id: str
    raf_path: Path
    track: str | None
    car: str | None
    car_class: str | None
    compound: str | None
    microsector_index: int
    operators: Mapping[str, bool]
    operator_intervals: Mapping[str, Sequence[tuple[float | None, float | None]]]


@dataclass(frozen=True)
class MicrosectorSample:
    """Telemetry slice paired with operator ground truth."""

    capture_id: str
    microsector_index: int
    track: str
    car: str
    car_class: str | None
    compound: str | None
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    records: Sequence[TelemetryRecord]
    labels: Mapping[str, bool]
    label_intervals: Mapping[str, Sequence[tuple[float, float]]]
    delta_nfr_series: Sequence[float] = ()
    nav_nu_f: float | Mapping[str, float] | None = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))


@dataclass(frozen=True)
class ParameterSet:
    """Candidate parameter payload for a detector."""

    identifier: str
    parameters: Mapping[str, object]


_DEFAULT_OPERATOR_GRID_SPECS: Mapping[str, Mapping[str, Sequence[object]] | Sequence[Mapping[str, object]]] = {
    "AL": {
        "window": (5, 7, 9),
        "lateral_threshold": (1.4, 1.6, 1.8),
        "load_threshold": (200.0, 250.0, 300.0),
    },
    "EN": {
        "window": (4, 6, 8),
        "psi_threshold": (0.8, 0.9, 1.0),
        "epi_norm_max": (100.0, 120.0, 140.0),
    },
    "IL": {
        "window": (5, 7),
        "base_threshold": (0.3, 0.35, 0.4),
        "speed_gain": (0.01, 0.012, 0.015),
    },
    "NAV": {
        "window": (3, 4, 5),
        "eps": (1e-3, 5e-3, 1e-2),
    },
    "NUL": {
        "window": (4, 6, 8),
        "active_nodes_delta_max": (-3, -2, -1),
        "epi_concentration_min": (0.5, 0.6, 0.7),
        "active_node_load_min": (200.0, 250.0, 300.0),
    },
    "OZ": {
        "window": (5, 7),
        "slip_threshold": (0.1, 0.12, 0.14),
        "yaw_threshold": (0.2, 0.25, 0.3),
    },
    "RA": {
        "window": (10, 12),
        "nu_band": ((0.8, 2.5), (1.0, 3.0)),
        "si_min": (0.5, 0.55),
        "delta_nfr_max": (10.0, 12.0),
        "k_min": (2, 3),
    },
    "REMESH": {
        "window": (6, 8, 10),
        "tau_candidates": ((0.2, 0.4, 0.6), (0.3, 0.5, 0.7)),
        "acf_min": (0.7, 0.75, 0.8),
        "min_repeats": (2, 3),
    },
    "SILENCE": {
        "window": (11, 15, 19),
        "load_threshold": (350.0, 400.0, 450.0),
        "accel_threshold": (0.7, 0.8, 0.9),
        "delta_nfr_threshold": (35.0, 45.0, 55.0),
    },
    "THOL": {
        "epi_accel_min": (0.6, 0.8, 1.0),
        "stability_window": (0.3, 0.4, 0.5),
        "stability_tolerance": (0.04, 0.05, 0.06),
    },
    "UM": {
        "window": (6, 8, 10),
        "rho_min": (0.6, 0.65, 0.7),
        "phase_max": (0.2, 0.25, 0.3),
        "min_duration": (0.25, 0.35, 0.45),
    },
    "VAL": {
        "window": (4, 6, 8),
        "epi_growth_min": (0.3, 0.4, 0.5),
        "active_nodes_delta_min": (1, 2),
        "active_node_load_min": (200.0, 250.0, 300.0),
    },
    "ZHIR": {
        "window": (6, 8, 10),
        "xi_min": (0.3, 0.35, 0.4),
        "min_persistence": (0.3, 0.4, 0.5),
        "phase_jump_min": (0.15, 0.2, 0.25),
    },
}


@dataclass
class EvaluationResult:
    """Aggregated metrics for a detector/parameter set combination."""

    operator_id: str
    combination: tuple[str, str, str]
    parameter_set: ParameterSet
    precision: float
    recall: float
    f1: float
    fp_per_minute: float
    support: int
    tp: int
    fp: int
    tn: int
    fn: int
    duration_minutes: float


_SENTINEL_IDENTIFIERS = {"__default__", "__unknown__"}


def _normalise_identifier(value: str | None) -> str | None:
    """Legacy generic normaliser retained for backwards compatibility."""

    token = _normalise_raw_token(value)
    if token is None:
        return None
    return token.lower()


def _normalise_raw_token(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.strip().lower() in _SENTINEL_IDENTIFIERS:
        return None
    return text


def _slugify_token(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


@lru_cache()
def _car_aliases() -> Mapping[str, str]:
    base = data_root() / "cars"
    aliases: dict[str, str] = {}
    for entry in base.glob("*.toml"):
        try:
            with entry.open("rb") as handle:
                payload = tomllib.load(handle)
        except OSError:  # pragma: no cover - resource missing
            continue
        canonical = str(payload.get("abbrev") or entry.stem).strip().upper()
        if not canonical:
            continue
        tokens = {
            canonical.lower(),
            _slugify_token(canonical),
        }
        name = payload.get("name")
        if name:
            label = str(name).strip()
            if label:
                tokens.add(label.lower())
                tokens.add(_slugify_token(label))
        for token in tokens:
            if token:
                aliases[token] = canonical
    return aliases


@lru_cache()
def _car_class_aliases() -> Mapping[str, str]:
    base = data_root() / "cars"
    aliases: dict[str, str] = {}
    for entry in base.glob("*.toml"):
        try:
            with entry.open("rb") as handle:
                payload = tomllib.load(handle)
        except OSError:  # pragma: no cover - resource missing
            continue
        raw_class = payload.get("lfs_class")
        label = _normalise_raw_token(str(raw_class) if raw_class is not None else None)
        if not label:
            continue
        tokens = {label.lower(), _slugify_token(label)}
        for token in tokens:
            if token:
                aliases[token] = label
    return aliases


@lru_cache()
def _track_aliases() -> Mapping[str, str]:
    base = data_root() / "tracks"
    mapping: dict[str, str] = {}
    canonical_targets: dict[str, str] = {}
    token_map: dict[str, set[str]] = {}

    for entry in base.glob("*.toml"):
        try:
            with entry.open("rb") as handle:
                payload = tomllib.load(handle)
        except OSError:  # pragma: no cover - resource missing
            continue
        configs = payload.get("config", {})
        for identifier, values in configs.items():
            canonical = str(identifier).strip().upper()
            if not canonical:
                continue
            alias_target_raw = None
            if isinstance(values, Mapping):
                alias_target_raw = values.get("alias_of")
                name = values.get("name")
            else:
                name = None
            alias_target = (
                str(alias_target_raw).strip().upper()
                if alias_target_raw is not None
                else canonical
            )
            canonical_targets[canonical] = alias_target or canonical
            tokens = token_map.setdefault(canonical, set())
            tokens.add(canonical.lower())
            slug = _slugify_token(canonical)
            if slug:
                tokens.add(slug)
            if name:
                label = _normalise_raw_token(str(name))
                if label:
                    tokens.add(label.lower())
                    slug_label = _slugify_token(label)
                    if slug_label:
                        tokens.add(slug_label)

    def resolve(identifier: str) -> str:
        current = identifier
        visited: set[str] = set()
        while True:
            if current in visited:
                return current
            visited.add(current)
            target = canonical_targets.get(current, current)
            if target == current:
                return current
            current = target

    for identifier, tokens in token_map.items():
        canonical = resolve(identifier)
        for token in tokens:
            if token:
                mapping[token] = canonical
    return mapping


def _normalise_car_identifier(value: str | None) -> str | None:
    token = _normalise_raw_token(value)
    if token is None:
        return None
    aliases = _car_aliases()
    lookup = token.lower()
    canonical = aliases.get(lookup) or aliases.get(_slugify_token(token))
    if canonical:
        return canonical
    fallback = normalise_car_model(token)
    if fallback:
        return fallback
    cleaned = "".join(ch for ch in token.upper() if ch.isalnum())
    return cleaned or None


def _normalise_car_class_identifier(value: str | None) -> str | None:
    token = _normalise_raw_token(value)
    if token is None:
        return None
    aliases = _car_class_aliases()
    lookup = token.lower()
    canonical = aliases.get(lookup) or aliases.get(_slugify_token(token))
    if canonical:
        return canonical
    return token.upper()


def _normalise_track_identifier(value: str | None) -> str | None:
    token = _normalise_raw_token(value)
    if token is None:
        return None
    aliases = _track_aliases()
    lookup = token.lower()
    canonical = aliases.get(lookup) or aliases.get(_slugify_token(token))
    if canonical:
        return canonical
    cleaned = _slugify_token(token).upper()
    if cleaned:
        return cleaned
    return token.upper()


def _normalise_compound_token(value: str | None) -> str | None:
    canonical = normalise_compound_identifier(value)
    if canonical:
        return canonical
    token = _normalise_raw_token(value)
    if token is None:
        return None
    return token.lower()


def _normalise_operator_labels(payload: Any) -> dict[str, bool]:
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        entries: dict[str, bool] = {}
        for key, value in payload.items():
            if key is None:
                continue
            identifier = normalize_structural_operator_identifier(str(key))
            if isinstance(value, Mapping):
                status = value.get("active")
                if status is None:
                    status = value.get("label")
                if status is None:
                    status = value.get("positive")
                if status is not None:
                    if isinstance(status, str):
                        token = status.strip().lower()
                        entries[identifier] = token not in {"", "0", "false", "no"}
                    else:
                        entries[identifier] = bool(status)
                    continue
            if isinstance(value, str):
                token = value.strip().lower()
                entries[identifier] = token not in {"", "0", "false", "no"}
            else:
                entries[identifier] = bool(value)
        return entries
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        entries: dict[str, bool] = {}
        for item in payload:
            if item is None:
                continue
            identifier = normalize_structural_operator_identifier(str(item))
            entries[identifier] = True
        return entries
    if isinstance(payload, str):
        tokens = [token.strip() for token in payload.split(",")]
        return {
            normalize_structural_operator_identifier(token): True
            for token in tokens
            if token
        }
    return {}


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer


def _parse_interval_payload(value: Any) -> list[tuple[float | None, float | None]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("intervals", "ranges", "windows"):
            nested = value.get(key)
            parsed = _parse_interval_payload(nested)
            if parsed:
                return parsed
        start = value.get("start")
        if start is None:
            start = value.get("begin") or value.get("t0")
        end = value.get("end")
        if end is None:
            end = value.get("stop") or value.get("t1")
        start_value = _coerce_float(start)
        end_value = _coerce_float(end)
        if start is None and end is None:
            return []
        return [(start_value, end_value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2:
            start_value = _coerce_float(value[0])
            end_value = _coerce_float(value[1])
            if start_value is not None or end_value is not None:
                return [(start_value, end_value)]
        intervals: list[tuple[float | None, float | None]] = []
        for item in value:
            intervals.extend(_parse_interval_payload(item))
        return intervals
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return []
        return _parse_interval_payload(decoded)
    return []


def _normalise_operator_intervals(
    payload: Any,
) -> dict[str, tuple[tuple[float | None, float | None], ...]]:
    intervals: dict[str, list[tuple[float | None, float | None]]] = {}

    def ensure(identifier: str) -> None:
        intervals.setdefault(identifier, [])

    def add(identifier: str, values: Iterable[tuple[float | None, float | None]]) -> None:
        bucket = intervals.setdefault(identifier, [])
        for start, end in values:
            bucket.append((start, end))

    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key is None:
                continue
            identifier = normalize_structural_operator_identifier(str(key))
            parsed = _parse_interval_payload(value)
            if parsed:
                add(identifier, parsed)
            else:
                ensure(identifier)
        return {key: tuple(value) for key, value in intervals.items()}
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for item in payload:
            if item is None:
                continue
            if isinstance(item, Mapping):
                operator_token = item.get("operator") or item.get("name") or item.get("id")
                if operator_token is None:
                    continue
                identifier = normalize_structural_operator_identifier(str(operator_token))
                parsed = _parse_interval_payload(item)
                if parsed:
                    add(identifier, parsed)
                else:
                    ensure(identifier)
            else:
                identifier = normalize_structural_operator_identifier(str(item))
                ensure(identifier)
        return {key: tuple(value) for key, value in intervals.items()}
    if isinstance(payload, str):
        parsed = _parse_interval_payload(payload)
        if parsed:
            # Without operator context we cannot associate the intervals.
            return {}
    return {}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_lines(path: Path) -> list[Any]:
    entries: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Invalid JSON entry on line {line_number} of '{path}': {exc.msg}"
                ) from exc
            if isinstance(payload, Mapping) and "captures" in payload:
                captures = payload.get("captures")
                if isinstance(captures, Sequence) and not isinstance(
                    captures, (str, bytes)
                ):
                    entries.extend(captures)
                else:
                    entries.append(payload)
            else:
                entries.append(payload)
    return entries


def _load_toml(path: Path) -> Any:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_labels(path: Path, *, raf_root: Path) -> list[LabelledMicrosector]:
    """Parse the ``--labels`` artefact into :class:`LabelledMicrosector` entries."""

    loader_map: dict[str, Callable[[Path], Any]] = {
        ".json": _load_json,
        ".jsonl": _load_json_lines,
        ".toml": _load_toml,
        ".tml": _load_toml,
        ".yaml": _load_yaml,
        ".yml": _load_yaml,
        ".csv": None,
    }
    suffix = path.suffix.lower()
    if suffix not in loader_map:
        raise ValueError(f"Unsupported label format: {path.suffix}")
    if suffix == ".csv":
        return list(_load_labels_csv(path, raf_root=raf_root))
    loader = loader_map[suffix]
    assert loader is not None
    data = loader(path)
    return list(_load_labels_mapping(data, raf_root=raf_root))


def _load_labels_csv(path: Path, *, raf_root: Path) -> Iterator[LabelledMicrosector]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                micro_index = int(str(row.get("microsector") or row.get("microsector_index")))
            except (TypeError, ValueError):
                LOGGER.debug("Skipping CSV row without a valid microsector index: %s", row)
                continue
            operator_source = row.get("operators") or row.get("operator")
            operators = _normalise_operator_labels(operator_source)
            operator_intervals = _normalise_operator_intervals(operator_source)
            extra_intervals_source = (
                row.get("intervals")
                or row.get("operator_intervals")
                or row.get("ranges")
            )
            if extra_intervals_source:
                extra_intervals = _normalise_operator_intervals(extra_intervals_source)
                for identifier, values in extra_intervals.items():
                    if identifier in operator_intervals:
                        operator_intervals[identifier] = (
                            operator_intervals[identifier] + values
                        )
                    else:
                        operator_intervals[identifier] = values
            if not operators and row.get("label") is not None:
                operators = {
                    normalize_structural_operator_identifier(str(row.get("operator"))): bool(
                        str(row.get("label")).strip() not in {"", "0", "false", "False"}
                    )
                }
                operator_intervals = {}
            raf_token = row.get("raf") or row.get("path") or row.get("capture")
            if not raf_token:
                LOGGER.debug("CSV row missing RAF reference: %s", row)
                continue
            raf_path = (raf_root / str(raf_token)).resolve()
            capture_id = str(row.get("capture_id") or row.get("capture") or raf_path.stem)
            yield LabelledMicrosector(
                capture_id=capture_id,
                raf_path=raf_path,
                track=_normalise_track_identifier(row.get("track") or row.get("track_name")),
                car=_normalise_car_identifier(row.get("car") or row.get("car_model")),
                car_class=_normalise_car_class_identifier(
                    row.get("car_class") or row.get("class")
                ),
                compound=_normalise_compound_token(
                    row.get("compound") or row.get("tyre") or row.get("tyre_compound")
                ),
                microsector_index=micro_index,
                operators=operators,
                operator_intervals={key: tuple(values) for key, values in operator_intervals.items()},
            )


def _load_labels_mapping(payload: Any, *, raf_root: Path) -> Iterator[LabelledMicrosector]:
    if isinstance(payload, Mapping):
        if "captures" in payload and isinstance(payload["captures"], Sequence):
            captures = payload["captures"]
        else:
            captures = [payload]
    elif isinstance(payload, Sequence):
        captures = payload
    else:
        raise ValueError("Labels payload must be a mapping or sequence")

    for entry in captures:
        if not isinstance(entry, Mapping):
            continue
        raf_token = entry.get("raf") or entry.get("path") or entry.get("capture")
        if not raf_token:
            LOGGER.debug("Skipping capture without RAF reference: %s", entry)
            continue
        raf_path = (raf_root / str(raf_token)).resolve()
        capture_id = str(entry.get("id") or entry.get("capture_id") or raf_path.stem)
        track = _normalise_track_identifier(entry.get("track") or entry.get("track_name"))
        car = _normalise_car_identifier(entry.get("car") or entry.get("car_model"))
        car_class = _normalise_car_class_identifier(entry.get("car_class") or entry.get("class"))
        compound = _normalise_compound_token(
            entry.get("compound") or entry.get("tyre") or entry.get("tyre_compound")
        )
        micro_payload = entry.get("microsectors") or entry.get("labels")
        if isinstance(micro_payload, Mapping):
            iterator = micro_payload.items()
        elif isinstance(micro_payload, Sequence):
            iterator = enumerate(micro_payload)
        else:
            LOGGER.debug("Capture missing microsector labels: %s", entry)
            continue
        for micro_index, micro_entry in iterator:
            index_value = micro_index
            operators = {}
            operator_intervals: dict[str, tuple[tuple[float | None, float | None], ...]] = {}
            if isinstance(micro_entry, Mapping):
                index_value = micro_entry.get("index", micro_index)
                operator_payload = (
                    micro_entry.get("operators")
                    or micro_entry.get("labels")
                    or micro_entry.get("events")
                )
                operators = _normalise_operator_labels(operator_payload)
                operator_intervals = _normalise_operator_intervals(operator_payload)
            elif isinstance(micro_entry, Sequence) and not isinstance(micro_entry, (str, bytes)):
                operators = _normalise_operator_labels(micro_entry)
                operator_intervals = _normalise_operator_intervals(micro_entry)
            else:
                operators = _normalise_operator_labels(micro_entry)
                operator_intervals = _normalise_operator_intervals(micro_entry)
            try:
                microsector_index = int(index_value)
            except (TypeError, ValueError):
                LOGGER.debug("Skipping microsector with invalid index: %s", micro_entry)
                continue
            yield LabelledMicrosector(
                capture_id=capture_id,
                raf_path=raf_path,
                track=track,
                car=car,
                car_class=car_class,
                compound=compound,
                microsector_index=microsector_index,
                operators=operators,
                operator_intervals=operator_intervals,
            )


def _microsector_span(microsector: Microsector) -> tuple[int, int]:
    indices: list[int] = []
    for sequence in microsector.phase_samples.values():
        indices.extend(int(value) for value in sequence)
    if not indices:
        raise ValueError("Microsector does not expose phase samples")
    start = min(indices)
    end = max(indices)
    return start, end


def _normalise_interval_bounds(
    intervals: Sequence[tuple[float | None, float | None]],
    *,
    default_start: float,
    default_end: float,
) -> list[tuple[float, float]]:
    resolved: list[tuple[float, float]] = []
    start_bound = min(default_start, default_end)
    end_bound = max(default_start, default_end)
    for raw_start, raw_end in intervals:
        start_time = default_start if raw_start is None else float(raw_start)
        end_time = default_end if raw_end is None else float(raw_end)
        if not math.isfinite(start_time):
            start_time = default_start
        if not math.isfinite(end_time):
            end_time = default_end
        if end_time < start_time:
            start_time, end_time = end_time, start_time
        start_time = max(start_bound, start_time)
        end_time = min(end_bound, end_time)
        if end_time <= start_time:
            continue
        resolved.append((start_time, end_time))
    resolved.sort(key=lambda item: item[0])
    return resolved


def _resolve_operator_truth(
    label: LabelledMicrosector,
    *,
    micro_start_time: float,
    micro_end_time: float,
) -> tuple[dict[str, bool], dict[str, tuple[tuple[float, float], ...]]]:
    operator_ids = set(label.operators)
    operator_ids.update(label.operator_intervals)
    if not operator_ids:
        return {}, {}

    labels: dict[str, bool] = {}
    intervals: dict[str, tuple[tuple[float, float], ...]] = {}

    for operator_id in sorted(operator_ids):
        positive = label.operators.get(operator_id)
        if positive is None:
            positive = bool(label.operator_intervals.get(operator_id))
        else:
            positive = bool(positive)
        resolved = _normalise_interval_bounds(
            label.operator_intervals.get(operator_id, ()),
            default_start=micro_start_time,
            default_end=micro_end_time,
        )
        if positive and not resolved and micro_end_time > micro_start_time:
            resolved = [(micro_start_time, micro_end_time)]
        labels[operator_id] = bool(positive)
        intervals[operator_id] = tuple(resolved)

    return labels, intervals


def _build_microsector_dataset(
    labels: Sequence[LabelledMicrosector],
    *,
    raf_root: Path,
) -> list[MicrosectorSample]:
    grouped: MutableMapping[Path, list[LabelledMicrosector]] = {}
    for entry in labels:
        grouped.setdefault(entry.raf_path, []).append(entry)

    extractor = EPIExtractor()
    dataset: list[MicrosectorSample] = []

    for raf_path, micro_labels in grouped.items():
        try:
            raf_file = read_raf(raf_path)
        except OSError as exc:
            LOGGER.warning("Failed to read RAF '%s': %s", raf_path, exc)
            continue
        metadata: dict[str, object] = {}
        compound = next((label.compound for label in micro_labels if label.compound), None)
        if compound:
            metadata["tyre_compound"] = compound
        try:
            records = raf_to_telemetry_records(raf_file, metadata=metadata)
        except Exception as exc:  # pragma: no cover - defensive catch
            LOGGER.warning("Failed to convert RAF '%s' to telemetry records: %s", raf_path, exc)
            continue
        if not records:
            LOGGER.debug("RAF '%s' produced no telemetry records", raf_path)
            continue
        try:
            bundles = extractor.extract(list(records))
        except Exception as exc:  # pragma: no cover - defensive catch
            LOGGER.warning("Failed to extract EPI bundles for '%s': %s", raf_path, exc)
            continue
        try:
            microsectors = segment_microsectors(records, bundles)
        except Exception as exc:  # pragma: no cover - defensive catch
            LOGGER.warning("Microsector segmentation failed for '%s': %s", raf_path, exc)
            continue
        if not microsectors:
            LOGGER.debug("RAF '%s' yielded no microsectors", raf_path)
            continue
        micro_map: dict[int, Microsector] = {micro.index: micro for micro in microsectors}
        for label in micro_labels:
            micro = micro_map.get(label.microsector_index)
            if micro is None:
                LOGGER.warning(
                    "Capture '%s' missing microsector %s in RAF '%s'",
                    label.capture_id,
                    label.microsector_index,
                    raf_path,
                )
                continue
            try:
                start_index, end_index = _microsector_span(micro)
            except ValueError as exc:
                LOGGER.warning(
                    "Microsector %s in '%s' is missing sample indices: %s",
                    label.microsector_index,
                    raf_path,
                    exc,
                )
                continue
            track = label.track or raf_file.header.track_name or "__unknown__"
            car_model = label.car or raf_file.header.car_model or "__unknown__"
            car_class = label.car_class
            compound = label.compound
            labels_map, interval_map = _resolve_operator_truth(
                label,
                micro_start_time=float(micro.start_time),
                micro_end_time=float(micro.end_time),
            )
            delta_series: list[float] = []
            nu_f_values: list[float] = []
            if bundles:
                bundle_len = len(bundles)
                if bundle_len > 0:
                    start_bundle = max(0, min(start_index, bundle_len - 1))
                    end_bundle = max(start_bundle, min(end_index, bundle_len - 1))
                    for bundle in bundles[start_bundle : end_bundle + 1]:
                        delta_value = _coerce_float(getattr(bundle, "delta_nfr", None))
                        if delta_value is None or not math.isfinite(delta_value):
                            delta_value = 0.0
                        delta_series.append(delta_value)
                        nu_candidates = (
                            getattr(bundle, "nu_f_dominant", None),
                            getattr(getattr(bundle, "track", None), "nu_f", None),
                            getattr(getattr(bundle, "driver", None), "nu_f", None),
                        )
                        for candidate in nu_candidates:
                            numeric = _coerce_float(candidate)
                            if numeric is None or not math.isfinite(numeric):
                                continue
                            nu_f_values.append(float(numeric))
                            break
            nav_nu_f: float | None = None
            if nu_f_values:
                try:
                    nav_nu_f = float(statistics.median(nu_f_values))
                except statistics.StatisticsError:  # pragma: no cover - defensive
                    nav_nu_f = None
            dataset.append(
                MicrosectorSample(
                    capture_id=label.capture_id,
                    microsector_index=label.microsector_index,
                    track=str(track),
                    car=str(car_model),
                    car_class=str(car_class) if car_class else None,
                    compound=str(compound) if compound else None,
                    start_index=start_index,
                    end_index=end_index,
                    start_time=float(micro.start_time),
                    end_time=float(micro.end_time),
                    records=records,
                    delta_nfr_series=tuple(delta_series),
                    nav_nu_f=nav_nu_f,
                    labels=labels_map or dict(label.operators),
                    label_intervals=interval_map,
                )
            )
    return dataset


def _filter_samples(
    samples: Sequence[MicrosectorSample],
    *,
    cars: set[str] | None,
    compounds: set[str] | None,
    tracks: set[str] | None,
) -> list[MicrosectorSample]:
    def include(sample: MicrosectorSample) -> bool:
        sample_car = _normalise_car_identifier(sample.car)
        sample_compound = _normalise_compound_token(sample.compound)
        sample_track = _normalise_track_identifier(sample.track)

        if cars and sample_car not in cars:
            return False
        if compounds and sample_compound not in compounds:
            return False
        if tracks and sample_track not in tracks:
            return False
        return True

    filtered = [sample for sample in samples if include(sample)]
    if not filtered:
        raise ValueError("Filters removed every microsector sample")
    return filtered


def _parameter_sets_from_definition(
    identifier: str, definition: Any
) -> list[ParameterSet]:
    param_sets: list[ParameterSet] = []
    if isinstance(definition, Sequence) and not isinstance(definition, (str, bytes)):
        for index, entry in enumerate(definition):
            if not isinstance(entry, Mapping):
                continue
            param_sets.append(
                ParameterSet(
                    identifier=f"{identifier}-{index}",
                    parameters=dict(entry),
                )
            )
    elif isinstance(definition, Mapping):
        keys = sorted(definition.keys())
        grids: list[list[tuple[str, object]]] = []
        for param_name in keys:
            options = definition[param_name]
            if isinstance(options, Mapping) and "values" in options:
                options = options["values"]
            if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
                grids.append([(param_name, option) for option in options])
            else:
                grids.append([(param_name, options)])
        for index, combination in enumerate(product(*grids)):
            params = {name: entry for name, entry in combination}
            param_sets.append(
                ParameterSet(
                    identifier=f"{identifier}-{index}",
                    parameters=params,
                )
            )
    else:
        raise ValueError(
            f"Operator '{identifier}' search space must be a mapping or sequence"
        )
    return param_sets


def _build_operator_grid_from_payload(
    payload: Mapping[str, Any]
) -> dict[str, list[ParameterSet]]:
    operator_grid: dict[str, list[ParameterSet]] = {}
    for key, value in payload.items():
        identifier = normalize_structural_operator_identifier(str(key))
        param_sets = _parameter_sets_from_definition(identifier, value)
        if not param_sets:
            raise ValueError(f"Operator '{key}' has no parameter sets to evaluate")
        operator_grid[identifier] = param_sets
    return operator_grid


@lru_cache(maxsize=1)
def _default_operator_grid() -> Mapping[str, list[ParameterSet]]:
    return _build_operator_grid_from_payload(_DEFAULT_OPERATOR_GRID_SPECS)


def _load_operator_grid(path: Path) -> dict[str, list[ParameterSet]]:
    loader_map: dict[str, Callable[[Path], Any]] = {
        ".json": _load_json,
        ".toml": _load_toml,
        ".tml": _load_toml,
        ".yaml": _load_yaml,
        ".yml": _load_yaml,
    }
    suffix = path.suffix.lower()
    if suffix not in loader_map:
        raise ValueError(f"Unsupported operator search space format: {path.suffix}")
    data = loader_map[suffix](path)
    if not isinstance(data, Mapping):
        raise ValueError("Operator search space must be a mapping")
    payload = data.get("operators") if "operators" in data else data
    if not isinstance(payload, Mapping):
        raise ValueError("Operator search space must be a mapping")
    return _build_operator_grid_from_payload(payload)


def _detector_callable(operator_id: str) -> Callable[..., Sequence[Mapping[str, object]]]:
    normalised = normalize_structural_operator_identifier(operator_id)
    func_name = f"detect_{normalised.lower()}"
    detector = getattr(operator_detection, func_name, None)
    if detector is None:
        raise ValueError(f"Unknown detector for operator '{operator_id}'")
    return detector


def _group_combinations(
    samples: Sequence[MicrosectorSample],
) -> tuple[dict[tuple[str, str, str], list[MicrosectorSample]], set[tuple[str, str]]]:
    grouped: dict[tuple[str, str, str], list[MicrosectorSample]] = {}
    invalid_pairs: set[tuple[str, str]] = set()
    for sample in samples:
        car_class = sample.car_class or "__default__"
        car_model = sample.car or "__unknown__"
        compound = sample.compound or "__default__"
        grouped.setdefault((car_class, car_model, compound), []).append(sample)

        canonical_car = normalise_car_model(sample.car)
        canonical_compound = normalise_compound_identifier(sample.compound)
        if canonical_car and canonical_compound:
            allowed = CAR_COMPOUND_COMPATIBILITY.get(canonical_car)
            if allowed is not None and canonical_compound not in allowed:
                invalid_pairs.add((str(sample.car), str(sample.compound)))
    return grouped, invalid_pairs


def _build_folds(samples: Sequence[MicrosectorSample], kfold: int) -> dict[str, int]:
    tracks = sorted({sample.track for sample in samples})
    if not tracks:
        return {}
    folds = max(1, min(kfold, len(tracks)))
    assignments: dict[str, int] = {}
    for index, track in enumerate(tracks):
        assignments[track] = index % folds
    return assignments


def _event_attribute(event: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(event, Mapping) and name in event:
        return event.get(name)
    return getattr(event, name, None)


def _normalise_event_interval(
    event: Mapping[str, Any] | Any,
    window: Sequence[TelemetryRecord],
    sample: MicrosectorSample,
) -> tuple[float, float] | None:
    start_time = _coerce_float(_event_attribute(event, "start_time"))
    end_time = _coerce_float(_event_attribute(event, "end_time"))
    duration = _coerce_float(_event_attribute(event, "duration"))

    def clamp_bounds(start: float, end: float) -> tuple[float, float]:
        start_bound = min(sample.start_time, sample.end_time)
        end_bound = max(sample.start_time, sample.end_time)
        start_clamped = min(max(start, start_bound), end_bound)
        end_clamped = min(max(end, start_bound), end_bound)
        if end_clamped < start_clamped:
            start_clamped, end_clamped = end_clamped, start_clamped
        return start_clamped, end_clamped

    if start_time is not None and end_time is not None:
        start_time, end_time = clamp_bounds(start_time, end_time)
        if end_time > start_time:
            return start_time, end_time
    if start_time is not None and duration is not None and duration >= 0.0:
        interval = clamp_bounds(start_time, start_time + duration)
        if interval[1] > interval[0]:
            return interval
    if end_time is not None and duration is not None and duration >= 0.0:
        interval = clamp_bounds(end_time - duration, end_time)
        if interval[1] > interval[0]:
            return interval

    start_index = _coerce_int(_event_attribute(event, "start_index"))
    end_index = _coerce_int(_event_attribute(event, "end_index"))
    if start_index is not None or end_index is not None:
        if not window:
            return None
        window_len = len(window)
        rel_start = start_index if start_index is not None else end_index
        if rel_start is None:
            return None
        rel_end = end_index if end_index is not None else rel_start
        rel_start = max(0, min(rel_start, window_len - 1))
        rel_end = max(rel_start, min(rel_end, window_len - 1))
        start_candidate = _coerce_float(getattr(window[rel_start], "timestamp", None))
        end_candidate = _coerce_float(getattr(window[rel_end], "timestamp", None))
        if start_candidate is None:
            start_candidate = sample.start_time
        if end_candidate is None:
            end_candidate = sample.end_time
        start_candidate, end_candidate = clamp_bounds(start_candidate, end_candidate)
        if end_candidate > start_candidate:
            return start_candidate, end_candidate

    if duration is not None and duration >= 0.0:
        interval = clamp_bounds(sample.start_time, sample.start_time + duration)
        if interval[1] > interval[0]:
            return interval

    fallback = clamp_bounds(sample.start_time, sample.end_time)
    if fallback[1] > fallback[0]:
        return fallback
    return None


def _normalise_event_intervals(
    events: Sequence[Mapping[str, Any] | Any],
    window: Sequence[TelemetryRecord],
    sample: MicrosectorSample,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for event in events:
        interval = _normalise_event_interval(event, window, sample)
        if interval is None:
            continue
        start_time, end_time = interval
        if end_time <= start_time:
            continue
        intervals.append((start_time, end_time))
    intervals.sort(key=lambda item: item[0])
    return intervals


def _interval_iou(
    lhs: tuple[float, float], rhs: tuple[float, float]
) -> float:
    start = max(lhs[0], rhs[0])
    end = min(lhs[1], rhs[1])
    intersection = max(0.0, end - start)
    if intersection <= 0.0:
        return 0.0
    lhs_length = max(0.0, lhs[1] - lhs[0])
    rhs_length = max(0.0, rhs[1] - rhs[0])
    union = lhs_length + rhs_length - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _match_intervals(
    truth: Sequence[tuple[float, float]],
    predictions: Sequence[tuple[float, float]],
    *,
    threshold: float,
) -> tuple[set[int], set[int]]:
    if not truth or not predictions:
        return set(), set()
    candidates: list[tuple[float, int, int]] = []
    for truth_index, pred_index in product(range(len(truth)), range(len(predictions))):
        score = _interval_iou(truth[truth_index], predictions[pred_index])
        if score >= threshold:
            candidates.append((score, truth_index, pred_index))
    candidates.sort(key=lambda item: item[0], reverse=True)
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    for _, truth_index, pred_index in candidates:
        if truth_index in matched_truth or pred_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(pred_index)
    return matched_truth, matched_predictions


def _evaluate_detector(
    operator_id: str,
    samples: Sequence[MicrosectorSample],
    parameter_set: ParameterSet,
    *,
    fold_assignments: Mapping[str, int],
) -> EvaluationResult | None:
    if not samples:
        return None
    detector = _detector_callable(operator_id)
    canonical_id = normalize_structural_operator_identifier(operator_id)
    tp = fp = tn = fn = 0
    total_duration = 0.0
    support = 0
    evaluated_samples = 0

    if fold_assignments:
        fold_indices = sorted(set(fold_assignments.values()))
    else:
        fold_indices = [0]

    for fold_index in fold_indices:
        for sample in samples:
            sample_fold = fold_assignments.get(sample.track, 0)
            if sample_fold != fold_index:
                continue
            truth_intervals = list(sample.label_intervals.get(operator_id, ()))
            is_positive = bool(sample.labels.get(operator_id)) or bool(truth_intervals)
            window = list(sample.records[sample.start_index : sample.end_index + 1])
            try:
                if canonical_id == "NAV":
                    delta_series = list(sample.delta_nfr_series)
                    if not delta_series and window:
                        for record in window:
                            delta_value = _coerce_float(getattr(record, "delta_nfr", None))
                            if delta_value is None or not math.isfinite(delta_value):
                                delta_series.append(0.0)
                            else:
                                delta_series.append(float(delta_value))
                    nu_f_value: float | Mapping[str, float] | None = sample.nav_nu_f
                    if not isinstance(nu_f_value, Mapping) and nu_f_value is None and delta_series:
                        try:
                            nu_f_value = float(
                                statistics.median(abs(value) for value in delta_series)
                            )
                        except statistics.StatisticsError:  # pragma: no cover - defensive
                            nu_f_value = None
                    events = detector(
                        delta_series,
                        nu_f=nu_f_value,
                        metadata=window,
                        **parameter_set.parameters,
                    )
                else:
                    events = detector(window, **parameter_set.parameters)
            except Exception as exc:
                LOGGER.debug(
                    "Detector '%s' failed for sample %s/%s: %s",
                    operator_id,
                    sample.capture_id,
                    sample.microsector_index,
                    exc,
                )
                events = ()
            prediction_intervals = _normalise_event_intervals(events, window, sample)
            matched_truth, matched_predictions = _match_intervals(
                truth_intervals, prediction_intervals, threshold=0.5
            )
            evaluated_samples += 1
            tp += len(matched_truth)
            unmatched_predictions = len(prediction_intervals) - len(matched_predictions)
            if unmatched_predictions > 0:
                fp += unmatched_predictions
            if truth_intervals:
                support += len(truth_intervals)
            elif is_positive:
                support += 1
            unmatched_truth = max(0, len(truth_intervals) - len(matched_truth))
            if unmatched_truth > 0:
                fn += unmatched_truth
            if not truth_intervals and not is_positive:
                if not prediction_intervals:
                    tn += 1
            total_duration += sample.duration_seconds

    if evaluated_samples == 0:
        return None

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    duration_minutes = total_duration / 60.0 if total_duration > 0 else 0.0
    if duration_minutes <= 0.0:
        fp_per_minute = math.inf if fp > 0 else 0.0
    else:
        fp_per_minute = fp / duration_minutes
    return EvaluationResult(
        operator_id=operator_id,
        combination=(samples[0].car_class or "__default__", samples[0].car, samples[0].compound or "__default__"),
        parameter_set=parameter_set,
        precision=precision,
        recall=recall,
        f1=f1,
        fp_per_minute=fp_per_minute,
        support=support,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        duration_minutes=duration_minutes,
    )


def _select_best(
    results: Sequence[EvaluationResult],
    *,
    fp_per_minute_max: float,
) -> EvaluationResult | None:
    if not results:
        return None
    eligible = [result for result in results if result.fp_per_minute <= fp_per_minute_max]
    candidates = eligible or list(results)
    return max(
        candidates,
        key=lambda entry: (
            round(entry.f1, 6),
            round(entry.recall, 6),
            round(entry.precision, 6),
            -round(entry.fp_per_minute, 6),
        ),
    )


def _ensure_mapping(
    root: MutableMapping[str, Any], path: Sequence[str]
) -> MutableMapping[str, Any]:
    cursor: MutableMapping[str, Any] = root
    for raw_key in path:
        key = str(raw_key)
        existing = cursor.get(key)
        if not isinstance(existing, MutableMapping):
            existing = {}
            cursor[key] = existing
        cursor = existing
    return cursor


def _merge_parameter_values(
    target: MutableMapping[str, Any], params: Mapping[str, object]
) -> None:
    for key, value in params.items():
        target[str(key)] = value


def _detector_function_name(operator_id: str) -> str:
    detector = _detector_callable(operator_id)
    name = getattr(detector, "__name__", None)
    if not isinstance(name, str):  # pragma: no cover - defensive
        normalised = normalize_structural_operator_identifier(operator_id)
        return f"detect_{normalised.lower()}"
    return name


def _materialise_best_params(
    selections: Sequence[EvaluationResult],
    *,
    output_path: Path,
) -> None:
    grouped: dict[str, list[EvaluationResult]] = {}
    for selection in selections:
        detector_name = _detector_function_name(selection.operator_id)
        grouped.setdefault(detector_name, []).append(selection)

    payload: dict[str, Any] = {}
    for detector_name, entries in sorted(grouped.items()):
        detector_payload = payload.setdefault(detector_name, {})
        if not isinstance(detector_payload, MutableMapping):  # pragma: no cover - defensive
            detector_payload = {}
            payload[detector_name] = detector_payload
        for selection in entries:
            car_class, car_model, compound = selection.combination
            class_key = str(car_class) if car_class is not None else "__default__"
            car_key = str(car_model) if car_model is not None else "__unknown__"
            compound_key = (
                str(compound) if compound is not None else "__default__"
            )
            params = selection.parameter_set.parameters
            if not params:
                continue

            if class_key and class_key != "__default__":
                if compound_key != "__default__":
                    path = ("classes", class_key, "compounds", compound_key)
                else:
                    path = ("classes", class_key, "defaults")
                section = _ensure_mapping(detector_payload, path)
                _merge_parameter_values(section, params)

            if car_key and car_key != "__unknown__":
                if compound_key != "__default__":
                    path = ("cars", car_key, "compounds", compound_key)
                else:
                    path = ("cars", car_key, "defaults")
                section = _ensure_mapping(detector_payload, path)
                _merge_parameter_values(section, params)

            if (not class_key or class_key == "__default__") and (
                not car_key or car_key == "__unknown__"
            ):
                if compound_key != "__default__":
                    path = ("compounds", compound_key)
                else:
                    path = ("defaults",)
                section = _ensure_mapping(detector_payload, path)
                _merge_parameter_values(section, params)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


def _write_curves_csv(
    results: Sequence[EvaluationResult],
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_operator: dict[str, list[EvaluationResult]] = {}
    for result in results:
        by_operator.setdefault(result.operator_id, []).append(result)
    for operator_id, entries in by_operator.items():
        path = output_dir / f"{operator_id.lower()}_curves.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "operator",
                    "car_class",
                    "car",
                    "compound",
                    "parameter_set",
                    "precision",
                    "recall",
                    "f1",
                    "fp_per_min",
                    "support",
                ]
            )
            for entry in sorted(
                entries,
                key=lambda item: (
                    item.combination,
                    item.parameter_set.identifier,
                ),
            ):
                writer.writerow(
                    [
                        operator_id,
                        entry.combination[0],
                        entry.combination[1],
                        entry.combination[2],
                        entry.parameter_set.identifier,
                        f"{entry.precision:.6f}",
                        f"{entry.recall:.6f}",
                        f"{entry.f1:.6f}",
                        f"{entry.fp_per_minute:.6f}",
                        entry.support,
                    ]
                )


def _write_confusion_csv(
    selections: Sequence[EvaluationResult],
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_operator: dict[str, list[EvaluationResult]] = {}
    for result in selections:
        by_operator.setdefault(result.operator_id, []).append(result)
    for operator_id, entries in by_operator.items():
        path = output_dir / f"{operator_id.lower()}_confusion.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "operator",
                    "car_class",
                    "car",
                    "compound",
                    "parameter_set",
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                    "duration_minutes",
                ]
            )
            for entry in sorted(entries, key=lambda item: item.combination):
                writer.writerow(
                    [
                        operator_id,
                        entry.combination[0],
                        entry.combination[1],
                        entry.combination[2],
                        entry.parameter_set.identifier,
                        entry.tp,
                        entry.fp,
                        entry.tn,
                        entry.fn,
                        f"{entry.duration_minutes:.6f}",
                    ]
                )


def _write_report(
    selections: Sequence[EvaluationResult],
    *,
    output_path: Path,
) -> None:
    lines: list[str] = ["# Detector calibration report", ""]
    by_operator: dict[str, list[EvaluationResult]] = {}
    for result in selections:
        by_operator.setdefault(result.operator_id, []).append(result)
    for operator_id, entries in sorted(by_operator.items()):
        label = canonical_operator_label(operator_id)
        lines.append(f"## {operator_id} · {label}")
        lines.append("")
        lines.append(
            "| Car class | Car | Compound | Precision | Recall | F1 | FP/min | Params | Support |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for entry in sorted(entries, key=lambda item: item.combination):
            params = ", ".join(
                f"{key}={value}" for key, value in sorted(entry.parameter_set.parameters.items())
            )
            lines.append(
                "| {car_class} | {car} | {compound} | {precision:.3f} | {recall:.3f} | {f1:.3f} | "
                "{fp_per_minute:.3f} | {params} | {support} |".format(
                    car_class=entry.combination[0],
                    car=entry.combination[1],
                    compound=entry.combination[2],
                    precision=entry.precision,
                    recall=entry.recall,
                    f1=entry.f1,
                    fp_per_minute=entry.fp_per_minute,
                    params=params or "—",
                    support=entry.support,
                )
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def calibrate_detectors(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raf_root = Path(args.raf_root).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()

    LOGGER.info("Loading labelled microsectors from %s", labels_path)
    label_entries = _load_labels(labels_path, raf_root=raf_root)
    if not label_entries:
        raise SystemExit("No labelled microsectors were found in the provided artefact")

    LOGGER.info("Building telemetry dataset from %d labelled microsectors", len(label_entries))
    samples = _build_microsector_dataset(label_entries, raf_root=raf_root)
    if not samples:
        raise SystemExit("No microsector samples were produced from the supplied RAF captures")

    def _normalised_filter_tokens(
        values: Sequence[str] | None,
        normaliser: Callable[[str | None], str | None],
    ) -> set[str]:
        tokens: set[str] = set()
        for value in values or []:
            canonical = normaliser(value)
            if canonical:
                tokens.add(canonical)
        return tokens

    cars = _normalised_filter_tokens(args.cars, _normalise_car_identifier)
    compounds = _normalised_filter_tokens(args.compounds, _normalise_compound_token)
    tracks = _normalised_filter_tokens(args.tracks, _normalise_track_identifier)
    if cars or compounds or tracks:
        try:
            samples = _filter_samples(
                samples,
                cars=cars or None,
                compounds=compounds or None,
                tracks=tracks or None,
            )
        except ValueError as exc:
            applied_filters: list[str] = []
            if cars:
                applied_filters.append(f"cars={sorted(cars)}")
            if compounds:
                applied_filters.append(f"compounds={sorted(compounds)}")
            if tracks:
                applied_filters.append(f"tracks={sorted(tracks)}")
            filters_text = ", ".join(applied_filters)
            context = f" ({filters_text})" if filters_text else ""
            raise SystemExit(
                "No microsector samples remained after applying the requested filters"
                f"{context}: {exc}"
            ) from exc

    requested_operators = [
        normalize_structural_operator_identifier(str(value))
        for value in args.operators
        if str(value).strip()
    ]
    requested_operators = list(dict.fromkeys(requested_operators))
    if not requested_operators:
        raise SystemExit("At least one operator identifier must be supplied")

    overrides: dict[str, list[ParameterSet]] = {}
    if args.operator_grid:
        override_path = Path(args.operator_grid).expanduser().resolve()
        LOGGER.info("Loading operator search space overrides from %s", override_path)
        overrides = _load_operator_grid(override_path)

    defaults = _default_operator_grid()
    operator_grid: dict[str, list[ParameterSet]] = {}
    missing: list[str] = []
    for identifier in requested_operators:
        if identifier in overrides:
            operator_grid[identifier] = list(overrides[identifier])
        elif identifier in defaults:
            operator_grid[identifier] = list(defaults[identifier])
        else:
            missing.append(identifier)

    if missing:
        raise SystemExit(
            "No parameter search space available for operators: "
            + ", ".join(sorted(missing))
        )

    extra_overrides = sorted(set(overrides) - set(operator_grid))
    if extra_overrides:
        LOGGER.info(
            "Ignoring overrides for operators not requested: %s",
            ", ".join(extra_overrides),
        )

    LOGGER.info(
        "Calibrating operators: %s",
        ", ".join(sorted(operator_grid.keys())),
    )

    LOGGER.info("Evaluating detectors across %d microsectors", len(samples))
    grouped, invalid_pairs = _group_combinations(samples)
    if invalid_pairs:
        formatted = ", ".join(
            f"({car}, {compound})" for car, compound in sorted(invalid_pairs)
        )
        raise SystemExit(
            "The labelled dataset includes unsupported car/tyre compound combinations: "
            + formatted
        )
    fold_assignments = _build_folds(samples, args.kfold)

    all_results: list[EvaluationResult] = []
    best_selections: list[EvaluationResult] = []

    for combination, combo_samples in sorted(grouped.items()):
        LOGGER.info(
            "Processing combination class=%s car=%s compound=%s (%d samples)",
            combination[0],
            combination[1],
            combination[2],
            len(combo_samples),
        )
        operators_in_combo: set[str] = set()
        for sample in combo_samples:
            for key in sample.labels.keys():
                identifier = normalize_structural_operator_identifier(str(key))
                if identifier in operator_grid:
                    operators_in_combo.add(identifier)
        for operator_id in sorted(operators_in_combo):
            candidates: list[EvaluationResult] = []
            for parameter_set in operator_grid[operator_id]:
                result = _evaluate_detector(
                    operator_id,
                    combo_samples,
                    parameter_set,
                    fold_assignments=fold_assignments,
                )
                if result is None:
                    continue
                candidates.append(result)
                all_results.append(result)
            if not candidates:
                LOGGER.warning(
                    "No successful evaluations for operator '%s' under combination %s",
                    operator_id,
                    combination,
                )
                continue
            selection = _select_best(candidates, fp_per_minute_max=args.fp_per_min_max)
            if selection:
                best_selections.append(selection)

    if not best_selections:
        raise SystemExit("No detector selections satisfied the evaluation constraints")

    best_params_path = output_dir / "best_params.yaml"
    LOGGER.info("Writing best parameter overrides to %s", best_params_path)
    _materialise_best_params(best_selections, output_path=best_params_path)

    curves_dir = output_dir / "curves"
    LOGGER.info("Writing calibration curves to %s", curves_dir)
    _write_curves_csv(all_results, output_dir=curves_dir)

    confusion_dir = output_dir / "confusion"
    LOGGER.info("Writing confusion matrices to %s", confusion_dir)
    _write_confusion_csv(best_selections, output_dir=confusion_dir)

    report_path = output_dir / "report.md"
    LOGGER.info("Writing summary report to %s", report_path)
    _write_report(best_selections, output_path=report_path)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate TNFR operator detectors")
    parser.add_argument("--raf-root", required=True, help="Directory containing RAF telemetry captures")
    parser.add_argument("--labels", required=True, help="Path to the labelled microsector artefact")
    parser.add_argument(
        "--operators",
        nargs="+",
        required=True,
        help=(
            "Structural operator identifiers to calibrate (e.g. NAV EN). "
            "Packaged search spaces are used unless --operator-grid overrides them."
        ),
    )
    parser.add_argument(
        "--operator-grid",
        help=(
            "Optional parameter grid file providing custom search spaces. "
            "When supplied it overrides the packaged defaults for the listed operators."
        ),
    )
    parser.add_argument("--out", required=True, help="Output directory for calibration artefacts")
    parser.add_argument(
        "--cars",
        nargs="*",
        help="Optional car filter (one or more identifiers). Calibration aborts if no samples remain.",
    )
    parser.add_argument(
        "--compounds",
        nargs="*",
        help="Optional compound filter (one or more identifiers). Calibration aborts if no samples remain.",
    )
    parser.add_argument(
        "--tracks",
        nargs="*",
        help="Optional track filter (one or more identifiers). Calibration aborts if no samples remain.",
    )
    parser.add_argument("--kfold", type=int, default=5, help="Number of cross-validation folds (grouped by track)")
    parser.add_argument(
        "--fp_per_min_max",
        type=float,
        default=0.5,
        help="Maximum allowed false positives per minute when ranking configurations",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    calibrate_detectors(args)


if __name__ == "__main__":
    main()

