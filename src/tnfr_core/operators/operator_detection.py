"""Detection utilities for on-track operator events.

Each detection routine analyses a windowed sequence of
:class:`~tnfr_core.operators.interfaces.SupportsTelemetrySample` objects and yields
event descriptors when the observed behaviour exceeds the
configured thresholds.  The detectors are intentionally lightweight so that
they can be executed on every microsector without adding measurable overhead
to the orchestration pipeline.
"""

from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path
from statistics import fmean, mean, pstdev
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from tnfr_lfs.resources import data_root, pack_root

from tnfr_core.config.loader import get_params, load_detection_config
from tnfr_core.operators.interfaces import SupportsTelemetrySample
from tnfr_core.operators.structural_time import compute_structural_timestamps

__all__ = [
    "OperatorEvent",
    "STRUCTURAL_OPERATOR_LABELS",
    "canonical_operator_label",
    "normalize_structural_operator_identifier",
    "silence_event_payloads",
    "detect_al",
    "detect_oz",
    "detect_il",
    "detect_silence",
    "detect_nav",
    "detect_en",
    "detect_um",
    "detect_ra",
    "detect_val",
    "detect_nul",
    "detect_thol",
    "detect_zhir",
    "detect_remesh",
]


STRUCTURAL_OPERATOR_LABELS: Mapping[str, str] = {
    "AL": "Support",
    "EN": "Reception",
    "IL": "Coherence",
    "OZ": "Dissonance",
    "UM": "Coupling",
    "RA": "Propagation",
    "SILENCE": "Structural silence",
    "VAL": "Amplification",
    "NUL": "Contraction",
    "THOL": "Auto-organisation",
    "ZHIR": "Transformation",
    "NAV": "Transition",
    "REMESH": "Remeshing",
}

_STRUCTURAL_IDENTIFIER_ALIASES: Mapping[str, str] = {
    "A'L": "AL",
    "A\u2019L": "AL",
    "E'N": "EN",
    "E\u2019N": "EN",
    "I'L": "IL",
    "I\u2019L": "IL",
    "O'Z": "OZ",
    "O\u2019Z": "OZ",
    "U'M": "UM",
    "U\u2019M": "UM",
    "R'A": "RA",
    "R\u2019A": "RA",
    "SH'A": "SILENCE",
    "SH\u2019A": "SILENCE",
    "SHA": "SILENCE",
    "VA'L": "VAL",
    "VA\u2019L": "VAL",
    "NU'L": "NUL",
    "NU\u2019L": "NUL",
    "T'HOL": "THOL",
    "T\u2019HOL": "THOL",
    "AUTOORGANISATION": "THOL",
    "AUTO ORGANISATION": "THOL",
    "AUTO-ORGANISATION": "THOL",
    "AUTOORGANIZATION": "THOL",
    "AUTO ORGANIZATION": "THOL",
    "AUTO-ORGANIZATION": "THOL",
    "AUTOORGANIZACION": "THOL",
    "AUTO ORGANIZACION": "THOL",
    "AUTO-ORGANIZACION": "THOL",
    "AUTOORGANIZACIÓN": "THOL",
    "AUTO ORGANIZACIÓN": "THOL",
    "AUTO-ORGANIZACIÓN": "THOL",
    "Z'HIR": "ZHIR",
    "Z\u2019HIR": "ZHIR",
    "NA'V": "NAV",
    "NA\u2019V": "NAV",
    "NAV": "NAV",
    "TRANSITION": "NAV",
    "TRANSICION": "NAV",
    "TRANSICIÓN": "NAV",
    "REMESH": "REMESH",
    "RE'MESH": "REMESH",
    "RE\u2019MESH": "REMESH",
}

try:
    if isinstance(STRUCTURAL_OPERATOR_LABELS, Mapping):
        labels = dict(STRUCTURAL_OPERATOR_LABELS)
        labels.setdefault("NAV", "Transition")
        labels.setdefault("THOL", "Auto-organisation")
        STRUCTURAL_OPERATOR_LABELS = labels
    else:
        STRUCTURAL_OPERATOR_LABELS = tuple(  # type: ignore[assignment]
            sorted(set(STRUCTURAL_OPERATOR_LABELS) | {"NAV", "THOL"})
        )
except Exception:
    pass

_DEPRECATED_STRUCTURAL_IDENTIFIER_ALIASES: Mapping[str, str] = {
    "SILENCIO": "SILENCE",
}


def normalize_structural_operator_identifier(identifier: str) -> str:
    """Return the canonical structural identifier for ``identifier``."""

    if not isinstance(identifier, str):
        return str(identifier)
    key = identifier.upper()
    if key in _DEPRECATED_STRUCTURAL_IDENTIFIER_ALIASES:
        warnings.warn(
            "The 'SILENCIO' structural identifier is deprecated; use 'SILENCE' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        key = _DEPRECATED_STRUCTURAL_IDENTIFIER_ALIASES[key]
    return _STRUCTURAL_IDENTIFIER_ALIASES.get(key, key)


def canonical_operator_label(identifier: str) -> str:
    """Return the canonical structural label for an operator identifier."""

    if not isinstance(identifier, str):
        return str(identifier)
    key = normalize_structural_operator_identifier(identifier)
    return STRUCTURAL_OPERATOR_LABELS.get(key, identifier)


def silence_event_payloads(
    events: Mapping[str, Sequence[Mapping[str, object]] | None] | None,
) -> Tuple[Mapping[str, object], ...]:
    """Return all silence payloads, accepting case-insensitive identifiers."""

    if not events:
        return ()

    collected: List[Mapping[str, object]] = []
    for name, payload in events.items():
        if normalize_structural_operator_identifier(name) != "SILENCE":
            continue
        if not payload:
            continue
        if isinstance(payload, SequenceABC) and not isinstance(payload, Mapping):
            collected.extend(payload)  # type: ignore[list-item]
        else:
            collected.append(payload)  # type: ignore[arg-type]
    return tuple(collected)


Number = float
DeltaSample = Union[Number, Mapping[str, Number]]


def _load_detection_table() -> Mapping[str, Any]:
    """Return the cached detection override table for the active pack root."""

    root = pack_root()
    return _load_detection_table_for_pack_root(str(root))


@lru_cache(maxsize=None)
def _load_detection_table_for_pack_root(root: str) -> Mapping[str, Any]:
    """Return detection overrides resolved for ``root``."""

    return load_detection_config(pack_root=Path(root))


# Preserve cache management API for tests and callers that relied on the
# previous :func:`functools.lru_cache` decorated helper.
_load_detection_table.cache_clear = _load_detection_table_for_pack_root.cache_clear  # type: ignore[attr-defined]
_load_detection_table.cache_info = _load_detection_table_for_pack_root.cache_info  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def _load_lfs_car_classes() -> Mapping[str, str]:
    """Return a mapping of car model abbreviations to LFS classes."""

    mapping: Dict[str, str] = {}

    try:
        cars_dir = data_root() / "cars"
    except Exception:  # pragma: no cover - defensive fallback
        return mapping

    if not cars_dir.exists():
        return mapping

    for manifest in sorted(cars_dir.glob("*.toml")):
        try:
            with manifest.open("rb") as buffer:
                payload = tomllib.load(buffer)
        except Exception:  # pragma: no cover - defensive fallback
            continue

        car_model = payload.get("abbrev") or manifest.stem
        lfs_class = payload.get("lfs_class")
        if not car_model or not lfs_class:
            continue

        mapping[str(car_model).strip().upper()] = str(lfs_class)

    return mapping


def _derive_lfs_class(car_model: str | None) -> str | None:
    """Return the LFS class inferred from ``car_model`` when available."""

    if not car_model:
        return None

    lookup = _load_lfs_car_classes()
    return lookup.get(str(car_model).strip().upper())


def _sequence_metadata(
    records: Sequence[SupportsTelemetrySample]
    | SupportsTelemetrySample
    | Mapping[str, object]
    | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Extract (car_class, car_model, track_name, tyre_compound) metadata."""

    car_class: str | None = None
    car_model: str | None = None
    track_name: str | None = None
    tyre_compound: str | None = None

    if not records:
        return car_class, car_model, track_name, tyre_compound

    if isinstance(records, MappingABC):
        candidates = [records]
    elif isinstance(records, SequenceABC) and not isinstance(records, (str, bytes, bytearray)):
        candidates = records
    else:
        candidates = [records]

    for sample in candidates:
        if sample is None:
            continue
        if isinstance(sample, MappingABC):
            getter = sample.get  # type: ignore[assignment]
        else:
            getter = None
        if car_class is None:
            car_class = (
                getter("car_class") if getter else getattr(sample, "car_class", None)
            )
        if car_model is None:
            car_model = (
                getter("car_model") if getter else getattr(sample, "car_model", None)
            )
        if track_name is None:
            track_name = (
                getter("track_name") if getter else getattr(sample, "track_name", None)
            )
        if tyre_compound is None:
            tyre_compound = (
                getter("tyre_compound")
                if getter
                else getattr(sample, "tyre_compound", None)
            )
        if all(value is not None for value in (car_model, track_name, tyre_compound)):
            break

    if car_class is None and car_model:
        derived = _derive_lfs_class(car_model)
        if derived is not None:
            car_class = derived

    return car_class, car_model, track_name, tyre_compound


def _resolve_detection_parameters(
    operator: str,
    *,
    defaults: Mapping[str, Any],
    provided: Mapping[str, Any],
    metadata_source: Sequence[SupportsTelemetrySample]
    | SupportsTelemetrySample
    | Mapping[str, object]
    | None,
) -> Dict[str, Any]:
    """Merge defaults, configuration overrides, and explicit parameters."""

    resolved = dict(defaults)
    table = _load_detection_table()
    config: Mapping[str, Any] | None = None
    if isinstance(table, MappingABC):
        operator_table = table.get(operator)
        if isinstance(operator_table, MappingABC):
            config = operator_table
        else:
            shared_keys = ("defaults", "tracks", "classes", "cars", "compounds")
            shared_payload = {
                key: table[key]
                for key in shared_keys
                if key in table
            }
            if shared_payload:
                config = shared_payload
    else:
        config = None

    if config is not None:
        car_class, car_model, track_name, tyre_compound = _sequence_metadata(metadata_source)
        overrides = get_params(
            config,
            car_class=car_class,
            car_model=car_model,
            track_name=track_name,
            tyre_compound=tyre_compound,
        )
        resolved.update(overrides)

    for key, value in provided.items():
        resolved[key] = value

    return resolved


def _with_detection_overrides(
    operator: str,
    *,
    metadata_argument: str | None,
    parameters: Sequence[str],
):
    """Decorator injecting detection overrides for ``operator``."""

    def decorator(func):  # type: ignore[no-untyped-def]
        signature = inspect.signature(func)
        defaults: Dict[str, Any] = {}
        for name in parameters:
            param = signature.parameters.get(name)
            if param is None or param.default is inspect._empty:
                raise ValueError(
                    f"{func.__name__} missing default value for parameter '{name}'"
                )
            defaults[name] = param.default

        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            metadata_source: (
                Sequence[SupportsTelemetrySample]
                | SupportsTelemetrySample
                | Mapping[str, object]
                | None
            ) = None
            if metadata_argument:
                metadata_source = bound.arguments.get(metadata_argument)

            provided = {name: kwargs[name] for name in parameters if name in kwargs}
            resolved = _resolve_detection_parameters(
                operator,
                defaults=defaults,
                provided=provided,
                metadata_source=metadata_source,
            )

            for name in parameters:
                if name in resolved:
                    bound.arguments[name] = resolved[name]

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def _is_mapping_sample(sample: DeltaSample) -> bool:
    return hasattr(sample, "keys") and hasattr(sample, "items")


def _rel_tol(target: float, eps: float) -> float:
    """Return relative tolerance (eps * |target|) with absolute fallback."""

    return max(eps, eps * abs(target))


def _closeness(value: float, target: float, eps: float) -> float:
    """Return closeness score in [0, 1] with linear drop outside tolerance."""

    tol = _rel_tol(target, eps)
    d = abs(value - target)
    if d <= tol:
        return max(0.0, 1.0 - (d / (tol + 1e-12)) * 0.5)
    return 0.0


@_with_detection_overrides(
    "detect_nav",
    metadata_argument="metadata",
    parameters=("window", "eps"),
)
def detect_nav(
    series: Sequence[DeltaSample],
    *,
    nu_f: Union[float, Mapping[str, float], None],
    window: int = 3,
    eps: float = 1e-3,
    metadata: Sequence[SupportsTelemetrySample]
    | SupportsTelemetrySample
    | Mapping[str, object]
    | None = None,
) -> List[Dict[str, Any]]:
    """Detect sustained ΔNFR ≈ νf (NA'V) runs."""

    events: List[Dict[str, Any]] = []
    if not series or window <= 0:
        return events

    if nu_f is None:
        if _is_mapping_sample(series[0]):
            raise ValueError("nu_f por nodo requerido para series por nodo")
        from statistics import median

        nu_f = float(median(abs(x) for x in series)) if series else 0.0

    def in_phase_and_score(sample: DeltaSample) -> float:
        if _is_mapping_sample(sample):
            assert isinstance(nu_f, Mapping)
            subscores = []
            for k, v in sample.items():
                target = float(nu_f.get(k, 0.0))
                subscores.append(_closeness(float(v), target, eps))
            return sum(subscores) / len(subscores) if subscores else 0.0
        assert not isinstance(nu_f, Mapping)
        return _closeness(float(sample), float(nu_f), eps)

    n = len(series)
    i = 0
    while i < n:
        score = in_phase_and_score(series[i])
        if score > 0.0:
            j = i + 1
            scores = [score]
            while j < n:
                s = in_phase_and_score(series[j])
                if s <= 0.0:
                    break
                scores.append(s)
                j += 1
            run_len = j - i
            if run_len >= window:
                severity = sum(scores) / len(scores)
                events.append({"severity": float(severity), "duration": int(run_len)})
            i = j
        else:
            i += 1
    return events


def _clean_numeric(value: object) -> float | None:
    """Return ``value`` as a finite float or ``None`` when unavailable."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _clean_sequence(values: Sequence[object]) -> List[float]:
    """Return all finite floats extracted from ``values``."""

    cleaned: List[float] = []
    for value in values:
        numeric = _clean_numeric(value)
        if numeric is not None:
            cleaned.append(numeric)
    return cleaned


def _span(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def _sample_times(records: Sequence[SupportsTelemetrySample]) -> List[float]:
    times: List[float] = []
    for index, record in enumerate(records):
        timestamp = _clean_numeric(getattr(record, "timestamp", index))
        if timestamp is None:
            timestamp = float(index)
        times.append(timestamp)
    return times


def _safe_dt(times: Sequence[float], index: int) -> float:
    if index <= 0:
        return 0.0
    dt = float(times[index] - times[index - 1])
    return dt if dt > 0.0 else 0.0


def _epi_norm(sample: SupportsTelemetrySample) -> float:
    nfr = _clean_numeric(getattr(sample, "nfr", 0.0)) or 0.0
    si = _clean_numeric(getattr(sample, "si", 0.0)) or 0.0
    return float(math.hypot(nfr, si))


def _epi_concentration(sample: SupportsTelemetrySample) -> float:
    nfr = abs(_clean_numeric(getattr(sample, "nfr", 0.0)) or 0.0)
    si = abs(_clean_numeric(getattr(sample, "si", 0.0)) or 0.0)
    if not math.isfinite(nfr):
        nfr = 0.0
    if not math.isfinite(si):
        si = 0.0
    total = nfr + si
    if total <= 1e-9:
        return 0.0
    return max(nfr, si) / total


def _phase_features(sample: SupportsTelemetrySample) -> List[float]:
    features: List[float] = []
    for attribute in (
        "mu_eff_front",
        "mu_eff_rear",
        "slip_angle",
        "slip_ratio",
        "si",
    ):
        numeric = _clean_numeric(getattr(sample, attribute, None))
        if numeric is None or not math.isfinite(numeric):
            features.append(math.nan)
        else:
            features.append(float(numeric))
    return features


def _phase_metric(sample: SupportsTelemetrySample) -> float:
    metrics = [value for value in _phase_features(sample) if math.isfinite(value)]
    return fmean(metrics) if metrics else 0.0


def _count_active_support_nodes(
    sample: SupportsTelemetrySample, *, load_min: float
) -> int:
    loads: List[float] = []
    for attribute in (
        "wheel_load_fl",
        "wheel_load_fr",
        "wheel_load_rl",
        "wheel_load_rr",
    ):
        numeric = _clean_numeric(getattr(sample, attribute, None))
        if numeric is not None and math.isfinite(numeric):
            loads.append(numeric)
    if not loads:
        for attribute in ("vertical_load_front", "vertical_load_rear"):
            numeric = _clean_numeric(getattr(sample, attribute, None))
            if numeric is not None and math.isfinite(numeric):
                loads.append(numeric * 0.5)
    active = 0
    for load in loads:
        if load >= load_min:
            active += 1
    return active


def _integrate_series(
    values: Sequence[float],
    times: Sequence[float],
    start_index: int,
    end_index: int,
) -> float:
    if end_index <= start_index:
        return 0.0
    integral = 0.0
    for index in range(start_index + 1, end_index + 1):
        dt = float(times[index] - times[index - 1])
        if dt <= 0.0:
            continue
        left = values[index - 1]
        right = values[index]
        integral += 0.5 * (left + right) * dt
    return float(integral)


def _sliding_window_max(values: np.ndarray, window: int) -> np.ndarray:
    """Return the rolling maximum over the trailing ``window`` samples."""

    length = int(values.size)
    if length == 0:
        return values.copy()

    window = max(1, int(window))
    if length == 1 or window == 1:
        return values.copy()

    if length < window:
        return np.maximum.accumulate(values)

    prefix = np.maximum.accumulate(values[: window - 1])
    trailing = sliding_window_view(values, window_shape=window)
    maxima = trailing.max(axis=-1)
    return np.concatenate((prefix, maxima))


def _compute_first_derivative(
    values: Sequence[float], times: Sequence[float]
) -> List[float]:
    derivative: List[float] = [0.0] * len(values)
    for index in range(1, len(values)):
        dt = float(times[index] - times[index - 1])
        if dt <= 0.0:
            derivative[index] = 0.0
            continue
        derivative[index] = (values[index] - values[index - 1]) / dt
    return derivative


def _compute_second_derivative(
    values: Sequence[float], times: Sequence[float]
) -> List[float]:
    first = _compute_first_derivative(values, times)
    second: List[float] = [0.0] * len(values)
    for index in range(2, len(values)):
        dt = float(times[index] - times[index - 1])
        if dt <= 0.0:
            continue
        second[index] = (first[index] - first[index - 1]) / dt
    return second


def _band_power(
    values: Sequence[float],
    times: Sequence[float],
    nu_band: Tuple[float, float],
) -> float:
    length = len(values)
    if length < 3:
        return 0.0
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        return 0.0
    sample_rate = (length - 1) / duration
    if sample_rate <= 0.0:
        return 0.0

    window = np.array(values, dtype=float)
    if not np.any(np.isfinite(window)):
        return 0.0
    window = np.nan_to_num(window - np.nanmean(window))
    spectrum = np.fft.rfft(window)
    frequencies = np.fft.rfftfreq(length, d=1.0 / sample_rate)
    lower, upper = nu_band
    mask = (frequencies >= lower) & (frequencies <= upper)
    if not np.any(mask):
        return 0.0
    power = np.abs(spectrum[mask]) ** 2
    return float(np.sum(power))


def _phase_lag(times: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return float("inf")
    a_index = _argmax_abs(a)
    b_index = _argmax_abs(b)
    if a_index is None or b_index is None:
        return float("inf")
    return abs(float(times[a_index] - times[b_index]))


def _argmax_abs(values: Sequence[float]) -> int | None:
    best_index: int | None = None
    best_value = 0.0
    for index, value in enumerate(values):
        try:
            numeric = abs(float(value))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        if best_index is None or numeric > best_value:
            best_value = numeric
            best_index = index
    return best_index


def _autocorrelation(series: Sequence[float], lag: int) -> float:
    if lag <= 0 or lag >= len(series):
        return 0.0
    x = np.asarray(series, dtype=float)
    if x.size < (lag + 2):
        return 0.0
    x = np.nan_to_num(x)
    a = x[:-lag]
    b = x[lag:]
    if a.size < 2 or b.size < 2:
        return 0.0
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    numerator = float(np.dot(a_centered, b_centered))
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


@_with_detection_overrides(
    "detect_en",
    metadata_argument="records",
    parameters=("window", "psi_threshold", "epi_norm_max"),
)
def detect_en(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 6,
    psi_threshold: float = 0.9,
    epi_norm_max: float = 120.0,
) -> List[Mapping[str, float | str | int]]:
    """Detect reception (EN) when ψ flux excites a rising low-norm EPI.

    The detector integrates a ψ-like flux assembled from suspension velocities,
    steering rate, yaw rate, and vertical micro-vibrations across a rolling
    ``window``.  An event is emitted whenever the integrated flux exceeds
    ``psi_threshold`` while the |EPI| proxy remains
    below ``epi_norm_max`` and is non-decreasing across the same window.
    ``NaN`` samples are ignored so that sparse telemetry does not trigger false
    positives.
    """

    if len(records) < 2:
        return []

    window = max(2, int(window))
    xp = np

    times = xp.asarray(_sample_times(records), dtype=float)
    dt = xp.concatenate((xp.zeros(1, dtype=times.dtype), xp.diff(times)))
    dt = xp.where(dt > 0.0, dt, 0.0)

    susp_front = xp.asarray(
        [
            abs(_clean_numeric(getattr(record, "suspension_velocity_front", 0.0)) or 0.0)
            for record in records
        ],
        dtype=float,
    )
    susp_rear = xp.asarray(
        [
            abs(_clean_numeric(getattr(record, "suspension_velocity_rear", 0.0)) or 0.0)
            for record in records
        ],
        dtype=float,
    )
    steer_series = xp.asarray(
        [
            _clean_numeric(getattr(record, "steer", 0.0)) or 0.0
            for record in records
        ],
        dtype=float,
    )
    yaw_rate = xp.asarray(
        [
            abs(_clean_numeric(getattr(record, "yaw_rate", 0.0)) or 0.0)
            for record in records
        ],
        dtype=float,
    )
    vertical_load = xp.asarray(
        [
            (
                float(value)
                if (value := _clean_numeric(getattr(record, "vertical_load", math.nan)))
                is not None
                else math.nan
            )
            for record in records
        ],
        dtype=float,
    )
    epi_norms = xp.asarray([_epi_norm(record) for record in records], dtype=float)

    safe_dt = xp.where(dt > 0.0, dt, 1.0)
    steer_diff = xp.concatenate((xp.zeros(1, dtype=steer_series.dtype), xp.diff(steer_series)))
    steer_rate = xp.where(dt > 0.0, xp.abs(steer_diff) / safe_dt, 0.0)

    vertical_prev = xp.concatenate((vertical_load[:1], vertical_load[:-1]))
    vertical_valid = xp.isfinite(vertical_load)
    vertical_prev_valid = xp.concatenate((xp.array([False]), vertical_valid[:-1]))
    vertical_delta = xp.abs(vertical_load - vertical_prev)
    micro_vibration = xp.where(
        vertical_valid & vertical_prev_valid & (dt > 0.0),
        vertical_delta / safe_dt,
        0.0,
    )

    psi_flux = (0.5 * (susp_front + susp_rear)) + steer_rate + yaw_rate + micro_vibration

    dt_segments = xp.diff(times)
    if dt_segments.size:
        dt_segments = xp.where(dt_segments > 0.0, dt_segments, 0.0)
    flux_avg = 0.5 * (psi_flux[1:] + psi_flux[:-1])
    segment_integral = flux_avg * dt_segments
    psi_cumulative = xp.concatenate((xp.zeros(1, dtype=psi_flux.dtype), xp.cumsum(segment_integral)))

    length = int(len(records))
    indices = xp.arange(length)
    start_indices = xp.maximum(indices - (window - 1), 0).astype(int)
    psi_integral = psi_cumulative[indices] - psi_cumulative[start_indices]
    start_times = times[start_indices]
    window_duration = xp.where(indices > start_indices, times - start_times, 0.0)
    epi_start = epi_norms[start_indices]
    epi_end = epi_norms
    epi_peak = _sliding_window_max(epi_norms, window)
    epi_rising = epi_end >= (epi_start - 1e-6)

    meets_threshold = (
        (psi_integral >= psi_threshold)
        & (epi_peak <= epi_norm_max)
        & epi_rising
    )

    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_integral = 0.0
    best_metrics: Dict[str, float] = {}

    for index in range(length):
        start_index = int(start_indices[index])
        psi_value = float(psi_integral[index])
        duration_value = float(window_duration[index])
        epi_start_value = float(epi_start[index])
        epi_end_value = float(epi_end[index])
        epi_peak_value = float(epi_peak[index])

        metrics = {
            "psi_integral": psi_value,
            "window_duration": duration_value,
            "epi_norm_start": epi_start_value,
            "epi_norm_end": epi_end_value,
            "epi_norm_peak": epi_peak_value,
        }

        if bool(meets_threshold[index]):
            if active_start is None:
                active_start = start_index
                peak_integral = psi_value
                best_metrics = metrics
            else:
                if psi_value >= peak_integral:
                    best_metrics = metrics
                peak_integral = max(peak_integral, psi_value)
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label("EN"),
                records,
                active_start,
                index - 1,
                peak_integral,
                psi_threshold,
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    "psi_threshold": float(psi_threshold),
                    "epi_norm_max": float(epi_norm_max),
                }
            )
            events.append(payload)
            active_start = None
            peak_integral = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label("EN"),
            records,
            active_start,
            len(records) - 1,
            peak_integral,
            psi_threshold,
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                "psi_threshold": float(psi_threshold),
                "epi_norm_max": float(epi_norm_max),
            }
        )
        events.append(payload)

    return events


@_with_detection_overrides(
    "detect_um",
    metadata_argument="records",
    parameters=("window", "rho_min", "phase_max", "min_duration"),
)
def detect_um(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 8,
    rho_min: float = 0.65,
    phase_max: float = 0.25,
    min_duration: float = 0.35,
) -> List[Mapping[str, float | str | int]]:
    """Detect coupling (UM) when slip responses align with yaw dynamics.

    Pairwise coupling across slip ratios, slip angles, steering input, and yaw
    rate is evaluated on sliding windows of size ``window``.  An opportunity is
    emitted when the strongest correlation exceeds ``rho_min`` while the
    resulting phase lag stays within ``phase_max`` for at least ``min_duration``
    seconds.
    """

    if len(records) < 2:
        return []

    window = max(3, int(window))
    from tnfr_core.operators.operators import pairwise_coupling_operator

    times = _sample_times(records)

    def _series(attribute: str, *, absolute: bool = False) -> List[float]:
        values: List[float] = []
        for record in records:
            numeric = _clean_numeric(getattr(record, attribute, 0.0))
            value = float(numeric) if numeric is not None else 0.0
            if absolute:
                value = abs(value)
            if not math.isfinite(value):
                value = 0.0
            values.append(value)
        return values

    slip_ratio_series = _series('slip_ratio', absolute=True)
    slip_angle_series = _series('slip_angle', absolute=True)
    steer_series = _series('steer')
    yaw_series = _series('yaw_rate')

    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_correlation = 0.0
    best_metrics: Dict[str, float] = {}

    for index in range(len(records)):
        start_index = max(0, index - window + 1)
        length = index - start_index + 1
        if length < 3:
            continue

        window_times = times[start_index : index + 1]
        ratio_window = slip_ratio_series[start_index : index + 1]
        angle_window = slip_angle_series[start_index : index + 1]
        steer_window = steer_series[start_index : index + 1]
        yaw_window = yaw_series[start_index : index + 1]

        correlations = pairwise_coupling_operator(
            {
                'slip_ratio': ratio_window,
                'slip_angle': angle_window,
                'steer': steer_window,
                'yaw': yaw_window,
            },
            pairs=(
                ('slip_ratio', 'yaw'),
                ('slip_angle', 'yaw'),
                ('steer', 'yaw'),
            ),
        )
        positive_correlations = [
            value
            for value in correlations.values()
            if math.isfinite(value) and value >= 0.0
        ]
        max_corr = max(positive_correlations) if positive_correlations else 0.0

        phase_lag = _phase_lag(window_times, steer_window, yaw_window)
        duration = window_times[-1] - window_times[0]
        meets_threshold = (
            positive_correlations
            and max_corr >= rho_min
            and math.isfinite(phase_lag)
            and phase_lag <= phase_max
            and duration > 0.0
        )

        metrics = {
            'max_coupling': float(max_corr),
            'phase_lag': float(phase_lag),
            'window_duration': float(duration),
            'slip_ratio_yaw': float(correlations.get('slip_ratio↔yaw', 0.0)),
            'slip_angle_yaw': float(correlations.get('slip_angle↔yaw', 0.0)),
            'steer_yaw': float(correlations.get('steer↔yaw', 0.0)),
        }

        if meets_threshold:
            if active_start is None:
                active_start = start_index
                peak_correlation = max_corr
                best_metrics = metrics
            else:
                if max_corr >= peak_correlation:
                    best_metrics = metrics
                peak_correlation = max(peak_correlation, max_corr)
        elif active_start is not None:
            end_index = index - 1
            duration_total = float(times[end_index] - times[active_start]) if end_index > active_start else 0.0
            if duration_total >= min_duration:
                event = _finalise_event(
                    canonical_operator_label('UM'),
                    records,
                    active_start,
                    end_index,
                    peak_correlation,
                    rho_min,
                )
                payload = event.as_mapping()
                payload.update(best_metrics)
                payload.update(
                    {
                        'rho_min': float(rho_min),
                        'phase_max': float(phase_max),
                        'min_duration': float(min_duration),
                    }
                )
                events.append(payload)
            active_start = None
            peak_correlation = 0.0
            best_metrics = {}

    if active_start is not None:
        end_index = len(records) - 1
        duration_total = float(times[end_index] - times[active_start]) if end_index > active_start else 0.0
        if duration_total >= min_duration:
            event = _finalise_event(
                canonical_operator_label('UM'),
                records,
                active_start,
                end_index,
                peak_correlation,
                rho_min,
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    'rho_min': float(rho_min),
                    'phase_max': float(phase_max),
                    'min_duration': float(min_duration),
                }
            )
            events.append(payload)

    return events


@_with_detection_overrides(
    "detect_ra",
    metadata_argument="records",
    parameters=("window", "nu_band", "si_min", "delta_nfr_max", "k_min"),
)
def detect_ra(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 12,
    nu_band: Tuple[float, float] = (1.0, 3.0),
    si_min: float = 0.55,
    delta_nfr_max: float = 12.0,
    k_min: int = 2,
) -> List[Mapping[str, float | str | int]]:
    """Detect propagation (RA) bursts sustained around a natural frequency band.

    The detector analyses ΔNFR spectra inside ``nu_band`` using a lightweight
    FFT projection.  Propagation is confirmed when the band power remains high
    while the mean sense index exceeds ``si_min`` and the ΔNFR dispersion stays
    below ``delta_nfr_max`` for at least ``k_min`` consecutive windows.
    """

    if len(records) < max(3, window):
        return []

    window = max(3, int(window))
    k_min = max(1, int(k_min))
    times = _sample_times(records)

    nfr_series: List[float] = []
    si_series: List[float] = []
    for record in records:
        nfr_value = _clean_numeric(getattr(record, 'nfr', 0.0))
        si_value = _clean_numeric(getattr(record, 'si', 0.0))
        nfr_series.append(float(nfr_value) if nfr_value is not None and math.isfinite(nfr_value) else 0.0)
        si_series.append(float(si_value) if si_value is not None and math.isfinite(si_value) else 0.0)

    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    first_window_start: int | None = None
    sustained_windows = 0
    peak_power = 0.0
    baseline_power = 0.0
    best_metrics: Dict[str, float] = {}
    baseline_metrics: Dict[str, float] = {}

    for index in range(len(records)):
        start_index = max(0, index - window + 1)
        length = index - start_index + 1
        if length < window:
            continue

        window_times = times[start_index : index + 1]
        nfr_window = nfr_series[start_index : index + 1]
        si_window = si_series[start_index : index + 1]

        band_power = _band_power(nfr_window, window_times, nu_band)
        si_mean = fmean(si_window) if si_window else 0.0
        dispersion = pstdev(nfr_window) if len(nfr_window) > 1 else 0.0

        meets_threshold = (
            band_power > 0.0
            and si_mean >= si_min
            and dispersion <= delta_nfr_max
        )

        metrics = {
            'band_power': float(band_power),
            'si_mean': float(si_mean),
            'delta_nfr_dispersion': float(dispersion),
        }

        if meets_threshold:
            sustained_windows += 1
            if first_window_start is None:
                first_window_start = start_index
            if sustained_windows >= k_min:
                if active_start is None:
                    active_start = first_window_start
                    peak_power = band_power
                    baseline_power = band_power
                    baseline_metrics = metrics
                    best_metrics = metrics
                else:
                    if band_power >= peak_power:
                        best_metrics = metrics
                    peak_power = max(peak_power, band_power)
        else:
            if active_start is not None and sustained_windows >= k_min:
                end_index = index - 1
                reference_si = max(
                    best_metrics.get('si_mean', baseline_metrics.get('si_mean', si_min)),
                    si_min,
                )
                threshold = max(baseline_power * (si_min / reference_si), 1e-9)
                event = _finalise_event(
                    canonical_operator_label('RA'),
                    records,
                    active_start,
                    end_index,
                    peak_power,
                    threshold,
                )
                payload = event.as_mapping()
                payload.update(best_metrics)
                payload.update(
                    {
                        'band_power_baseline': float(baseline_power),
                        'severity_threshold': float(threshold),
                        'peak_band_power': float(peak_power),
                        'nu_band': tuple(float(value) for value in nu_band),
                        'si_min': float(si_min),
                        'delta_nfr_max': float(delta_nfr_max),
                        'k_min': int(k_min),
                    }
                )
                events.append(payload)
            active_start = None
            first_window_start = None
            sustained_windows = 0
            peak_power = 0.0
            baseline_power = 0.0
            best_metrics = {}
            baseline_metrics = {}

    if active_start is not None and sustained_windows >= k_min:
        end_index = len(records) - 1
        reference_si = max(
            best_metrics.get('si_mean', baseline_metrics.get('si_mean', si_min)),
            si_min,
        )
        threshold = max(baseline_power * (si_min / reference_si), 1e-9)
        event = _finalise_event(
            canonical_operator_label('RA'),
            records,
            active_start,
            end_index,
            peak_power,
            threshold,
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                'band_power_baseline': float(baseline_power),
                'severity_threshold': float(threshold),
                'peak_band_power': float(peak_power),
                'nu_band': tuple(float(value) for value in nu_band),
                'si_min': float(si_min),
                'delta_nfr_max': float(delta_nfr_max),
                'k_min': int(k_min),
            }
        )
        events.append(payload)

    return events


@_with_detection_overrides(
    "detect_val",
    metadata_argument="records",
    parameters=("window", "epi_growth_min", "active_nodes_delta_min", "active_node_load_min"),
)
def detect_val(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 6,
    epi_growth_min: float = 0.4,
    active_nodes_delta_min: int = 1,
    active_node_load_min: float = 250.0,
) -> List[Mapping[str, float | str | int]]:
    """Detect amplification (VAL) phases where EPI growth recruits support nodes.

    The detector tracks the rate of change of the |EPI| proxy across the rolling
    ``window`` together with the change in active support nodes.  An event is
    emitted once the growth rate exceeds ``epi_growth_min`` while the active node
    count increases by at least ``active_nodes_delta_min``.
    """

    if len(records) < max(3, window):
        return []

    window = max(3, int(window))
    times = _sample_times(records)

    epi_norms = [_epi_norm(record) for record in records]
    active_nodes = [
        _count_active_support_nodes(record, load_min=float(active_node_load_min))
        for record in records
    ]

    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_growth = 0.0
    best_metrics: Dict[str, float] = {}

    for index in range(len(records)):
        start_index = max(0, index - window + 1)
        if index == start_index:
            continue

        duration = float(times[index] - times[start_index])
        if duration <= 0.0:
            continue

        epi_start = epi_norms[start_index]
        epi_end = epi_norms[index]
        growth_rate = (epi_end - epi_start) / duration

        nodes_start = active_nodes[start_index]
        nodes_end = active_nodes[index]
        node_delta = nodes_end - nodes_start

        meets_threshold = (
            growth_rate >= epi_growth_min
            and node_delta >= active_nodes_delta_min
        )

        metrics = {
            'epi_growth_rate': float(growth_rate),
            'epi_norm_start': float(epi_start),
            'epi_norm_end': float(epi_end),
            'active_nodes_start': int(nodes_start),
            'active_nodes_end': int(nodes_end),
            'active_nodes_delta': int(node_delta),
            'window_duration': float(duration),
        }

        if meets_threshold:
            if active_start is None:
                active_start = start_index
                peak_growth = growth_rate
                best_metrics = metrics
            else:
                if growth_rate >= peak_growth:
                    best_metrics = metrics
                peak_growth = max(peak_growth, growth_rate)
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label('VAL'),
                records,
                active_start,
                index - 1,
                peak_growth,
                epi_growth_min,
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    'epi_growth_min': float(epi_growth_min),
                    'active_nodes_delta_min': int(active_nodes_delta_min),
                    'active_node_load_min': float(active_node_load_min),
                }
            )
            events.append(payload)
            active_start = None
            peak_growth = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label('VAL'),
            records,
            active_start,
            len(records) - 1,
            peak_growth,
            epi_growth_min,
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                'epi_growth_min': float(epi_growth_min),
                'active_nodes_delta_min': int(active_nodes_delta_min),
                'active_node_load_min': float(active_node_load_min),
            }
        )
        events.append(payload)

    return events


@_with_detection_overrides(
    "detect_nul",
    metadata_argument="records",
    parameters=("window", "active_nodes_delta_max", "epi_concentration_min", "active_node_load_min"),
)
def detect_nul(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 6,
    active_nodes_delta_max: int = -1,
    epi_concentration_min: float = 0.6,
    active_node_load_min: float = 250.0,
) -> List[Mapping[str, float | str | int]]:
    """Detect contraction (NUL) by shrinking support under rising EPI concentration.

    Contraction is flagged when the active support node count decreases by at
    least ``active_nodes_delta_max`` (a negative value) while the EPI
    concentration measured across the window exceeds ``epi_concentration_min``.
    """

    if len(records) < max(3, window):
        return []

    window = max(3, int(window))
    times = _sample_times(records)

    active_nodes = [
        _count_active_support_nodes(record, load_min=float(active_node_load_min))
        for record in records
    ]
    epi_concentration = [_epi_concentration(record) for record in records]

    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_concentration = 0.0
    best_metrics: Dict[str, float] = {}

    for index in range(len(records)):
        start_index = max(0, index - window + 1)
        if index == start_index:
            continue

        nodes_start = active_nodes[start_index]
        nodes_end = active_nodes[index]
        node_delta = nodes_end - nodes_start

        concentration_window = epi_concentration[start_index : index + 1]
        concentration_peak = max(concentration_window) if concentration_window else 0.0

        meets_threshold = (
            node_delta <= active_nodes_delta_max
            and concentration_peak >= epi_concentration_min
        )

        metrics = {
            'active_nodes_start': int(nodes_start),
            'active_nodes_end': int(nodes_end),
            'active_nodes_delta': int(node_delta),
            'epi_concentration_peak': float(concentration_peak),
            'window_duration': float(times[index] - times[start_index]),
        }

        if meets_threshold:
            if active_start is None:
                active_start = start_index
                peak_concentration = concentration_peak
                best_metrics = metrics
            else:
                if concentration_peak >= peak_concentration:
                    best_metrics = metrics
                peak_concentration = max(peak_concentration, concentration_peak)
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label('NUL'),
                records,
                active_start,
                index - 1,
                peak_concentration,
                epi_concentration_min,
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    'active_nodes_delta_max': int(active_nodes_delta_max),
                    'epi_concentration_min': float(epi_concentration_min),
                    'active_node_load_min': float(active_node_load_min),
                }
            )
            events.append(payload)
            active_start = None
            peak_concentration = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label('NUL'),
            records,
            active_start,
            len(records) - 1,
            peak_concentration,
            epi_concentration_min,
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                'active_nodes_delta_max': int(active_nodes_delta_max),
                'epi_concentration_min': float(epi_concentration_min),
                'active_node_load_min': float(active_node_load_min),
            }
        )
        events.append(payload)

    return events


@_with_detection_overrides(
    "detect_thol",
    metadata_argument="records",
    parameters=("epi_accel_min", "stability_window", "stability_tolerance"),
)
def detect_thol(
    records: Sequence[SupportsTelemetrySample],
    *,
    epi_accel_min: float = 0.8,
    stability_window: float = 0.4,
    stability_tolerance: float = 0.05,
) -> List[Mapping[str, float | str | int]]:
    """Detect auto-organisation (THOL) from EPI acceleration followed by stability.

    The detector observes the second derivative of the |EPI| proxy and emits an
    event when it exceeds ``epi_accel_min`` and is followed by a period of
    duration ``stability_window`` where the first derivative remains within
    ``±stability_tolerance``.
    """

    if len(records) < 3:
        return []

    times = _sample_times(records)
    epi_norms = [_epi_norm(record) for record in records]
    first_derivative = _compute_first_derivative(epi_norms, times)
    second_derivative = _compute_second_derivative(epi_norms, times)

    events: List[Mapping[str, float | str | int]] = []
    index = 2
    while index < len(records):
        accel = second_derivative[index]
        if accel < epi_accel_min:
            index += 1
            continue

        stability_limit = times[index] + stability_window
        j = index + 1
        last_stable: int | None = None
        stable_run = 0
        triggered = False

        while j < len(records) and times[j] <= stability_limit + 1e-9:
            if abs(first_derivative[j]) <= stability_tolerance:
                stable_run += 1
                last_stable = j
                if times[last_stable] - times[index] >= stability_window:
                    start_index = max(0, index - 1)
                    end_index = last_stable
                    event = _finalise_event(
                        canonical_operator_label('THOL'),
                        records,
                        start_index,
                        end_index,
                        accel,
                        epi_accel_min,
                    )
                    payload = event.as_mapping()
                    payload.update(
                        {
                            'epi_second_derivative': float(accel),
                            'stability_duration': float(times[end_index] - times[index]),
                            'stability_tolerance': float(stability_tolerance),
                            'stable_samples': int(stable_run),
                        }
                    )
                    events.append(payload)
                    index = end_index + 1
                    triggered = True
                    break
            else:
                stable_run = 0
                last_stable = None
            j += 1

        if not triggered:
            index += 1

    return events


@_with_detection_overrides(
    "detect_zhir",
    metadata_argument="records",
    parameters=("window", "xi_min", "min_persistence", "phase_jump_min"),
)
def detect_zhir(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 8,
    xi_min: float = 0.35,
    min_persistence: float = 0.4,
    phase_jump_min: float = 0.2,
) -> List[Mapping[str, float | str | int]]:
    """Detect transformation (ZHIR) events via phase change-point analysis.

    The detector applies a bidirectional window change detector on phase metrics
    (μ regime, slip angles/ratios, Si) and confirms the event once the |EPI|
    derivative exceeds ``xi_min`` for at least ``min_persistence`` seconds.
    """

    if len(records) < max(3, window):
        return []

    window = max(4, int(window))
    half = max(1, window // 2)
    times = _sample_times(records)
    phase_vectors = np.array([_phase_features(record) for record in records], dtype=float)
    if phase_vectors.ndim == 1:
        phase_vectors = phase_vectors.reshape(len(records), -1)
    phase_metrics = [_phase_metric(record) for record in records]
    epi_norms = [_epi_norm(record) for record in records]
    epi_derivative = _compute_first_derivative(epi_norms, times)

    events: List[Mapping[str, float | str | int]] = []
    index = half
    while index < len(records) - half:
        past = phase_metrics[index - half : index]
        future = phase_metrics[index : index + half]
        if not past or not future:
            index += 1
            continue
        with np.errstate(invalid='ignore'):
            past_mean = np.nanmean(phase_vectors[index - half : index], axis=0)
            future_mean = np.nanmean(phase_vectors[index : index + half], axis=0)

        valid = np.isfinite(past_mean) & np.isfinite(future_mean)
        if not np.any(valid):
            index += 1
            continue

        diff = future_mean[valid] - past_mean[valid]
        phase_jump = float(np.linalg.norm(diff))
        derivative = abs(epi_derivative[index])

        if phase_jump >= phase_jump_min and derivative >= xi_min:
            start_index = max(0, index - half)
            end_index = index
            while (
                end_index + 1 < len(records)
                and abs(epi_derivative[end_index + 1]) >= xi_min
            ):
                end_index += 1
            duration = float(times[end_index] - times[start_index])
            if duration >= min_persistence:
                event = _finalise_event(
                    canonical_operator_label('ZHIR'),
                    records,
                    start_index,
                    end_index,
                    phase_jump,
                    phase_jump_min,
                )
                payload = event.as_mapping()
                payload.update(
                    {
                        'phase_jump': float(phase_jump),
                        'epi_derivative': float(derivative),
                        'persistence': float(duration),
                    }
                )
                payload.update(
                    {
                        'xi_min': float(xi_min),
                        'min_persistence': float(min_persistence),
                        'phase_jump_min': float(phase_jump_min),
                    }
                )
                events.append(payload)
                index = end_index + 1
                continue
        index += 1

    return events


@_with_detection_overrides(
    "detect_remesh",
    metadata_argument="records",
    parameters=("window", "tau_candidates", "acf_min", "min_repeats"),
)
def detect_remesh(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 8,
    tau_candidates: Sequence[float] = (0.2, 0.4, 0.6),
    acf_min: float = 0.75,
    min_repeats: int = 2,
) -> List[Mapping[str, float | str | int]]:
    """Detect remeshing (REMESH) by repeating spatial patterns.

    Autocorrelation scores are evaluated at lags ``tau_candidates`` and an event
    is emitted when at least ``min_repeats`` candidates exceed ``acf_min``.
    """

    if len(records) < max(3, window):
        return []

    window = max(3, int(window))
    times = _sample_times(records)
    line_series = [
        float(_clean_numeric(getattr(record, 'line_deviation', 0.0)) or 0.0)
        for record in records
    ]

    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_similarity = 0.0
    best_metrics: Dict[str, float] = {}

    for index in range(len(records)):
        start_index = max(0, index - window + 1)
        length = index - start_index + 1
        if length < 3:
            continue

        window_times = times[start_index : index + 1]
        line_window = line_series[start_index : index + 1]
        duration = float(window_times[-1] - window_times[0])
        if duration <= 0.0:
            continue
        avg_dt = duration / max(length - 1, 1)

        matches = 0
        best_corr = 0.0
        for tau in tau_candidates:
            if tau <= 0.0:
                continue
            lag = max(1, int(round(tau / avg_dt)))
            if lag >= length:
                continue
            corr = _autocorrelation(line_window, lag)
            if corr >= acf_min:
                matches += 1
            best_corr = max(best_corr, corr)

        meets_threshold = matches >= min_repeats and best_corr >= acf_min

        metrics = {
            'matched_lags': int(matches),
            'best_correlation': float(best_corr),
            'avg_dt': float(avg_dt),
        }

        if meets_threshold:
            if active_start is None:
                active_start = start_index
                peak_similarity = best_corr
                best_metrics = metrics
            else:
                if best_corr >= peak_similarity:
                    best_metrics = metrics
                peak_similarity = max(peak_similarity, best_corr)
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label('REMESH'),
                records,
                active_start,
                index - 1,
                peak_similarity,
                acf_min,
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    'tau_candidates': tuple(float(value) for value in tau_candidates),
                    'acf_min': float(acf_min),
                    'min_repeats': int(min_repeats),
                }
            )
            events.append(payload)
            active_start = None
            peak_similarity = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label('REMESH'),
            records,
            active_start,
            len(records) - 1,
            peak_similarity,
            acf_min,
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                'tau_candidates': tuple(float(value) for value in tau_candidates),
                'acf_min': float(acf_min),
                'min_repeats': int(min_repeats),
            }
        )
        events.append(payload)

    return events

@dataclass(frozen=True)
class OperatorEvent:
    """Summary of a detected operator opportunity."""

    name: str
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    duration: float
    severity: float
    peak_value: float

    def as_mapping(self) -> Mapping[str, float | str | int]:
        return {
            "name": self.name,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "severity": self.severity,
            "peak_value": self.peak_value,
        }


def _window(
    records: Sequence[SupportsTelemetrySample], start: int, size: int
) -> Sequence[SupportsTelemetrySample]:
    lower = max(0, start - size + 1)
    return records[lower : start + 1]


def _finalise_event(
    name: str,
    records: Sequence[SupportsTelemetrySample],
    start_index: int,
    end_index: int,
    peak_value: float,
    threshold: float,
) -> OperatorEvent:
    start_index = max(0, start_index)
    end_index = max(start_index, end_index)
    start_time = float(records[start_index].timestamp)
    end_time = float(records[end_index].timestamp)
    duration = max(0.0, end_time - start_time)
    severity = peak_value / (threshold + 1e-9)
    return OperatorEvent(
        name=name,
        start_index=start_index,
        end_index=end_index,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        severity=max(0.0, severity),
        peak_value=peak_value,
    )


@_with_detection_overrides(
    "detect_al",
    metadata_argument="records",
    parameters=("window", "lateral_threshold", "load_threshold"),
)
def detect_al(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 5,
    lateral_threshold: float = 1.6,
    load_threshold: float = 250.0,
) -> List[Mapping[str, float | str | int]]:
    """Detect lateral support (AL) opportunities.

    The detector evaluates absolute lateral acceleration within a sliding
    window and confirms that the accompanying load transfer is significant
    before emitting an event.
    """

    if not records:
        return []

    events: List[OperatorEvent] = []
    active_start: int | None = None
    peak_value = 0.0
    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue
        lateral_avg = mean(abs(sample.lateral_accel) for sample in window_records)
        load_span = max(sample.vertical_load for sample in window_records) - min(
            sample.vertical_load for sample in window_records
        )
        meets_threshold = lateral_avg >= lateral_threshold and load_span >= load_threshold
        if meets_threshold:
            if active_start is None:
                active_start = index - window + 1
                peak_value = lateral_avg
            else:
                peak_value = max(peak_value, lateral_avg)
        elif active_start is not None:
            events.append(
                _finalise_event(
                    canonical_operator_label("AL"),
                    records,
                    active_start,
                    index - 1,
                    peak_value,
                    lateral_threshold,
                )
            )
            active_start = None
            peak_value = 0.0

    if active_start is not None:
        events.append(
            _finalise_event(
                canonical_operator_label("AL"),
                records,
                active_start,
                len(records) - 1,
                peak_value,
                lateral_threshold,
            )
        )

    return [event.as_mapping() for event in events]


@_with_detection_overrides(
    "detect_oz",
    metadata_argument="records",
    parameters=("window", "slip_threshold", "yaw_threshold"),
)
def detect_oz(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 5,
    slip_threshold: float = 0.12,
    yaw_threshold: float = 0.25,
) -> List[Mapping[str, float | str | int]]:
    """Detect oversteer (OZ) excursions."""

    if not records:
        return []

    events: List[OperatorEvent] = []
    active_start: int | None = None
    peak_value = 0.0
    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue
        slip_avg = mean(abs(sample.slip_angle) for sample in window_records)
        yaw_avg = mean(abs(sample.yaw_rate) for sample in window_records)
        meets_threshold = slip_avg >= slip_threshold and yaw_avg >= yaw_threshold
        if meets_threshold:
            if active_start is None:
                active_start = index - window + 1
                peak_value = max(slip_avg, yaw_avg)
            else:
                peak_value = max(peak_value, slip_avg, yaw_avg)
        elif active_start is not None:
            events.append(
                _finalise_event(
                    canonical_operator_label("OZ"),
                    records,
                    active_start,
                    index - 1,
                    peak_value,
                    max(slip_threshold, yaw_threshold),
                )
            )
            active_start = None
            peak_value = 0.0

    if active_start is not None:
        events.append(
            _finalise_event(
                canonical_operator_label("OZ"),
                records,
                active_start,
                len(records) - 1,
                peak_value,
                max(slip_threshold, yaw_threshold),
            )
        )

    return [event.as_mapping() for event in events]


@_with_detection_overrides(
    "detect_il",
    metadata_argument="records",
    parameters=("window", "base_threshold", "speed_gain"),
)
def detect_il(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 5,
    base_threshold: float = 0.35,
    speed_gain: float = 0.012,
) -> List[Mapping[str, float | str | int]]:
    """Detect ideal-line (IL) deviations with a speed dependent threshold."""

    if not records:
        return []

    events: List[OperatorEvent] = []
    active_start: int | None = None
    peak_value = 0.0
    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue
        speed_samples = [abs(sample.speed) for sample in window_records if math.isfinite(sample.speed)]
        mean_speed = mean(speed_samples) if speed_samples else 0.0
        threshold = base_threshold + (speed_gain * mean_speed)
        deviation_samples = [
            abs(sample.line_deviation)
            for sample in window_records
            if math.isfinite(sample.line_deviation)
        ]
        deviation_peak = max(deviation_samples) if deviation_samples else 0.0
        meets_threshold = deviation_peak >= threshold
        if meets_threshold:
            if active_start is None:
                active_start = index - window + 1
                peak_value = deviation_peak
            else:
                peak_value = max(peak_value, deviation_peak)
        elif active_start is not None:
            events.append(
                _finalise_event(
                    canonical_operator_label("IL"),
                    records,
                    active_start,
                    index - 1,
                    peak_value,
                    threshold,
                )
            )
            active_start = None
            peak_value = 0.0

    if active_start is not None:
        final_records = _window(records, len(records) - 1, window)
        final_speed_samples = [
            abs(sample.speed) for sample in final_records if math.isfinite(sample.speed)
        ]
        mean_speed = mean(final_speed_samples) if final_speed_samples else 0.0
        threshold = base_threshold + (speed_gain * mean_speed)
        events.append(
            _finalise_event(
                canonical_operator_label("IL"),
                records,
                active_start,
                len(records) - 1,
                peak_value,
                threshold,
            )
        )

    return [event.as_mapping() for event in events]


@_with_detection_overrides(
    "detect_silence",
    metadata_argument="records",
    parameters=("window", "load_threshold", "accel_threshold", "delta_nfr_threshold"),
)
def detect_silence(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 15,
    load_threshold: float = 400.0,
    accel_threshold: float = 0.8,
    delta_nfr_threshold: float = 45.0,
    structural_window: int = 11,
    structural_density_threshold: float = 0.2,
    min_duration: float = 0.8,
) -> List[Mapping[str, float | str | int]]:
    """Detect structural silence intervals with low dynamic activation."""

    if not records:
        return []

    window = max(2, int(window))
    structural_window = max(3, int(structural_window))
    structural_axis = compute_structural_timestamps(
        records,
        window_size=structural_window,
        base_timestamp=records[0].timestamp,
    )

    densities: List[float] = [0.0]
    for index in range(1, len(records)):
        current = float(records[index].timestamp)
        previous = float(records[index - 1].timestamp)
        chronological_dt = max(0.0, current - previous)
        if chronological_dt <= 1e-9:
            densities.append(0.0)
            continue
        structural_dt = max(0.0, structural_axis[index] - structural_axis[index - 1])
        density = max(0.0, (structural_dt / (chronological_dt + 1e-9)) - 1.0)
        densities.append(density)

    def _slack(value: float, threshold: float) -> float:
        if threshold <= 1e-9:
            return 0.0
        return max(0.0, (threshold - value) / threshold)

    def _event_metrics(start_index: int, end_index: int) -> tuple[dict[str, float], float]:
        samples = records[start_index : end_index + 1]
        load_values = [float(sample.vertical_load) for sample in samples]
        load_span = max(load_values) - min(load_values) if load_values else 0.0
        lat_mean = mean(abs(sample.lateral_accel) for sample in samples)
        long_mean = mean(abs(sample.longitudinal_accel) for sample in samples)
        accel_mean = 0.5 * (lat_mean + long_mean)
        nfr_values = [float(getattr(sample, "nfr", 0.0)) for sample in samples]
        delta_nfr_span = max(nfr_values) - min(nfr_values) if nfr_values else 0.0
        density_slice = densities[start_index : end_index + 1]
        structural_density_mean = mean(density_slice) if density_slice else 0.0
        slack_components = (
            _slack(load_span, load_threshold),
            _slack(accel_mean, accel_threshold),
            _slack(delta_nfr_span, delta_nfr_threshold),
            _slack(structural_density_mean, structural_density_threshold),
        )
        slack = min(slack_components)
        return (
            {
                "load_span": float(load_span),
                "accel_mean": float(accel_mean),
                "delta_nfr_span": float(delta_nfr_span),
                "structural_density_mean": float(structural_density_mean),
            },
            float(slack),
        )

    def _append_event(start_index: int, end_index: int) -> None:
        nonlocal events
        start_index = max(0, start_index)
        end_index = max(start_index, end_index)
        metrics, slack = _event_metrics(start_index, end_index)
        if slack <= 0.0:
            return
        event = _finalise_event(
            canonical_operator_label("SILENCE"),
            records,
            start_index,
            end_index,
            slack,
            1.0,
        )
        if event.duration < float(min_duration):
            return
        payload = event.as_mapping()
        payload.update(metrics)
        structural_start = float(structural_axis[start_index])
        structural_end = float(structural_axis[end_index])
        payload.update(
            {
                "structural_start": structural_start,
                "structural_end": structural_end,
                "structural_duration": max(0.0, structural_end - structural_start),
                "load_threshold": float(load_threshold),
                "accel_threshold": float(accel_threshold),
                "delta_nfr_threshold": float(delta_nfr_threshold),
                "structural_density_threshold": float(structural_density_threshold),
                "structural_window": int(structural_window),
                "slack": float(slack),
            }
        )
        events.append(payload)

    events: List[dict[str, float | str | int]] = []
    active_start: int | None = None

    for index in range(len(records)):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue
        load_values = [float(sample.vertical_load) for sample in window_records]
        load_span = max(load_values) - min(load_values) if load_values else 0.0
        lat_mean = mean(abs(sample.lateral_accel) for sample in window_records)
        long_mean = mean(abs(sample.longitudinal_accel) for sample in window_records)
        accel_mean = 0.5 * (lat_mean + long_mean)
        nfr_values = [float(getattr(sample, "nfr", 0.0)) for sample in window_records]
        delta_nfr_span = max(nfr_values) - min(nfr_values) if nfr_values else 0.0
        density_window = densities[max(0, index - window + 1) : index + 1]
        structural_density_mean = mean(density_window) if density_window else 0.0

        meets_thresholds = (
            load_span <= load_threshold
            and accel_mean <= accel_threshold
            and delta_nfr_span <= delta_nfr_threshold
            and structural_density_mean <= structural_density_threshold
        )

        if meets_thresholds:
            if active_start is None:
                active_start = index - window + 1
        elif active_start is not None:
            _append_event(active_start, index - 1)
            active_start = None

    if active_start is not None:
        _append_event(active_start, len(records) - 1)

    return events
