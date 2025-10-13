"""Detection utilities for on-track operator events.

Each detection routine analyses a windowed sequence of
:class:`~tnfr_core.operators.interfaces.SupportsTelemetrySample` objects and yields
event descriptors when the observed behaviour exceeds the
configured thresholds.  The detectors are intentionally lightweight so that
they can be executed on every microsector without adding measurable overhead
to the orchestration pipeline.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

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


def detect_nav(
    series: Sequence[DeltaSample],
    *,
    nu_f: Union[float, Mapping[str, float], None],
    window: int = 3,
    eps: float = 1e-3,
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


def detect_en(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 5,
    si_threshold: float = 0.6,
    nfr_span_threshold: float = 35.0,
    throttle_threshold: float = 0.3,
) -> List[Mapping[str, float | str | int]]:
    """Detect reception (EN) intervals where Si stabilises incoming ΔNFR.

    The heuristic analyses rolling windows looking for a sustained sense index
    above ``si_threshold`` while ΔNFR variation and the throttle budget remain
    within the ``nfr_span_threshold`` and ``throttle_threshold`` limits.
    Windows failing to gather enough samples (``len(window) < window``) are
    skipped, mirroring the existing structural detectors.
    """

    if not records:
        return []

    window = max(2, int(window))
    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_value = 0.0
    best_metrics: Dict[str, float] = {}

    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue

        si_samples = _clean_sequence([getattr(sample, "si", 0.0) for sample in window_records])
        si_mean = mean(si_samples) if si_samples else 0.0
        nfr_values = _clean_sequence([getattr(sample, "nfr", 0.0) for sample in window_records])
        nfr_span = _span(nfr_values)
        throttle_samples = _clean_sequence(
            [getattr(sample, "throttle", 0.0) for sample in window_records]
        )
        throttle_mean = mean(throttle_samples) if throttle_samples else 0.0
        yaw_samples = _clean_sequence(
            [abs(getattr(sample, "yaw_rate", 0.0)) for sample in window_records]
        )
        yaw_rate_mean = mean(yaw_samples) if yaw_samples else 0.0

        meets_threshold = (
            si_mean >= si_threshold
            and nfr_span <= nfr_span_threshold
            and throttle_mean >= throttle_threshold
        )

        metrics = {
            "si_mean": float(si_mean),
            "nfr_span": float(nfr_span),
            "throttle_mean": float(throttle_mean),
            "yaw_rate_mean": float(yaw_rate_mean),
        }

        if meets_threshold:
            if active_start is None:
                active_start = max(0, index - window + 1)
                peak_value = si_mean
                best_metrics = metrics
            else:
                peak_value = max(peak_value, si_mean)
                if si_mean >= best_metrics.get("si_mean", 0.0):
                    best_metrics = metrics
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label("EN"),
                records,
                active_start,
                index - 1,
                peak_value,
                si_threshold,
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    "si_threshold": float(si_threshold),
                    "nfr_span_threshold": float(nfr_span_threshold),
                    "throttle_threshold": float(throttle_threshold),
                }
            )
            events.append(payload)
            active_start = None
            peak_value = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label("EN"),
            records,
            active_start,
            len(records) - 1,
            peak_value,
            si_threshold,
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                "si_threshold": float(si_threshold),
                "nfr_span_threshold": float(nfr_span_threshold),
                "throttle_threshold": float(throttle_threshold),
            }
        )
        events.append(payload)

    return events


def detect_um(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 5,
    mu_delta_threshold: float = 0.2,
    load_ratio_threshold: float = 0.08,
    suspension_delta_threshold: float = 0.015,
) -> List[Mapping[str, float | str | int]]:
    """Detect coupling (UM) deviations between front and rear nodes.

    Coupling is flagged when the mean μ effectiveness gap exceeds
    ``mu_delta_threshold`` while the front/rear load ratio and suspension
    velocities diverge beyond ``load_ratio_threshold`` and
    ``suspension_delta_threshold`` respectively.
    """

    if not records:
        return []

    window = max(2, int(window))
    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_value = 0.0
    best_metrics: Dict[str, float] = {}

    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue

        mu_front_samples = _clean_sequence(
            [getattr(sample, "mu_eff_front", 0.0) for sample in window_records]
        )
        mu_rear_samples = _clean_sequence(
            [getattr(sample, "mu_eff_rear", 0.0) for sample in window_records]
        )
        mu_front_mean = mean(mu_front_samples) if mu_front_samples else 0.0
        mu_rear_mean = mean(mu_rear_samples) if mu_rear_samples else 0.0
        mu_delta = abs(mu_front_mean - mu_rear_mean)

        load_ratios: List[float] = []
        for sample in window_records:
            front = _clean_numeric(getattr(sample, "vertical_load_front", 0.0))
            rear = _clean_numeric(getattr(sample, "vertical_load_rear", 0.0))
            if front is None or rear is None:
                continue
            total = front + rear
            if total <= 1e-9:
                continue
            load_ratios.append(front / total)
        load_ratio_delta = _span(load_ratios)

        suspension_front = _clean_sequence(
            [getattr(sample, "suspension_velocity_front", 0.0) for sample in window_records]
        )
        suspension_rear = _clean_sequence(
            [getattr(sample, "suspension_velocity_rear", 0.0) for sample in window_records]
        )
        suspension_delta = abs(
            (mean(suspension_front) if suspension_front else 0.0)
            - (mean(suspension_rear) if suspension_rear else 0.0)
        )

        meets_threshold = (
            mu_delta >= mu_delta_threshold
            and load_ratio_delta >= load_ratio_threshold
            and suspension_delta >= suspension_delta_threshold
        )

        metrics = {
            "mu_front_mean": float(mu_front_mean),
            "mu_rear_mean": float(mu_rear_mean),
            "mu_delta": float(mu_delta),
            "load_ratio_delta": float(load_ratio_delta),
            "suspension_velocity_delta": float(suspension_delta),
        }

        if meets_threshold:
            if active_start is None:
                active_start = max(0, index - window + 1)
                peak_value = mu_delta
                best_metrics = metrics
            else:
                peak_value = max(peak_value, mu_delta)
                if mu_delta >= best_metrics.get("mu_delta", 0.0):
                    best_metrics = metrics
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label("UM"),
                records,
                active_start,
                index - 1,
                peak_value,
                mu_delta_threshold,
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    "mu_delta_threshold": float(mu_delta_threshold),
                    "load_ratio_threshold": float(load_ratio_threshold),
                    "suspension_delta_threshold": float(suspension_delta_threshold),
                }
            )
            events.append(payload)
            active_start = None
            peak_value = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label("UM"),
            records,
            active_start,
            len(records) - 1,
            peak_value,
            mu_delta_threshold,
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                "mu_delta_threshold": float(mu_delta_threshold),
                "load_ratio_threshold": float(load_ratio_threshold),
                "suspension_delta_threshold": float(suspension_delta_threshold),
            }
        )
        events.append(payload)

    return events


def detect_ra(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 5,
    nfr_rate_threshold: float = 25.0,
    si_span_threshold: float = 0.15,
    speed_threshold: float = 20.0,
) -> List[Mapping[str, float | str | int]]:
    """Detect propagation (RA) bursts from ΔNFR diffusion.

    The detector tracks the mean ΔNFR rate of change per window. Propagation is
    reported when that mean exceeds ``nfr_rate_threshold`` while the sense index
    span is above ``si_span_threshold`` and the mean absolute speed stays above
    ``speed_threshold``.
    """

    if not records:
        return []

    window = max(2, int(window))
    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_value = 0.0
    best_metrics: Dict[str, float] = {}

    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue

        nfr_sequence = [
            _clean_numeric(getattr(sample, "nfr", 0.0)) for sample in window_records
        ]
        nfr_rates: List[float] = []
        prev_value: float | None = None
        for value in nfr_sequence:
            if value is None:
                continue
            if prev_value is not None:
                nfr_rates.append(abs(value - prev_value))
            prev_value = value
        nfr_rate_mean = mean(nfr_rates) if nfr_rates else 0.0

        si_values = _clean_sequence([getattr(sample, "si", 0.0) for sample in window_records])
        si_span = _span(si_values)

        speed_samples = _clean_sequence(
            [abs(getattr(sample, "speed", 0.0)) for sample in window_records]
        )
        speed_mean = mean(speed_samples) if speed_samples else 0.0

        meets_threshold = (
            nfr_rate_mean >= nfr_rate_threshold
            and si_span >= si_span_threshold
            and speed_mean >= speed_threshold
        )

        metrics = {
            "nfr_rate_mean": float(nfr_rate_mean),
            "si_span": float(si_span),
            "speed_mean": float(speed_mean),
        }

        if meets_threshold:
            if active_start is None:
                active_start = max(0, index - window + 1)
                peak_value = nfr_rate_mean
                best_metrics = metrics
            else:
                peak_value = max(peak_value, nfr_rate_mean)
                if nfr_rate_mean >= best_metrics.get("nfr_rate_mean", 0.0):
                    best_metrics = metrics
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label("RA"),
                records,
                active_start,
                index - 1,
                peak_value,
                nfr_rate_threshold,
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    "nfr_rate_threshold": float(nfr_rate_threshold),
                    "si_span_threshold": float(si_span_threshold),
                    "speed_threshold": float(speed_threshold),
                }
            )
            events.append(payload)
            active_start = None
            peak_value = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label("RA"),
            records,
            active_start,
            len(records) - 1,
            peak_value,
            nfr_rate_threshold,
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                "nfr_rate_threshold": float(nfr_rate_threshold),
                "si_span_threshold": float(si_span_threshold),
                "speed_threshold": float(speed_threshold),
            }
        )
        events.append(payload)

    return events


def detect_val(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 5,
    lateral_threshold: float = 1.8,
    throttle_threshold: float = 0.55,
    load_span_threshold: float = 600.0,
) -> List[Mapping[str, float | str | int]]:
    """Detect amplification (VAL) opportunities based on support demand.

    Windows that exceed the lateral, throttle, and vertical load thresholds are
    marked as amplification candidates. The detector focuses on the maximum
    lateral acceleration and throttle observed within the window.
    """

    if not records:
        return []

    window = max(2, int(window))
    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_value = 0.0
    best_metrics: Dict[str, float] = {}

    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue

        lateral_samples = _clean_sequence(
            [abs(getattr(sample, "lateral_accel", 0.0)) for sample in window_records]
        )
        lateral_peak = max(lateral_samples) if lateral_samples else 0.0
        throttle_samples = _clean_sequence(
            [getattr(sample, "throttle", 0.0) for sample in window_records]
        )
        throttle_peak = max(throttle_samples) if throttle_samples else 0.0
        load_values = _clean_sequence(
            [getattr(sample, "vertical_load", 0.0) for sample in window_records]
        )
        load_span = _span(load_values)
        longitudinal_samples = _clean_sequence(
            [abs(getattr(sample, "longitudinal_accel", 0.0)) for sample in window_records]
        )
        longitudinal_mean = mean(longitudinal_samples) if longitudinal_samples else 0.0

        meets_threshold = (
            lateral_peak >= lateral_threshold
            and throttle_peak >= throttle_threshold
            and load_span >= load_span_threshold
        )

        metrics = {
            "lateral_peak": float(lateral_peak),
            "throttle_peak": float(throttle_peak),
            "load_span": float(load_span),
            "longitudinal_mean": float(longitudinal_mean),
        }

        if meets_threshold:
            if active_start is None:
                active_start = max(0, index - window + 1)
                peak_value = max(lateral_peak, throttle_peak)
                best_metrics = metrics
            else:
                peak_value = max(peak_value, lateral_peak, throttle_peak)
                if peak_value >= max(best_metrics.get("lateral_peak", 0.0), best_metrics.get("throttle_peak", 0.0)):
                    best_metrics = metrics
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label("VAL"),
                records,
                active_start,
                index - 1,
                peak_value,
                max(lateral_threshold, throttle_threshold),
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    "lateral_threshold": float(lateral_threshold),
                    "throttle_threshold": float(throttle_threshold),
                    "load_span_threshold": float(load_span_threshold),
                }
            )
            events.append(payload)
            active_start = None
            peak_value = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label("VAL"),
            records,
            active_start,
            len(records) - 1,
            peak_value,
            max(lateral_threshold, throttle_threshold),
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                "lateral_threshold": float(lateral_threshold),
                "throttle_threshold": float(throttle_threshold),
                "load_span_threshold": float(load_span_threshold),
            }
        )
        events.append(payload)

    return events


def detect_nul(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 5,
    decel_threshold: float = 1.6,
    speed_drop_threshold: float = 5.0,
    brake_pressure_threshold: float = 0.45,
) -> List[Mapping[str, float | str | int]]:
    """Detect contraction (NUL) phases driven by heavy braking.

    The routine searches for windows with a large negative longitudinal
    acceleration peak (``decel_threshold``), a significant speed drop, and a
    sustained brake pressure budget.
    """

    if not records:
        return []

    window = max(2, int(window))
    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_value = 0.0
    best_metrics: Dict[str, float] = {}

    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue

        longitudinal_samples = _clean_sequence(
            [getattr(sample, "longitudinal_accel", 0.0) for sample in window_records]
        )
        longitudinal_min = min(longitudinal_samples) if longitudinal_samples else 0.0
        decel_peak = abs(longitudinal_min)

        speed_samples = _clean_sequence(
            [getattr(sample, "speed", 0.0) for sample in window_records]
        )
        speed_drop = _span(speed_samples)

        brake_samples = _clean_sequence(
            [getattr(sample, "brake_pressure", 0.0) for sample in window_records]
        )
        brake_mean = mean(brake_samples) if brake_samples else 0.0

        meets_threshold = (
            decel_peak >= decel_threshold
            and speed_drop >= speed_drop_threshold
            and brake_mean >= brake_pressure_threshold
        )

        metrics = {
            "decel_peak": float(decel_peak),
            "speed_drop": float(speed_drop),
            "brake_pressure_mean": float(brake_mean),
        }

        if meets_threshold:
            if active_start is None:
                active_start = max(0, index - window + 1)
                peak_value = decel_peak
                best_metrics = metrics
            else:
                peak_value = max(peak_value, decel_peak)
                if decel_peak >= best_metrics.get("decel_peak", 0.0):
                    best_metrics = metrics
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label("NUL"),
                records,
                active_start,
                index - 1,
                peak_value,
                decel_threshold,
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    "decel_threshold": float(decel_threshold),
                    "speed_drop_threshold": float(speed_drop_threshold),
                    "brake_pressure_threshold": float(brake_pressure_threshold),
                }
            )
            events.append(payload)
            active_start = None
            peak_value = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label("NUL"),
            records,
            active_start,
            len(records) - 1,
            peak_value,
            decel_threshold,
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                "decel_threshold": float(decel_threshold),
                "speed_drop_threshold": float(speed_drop_threshold),
                "brake_pressure_threshold": float(brake_pressure_threshold),
            }
        )
        events.append(payload)

    return events


def detect_thol(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 7,
    suspension_span_threshold: float = 0.04,
    steer_span_threshold: float = 0.1,
    yaw_rate_span_threshold: float = 0.15,
) -> List[Mapping[str, float | str | int]]:
    """Detect auto-organisation (THOL) phases during regime transitions.

    Auto-organisation is identified when suspension activity spikes while
    steering and yaw variations remain tightly bounded, suggesting the chassis
    is reorganising without large heading changes.
    """

    if not records:
        return []

    window = max(3, int(window))
    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_value = 0.0
    best_metrics: Dict[str, float] = {}

    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue

        suspension_front = _clean_sequence(
            [getattr(sample, "suspension_velocity_front", 0.0) for sample in window_records]
        )
        suspension_rear = _clean_sequence(
            [getattr(sample, "suspension_velocity_rear", 0.0) for sample in window_records]
        )
        suspension_activity = _span(suspension_front) + _span(suspension_rear)

        steer_values = _clean_sequence(
            [getattr(sample, "steer", 0.0) for sample in window_records]
        )
        steer_span = _span(steer_values)

        yaw_values = _clean_sequence(
            [getattr(sample, "yaw_rate", 0.0) for sample in window_records]
        )
        yaw_span = _span(yaw_values)

        meets_threshold = (
            suspension_activity >= suspension_span_threshold
            and steer_span <= steer_span_threshold
            and yaw_span <= yaw_rate_span_threshold
        )

        metrics = {
            "suspension_activity": float(suspension_activity),
            "steer_span": float(steer_span),
            "yaw_rate_span": float(yaw_span),
        }

        if meets_threshold:
            if active_start is None:
                active_start = max(0, index - window + 1)
                peak_value = suspension_activity
                best_metrics = metrics
            else:
                peak_value = max(peak_value, suspension_activity)
                if suspension_activity >= best_metrics.get("suspension_activity", 0.0):
                    best_metrics = metrics
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label("THOL"),
                records,
                active_start,
                index - 1,
                peak_value,
                suspension_span_threshold,
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    "suspension_span_threshold": float(suspension_span_threshold),
                    "steer_span_threshold": float(steer_span_threshold),
                    "yaw_rate_span_threshold": float(yaw_rate_span_threshold),
                }
            )
            events.append(payload)
            active_start = None
            peak_value = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label("THOL"),
            records,
            active_start,
            len(records) - 1,
            peak_value,
            suspension_span_threshold,
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                "suspension_span_threshold": float(suspension_span_threshold),
                "steer_span_threshold": float(steer_span_threshold),
                "yaw_rate_span_threshold": float(yaw_rate_span_threshold),
            }
        )
        events.append(payload)

    return events


def detect_zhir(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 7,
    si_delta_threshold: float = 0.25,
    nfr_delta_threshold: float = 50.0,
    line_deviation_threshold: float = 0.4,
) -> List[Mapping[str, float | str | int]]:
    """Detect transformation (ZHIR) bursts reflecting deep archetype shifts."""

    if not records:
        return []

    window = max(3, int(window))
    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_value = 0.0
    best_metrics: Dict[str, float] = {}

    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue

        si_values = _clean_sequence([getattr(sample, "si", 0.0) for sample in window_records])
        nfr_values = _clean_sequence([getattr(sample, "nfr", 0.0) for sample in window_records])
        line_values = _clean_sequence(
            [getattr(sample, "line_deviation", 0.0) for sample in window_records]
        )

        if len(si_values) >= 2:
            si_delta = abs(si_values[-1] - si_values[0])
        else:
            si_delta = 0.0
        if len(nfr_values) >= 2:
            nfr_delta = abs(nfr_values[-1] - nfr_values[0])
        else:
            nfr_delta = 0.0
        line_span = _span(line_values)

        meets_threshold = (
            si_delta >= si_delta_threshold
            and nfr_delta >= nfr_delta_threshold
            and line_span >= line_deviation_threshold
        )

        metrics = {
            "si_delta": float(si_delta),
            "nfr_delta": float(nfr_delta),
            "line_deviation_span": float(line_span),
        }

        if meets_threshold:
            if active_start is None:
                active_start = max(0, index - window + 1)
                peak_value = max(si_delta, nfr_delta)
                best_metrics = metrics
            else:
                peak_value = max(peak_value, si_delta, nfr_delta)
                if peak_value >= max(
                    best_metrics.get("si_delta", 0.0), best_metrics.get("nfr_delta", 0.0)
                ):
                    best_metrics = metrics
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label("ZHIR"),
                records,
                active_start,
                index - 1,
                peak_value,
                max(si_delta_threshold, nfr_delta_threshold),
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    "si_delta_threshold": float(si_delta_threshold),
                    "nfr_delta_threshold": float(nfr_delta_threshold),
                    "line_deviation_threshold": float(line_deviation_threshold),
                }
            )
            events.append(payload)
            active_start = None
            peak_value = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label("ZHIR"),
            records,
            active_start,
            len(records) - 1,
            peak_value,
            max(si_delta_threshold, nfr_delta_threshold),
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                "si_delta_threshold": float(si_delta_threshold),
                "nfr_delta_threshold": float(nfr_delta_threshold),
                "line_deviation_threshold": float(line_deviation_threshold),
            }
        )
        events.append(payload)

    return events


def detect_remesh(
    records: Sequence[SupportsTelemetrySample],
    *,
    window: int = 5,
    line_gradient_threshold: float = 0.25,
    yaw_rate_gradient_threshold: float = 0.25,
    structural_gap_threshold: float = 0.6,
) -> List[Mapping[str, float | str | int]]:
    """Detect remeshing (REMESH) opportunities from structural discontinuities."""

    if not records:
        return []

    window = max(2, int(window))
    events: List[Mapping[str, float | str | int]] = []
    active_start: int | None = None
    peak_value = 0.0
    best_metrics: Dict[str, float] = {}

    for index, record in enumerate(records):
        window_records = _window(records, index, window)
        if len(window_records) < window:
            continue

        line_gradients: List[float] = []
        yaw_gradients: List[float] = []
        structural_gaps: List[float] = []

        prev_line: float | None = None
        prev_yaw: float | None = None
        prev_struct: float | None = None
        prev_time: float | None = None

        for sample in window_records:
            line_value = _clean_numeric(getattr(sample, "line_deviation", 0.0))
            yaw_value = _clean_numeric(getattr(sample, "yaw_rate", 0.0))
            struct_value = _clean_numeric(getattr(sample, "structural_timestamp", None))
            time_value = _clean_numeric(getattr(sample, "timestamp", 0.0))

            if line_value is not None and prev_line is not None:
                line_gradients.append(abs(line_value - prev_line))
            if yaw_value is not None and prev_yaw is not None:
                yaw_gradients.append(abs(yaw_value - prev_yaw))
            if (
                struct_value is not None
                and prev_struct is not None
                and time_value is not None
                and prev_time is not None
            ):
                dt = time_value - prev_time
                if dt > 1e-9:
                    structural_delta = struct_value - prev_struct
                    structural_gaps.append(abs(structural_delta - dt))

            prev_line = line_value if line_value is not None else prev_line
            prev_yaw = yaw_value if yaw_value is not None else prev_yaw
            prev_struct = struct_value if struct_value is not None else prev_struct
            prev_time = time_value if time_value is not None else prev_time

        line_gradient_mean = mean(line_gradients) if line_gradients else 0.0
        yaw_gradient_mean = mean(yaw_gradients) if yaw_gradients else 0.0
        structural_gap_mean = mean(structural_gaps) if structural_gaps else 0.0

        meets_threshold = (
            line_gradient_mean >= line_gradient_threshold
            and yaw_gradient_mean >= yaw_rate_gradient_threshold
            and structural_gap_mean >= structural_gap_threshold
        )

        metrics = {
            "line_gradient_mean": float(line_gradient_mean),
            "yaw_rate_gradient_mean": float(yaw_gradient_mean),
            "structural_gap_mean": float(structural_gap_mean),
        }

        if meets_threshold:
            if active_start is None:
                active_start = max(0, index - window + 1)
                peak_value = max(line_gradient_mean, yaw_gradient_mean, structural_gap_mean)
                best_metrics = metrics
            else:
                candidate_peak = max(line_gradient_mean, yaw_gradient_mean, structural_gap_mean)
                peak_value = max(peak_value, candidate_peak)
                if candidate_peak >= max(
                    best_metrics.get("line_gradient_mean", 0.0),
                    best_metrics.get("yaw_rate_gradient_mean", 0.0),
                    best_metrics.get("structural_gap_mean", 0.0),
                ):
                    best_metrics = metrics
        elif active_start is not None:
            event = _finalise_event(
                canonical_operator_label("REMESH"),
                records,
                active_start,
                index - 1,
                peak_value,
                max(line_gradient_threshold, yaw_rate_gradient_threshold),
            )
            payload = event.as_mapping()
            payload.update(best_metrics)
            payload.update(
                {
                    "line_gradient_threshold": float(line_gradient_threshold),
                    "yaw_rate_gradient_threshold": float(yaw_rate_gradient_threshold),
                    "structural_gap_threshold": float(structural_gap_threshold),
                }
            )
            events.append(payload)
            active_start = None
            peak_value = 0.0
            best_metrics = {}

    if active_start is not None:
        event = _finalise_event(
            canonical_operator_label("REMESH"),
            records,
            active_start,
            len(records) - 1,
            peak_value,
            max(line_gradient_threshold, yaw_rate_gradient_threshold),
        )
        payload = event.as_mapping()
        payload.update(best_metrics)
        payload.update(
            {
                "line_gradient_threshold": float(line_gradient_threshold),
                "yaw_rate_gradient_threshold": float(yaw_rate_gradient_threshold),
                "structural_gap_threshold": float(structural_gap_threshold),
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
