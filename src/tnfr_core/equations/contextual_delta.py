"""Context-aware ΔNFR adjustments.

This module centralises the heuristics required to contextualise ΔNFR values
based on curvature, track surface grip and traffic density cues.  The
calibration data is loaded from the embedded TOML metadata living under
``tnfr_lfs/data`` so downstream modules can rely on a single source of truth
for the adjustment matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

try:  # pragma: no cover - ``tomllib`` only ships with Python ≥ 3.11
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fall back to ``tomli`` in 3.10
    import tomli as tomllib  # type: ignore

from tnfr_lfs.data import CONTEXT_FACTORS_RESOURCE
from tnfr_core.operators.interfaces import SupportsContextBundle, SupportsContextRecord

__all__ = [
    "ContextFactors",
    "ContextMatrix",
    "apply_contextual_delta",
    "load_context_matrix",
    "resolve_microsector_context",
    "resolve_context_from_bundle",
    "resolve_context_from_record",
    "resolve_series_context",
]


@dataclass(frozen=True)
class ContextFactors:
    """Triplet of multiplicative factors applied to ΔNFR."""

    curve: float = 1.0
    surface: float = 1.0
    traffic: float = 1.0

    def as_mapping(self) -> Mapping[str, float]:
        return {"curve": self.curve, "surface": self.surface, "traffic": self.traffic}

    @property
    def multiplier(self) -> float:
        return self.curve * self.surface * self.traffic


@dataclass(frozen=True)
class ContextMatrix:
    """Calibration payload loaded from ``context_factors.toml``."""

    curve_bands: tuple[tuple[float | None, float, str | None], ...]
    surface_bands: tuple[tuple[float | None, float | None, float, str | None], ...]
    traffic_bands: tuple[tuple[float | None, float, str | None], ...]
    min_multiplier: float
    max_multiplier: float
    surface_reference_load: float
    traffic_speed_reference: float
    traffic_direction_reference: float
    traffic_longitudinal_reference: float

    def curve_factor(self, value: float) -> float:
        for limit, factor, _ in self.curve_bands:
            if limit is None or value <= limit:
                return factor
        return self.curve_bands[-1][1]

    def curve_band(self, value: float) -> tuple[float | None, float, str | None]:
        for band in self.curve_bands:
            limit, _, _ = band
            if limit is None or value <= limit:
                return band
        return self.curve_bands[-1]

    def surface_factor(self, ratio: float) -> float:
        for lower, upper, factor, _ in self.surface_bands:
            lower_ok = lower is None or ratio >= lower
            upper_ok = upper is None or ratio <= upper
            if lower_ok and upper_ok:
                return factor
        return self.surface_bands[-1][2]

    def surface_band(
        self, ratio: float
    ) -> tuple[float | None, float | None, float, str | None]:
        for band in self.surface_bands:
            lower, upper, _, _ = band
            lower_ok = lower is None or ratio >= lower
            upper_ok = upper is None or ratio <= upper
            if lower_ok and upper_ok:
                return band
        return self.surface_bands[-1]

    def traffic_factor(self, load: float) -> float:
        for limit, factor, _ in self.traffic_bands:
            if limit is None or load <= limit:
                return factor
        return self.traffic_bands[-1][1]

    def traffic_band(self, load: float) -> tuple[float | None, float, str | None]:
        for band in self.traffic_bands:
            limit, _, _ = band
            if limit is None or load <= limit:
                return band
        return self.traffic_bands[-1]


def _parse_curve_bands(
    payload: Mapping[str, object]
) -> tuple[tuple[float | None, float, str | None], ...]:
    bands: list[tuple[float | None, float, str | None]] = []
    for entry in payload.get("bands", []) or []:
        limit = entry.get("max") if isinstance(entry, Mapping) else None
        factor = entry.get("factor") if isinstance(entry, Mapping) else None
        label = entry.get("label") if isinstance(entry, Mapping) else None
        if factor is None:
            continue
        bands.append(
            (
                float(limit) if limit is not None else None,
                float(factor),
                str(label) if label is not None else None,
            )
        )
    if not bands:
        bands.append((None, 1.0, None))
    bands.sort(key=lambda item: float("inf") if item[0] is None else item[0])
    return tuple(bands)


def _parse_surface_bands(
    payload: Mapping[str, object]
) -> tuple[tuple[float | None, float | None, float, str | None], ...]:
    bands: list[tuple[float | None, float | None, float, str | None]] = []
    for entry in payload.get("bands", []) or []:
        if not isinstance(entry, Mapping):
            continue
        lower = entry.get("min")
        upper = entry.get("max")
        factor = entry.get("factor", 1.0)
        label = entry.get("label")
        bands.append(
            (
                float(lower) if lower is not None else None,
                float(upper) if upper is not None else None,
                float(factor),
                str(label) if label is not None else None,
            )
        )
    if not bands:
        bands.append((None, None, 1.0, None))
    bands.sort(
        key=lambda item: (
            float("-inf") if item[0] is None else item[0],
            float("inf") if item[1] is None else item[1],
        )
    )
    return tuple(bands)


def _parse_traffic_bands(
    payload: Mapping[str, object]
) -> tuple[tuple[float | None, float, str | None], ...]:
    bands: list[tuple[float | None, float, str | None]] = []
    for entry in payload.get("bands", []) or []:
        if not isinstance(entry, Mapping):
            continue
        limit = entry.get("max")
        factor = entry.get("factor", 1.0)
        label = entry.get("label")
        bands.append(
            (
                float(limit) if limit is not None else None,
                float(factor),
                str(label) if label is not None else None,
            )
        )
    if not bands:
        bands.append((None, 1.0, None))
    bands.sort(key=lambda item: float("inf") if item[0] is None else item[0])
    return tuple(bands)


@lru_cache()
def load_context_matrix(track: str | None = None) -> ContextMatrix:
    """Return the context matrix for ``track`` (defaults to the generic profile)."""

    with CONTEXT_FACTORS_RESOURCE.open("rb") as handle:
        payload = tomllib.load(handle)
    dataset = payload.get(track or "defaults") or payload.get("defaults") or {}
    limits = dataset.get("min_multiplier"), dataset.get("max_multiplier")
    min_multiplier = float(limits[0]) if limits[0] is not None else 0.6
    max_multiplier = float(limits[1]) if limits[1] is not None else 1.4

    curve_data = dataset.get("curve") if isinstance(dataset, Mapping) else {}
    surface_data = dataset.get("surface") if isinstance(dataset, Mapping) else {}
    traffic_data = dataset.get("traffic") if isinstance(dataset, Mapping) else {}

    curve_bands = _parse_curve_bands(curve_data if isinstance(curve_data, Mapping) else {})
    surface_bands = _parse_surface_bands(surface_data if isinstance(surface_data, Mapping) else {})
    traffic_bands = _parse_traffic_bands(traffic_data if isinstance(traffic_data, Mapping) else {})

    surface_reference = float(surface_data.get("reference_load", 5000.0)) if isinstance(surface_data, Mapping) else 5000.0
    speed_reference = float(traffic_data.get("speed_drop_reference", 10.0)) if isinstance(traffic_data, Mapping) else 10.0
    direction_reference = float(traffic_data.get("direction_change_reference", 2.5)) if isinstance(traffic_data, Mapping) else 2.5
    longitudinal_reference = float(traffic_data.get("longitudinal_reference", 0.5)) if isinstance(traffic_data, Mapping) else 0.5

    return ContextMatrix(
        curve_bands=curve_bands,
        surface_bands=surface_bands,
        traffic_bands=traffic_bands,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
        surface_reference_load=surface_reference,
        traffic_speed_reference=max(speed_reference, 1e-6),
        traffic_direction_reference=max(direction_reference, 1e-6),
        traffic_longitudinal_reference=max(longitudinal_reference, 1e-6),
    )


def _resolve_multiplier(factors: ContextFactors | Mapping[str, float]) -> float:
    if isinstance(factors, ContextFactors):
        return factors.multiplier
    curve = float(factors.get("curve", 1.0)) if isinstance(factors, Mapping) else 1.0
    surface = float(factors.get("surface", 1.0)) if isinstance(factors, Mapping) else 1.0
    traffic = float(factors.get("traffic", 1.0)) if isinstance(factors, Mapping) else 1.0
    return curve * surface * traffic


def apply_contextual_delta(
    delta_value: float,
    factors: ContextFactors | Mapping[str, float],
    *,
    context_matrix: ContextMatrix | None = None,
) -> float:
    """Return ``delta_value`` scaled by the contextual factor matrix."""

    matrix = context_matrix or load_context_matrix()
    multiplier = _resolve_multiplier(factors)
    multiplier = max(matrix.min_multiplier, min(matrix.max_multiplier, multiplier))
    return delta_value * multiplier


def resolve_context_from_record(
    matrix: ContextMatrix,
    record: SupportsContextRecord | Mapping[str, object],
    *,
    baseline_vertical_load: float | None = None,
) -> ContextFactors:
    """Derive factors from a :class:`~tnfr_core.equations.epi.TelemetryRecord`-like payload."""

    if isinstance(record, Mapping):
        lateral = abs(float(record.get("lateral_accel", 0.0)))
        vertical = float(record.get("vertical_load", 0.0))
        long_accel = abs(float(record.get("longitudinal_accel", 0.0)))
    elif isinstance(record, SupportsContextRecord):
        lateral = abs(float(record.lateral_accel))
        vertical = float(record.vertical_load)
        long_accel = abs(float(record.longitudinal_accel))
    else:  # pragma: no cover - defensive fallback for unexpected payloads
        lateral = abs(float(getattr(record, "lateral_accel", 0.0)))
        vertical = float(getattr(record, "vertical_load", 0.0))
        long_accel = abs(float(getattr(record, "longitudinal_accel", 0.0)))

    curve_factor = matrix.curve_factor(lateral)

    reference = baseline_vertical_load or matrix.surface_reference_load
    reference = reference if reference > 1e-6 else matrix.surface_reference_load
    reference = reference if reference > 1e-6 else 1.0
    surface_ratio = vertical / reference
    surface_factor = matrix.surface_factor(surface_ratio)

    traffic_load = long_accel / matrix.traffic_longitudinal_reference
    traffic_factor = matrix.traffic_factor(traffic_load)

    return ContextFactors(curve_factor, surface_factor, traffic_factor)


def resolve_context_from_bundle(
    matrix: ContextMatrix, bundle: SupportsContextBundle | Mapping[str, object]
) -> ContextFactors:
    """Resolve factors using the information stored inside an :class:`EPIBundle`."""

    if isinstance(bundle, Mapping):
        chassis = bundle.get("chassis")
        tyres = bundle.get("tyres")
        transmission = bundle.get("transmission")
    elif isinstance(bundle, SupportsContextBundle):
        chassis = bundle.chassis
        tyres = bundle.tyres
        transmission = bundle.transmission
    else:  # pragma: no cover - defensive fallback for unexpected payloads
        chassis = getattr(bundle, "chassis", None)
        tyres = getattr(bundle, "tyres", None)
        transmission = getattr(bundle, "transmission", None)

    lateral = (
        abs(float(getattr(chassis, "lateral_accel", 0.0))) if chassis is not None else 0.0
    )
    curve_factor = matrix.curve_factor(lateral)

    load_value = float(getattr(tyres, "load", 0.0)) if tyres is not None else 0.0
    reference = matrix.surface_reference_load
    surface_ratio = load_value / reference if load_value > 0.0 and reference > 1e-6 else 1.0
    surface_factor = matrix.surface_factor(surface_ratio)

    long_accel = float(getattr(chassis, "longitudinal_accel", 0.0))
    if long_accel == 0.0 and transmission is not None:
        long_accel = float(getattr(transmission, "longitudinal_accel", 0.0))
    traffic_load = abs(long_accel) / matrix.traffic_longitudinal_reference
    traffic_factor = matrix.traffic_factor(traffic_load)

    return ContextFactors(curve_factor, surface_factor, traffic_factor)


def resolve_microsector_context(
    matrix: ContextMatrix,
    *,
    curvature: float,
    grip_rel: float,
    speed_drop: float,
    direction_changes: float,
) -> ContextFactors:
    """Resolve aggregate factors for a microsector from its summary metrics."""

    curve_factor = matrix.curve_factor(max(curvature, 0.0))
    surface_factor = matrix.surface_factor(max(grip_rel, 1e-9))
    traffic_speed_component = speed_drop / matrix.traffic_speed_reference
    traffic_direction_component = direction_changes / matrix.traffic_direction_reference
    traffic_score = max(0.0, 0.5 * (traffic_speed_component + traffic_direction_component))
    traffic_factor = matrix.traffic_factor(traffic_score)
    return ContextFactors(curve_factor, surface_factor, traffic_factor)


def resolve_series_context(
    series: Sequence[SupportsContextBundle | SupportsContextRecord | Mapping[str, object]],
    *,
    matrix: ContextMatrix | None = None,
    baseline_vertical_load: float | None = None,
) -> list[ContextFactors]:
    """Return context factors for every element in ``series``."""

    matrix = matrix or load_context_matrix()
    factors: list[ContextFactors] = []
    for entry in series:
        if isinstance(entry, SupportsContextBundle):
            factors.append(resolve_context_from_bundle(matrix, entry))
        elif isinstance(entry, Mapping) and "tyres" in entry:
            factors.append(resolve_context_from_bundle(matrix, entry))
        else:
            factors.append(
                resolve_context_from_record(
                    matrix,
                    entry,
                    baseline_vertical_load=baseline_vertical_load,
                )
            )
    return factors
