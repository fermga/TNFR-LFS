"""Auto-calibrate per-car physical specs from live LFS telemetry.

When the car is at rest on a flat surface, the four vertical wheel loads sum
to ``mass·g`` and the front/rear ratio yields ``weight_dist_front`` directly.
This works for *any* car including paid-content mods, because OutSim Mode 2
emits the per-wheel vertical load regardless of the car model.

Calibrations are persisted to a small JSON store keyed by the OutGauge ``car``
field (which is the LFS short name for stock cars and the mod ID for mods).
At observe-time, we prefer (1) a saved calibration, (2) the bundled
``_CAR_SPECS`` defaults, (3) generic Formula-class defaults.
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .constants import GRAVITY
from .live import TelemetrySample
from .observables import _CAR_SPECS, CarSpec

_LOG = logging.getLogger(__name__)

# Defaults for unknown classes — wheelbase / track / CG cannot be observed
# from telemetry, so we keep generic Formula values. mass + weight_dist
# are overwritten by the rest-state measurement.
_GENERIC_FALLBACK = {
    "wheelbase_m": 2.55,
    "track_front_m": 1.50,
    "track_rear_m": 1.40,
    "cg_height_m": 0.30,
    "driven": "RWD",
}

# Window of samples used to detect "at rest". 1 s @ 100 Hz.
_REST_WINDOW = 100
# Thresholds: speed and acceleration both close to zero, throttle/brake idle.
# NOTE: real LFS rest telemetry shows ~0.06 throttle floor and ~0.3 m/s² accel
# noise from engine vibration even when fully stopped — keep generous margins.
_REST_SPEED_MS = 1.0
_REST_ACCEL_MS2 = 1.0
_REST_INPUT = 0.15


def default_store_path() -> Path:
    """Return the default JSON store location: ``~/.lfs-telemetry/cars.json``."""
    env = os.environ.get("LFS_TELEMETRY_CAR_STORE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".lfs-telemetry" / "cars.json"


@dataclass(slots=True)
class CarCalibration:
    """A measured calibration for one car id.

    ``car_id`` is the raw OutGauge car field (uppercased, stripped). For stock
    cars this is the 4-letter short name (e.g. ``FBM``); for mods it's the
    LFS mod id (typically 4-6 alphanumeric chars).
    """

    car_id: str
    mass_kg: float
    weight_dist_front: float
    sample_count: int
    sum_load_n: float          # mean total static load observed (≈ mass·g)
    front_fraction: float      # mean (FL+FR) / total
    left_fraction: float       # mean (FL+RL) / total — for asymmetry detection
    mu_lat: float | None = None      # measured peak lateral grip (a_y/g)
    mu_long: float | None = None     # measured peak longitudinal grip
    mu_lat_aero_k: float | None = None  # μ_lat slope vs v², (m/s)^-2
    mu_sample_count: int = 0

    def to_dict(self) -> dict:
        d = {
            "car_id": self.car_id,
            "mass_kg": round(self.mass_kg, 2),
            "weight_dist_front": round(self.weight_dist_front, 4),
            "left_fraction": round(self.left_fraction, 4),
            "sample_count": self.sample_count,
            "sum_load_n": round(self.sum_load_n, 2),
            "front_fraction": round(self.front_fraction, 4),
        }
        if self.mu_lat is not None:
            d["mu_lat"] = round(self.mu_lat, 4)
        if self.mu_long is not None:
            d["mu_long"] = round(self.mu_long, 4)
        if self.mu_lat_aero_k is not None:
            d["mu_lat_aero_k"] = round(self.mu_lat_aero_k, 8)
        if self.mu_sample_count:
            d["mu_sample_count"] = self.mu_sample_count
        return d

    @classmethod
    def from_dict(cls, data: dict) -> CarCalibration:
        return cls(
            car_id=str(data["car_id"]).upper(),
            mass_kg=float(data["mass_kg"]),
            weight_dist_front=float(data["weight_dist_front"]),
            sample_count=int(data.get("sample_count", 0)),
            sum_load_n=float(data.get("sum_load_n", 0.0)),
            front_fraction=float(data.get("front_fraction", 0.5)),
            left_fraction=float(data.get("left_fraction", 0.5)),
            mu_lat=(float(data["mu_lat"]) if data.get("mu_lat") is not None
                    else None),
            mu_long=(float(data["mu_long"]) if data.get("mu_long") is not None
                     else None),
            mu_lat_aero_k=(float(data["mu_lat_aero_k"])
                           if data.get("mu_lat_aero_k") is not None else None),
            mu_sample_count=int(data.get("mu_sample_count", 0)),
        )

    def to_spec(self) -> CarSpec:
        """Promote this calibration to a full :class:`CarSpec`.

        Uses bundled ``_CAR_SPECS`` for geometry (wheelbase/track/CG) when
        the car id is a known stock car, otherwise falls back to generic
        Formula geometry. Mass and weight distribution come from the
        measurement itself.
        """
        key = self.car_id[:4]
        bundled = _CAR_SPECS.get(key, {})
        merged = dict(_GENERIC_FALLBACK)
        merged.update({k: v for k, v in bundled.items()
                       if k in {"wheelbase_m", "track_front_m",
                                "track_rear_m", "cg_height_m", "driven"}})
        if self.mass_kg > 0:
            merged["mass_kg"] = self.mass_kg
        elif "mass_kg" in bundled:
            merged["mass_kg"] = bundled["mass_kg"]
        if self.sample_count > 0 or "weight_dist_front" not in bundled:
            merged["weight_dist_front"] = (
                self.weight_dist_front if self.sample_count > 0
                else bundled.get("weight_dist_front", 0.5)
            )
        else:
            merged["weight_dist_front"] = bundled["weight_dist_front"]
        if self.mu_lat is not None:
            merged["mu_lat"] = self.mu_lat
        if self.mu_long is not None:
            merged["mu_long"] = self.mu_long
        if self.mu_lat_aero_k is not None:
            merged["mu_lat_aero_k"] = self.mu_lat_aero_k
        return CarSpec(**merged)  # type: ignore[arg-type]


class CarSpecStore:
    """JSON-backed store of per-car calibrations."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_store_path()
        self._cache: dict[str, CarCalibration] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("could not read %s: %s", self.path, exc)
            return
        for cid, data in (raw.get("cars") or {}).items():
            try:
                self._cache[cid.upper()] = CarCalibration.from_dict(data)
            except (KeyError, TypeError, ValueError) as exc:
                _LOG.warning("skipping invalid entry %s: %s", cid, exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "cars": {cid: cal.to_dict() for cid, cal in self._cache.items()},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, car_id: str | None) -> CarCalibration | None:
        if not car_id:
            return None
        self.load()
        return self._cache.get(car_id.strip().upper())

    def put(self, calibration: CarCalibration) -> None:
        self.load()
        self._cache[calibration.car_id.upper()] = calibration

    def all(self) -> dict[str, CarCalibration]:
        self.load()
        return dict(self._cache)

    def spec_for(self, car_id: str | None) -> CarSpec:
        """Resolve a CarSpec using (1) calibration store, (2) bundled
        ``_CAR_SPECS``, (3) generic Formula defaults."""
        cal = self.get(car_id)
        if cal is not None:
            return cal.to_spec()
        # fall through to bundled / defaults
        from .observables import car_spec_for
        return car_spec_for(car_id)

    def update_mu_from_df(self, car_id: str, df) -> CarCalibration | None:
        """Estimate μ_lat/μ_long from an observed-telemetry DataFrame and
        merge them into the saved (or default) calibration for ``car_id``.

        Returns the updated :class:`CarCalibration` (also persisted in
        memory; call :meth:`save` to write to disk) or ``None`` if there
        was insufficient data to estimate either coefficient.

        The DataFrame must contain at least ``lat_accel`` and
        ``long_accel`` columns (the output of :func:`observe_window`).
        Mass and weight distribution are taken from any prior rest
        calibration; if none exists, generic placeholders are stored so
        that μ values can still be queried via :meth:`spec_for`.
        """
        import math

        from .calibrate import estimate_mu_lat, estimate_mu_lat_curve, estimate_mu_long

        mu0, k_aero, _n_bins = estimate_mu_lat_curve(df)
        mu_lat = mu0 if math.isfinite(mu0) else estimate_mu_lat(df)
        mu_long = estimate_mu_long(df)
        if not (math.isfinite(mu_lat) or math.isfinite(mu_long)):
            return None
        key = car_id.strip().upper()
        existing = self.get(key)
        if existing is None:
            existing = CarCalibration(
                car_id=key, mass_kg=0.0, weight_dist_front=0.5,
                sample_count=0, sum_load_n=0.0,
                front_fraction=0.5, left_fraction=0.5,
            )
        if math.isfinite(mu_lat):
            existing.mu_lat = mu_lat * 0.95
            existing.mu_lat_aero_k = (k_aero * 0.95
                                      if math.isfinite(k_aero) else 0.0)
        if math.isfinite(mu_long):
            existing.mu_long = mu_long * 0.95
        existing.mu_sample_count = len(df)
        self.put(existing)
        return existing


def _is_at_rest(samples: Iterable[TelemetrySample]) -> bool:
    """Return True if the entire window looks like a stationary, on-track car."""
    return _rest_failure(samples) is None


def _rest_failure(samples: Iterable[TelemetrySample]) -> str | None:
    """Return None if the window looks at rest, else a human-readable reason."""
    for s in samples:
        if not s.is_complete:
            return "incomplete sample"
        og = s.outgauge
        os_pkt = s.outsim
        if og is None or os_pkt is None:
            return "missing outsim/outgauge"
        if abs(og.speed_ms) > _REST_SPEED_MS:
            return f"moving (speed={og.speed_ms:.2f} m/s > {_REST_SPEED_MS})"
        if og.throttle > _REST_INPUT:
            return f"throttle={og.throttle:.2f} > {_REST_INPUT}"
        if og.brake > _REST_INPUT:
            return f"brake={og.brake:.2f} > {_REST_INPUT}"
        ax, ay, _ = os_pkt.accel
        mag = (ax * ax + ay * ay) ** 0.5
        if mag > _REST_ACCEL_MS2:
            return f"accel={mag:.2f} m/s² > {_REST_ACCEL_MS2}"
        ext = s.outsim2
        if ext is None or not ext.wheels or len(ext.wheels) != 4:
            return "no extended wheel data"
        for w in ext.wheels:
            if not w.touching:
                return "wheel airborne"
    return None


def _measure(samples: list[TelemetrySample], car_id: str) -> CarCalibration:
    """Average the four wheel loads across the window into a calibration."""
    from ..telemetry.observables import CORNERS
    from ..telemetry.protocol.packets import WHEEL_ORDER

    # Map LFS native order (RL,RR,FL,FR) into FL/FR/RL/RR.
    idx = {name: WHEEL_ORDER.index(name) for name in CORNERS}
    totals = dict.fromkeys(CORNERS, 0.0)
    n = 0
    for s in samples:
        ext = s.outsim2
        if ext is None or not ext.wheels:
            continue
        for name, i in idx.items():
            totals[name] += ext.wheels[i].vertical_load_n
        n += 1
    if n == 0:
        raise ValueError("no usable wheel samples")
    means = {k: v / n for k, v in totals.items()}
    total = sum(means.values())
    front = means["FL"] + means["FR"]
    left = means["FL"] + means["RL"]
    return CarCalibration(
        car_id=car_id.strip().upper(),
        mass_kg=total / GRAVITY,
        weight_dist_front=front / total if total > 0 else 0.5,
        sample_count=n,
        sum_load_n=total,
        front_fraction=front / total if total > 0 else 0.5,
        left_fraction=left / total if total > 0 else 0.5,
    )


class RestCalibrator:
    """Streaming detector that emits a calibration when the car is at rest.

    Usage::

        cal = RestCalibrator()
        for sample in stream:
            new = cal.feed(sample)
            if new is not None:
                store.put(new); store.save()
    """

    def __init__(self, window: int = _REST_WINDOW) -> None:
        self.window = window
        self._buffer: deque[TelemetrySample] = deque(maxlen=window)

    def feed(self, sample: TelemetrySample) -> CarCalibration | None:
        self._buffer.append(sample)
        if len(self._buffer) < self.window:
            return None
        og = sample.outgauge
        if og is None:
            return None
        car_id = (og.car or "").strip()
        if not car_id:
            return None
        if not _is_at_rest(self._buffer):
            return None
        cal = _measure(list(self._buffer), car_id)
        # consume the window so we don't re-emit on every tick
        self._buffer.clear()
        return cal

    def diagnose(self) -> str | None:
        """Return why the current buffer is *not* at rest, or None if it is.

        Useful for live UIs / CLIs that want to tell the user why the
        calibration hasn't fired yet.
        """
        if len(self._buffer) < self.window:
            return f"buffering ({len(self._buffer)}/{self.window})"
        return _rest_failure(self._buffer)


__all__ = [
    "CarCalibration",
    "CarSpecStore",
    "RestCalibrator",
    "default_store_path",
]
