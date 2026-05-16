"""Telemetry → structural observations.

LFS exposes only a *subset* of true vehicle state through OutSim/OutGauge
(no per-tyre forces, slip, temps). We therefore reconstruct **structural
proxies** that are consistent with the nodal equation:

* per-corner *vertical load proxy* from longitudinal + lateral acceleration
  (quasi-static load transfer with a Formula-class default mass distribution);
* per-corner *lateral demand* from yaw-rate and steered axle geometry;
* per-corner *longitudinal demand* from accel/brake plus driven-axle policy;
* a *thermal proxy* that integrates ``demand²·dt`` per corner (since LFS
  does not expose tyre temps in OutSim/OutGauge, this proxy stands in for
  the structural strain accumulated by each corner).

These quantities are intentionally dimensionless or normalized so that the
downstream analysis layers can consume corner observables uniformly.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from .live import TelemetrySample
from .protocol.packets import WHEEL_ORDER


CORNERS: tuple[str, ...] = ("FL", "FR", "RL", "RR")


# Per-car physical defaults indexed by LFS short car name.
# weight_dist_front comes from official LFS car data sheets; cg_height is a
# best-effort estimate. Mass values are unladen + driver (~75 kg).
#
# mu_lat / mu_long are *peak* tyre coefficients used by the quasi-static
# racing-line model. They are conservative defaults derived from category:
#   - road tyres, no aero  : ~1.00 / 0.95
#   - sport / road super   : ~1.10 / 1.00
#   - GTR (slicks + aero)  : ~1.30 / 1.20
#   - Formula (slicks+wing): ~1.45-1.70 / 1.25-1.45
# Real values vary with setup, fuel and tyre wear; use ``calibrate.py`` to
# refine them from telemetry per car/track.
_CAR_SPECS: dict[str, dict[str, float | str]] = {
    # ---- Road / street ----
    "UF1": dict(mass_kg=725.0,  wheelbase_m=2.13, track_front_m=1.30,
                track_rear_m=1.30, cg_height_m=0.50,
                weight_dist_front=0.62, driven="FWD",
                mu_lat=0.95, mu_long=0.90),
    "XFG": dict(mass_kg=975.0,  wheelbase_m=2.43, track_front_m=1.40,
                track_rear_m=1.40, cg_height_m=0.50,
                weight_dist_front=0.61, driven="FWD",
                mu_lat=1.00, mu_long=0.95),
    "XRG": dict(mass_kg=1100.0, wheelbase_m=2.55, track_front_m=1.42,
                track_rear_m=1.42, cg_height_m=0.50,
                weight_dist_front=0.55, driven="RWD",
                mu_lat=1.00, mu_long=0.95),
    "LX4": dict(mass_kg=620.0,  wheelbase_m=2.30, track_front_m=1.45,
                track_rear_m=1.40, cg_height_m=0.40,
                weight_dist_front=0.46, driven="RWD",
                mu_lat=1.10, mu_long=1.00),
    "LX6": dict(mass_kg=665.0,  wheelbase_m=2.30, track_front_m=1.45,
                track_rear_m=1.40, cg_height_m=0.40,
                weight_dist_front=0.47, driven="RWD",
                mu_lat=1.15, mu_long=1.05),
    "RB4": dict(mass_kg=1180.0, wheelbase_m=2.50, track_front_m=1.46,
                track_rear_m=1.46, cg_height_m=0.50,
                weight_dist_front=0.55, driven="AWD",
                mu_lat=1.05, mu_long=1.00),
    "FXO": dict(mass_kg=1180.0, wheelbase_m=2.55, track_front_m=1.48,
                track_rear_m=1.48, cg_height_m=0.50,
                weight_dist_front=0.60, driven="FWD",
                mu_lat=1.05, mu_long=1.00),
    "XRT": dict(mass_kg=1200.0, wheelbase_m=2.55, track_front_m=1.46,
                track_rear_m=1.46, cg_height_m=0.50,
                weight_dist_front=0.55, driven="RWD",
                mu_lat=1.05, mu_long=1.00),
    "RAC": dict(mass_kg=840.0,  wheelbase_m=2.36, track_front_m=1.45,
                track_rear_m=1.42, cg_height_m=0.42,
                weight_dist_front=0.46, driven="RWD",
                mu_lat=1.15, mu_long=1.05),
    "FZ5": dict(mass_kg=1380.0, wheelbase_m=2.27, track_front_m=1.50,
                track_rear_m=1.55, cg_height_m=0.45,
                weight_dist_front=0.40, driven="RWD",
                mu_lat=1.10, mu_long=1.00),
    "FZ50": dict(mass_kg=1380.0, wheelbase_m=2.27, track_front_m=1.50,
                 track_rear_m=1.55, cg_height_m=0.45,
                 weight_dist_front=0.40, driven="RWD",
                 mu_lat=1.10, mu_long=1.00),
    # ---- GTR (slicks + aero) ----
    "UFR": dict(mass_kg=720.0,  wheelbase_m=2.13, track_front_m=1.40,
                track_rear_m=1.40, cg_height_m=0.40,
                weight_dist_front=0.55, driven="FWD",
                mu_lat=1.30, mu_long=1.20),
    "XFR": dict(mass_kg=950.0,  wheelbase_m=2.43, track_front_m=1.50,
                track_rear_m=1.50, cg_height_m=0.40,
                weight_dist_front=0.58, driven="FWD",
                mu_lat=1.30, mu_long=1.20),
    "FXR": dict(mass_kg=1180.0, wheelbase_m=2.55, track_front_m=1.55,
                track_rear_m=1.55, cg_height_m=0.40,
                weight_dist_front=0.55, driven="AWD",
                mu_lat=1.35, mu_long=1.25, mu_lat_aero_k=5e-5),
    "XRR": dict(mass_kg=1180.0, wheelbase_m=2.55, track_front_m=1.55,
                track_rear_m=1.55, cg_height_m=0.40,
                weight_dist_front=0.50, driven="RWD",
                mu_lat=1.35, mu_long=1.25, mu_lat_aero_k=5e-5),
    "FZR": dict(mass_kg=1180.0, wheelbase_m=2.27, track_front_m=1.55,
                track_rear_m=1.60, cg_height_m=0.40,
                weight_dist_front=0.42, driven="RWD",
                mu_lat=1.35, mu_long=1.25, mu_lat_aero_k=5e-5),
    # ---- Formula / single-seater ----
    # FBM mass tuned from live LFS telemetry (~5400 N rest sum incl. ~50% fuel,
    # 75 kg driver). LFS spec sheet: 455 kg dry, ~140 hp 1.2L turbo, RWD.
    "FBM": dict(mass_kg=545.0, wheelbase_m=2.59, track_front_m=1.42,
                track_rear_m=1.36, cg_height_m=0.28,
                weight_dist_front=0.447, driven="RWD",
                mu_lat=1.40, mu_long=1.20),
    "MRT": dict(mass_kg=235.0, wheelbase_m=1.75, track_front_m=1.30,
                track_rear_m=1.30, cg_height_m=0.28,
                weight_dist_front=0.40, driven="RWD",
                mu_lat=1.20, mu_long=1.10),
    "FOX": dict(mass_kg=600.0, wheelbase_m=2.55, track_front_m=1.50,
                track_rear_m=1.40, cg_height_m=0.30,
                weight_dist_front=0.42, driven="RWD",
                mu_lat=1.50, mu_long=1.30, mu_lat_aero_k=1e-4),
    "FO8": dict(mass_kg=640.0, wheelbase_m=2.80, track_front_m=1.65,
                track_rear_m=1.55, cg_height_m=0.30,
                weight_dist_front=0.42, driven="RWD",
                mu_lat=1.60, mu_long=1.35, mu_lat_aero_k=1.5e-4),
    "BF1": dict(mass_kg=605.0, wheelbase_m=3.10, track_front_m=1.80,
                track_rear_m=1.55, cg_height_m=0.28,
                weight_dist_front=0.45, driven="RWD",
                mu_lat=1.70, mu_long=1.45, mu_lat_aero_k=2.5e-4),
}


@dataclass(slots=True, frozen=True)
class CarSpec:
    """Static car parameters used to map global telemetry to per-corner load.

    Defaults are tuned for an LFS *Formula* class (FOX/FO8). The mass split
    and CG height are quasi-canonical; tweak per car for higher fidelity.
    """

    mass_kg: float = 600.0           # incl. driver, full tank
    wheelbase_m: float = 2.55
    track_front_m: float = 1.50
    track_rear_m: float = 1.40
    cg_height_m: float = 0.30
    weight_dist_front: float = 0.42  # fraction of static load on front axle
    driven: str = "RWD"              # "RWD" | "FWD" | "AWD"
    mu_lat: float = 1.40             # peak lateral tyre coefficient at v=0
    mu_long: float = 1.20            # peak longitudinal tyre coefficient
    # Effective lateral grip grows with downforce: F_aero ∝ v^2 → normalized
    # μ gain ∝ v^2. ``mu_lat_aero_k`` is the slope of μ_lat over v^2 in
    # (m/s)^-2. Zero for road cars without aero. Tuned per-car via
    # :func:`lfs_telemetry.telemetry.calibrate.estimate_mu_lat_curve`.
    mu_lat_aero_k: float = 0.0
    g: float = 9.81

    def mu_lat_at(self, speed_ms: float | np.ndarray) -> float | np.ndarray:
        """Effective lateral grip coefficient at the given speed.

        Linear in v² (downforce model). Capped at ``2 × mu_lat`` so the
        extrapolation doesn't blow up at very high v.
        """
        if self.mu_lat_aero_k == 0.0:
            return self.mu_lat
        v2 = np.asarray(speed_ms, dtype=float) ** 2
        mu = self.mu_lat + self.mu_lat_aero_k * v2
        return np.minimum(mu, 2.0 * self.mu_lat)

    def static_corner_loads_n(self) -> dict[str, float]:
        wf = self.weight_dist_front
        wr = 1.0 - wf
        front = self.mass_kg * self.g * wf / 2.0
        rear = self.mass_kg * self.g * wr / 2.0
        return {"FL": front, "FR": front, "RL": rear, "RR": rear}


def car_spec_for(car_name: str | None) -> CarSpec:
    """Return a :class:`CarSpec` tuned for the given LFS car short name.

    Resolution order:

    1. User registry at ``$LFS_TELEMETRY_CARS_JSON`` or ``./config/cars.json``
       (entries here override or extend the built-in table — useful for
       mod cars whose ``.vob`` files are encrypted and not parseable).
    2. Official ``<car>_CAR_info.bin`` exported by LFS, located under
       ``$LFS_TELEMETRY_CAR_INFO_DIR`` (or ``./assets/source/`` /
       ``./assets/source/cars/`` as last-resort defaults). This is the
       ground truth for mass / wheelbase / track / CG / drivetrain.
    3. Built-in ``_CAR_SPECS`` table for the official 20 LFS cars.
    4. Generic FOX-class defaults if nothing is found.
    """
    if not car_name:
        return CarSpec()
    key = car_name.strip().upper()
    user = _load_user_registry()
    data = user.get(key) or user.get(key[:4]) or user.get(key[:3])
    if data is None:
        bin_kwargs = _load_car_info_bin_kwargs(key)
        if bin_kwargs is not None:
            data = bin_kwargs
    if data is None:
        data = _CAR_SPECS.get(key[:4]) or _CAR_SPECS.get(key[:3])
    if data is None:
        return CarSpec()
    # Filter to known fields so unknown JSON keys don't blow up the dataclass.
    known = {f for f in CarSpec.__dataclass_fields__}
    clean = {k: v for k, v in data.items() if k in known}
    return CarSpec(**clean)  # type: ignore[arg-type]


_CAR_INFO_BIN_CACHE: dict[str, dict | None] = {}
if TYPE_CHECKING:  # pragma: no cover
    from .car_info_bin import CarInfoBin
_CAR_INFO_BIN_FULL_CACHE: dict[str, "CarInfoBin | None"] = {}
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]

# Guards module-level caches (_CAR_INFO_BIN_CACHE, _CAR_INFO_BIN_FULL_CACHE,
# _REGISTRY_CACHE, _REGISTRY_PATH_CACHE). Live capture and Studio UI run on
# different threads, both of which can trigger registry lookups; the lock
# turns the check-then-fill pattern into a critical section. Reentrant so
# _load_car_info_bin_kwargs can call load_car_info_bin_for under the same
# lock without deadlocking.
_CACHE_LOCK = threading.RLock()


def _asset_search_dirs(env_var: str, *subpath: str) -> list[Path]:
    """Build the standard search path for asset files.

    Order: ``$env_var`` (if set) → ``./<subpath>`` → ``<package_root>/<subpath>``.
    """
    out: list[Path] = []
    env = os.environ.get(env_var)
    if env:
        out.append(Path(env))
    out.append(Path.cwd().joinpath(*subpath))
    out.append(_PACKAGE_ROOT.joinpath(*subpath))
    return out


def _car_info_bin_search_dirs() -> list[Path]:
    """Search dirs used to locate ``<key>_CAR_info.bin`` exports."""
    search = _asset_search_dirs(
        "LFS_TELEMETRY_CAR_INFO_DIR", "assets", "source", "cars")
    # Fall back to the parent dir (assets/source) when the cars/ folder
    # is absent — LFS exports may sit at either level.
    search.extend([d.parent for d in list(search) if d.name == "cars"])
    return search


def _car_info_bin_candidates(key: str) -> tuple[str, ...]:
    return (f"{key}_CAR_info.bin", f"{key[:4]}_CAR_info.bin",
            f"{key[:3]}_CAR_info.bin")


def load_car_info_bin_for(key: str) -> "CarInfoBin | None":
    """Locate and parse ``<key>_CAR_info.bin``, returning the full record.

    Mirrors :func:`_load_car_info_bin_kwargs` (same search path and
    candidate filenames) but returns the entire :class:`CarInfoBin`
    instead of just :class:`CarSpec` kwargs. Cached per ``key``.
    """
    key = (key or "").upper()
    if not key:
        return None
    with _CACHE_LOCK:
        if key in _CAR_INFO_BIN_FULL_CACHE:
            return _CAR_INFO_BIN_FULL_CACHE[key]
        from .car_info_bin import parse_car_info_bin
        for d in _car_info_bin_search_dirs():
            for name in _car_info_bin_candidates(key):
                p = d / name
                if p.exists():
                    try:
                        info = parse_car_info_bin(p)
                    except (OSError, ValueError, struct.error):
                        continue
                    _CAR_INFO_BIN_FULL_CACHE[key] = info
                    return info
        _CAR_INFO_BIN_FULL_CACHE[key] = None
        return None


def _load_car_info_bin_kwargs(key: str) -> dict | None:
    """Locate ``<key>_CAR_info.bin`` and return CarSpec kwargs, or None.

    Cached per ``key`` so repeated lookups are O(1). Use
    :func:`reload_car_registry` to clear.
    """
    if key in _CAR_INFO_BIN_CACHE:
        return _CAR_INFO_BIN_CACHE[key]
    with _CACHE_LOCK:
        if key in _CAR_INFO_BIN_CACHE:
            return _CAR_INFO_BIN_CACHE[key]
        info = load_car_info_bin_for(key)
        if info is None:
            _CAR_INFO_BIN_CACHE[key] = None
            return None
        kwargs = info.to_car_spec_kwargs()
        _CAR_INFO_BIN_CACHE[key] = kwargs
        return kwargs


_REGISTRY_CACHE: dict[str, dict[str, float | str]] | None = None
_REGISTRY_PATH_CACHE: Path | None = None


def _load_user_registry() -> dict[str, dict[str, float | str]]:
    """Load and cache the user JSON registry, if present.

    Resolves the path from ``$LFS_TELEMETRY_CARS_JSON``, falling back to
    ``<cwd>/config/cars.json`` and ``<package>/config/cars.json``.
    """
    global _REGISTRY_CACHE, _REGISTRY_PATH_CACHE
    # Same env→cwd→package fallback order as _car_info_bin_search_dirs,
    # but the env var is treated as a direct file path (not a dir).
    candidates = _asset_search_dirs(
        "LFS_TELEMETRY_CARS_JSON", "config", "cars.json"
    )
    path = next((p for p in candidates if p.exists()), None)
    with _CACHE_LOCK:
        if path != _REGISTRY_PATH_CACHE:
            _REGISTRY_CACHE = None
            _REGISTRY_PATH_CACHE = path
        if _REGISTRY_CACHE is not None:
            return _REGISTRY_CACHE
        if path is None:
            _REGISTRY_CACHE = {}
            return _REGISTRY_CACHE
        try:
            with path.open("r", encoding="utf-8") as fh:
                blob = json.load(fh)
            cars = blob.get("cars", {}) if isinstance(blob, dict) else {}
            _REGISTRY_CACHE = {str(k).upper(): v for k, v in cars.items()
                               if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError):
            _REGISTRY_CACHE = {}
        return _REGISTRY_CACHE


def reload_car_registry() -> None:
    """Force the user registry to be re-read on the next lookup."""
    global _REGISTRY_CACHE, _REGISTRY_PATH_CACHE
    with _CACHE_LOCK:
        _REGISTRY_CACHE = None
        _REGISTRY_PATH_CACHE = None
        _CAR_INFO_BIN_CACHE.clear()
        _CAR_INFO_BIN_FULL_CACHE.clear()


@dataclass(slots=True)
class StructuralObservation:
    """Per-sample structural state for the car-as-NFR."""

    time_s: float
    speed_ms: float
    yaw_rate: float                                  # rad/s
    pitch: float                                     # rad
    roll: float                                      # rad
    long_accel: float                                # m/s² (car frame, +fwd)
    lat_accel: float                                 # m/s² (car frame, +right)
    vert_accel: float                                # m/s² (car frame, +down approx)
    throttle: float
    brake: float
    corner_load_n: dict[str, float] = field(default_factory=dict)
    corner_lateral_demand: dict[str, float] = field(default_factory=dict)
    corner_long_demand: dict[str, float] = field(default_factory=dict)


def observe_sample(sample: TelemetrySample, spec: CarSpec) -> StructuralObservation:
    """Project one fused sample into structural observables.

    When ``sample.outsim2.wheels`` is present (extended OutSim with
    OSO_WHEELS), use real per-wheel ``vertical_load_n``, ``slip_ratio`` and
    ``tan_slip_angle`` instead of quasi-static estimates.
    """
    if not sample.is_complete:
        raise ValueError("sample must contain both OutSim and OutGauge data")
    os_pkt = sample.outsim
    og_pkt = sample.outgauge
    assert os_pkt is not None and og_pkt is not None

    # OutSim accel is in the car local frame: +x forward, +y right, +z down.
    long_a, lat_a, vert_a = os_pkt.accel
    yaw_rate = os_pkt.ang_vel[2]

    real_wheels = (
        sample.outsim2.wheels if sample.outsim2 is not None else None
    )

    if real_wheels is not None and len(real_wheels) == 4:
        # Real per-corner data path. WHEEL_ORDER from LFS is RL, RR, FL, FR;
        # CORNERS uses FL, FR, RL, RR — remap explicitly.
        by_lfs = dict(zip(WHEEL_ORDER, real_wheels))
        loads = {c: float(by_lfs[c].vertical_load_n) for c in CORNERS}
        # Lateral demand: load × |tan(slip angle)| (proxy for cornering work).
        lat_demand = {
            c: float(by_lfs[c].vertical_load_n
                     * abs(by_lfs[c].tan_slip_angle))
            for c in CORNERS
        }
        # Longitudinal demand: load × |slip ratio| (proxy for tractive work).
        long_demand = {
            c: float(by_lfs[c].vertical_load_n
                     * abs(by_lfs[c].slip_ratio))
            for c in CORNERS
        }
    else:
        # Quasi-static fallback (basic OutSim only).
        static = spec.static_corner_loads_n()
        dF_long = (
            spec.mass_kg * long_a * spec.cg_height_m / spec.wheelbase_m
        )
        dF_lat_front = (
            spec.mass_kg * lat_a * spec.cg_height_m / spec.track_front_m
            * spec.weight_dist_front
        )
        dF_lat_rear = (
            spec.mass_kg * lat_a * spec.cg_height_m / spec.track_rear_m
            * (1.0 - spec.weight_dist_front)
        )
        loads = {
            "FL": max(0.0, static["FL"] - dF_long / 2.0 - dF_lat_front / 2.0),
            "FR": max(0.0, static["FR"] - dF_long / 2.0 + dF_lat_front / 2.0),
            "RL": max(0.0, static["RL"] + dF_long / 2.0 - dF_lat_rear / 2.0),
            "RR": max(0.0, static["RR"] + dF_long / 2.0 + dF_lat_rear / 2.0),
        }
        a_y_g = lat_a / spec.g
        lat_demand = {c: loads[c] * abs(a_y_g) for c in CORNERS}
        a_x_g = long_a / spec.g
        driven = spec.driven.upper()
        drive_corners = {
            "RWD": ("RL", "RR"),
            "FWD": ("FL", "FR"),
            "AWD": CORNERS,
        }.get(driven, ("RL", "RR"))
        long_demand = {}
        for c in CORNERS:
            if a_x_g >= 0.0:
                long_demand[c] = loads[c] * a_x_g if c in drive_corners else 0.0
            else:
                long_demand[c] = loads[c] * abs(a_x_g)

    return StructuralObservation(
        time_s=sample.time_ms / 1000.0,
        speed_ms=og_pkt.speed_ms,
        yaw_rate=yaw_rate,
        pitch=os_pkt.pitch,
        roll=os_pkt.roll,
        long_accel=long_a,
        lat_accel=lat_a,
        vert_accel=vert_a,
        throttle=og_pkt.throttle,
        brake=og_pkt.brake,
        corner_load_n=loads,
        corner_lateral_demand=lat_demand,
        corner_long_demand=long_demand,
    )


def observe_window(
    samples: Iterable[TelemetrySample],
    spec: CarSpec | None = None,
) -> pd.DataFrame:
    """Vectorize a stream of samples into a tidy DataFrame for analysis."""
    spec = spec or CarSpec()
    rows = []
    for s in samples:
        if not s.is_complete:
            continue
        obs = observe_sample(s, spec)
        row: dict[str, float] = {
            "time_s": obs.time_s,
            "speed_ms": obs.speed_ms,
            "yaw_rate": obs.yaw_rate,
            "pitch": obs.pitch,
            "roll": obs.roll,
            "long_accel": obs.long_accel,
            "lat_accel": obs.lat_accel,
            "vert_accel": obs.vert_accel,
            "throttle": obs.throttle,
            "brake": obs.brake,
        }
        for c in CORNERS:
            row[f"load_{c}"] = obs.corner_load_n[c]
            row[f"lat_dem_{c}"] = obs.corner_lateral_demand[c]
            row[f"long_dem_{c}"] = obs.corner_long_demand[c]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("time_s").reset_index(drop=True)
    return df
