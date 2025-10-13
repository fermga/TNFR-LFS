"""Parity tests between the local TNFR engine and the canonical extra."""

from __future__ import annotations

import importlib
import math
import os
import sys
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("tnfr")


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


if not _is_truthy(os.getenv("TNFR_CANONICAL")):
    pytest.skip("Canonical TNFR extra not active", allow_module_level=True)


_DEFAULT_REL_TOLERANCES: Mapping[str, float] = {
    "epi": 5e-5,
    "delta_nfr": 5e-5,
    "sense_index": 5e-5,
    "nu_f": 5e-4,
    "phase_alignment": 1e-3,
}


def _resolve_rel_tol(metric: str) -> float:
    env_key = f"TNFR_PARITY_{metric.upper()}_REL_TOL"
    override = os.getenv(env_key)
    if override is not None:
        try:
            return float(override)
        except ValueError:
            raise ValueError(f"Invalid tolerance override for {metric}: {override!r}") from None
    return float(_DEFAULT_REL_TOLERANCES[metric])


def _clear_tnfr_core_modules() -> None:
    for name in list(sys.modules):
        if name == "tnfr_core" or name.startswith("tnfr_core."):
            sys.modules.pop(name, None)


def _load_tnfr_core(canonical_flag: str) -> Any:
    original = os.getenv("TNFR_CANONICAL")
    try:
        os.environ["TNFR_CANONICAL"] = canonical_flag
        _clear_tnfr_core_modules()
        return importlib.import_module("tnfr_core")
    finally:
        if original is None:
            os.environ.pop("TNFR_CANONICAL", None)
        else:
            os.environ["TNFR_CANONICAL"] = original


def _generate_synthetic_payload(
    *,
    sample_count: int = 64,
    sample_rate: float = 20.0,
    time_scale: float = 1.0,
    load_offset: float = 0.0,
) -> List[Dict[str, Any]]:
    dt = time_scale / sample_rate
    base_load = 5000.0 + load_offset
    payload: List[Dict[str, Any]] = []
    for index in range(sample_count):
        t = index * dt
        vertical_load = base_load + 320.0 * math.sin(2.0 * math.pi * 0.45 * t)
        slip_ratio = 0.03 + 0.01 * math.sin(2.0 * math.pi * 0.3 * t + 0.2)
        lateral_accel = 1.8 * math.sin(2.0 * math.pi * 0.25 * t)
        longitudinal_accel = 0.9 * math.cos(2.0 * math.pi * 0.33 * t)
        yaw = 0.02 * math.sin(2.0 * math.pi * 0.18 * t)
        pitch = 0.01 * math.cos(2.0 * math.pi * 0.12 * t)
        roll = 0.015 * math.sin(2.0 * math.pi * 0.21 * t + 0.3)
        brake_pressure = 0.35 + 0.25 * max(0.0, math.sin(2.0 * math.pi * 0.1 * t))
        locking = -0.1 * abs(math.sin(2.0 * math.pi * 0.35 * t))
        nfr = 1.0 + 0.05 * math.sin(2.0 * math.pi * 0.4 * t + 0.1)
        si = 0.55 + 0.12 * math.cos(2.0 * math.pi * 0.38 * t)
        speed = 62.0 + 6.0 * math.sin(2.0 * math.pi * 0.22 * t + 0.4)
        yaw_rate = 0.45 * math.cos(2.0 * math.pi * 0.3 * t)
        slip_angle = 0.025 * math.sin(2.0 * math.pi * 0.36 * t + 0.15)
        steer = 0.12 * math.sin(2.0 * math.pi * 0.28 * t)
        throttle = 0.52 + 0.28 * max(0.0, math.sin(2.0 * math.pi * 0.18 * t + 0.6))
        gear = 3 + (index // 16) % 3
        vertical_load_front = 0.53 * vertical_load
        vertical_load_rear = vertical_load - vertical_load_front
        mu_eff_front = 1.38 + 0.04 * math.sin(2.0 * math.pi * 0.27 * t)
        mu_eff_rear = 1.33 + 0.05 * math.cos(2.0 * math.pi * 0.29 * t)
        mu_eff_front_lateral = mu_eff_front * (0.92 + 0.02 * math.sin(2.0 * math.pi * 0.24 * t))
        mu_eff_front_longitudinal = mu_eff_front * (0.88 + 0.03 * math.cos(2.0 * math.pi * 0.26 * t))
        mu_eff_rear_lateral = mu_eff_rear * (0.9 + 0.025 * math.sin(2.0 * math.pi * 0.31 * t))
        mu_eff_rear_longitudinal = mu_eff_rear * (0.86 + 0.02 * math.cos(2.0 * math.pi * 0.33 * t))
        suspension_travel_front = 0.03 + 0.005 * math.sin(2.0 * math.pi * 0.6 * t)
        suspension_travel_rear = 0.028 + 0.004 * math.sin(2.0 * math.pi * 0.52 * t + 0.2)
        front_velocity = 0.005 * 2.0 * math.pi * 0.6 * math.cos(2.0 * math.pi * 0.6 * t) * time_scale
        rear_velocity = 0.004 * 2.0 * math.pi * 0.52 * math.cos(2.0 * math.pi * 0.52 * t + 0.2) * time_scale
        payload.append(
            {
                "timestamp": t,
                "vertical_load": vertical_load,
                "slip_ratio": slip_ratio,
                "lateral_accel": lateral_accel,
                "longitudinal_accel": longitudinal_accel,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "brake_pressure": brake_pressure,
                "locking": locking,
                "nfr": nfr,
                "si": si,
                "speed": speed,
                "yaw_rate": yaw_rate,
                "slip_angle": slip_angle,
                "steer": steer,
                "throttle": throttle,
                "gear": gear,
                "vertical_load_front": vertical_load_front,
                "vertical_load_rear": vertical_load_rear,
                "mu_eff_front": mu_eff_front,
                "mu_eff_rear": mu_eff_rear,
                "mu_eff_front_lateral": mu_eff_front_lateral,
                "mu_eff_front_longitudinal": mu_eff_front_longitudinal,
                "mu_eff_rear_lateral": mu_eff_rear_lateral,
                "mu_eff_rear_longitudinal": mu_eff_rear_longitudinal,
                "suspension_travel_front": suspension_travel_front,
                "suspension_travel_rear": suspension_travel_rear,
                "suspension_velocity_front": front_velocity,
                "suspension_velocity_rear": rear_velocity,
            }
        )
    return payload


_NU_F_NODES: Tuple[str, ...] = (
    "tyres",
    "suspension",
    "chassis",
    "brakes",
    "transmission",
    "track",
    "driver",
)


def _summarise_outputs(bundles: Iterable[Any], microsectors: Iterable[Any]) -> Dict[str, Any]:
    bundle_list = list(bundles)
    microsector_list = list(microsectors)
    epi = np.array([float(bundle.epi) for bundle in bundle_list], dtype=float)
    delta_nfr = np.array([float(bundle.delta_nfr) for bundle in bundle_list], dtype=float)
    sense_index = np.array([float(bundle.sense_index) for bundle in bundle_list], dtype=float)
    nu_f_matrix = np.array(
        [
            [float(getattr(bundle, node).nu_f) for node in _NU_F_NODES]
            for bundle in bundle_list
        ],
        dtype=float,
    )
    alignment_totals: Dict[str, float] = {}
    alignment_counts: Dict[str, int] = {}
    active_phases: List[str] = []
    for sector in microsector_list:
        active_phases.append(str(sector.active_phase))
        for phase, value in getattr(sector, "phase_alignment", {}).items():
            phase_key = str(phase)
            alignment_totals[phase_key] = alignment_totals.get(phase_key, 0.0) + float(value)
            alignment_counts[phase_key] = alignment_counts.get(phase_key, 0) + 1
    phase_alignment = {
        phase: alignment_totals[phase] / alignment_counts[phase]
        for phase in alignment_totals
        if alignment_counts[phase] > 0
    }
    return {
        "epi": epi,
        "delta_nfr": delta_nfr,
        "sense_index": sense_index,
        "nu_f": nu_f_matrix,
        "phase_alignment": phase_alignment,
        "active_phases": tuple(active_phases),
    }


def _compute_engine_outputs(samples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fallback_core = _load_tnfr_core("0")
    fallback_records = [fallback_core.TelemetryRecord(**row) for row in samples]
    fallback_bundles = fallback_core.EPIExtractor().extract(fallback_records)
    fallback_micro = fallback_core.segment_microsectors(fallback_records, fallback_bundles)
    fallback_summary = _summarise_outputs(fallback_bundles, fallback_micro)

    canonical_core = _load_tnfr_core("1")
    canonical_records = [canonical_core.TelemetryRecord(**row) for row in samples]
    canonical_bundles = canonical_core.EPIExtractor().extract(canonical_records)
    canonical_micro = canonical_core.segment_microsectors(canonical_records, canonical_bundles)
    canonical_summary = _summarise_outputs(canonical_bundles, canonical_micro)

    return fallback_summary, canonical_summary


def _assert_close(metric: str, lhs: np.ndarray, rhs: np.ndarray) -> None:
    assert lhs.shape == rhs.shape, f"Shape mismatch for {metric}: {lhs.shape} vs {rhs.shape}"
    if lhs.size == 0:
        return
    tol = _resolve_rel_tol(metric)
    npt.assert_allclose(lhs, rhs, rtol=tol, atol=tol)


def _assert_phase_alignment_close(lhs: Mapping[str, float], rhs: Mapping[str, float]) -> None:
    tol = _resolve_rel_tol("phase_alignment")
    assert set(lhs) == set(rhs), f"Phase keys differ: {set(lhs)} vs {set(rhs)}"
    for phase in sorted(lhs):
        l_value = float(lhs[phase])
        r_value = float(rhs[phase])
        if l_value == r_value == 0.0:
            continue
        diff = abs(l_value - r_value)
        reference = max(abs(l_value), abs(r_value), 1.0)
        assert diff <= tol * reference, f"Phase {phase} differs by {diff} (> {tol * reference})"


def _assert_parity(samples: List[Dict[str, Any]]) -> None:
    fallback_summary, canonical_summary = _compute_engine_outputs(samples)

    _assert_close("epi", fallback_summary["epi"], canonical_summary["epi"])
    _assert_close("delta_nfr", fallback_summary["delta_nfr"], canonical_summary["delta_nfr"])
    _assert_close("sense_index", fallback_summary["sense_index"], canonical_summary["sense_index"])
    _assert_close("nu_f", fallback_summary["nu_f"], canonical_summary["nu_f"])
    _assert_phase_alignment_close(
        fallback_summary["phase_alignment"], canonical_summary["phase_alignment"]
    )
    assert fallback_summary["active_phases"] == canonical_summary["active_phases"]


def test_tnfr_engine_parity_baseline() -> None:
    samples = _generate_synthetic_payload()
    _assert_parity(samples)


@given(
    time_scale=st.floats(min_value=0.5, max_value=1.8, allow_nan=False, allow_infinity=False),
    load_offset=st.floats(min_value=-600.0, max_value=600.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=12, deadline=None)
def test_tnfr_engine_parity_under_scaling(time_scale: float, load_offset: float) -> None:
    samples = _generate_synthetic_payload(time_scale=time_scale, load_offset=load_offset)
    _assert_parity(samples)

