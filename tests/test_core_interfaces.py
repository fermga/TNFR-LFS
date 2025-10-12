"""Tests covering structural typing protocols for EPI bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

import pytest

from tnfr_lfs.core.interfaces import (
    SupportsBrakesNode,
    SupportsChassisNode,
    SupportsDriverNode,
    SupportsEPIBundle,
    SupportsSuspensionNode,
    SupportsTrackNode,
    SupportsTransmissionNode,
    SupportsTyresNode,
)
from tnfr_lfs.core.metrics import WindowMetrics, compute_window_metrics
from tnfr_lfs.core.operators import DissonanceBreakdown, dissonance_breakdown_operator
from tests.helpers import build_telemetry_record


@dataclass
class MinimalTyresNode:
    delta_nfr: float = 0.0
    sense_index: float = 1.0
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    load: float = 0.0
    slip_ratio: float = 0.0
    mu_eff_front: float = 0.0
    mu_eff_rear: float = 0.0
    mu_eff_front_lateral: float = 0.0
    mu_eff_front_longitudinal: float = 0.0
    mu_eff_rear_lateral: float = 0.0
    mu_eff_rear_longitudinal: float = 0.0
    tyre_temp_fl: float = 0.0
    tyre_temp_fr: float = 0.0
    tyre_temp_rl: float = 0.0
    tyre_temp_rr: float = 0.0
    tyre_temp_fl_inner: float = 0.0
    tyre_temp_fr_inner: float = 0.0
    tyre_temp_rl_inner: float = 0.0
    tyre_temp_rr_inner: float = 0.0
    tyre_temp_fl_middle: float = 0.0
    tyre_temp_fr_middle: float = 0.0
    tyre_temp_rl_middle: float = 0.0
    tyre_temp_rr_middle: float = 0.0
    tyre_temp_fl_outer: float = 0.0
    tyre_temp_fr_outer: float = 0.0
    tyre_temp_rl_outer: float = 0.0
    tyre_temp_rr_outer: float = 0.0
    tyre_pressure_fl: float = 0.0
    tyre_pressure_fr: float = 0.0
    tyre_pressure_rl: float = 0.0
    tyre_pressure_rr: float = 0.0


@dataclass
class MinimalSuspensionNode:
    delta_nfr: float = 0.0
    sense_index: float = 1.0
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    travel_front: float = 0.0
    travel_rear: float = 0.0
    velocity_front: float = 0.0
    velocity_rear: float = 0.0


@dataclass
class MinimalChassisNode:
    delta_nfr: float = 0.0
    sense_index: float = 1.0
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    yaw_rate: float = 0.0
    lateral_accel: float = 0.0
    longitudinal_accel: float = 0.0


@dataclass
class MinimalBrakesNode:
    delta_nfr: float = 0.0
    sense_index: float = 1.0
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    brake_pressure: float = 0.0
    locking: float = 0.0
    brake_temp_fl: float = 0.0
    brake_temp_fr: float = 0.0
    brake_temp_rl: float = 0.0
    brake_temp_rr: float = 0.0
    brake_temp_peak: float = 0.0
    brake_temp_mean: float = 0.0


@dataclass
class MinimalTransmissionNode:
    delta_nfr: float = 0.0
    sense_index: float = 1.0
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    throttle: float = 0.0
    gear: int = 0
    speed: float = 0.0
    longitudinal_accel: float = 0.0
    rpm: float = 0.0
    line_deviation: float = 0.0


@dataclass
class MinimalTrackNode:
    delta_nfr: float = 0.0
    sense_index: float = 1.0
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    axle_load_balance: float = 0.0
    axle_velocity_balance: float = 0.0
    yaw: float = 0.0
    lateral_accel: float = 0.0
    gradient: float = 0.0


@dataclass
class MinimalDriverNode:
    delta_nfr: float = 0.0
    sense_index: float = 1.0
    nu_f: float = 0.0
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    steer: float = 0.0
    throttle: float = 0.0
    style_index: float = 0.0


@dataclass
class MinimalBundle:
    timestamp: float
    epi: float
    delta_nfr: float
    sense_index: float
    tyres: MinimalTyresNode
    suspension: MinimalSuspensionNode
    chassis: MinimalChassisNode
    brakes: MinimalBrakesNode
    transmission: MinimalTransmissionNode
    track: MinimalTrackNode
    driver: MinimalDriverNode
    structural_timestamp: float | None = None
    delta_breakdown: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    node_evolution: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    dEPI_dt: float = 0.0
    integrated_epi: float = 0.0
    delta_nfr_proj_longitudinal: float = 0.0
    delta_nfr_proj_lateral: float = 0.0
    nu_f_classification: str = ""
    nu_f_category: str = ""
    nu_f_label: str = ""
    nu_f_dominant: float = 0.0
    coherence_index: float = 0.0
    ackermann_parallel_index: float = 0.0


@pytest.mark.parametrize("bundle_scale", (0.1, 0.2, 0.3))
def test_minimal_bundle_satisfies_protocols(bundle_scale: float) -> None:
    record = build_telemetry_record(
        bundle_scale * 10.0,
        yaw_rate=bundle_scale,
        steer=bundle_scale,
        throttle=bundle_scale,
        vertical_load=4800.0 + (bundle_scale * 100.0),
        suspension_travel_front=0.01 * bundle_scale,
        suspension_travel_rear=0.008 * bundle_scale,
        suspension_velocity_front=0.02 * bundle_scale,
        suspension_velocity_rear=0.018 * bundle_scale,
        rpm=5000.0 * bundle_scale,
        line_deviation=0.0,
    )
    tyres = MinimalTyresNode(
        delta_nfr=bundle_scale,
        sense_index=1.0 - (bundle_scale * 0.1),
        load=record.vertical_load,
        slip_ratio=record.slip_ratio,
        mu_eff_front=record.mu_eff_front,
        mu_eff_rear=record.mu_eff_rear,
        mu_eff_front_lateral=record.mu_eff_front_lateral,
        mu_eff_front_longitudinal=record.mu_eff_front_longitudinal,
        mu_eff_rear_lateral=record.mu_eff_rear_lateral,
        mu_eff_rear_longitudinal=record.mu_eff_rear_longitudinal,
    )
    suspension = MinimalSuspensionNode(
        delta_nfr=bundle_scale * 0.5,
        sense_index=1.0 - (bundle_scale * 0.05),
        travel_front=record.suspension_travel_front,
        travel_rear=record.suspension_travel_rear,
        velocity_front=record.suspension_velocity_front,
        velocity_rear=record.suspension_velocity_rear,
    )
    chassis = MinimalChassisNode(
        delta_nfr=bundle_scale * 0.2,
        yaw_rate=record.yaw_rate,
        lateral_accel=record.lateral_accel,
        longitudinal_accel=record.longitudinal_accel,
    )
    brakes = MinimalBrakesNode(
        delta_nfr=bundle_scale * 0.1,
        brake_pressure=record.brake_pressure,
        locking=record.locking,
    )
    transmission = MinimalTransmissionNode(
        delta_nfr=bundle_scale * 0.05,
        throttle=record.throttle,
        gear=record.gear,
        speed=record.speed,
        longitudinal_accel=record.longitudinal_accel,
        rpm=record.rpm,
        line_deviation=record.line_deviation,
    )
    track = MinimalTrackNode(
        delta_nfr=bundle_scale * 0.07,
        axle_load_balance=record.vertical_load_front - record.vertical_load_rear,
        axle_velocity_balance=
            record.suspension_velocity_front - record.suspension_velocity_rear,
        yaw=record.yaw,
        lateral_accel=record.lateral_accel,
    )
    driver = MinimalDriverNode(
        delta_nfr=bundle_scale * 0.03,
        steer=record.steer,
        throttle=record.throttle,
        style_index=record.si,
    )
    bundle = MinimalBundle(
        timestamp=record.timestamp,
        structural_timestamp=record.timestamp,
        epi=bundle_scale * 2.0,
        delta_nfr=bundle_scale,
        sense_index=record.si,
        tyres=tyres,
        suspension=suspension,
        chassis=chassis,
        brakes=brakes,
        transmission=transmission,
        track=track,
        driver=driver,
        dEPI_dt=bundle_scale * 0.01,
        integrated_epi=bundle_scale * 0.02,
        delta_nfr_proj_longitudinal=bundle_scale * 0.6,
        delta_nfr_proj_lateral=bundle_scale * 0.4,
        ackermann_parallel_index=bundle_scale * 0.05,
    )

    assert isinstance(tyres, SupportsTyresNode)
    assert isinstance(suspension, SupportsSuspensionNode)
    assert isinstance(chassis, SupportsChassisNode)
    assert isinstance(brakes, SupportsBrakesNode)
    assert isinstance(transmission, SupportsTransmissionNode)
    assert isinstance(track, SupportsTrackNode)
    assert isinstance(driver, SupportsDriverNode)
    assert isinstance(bundle, SupportsEPIBundle)

    metrics = compute_window_metrics([record], bundles=[bundle])
    assert isinstance(metrics, WindowMetrics)

    breakdown = dissonance_breakdown_operator([bundle.delta_nfr], bundle.delta_nfr, bundles=[bundle])
    assert isinstance(breakdown, DissonanceBreakdown)
    assert breakdown.total_events >= 0
