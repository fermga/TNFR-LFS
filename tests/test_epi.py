import pytest

from tnfr_lfs.core.epi import DeltaCalculator, EPIExtractor, TelemetryRecord, delta_nfr_by_node
from tnfr_lfs.core.epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    EPIBundle,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)


def test_delta_calculation_against_baseline(synthetic_records):
    baseline = DeltaCalculator.derive_baseline(synthetic_records)
    sample = synthetic_records[0]
    bundle = DeltaCalculator.compute_bundle(sample, baseline, epi_value=0.0)

    assert isinstance(bundle, EPIBundle)
    assert bundle.delta_nfr == pytest.approx(sample.nfr - baseline.nfr)
    assert 0.0 <= bundle.sense_index <= 1.0
    assert bundle.track.delta_nfr + bundle.suspension.delta_nfr != 0.0
    assert bundle.dEPI_dt == pytest.approx(
        sum(
            getattr(bundle, node).nu_f * getattr(bundle, node).delta_nfr
            for node in ("tyres", "suspension", "chassis", "brakes", "transmission", "track", "driver")
        ),
        rel=1e-6,
    )
    assert bundle.integrated_epi == pytest.approx(bundle.epi)
    assert set(bundle.node_evolution) >= {
        "tyres",
        "suspension",
        "chassis",
        "brakes",
        "transmission",
        "track",
        "driver",
    }
    for node in ("tyres", "suspension", "chassis", "brakes", "transmission", "track", "driver"):
        integral, derivative = bundle.node_evolution[node]
        node_model = getattr(bundle, node)
        assert node_model.dEPI_dt == pytest.approx(derivative)
        assert node_model.integrated_epi == pytest.approx(integral)


def test_delta_nfr_by_node_emphasises_braking_signals():
    baseline = TelemetryRecord(
        timestamp=0.0,
        vertical_load=5000.0,
        slip_ratio=0.02,
        lateral_accel=0.5,
        longitudinal_accel=0.1,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        brake_pressure=0.1,
        locking=0.0,
        nfr=500.0,
        si=0.85,
        speed=22.0,
        yaw_rate=0.05,
        slip_angle=0.01,
        steer=0.1,
        throttle=0.6,
        gear=4,
        vertical_load_front=2600.0,
        vertical_load_rear=2400.0,
        mu_eff_front=0.95,
        mu_eff_rear=0.9,
        suspension_travel_front=0.52,
        suspension_travel_rear=0.48,
        suspension_velocity_front=0.0,
        suspension_velocity_rear=0.0,
    )
    sample = TelemetryRecord(
        timestamp=0.1,
        vertical_load=5250.0,
        slip_ratio=0.06,
        lateral_accel=0.6,
        longitudinal_accel=-0.6,
        yaw=0.12,
        pitch=0.03,
        roll=0.02,
        brake_pressure=0.85,
        locking=1.0,
        nfr=508.0,
        si=0.78,
        speed=20.5,
        yaw_rate=0.65,
        slip_angle=0.08,
        steer=0.55,
        throttle=0.4,
        gear=3,
        vertical_load_front=3150.0,
        vertical_load_rear=2100.0,
        mu_eff_front=1.25,
        mu_eff_rear=0.85,
        suspension_travel_front=0.61,
        suspension_travel_rear=0.39,
        suspension_velocity_front=0.45,
        suspension_velocity_rear=-0.45,
        reference=baseline,
    )

    distribution = delta_nfr_by_node(sample)

    assert pytest.approx(sum(distribution.values()), rel=1e-6) == sample.nfr - baseline.nfr
    assert distribution["brakes"] > distribution["driver"]
    assert distribution["brakes"] >= distribution["transmission"]
    assert distribution["tyres"] > 0


def test_delta_nfr_by_node_conserves_total_with_extended_fields():
    baseline = TelemetryRecord(
        timestamp=0.0,
        vertical_load=4800.0,
        slip_ratio=0.01,
        lateral_accel=0.4,
        longitudinal_accel=0.2,
        yaw=-0.05,
        pitch=0.01,
        roll=0.02,
        brake_pressure=0.05,
        locking=0.0,
        nfr=480.0,
        si=0.9,
        speed=18.0,
        yaw_rate=0.02,
        slip_angle=0.005,
        steer=0.04,
        throttle=0.7,
        gear=3,
        vertical_load_front=2500.0,
        vertical_load_rear=2300.0,
        mu_eff_front=0.8,
        mu_eff_rear=0.75,
        suspension_travel_front=0.52,
        suspension_travel_rear=0.48,
        suspension_velocity_front=0.0,
        suspension_velocity_rear=0.0,
    )
    sample = TelemetryRecord(
        timestamp=0.05,
        vertical_load=5050.0,
        slip_ratio=0.09,
        lateral_accel=0.95,
        longitudinal_accel=-0.2,
        yaw=0.08,
        pitch=-0.03,
        roll=0.05,
        brake_pressure=0.6,
        locking=0.3,
        nfr=498.0,
        si=0.76,
        speed=21.0,
        yaw_rate=0.42,
        slip_angle=0.09,
        steer=0.2,
        throttle=0.6,
        gear=2,
        vertical_load_front=3000.0,
        vertical_load_rear=2050.0,
        mu_eff_front=1.4,
        mu_eff_rear=0.95,
        suspension_travel_front=0.6,
        suspension_travel_rear=0.4,
        suspension_velocity_front=0.8,
        suspension_velocity_rear=-0.75,
        reference=baseline,
    )

    node_deltas = delta_nfr_by_node(sample)
    assert pytest.approx(sum(node_deltas.values()), rel=1e-6) == sample.nfr - baseline.nfr
    assert node_deltas["tyres"] > node_deltas["driver"]
    assert node_deltas["suspension"] != pytest.approx(0.0)


def test_epi_extractor_creates_structured_nodes(synthetic_bundles):
    assert len(synthetic_bundles) == 17
    pivot = synthetic_bundles[5]

    assert isinstance(pivot.tyres, TyresNode)
    assert isinstance(pivot.suspension, SuspensionNode)
    assert isinstance(pivot.chassis, ChassisNode)
    assert isinstance(pivot.brakes, BrakesNode)
    assert isinstance(pivot.transmission, TransmissionNode)
    assert isinstance(pivot.track, TrackNode)
    assert isinstance(pivot.driver, DriverNode)
    assert 0.0 <= pivot.tyres.sense_index <= 1.0
    assert pivot.tyres.nu_f > 0
    assert pivot.suspension.nu_f > 0
    assert 0.0 <= pivot.track.sense_index <= 1.0
    assert sum(
        node.delta_nfr for node in (
            pivot.tyres,
            pivot.suspension,
            pivot.chassis,
            pivot.brakes,
            pivot.transmission,
            pivot.track,
            pivot.driver,
        )
    ) == pytest.approx(pivot.delta_nfr, rel=1e-6)
    assert pivot.dEPI_dt != pytest.approx(0.0)
    previous = synthetic_bundles[4]
    dt = pivot.timestamp - previous.timestamp
    expected_integral = previous.integrated_epi + pivot.dEPI_dt * dt
    assert pivot.integrated_epi == pytest.approx(expected_integral, rel=1e-3)
    nodal_derivative_sum = sum(
        getattr(pivot, node).dEPI_dt
        for node in ("tyres", "suspension", "chassis", "brakes", "transmission", "track", "driver")
    )
    assert nodal_derivative_sum == pytest.approx(pivot.dEPI_dt, rel=1e-6)
    nodal_integral_sum = sum(
        getattr(pivot, node).integrated_epi
        for node in ("tyres", "suspension", "chassis", "brakes", "transmission", "track", "driver")
    )
    assert nodal_integral_sum == pytest.approx(pivot.dEPI_dt * dt, rel=1e-6, abs=1e-9)
    for node in ("tyres", "suspension", "chassis", "brakes", "transmission", "track", "driver"):
        integral, derivative = pivot.node_evolution[node]
        model = getattr(pivot, node)
        assert integral == pytest.approx(model.integrated_epi, rel=1e-6, abs=1e-9)
        assert derivative == pytest.approx(model.dEPI_dt, rel=1e-6)


def test_epi_weights_shift_balance_between_load_and_slip(synthetic_records):
    default_results = EPIExtractor().extract(synthetic_records)
    slip_focused = EPIExtractor(load_weight=0.2, slip_weight=0.8).extract(synthetic_records)

    assert default_results[0].epi != pytest.approx(slip_focused[0].epi)
    assert default_results[-1].epi != pytest.approx(slip_focused[-1].epi)
    assert all(0.0 <= bundle.sense_index <= 1.0 for bundle in slip_focused)
