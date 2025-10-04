import math

import pytest

from tnfr_lfs.core.epi import (
    DeltaCalculator,
    EPIExtractor,
    NaturalFrequencyAnalyzer,
    NaturalFrequencySettings,
    TelemetryRecord,
    delta_nfr_by_node,
    resolve_nu_f_by_node,
)
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


def _frequency_record(
    timestamp: float,
    *,
    steer: float,
    throttle: float,
    brake: float,
    suspension: float,
) -> TelemetryRecord:
    base = {
        "timestamp": timestamp,
        "vertical_load": 5200.0 + suspension * 150.0,
        "slip_ratio": 0.02,
        "lateral_accel": 0.6 + steer * 0.1,
        "longitudinal_accel": 0.2 + (throttle - brake) * 0.05,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "brake_pressure": max(0.0, brake),
        "locking": 0.0,
        "nfr": 500.0,
        "si": 0.82,
        "speed": 45.0,
        "yaw_rate": steer * 0.15,
        "slip_angle": 0.01,
        "steer": steer,
        "throttle": max(0.0, min(1.0, throttle)),
        "gear": 3,
        "vertical_load_front": 2600.0 + suspension * 40.0,
        "vertical_load_rear": 2600.0 - suspension * 40.0,
        "mu_eff_front": 1.05,
        "mu_eff_rear": 1.05,
        "mu_eff_front_lateral": 1.05,
        "mu_eff_front_longitudinal": 1.0,
        "mu_eff_rear_lateral": 1.05,
        "mu_eff_rear_longitudinal": 1.0,
        "suspension_travel_front": 0.02 + suspension * 0.01,
        "suspension_travel_rear": 0.02 + suspension * 0.01,
        "suspension_velocity_front": suspension,
        "suspension_velocity_rear": suspension * 0.92,
    }
    return TelemetryRecord(**base)


def _synthetic_frequency_series(
    frequency: float,
    *,
    sample_rate: float,
    duration: float,
    noise_level: float,
) -> list[TelemetryRecord]:
    total_samples = int(duration * sample_rate)
    records: list[TelemetryRecord] = []
    for index in range(total_samples):
        timestamp = index / sample_rate
        base_wave = math.sin(2.0 * math.pi * frequency * timestamp)
        noise_wave = noise_level * math.sin(
            2.0 * math.pi * (frequency * 0.5) * timestamp + math.pi / 4.0
        )
        steer = 0.3 * base_wave + noise_wave
        throttle = 0.55 + 0.25 * base_wave + noise_level * math.sin(
            2.0 * math.pi * (frequency * 1.2) * timestamp + math.pi / 6.0
        )
        brake = 0.25 + 0.15 * math.sin(2.0 * math.pi * frequency * timestamp + math.pi / 3.0)
        brake += noise_level * 0.5 * math.sin(2.0 * math.pi * (frequency * 0.7) * timestamp)
        suspension = 0.08 * base_wave + 0.03 * math.sin(
            2.0 * math.pi * (frequency * 1.3) * timestamp + math.pi / 8.0
        )
        records.append(
            _frequency_record(
                timestamp,
                steer=steer,
                throttle=throttle,
                brake=brake,
                suspension=suspension,
            )
        )
    return records


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

    assert bundle.tyres.load == pytest.approx(sample.vertical_load)
    assert bundle.tyres.slip_ratio == pytest.approx(sample.slip_ratio)
    assert bundle.tyres.mu_eff_front == pytest.approx(sample.mu_eff_front)
    assert bundle.tyres.mu_eff_rear == pytest.approx(sample.mu_eff_rear)
    assert bundle.suspension.travel_front == pytest.approx(sample.suspension_travel_front)
    assert bundle.suspension.travel_rear == pytest.approx(sample.suspension_travel_rear)
    assert bundle.suspension.velocity_front == pytest.approx(sample.suspension_velocity_front)
    assert bundle.suspension.velocity_rear == pytest.approx(sample.suspension_velocity_rear)
    assert bundle.chassis.yaw == pytest.approx(sample.yaw)
    assert bundle.chassis.pitch == pytest.approx(sample.pitch)
    assert bundle.chassis.roll == pytest.approx(sample.roll)
    assert bundle.chassis.yaw_rate == pytest.approx(sample.yaw_rate)
    assert bundle.chassis.lateral_accel == pytest.approx(sample.lateral_accel)
    assert bundle.chassis.longitudinal_accel == pytest.approx(sample.longitudinal_accel)
    assert bundle.brakes.brake_pressure == pytest.approx(sample.brake_pressure)
    assert bundle.brakes.locking == pytest.approx(sample.locking)
    assert bundle.transmission.throttle == pytest.approx(sample.throttle)
    assert bundle.transmission.gear == sample.gear
    assert bundle.transmission.speed == pytest.approx(sample.speed)
    assert bundle.transmission.longitudinal_accel == pytest.approx(sample.longitudinal_accel)
    assert bundle.track.axle_load_balance == pytest.approx(
        sample.vertical_load_front - sample.vertical_load_rear
    )
    assert bundle.track.axle_velocity_balance == pytest.approx(
        sample.suspension_velocity_front - sample.suspension_velocity_rear
    )
    assert bundle.track.yaw == pytest.approx(sample.yaw)
    assert bundle.track.lateral_accel == pytest.approx(sample.lateral_accel)
    assert bundle.driver.steer == pytest.approx(sample.steer)
    assert bundle.driver.throttle == pytest.approx(sample.throttle)
    assert bundle.driver.style_index == pytest.approx(sample.si)


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
        mu_eff_front_lateral=0.95,
        mu_eff_front_longitudinal=0.88,
        mu_eff_rear_lateral=0.9,
        mu_eff_rear_longitudinal=0.82,
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
        mu_eff_front_lateral=1.25,
        mu_eff_front_longitudinal=1.12,
        mu_eff_rear_lateral=0.85,
        mu_eff_rear_longitudinal=0.74,
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
        mu_eff_front_lateral=0.8,
        mu_eff_front_longitudinal=0.72,
        mu_eff_rear_lateral=0.75,
        mu_eff_rear_longitudinal=0.68,
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
        mu_eff_front_lateral=1.4,
        mu_eff_front_longitudinal=1.18,
        mu_eff_rear_lateral=0.95,
        mu_eff_rear_longitudinal=0.82,
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


def test_epi_extractor_creates_structured_nodes(synthetic_bundles, synthetic_records):
    assert len(synthetic_bundles) == 17
    pivot = synthetic_bundles[5]
    source_record = synthetic_records[5]

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
    assert pivot.tyres.load == pytest.approx(source_record.vertical_load)
    assert pivot.suspension.velocity_front == pytest.approx(source_record.suspension_velocity_front)
    assert pivot.chassis.yaw_rate == pytest.approx(source_record.yaw_rate)
    assert pivot.brakes.locking == pytest.approx(source_record.locking)
    assert pivot.transmission.gear == source_record.gear
    assert pivot.track.axle_load_balance == pytest.approx(
        source_record.vertical_load_front - source_record.vertical_load_rear
    )
    assert pivot.driver.steer == pytest.approx(source_record.steer)


def test_delta_breakdown_matches_node_totals(synthetic_bundles):
    bundle = synthetic_bundles[3]

    assert bundle.delta_breakdown
    accumulated = 0.0
    for node, breakdown in bundle.delta_breakdown.items():
        node_total = sum(breakdown.values())
        model = getattr(bundle, node)
        assert node_total == pytest.approx(model.delta_nfr, rel=1e-6, abs=1e-9)
        accumulated += node_total
    assert accumulated == pytest.approx(bundle.delta_nfr, rel=1e-6, abs=1e-9)


def test_epi_weights_shift_balance_between_load_and_slip(synthetic_records):
    default_results = EPIExtractor().extract(synthetic_records)
    slip_focused = EPIExtractor(load_weight=0.2, slip_weight=0.8).extract(synthetic_records)

    assert default_results[0].epi != pytest.approx(slip_focused[0].epi)
    assert default_results[-1].epi != pytest.approx(slip_focused[-1].epi)
    assert all(0.0 <= bundle.sense_index <= 1.0 for bundle in slip_focused)


def test_natural_frequency_analysis_converges_to_dominant_signal():
    frequency = 2.4  # Hz
    sample_rate = 40.0
    duration = 6.0
    settings = NaturalFrequencySettings(
        min_window_seconds=2.0,
        max_window_seconds=5.0,
        bandpass_low_hz=0.5,
        bandpass_high_hz=5.0,
        smoothing_alpha=0.35,
        vehicle_frequency={"__default__": 1.5, "test_proto": 2.1},
        min_multiplier=0.5,
        max_multiplier=2.5,
    )
    analyzer = NaturalFrequencyAnalyzer(settings)
    car_model = "test_proto"

    records = _synthetic_frequency_series(
        frequency,
        sample_rate=sample_rate,
        duration=duration,
        noise_level=0.05,
    )

    base_reference = resolve_nu_f_by_node(records[-1]).by_node
    driver_history: list[float] = []
    last_map: dict[str, float] = {}
    warmup_samples = int(settings.min_window_seconds * sample_rate)
    for index, record in enumerate(records):
        last_snapshot = resolve_nu_f_by_node(
            record,
            analyzer=analyzer,
            car_model=car_model,
        )
        last_map = last_snapshot.by_node
        if index >= warmup_samples:
            driver_history.append(last_map["driver"])

    assert last_map  # Analyzer produced an updated map.
    vehicle_frequency = settings.resolve_vehicle_frequency(car_model)
    expected_ratio = frequency / vehicle_frequency
    expected_ratio = max(settings.min_multiplier, min(settings.max_multiplier, expected_ratio))

    base_final = resolve_nu_f_by_node(records[-1]).by_node
    assert last_snapshot.classification in {"óptima", "alta"}
    assert "ν_f" in last_snapshot.frequency_label
    assert last_snapshot.coherence_index == pytest.approx(0.0)
    for node in ("driver", "suspension", "transmission", "brakes", "tyres"):
        measured_ratio = last_map[node] / base_final[node]
        assert measured_ratio == pytest.approx(expected_ratio, rel=0.25)

    assert driver_history, "warm-up period should yield samples for smoothing analysis"
    driver_steps = [
        abs(curr - prev) for prev, curr in zip(driver_history[:-1], driver_history[1:])
    ]
    if driver_steps:
        max_step = max(driver_steps)
        assert max_step < base_reference["driver"] * 0.6
