import math

from dataclasses import replace
from statistics import mean
from typing import Dict, Sequence

import numpy as np
import pytest

from tests.helpers import build_frequency_record, build_telemetry_record

from tnfr_lfs.core.cache_settings import CacheOptions
from tnfr_lfs.core import cache as cache_helpers
from tnfr_lfs.core import epi as epi_module
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
from tnfr_lfs.core.spectrum import (
    apply_window,
    cross_spectrum,
    detrend,
    estimate_sample_rate,
    hann_window,
    power_spectrum,
)


@pytest.fixture(autouse=True)
def _clear_epi_cache_state():
    cache_helpers.configure_cache_from_options(CacheOptions())
    cache_helpers.clear_delta_cache()
    cache_helpers.clear_dynamic_cache()
    yield
    cache_helpers.configure_cache_from_options(CacheOptions())
    cache_helpers.clear_delta_cache()
    cache_helpers.clear_dynamic_cache()


def test_should_use_delta_cache_defaults_to_global_state():
    cache_helpers.configure_cache(enable_delta_cache=False)
    assert not cache_helpers.should_use_delta_cache(None)

    cache_helpers.configure_cache(enable_delta_cache=True)
    assert cache_helpers.should_use_delta_cache(None)


def test_should_use_delta_cache_honours_override():
    options = CacheOptions(enable_delta_cache=False, nu_f_cache_size=64, telemetry_cache_size=1)
    assert not cache_helpers.should_use_delta_cache(options)

    enabled = CacheOptions(enable_delta_cache=True, nu_f_cache_size=64, telemetry_cache_size=1)
    assert cache_helpers.should_use_delta_cache(enabled)


def test_should_use_dynamic_cache_defaults_to_global_state():
    cache_helpers.configure_cache(nu_f_cache_size=0)
    assert not cache_helpers.should_use_dynamic_cache(None)

    cache_helpers.configure_cache(nu_f_cache_size=16)
    assert cache_helpers.should_use_dynamic_cache(None)


def test_should_use_dynamic_cache_honours_override():
    disabled = CacheOptions(enable_delta_cache=True, nu_f_cache_size=0, telemetry_cache_size=1)
    assert not cache_helpers.should_use_dynamic_cache(disabled)

    enabled = CacheOptions(enable_delta_cache=False, nu_f_cache_size=8, telemetry_cache_size=1)
    assert cache_helpers.should_use_dynamic_cache(enabled)


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
            build_frequency_record(
                timestamp,
                steer=steer,
                throttle=throttle,
                brake=brake,
                suspension=suspension,
            )
        )
    return records


def test_power_spectrum_identifies_peak_frequency():
    sample_rate = 100.0
    duration = 2.0
    dominant_frequency = 5.0
    time = np.arange(int(sample_rate * duration), dtype=float) / sample_rate
    signal = np.sin(2.0 * math.pi * dominant_frequency * time)

    spectrum = power_spectrum(signal, sample_rate)

    assert spectrum
    frequencies = np.array([entry[0] for entry in spectrum], dtype=float)
    energy = np.array([entry[1] for entry in spectrum], dtype=float)
    peak_frequency = float(frequencies[np.argmax(energy)])

    assert math.isclose(peak_frequency, dominant_frequency, rel_tol=1e-2, abs_tol=1e-2)


def _legacy_cross_spectrum(
    input_series: Sequence[float], response_series: Sequence[float], sample_rate: float
) -> list[tuple[float, float, float]]:
    input_values = np.asarray(input_series, dtype=float)
    response_values = np.asarray(response_series, dtype=float)
    length = min(len(input_values), len(response_values))
    if length < 2 or sample_rate <= 0.0:
        return []

    input_values = np.asarray(detrend(input_values)[-length:], dtype=float)
    response_values = np.asarray(detrend(response_values)[-length:], dtype=float)
    window = np.asarray(hann_window(length), dtype=float)
    input_windowed = np.asarray(apply_window(input_values, window), dtype=float)
    response_windowed = np.asarray(apply_window(response_values, window), dtype=float)

    indices = np.arange(length, dtype=float)
    spectrum: list[tuple[float, float, float]] = []
    for harmonic in range(1, length // 2 + 1):
        angle_factor = -2.0 * math.pi * harmonic / length
        angles = angle_factor * indices
        cos_values = np.cos(angles)
        sin_values = np.sin(angles)
        x_real = float(np.dot(input_windowed, cos_values))
        x_imag = float(np.dot(input_windowed, sin_values))
        y_real = float(np.dot(response_windowed, cos_values))
        y_imag = float(np.dot(response_windowed, sin_values))
        cross_real = x_real * y_real + x_imag * y_imag
        cross_imag = x_imag * y_real - x_real * y_imag
        frequency = harmonic * sample_rate / length
        spectrum.append((frequency, cross_real, cross_imag))
    return spectrum


def test_cross_spectrum_matches_legacy_reference():
    sample_rate = 120.0
    duration = 2.0
    primary_frequency = 7.0
    secondary_frequency = 14.0
    time = np.arange(int(sample_rate * duration), dtype=float) / sample_rate

    input_series = 0.8 * np.sin(2.0 * math.pi * primary_frequency * time)
    input_series += 0.35 * np.sin(2.0 * math.pi * secondary_frequency * time + math.pi / 5.0)

    response_series = 1.2 * np.sin(2.0 * math.pi * primary_frequency * time + math.pi / 3.0)
    response_series += 0.1 * np.sin(2.0 * math.pi * secondary_frequency * time - math.pi / 6.0)

    reference = _legacy_cross_spectrum(input_series, response_series, sample_rate)
    spectrum = cross_spectrum(input_series, response_series, sample_rate)

    filtered = [entry for entry in spectrum if entry[0] > 1e-9]
    assert filtered
    assert len(filtered) == len(reference)

    for (frequency, real, imag), (ref_frequency, ref_real, ref_imag) in zip(filtered, reference):
        assert math.isclose(frequency, ref_frequency, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(real, ref_real, rel_tol=1e-9, abs_tol=1e-6)
        assert math.isclose(imag, ref_imag, rel_tol=1e-9, abs_tol=1e-6)


def _legacy_dynamic_multipliers(
    history: Sequence[TelemetryRecord],
    settings: NaturalFrequencySettings,
    car_model: str | None,
) -> tuple[dict[str, float], float]:
    history = list(history)
    if len(history) < 2:
        return {}, 0.0

    duration = history[-1].timestamp - history[0].timestamp
    if duration < max(0.0, settings.min_window_seconds - 1e-6):
        return {}, 0.0

    sample_rate = estimate_sample_rate(history)
    if sample_rate <= 0.0:
        return {}, 0.0

    min_samples = max(4, int(settings.min_window_seconds * sample_rate))
    if len(history) < min_samples:
        return {}, 0.0

    steer_series = [float(record.steer) for record in history]
    throttle_series = [float(record.throttle) for record in history]
    brake_series = [float(record.brake_pressure) for record in history]
    suspension_front = [float(record.suspension_velocity_front) for record in history]
    suspension_rear = [float(record.suspension_velocity_rear) for record in history]
    suspension_combined = [
        (front + rear) * 0.5 for front, rear in zip(suspension_front, suspension_rear)
    ]

    low = max(0.0, settings.bandpass_low_hz)
    high = max(low, settings.bandpass_high_hz)

    def _legacy_dominant_frequency(series: Sequence[float]) -> float:
        spectrum = power_spectrum(series, sample_rate)
        band = [entry for entry in spectrum if low <= entry[0] <= high]
        if not band:
            return 0.0
        frequency, energy = max(band, key=lambda entry: entry[1])
        if energy <= 1e-9:
            return 0.0
        return frequency

    def _legacy_dominant_cross(
        x_series: Sequence[float], y_series: Sequence[float]
    ) -> float:
        spectrum = cross_spectrum(x_series, y_series, sample_rate)
        band = [entry for entry in spectrum if low <= entry[0] <= high]
        if not band:
            return 0.0
        frequency, real, imag = max(
            band, key=lambda entry: math.hypot(entry[1], entry[2])
        )
        magnitude = math.hypot(real, imag)
        if magnitude <= 1e-9:
            return 0.0
        return frequency

    steer_freq = _legacy_dominant_frequency(steer_series)
    throttle_freq = _legacy_dominant_frequency(throttle_series)
    brake_freq = _legacy_dominant_frequency(brake_series)
    suspension_freq = _legacy_dominant_frequency(suspension_combined)
    tyre_freq = _legacy_dominant_cross(steer_series, suspension_combined)

    vehicle_frequency = settings.resolve_vehicle_frequency(car_model)

    def _normalise(frequency: float) -> float:
        if frequency <= 0.0:
            return 1.0
        ratio = frequency / vehicle_frequency
        ratio = max(settings.min_multiplier, min(settings.max_multiplier, ratio))
        return ratio

    dominant_frequency = 0.0
    multipliers: dict[str, float] = {}
    if steer_freq > 0.0:
        multipliers["driver"] = _normalise(steer_freq)
        dominant_frequency = steer_freq
    if throttle_freq > 0.0:
        multipliers["transmission"] = _normalise(throttle_freq)
        if throttle_freq > dominant_frequency:
            dominant_frequency = throttle_freq
    if brake_freq > 0.0:
        multipliers["brakes"] = _normalise(brake_freq)
        if brake_freq > dominant_frequency:
            dominant_frequency = brake_freq
    if suspension_freq > 0.0:
        value = _normalise(suspension_freq)
        multipliers["suspension"] = value
        multipliers.setdefault("chassis", value)
        if suspension_freq > dominant_frequency:
            dominant_frequency = suspension_freq
    if tyre_freq > 0.0:
        value = _normalise(tyre_freq)
        multipliers["tyres"] = value
        multipliers["chassis"] = value
        if tyre_freq > dominant_frequency:
            dominant_frequency = tyre_freq
    elif suspension_freq > 0.0 and steer_freq > 0.0:
        blended = (multipliers["suspension"] + multipliers["driver"]) * 0.5
        multipliers["tyres"] = blended

    return multipliers, dominant_frequency


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


def test_vectorised_baseline_matches_scalar_means(synthetic_records):
    """Vectorised baseline derivation must remain numerically compatible."""

    baseline = DeltaCalculator.derive_baseline(synthetic_records)

    float_fields = (
        "vertical_load",
        "slip_ratio",
        "slip_ratio_fl",
        "slip_ratio_fr",
        "slip_ratio_rl",
        "slip_ratio_rr",
        "lateral_accel",
        "longitudinal_accel",
        "yaw",
        "pitch",
        "roll",
        "brake_pressure",
        "locking",
        "nfr",
        "si",
        "speed",
        "yaw_rate",
        "slip_angle",
        "slip_angle_fl",
        "slip_angle_fr",
        "slip_angle_rl",
        "slip_angle_rr",
        "steer",
        "throttle",
        "vertical_load_front",
        "vertical_load_rear",
        "mu_eff_front",
        "mu_eff_rear",
        "mu_eff_front_lateral",
        "mu_eff_front_longitudinal",
        "mu_eff_rear_lateral",
        "mu_eff_rear_longitudinal",
        "suspension_travel_front",
        "suspension_travel_rear",
        "suspension_velocity_front",
        "suspension_velocity_rear",
    )

    for field in float_fields:
        expected = mean(getattr(record, field) for record in synthetic_records)
        value = getattr(baseline, field)
        if math.isnan(expected):
            assert math.isnan(value)
        else:
            assert value == pytest.approx(expected)

    expected_gear = int(round(mean(record.gear for record in synthetic_records)))
    assert baseline.gear == expected_gear


def test_dynamic_multipliers_vectorisation_matches_legacy(synthetic_records):
    analyzer = NaturalFrequencyAnalyzer()
    history = synthetic_records[:128]

    legacy_multipliers, legacy_frequency = _legacy_dynamic_multipliers(
        history, analyzer.settings, None
    )
    vectorised_multipliers, vectorised_frequency = analyzer._compute_dynamic_multipliers_raw(
        history, None
    )

    assert vectorised_frequency == pytest.approx(legacy_frequency)
    assert vectorised_multipliers.keys() == legacy_multipliers.keys()
    for node, expected in legacy_multipliers.items():
        assert vectorised_multipliers[node] == pytest.approx(expected)


def test_delta_nfr_by_node_emphasises_braking_signals():
    baseline = build_telemetry_record(
        timestamp=0.0,
        slip_ratio=0.02,
        lateral_accel=0.5,
        longitudinal_accel=0.1,
        brake_pressure=0.1,
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
    )
    sample = build_telemetry_record(
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
    baseline = build_telemetry_record(
        timestamp=0.0,
        vertical_load=4800.0,
        slip_ratio=0.01,
        lateral_accel=0.4,
        longitudinal_accel=0.2,
        yaw=-0.05,
        pitch=0.01,
        roll=0.02,
        brake_pressure=0.05,
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
    )
    sample = build_telemetry_record(
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
    assert last_snapshot.classification in {"optimal", "high"}
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


def test_delta_nfr_cache_reuses_result(monkeypatch):
    baseline = build_frequency_record(0.0, steer=0.1, throttle=0.6, brake=0.1, suspension=0.02)
    record = replace(
        build_frequency_record(0.1, steer=0.2, throttle=0.4, brake=0.2, suspension=0.05),
        reference=baseline,
        nfr=baseline.nfr + 10.0,
    )

    call_count = {"value": 0}

    def _fake_compute(target: TelemetryRecord) -> Dict[str, float]:
        call_count["value"] += 1
        return {"tyres": 1.0, "suspension": 0.5}

    monkeypatch.setattr(
        epi_module,
        "_delta_nfr_by_node_uncached",
        _fake_compute,
    )

    first = delta_nfr_by_node(record)
    second = delta_nfr_by_node(record)

    assert call_count["value"] == 1
    assert first == second

    cache_helpers.invalidate_delta_record(record)
    third = delta_nfr_by_node(record)
    assert call_count["value"] == 2
    assert third == first


def test_delta_nfr_cache_disable(monkeypatch):
    baseline = build_frequency_record(0.0, steer=0.1, throttle=0.6, brake=0.1, suspension=0.02)
    record = replace(
        build_frequency_record(0.1, steer=0.2, throttle=0.4, brake=0.2, suspension=0.05),
        reference=baseline,
        nfr=baseline.nfr + 8.0,
    )

    call_count = {"value": 0}

    def _fake_compute(target: TelemetryRecord) -> Dict[str, float]:
        call_count["value"] += 1
        return {"tyres": 1.0}

    monkeypatch.setattr(
        epi_module,
        "_delta_nfr_by_node_uncached",
        _fake_compute,
    )

    cache_helpers.configure_cache_from_options(CacheOptions(enable_delta_cache=False))

    delta_nfr_by_node(record)
    delta_nfr_by_node(record)

    assert call_count["value"] == 2


def test_delta_nfr_cache_options_override(monkeypatch):
    baseline = build_frequency_record(0.0, steer=0.1, throttle=0.6, brake=0.1, suspension=0.02)
    record = replace(
        build_frequency_record(0.1, steer=0.2, throttle=0.4, brake=0.2, suspension=0.05),
        reference=baseline,
        nfr=baseline.nfr + 6.0,
    )

    call_count = {"value": 0}

    def _fake_compute(target: TelemetryRecord) -> Dict[str, float]:
        call_count["value"] += 1
        return {"tyres": 1.0}

    monkeypatch.setattr(
        epi_module,
        "_delta_nfr_by_node_uncached",
        _fake_compute,
    )

    options = CacheOptions(enable_delta_cache=False, nu_f_cache_size=64, telemetry_cache_size=1)

    delta_nfr_by_node(record, cache_options=options)
    delta_nfr_by_node(record, cache_options=options)

    assert call_count["value"] == 2


def test_dynamic_multiplier_cache_invalidation(monkeypatch):
    settings = NaturalFrequencySettings(
        min_window_seconds=0.05,
        max_window_seconds=0.12,
        bandpass_low_hz=0.1,
        bandpass_high_hz=4.0,
    )
    analyzer = NaturalFrequencyAnalyzer(settings)

    samples = [
        build_frequency_record(0.00, steer=0.1, throttle=0.5, brake=0.2, suspension=0.03),
        build_frequency_record(0.05, steer=0.2, throttle=0.6, brake=0.1, suspension=0.04),
        build_frequency_record(0.10, steer=0.15, throttle=0.55, brake=0.12, suspension=0.02),
    ]
    for sample in samples:
        analyzer._append_record(sample)

    compute_calls = {"count": 0}

    def _fake_dynamic(self, history, car_model):
        compute_calls["count"] += 1
        return {"tyres": 1.1, "driver": 0.9}, 2.0

    monkeypatch.setattr(
        NaturalFrequencyAnalyzer,
        "_compute_dynamic_multipliers_raw",
        _fake_dynamic,
        raising=False,
    )

    invalidated: list[float] = []

    def _fake_invalidate(record: TelemetryRecord) -> None:
        invalidated.append(record.timestamp)

    monkeypatch.setattr(epi_module, "invalidate_dynamic_record", _fake_invalidate)

    first = analyzer._dynamic_multipliers("proto_car")
    second = analyzer._dynamic_multipliers("proto_car")

    assert compute_calls["count"] == 1
    assert first == second

    analyzer._append_record(
        build_frequency_record(0.20, steer=0.25, throttle=0.65, brake=0.15, suspension=0.05)
    )

    third = analyzer._dynamic_multipliers("proto_car")
    assert compute_calls["count"] == 2
    assert third == first
    assert invalidated  # Old history entries should trigger invalidation.


def test_dynamic_multiplier_cache_disable(monkeypatch):
    settings = NaturalFrequencySettings(
        min_window_seconds=0.05,
        max_window_seconds=0.12,
        bandpass_low_hz=0.1,
        bandpass_high_hz=4.0,
    )
    analyzer = NaturalFrequencyAnalyzer(
        settings,
        cache_options=CacheOptions(
            enable_delta_cache=True, nu_f_cache_size=0, telemetry_cache_size=1
        ),
    )

    samples = [
        build_frequency_record(0.00, steer=0.1, throttle=0.5, brake=0.2, suspension=0.03),
        build_frequency_record(0.05, steer=0.2, throttle=0.6, brake=0.1, suspension=0.04),
    ]
    for sample in samples:
        analyzer._append_record(sample)

    compute_calls = {"count": 0}

    def _fake_dynamic(self, history, car_model):
        compute_calls["count"] += 1
        return {"tyres": 1.0}, 2.0

    monkeypatch.setattr(
        NaturalFrequencyAnalyzer,
        "_compute_dynamic_multipliers_raw",
        _fake_dynamic,
        raising=False,
    )

    analyzer._dynamic_multipliers("FZR")
    analyzer._dynamic_multipliers("FZR")

    assert compute_calls["count"] == 2
