from __future__ import annotations

from typing import Sequence

import math

from tnfr_core.epi import (
    NaturalFrequencySnapshot,
    apply_plugin_nu_f_snapshot,
    resolve_plugin_nu_f,
)
from tnfr_core.operators import plugin_coherence_operator
from tnfr_lfs.plugins import TNFRPlugin

from tests.helpers import build_telemetry_record


class SamplePlugin(TNFRPlugin):
    """Test helper implementing the plugin hooks for observability."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        super().__init__(*args, **kwargs)
        self.nu_f_updates: list[NaturalFrequencySnapshot] = []
        self.coherence_updates: list[tuple[float, Sequence[float] | None]] = []
        self.reset_count = 0

    def on_reset(self) -> None:
        self.reset_count += 1

    def on_nu_f_updated(self, snapshot: NaturalFrequencySnapshot) -> None:
        self.nu_f_updates.append(snapshot)

    def on_coherence_updated(
        self, coherence_index: float, series: Sequence[float] | None = None
    ) -> None:
        self.coherence_updates.append((coherence_index, None if series is None else tuple(series)))
def test_apply_snapshot_updates_plugin_state() -> None:
    plugin = SamplePlugin("sample", "Sample plugin", "1.0.0")
    snapshot = NaturalFrequencySnapshot(
        by_node={"tyres": 0.18, "driver": 0.07},
        dominant_frequency=2.2,
        classification="optimal",
        category="generic",
        target_band=(1.6, 2.4),
        coherence_index=0.45,
    )

    result = apply_plugin_nu_f_snapshot(plugin, snapshot)

    assert result is snapshot
    assert plugin.nu_f == snapshot.by_node
    assert math.isclose(plugin.coherence_index, snapshot.coherence_index)
    assert plugin.nu_f_updates[-1] is snapshot


def test_resolve_plugin_nu_f_uses_pipeline_objects() -> None:
    plugin = SamplePlugin("integration", "Integration test", "1.0.0")
    history = [
        build_telemetry_record(
            0.0,
            nfr=510.0,
            si=0.82,
            vertical_load=5000.0,
            slip_ratio=0.01,
            lateral_accel=0.5 + 0.1 * 0.1,
            longitudinal_accel=0.3,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            brake_pressure=0.1,
            locking=0.0,
            speed=45.0,
            yaw_rate=0.1 * 0.2,
            slip_angle=0.01,
            steer=0.1,
            throttle=0.5,
            gear=3,
            vertical_load_front=2500.0,
            vertical_load_rear=2500.0,
            mu_eff_front=1.05,
            mu_eff_rear=1.04,
            mu_eff_front_lateral=1.05,
            mu_eff_front_longitudinal=1.0,
            mu_eff_rear_lateral=1.05,
            mu_eff_rear_longitudinal=1.0,
            suspension_travel_front=0.03,
            suspension_travel_rear=0.031,
            suspension_velocity_front=0.02,
            suspension_velocity_rear=0.018,
        ),
        build_telemetry_record(
            0.1,
            nfr=510.0,
            si=0.82,
            vertical_load=5000.0,
            slip_ratio=0.01,
            lateral_accel=0.5 + 0.3 * 0.1,
            longitudinal_accel=0.3,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            brake_pressure=0.1,
            locking=0.0,
            speed=45.0,
            yaw_rate=0.3 * 0.2,
            slip_angle=0.01,
            steer=0.3,
            throttle=0.5,
            gear=3,
            vertical_load_front=2500.0,
            vertical_load_rear=2500.0,
            mu_eff_front=1.05,
            mu_eff_rear=1.04,
            mu_eff_front_lateral=1.05,
            mu_eff_front_longitudinal=1.0,
            mu_eff_rear_lateral=1.05,
            mu_eff_rear_longitudinal=1.0,
            suspension_travel_front=0.03,
            suspension_travel_rear=0.031,
            suspension_velocity_front=0.02,
            suspension_velocity_rear=0.018,
        ),
    ]
    target = build_telemetry_record(
        0.2,
        nfr=510.0,
        si=0.82,
        vertical_load=5000.0,
        slip_ratio=0.01,
        lateral_accel=0.5 + 0.25 * 0.1,
        longitudinal_accel=0.3,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        brake_pressure=0.1,
        locking=0.0,
        speed=45.0,
        yaw_rate=0.25 * 0.2,
        slip_angle=0.01,
        steer=0.25,
        throttle=0.5,
        gear=3,
        vertical_load_front=2500.0,
        vertical_load_rear=2500.0,
        mu_eff_front=1.05,
        mu_eff_rear=1.04,
        mu_eff_front_lateral=1.05,
        mu_eff_front_longitudinal=1.0,
        mu_eff_rear_lateral=1.05,
        mu_eff_rear_longitudinal=1.0,
        suspension_travel_front=0.03,
        suspension_travel_rear=0.031,
        suspension_velocity_front=0.02,
        suspension_velocity_rear=0.018,
    )

    snapshot = resolve_plugin_nu_f(plugin, target, history=history)

    assert snapshot.by_node == plugin.nu_f
    assert math.isfinite(snapshot.dominant_frequency)
    assert plugin.nu_f_updates[-1] == snapshot
    assert math.isclose(plugin.coherence_index, snapshot.coherence_index)


def test_plugin_coherence_operator_synchronises_state() -> None:
    plugin = SamplePlugin("coherence", "Coherence tracker", "1.0.0")
    series = [0.1, 0.2, 0.4, 0.35]

    smoothed = plugin_coherence_operator(plugin, series, window=3)

    assert len(smoothed) == len(series)
    assert math.isclose(plugin.coherence_index, smoothed[-1])
    assert plugin.coherence_updates[-1][1] == tuple(smoothed)


def test_reset_state_clears_cached_values() -> None:
    plugin = SamplePlugin("reset", "Resettable", "1.0.0")
    snapshot = NaturalFrequencySnapshot(
        by_node={"tyres": 0.2},
        dominant_frequency=2.5,
        classification="optimal",
        category="generic",
        target_band=(1.6, 2.4),
        coherence_index=0.6,
    )
    apply_plugin_nu_f_snapshot(plugin, snapshot)
    plugin.reset_state()

    assert plugin.reset_count == 1
    assert plugin.nu_f == {}
    assert plugin.coherence_index == 0.0
