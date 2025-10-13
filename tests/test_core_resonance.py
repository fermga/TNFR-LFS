import math

import pytest

from tests.helpers import build_dynamic_record

from tnfr_core.resonance import analyse_modal_resonance


def test_modal_analysis_reports_excitation_ratio() -> None:
    records = []
    for index in range(200):
        timestamp = index * 0.05
        yaw = math.sin(2.0 * math.pi * 1.0 * timestamp)
        roll = math.sin(2.0 * math.pi * 0.6 * timestamp)
        pitch = math.sin(2.0 * math.pi * 0.8 * timestamp)
        steer = math.sin(2.0 * math.pi * 0.8 * timestamp)
        suspension_front = 0.5 * math.sin(2.0 * math.pi * 0.8 * timestamp)
        suspension_rear = 0.45 * math.sin(2.0 * math.pi * 0.8 * timestamp)
        records.append(
            build_dynamic_record(
                timestamp,
                5000.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.8,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                steer=steer,
                throttle=0.5,
                speed=120.0,
                gear=4,
                vertical_load_front=2500.0,
                vertical_load_rear=2500.0,
                mu_eff_front=1.0,
                mu_eff_rear=1.0,
                mu_eff_front_lateral=1.0,
                mu_eff_front_longitudinal=1.0,
                mu_eff_rear_lateral=1.0,
                mu_eff_rear_longitudinal=1.0,
                suspension_travel_front=0.02,
                suspension_travel_rear=0.02,
                suspension_velocity_front=suspension_front,
                suspension_velocity_rear=suspension_rear,
            )
        )

    analysis = analyse_modal_resonance(records)

    yaw_axis = analysis["yaw"]
    pitch_axis = analysis["pitch"]

    assert yaw_axis.nu_exc == pytest.approx(0.8, rel=0.1)
    assert yaw_axis.rho == pytest.approx(0.8, rel=0.15)
    assert pitch_axis.rho == pytest.approx(1.0, rel=0.1)
