"""Regression coverage for the native LFS exporter."""

from __future__ import annotations

import importlib
import struct

import pytest

from tnfr_lfs.exporters.setup_plan import SetupChange, SetupPlan


def _build_plan() -> SetupPlan:
    return SetupPlan(
        car_model="XFG",
        session="practice",
        changes=(
            SetupChange("front_camber_deg", -2.5, "", ""),
            SetupChange("rear_camber_deg", -1.8, "", ""),
            SetupChange("front_toe_deg", 0.1, "", ""),
            SetupChange("rear_toe_deg", -0.2, "", ""),
            SetupChange("caster_deg", 3.4, "", ""),
            SetupChange("front_ride_height", 55.0, "", ""),
            SetupChange("rear_ride_height", 60.0, "", ""),
            SetupChange("front_spring_stiffness", 75.0, "", ""),
            SetupChange("rear_spring_stiffness", 80.0, "", ""),
            SetupChange("front_rebound_clicks", 12.0, "", ""),
            SetupChange("rear_rebound_clicks", 10.0, "", ""),
            SetupChange("front_compression_clicks", 8.0, "", ""),
            SetupChange("rear_compression_clicks", 9.0, "", ""),
            SetupChange("front_arb_steps", 3200.0, "", ""),
            SetupChange("rear_arb_steps", 2800.0, "", ""),
            SetupChange("front_tyre_pressure", 190.0, "", ""),
            SetupChange("rear_tyre_pressure", 195.0, "", ""),
            SetupChange("brake_bias_pct", 68.5, "", ""),
            SetupChange("diff_power_lock", 45.0, "", ""),
            SetupChange("diff_coast_lock", 35.0, "", ""),
            SetupChange("diff_preload_nm", 80.0, "", ""),
            SetupChange("final_drive_ratio", 3.72, "", ""),
            SetupChange("gear_1_ratio", 3.10, "", ""),
            SetupChange("gear_2_ratio", 2.12, "", ""),
            SetupChange("gear_3_ratio", 1.55, "", ""),
            SetupChange("gear_4_ratio", 1.22, "", ""),
            SetupChange("gear_5_ratio", 1.00, "", ""),
            SetupChange("gear_6_ratio", 0.88, "", ""),
            SetupChange("rear_wing_angle", 12.0, "", ""),
        ),
    )


@pytest.fixture(autouse=True)
def _enable_native_export(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TNFR_LFS_NATIVE_EXPORT", "1")
    # Ensure the module observes the updated environment variable.
    module = importlib.import_module("tnfr_lfs.exporters.lfs_native")
    importlib.reload(module)
    return module


def test_encode_native_setup_produces_expected_layout(_enable_native_export):
    module = importlib.import_module("tnfr_lfs.exporters.lfs_native")
    payload = module.encode_native_setup(_build_plan())

    assert payload[:6] == b"SRSETT"
    assert payload[7] == 252
    assert payload[8] == 2
    assert payload[12] & 0x80  # modern setup flag set

    # Camber bytes represent +/-4.5° around the midpoint of 45.
    assert payload[120] == 20  # front left camber
    assert payload[121] == 20  # front right camber
    assert payload[80] == 27  # rear left camber
    assert payload[81] == 27  # rear right camber

    # Toe values are encoded in 0.1° steps with 9 representing 0°.
    assert payload[116] == 10
    assert payload[76] == 7

    # Caster stored as tenths of a degree.
    assert payload[117] == 34

    # Ride heights and spring data are stored as little-endian floats.
    assert struct.unpack_from("<f", payload, 92)[0] == pytest.approx(55.0)
    assert struct.unpack_from("<f", payload, 52)[0] == pytest.approx(60.0)
    assert struct.unpack_from("<f", payload, 96)[0] == pytest.approx(75.0)
    assert struct.unpack_from("<f", payload, 56)[0] == pytest.approx(80.0)
    assert struct.unpack_from("<f", payload, 104)[0] == pytest.approx(12.0)
    assert struct.unpack_from("<f", payload, 64)[0] == pytest.approx(10.0)
    assert struct.unpack_from("<f", payload, 100)[0] == pytest.approx(8.0)
    assert struct.unpack_from("<f", payload, 60)[0] == pytest.approx(9.0)
    assert struct.unpack_from("<f", payload, 108)[0] == pytest.approx(3200.0)
    assert struct.unpack_from("<f", payload, 68)[0] == pytest.approx(2800.0)

    # Tyre pressures are stored as unsigned words in kPa.
    assert struct.unpack_from("<H", payload, 128)[0] == 190
    assert struct.unpack_from("<H", payload, 88)[0] == 195

    # Brake balance is stored as half-percent increments.
    assert payload[26] == 137

    # Differential settings.
    assert payload[86] == 45
    assert payload[87] == 35
    assert payload[83] == 8

    # Wing angle is stored as a raw byte.
    assert payload[20] == 12

    # Final drive and gear ratios stored as floats.
    assert struct.unpack_from("<f", payload, 28)[0] == pytest.approx(3.72)
    assert struct.unpack_from("<f", payload, 32)[0] == pytest.approx(3.10)
    assert struct.unpack_from("<f", payload, 36)[0] == pytest.approx(2.12)
    assert struct.unpack_from("<f", payload, 40)[0] == pytest.approx(1.55)
    assert struct.unpack_from("<f", payload, 44)[0] == pytest.approx(1.22)
    assert struct.unpack_from("<f", payload, 48)[0] == pytest.approx(1.00)
    assert struct.unpack_from("<f", payload, 72)[0] == pytest.approx(0.88)
