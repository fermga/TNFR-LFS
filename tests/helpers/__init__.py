"""Convenience re-exports for test helpers."""

from .abtest import DummyBundle, build_metrics, scale_samples
from .constants import BASE_NU_F, SUPPORTED_CAR_MODELS
from .profile_manager import preloaded_profile_manager
from .setup import build_minimal_setup_plan, build_native_export_plan, build_setup_plan
from .steering import build_steering_bundle, build_steering_record
from .telemetry import (
    build_calibration_record,
    build_contextual_delta_record,
    build_dynamic_record,
    build_frequency_record,
    build_resonance_record,
    build_telemetry_record,
)
from .udp import QueueUDPSocket, make_select_stub, make_wait_stub

__all__ = [
    "preloaded_profile_manager",
    "build_steering_bundle",
    "build_steering_record",
    "build_calibration_record",
    "build_contextual_delta_record",
    "build_dynamic_record",
    "build_frequency_record",
    "build_resonance_record",
    "build_telemetry_record",
    "DummyBundle",
    "build_metrics",
    "scale_samples",
    "BASE_NU_F",
    "SUPPORTED_CAR_MODELS",
    "build_setup_plan",
    "build_native_export_plan",
    "build_minimal_setup_plan",
    "QueueUDPSocket",
    "make_select_stub",
    "make_wait_stub",
]
