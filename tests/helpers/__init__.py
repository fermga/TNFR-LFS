"""Convenience re-exports for test helpers."""

from .abtest import DummyBundle, build_metrics, scale_samples
from .cli import instrument_prepare_pack_context
from .constants import BASE_NU_F, SUPPORTED_CAR_MODELS
from .epi import build_balanced_bundle, build_epi_bundle, build_epi_nodes
from .microsector import build_goal, build_microsector
from .packets import build_outgauge_packet, build_outsim_packet
from .plugins import (
    plugin_registry_state,
    write_plugin_config_text,
    write_plugin_manager_config,
    write_plugin_module,
)
from .profile_manager import preloaded_profile_manager
from .setup import build_minimal_setup_plan, build_native_export_plan, build_setup_plan
from .steering import (
    build_parallel_window_metrics,
    build_steering_bundle,
    build_steering_record,
)
from .telemetry import (
    build_calibration_record,
    build_contextual_delta_record,
    build_dynamic_record,
    build_frequency_record,
    build_resonance_record,
    build_telemetry_record,
)
from .udp import (
    QueueUDPSocket,
    build_outgauge_payload,
    build_outsim_payload,
    make_select_stub,
    make_wait_stub,
    pad_outgauge_field,
)

__all__ = [
    "preloaded_profile_manager",
    "build_parallel_window_metrics",
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
    "build_epi_bundle",
    "build_balanced_bundle",
    "build_epi_nodes",
    "build_goal",
    "build_microsector",
    "build_setup_plan",
    "build_native_export_plan",
    "build_minimal_setup_plan",
    "plugin_registry_state",
    "write_plugin_config_text",
    "write_plugin_manager_config",
    "write_plugin_module",
    "build_outsim_packet",
    "build_outgauge_packet",
    "QueueUDPSocket",
    "pad_outgauge_field",
    "build_outgauge_payload",
    "build_outsim_payload",
    "make_select_stub",
    "make_wait_stub",
    "instrument_prepare_pack_context",
]
