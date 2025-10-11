"""Convenience re-exports for test helpers."""

from .abtest import DummyBundle, build_metrics, scale_samples
from .cli import (
    DummyRecord,
    build_load_parquet_args,
    build_persist_parquet_args,
    instrument_prepare_pack_context,
    run_cli_in_tmp,
)
from .osd import DummyHUD, _populate_hud, _window_metrics_from_parallel_turn
from .constants import BASE_NU_F, SUPPORTED_CAR_MODELS
from .epi import (
    build_axis_bundle,
    build_balanced_bundle,
    build_epi_bundle,
    build_node_bundle,
    build_epi_nodes,
    build_operator_bundle,
    build_rich_bundle,
    build_support_bundle,
    build_udr_bundle_series,
)
from .microsector import build_goal, build_microsector
from .packets import build_outgauge_packet, build_outsim_packet
from .packs import (
    MINIMAL_DATA_CAR,
    PackBuilder,
    create_brake_thermal_pack,
    create_cli_config_pack,
    create_config_pack,
    pack_builder,
)
from .plugins import (
    plugin_registry_state,
    write_plugin_config_text,
    write_plugin_manager_config,
    write_plugin_module,
)
from .replay_bundle import RowToRecordCounter, monkeypatch_row_to_record_counter
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
    append_once_on_wait,
    QueueUDPSocket,
    build_outgauge_payload,
    build_outsim_payload,
    extend_queue_on_wait,
    make_select_stub,
    make_wait_stub,
    pad_outgauge_field,
    raise_gaierror,
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
    "build_node_bundle",
    "build_balanced_bundle",
    "build_epi_nodes",
    "build_support_bundle",
    "build_operator_bundle",
    "build_axis_bundle",
    "build_udr_bundle_series",
    "build_rich_bundle",
    "build_goal",
    "build_microsector",
    "build_setup_plan",
    "build_native_export_plan",
    "build_minimal_setup_plan",
    "plugin_registry_state",
    "write_plugin_config_text",
    "write_plugin_manager_config",
    "write_plugin_module",
    "RowToRecordCounter",
    "monkeypatch_row_to_record_counter",
    "build_outsim_packet",
    "build_outgauge_packet",
    "QueueUDPSocket",
    "append_once_on_wait",
    "pad_outgauge_field",
    "build_outgauge_payload",
    "build_outsim_payload",
    "extend_queue_on_wait",
    "make_select_stub",
    "make_wait_stub",
    "raise_gaierror",
    "instrument_prepare_pack_context",
    "run_cli_in_tmp",
    "DummyRecord",
    "build_load_parquet_args",
    "build_persist_parquet_args",
    "DummyHUD",
    "_populate_hud",
    "_window_metrics_from_parallel_turn",
    "MINIMAL_DATA_CAR",
    "PackBuilder",
    "create_cli_config_pack",
    "create_config_pack",
    "create_brake_thermal_pack",
    "pack_builder",
]
