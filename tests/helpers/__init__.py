"""Convenience re-exports for test helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

_MODULE_EXPORTS: Tuple[Tuple[str, Iterable[str]], ...] = (
    ("profile_manager", ("preloaded_profile_manager",)),
    (
        "steering",
        (
            "build_parallel_window_metrics",
            "build_steering_bundle",
            "build_steering_record",
        ),
    ),
    (
        "telemetry",
        (
            "build_calibration_record",
            "build_contextual_delta_record",
            "build_dynamic_record",
            "build_frequency_record",
            "build_resonance_record",
            "build_telemetry_record",
        ),
    ),
    ("abtest", ("DummyBundle", "build_metrics", "scale_samples")),
    ("constants", ("BASE_NU_F", "SUPPORTED_CAR_MODELS")),
    (
        "epi",
        (
            "build_epi_bundle",
            "build_node_bundle",
            "build_balanced_bundle",
            "build_epi_nodes",
            "build_support_bundle",
            "build_operator_bundle",
            "build_axis_bundle",
            "build_udr_bundle_series",
            "build_rich_bundle",
        ),
    ),
    ("microsector", ("build_goal", "build_microsector")),
    (
        "setup",
        (
            "build_setup_plan",
            "build_native_export_plan",
            "build_minimal_setup_plan",
        ),
    ),
    (
        "plugins",
        (
            "plugin_registry_state",
            "write_plugin_config_text",
            "write_plugin_manager_config",
            "write_plugin_module",
        ),
    ),
    ("replay_bundle", ("RowToRecordCounter", "monkeypatch_row_to_record_counter")),
    (
        "packets",
        (
            "build_extended_outsim_packet",
            "build_extended_outsim_payload",
            "build_outgauge_packet",
            "build_outgauge_sample",
            "build_outsim_packet",
            "build_outsim_sample",
            "build_sample_outgauge_packet",
            "build_synthetic_packet_pair",
            "simple_outsim_namespace",
        ),
    ),
    (
        "udp",
        (
            "QueueUDPSocket",
            "append_once_on_wait",
            "pad_outgauge_field",
            "build_outgauge_payload",
            "build_outsim_payload",
            "extend_queue_on_wait",
            "make_select_stub",
            "make_wait_stub",
            "raise_gaierror",
        ),
    ),
    (
        "cli",
        (
            "instrument_prepare_pack_context",
            "run_cli_in_tmp",
            "DummyRecord",
            "build_load_parquet_args",
            "build_persist_parquet_args",
        ),
    ),
    ("osd", ("DummyHUD", "_populate_hud", "_window_metrics_from_parallel_turn")),
    (
        "packs",
        (
            "MINIMAL_DATA_CAR",
            "PackBuilder",
            "create_cli_config_pack",
            "create_config_pack",
            "create_brake_thermal_pack",
            "pack_builder",
        ),
    ),
)

_NAME_TO_MODULE: Dict[str, str] = {
    name: module for module, names in _MODULE_EXPORTS for name in names
}

__all__ = [name for _, names in _MODULE_EXPORTS for name in names]


def __getattr__(name: str) -> Any:
    try:
        module_name = _NAME_TO_MODULE[name]
    except KeyError as exc:  # pragma: no cover - standard AttributeError path
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__() -> List[str]:
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:
    from .abtest import DummyBundle, build_metrics, scale_samples
    from .cli import (
        DummyRecord,
        build_load_parquet_args,
        build_persist_parquet_args,
        instrument_prepare_pack_context,
        run_cli_in_tmp,
    )
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
    from .osd import DummyHUD, _populate_hud, _window_metrics_from_parallel_turn
    from .packets import (
        build_extended_outsim_packet,
        build_extended_outsim_payload,
        build_outgauge_packet,
        build_outgauge_sample,
        build_outsim_packet,
        build_outsim_sample,
        build_sample_outgauge_packet,
        build_synthetic_packet_pair,
        simple_outsim_namespace,
    )
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
    from .profile_manager import preloaded_profile_manager
    from .replay_bundle import RowToRecordCounter, monkeypatch_row_to_record_counter
    from .setup import (
        build_minimal_setup_plan,
        build_native_export_plan,
        build_setup_plan,
    )
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
        append_once_on_wait,
        build_outgauge_payload,
        build_outsim_payload,
        extend_queue_on_wait,
        make_select_stub,
        make_wait_stub,
        pad_outgauge_field,
        raise_gaierror,
    )
