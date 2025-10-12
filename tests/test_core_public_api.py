"""Verify that tnfr_lfs.core public modules expose the expected symbols."""

from __future__ import annotations

import importlib

import pytest


MODULE_PUBLIC_EXPORTS: dict[str, set[str]] = {
    "tnfr_lfs.core.epi": {
        "TelemetryRecord",
        "EPIExtractor",
        "NaturalFrequencyAnalyzer",
        "NaturalFrequencySnapshot",
        "NaturalFrequencySettings",
        "DeltaCalculator",
        "delta_nfr_by_node",
        "apply_plugin_nu_f_snapshot",
        "resolve_plugin_nu_f",
    },
    "tnfr_lfs.core.epi_models": {"EPIBundle"},
    "tnfr_lfs.core.coherence": {"compute_node_delta_nfr", "sense_index"},
    "tnfr_lfs.core.segmentation": {
        "Goal",
        "Microsector",
        "detect_quiet_microsector_streaks",
        "microsector_stability_metrics",
        "segment_microsectors",
    },
    "tnfr_lfs.core.resonance": {
        "ModalPeak",
        "ModalAnalysis",
        "analyse_modal_resonance",
    },
    "tnfr_lfs.core.structural_time": {
        "compute_structural_timestamps",
        "resolve_time_axis",
    },
    "tnfr_lfs.core.delta_utils": set(),
}


@pytest.mark.parametrize("module_name", sorted(MODULE_PUBLIC_EXPORTS))
def test_module_all_matches_expected(module_name: str) -> None:
    module = importlib.import_module(module_name)
    expected = MODULE_PUBLIC_EXPORTS[module_name]
    assert hasattr(module, "__all__"), f"{module_name} is missing __all__"
    assert set(module.__all__) == expected


@pytest.mark.parametrize("module_name", sorted(MODULE_PUBLIC_EXPORTS))
def test_star_import_only_exposes_expected(module_name: str) -> None:
    expected = MODULE_PUBLIC_EXPORTS[module_name]
    namespace: dict[str, object] = {}
    exec(f"from {module_name} import *", {}, namespace)
    exported = {name for name in namespace if not name.startswith("_")}
    assert exported == expected
