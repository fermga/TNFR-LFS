"""Contract tests covering the TNFR theory public surface."""

from importlib import import_module

import pytest

from tnfr_core import metrics as core_metrics
from tnfr_core import operator_detection


STRUCTURAL_LABELS = {
    "AL": "Support",
    "EN": "Reception",
    "IL": "Coherence",
    "NAV": "Transition",
    "NUL": "Contraction",
    "OZ": "Dissonance",
    "RA": "Propagation",
    "REMESH": "Remeshing",
    "SILENCE": "Structural silence",
    "THOL": "Auto-organisation",
    "UM": "Coupling",
    "VAL": "Amplification",
    "ZHIR": "Transformation",
}


@pytest.mark.parametrize("detector_name", [
    "detect_al",
    "detect_oz",
    "detect_il",
    "detect_silence",
    "detect_nav",
])
def test_operator_detectors_are_public(detector_name: str) -> None:
    detector = getattr(operator_detection, detector_name)
    assert callable(detector), detector_name


@pytest.mark.parametrize("metric_name", [
    "coherence_total",
    "phase_synchrony_index",
    "psi_norm",
    "psi_support",
    "bifurcation_threshold",
    "mutation_threshold",
])
def test_metric_utilities_are_public(metric_name: str) -> None:
    metric_callable = getattr(core_metrics, metric_name)
    assert callable(metric_callable), metric_name


def test_canonical_operator_label_recognises_structural_codes() -> None:
    """Every structural detector code maps to its canonical label."""

    for code, label in STRUCTURAL_LABELS.items():
        assert operator_detection.canonical_operator_label(code) == label
        # mixed-case identifiers stay compatible with previous detector outputs
        assert operator_detection.canonical_operator_label(code.lower()) == label


def test_theory_audit_script_is_importable() -> None:
    """The auxiliary audit script remains importable for documentation hooks."""

    module = import_module("tools.tnfr_theory_audit")
    assert hasattr(module, "THEORY_OPERATORS")
    assert hasattr(module, "THEORY_VARIABLES")
