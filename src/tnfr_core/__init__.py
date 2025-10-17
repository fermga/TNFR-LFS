"""Core computation utilities for TNFR."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any

from tnfr_core._canonical import (
    CANONICAL_ENV_VALUE,
    CANONICAL_REQUESTED,
    TNFR_AVAILABLE,
    TNFR_MINIMUM_VERSION,
    CanonicalImportError,
    import_tnfr,
    require_tnfr,
)

_equations = import_module("tnfr_core.equations")
_metrics = import_module("tnfr_core.metrics")
_operators = import_module("tnfr_core.operators")
_runtime = import_module("tnfr_core.runtime")
_signal = import_module("tnfr_core.signal")

_BASE_EXPORTS = [
    "CANONICAL_ENV_VALUE",
    "CANONICAL_REQUESTED",
    "TNFR_AVAILABLE",
    "TNFR_MINIMUM_VERSION",
    "CanonicalImportError",
    "import_tnfr",
    "require_tnfr",
]

__all__ = list(
    dict.fromkeys(
        [
            *_BASE_EXPORTS,
            *_equations.__all__,
            *_metrics.__all__,
            *_operators.__all__,
            *_signal.__all__,
            *_runtime.__all__,
            *_signal.__all__,
        ]
    )
)

# Public handles to the structured namespaces.
equations = _equations
metrics = _metrics
operators = _operators
runtime = _runtime
signal = _signal

_MODULE_ALIASES = {
    "archetypes": import_module("tnfr_core.equations.archetypes"),
    "coherence": import_module("tnfr_core.equations.coherence"),
    "constants": import_module("tnfr_core.equations.constants"),
    "contextual_delta": import_module("tnfr_core.equations.contextual_delta"),
    "delta_utils": import_module("tnfr_core.equations.delta_utils"),
    "dissonance": import_module("tnfr_core.equations.dissonance"),
    "epi": import_module("tnfr_core.equations.epi"),
    "epi_models": import_module("tnfr_core.equations.epi_models"),
    "phases": import_module("tnfr_core.equations.phases"),
    "utils": import_module("tnfr_core.equations.utils"),
    "coherence_calibration": import_module("tnfr_core.metrics.coherence_calibration"),
    "resonance": import_module("tnfr_core.metrics.resonance"),
    "segmentation": import_module("tnfr_core.metrics.segmentation"),
    "spectrum": import_module("tnfr_core.metrics.spectrum"),
    "operator_detection": import_module("tnfr_core.operators.operator_detection"),
    "operator_labels": import_module("tnfr_core.operators.operator_labels"),
    "cache": import_module("tnfr_core.operators.cache"),
    "cache_settings": import_module("tnfr_core.operators.cache_settings"),
    "structural_time": import_module("tnfr_core.operators.structural_time"),
    "interfaces": import_module("tnfr_core.operators.interfaces"),
    "runtime": _runtime,
    "runtime_shared": import_module("tnfr_core.runtime.shared"),
    "signal": _signal,
    "signal_spectrum": import_module("tnfr_core.signal.spectrum"),
}

for alias, module in _MODULE_ALIASES.items():
    globals()[alias] = module
    sys.modules[f"{__name__}.{alias}"] = module
    if hasattr(module, "__all__"):
        for item in module.__all__:
            if item not in __all__:
                __all__.append(item)

__all__ = list(dict.fromkeys(__all__ + list(_MODULE_ALIASES.keys())))


def __getattr__(name: str) -> Any:
    for module in (_equations, _metrics, _operators, _signal, _runtime):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    names = {"equations", "metrics", "operators", "signal", *__all__}
    names.update(key for key in globals() if not key.startswith("_"))
    return sorted(names)
