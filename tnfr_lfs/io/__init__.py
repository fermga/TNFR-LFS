"""IO utilities for TNFR × LFS (deprecated; use :mod:`tnfr_lfs.ingestion.offline`)."""

from __future__ import annotations

import warnings
from typing import Any

_WARNED_NAMES: set[str] = set()

__all__ = [
    "write_run",
    "iter_run",
    "DeterministicReplayer",
    "ProfileManager",
    "ProfileObjectives",
    "ProfileSnapshot",
    "ProfileTolerances",
    "StintMetrics",
    "AeroProfile",
    "load_playbook",
    "ReplayCSVBundleReader",
    "read_raf",
    "raf_to_telemetry_records",
    "RafHeader",
    "RafCarStatic",
    "RafWheelStatic",
    "RafWheelFrame",
    "RafFrame",
    "RafFile",
]

def __getattr__(name: str) -> Any:
    from tnfr_lfs.ingestion import offline as _offline

    if hasattr(_offline, name):
        if name not in _WARNED_NAMES:
            warnings.warn(
                "'tnfr_lfs.io' is deprecated and will be removed in a future release; "
                "import from 'tnfr_lfs.ingestion.offline' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _WARNED_NAMES.add(name)
        value = getattr(_offline, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tnfr_lfs.io' has no attribute {name!r}")


def __dir__() -> list[str]:
    from tnfr_lfs.ingestion import offline as _offline

    return sorted(set(__all__) | set(dir(_offline)))
