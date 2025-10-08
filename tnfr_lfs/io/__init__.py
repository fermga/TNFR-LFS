"""IO utilities for TNFR × LFS."""

from .logs import DeterministicReplayer, iter_run, write_run
from .playbook import load_playbook
from .replay_csv_bundle import ReplayCSVBundleReader
from .raf import (
    RafCarStatic,
    RafFile,
    RafFrame,
    RafHeader,
    RafWheelFrame,
    RafWheelStatic,
    raf_to_telemetry_records,
    read_raf,
)
from .profiles import (
    AeroProfile,
    ProfileManager,
    ProfileObjectives,
    ProfileSnapshot,
    ProfileTolerances,
    StintMetrics,
)

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
