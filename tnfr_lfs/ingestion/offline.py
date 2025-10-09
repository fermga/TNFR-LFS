"""Utilities for ingesting telemetry from offline artefacts."""

from __future__ import annotations

from tnfr_lfs.io.logs import DeterministicReplayer, iter_run, write_run
from tnfr_lfs.io.playbook import load_playbook
from tnfr_lfs.io.profiles import (
    AeroProfile,
    ProfileManager,
    ProfileObjectives,
    ProfileSnapshot,
    ProfileTolerances,
    StintMetrics,
)
from tnfr_lfs.io.raf import (
    RafCarStatic,
    RafFile,
    RafFrame,
    RafHeader,
    RafWheelFrame,
    RafWheelStatic,
    raf_to_telemetry_records,
    read_raf,
)
from tnfr_lfs.io.replay_csv_bundle import ReplayCSVBundleReader

__all__ = [
    "AeroProfile",
    "DeterministicReplayer",
    "ProfileManager",
    "ProfileObjectives",
    "ProfileSnapshot",
    "ProfileTolerances",
    "RafCarStatic",
    "RafFile",
    "RafFrame",
    "RafHeader",
    "RafWheelFrame",
    "RafWheelStatic",
    "ReplayCSVBundleReader",
    "StintMetrics",
    "iter_run",
    "load_playbook",
    "raf_to_telemetry_records",
    "read_raf",
    "write_run",
]
