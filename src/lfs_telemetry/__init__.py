"""LFS Race Engineer: real-time telemetry capture and Studio overlay for Live for Speed.

Pipeline: LFS UDP/TCP telemetry → live ``StintTelemetry`` / ``LapTelemetry``
primitives → PySide6 Studio overlay (``lfs_telemetry.studio``).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("lfs-race-engineer")
except PackageNotFoundError:  # pragma: no cover - editable/source tree
    __version__ = "0.0.0+local"
