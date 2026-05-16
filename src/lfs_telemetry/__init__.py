"""LFS Race Engineer: real-time telemetry capture and Studio overlay for Live for Speed.

Pipeline: LFS UDP/TCP telemetry → live ``StintTelemetry`` / ``LapTelemetry``
primitives → PySide6 Studio overlay (``lfs_telemetry.studio``).
"""

from __future__ import annotations

__version__ = "0.2.0"
