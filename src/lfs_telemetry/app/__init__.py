"""Capture-process support modules for the LFS Race Engineer Studio.

This package used to ship a Dash/Plotly viewer; that legacy UI was
removed in favour of the PySide6 ``studio`` package. The two modules
that survive (:mod:`capture_runner`, :mod:`state`) are still consumed
by the Studio Capture / Live tabs to spawn and manage the headless
``lfs-telemetry capture`` subprocess.
"""

from __future__ import annotations

from .. import __version__

__all__ = ["__version__"]
