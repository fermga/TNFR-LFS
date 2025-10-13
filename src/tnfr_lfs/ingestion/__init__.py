"""Deprecated import shim for :mod:`tnfr_lfs.telemetry`."""

from __future__ import annotations

import importlib
import sys
import warnings

_TELEMETRY_MODULE = importlib.import_module("tnfr_lfs.telemetry")

warnings.warn(
    "'tnfr_lfs.ingestion' is deprecated and will be removed in a future release; "
    "import from 'tnfr_lfs.telemetry' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = sorted(_TELEMETRY_MODULE.__all__)

for name in __all__:
    globals()[name] = getattr(_TELEMETRY_MODULE, name)

for submodule in [
    "_reorder_buffer",
    "_socket_poll",
    "config_loader",
    "fusion",
    "insim",
    "live",
    "offline",
    "outgauge_udp",
    "outsim_client",
    "outsim_udp",
    "pools",
    "track_loader",
]:
    module = importlib.import_module(f"tnfr_lfs.telemetry.{submodule}")
    sys.modules[f"{__name__}.{submodule}"] = module
