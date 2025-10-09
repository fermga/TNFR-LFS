"""Removed module shim for `tnfr_lfs.acquisition`."""

from __future__ import annotations

import importlib

# Import the new namespace so the error message can point to the correct module.
importlib.import_module("tnfr_lfs.ingestion")

raise ImportError(
    "'tnfr_lfs.acquisition' has been removed; import from 'tnfr_lfs.ingestion' instead."
)
