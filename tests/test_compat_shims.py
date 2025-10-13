"""Regression tests for deprecated import shims in :mod:`tnfr_lfs`."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable

import pytest


_SHIM_MODULES = (
    "tnfr_lfs.cache_settings",
    "tnfr_lfs.ingestion",
    "tnfr_lfs.io",
    "tnfr_lfs.processing",
)


def _purge_modules(names: Iterable[str]) -> None:
    prefixes = tuple(f"{name}." for name in names)
    for module_name in list(sys.modules):
        if module_name in names or module_name.startswith(prefixes):
            sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True)
def reset_shim_modules() -> None:
    """Ensure shim modules are freshly imported for each test."""

    _purge_modules(_SHIM_MODULES)
    try:
        yield
    finally:
        _purge_modules(_SHIM_MODULES)


def test_cache_settings_shim_warns_and_exports() -> None:
    """Importing :mod:`tnfr_lfs.cache_settings` warns and exposes key symbols."""

    warning_pattern = (
        "'tnfr_lfs.cache_settings' is deprecated and will be removed in a future release. "
        "Please import from 'tnfr_core.operators.cache_settings' instead."
    )

    with pytest.warns(DeprecationWarning, match=warning_pattern):
        cache_settings = importlib.import_module("tnfr_lfs.cache_settings")

    from tnfr_core.operators.cache_settings import CacheOptions as core_cache_options

    assert cache_settings.CacheOptions is core_cache_options


def test_ingestion_shim_warns_and_exports() -> None:
    """Importing :mod:`tnfr_lfs.ingestion` warns and mirrors telemetry exports."""

    warning_pattern = (
        "'tnfr_lfs.ingestion' is deprecated and will be removed in a future release; "
        "import from 'tnfr_lfs.telemetry' instead."
    )

    with pytest.warns(DeprecationWarning, match=warning_pattern):
        ingestion = importlib.import_module("tnfr_lfs.ingestion")

    from tnfr_lfs.telemetry import iter_run as telemetry_iter_run
    from tnfr_lfs.telemetry import offline as telemetry_offline

    assert ingestion.iter_run is telemetry_iter_run
    assert importlib.import_module("tnfr_lfs.ingestion.offline") is telemetry_offline


def test_io_shim_warns_on_attribute_access() -> None:
    """Accessing :mod:`tnfr_lfs.io` members triggers the deprecation warning."""

    warning_pattern = (
        "'tnfr_lfs.io' is deprecated and will be removed in a future release; "
        "import from 'tnfr_lfs.telemetry.offline' instead."
    )

    io_module = importlib.import_module("tnfr_lfs.io")

    with pytest.warns(DeprecationWarning, match=warning_pattern):
        replay_reader = io_module.ReplayCSVBundleReader

    from tnfr_lfs.telemetry.offline.replay_csv_bundle import (
        ReplayCSVBundleReader as offline_replay_reader,
    )

    assert replay_reader is offline_replay_reader


def test_processing_shim_warns_and_exports() -> None:
    """Importing :mod:`tnfr_lfs.processing` warns and exposes insights helpers."""

    warning_pattern = (
        "'tnfr_lfs.processing' is deprecated; use 'tnfr_lfs.analysis.insights' instead."
    )

    with pytest.warns(DeprecationWarning, match=warning_pattern):
        processing = importlib.import_module("tnfr_lfs.processing")

    from tnfr_lfs.analysis.insights import compute_insights as insights_compute

    assert processing.compute_insights is insights_compute
