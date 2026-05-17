"""Shared pytest fixtures for the test suite."""
from __future__ import annotations

import pytest

from lfs_telemetry import lfs_paths


@pytest.fixture(autouse=True)
def _clear_static_autodetect_cache():
    """Reset the lru_cache so monkeypatched module attributes apply."""
    lfs_paths._static_autodetect_candidates.cache_clear()
    yield
    lfs_paths._static_autodetect_candidates.cache_clear()
