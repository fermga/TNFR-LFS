"""Regression tests for the :mod:`tnfr_lfs.core` compatibility bridge."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_core_shim_issues_deprecation_warning():
    """Importing :mod:`tnfr_lfs.core` emits the expected deprecation warning."""

    sys.modules.pop("tnfr_lfs.core", None)

    warning_pattern = (
        "'tnfr_lfs.core' is deprecated and will be removed in a future release; "
        "import from 'tnfr_core' instead."
    )

    with pytest.warns(DeprecationWarning, match=warning_pattern):
        importlib.import_module("tnfr_lfs.core")
