"""Tests for the package version metadata."""

import importlib

import pytest
from packaging.version import Version

import tnfr_lfs
from tnfr_lfs import _version as version_module


def test_version_is_semver_patch():
    version = Version(tnfr_lfs.__version__)

    assert len(version.release) == 3, (
        "tnfr_lfs.__version__ must contain exactly three release components"
    )


def test_version_override_from_semantic_release(monkeypatch):
    monkeypatch.setenv("PYTHON_SEMANTIC_RELEASE_VERSION", "9.8.7")
    importlib.reload(version_module)
    reloaded = importlib.reload(tnfr_lfs)

    assert reloaded.__version__ == "9.8.7"

    monkeypatch.setenv("PYTHON_SEMANTIC_RELEASE_VERSION", "invalid-version")
    with pytest.raises(RuntimeError):
        importlib.reload(version_module)

    monkeypatch.delenv("PYTHON_SEMANTIC_RELEASE_VERSION", raising=False)
    importlib.reload(version_module)
    importlib.reload(tnfr_lfs)
