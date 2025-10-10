"""Tests for the package version metadata."""

from packaging.version import Version

import tnfr_lfs


def test_version_is_semver_patch():
    version = Version(tnfr_lfs.__version__)

    assert len(version.release) == 3, (
        "tnfr_lfs.__version__ must contain exactly three release components"
    )
