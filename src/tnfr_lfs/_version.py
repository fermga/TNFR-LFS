"""Utilities for retrieving and validating the package version."""

def _version_from_sources() -> str:
    """Return the version parsed from repository sources.

    This is a fallback mechanism for development environments where the
    distribution metadata has not been generated yet.
    """

    from pathlib import Path
    import re

    candidates = []
    resolved = Path(__file__).resolve()
    parents = resolved.parents
    if len(parents) >= 2:
        candidates.append(parents[1] / "CHANGELOG.md")
    if len(parents) >= 3:
        candidates.append(parents[2] / "CHANGELOG.md")

    for changelog in candidates:
        if not changelog.is_file():
            continue
        for line in changelog.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^## v(?P<version>\d+\.\d+\.\d+)\b", line)
            if match:
                return match.group("version")

    raise RuntimeError(
        "Unable to determine the 'tnfr_lfs' version from package metadata or "
        "repository sources."
    )


from importlib import metadata

from packaging.version import InvalidVersion, Version


def _load_version() -> str:
    """Return the validated package version.

    The version is loaded from the installed package metadata and must conform
    to the ``MAJOR.MINOR.PATCH`` semantic versioning scheme.
    """

    package_name = "tnfr_lfs"

    try:
        raw_version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        raw_version = _version_from_sources()

    try:
        parsed = Version(raw_version)
    except InvalidVersion as exc:
        raise RuntimeError(
            "Invalid version string for 'tnfr_lfs': "
            f"{raw_version!r}. Expected a semantic version."
        ) from exc

    if len(parsed.release) != 3:
        raise RuntimeError(
            "The 'tnfr_lfs' version must follow the MAJOR.MINOR.PATCH format. "
            f"Found: {raw_version!r}."
        )

    return raw_version


__version__ = _load_version()

__all__ = ["__version__"]
