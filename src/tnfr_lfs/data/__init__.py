"""Embedded data resources for TNFR × LFS."""

from __future__ import annotations

from importlib import resources

__all__ = [
    "THRESHOLD_PROFILES_RESOURCE",
    "FUSION_CALIBRATION_RESOURCE",
    "CONTEXT_FACTORS_RESOURCE",
    "PLAYBOOK_RESOURCE",
    "__doc__",
]


def _resource(name: str):
    return resources.files(__name__).joinpath(name)


THRESHOLD_PROFILES_RESOURCE = _resource("threshold_profiles.toml")
FUSION_CALIBRATION_RESOURCE = _resource("fusion_calibration.toml")
CONTEXT_FACTORS_RESOURCE = _resource("context_factors.toml")
PLAYBOOK_RESOURCE = _resource("playbooks/tnfr_playbook.toml")
