"""Embedded TNFR playbook resources."""

from importlib import resources

__all__ = [
    "PLAYBOOK_RESOURCE",
]

PLAYBOOK_RESOURCE = resources.files(__name__).joinpath("tnfr_playbook.toml")
