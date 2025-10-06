"""Helpers to load TNFR playbook suggestions."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, MutableMapping, Sequence

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from importlib import resources

_DEFAULT_RESOURCE = ("tnfr_lfs.data.playbooks", "tnfr_playbook.toml")


def _iter_actions(value: object) -> Iterable[str]:
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, Mapping):
        actions = value.get("actions")
        yield from _iter_actions(actions)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _iter_actions(item)
        return


def load_playbook(source: str | Path | None = None) -> Mapping[str, tuple[str, ...]]:
    """Load the bundled TNFR playbook suggestions.

    Parameters
    ----------
    source:
        Optional filesystem path to a TOML playbook. When omitted the embedded
        resource is used.
    """

    if source is None:
        package, name = _DEFAULT_RESOURCE
        with resources.open_binary(package, name) as handle:
            payload = tomllib.load(handle)
    else:
        path = Path(source)
        with path.open("rb") as handle:
            payload = tomllib.load(handle)

    rules: MutableMapping[str, tuple[str, ...]] = {}
    section = payload.get("rules") if isinstance(payload, Mapping) else None
    if isinstance(section, Mapping):
        for raw_key, raw_value in section.items():
            key = str(raw_key).strip()
            if not key:
                continue
            actions = tuple(dict.fromkeys(action.strip() for action in _iter_actions(raw_value) if action.strip()))
            if actions:
                rules[key] = actions
    return MappingProxyType(dict(rules))


__all__ = ["load_playbook"]
