from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from tnfr_lfs.plugins.template import canonical_plugins_mapping, render_plugins_template

from tests.helpers import plugin_registry_state


_CANONICAL_MAPPING, _CANONICAL_SOURCE = canonical_plugins_mapping()


@pytest.fixture(autouse=True)
def plugin_registry():
    with plugin_registry_state() as register:
        yield register


@pytest.fixture()
def canonical_plugins_payload() -> tuple[dict[str, Any], Path]:
    """Return a mutable copy of the canonical plugin configuration payload."""

    return copy.deepcopy(_CANONICAL_MAPPING), _CANONICAL_SOURCE


@pytest.fixture()
def canonical_plugins_template_text(
    canonical_plugins_payload: tuple[dict[str, Any], Path]
) -> str:
    mapping, _ = canonical_plugins_payload
    return render_plugins_template(mapping)
