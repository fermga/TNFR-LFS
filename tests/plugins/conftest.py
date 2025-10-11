from __future__ import annotations

import pytest

from tests.helpers import plugin_registry_state


@pytest.fixture(autouse=True)
def plugin_registry():
    with plugin_registry_state() as register:
        yield register
