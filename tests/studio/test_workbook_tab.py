"""Smoke tests for the MoTeC-style :class:`WorkbookTab`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lfs_telemetry.studio.app import create_app  # noqa: E402
from lfs_telemetry.studio.models import LapLoader  # noqa: E402
from lfs_telemetry.studio.signals import SignalBus  # noqa: E402
from lfs_telemetry.studio.widgets.workbook_tab import (  # noqa: E402
    WorkbookTab,
    _ComponentCard,
)
from lfs_telemetry.studio.workbooks import (  # noqa: E402
    Component,
    Workbook,
    Worksheet,
    builtin_template_names,
)


@pytest.fixture(scope="module")
def qapp():
    return create_app([sys.argv[0]])


@pytest.fixture
def tab(qapp, tmp_path, monkeypatch):
    # Sandbox the user-workbook directory so saves don't pollute
    # %APPDATA%.
    from lfs_telemetry.studio import workbooks as wb_mod

    monkeypatch.setattr(wb_mod, "user_workbooks_dir", lambda: tmp_path)
    bus = SignalBus()
    loader = LapLoader(bus)
    tab = WorkbookTab(loader, bus)
    yield tab
    tab.deleteLater()


def test_workbook_tab_constructs_with_default(tab):
    # Default workbook has multiple worksheets and at least one
    # component each.
    assert tab._workbook.worksheets, "default workbook should be non-empty"
    assert tab._ws_tabs.count() == len(tab._workbook.worksheets)


def test_workbook_tab_lists_every_builtin_template(tab):
    labels = [tab._wb_combo.itemText(i) for i in range(tab._wb_combo.count())]
    for name in builtin_template_names():
        assert any(name in label for label in labels), name


def test_add_worksheet_and_component_round_trip(tab):
    start = len(tab._workbook.worksheets)
    tab._workbook.worksheets.append(Worksheet(title="extra"))
    tab._rebuild_worksheet_tabs()
    assert tab._ws_tabs.count() == start + 1

    tab._ws_tabs.setCurrentIndex(start)  # the new "extra" sheet
    tab._add_component(kind="graph")
    bundle = tab._worksheet_widgets[start]
    _splitter, cards = bundle
    assert len(cards) == 1
    assert isinstance(cards[0], _ComponentCard)
    assert cards[0].component.type == "graph"


def test_channels_changed_routes_to_active_card(tab):
    # Force a known active card with a graph component.
    bundle = tab._worksheet_widgets[tab._ws_tabs.currentIndex()]
    _splitter, cards = bundle
    target = next((c for c in cards if c.component.type == "graph"), None)
    if target is None:
        pytest.skip("first worksheet has no graph card")
    tab._on_card_activated(target)
    new_channels = ["throttle", "brake"]
    tab._signals.channels_changed.emit(new_channels)
    assert target.component.channels == new_channels


def test_axis_kind_propagates_to_every_card(tab):
    tab._axis_combo.setCurrentIndex(1)  # "time"
    assert tab._axis_kind == "time"
    for _s, cards in tab._worksheet_widgets.values():
        for c in cards:
            chart = c.chart()
            if chart is not None:
                assert chart._axis_kind == "time"
