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
    _ComponentEditorDialog,
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


def test_bar_component_renders_without_error(tab):
    # Add a bar component on a fresh worksheet so we control the cards.
    tab._workbook.worksheets.append(Worksheet(title="bars"))
    tab._rebuild_worksheet_tabs()
    idx = len(tab._workbook.worksheets) - 1
    tab._ws_tabs.setCurrentIndex(idx)
    ws = tab._workbook.worksheets[idx]
    ws.components.append(Component(
        type="bar",
        title="tyre temps",
        channels=[
            "wheel_FL_air_temp_c",
            "wheel_FR_air_temp_c",
            "wheel_RL_air_temp_c",
            "wheel_RR_air_temp_c",
        ],
    ))
    tab._rebuild_worksheet_tabs()
    tab._ws_tabs.setCurrentIndex(idx)
    _splitter, cards = tab._worksheet_widgets[idx]
    assert len(cards) == 1
    bar_card = cards[0]
    assert bar_card.component.type == "bar"
    assert bar_card.chart() is None  # bar cards have no MultiChannelChart
    assert getattr(bar_card, "_bar", None) is not None


def test_card_move_reorders_components(tab):
    # Build a fresh worksheet with two graph components so order
    # changes are unambiguous.
    tab._workbook.worksheets.append(Worksheet(title="move-test"))
    tab._rebuild_worksheet_tabs()
    idx = len(tab._workbook.worksheets) - 1
    tab._ws_tabs.setCurrentIndex(idx)
    ws = tab._workbook.worksheets[idx]
    ws.components.append(Component(type="graph", title="A", channels=[]))
    ws.components.append(Component(type="graph", title="B", channels=[]))
    tab._rebuild_worksheet_tabs()
    tab._ws_tabs.setCurrentIndex(idx)
    _splitter, cards = tab._worksheet_widgets[idx]
    assert [c.component.title for c in cards] == ["A", "B"]

    tab._on_card_move(cards[0], +1)  # move A down
    ws = tab._workbook.worksheets[idx]
    assert [c.title for c in ws.components] == ["B", "A"]


def test_splitter_sizes_persist_round_trip(tab):
    # Pick a worksheet that already has >=2 cards so sizes are
    # meaningful; default Driver Inputs sheet has two graphs.
    target_idx = None
    for i, ws in enumerate(tab._workbook.worksheets):
        if len(ws.components) >= 2:
            target_idx = i
            break
    if target_idx is None:
        pytest.skip("default workbook lacks a multi-card worksheet")
    tab._ws_tabs.setCurrentIndex(target_idx)
    splitter, cards = tab._worksheet_widgets[target_idx]

    sizes = [120, 240] + [100] * (len(cards) - 2)
    tab._persist_splitter_sizes(target_idx, sizes)

    restored = tab._restore_splitter_sizes(target_idx, len(cards))
    assert restored == sizes

    # Mismatched card count falls back to defaults.
    fallback = tab._restore_splitter_sizes(target_idx, len(cards) + 1)
    assert fallback == [100] * (len(cards) + 1)

    # Reorder clears the persisted sizes for that sheet.
    if len(cards) >= 2:
        tab._on_card_move(cards[0], +1)
        # After clear, restore returns defaults again.
        wiped = tab._restore_splitter_sizes(target_idx, len(cards))
        assert wiped == [100] * len(cards)


def test_component_editor_dialog_round_trip(qapp):
    comp = Component(
        type="graph",
        title="Driver",
        channels=["throttle"],
        options={"overlay": True, "normalize": False},
    )
    dlg = _ComponentEditorDialog(comp, ["throttle", "brake", "speed_ms"])
    assert dlg.title == "Driver"
    assert dlg.type == "graph"
    assert dlg.channels == ["throttle"]
    # Toggle a second channel and confirm it shows up.
    for i in range(dlg._channels_list.count()):
        item = dlg._channels_list.item(i)
        if item.data(0x0100) == "brake":  # Qt.UserRole == 0x0100
            from PySide6.QtCore import Qt as _Qt
            item.setCheckState(_Qt.CheckState.Checked)
    assert "brake" in dlg.channels


def test_component_editor_keeps_off_schema_channels(qapp):
    comp = Component(
        type="graph",
        title="X",
        channels=["nonexistent_channel"],
        options={},
    )
    dlg = _ComponentEditorDialog(comp, ["throttle", "brake"])
    # The off-schema channel is listed and checked by default so the
    # user can keep it without surprise.
    assert "nonexistent_channel" in dlg.channels


def test_component_editor_caps_at_max(qapp):
    comp = Component(type="graph", title="X", channels=[], options={})
    cols = [f"col_{i}" for i in range(12)]
    dlg = _ComponentEditorDialog(comp, cols)
    from PySide6.QtCore import Qt as _Qt
    for i in range(9):  # tick 9 — over the cap of 8
        dlg._channels_list.item(i).setCheckState(_Qt.CheckState.Checked)
    # Direct accept must be blocked; _maybe_accept won't call accept().
    assert len(dlg.channels) == 9
    # Patch QMessageBox.warning so the modal popup doesn't hang the
    # offscreen test harness.
    from PySide6.QtWidgets import QMessageBox
    import lfs_telemetry.studio.widgets.workbook_tab as wt_mod
    original = wt_mod.QMessageBox.warning
    wt_mod.QMessageBox.warning = staticmethod(
        lambda *a, **kw: QMessageBox.StandardButton.Ok
    )
    try:
        dlg._maybe_accept()
    finally:
        wt_mod.QMessageBox.warning = original
    # Dialog still open (no result code set) — verify accept was not
    # actually triggered by checking it's not visible-accepted.
    assert dlg.result() != 1  # 1 == Accepted


def test_card_edit_updates_component_in_place(tab):
    # Pick a worksheet with at least one graph card.
    target_idx = None
    target_card = None
    for i, (_s, cards) in tab._worksheet_widgets.items():
        for c in cards:
            if c.component.type == "graph":
                target_idx = i
                target_card = c
                break
        if target_card is not None:
            break
    if target_card is None:
        pytest.skip("default workbook has no graph card to edit")

    tab._ws_tabs.setCurrentIndex(target_idx)
    # Monkey-patch the dialog so the test runs headless without modal
    # interaction. We construct a real dialog, mutate it, then short
    # circuit exec().
    from lfs_telemetry.studio.widgets import workbook_tab as wt_mod

    class _StubDlg:
        def __init__(self, comp, cols, parent=None):
            self._comp = comp
        def exec(self):
            from PySide6.QtWidgets import QDialog as _QD
            return _QD.DialogCode.Accepted
        @property
        def title(self): return "renamed"
        @property
        def type(self): return self._comp.type
        @property
        def channels(self): return ["throttle"]
        @property
        def options(self):
            return {"overlay": False, "normalize": True}

    monkey_target = wt_mod._ComponentEditorDialog
    wt_mod._ComponentEditorDialog = _StubDlg
    try:
        tab._on_card_edit(target_card)
    finally:
        wt_mod._ComponentEditorDialog = monkey_target

    assert target_card.component.title == "renamed"
    assert target_card.component.channels == ["throttle"]
    assert target_card.component.options["overlay"] is False
    assert target_card.component.options["normalize"] is True


# ---------------------------------------------------------------------------
# Phase 4a: dirty marker + auto-save on close
# ---------------------------------------------------------------------------


def test_dirty_marker_starts_clean(tab):
    assert tab._dirty is False
    # Save button shows base label (no bullet) on a freshly-loaded
    # default workbook.
    assert "•" not in tab._save_btn.text()


def test_dirty_marker_set_by_mutators(tab):
    # Add a worksheet → dirty.
    tab._workbook.worksheets.append(Worksheet(title="hot"))
    tab._rebuild_worksheet_tabs()
    tab._mark_dirty()  # explicit because _rebuild bypasses _add_worksheet
    assert tab._dirty is True
    assert tab._save_btn.text().endswith("•")


def test_add_component_marks_dirty(tab):
    tab._mark_clean()
    tab._ws_tabs.setCurrentIndex(0)
    tab._add_component(kind="graph")
    assert tab._dirty is True


def test_channels_changed_marks_dirty(tab):
    tab._mark_clean()
    # Pick an active graph card.
    for _s, cards in tab._worksheet_widgets.values():
        for c in cards:
            if c.component.type == "graph":
                tab._on_card_activated(c)
                break
        if tab._active_card is not None:
            break
    if tab._active_card is None:
        pytest.skip("no graph card to receive channel updates")
    new_channels = ["throttle", "brake", "speed_ms"]
    tab._signals.channels_changed.emit(new_channels)
    assert tab._dirty is True
    # Re-emitting the same list should be a no-op (no spurious dirty).
    tab._mark_clean()
    tab._signals.channels_changed.emit(new_channels)
    assert tab._dirty is False


def test_save_workbook_clears_dirty(tab, monkeypatch):
    # Force the workbook to look like a user file so _save_workbook
    # overwrites silently without prompting.
    from lfs_telemetry.studio.widgets import workbook_tab as wt_mod

    saved: list = []
    monkeypatch.setattr(
        wt_mod, "save_user_workbook",
        lambda wb: saved.append(wb) or Path("dummy.json"),
    )
    tab._workbook.name = "test_wb"
    tab._workbook_origin = ("user", "test_wb.json")
    tab._mark_dirty()
    ok = tab._save_workbook(prompt_name=False)
    assert ok is True
    assert tab._dirty is False
    assert len(saved) == 1


def test_flush_on_close_cancel_aborts(tab):
    from PySide6.QtWidgets import QMessageBox
    from lfs_telemetry.studio.widgets import workbook_tab as wt_mod

    tab._mark_dirty()

    class _Evt:
        def __init__(self):
            self.ignored = False

        def ignore(self):
            self.ignored = True

    original = wt_mod.QMessageBox.question
    wt_mod.QMessageBox.question = staticmethod(
        lambda *a, **kw: QMessageBox.StandardButton.Cancel
    )
    try:
        evt = _Evt()
        proceed = tab.flush_on_close(evt)
    finally:
        wt_mod.QMessageBox.question = original

    assert proceed is False
    assert evt.ignored is True
    assert tab._dirty is True  # still dirty


def test_flush_on_close_discard_proceeds(tab):
    from PySide6.QtWidgets import QMessageBox
    from lfs_telemetry.studio.widgets import workbook_tab as wt_mod

    tab._mark_dirty()
    original = wt_mod.QMessageBox.question
    wt_mod.QMessageBox.question = staticmethod(
        lambda *a, **kw: QMessageBox.StandardButton.Discard
    )
    try:
        proceed = tab.flush_on_close(None)
    finally:
        wt_mod.QMessageBox.question = original

    assert proceed is True


def test_flush_on_close_clean_is_noop(tab):
    tab._mark_clean()
    # No QMessageBox patching needed — should never be invoked.
    assert tab.flush_on_close(None) is True


def test_confirm_discard_save_path(tab, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from lfs_telemetry.studio.widgets import workbook_tab as wt_mod

    saved: list = []
    monkeypatch.setattr(
        wt_mod, "save_user_workbook",
        lambda wb: saved.append(wb) or Path("dummy.json"),
    )
    tab._workbook.name = "confirm_wb"
    tab._workbook_origin = ("user", "confirm_wb.json")
    tab._mark_dirty()

    original = wt_mod.QMessageBox.question
    wt_mod.QMessageBox.question = staticmethod(
        lambda *a, **kw: QMessageBox.StandardButton.Save
    )
    try:
        ok = tab._confirm_discard_if_dirty()
    finally:
        wt_mod.QMessageBox.question = original

    assert ok is True
    assert tab._dirty is False
    assert len(saved) == 1
