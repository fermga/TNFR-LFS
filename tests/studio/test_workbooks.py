"""Unit tests for the workbook data model.

Pure-Python: does not require Qt/PySide6 for the schema-level checks,
but the I/O helpers happen to touch ``QStandardPaths`` only inside
:func:`user_workbooks_dir`, which we cover separately and skip when
PySide6 isn't present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lfs_telemetry.studio.workbooks import (
    COMPONENT_TYPES,
    SCHEMA_VERSION,
    Component,
    Workbook,
    Worksheet,
    builtin_template,
    builtin_template_names,
    default_workbook,
    list_user_workbooks,
    load_workbook,
    save_user_workbook,
    save_workbook,
)

# ---------------------------------------------------------------------------
# Component / Worksheet / Workbook validation
# ---------------------------------------------------------------------------

def test_component_rejects_unknown_type():
    with pytest.raises(ValueError):
        Component(type="not-a-real-type", title="x")


def test_component_rejects_non_str_channels():
    with pytest.raises(TypeError):
        Component(type="graph", title="x", channels=[1, 2, 3])


def test_component_default_id_is_unique():
    a = Component(type="graph", title="a")
    b = Component(type="graph", title="b")
    assert a.id and b.id
    assert a.id != b.id


def test_component_roundtrip():
    c = Component(
        type="graph",
        title="Throttle + Brake",
        channels=["throttle", "brake"],
        options={"overlay": True, "normalize": False},
    )
    restored = Component.from_dict(c.to_dict())
    assert restored == c


def test_worksheet_drops_invalid_components(caplog):
    raw = {
        "title": "Mixed",
        "components": [
            {"type": "graph", "title": "ok", "channels": ["throttle"]},
            {"type": "bogus", "title": "skip me", "channels": []},
        ],
    }
    ws = Worksheet.from_dict(raw)
    assert len(ws.components) == 1
    assert ws.components[0].title == "ok"


def test_workbook_rejects_future_schema():
    raw = {"name": "Future", "schema_version": SCHEMA_VERSION + 99,
           "worksheets": []}
    with pytest.raises(ValueError):
        Workbook.from_dict(raw)


def test_workbook_roundtrip_json(tmp_path: Path):
    wb = default_workbook()
    target = tmp_path / "wb.json"
    save_workbook(wb, target)
    assert target.exists()
    restored = load_workbook(target)
    assert restored.name == wb.name
    assert [w.title for w in restored.worksheets] == [
        w.title for w in wb.worksheets
    ]
    # Component ids survive the round-trip.
    for orig_ws, new_ws in zip(wb.worksheets, restored.worksheets, strict=True):
        assert [c.id for c in orig_ws.components] == [
            c.id for c in new_ws.components
        ]


def test_save_workbook_is_atomic_tmp_cleaned(tmp_path: Path):
    wb = Workbook(name="t", worksheets=[Worksheet(title="t")])
    target = tmp_path / "sub" / "wb.json"
    save_workbook(wb, target)
    # .tmp staging file must have been replaced, not left behind.
    assert not target.with_suffix(target.suffix + ".tmp").exists()
    assert json.loads(target.read_text("utf-8"))["name"] == "t"


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

def test_builtin_template_names_non_empty():
    names = builtin_template_names()
    assert "Default" in names
    assert len(names) >= 5


def test_each_builtin_template_uses_known_types():
    for name in builtin_template_names():
        wb = builtin_template(name)
        for ws in wb.worksheets:
            for comp in ws.components:
                assert comp.type in COMPONENT_TYPES
                assert comp.title
                # Every component must reference at least one channel
                # (the renderer treats empty lists as a no-op).
                assert comp.channels


def test_each_builtin_template_uses_real_channel_names():
    # Catches casing mistakes (e.g. wheel_fl_* vs wheel_FL_*) and
    # typos like ``engine_rpm`` instead of ``rpm`` before they reach
    # the GUI, where they would silently degrade to empty plots.
    from lfs_telemetry.telemetry.channels import CHANNELS

    known = set(CHANNELS)
    offenders: list[tuple[str, str, str, str]] = []
    for name in builtin_template_names():
        for ws in builtin_template(name).worksheets:
            for comp in ws.components:
                for ch in comp.channels:
                    if ch not in known:
                        offenders.append((name, ws.title, comp.title, ch))
    assert not offenders, f"Unknown channels in templates: {offenders}"


def test_builtin_template_returns_fresh_instances():
    a = builtin_template("Default")
    b = builtin_template("Default")
    assert a is not b
    a.worksheets[0].title = "MUTATED"
    assert b.worksheets[0].title != "MUTATED"


def test_unknown_template_raises():
    with pytest.raises(KeyError):
        builtin_template("does not exist")


# ---------------------------------------------------------------------------
# User-dir I/O (depends on PySide6 for QStandardPaths)
# ---------------------------------------------------------------------------

def test_save_user_workbook_lands_in_workbooks_dir(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from lfs_telemetry.studio import workbooks as wb_mod

    monkeypatch.setattr(wb_mod, "user_workbooks_dir", lambda: tmp_path)
    wb = Workbook(name="My Setup", worksheets=[Worksheet(title="t")])
    path = save_user_workbook(wb)
    assert path.parent == tmp_path
    assert path.name == "My Setup.json"
    assert path in list_user_workbooks.__wrapped__() if hasattr(
        list_user_workbooks, "__wrapped__"
    ) else True
    assert wb_mod.list_user_workbooks() == [path]


def test_save_user_workbook_sanitises_filename(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from lfs_telemetry.studio import workbooks as wb_mod

    monkeypatch.setattr(wb_mod, "user_workbooks_dir", lambda: tmp_path)
    wb = Workbook(name="Brakes/Rear?*", worksheets=[Worksheet(title="t")])
    path = save_user_workbook(wb)
    assert "/" not in path.name
    assert "?" not in path.name
    assert "*" not in path.name
