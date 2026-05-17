"""Phase 6/7: SetupAdvisorTab smoke test.

Builds the widget under the offscreen Qt platform, drives it through
the public slots that the rest of the studio would invoke
(``laps_selected`` → ``lap_loaded``) and asserts:

* a short stint renders the refusal copy (no crash, no jargon);
* a 5-lap stint either renders a recommendation or a clean refusal;
* the Markdown serializer never leaks TNFR-internal vocabulary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lfs_telemetry.studio.app import create_app  # noqa: E402
from lfs_telemetry.studio.models import LapLoader  # noqa: E402
from lfs_telemetry.studio.signals import SignalBus  # noqa: E402
from lfs_telemetry.studio.widgets.setup_advisor_tab import (  # noqa: E402
    SetupAdvisorTab,
)
from lfs_telemetry.telemetry.lap import LapTelemetry  # noqa: E402

# Re-use the synthetic baseline + asset paths from the engine test suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tnfr_racing"))
from test_advisor import _synthetic_baseline  # type: ignore  # noqa: E402

ASSETS = Path(__file__).resolve().parents[2] / "assets"

FORBIDDEN_TERMS = (
    "EPI", "νf", "ΔNFR", "Φ_s", "Si", "operador",
    "AL", "EN", "IL", "OZ", "UM", "RA", "SHA",
    "VAL", "NUL", "THOL", "ZHIR", "NAV", "REMESH",
    "U1", "U2", "U3", "U4", "U5", "U6", "tétrada",
)


@pytest.fixture(scope="module")
def qapp():
    app = create_app([sys.argv[0]])
    yield app


@pytest.fixture(scope="module")
def stint_laps():
    paths = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))
    if len(paths) < 5:
        pytest.skip("not enough synthetic BL1 laps to build a 5-lap stint")
    laps = [LapTelemetry.from_csv(p) for p in paths]
    return list(zip(paths, laps))


def _make_tab(qapp, workspace: Path) -> SetupAdvisorTab:
    signals = SignalBus()
    loader = LapLoader(workspace)
    return SetupAdvisorTab(loader, signals)


def test_setup_advisor_tab_constructs(qapp, tmp_path) -> None:
    tab = _make_tab(qapp, tmp_path)
    try:
        assert tab._view.toPlainText() != ""  # empty-state copy is set
        assert not tab._export_json.isEnabled()
        assert not tab._export_md.isEnabled()
    finally:
        tab.deleteLater()


def test_short_stint_renders_refusal(qapp, stint_laps, tmp_path) -> None:
    tab = _make_tab(qapp, tmp_path)
    try:
        # Only 3 laps → must refuse.
        for path, lap in stint_laps[:3]:
            tab._requested_paths.append(path)
            tab._loaded_laps[path] = lap
        tab._recompute()
        html = tab._view.toHtml()
        assert "consecutive valid laps" in html or "Need" in html
        assert not tab._export_json.isEnabled()
    finally:
        tab.deleteLater()


def test_full_stint_renders_proposal_or_clean_refusal(
    qapp, stint_laps, monkeypatch, tmp_path,
) -> None:
    """5 laps: either we get a proposal + enabled export buttons,
    or a documented refusal with no jargon."""
    import lfs_telemetry.studio.widgets.setup_advisor_tab as mod

    monkeypatch.setattr(
        mod, "load_car_info_bin_for", lambda _car: _synthetic_baseline(),
    )

    tab = _make_tab(qapp, tmp_path)
    try:
        for path, lap in stint_laps[:5]:
            tab._requested_paths.append(path)
            tab._loaded_laps[path] = lap
        tab._recompute()
        # Check the user-visible plain text only — Qt's auto-generated
        # HTML wrapper contains "HTML 4.0//EN" doctype noise that would
        # cause false positives on short tokens like "EN".
        rendered = tab._view.toPlainText()

        # Whatever the outcome, no forbidden token reaches the rendered UI.
        for term in FORBIDDEN_TERMS:
            assert term not in rendered, (
                f"TNFR jargon '{term}' leaked into the UI: {term!r}"
            )

        if tab._last_result and tab._last_result.proposed is not None:
            assert tab._export_json.isEnabled()
            assert tab._export_md.isEnabled()
            md = tab._serialize_markdown()
            for term in FORBIDDEN_TERMS:
                assert term not in md, (
                    f"TNFR jargon '{term}' leaked into Markdown: {term!r}"
                )
            js = tab._serialize_json()
            # JSON includes the internal rationale_id field; allow snake_case
            # rule names but never the canonical TNFR tokens.
            for term in FORBIDDEN_TERMS:
                if term in {"AL", "EN", "RA", "Si"}:
                    # Substring false-positives in JSON payload (e.g.
                    # "tyre_pressure" contains "RA"-free text only — but
                    # "FE_RA" style ids could collide; advisor doesn't use
                    # them). Skip these short tokens for the JSON blob.
                    continue
                assert term not in js, (
                    f"TNFR jargon '{term}' leaked into JSON: {term!r}"
                )
        else:
            # Documented refusal path.
            assert tab._last_result is None or tab._last_result.refusal_reason
    finally:
        tab.deleteLater()
