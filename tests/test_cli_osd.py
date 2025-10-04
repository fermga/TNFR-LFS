"""Tests for the live HUD OSD helper."""

from __future__ import annotations

from tnfr_lfs.acquisition import ButtonEvent, ButtonLayout, OverlayManager
from tnfr_lfs.cli.osd import HUDPager, TelemetryHUD


def _populate_hud(records) -> TelemetryHUD:
    hud = TelemetryHUD(car_model="generic_gt", track_name="valencia", plan_interval=0.0)
    for record in records:
        hud.append(record)
    return hud


def test_osd_pages_fit_within_button_limit(synthetic_records):
    hud = _populate_hud(synthetic_records[:120])
    pages = hud.pages()
    assert len(pages) == 3
    for page in pages:
        assert len(page.encode("utf8")) <= OverlayManager.MAX_BUTTON_TEXT - 1

    page_a, page_b, page_c = pages
    assert "ΔNFR" in page_a and "Acop" in page_a
    assert "ΔNFR" in page_b and "Modo" in page_b
    assert ("Hint" in page_c) or ("Plan" in page_c)


def test_hud_pager_cycles_on_button_click(synthetic_records):
    hud = _populate_hud(synthetic_records[:90])
    pager = HUDPager(hud.pages())
    layout = ButtonLayout().clamp()

    first_page = pager.current()
    click_event = ButtonEvent(
        ucid=layout.ucid,
        click_id=layout.click_id,
        inst=layout.inst,
        type_in=0,
    )
    assert pager.handle_event(click_event, layout)
    assert pager.current() != first_page

    unchanged_event = ButtonEvent(
        ucid=layout.ucid,
        click_id=layout.click_id + 1,
        inst=layout.inst,
        type_in=0,
    )
    current = pager.current()
    assert not pager.handle_event(unchanged_event, layout)
    assert pager.current() == current

    typed_event = ButtonEvent(
        ucid=layout.ucid,
        click_id=layout.click_id,
        inst=layout.inst,
        type_in=1,
    )
    assert not pager.handle_event(typed_event, layout)
