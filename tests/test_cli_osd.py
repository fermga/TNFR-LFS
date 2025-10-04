"""Tests for the live HUD OSD helper."""

from __future__ import annotations

from typing import Tuple

from tnfr_lfs.acquisition import ButtonEvent, ButtonLayout, MacroQueue, OverlayManager
from tnfr_lfs.cli import osd as osd_module
from tnfr_lfs.cli.osd import HUDPager, MacroStatus, OSDController, TelemetryHUD
from tnfr_lfs.exporters.setup_plan import SetupChange, SetupPlan


def _populate_hud(records) -> TelemetryHUD:
    hud = TelemetryHUD(car_model="generic_gt", track_name="valencia", plan_interval=0.0)
    for record in records:
        hud.append(record)
    return hud


class DummyHUD:
    def __init__(self, plan: SetupPlan) -> None:
        self._plan = plan
        self._status = MacroStatus()
        self._pages: Tuple[str, str, str, str] = (
            "Página A",
            "Página B",
            "Página C",
            "Aplicar recomendaciones",
        )

    def pages(self) -> Tuple[str, str, str, str]:
        return self._pages

    def plan(self) -> SetupPlan | None:
        return self._plan

    def update_macro_status(self, status: MacroStatus) -> None:
        self._status = status
        warnings = [f"⚠️ {warning}" for warning in status.warnings if warning]
        lines = ["Aplicar recomendaciones"]
        if status.queue_size:
            lines.append(f"Cola macros {status.queue_size}")
        if warnings:
            lines.extend(warnings)
        self._pages = (
            self._pages[0],
            self._pages[1],
            self._pages[2],
            "\n".join(lines),
        )


def test_osd_pages_fit_within_button_limit(synthetic_records):
    hud = _populate_hud(synthetic_records[:120])
    pages = hud.pages()
    assert len(pages) == 4
    for page in pages:
        assert len(page.encode("utf8")) <= OverlayManager.MAX_BUTTON_TEXT - 1

    page_a, page_b, page_c, page_d = pages
    assert "ΔNFR" in page_a and "Si" in page_a and "∇Acop" in page_a
    assert "ΔNFR" in page_b and "Modo" in page_b
    assert ("Hint" in page_c) or ("Plan" in page_c) or ("bias" in page_c.lower())
    assert "Aplicar" in page_d


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


def test_macro_queue_respects_timing():
    events: list[tuple[float, str]] = []
    current_time = 0.0

    def fake_time() -> float:
        return current_time

    def sender(command: str) -> None:
        events.append((fake_time(), command))

    queue = MacroQueue(sender, min_interval=0.2, time_fn=fake_time)
    queue.enqueue_press_sequence(["F12", "+", "+"], spacing=0.25)

    assert len(queue) == 3
    assert queue.pending()[0] == "/press F12"

    dispatched = queue.tick()
    assert dispatched == 1
    assert events[0][1] == "/press F12"
    assert len(queue) == 2

    current_time += 0.2
    assert queue.tick() == 0  # spacing not yet elapsed
    current_time += 0.1
    assert queue.tick() == 1
    assert events[-1][1] == "/press +"


def test_hud_macro_status_injects_warnings():
    change = SetupChange(
        parameter="Presión delantera",
        delta=1.5,
        rationale="",
        expected_effect="",
    )
    status = MacroStatus(
        next_change=change,
        warnings=("Abre el menú de boxes (F12)", "Detén el coche antes de aplicar"),
        queue_size=2,
    )
    page_d = osd_module._render_page_d((change,), status)
    assert "Cola macros 2" in page_d
    assert page_d.count("⚠️") == 2


def test_osd_controller_applies_macro_on_trigger():
    plan = SetupPlan(
        car_model="generic_gt",
        session=None,
        changes=(
            SetupChange(parameter="Alerón", delta=2.0, rationale="", expected_effect=""),
            SetupChange(parameter="Amortiguador", delta=-1.0, rationale="", expected_effect=""),
        ),
        rationales=(),
        expected_effects=(),
        sensitivities={},
    )
    hud = DummyHUD(plan)
    controller = OSDController(
        host="127.0.0.1",
        outsim_port=1,
        outgauge_port=2,
        insim_port=3,
        insim_keepalive=1.0,
        hud=hud,
    )
    layout = controller.layout

    sent_commands: list[str] = []
    current_time = 0.0

    def fake_time() -> float:
        return current_time

    controller._macro_queue = MacroQueue(sent_commands.append, min_interval=0.05, time_fn=fake_time)
    controller._car_stopped = True
    controller.set_menu_open(True)
    controller._pager.update(controller.hud.pages())
    while controller._pager.index < len(controller._pager.pages) - 1:
        controller._pager.advance()

    event = ButtonEvent(
        ucid=layout.ucid,
        click_id=layout.click_id,
        inst=layout.inst,
        type_in=0,
        flags=controller.APPLY_TRIGGER_FLAG,
    )
    controller._handle_button_event(event)
    assert len(controller._macro_queue) > 0
    assert controller._resolve_next_change() is not None

    # Dispatch queued commands
    controller._macro_queue.tick()
    assert sent_commands
    assert sent_commands[0].startswith("/press F12")


def test_osd_controller_blocks_macro_when_preflight_fails():
    plan = SetupPlan(
        car_model="generic_gt",
        session=None,
        changes=(
            SetupChange(parameter="Alerón", delta=2.0, rationale="", expected_effect=""),
        ),
        rationales=(),
        expected_effects=(),
        sensitivities={},
    )
    hud = DummyHUD(plan)
    controller = OSDController(
        host="127.0.0.1",
        outsim_port=1,
        outgauge_port=2,
        insim_port=3,
        insim_keepalive=1.0,
        hud=hud,
    )
    layout = controller.layout

    commands: list[str] = []
    controller._macro_queue = MacroQueue(commands.append, min_interval=0.05)
    controller._car_stopped = False
    controller.set_menu_open(False)
    controller._pager.update(controller.hud.pages())
    while controller._pager.index < len(controller._pager.pages) - 1:
        controller._pager.advance()

    warning_event = ButtonEvent(
        ucid=layout.ucid,
        click_id=layout.click_id,
        inst=layout.inst,
        type_in=0,
        flags=controller.APPLY_TRIGGER_FLAG,
    )
    controller._handle_button_event(warning_event)
    assert len(commands) == 0
    page_d = controller.hud.pages()[3]
    assert "⚠️" in page_d
