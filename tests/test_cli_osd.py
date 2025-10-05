"""Tests for the live HUD OSD helper."""

from __future__ import annotations

from typing import Tuple
from types import SimpleNamespace

from tnfr_lfs.acquisition import ButtonEvent, ButtonLayout, MacroQueue, OverlayManager
from tnfr_lfs.cli import osd as osd_module
from tnfr_lfs.cli.osd import HUDPager, MacroStatus, OSDController, TelemetryHUD
from tnfr_lfs.exporters.setup_plan import SetupChange, SetupPlan
from tnfr_lfs.core.metrics import (
    AeroAxisCoherence,
    AeroBandCoherence,
    AeroCoherence,
    BrakeHeadroom,
    SlideCatchBudget,
    WindowMetrics,
)


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
    assert "ΔNFR" in page_a and "∇Acop" in page_a
    assert "C(t)" in page_a
    assert "[" in page_a  # ΔNFR gauge present
    if "Sin microsector activo" not in page_a:
        assert any(char in page_a for char in "▁▂▃▄▅▆▇█")
        assert "ν_f~" in page_a or "ν_f~" in page_b
        assert "Si↺" in page_a or "Si plan" in page_c
    assert "Líder" in page_b and "ν_f" in page_b
    assert "C(t)" in page_b
    assert "dSi" in page_c
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
        clamped_parameters=(),
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
        clamped_parameters=(),
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


def test_render_page_c_marks_riesgos(synthetic_records):
    hud = _populate_hud(synthetic_records[:60])
    thresholds = hud._thresholds
    plan = SetupPlan(
        car_model="generic_gt",
        session=None,
        changes=(
            SetupChange(
                parameter="front_arb_steps",
                delta=1.0,
                rationale="",
                expected_effect="Mejor apoyo",
            ),
        ),
        rationales=(),
        expected_effects=(),
        sensitivities={"sense_index": {"front_arb_steps": 0.12}},
        clamped_parameters=("front_arb_steps",),
    )
    output = osd_module._render_page_c(None, plan, thresholds, None)
    assert "riesgos" in output
    assert "front_arb_steps" in output
    assert "dSi" in output


def test_render_page_c_includes_aero_guidance(synthetic_records):
    hud = _populate_hud(synthetic_records[:60])
    thresholds = hud._thresholds
    plan = SetupPlan(
        car_model="generic_gt",
        session=None,
        changes=(),
        rationales=(),
        expected_effects=(),
        sensitivities={},
        aero_guidance="Alta velocidad → sube alerón trasero",
        aero_metrics={
            "low_speed_imbalance": 0.02,
            "high_speed_imbalance": 0.3,
            "aero_mechanical_coherence": 0.68,
        },
    )
    output = osd_module._render_page_c(None, plan, thresholds, None)
    assert "Aero Alta velocidad" in output
    assert "Δaero alta" in output
    assert "C(a/m) 0.68" in output


def test_render_page_a_includes_wave_when_active_phase():
    from types import SimpleNamespace

    bundles = [
        SimpleNamespace(delta_nfr=value, sense_index=0.5 + idx * 0.05)
        for idx, value in enumerate((-0.3, -0.1, 0.1, 0.3, 0.6))
    ]
    phase_samples = {
        phase: (idx,)
        for idx, phase in enumerate(osd_module.PHASE_SEQUENCE)
    }
    microsector = SimpleNamespace(index=0, phase_samples=phase_samples)
    goal = SimpleNamespace(target_delta_nfr=0.4, target_sense_index=0.8)
    active = osd_module.ActivePhase(microsector=microsector, phase="apex", goal=goal)
    aero = AeroCoherence(
        low_speed=AeroBandCoherence(
            total=AeroAxisCoherence(0.12, 0.18),
            samples=6,
        ),
        high_speed=AeroBandCoherence(
            total=AeroAxisCoherence(0.22, 0.08),
            samples=9,
        ),
    )
    window_metrics = WindowMetrics(
        si=0.7,
        si_variance=0.0012,
        d_nfr_couple=0.1,
        d_nfr_res=-0.2,
        d_nfr_flat=0.05,
        nu_f=1.2,
        nu_exc=0.9,
        rho=0.75,
        phase_lag=0.0,
        phase_alignment=1.0,
        useful_dissonance_ratio=0.6,
        useful_dissonance_percentage=60.0,
        coherence_index=0.4,
        ackermann_parallel_index=0.0,
        slide_catch_budget=SlideCatchBudget(),
        support_effective=0.18,
        load_support_ratio=0.000036,
        structural_expansion_longitudinal=0.12,
        structural_contraction_longitudinal=0.08,
        structural_expansion_lateral=0.14,
        structural_contraction_lateral=0.02,
        bottoming_ratio_front=0.0,
        bottoming_ratio_rear=0.0,
        mu_usage_front_ratio=0.0,
        mu_usage_rear_ratio=0.0,
        phase_mu_usage_front_ratio=0.0,
        phase_mu_usage_rear_ratio=0.0,
        exit_gear_match=0.85,
        shift_stability=0.9,
        frequency_label="ν_f óptima 1.95Hz (obj 1.90-2.20Hz)",
        aero_coherence=aero,
        aero_mechanical_coherence=0.64,
        epi_derivative_abs=0.12,
        brake_headroom=BrakeHeadroom(),
    )
    output = osd_module._render_page_a(active, bundles[-1], 0.2, window_metrics, bundles)
    curve_label = f"Curva {microsector.index + 1}"
    assert curve_label in output
    assert osd_module.HUD_PHASE_LABELS.get(active.phase, active.phase.capitalize()) in output
    assert "ν_f~" in output
    assert "[" in output  # ΔNFR gauge
    assert "C(t) 0.40" in output
    assert "∇Acop" in output
    aero_line = osd_module._truncate_line(
        "Δaero alta "
        f"{window_metrics.aero_coherence.high_speed_imbalance:+.2f}"
        f" · baja {window_metrics.aero_coherence.low_speed_imbalance:+.2f}"
    )
    if len((output + "\n" + aero_line).encode("utf8")) <= osd_module.PAYLOAD_LIMIT:
        assert "Δaero" in output
    amc_line = osd_module._truncate_line(
        f"C(a/m) {window_metrics.aero_mechanical_coherence:.2f}"
    )
    if len((output + "\n" + amc_line).encode("utf8")) <= osd_module.PAYLOAD_LIMIT:
        assert "C(a/m)" in output


def test_render_page_a_displays_brake_meter_on_severe_events():
    bundles = [
        SimpleNamespace(
            delta_nfr=-0.2,
            delta_nfr_longitudinal=-0.18,
            delta_nfr_lateral=-0.04,
            sense_index=0.78,
        )
    ]
    phase_samples = {
        phase: (0,)
        for phase in osd_module.PHASE_SEQUENCE
    }
    operator_events = {
        "OZ": (
            {
                "delta_nfr_threshold": 0.28,
                "delta_nfr_peak": 0.35,
                "delta_nfr_avg": 0.3,
                "delta_nfr_ratio": 1.25,
                "surface_label": "low_grip",
            },
        )
    }
    microsector = SimpleNamespace(
        index=3,
        phase_samples=phase_samples,
        operator_events=operator_events,
    )
    goal = SimpleNamespace(
        target_delta_nfr=0.3,
        target_sense_index=0.82,
        target_delta_nfr_long=0.1,
        target_delta_nfr_lat=0.05,
        delta_axis_weights={"longitudinal": 0.6, "lateral": 0.4},
    )
    active = osd_module.ActivePhase(microsector=microsector, phase="entry1", goal=goal)
    aero = AeroCoherence(
        low_speed=AeroBandCoherence(
            total=AeroAxisCoherence(0.12, 0.18),
            samples=6,
        ),
        high_speed=AeroBandCoherence(
            total=AeroAxisCoherence(0.22, 0.08),
            samples=9,
        ),
    )
    window_metrics = WindowMetrics(
        si=0.72,
        si_variance=0.0008,
        d_nfr_couple=0.18,
        d_nfr_res=-0.12,
        d_nfr_flat=0.05,
        nu_f=1.3,
        nu_exc=0.95,
        rho=0.78,
        phase_lag=0.05,
        phase_alignment=0.9,
        useful_dissonance_ratio=0.64,
        useful_dissonance_percentage=58.0,
        coherence_index=0.45,
        ackermann_parallel_index=0.0,
        slide_catch_budget=SlideCatchBudget(),
        support_effective=0.2,
        load_support_ratio=0.000038,
        structural_expansion_longitudinal=0.16,
        structural_contraction_longitudinal=0.04,
        structural_expansion_lateral=0.1,
        structural_contraction_lateral=0.06,
        bottoming_ratio_front=0.0,
        bottoming_ratio_rear=0.0,
        mu_usage_front_ratio=0.0,
        mu_usage_rear_ratio=0.0,
        phase_mu_usage_front_ratio=0.0,
        phase_mu_usage_rear_ratio=0.0,
        exit_gear_match=0.78,
        shift_stability=0.88,
        frequency_label="ν_f óptima 1.95Hz (obj 1.90-2.20Hz)",
        aero_coherence=aero,
        aero_mechanical_coherence=0.58,
        epi_derivative_abs=0.09,
        brake_headroom=BrakeHeadroom(),
    )
    page = osd_module._render_page_a(active, bundles[0], 0.2, window_metrics, bundles)
    assert "ΔNFR frenada" in page
    assert "low_grip" in page
    assert "OZ" in page


def test_render_page_a_includes_no_tocar_notice():
    from types import SimpleNamespace

    bundles = [SimpleNamespace(delta_nfr=0.2, sense_index=0.8)] * 5
    phase_samples = {phase: (0,) for phase in osd_module.PHASE_SEQUENCE}
    microsector = SimpleNamespace(
        index=0,
        phase_samples=phase_samples,
        filtered_measures={
            "si_variance": 0.0004,
            "epi_derivative_abs": 0.05,
        },
        operator_events={
            "SILENCIO": (
                {
                    "duration": 0.8,
                    "slack": 0.5,
                    "structural_density_mean": 0.04,
                },
            )
        },
        start_time=0.0,
        end_time=1.0,
    )
    goal = SimpleNamespace(
        target_delta_nfr=0.2,
        target_sense_index=0.8,
        target_delta_nfr_long=0.05,
        target_delta_nfr_lat=0.03,
        delta_axis_weights={"longitudinal": 0.6, "lateral": 0.4},
    )
    active = osd_module.ActivePhase(microsector=microsector, phase="entry1", goal=goal)
    window_metrics = WindowMetrics(
        si=0.8,
        si_variance=0.0004,
        d_nfr_couple=0.1,
        d_nfr_res=0.05,
        d_nfr_flat=0.02,
        nu_f=1.2,
        nu_exc=1.1,
        rho=0.9,
        phase_lag=0.0,
        phase_alignment=0.95,
        useful_dissonance_ratio=0.2,
        useful_dissonance_percentage=20.0,
        coherence_index=0.5,
        ackermann_parallel_index=0.0,
        slide_catch_budget=SlideCatchBudget(),
        support_effective=0.15,
        load_support_ratio=0.00003,
        structural_expansion_longitudinal=0.12,
        structural_contraction_longitudinal=0.04,
        structural_expansion_lateral=0.1,
        structural_contraction_lateral=0.02,
        bottoming_ratio_front=0.0,
        bottoming_ratio_rear=0.0,
        mu_usage_front_ratio=0.0,
        mu_usage_rear_ratio=0.0,
        phase_mu_usage_front_ratio=0.0,
        phase_mu_usage_rear_ratio=0.0,
        exit_gear_match=0.9,
        shift_stability=1.0,
        frequency_label="",
        aero_coherence=AeroCoherence(),
        aero_mechanical_coherence=0.4,
        epi_derivative_abs=0.05,
        brake_headroom=BrakeHeadroom(),
    )
    page = osd_module._render_page_a(
        active,
        bundles[-1],
        0.2,
        window_metrics,
        bundles,
        quiet_sequences=((0, 1, 2),),
    )
    assert "no tocar" in page.lower()


def test_silence_event_meter_renders_when_present() -> None:
    operator_events = {
        "SILENCIO": (
            {
                "duration": 1.8,
                "structural_density_mean": 0.03,
                "load_span": 120.0,
                "slack": 0.4,
            },
        )
    }
    microsector = SimpleNamespace(
        index=1,
        operator_events=operator_events,
        start_time=0.0,
        end_time=2.4,
    )
    line = osd_module._silence_event_meter(microsector)
    assert line is not None
    assert "Silencio" in line


def test_brake_meter_skips_when_silence_dominates() -> None:
    operator_events = {
        "SILENCIO": ({"duration": 2.0},),
        "OZ": (
            {
                "delta_nfr_threshold": 0.28,
                "delta_nfr_peak": 0.45,
                "delta_nfr_ratio": 1.2,
                "surface_label": "test_surface",
            },
        ),
    }
    microsector = SimpleNamespace(
        operator_events=operator_events,
        start_time=0.0,
        end_time=2.0,
    )
    assert osd_module._brake_event_meter(microsector) is None
