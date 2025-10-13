"""Tests for the live HUD OSD helper."""

from __future__ import annotations

import math
import pytest
from types import SimpleNamespace

from tnfr_lfs.ingestion.live import ButtonEvent, ButtonLayout, MacroQueue, OverlayManager
from tnfr_lfs.cli import osd as osd_module
from tnfr_lfs.cli import app as cli_app
from tnfr_lfs.cli.osd import HUDPager, MacroStatus, OSDController, TelemetryHUD
from tnfr_lfs.exporters.setup_plan import SetupChange, SetupPlan
from tnfr_lfs.core.metrics import (
    AeroAxisCoherence,
    AeroBalanceDrift,
    AeroBalanceDriftBin,
    AeroBandCoherence,
    AeroCoherence,
    BrakeHeadroom,
    LockingWindowScore,
    SlideCatchBudget,
    WindowMetrics,
)
from tnfr_lfs.core.operator_detection import canonical_operator_label
from tnfr_lfs.recommender.rules import RuleProfileObjectives
from tnfr_lfs.core.epi import TelemetryRecord
from tests.helpers import (
    DummyHUD,
    _populate_hud,
    _window_metrics_from_parallel_turn,
    build_minimal_setup_plan,
    build_parallel_window_metrics,
)


def test_osd_pages_fit_within_button_limit(synthetic_records):
    hud = _populate_hud(synthetic_records[:120])
    pages = hud.pages()
    assert len(pages) == 4
    for page in pages:
        assert len(page.encode("utf8")) <= OverlayManager.MAX_BUTTON_TEXT - 1

    page_a, page_b, page_c, page_d = pages
    assert "ΔNFR" in page_a
    assert "C(t)" in page_a
    assert "[" in page_a  # ΔNFR gauge present
    assert "ENTRY Δ" in page_a
    if "No active microsector" not in page_a:
        assert any(char in page_a for char in "▁▂▃▄▅▆▇█")
        assert "ν_f~" in page_a or "ν_f~" in page_b
        assert "Si↺" in page_a or "Si plan" in page_c
    assert "Leader" in page_b and "ν_f" in page_b
    assert "C(t)" in page_b
    assert "ΔNFR phase map" in page_c
    assert "Apply" in page_d


def test_entropy_indicator_uses_thresholds() -> None:
    metrics = SimpleNamespace(delta_nfr_entropy=0.72, node_entropy=0.48)
    thresholds = {
        "delta_entropy_green": 0.70,
        "delta_entropy_amber": 0.50,
        "node_entropy_green": 0.60,
        "node_entropy_amber": 0.45,
    }
    line = osd_module._entropy_indicator_line(metrics, thresholds)
    assert line is not None
    assert "🟠" in line
    assert "0.72" in line and "0.48" in line


def test_lap_integral_series_and_cov_indicator() -> None:
    records = []
    bundles = []
    timestamp = 0.0
    for lap in (0, 0, 1, 1, 2, 2, None):
        records.append(
            SimpleNamespace(timestamp=timestamp, structural_timestamp=timestamp, lap=lap)
        )
        magnitude = 0.0 if lap is None else 1.0 + (lap or 0.0)
        bundles.append(SimpleNamespace(delta_nfr=magnitude))
        timestamp += 1.0

    values = osd_module._lap_integral_series(records, bundles)
    assert values == pytest.approx((2.0, 4.0, 6.0))
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    cov = math.sqrt(variance) / mean_value
    thresholds = {"integral_cov_green": 0.25, "integral_cov_amber": 0.45}
    line = osd_module._integral_cov_line(cov, len(values), thresholds)
    assert line is not None
    assert "🟠" in line
    assert f"n {len(values)}" in line


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
        parameter="Front pressure",
        delta=1.5,
        rationale="",
        expected_effect="",
    )
    status = MacroStatus(
        next_change=change,
        warnings=("Open the pit menu (F12)", "Stop the car before applying"),
        queue_size=2,
    )
    page_d = osd_module._render_page_d((change,), status)
    assert "Macro queue 2" in page_d
    assert page_d.count("⚠️") == 2


def test_osd_controller_applies_macro_on_trigger():
    plan = build_minimal_setup_plan(
        car_model="FZR",
        changes=[
            {"parameter": "Wing", "delta": 2.0},
            {"parameter": "Damper", "delta": -1.0},
        ],
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
    plan = build_minimal_setup_plan(
        car_model="FZR",
        changes=[{"parameter": "Wing", "delta": 2.0}],
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


def _empty_render_kwargs() -> dict[str, object]:
    return {}


def _operational_checklist_kwargs() -> dict[str, object]:
    window_metrics = SimpleNamespace(
        brake_headroom=SimpleNamespace(value=0.35),
        aero_coherence=SimpleNamespace(high_speed_imbalance=0.18),
    )
    bundles = [
        SimpleNamespace(delta_nfr=3.0, sense_index=0.72, timestamp=0.0),
        SimpleNamespace(delta_nfr=2.5, sense_index=0.82, timestamp=1.0),
    ]
    objectives = RuleProfileObjectives(
        target_delta_nfr=0.0,
        target_sense_index=0.75,
        target_brake_headroom=0.4,
    )
    session_hints = {"delta_reference": 6.0, "aero_reference": 0.12}
    return {
        "sense_state": {"average": 0.78},
        "window_metrics": window_metrics,
        "objectives": objectives,
        "session_hints": session_hints,
        "bundles": bundles,
    }


_RENDER_PAGE_C_CASES = [
    {
        "case_id": "marks-risks",
        "plan_kwargs": {
            "car_model": "FZR",
            "sci": 0.842,
            "changes": [
                {
                    "parameter": "front_arb_steps",
                    "delta": 1.0,
                    "expected_effect": "Better support",
                }
            ],
            "sensitivities": {"sense_index": {"front_arb_steps": 0.12}},
            "clamped_parameters": ("front_arb_steps",),
            "sci_breakdown": {
                "sense": 0.512,
                "delta": 0.198,
                "udr": 0.085,
                "bottoming": 0.025,
                "aero": 0.012,
            },
        },
        "expected_tokens": ("risks", "front_arb_steps", "dSi", "SCI 0.842", "sense 0.512"),
        "render_kwargs_factory": _empty_render_kwargs,
        "min_warning_count": 0,
    },
    {
        "case_id": "aero-guidance",
        "plan_kwargs": {
            "car_model": "FZR",
            "sci": 0.731,
            "aero_guidance": "High speed → add rear wing",
            "aero_metrics": {
                "low_speed_imbalance": 0.02,
                "high_speed_imbalance": 0.3,
                "aero_mechanical_coherence": 0.68,
            },
        },
        "expected_tokens": (
            "Aero High speed",
            "Δaero high",
            "C(c/d/a) 0.68",
        ),
        "render_kwargs_factory": _empty_render_kwargs,
        "min_warning_count": 0,
    },
    {
        "case_id": "phase-axis-summary",
        "plan_kwargs": {
            "car_model": "FZR",
            "sci": 0.731,
            "phase_axis_targets": {
                "entry": {"longitudinal": 0.4, "lateral": 0.1},
                "apex": {"longitudinal": 0.05, "lateral": 0.3},
            },
            "phase_axis_weights": {
                "entry": {"longitudinal": 0.7, "lateral": 0.3},
                "apex": {"longitudinal": 0.2, "lateral": 0.8},
            },
        },
        "expected_tokens": ("ΔNFR phase map", "⇈+0.40", "Entry ∥ ⇈+0.40"),
        "render_kwargs_factory": _empty_render_kwargs,
        "min_warning_count": 0,
    },
    {
        "case_id": "operational-checklist",
        "plan_kwargs": {"car_model": "FZR"},
        "expected_tokens": (
            "Checklist",
            "Si.78≥.75",
            "Δ∫5.5≤6",
            "Hd.35≥.4",
            "Δμ.18≤.12",
        ),
        "render_kwargs_factory": _operational_checklist_kwargs,
        "min_warning_count": 1,
    },
]


@pytest.mark.parametrize(
    "case",
    _RENDER_PAGE_C_CASES,
    ids=[case["case_id"] for case in _RENDER_PAGE_C_CASES],
)
def test_render_page_c_renders_expected_sections(osd_hud, case):
    hud = osd_hud(stop=60)
    thresholds = hud._thresholds
    plan = build_minimal_setup_plan(**case["plan_kwargs"])
    render_kwargs = case["render_kwargs_factory"]()
    output = osd_module._render_page_c(None, plan, thresholds, None, **render_kwargs)
    for token in case["expected_tokens"]:
        assert token in output, f"{case['case_id']} missing {token}"
    if case.get("min_warning_count", 0):
        assert output.count("⚠️") >= case["min_warning_count"]


def test_build_setup_plan_includes_phase_axis_summary() -> None:
    base_plan = SimpleNamespace(
        recommendations=[],
        decision_vector={},
        sensitivities={},
        phase_sensitivities={},
        sci=0.0,
        sci_breakdown={},
        objective_breakdown={},
    )
    microsectors = [
        SimpleNamespace(
            goals=(
                SimpleNamespace(
                    phase="entry1",
                    target_delta_nfr_long=0.4,
                    target_delta_nfr_lat=0.1,
                    delta_axis_weights={"longitudinal": 0.7, "lateral": 0.3},
                ),
                SimpleNamespace(
                    phase="apex3a",
                    target_delta_nfr_long=0.05,
                    target_delta_nfr_lat=0.3,
                    delta_axis_weights={"longitudinal": 0.2, "lateral": 0.8},
                ),
            ),
        )
    ]
    plan = osd_module._build_setup_plan(base_plan, "FZR", None, microsectors)
    assert plan.phase_axis_summary["longitudinal"]["entry"] == "⇈+0.40"
    assert any("Entry ∥" in hint for hint in plan.phase_axis_suggestions)


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
    sc_metrics = _window_metrics_from_parallel_turn(
        ((0.082, 0.012), (0.074, 0.018), (0.068, 0.022))
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
        phase_synchrony_index=1.0,
        motor_latency_ms=0.0,
        phase_motor_latency_ms={},
        useful_dissonance_ratio=0.6,
        useful_dissonance_percentage=60.0,
        coherence_index=0.4,
        ackermann_parallel_index=sc_metrics.ackermann_parallel_index,
        slide_catch_budget=sc_metrics.slide_catch_budget,
        locking_window_score=LockingWindowScore(),
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
        mu_balance=0.01,
        mu_symmetry={"window": {"front": 0.02, "rear": -0.03}},
        exit_gear_match=0.85,
        shift_stability=0.9,
        frequency_label="ν_f optimal 1.95Hz (obj 1.90-2.20Hz)",
        aero_coherence=aero,
        aero_mechanical_coherence=0.64,
        epi_derivative_abs=0.12,
        brake_headroom=BrakeHeadroom(),
    )
    gradient_line = osd_module._gradient_line(window_metrics)
    assert "Φsync" in gradient_line
    assert "μβ" in gradient_line
    assert "μΦF" in gradient_line
    assert "μΦR" in gradient_line
    assert "τmot" in gradient_line
    output = osd_module._render_page_a(active, bundles[-1], 0.2, window_metrics, bundles)
    curve_label = f"Corner {microsector.index + 1}"
    assert curve_label in output
    assert osd_module.HUD_PHASE_LABELS.get(active.phase, active.phase.capitalize()) in output
    assert "ν_f" in output
    assert "[" in output  # ΔNFR gauge
    assert "C(t) 0.40" in output
    assert "∇Coupl" in output
    aero_line = osd_module._truncate_line(
        "Δaero high "
        f"{window_metrics.aero_coherence.high_speed_imbalance:+.2f}"
        f" · low {window_metrics.aero_coherence.low_speed_imbalance:+.2f}"
    )
    if len((output + "\n" + aero_line).encode("utf8")) <= osd_module.PAYLOAD_LIMIT:
        assert "Δaero" in output
    amc_line = osd_module._truncate_line(
        f"C(c/d/a) {window_metrics.aero_mechanical_coherence:.2f}"
    )
    if len((output + "\n" + amc_line).encode("utf8")) <= osd_module.PAYLOAD_LIMIT:
        assert "C(c/d/a)" in output


def test_aero_drift_line_prefers_balance_segments() -> None:
    drift = AeroBalanceDrift(
        mu_tolerance=0.03,
        high_speed=AeroBalanceDriftBin(
            samples=12,
            mu_delta=0.05,
            mu_ratio=1.08,
            mu_balance=0.09,
            mu_symmetry_front=0.02,
            mu_symmetry_rear=-0.07,
            rake_mean=math.radians(1.5),
        ),
    )
    line = osd_module._aero_drift_line(drift)
    assert line is not None
    assert "μβ⚠️" in line
    assert "μΦR" in line
    assert "μΔ" not in line


def test_aero_drift_line_falls_back_to_mu_delta() -> None:
    drift = AeroBalanceDrift(
        mu_tolerance=0.02,
        high_speed=AeroBalanceDriftBin(
            samples=10,
            mu_delta=-0.06,
            mu_ratio=0.94,
            mu_balance=0.0,
            mu_symmetry_front=0.0,
            mu_symmetry_rear=0.0,
            rake_mean=math.radians(-1.2),
        ),
    )
    line = osd_module._aero_drift_line(drift)
    assert line is not None
    assert "μΔ" in line
    assert "μƒ/μr" in line
    assert "μβ" not in line


def test_render_page_a_displays_brake_meter_on_severe_events():
    bundles = [
        SimpleNamespace(
            delta_nfr=-0.2,
            delta_nfr_proj_longitudinal=-0.18,
            delta_nfr_proj_lateral=-0.04,
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
                "name": canonical_operator_label("OZ"),
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
    sc_metrics = _window_metrics_from_parallel_turn(
        ((0.09, 0.008), (0.084, 0.012), (0.078, 0.018))
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
        phase_synchrony_index=0.92,
        motor_latency_ms=0.0,
        phase_motor_latency_ms={},
        useful_dissonance_ratio=0.64,
        useful_dissonance_percentage=58.0,
        coherence_index=0.45,
        ackermann_parallel_index=sc_metrics.ackermann_parallel_index,
        slide_catch_budget=sc_metrics.slide_catch_budget,
        locking_window_score=LockingWindowScore(),
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
        mu_balance=-0.02,
        mu_symmetry={"window": {"front": -0.01, "rear": 0.03}},
        exit_gear_match=0.78,
        shift_stability=0.88,
        frequency_label="ν_f optimal 1.95Hz (obj 1.90-2.20Hz)",
        aero_coherence=aero,
        aero_mechanical_coherence=0.58,
        epi_derivative_abs=0.09,
        brake_headroom=BrakeHeadroom(),
    )
    page = osd_module._render_page_a(active, bundles[0], 0.2, window_metrics, bundles)
    assert "ΔNFR braking" in page
    assert "low_grip" in page
    assert canonical_operator_label("OZ") in page


def test_brake_headroom_line_renders_summary() -> None:
    headroom = BrakeHeadroom(
        value=0.35,
        fade_ratio=0.18,
        fade_slope=0.55,
        temperature_peak=660.0,
        temperature_mean=640.0,
        ventilation_alert="attention",
        ventilation_index=0.6,
    )
    line = osd_module._brake_headroom_line(headroom)
    assert line is not None
    assert line.startswith("Brake")
    assert "HR 0.35" in line
    assert "fade" in line
    assert "vent attention" in line


def test_brake_headroom_line_with_missing_temperature_data() -> None:
    headroom = BrakeHeadroom(
        value=0.42,
        fade_ratio=math.nan,
        fade_slope=math.nan,
        temperature_peak=math.nan,
        temperature_mean=math.nan,
        ventilation_index=math.nan,
        temperature_available=False,
        fade_available=False,
    )
    line = osd_module._brake_headroom_line(headroom)
    assert line is not None
    assert "no data" in line


def test_thermal_dispersion_lines_render_fallback_without_telemetry() -> None:
    microsector = SimpleNamespace(filtered_measures={})
    lines = osd_module._thermal_dispersion_lines(microsector)
    assert "T° no data" in lines
    assert any(
        line.startswith("Pbar") and "no data" in line for line in lines
    )


def test_thermal_dispersion_lines_render_values_when_available() -> None:
    measures = {
        "tyre_temp_fl": 88.2,
        "tyre_temp_fl_std": 1.5,
        "tyre_temp_fr": 89.0,
        "tyre_temp_fr_std": 1.2,
        "tyre_temp_rl": 86.3,
        "tyre_temp_rl_std": 1.1,
        "tyre_temp_rr": 87.1,
        "tyre_temp_rr_std": 1.4,
        "brake_temp_fl": 420.0,
        "brake_temp_fl_std": 5.0,
        "brake_temp_fr": 415.0,
        "brake_temp_fr_std": 4.5,
        "brake_temp_rl": 400.0,
        "brake_temp_rl_std": 4.0,
        "brake_temp_rr": 405.0,
        "brake_temp_rr_std": 4.2,
        "tyre_pressure_fl": 1.82,
        "tyre_pressure_fl_std": 0.012,
        "tyre_pressure_fr": 1.83,
        "tyre_pressure_fr_std": 0.011,
        "tyre_pressure_rl": 1.80,
        "tyre_pressure_rl_std": 0.010,
        "tyre_pressure_rr": 1.81,
        "tyre_pressure_rr_std": 0.009,
    }
    microsector = SimpleNamespace(filtered_measures=measures)
    lines = osd_module._thermal_dispersion_lines(microsector)
    assert lines
    assert all("no data" not in line for line in lines)
    assert any("T° FL" in line for line in lines)


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
            "SILENCE": (
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
    sc_metrics = _window_metrics_from_parallel_turn(
        ((0.07, 0.02), (0.062, 0.024), (0.056, 0.03))
    )
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
        phase_synchrony_index=0.97,
        motor_latency_ms=0.0,
        phase_motor_latency_ms={},
        useful_dissonance_ratio=0.2,
        useful_dissonance_percentage=20.0,
        coherence_index=0.5,
        ackermann_parallel_index=sc_metrics.ackermann_parallel_index,
        slide_catch_budget=sc_metrics.slide_catch_budget,
        locking_window_score=LockingWindowScore(),
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
        mu_balance=0.0,
        mu_symmetry={"window": {"front": 0.0, "rear": 0.0}},
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
    assert "leave untouched" in page.lower()


def test_silence_event_meter_renders_when_present() -> None:
    operator_events = {
        "SILENCE": (
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
    assert "Silence" in line


def test_brake_meter_skips_when_silence_dominates() -> None:
    operator_events = {
        "SILENCE": ({"duration": 2.0},),
        "OZ": (
            {
                "name": canonical_operator_label("OZ"),
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


def test_generate_out_reports_includes_phase_aliases(
    tmp_path,
    synthetic_records,
    synthetic_bundles,
    synthetic_microsectors,
) -> None:
    reports = cli_app._generate_out_reports(
        synthetic_records,
        synthetic_bundles,
        synthetic_microsectors,
        tmp_path,
        metrics={},
    )
    samples_payload = reports["phase_samples"]["data"]
    assert samples_payload
    phase_samples = samples_payload[0]["phase_samples"]
    assert "entry1" in phase_samples
    assert "entry" in phase_samples
    axis_targets_payload = reports["phase_axis_targets"]["data"]
    assert axis_targets_payload
    axis_targets = axis_targets_payload[0]["phase_axis_targets"]
    assert "entry1" in axis_targets
    assert "entry" in axis_targets
    metrics_payload = reports["phase_metrics"]["data"]
    assert metrics_payload
    phase_metrics = metrics_payload[0]["phase_delta_nfr_std"]
    assert "entry1" in phase_metrics
    assert "entry" in phase_metrics
    phase_entropy = metrics_payload[0]["phase_delta_nfr_entropy"]
    assert "entry1" in phase_entropy
    assert "entry" in phase_entropy
    phase_node_entropy = metrics_payload[0]["phase_node_entropy"]
    assert "entry1" in phase_node_entropy
    assert "entry" in phase_node_entropy


def test_sense_index_map_filters_sample_metrics() -> None:
    class DummyMicrosector:
        def __init__(self) -> None:
            self.index = 0
            self.phase_samples = {"entry": (0,), "apex": (), "exit": ()}
            self.filtered_measures = {
                "mu_balance": 0.12,
                "mu_balance_samples": 6,
                "mu_symmetry_front": 0.04,
                "mu_symmetry_rear": -0.03,
                "tyre_temp_fl": 88.0,
            }
            self.window_occupancy = {
                "entry": {"delta": 0.4},
                "apex": {},
                "exit": {},
            }
            self.last_mutation = None
            self.grip_rel = 0.98

        def phase_indices(self, phase: str):
            return self.phase_samples.get(phase, ())

    bundles = [SimpleNamespace(sense_index=0.76)]
    microsector = DummyMicrosector()
    results = cli_app._sense_index_map(bundles, [microsector])
    assert results
    entry = results[0]
    filtered = entry["filtered_measures"]
    assert "mu_balance_samples" not in filtered
    assert filtered["mu_balance"] == pytest.approx(0.12, rel=1e-6)
    samples = entry.get("sample_measures")
    assert samples is not None
    assert samples["mu_balance_samples"] == pytest.approx(6.0)
