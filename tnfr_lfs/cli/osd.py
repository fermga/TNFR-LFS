"""Runtime HUD controller for the on-screen display (OSD)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Deque, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from ..acquisition import (
    ButtonEvent,
    ButtonLayout,
    InSimClient,
    MacroQueue,
    OverlayManager,
    OutGaugeUDPClient,
    OutSimUDPClient,
    TelemetryFusion,
)
from ..core.epi import EPIExtractor, TelemetryRecord
from ..core.metrics import WindowMetrics, compute_window_metrics
from ..core.operators import orchestrate_delta_metrics
from ..core.phases import PHASE_SEQUENCE, phase_family
from ..core.resonance import ModalAnalysis, ModalPeak, analyse_modal_resonance
from ..core.segmentation import Goal, Microsector, segment_microsectors
from ..exporters.setup_plan import SetupChange, SetupPlan
from ..recommender import RecommendationEngine, SetupPlanner
from ..recommender.rules import NODE_LABELS, ThresholdProfile

PAYLOAD_LIMIT = OverlayManager.MAX_BUTTON_TEXT - 1
DEFAULT_UPDATE_RATE = 6.0
DEFAULT_PLAN_INTERVAL = 5.0

NODE_AXIS_LABELS = {
    "yaw": "guiñada",
    "roll": "balanceo",
    "pitch": "cabeceo",
}

MODAL_AXIS_SUMMARY_LABELS = {
    "yaw": "guiñada",
    "roll": "balanceo",
    "pitch": "suspensión",
}

SPARKLINE_BLOCKS = "▁▂▃▄▅▆▇█"

HUD_PHASE_LABELS = {
    "entry1": "Entrada 1",
    "entry2": "Entrada 2",
    "apex3a": "Vértice 3A",
    "apex3b": "Vértice 3B",
    "exit4": "Salida 4",
}


@dataclass(frozen=True)
class ActivePhase:
    """Resolved microsector/phase context for the current record."""

    microsector: Microsector
    phase: str
    goal: Goal | None


@dataclass(frozen=True)
class MacroStatus:
    """State exposed in the “Aplicar” HUD page."""

    next_change: SetupChange | None = None
    warnings: Tuple[str, ...] = ()
    queue_size: int = 0


class HUDPager:
    """Track the active HUD page and react to button clicks."""

    def __init__(self, pages: Sequence[str] | None = None) -> None:
        self._pages: Tuple[str, ...] = tuple(pages or ())
        if not self._pages:
            self._pages = (
                "Esperando telemetría…",
                "ΔNFR nodal en espera",
                "Plan en preparación…",
                "Aplicar en espera…",
            )
        self._index = 0

    @property
    def pages(self) -> Tuple[str, ...]:
        return self._pages

    def update(self, pages: Sequence[str]) -> None:
        if not pages:
            return
        self._pages = tuple(pages)
        if self._index >= len(self._pages):
            self._index = 0

    def current(self) -> str:
        return self._pages[self._index]

    @property
    def index(self) -> int:
        return self._index

    def advance(self) -> None:
        if not self._pages:
            return
        self._index = (self._index + 1) % len(self._pages)

    def handle_event(self, event: ButtonEvent | None, layout: ButtonLayout) -> bool:
        if event is None:
            return False
        if event.type_in != 0:
            return False
        if event.click_id != layout.click_id:
            return False
        self.advance()
        return True


class TelemetryHUD:
    """Maintain rolling telemetry and derive HUD snapshots."""

    def __init__(
        self,
        *,
        window: int = 256,
        car_model: str = "XFG",
        track_name: str = "generic",
        extractor: EPIExtractor | None = None,
        recommendation_engine: RecommendationEngine | None = None,
        setup_planner: SetupPlanner | None = None,
        plan_interval: float = DEFAULT_PLAN_INTERVAL,
        time_fn=monotonic,
    ) -> None:
        self._records: Deque[TelemetryRecord] = deque(maxlen=max(8, int(window)))
        self.extractor = extractor or EPIExtractor()
        self.car_model = car_model
        self.track_name = track_name
        self.recommendation_engine = recommendation_engine or RecommendationEngine(
            car_model=car_model, track_name=track_name
        )
        self.setup_planner = setup_planner or SetupPlanner(
            recommendation_engine=self.recommendation_engine
        )
        self._operator_state: MutableMapping[str, Dict[str, Dict[str, object]]] = {}
        self._microsectors: Sequence[Microsector] = ()
        self._bundles: Sequence[object] = ()
        self._pages: Tuple[str, str, str, str] = (
            "Esperando telemetría…",
            "ΔNFR nodal en espera",
            "Plan en preparación…",
            "Aplicar en espera…",
        )
        self._dirty = True
        self._plan_interval = max(0.0, float(plan_interval))
        self._time_fn = time_fn
        self._last_plan_time = -math.inf
        self._cached_plan: SetupPlan | None = None
        self._macro_status = MacroStatus()
        self._thresholds: ThresholdProfile = (
            self.recommendation_engine._resolve_context(  # type: ignore[attr-defined]
                car_model, track_name
            ).thresholds
        )

    def append(self, record: TelemetryRecord) -> None:
        self._records.append(record)
        self._dirty = True

    def pages(self) -> Tuple[str, str, str, str]:
        if self._dirty:
            self._recompute()
        return self._pages

    def plan(self) -> SetupPlan | None:
        if self._dirty:
            self._recompute()
        return self._cached_plan

    def update_macro_status(self, status: "MacroStatus") -> None:
        if status == self._macro_status:
            return
        self._macro_status = status
        self._dirty = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _recompute(self) -> None:
        records = list(self._records)
        if len(records) < 8:
            self._pages = (
                "Esperando telemetría…",
                "ΔNFR nodal en espera",
                "Plan en preparación…",
                "Aplicar en espera…",
            )
            self._dirty = False
            return

        bundles = self.extractor.extract(records)
        if not bundles:
            self._dirty = False
            return

        self._bundles = bundles
        self._microsectors = segment_microsectors(
            records,
            bundles,
            operator_state=self._operator_state,
            phase_weight_overrides=self._thresholds.phase_weights
            if self._thresholds.phase_weights
            else None,
        )
        active = _resolve_active_phase(self._microsectors, len(records) - 1)

        bundle = bundles[-1]
        goal_delta = active.goal.target_delta_nfr if active and active.goal else 0.0
        goal_si = active.goal.target_sense_index if active and active.goal else 0.75
        orchestrate_delta_metrics(
            [records],
            goal_delta,
            goal_si,
            microsectors=self._microsectors,
            phase_weights=self._thresholds.phase_weights,
        )
        phase_indices: Sequence[int] | None = None
        if active:
            phase_indices = active.microsector.phase_samples.get(active.phase)
        window_metrics = compute_window_metrics(
            records,
            phase_indices=phase_indices,
            bundles=bundles,
        )
        resonance = analyse_modal_resonance(records)
        recommendations = self.recommendation_engine.generate(
            bundles,
            self._microsectors,
            car_model=self.car_model,
            track_name=self.track_name,
        )
        alignment_hint = None
        if active and active.goal:
            target_alignment = float(active.goal.target_phase_alignment)
            target_lag = float(active.goal.target_phase_lag)
            measured_alignment = float(window_metrics.phase_alignment)
            measured_lag = float(window_metrics.phase_lag)
            if (
                measured_alignment < target_alignment - 0.1
                or measured_alignment < 0.0
                or abs(measured_lag - target_lag) > 0.2
            ):
                alignment_hint = (
                    f"θ {measured_lag:+.2f}rad · Siφ {measured_alignment:+.2f}"
                    f" (obj θ {target_lag:+.2f} · Siφ {target_alignment:+.2f})"
                )
        phase_hint = None
        if active:
            active_family = phase_family(active.phase)
            for recommendation in recommendations:
                if phase_family(recommendation.category) == active_family:
                    phase_hint = recommendation.message
                    if alignment_hint:
                        phase_hint = f"{phase_hint} · {alignment_hint}"
                    break
        if phase_hint is None and alignment_hint:
            phase_hint = alignment_hint

        now = self._time_fn()
        if self._plan_interval == 0.0 or now >= self._last_plan_time + self._plan_interval:
            try:
                raw_plan = self.setup_planner.plan(
                    bundles,
                    microsectors=self._microsectors,
                    car_model=self.car_model,
                    track_name=self.track_name,
                )
                try:
                    decision_space = self.setup_planner._space_for_car(self.car_model)
                except Exception:
                    decision_space = None
                self._cached_plan = _build_setup_plan(
                    raw_plan,
                    self.car_model,
                    decision_space,
                    microsectors=self._microsectors,
                )
            except Exception:
                self._cached_plan = None
            self._last_plan_time = now

        tolerance = 0.0
        if active:
            tolerance = self._thresholds.tolerance_for_phase(active.phase)

        top_changes: Tuple[SetupChange, ...] = ()
        if self._cached_plan:
            top_changes = tuple(self._cached_plan.changes[:3])

        self._pages = (
            _render_page_a(
                active,
                bundle,
                tolerance,
                window_metrics,
                self._bundles,
            ),
            _render_page_b(bundle, resonance),
            _render_page_c(
                phase_hint,
                self._cached_plan,
                self._thresholds,
                active,
            ),
            _render_page_d(top_changes, self._macro_status),
        )
        self._dirty = False


def _resolve_active_phase(
    microsectors: Sequence[Microsector], sample_index: int
) -> ActivePhase | None:
    for microsector in reversed(microsectors):
        for phase in PHASE_SEQUENCE:
            bounds = microsector.phase_boundaries.get(phase)
            if bounds is None:
                continue
            start, end = bounds
            if start <= sample_index < end:
                goal = next((goal for goal in microsector.goals if goal.phase == phase), None)
                return ActivePhase(microsector=microsector, phase=phase, goal=goal)
    return None


def _render_page_a(
    active: ActivePhase | None,
    bundle,
    tolerance: float,
    window_metrics: WindowMetrics,
    bundles: Sequence[object] | None = None,
) -> str:
    if not active:
        return _ensure_limit(
            "Sin microsector activo\nΔNFR -- obj -- ±0.00\n"
            + _gradient_line(window_metrics)
        )

    curve_label = f"Curva {active.microsector.index + 1}"
    phase_label = HUD_PHASE_LABELS.get(active.phase, active.phase.capitalize())
    current_delta = getattr(bundle, "delta_nfr", 0.0)
    goal_delta = active.goal.target_delta_nfr if active.goal else 0.0
    spark_delta = _phase_sparkline(active.microsector, bundles, "delta_nfr")
    spark_si = _phase_sparkline(active.microsector, bundles, "sense_index")
    long_component = getattr(bundle, "delta_nfr_longitudinal", 0.0)
    lat_component = getattr(bundle, "delta_nfr_lateral", 0.0)
    if active.goal:
        goal_long = getattr(active.goal, "target_delta_nfr_long", 0.0)
        goal_lat = getattr(active.goal, "target_delta_nfr_lat", 0.0)
        axis_weights = getattr(active.goal, "delta_axis_weights", {})
    else:
        goal_long = 0.0
        goal_lat = 0.0
        axis_weights = {}
    weight_long = float(axis_weights.get("longitudinal", 0.5))
    weight_lat = float(axis_weights.get("lateral", 0.5))
    lines = [
        f"{curve_label} · {phase_label}",
        f"ΔNFR {current_delta:+.2f} obj {goal_delta:+.2f} ±{tolerance:.2f}",
        f"Δ∥ {long_component:+.2f} obj {goal_long:+.2f} · Δ⊥ {lat_component:+.2f} obj {goal_lat:+.2f}",
        f"w∥ {weight_long:.2f} · w⊥ {weight_lat:.2f}",
        _gradient_line(window_metrics),
    ]
    if spark_delta and spark_si:
        lines.append(_truncate_line(f"Fases Δ{spark_delta} · Si{spark_si}"))
    return _ensure_limit("\n".join(lines))


def _gradient_line(window_metrics: WindowMetrics) -> str:
    return (
        f"Si {window_metrics.si:.2f} · ∇Acop {window_metrics.d_nfr_couple:+.2f}"
        f" · ∇Res {window_metrics.d_nfr_res:+.2f} · ∇Flat {window_metrics.d_nfr_flat:+.2f}"
        f" · ν_f {window_metrics.nu_f:.2f}Hz/ν_exc {window_metrics.nu_exc:.2f}Hz"
        f" · ρ {window_metrics.rho:.2f} · θ {window_metrics.phase_lag:+.2f}rad"
        f" · Siφ {window_metrics.phase_alignment:+.2f}"
        f" · UDR {window_metrics.useful_dissonance_ratio:.2f}"
    )


def _phase_sparkline(
    microsector: Microsector,
    bundles: Sequence[object] | None,
    attribute: str,
) -> str:
    if not bundles:
        return ""
    values: List[float] = []
    for phase in PHASE_SEQUENCE:
        indices = microsector.phase_samples.get(phase, ())
        samples: List[float] = []
        for idx in indices:
            if 0 <= idx < len(bundles):
                samples.append(float(getattr(bundles[idx], attribute, 0.0)))
        values.append(sum(samples) / len(samples) if samples else 0.0)
    if not values:
        return ""
    max_abs = max(1e-9, max(abs(value) for value in values))
    spark_chars: List[str] = []
    for value in values:
        ratio = (value / max_abs + 1.0) * 0.5
        index = max(0, min(len(SPARKLINE_BLOCKS) - 1, int(round(ratio * (len(SPARKLINE_BLOCKS) - 1)))))
        spark_chars.append(SPARKLINE_BLOCKS[index])
    return "".join(spark_chars)


def _render_page_b(bundle, resonance: Mapping[str, ModalAnalysis]) -> str:
    node_values = _nodal_delta_map(bundle)
    if not node_values:
        return "ΔNFR nodal\nDatos insuficientes"
    ordered = sorted(node_values.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    max_mag = max(abs(value) for _, value in ordered) or 1.0
    leader = ordered[0][0] if ordered else "--"
    lines = [f"ΔNFR nodal · Líder → {leader}"]
    for name, value in ordered:
        bar = _bar_for_value(value, max_mag)
        lines.append(f"{name:<12}{value:+.2f} {bar}")
    lines.extend(_modal_axis_lines(resonance))
    return _ensure_limit("\n".join(lines))


def _modal_axis_lines(resonance: Mapping[str, ModalAnalysis]) -> List[str]:
    ordered_axes = sorted(
        (
            (axis, analysis)
            for axis, analysis in resonance.items()
            if analysis.peaks
        ),
        key=lambda item: item[1].total_energy,
        reverse=True,
    )
    lines: List[str] = []
    for axis, analysis in ordered_axes:
        peak = max(analysis.peaks, key=lambda p: p.energy, default=None)
        if peak is None:
            continue
        axis_label = MODAL_AXIS_SUMMARY_LABELS.get(axis, NODE_AXIS_LABELS.get(axis, axis))
        ratio_label = f"{analysis.rho:.2f}"
        if analysis.rho > 0.0 and analysis.rho < 0.7:
            ratio_label += "⚠️"
        lines.append(
            f"ν_f {axis_label:<10}{peak.frequency:.1f}Hz ν_exc {analysis.nu_exc:.1f}Hz"
            f" ρ {ratio_label} {peak.classification}"
        )
    return lines


def _format_riesgos(parameters: Sequence[str]) -> str:
    unique = [param for param in dict.fromkeys(parameters) if param]
    return " · ".join(unique)


def _format_sensitivities(derivatives: Mapping[str, float], limit: int = 3) -> str:
    ordered = sorted(
        ((param, value) for param, value in derivatives.items()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    parts: List[str] = []
    for param, value in ordered[:limit]:
        parts.append(f"{param} {value:+.2f}")
    return " · ".join(parts)


def _render_page_c(
    phase_hint: str | None,
    plan: SetupPlan | None,
    thresholds: ThresholdProfile,
    active: ActivePhase | None,
) -> str:
    lines: List[str] = []
    if active and phase_hint:
        lines.append(_truncate_line(f"Hint {phase_hint}"))
    elif active:
        tolerance = thresholds.tolerance_for_phase(active.phase)
        phase_label = HUD_PHASE_LABELS.get(active.phase, active.phase)
        lines.append(
            _truncate_line(
                f"Hint objetivo {phase_label}: {active.goal.target_delta_nfr:+.2f} ±{tolerance:.2f}"
                if active.goal
                else f"Perfil {phase_label} ±{tolerance:.2f}"
            )
        )
    if plan and plan.changes:
        for change in plan.changes[:3]:
            delta = f"{change.delta:+.2f}"
            effect = change.expected_effect.strip()
            lines.append(_truncate_line(f"{change.parameter}: {delta} → {effect}"))
    else:
        lines.append("Plan en preparación…")
    if plan:
        if getattr(plan, "clamped_parameters", ()):  # compatibility with older plans
            riesgos = _format_riesgos(plan.clamped_parameters)
            if riesgos:
                lines.append(_truncate_line(f"riesgos: {riesgos}"))
        dsi_line = _format_sensitivities(plan.sensitivities.get("sense_index", {}))
        if dsi_line:
            lines.append(_truncate_line(f"dSi {dsi_line}"))
    return _ensure_limit("\n".join(lines))


def _render_page_d(
    top_changes: Sequence[SetupChange],
    status: MacroStatus,
) -> str:
    lines: List[str] = ["Aplicar recomendaciones"]
    if top_changes:
        for index, change in enumerate(top_changes, start=1):
            descriptor = _truncate_line(
                f"{index}. {change.parameter}: {change.delta:+.2f}"
            )
            lines.append(descriptor)
    else:
        lines.append("Sin recomendaciones disponibles")

    if status.next_change:
        change = status.next_change
        lines.append(
            _truncate_line(f"Siguiente → {change.parameter} {change.delta:+.2f}")
        )
    else:
        lines.append("Siguiente → --")

    if status.queue_size:
        lines.append(f"Cola macros {status.queue_size}")

    lines.append("APLICAR SIGUIENTE (Shift+Clic)")

    for warning in status.warnings:
        if not warning:
            continue
        lines.append(_truncate_line(f"⚠️ {warning}"))

    return _ensure_limit("\n".join(lines))


def _nodal_delta_map(bundle) -> Mapping[str, float]:
    nodes = {
        "tyres": getattr(bundle, "tyres", None),
        "suspension": getattr(bundle, "suspension", None),
        "chassis": getattr(bundle, "chassis", None),
        "brakes": getattr(bundle, "brakes", None),
        "transmission": getattr(bundle, "transmission", None),
        "track": getattr(bundle, "track", None),
        "driver": getattr(bundle, "driver", None),
    }
    resolved: Dict[str, float] = {}
    for key, node in nodes.items():
        if node is None:
            continue
        label = NODE_LABELS.get(key, key)
        resolved[label] = float(getattr(node, "delta_nfr", 0.0))
    return resolved


def _bar_for_value(value: float, max_value: float, width: int = 10) -> str:
    ratio = min(1.0, abs(value) / max(1e-9, max_value))
    filled = max(0, min(width, int(round(ratio * width))))
    return "#" * filled + "." * (width - filled)


def _dominant_peak(resonance: Mapping[str, ModalAnalysis]) -> Tuple[str, ModalPeak] | None:
    best: Tuple[str, ModalPeak] | None = None
    for axis, analysis in resonance.items():
        for peak in analysis.peaks:
            if best is None or peak.energy > best[1].energy:
                best = (axis, peak)
    return best


def _truncate_line(value: str, limit: int = 78) -> str:
    encoded = value.encode("utf8")
    if len(encoded) <= limit:
        return value
    while encoded and len(encoded) > limit:
        value = value[:-1]
        encoded = value.encode("utf8")
    return value


def _ensure_limit(payload: str) -> str:
    encoded = payload.encode("utf8")
    if len(encoded) <= PAYLOAD_LIMIT:
        return payload
    while encoded and len(encoded) > PAYLOAD_LIMIT:
        payload = payload[:-1]
        encoded = payload.encode("utf8")
    return payload


class OSDController:
    """High-level orchestrator that drives the HUD overlay."""

    def __init__(
        self,
        *,
        host: str,
        outsim_port: int,
        outgauge_port: int,
        insim_port: int,
        insim_keepalive: float,
        update_rate: float = DEFAULT_UPDATE_RATE,
        car_model: str = "XFG",
        track_name: str = "generic",
        layout: ButtonLayout | None = None,
        fusion: TelemetryFusion | None = None,
        hud: TelemetryHUD | None = None,
    ) -> None:
        self.host = host
        self.outsim_port = outsim_port
        self.outgauge_port = outgauge_port
        self.insim_port = insim_port
        self.insim_keepalive = max(1.0, float(insim_keepalive))
        self.layout = (layout or ButtonLayout()).clamp()
        self.update_period = 1.0 / max(1.0, float(update_rate))
        self.fusion = fusion or TelemetryFusion()
        self.hud = hud or TelemetryHUD(
            car_model=car_model,
            track_name=track_name,
            plan_interval=DEFAULT_PLAN_INTERVAL,
        )
        self._pager = HUDPager(self.hud.pages())
        self._idle_backoff = 0.01
        self._macro_queue: MacroQueue | None = None
        self._menu_open = False
        self._menu_open_last_seen = 0.0
        self._car_stopped = True
        self._next_change_index = 0
        self._last_plan_signature: Tuple[Tuple[str, float], ...] = ()

    APPLY_TRIGGER_FLAG = 0x10
    STOPPED_SPEED_THRESHOLD = 1.0

    def run(self) -> str:
        outsim: OutSimUDPClient | None = None
        outgauge: OutGaugeUDPClient | None = None
        insim: InSimClient | None = None
        overlay: OverlayManager | None = None
        last_render = 0.0
        try:
            outsim = OutSimUDPClient(host=self.host, port=self.outsim_port)
            outgauge = OutGaugeUDPClient(host=self.host, port=self.outgauge_port)
            insim = InSimClient(
                host=self.host,
                port=self.insim_port,
                keepalive_interval=self.insim_keepalive,
            )
            insim.connect()
            insim.subscribe_controls()
            overlay = OverlayManager(insim, layout=self.layout)
            overlay.show("Inicializando HUD…")
            self._macro_queue = MacroQueue(insim.send_command)
            self._refresh_macro_status()
            last_render = monotonic()
            try:
                while True:
                    outsim_packet = outsim.recv()
                    outgauge_packet = outgauge.recv()
                    if outsim_packet and outgauge_packet:
                        record = self.fusion.fuse(outsim_packet, outgauge_packet)
                        self.hud.append(record)
                        self._update_vehicle_state(record)
                        self._infer_menu_state(outgauge_packet)
                        self._pager.update(self.hud.pages())
                        self._refresh_macro_status()
                    else:
                        sleep(self._idle_backoff)

                    event = overlay.poll_button(0.0)
                    self._handle_button_event(event)

                    now = monotonic()
                    if now - last_render >= self.update_period:
                        overlay.show(self._pager.current())
                        last_render = now

                    overlay.tick()
                    if self._macro_queue and self._macro_queue.tick():
                        self._refresh_macro_status()
            except KeyboardInterrupt:
                return "HUD detenido por el usuario."
        finally:
            if overlay is not None:
                overlay.clear()
                overlay.close()
            if insim is not None:
                insim.close()
            if outsim is not None:
                outsim.close()
            if outgauge is not None:
                outgauge.close()
        return "HUD finalizado."

    def set_menu_open(self, open_state: bool) -> None:
        self._menu_open = bool(open_state)
        if open_state:
            self._menu_open_last_seen = monotonic()
        self._refresh_macro_status()

    def _handle_button_event(self, event: ButtonEvent | None) -> None:
        if event is None:
            return
        if event.type_in != 0:
            return
        if event.click_id != self.layout.click_id:
            return

        if self._pager.index == len(self._pager.pages) - 1 and self._should_trigger_macro(event):
            self._attempt_enqueue_macro()
        else:
            self._pager.advance()

    def _should_trigger_macro(self, event: ButtonEvent) -> bool:
        if event.flags & self.APPLY_TRIGGER_FLAG:
            return True
        if event.typed_char and event.typed_char.lower() in {"a", "\r"}:
            return True
        return False

    def _attempt_enqueue_macro(self) -> None:
        queue = self._macro_queue
        if queue is None:
            return
        next_change = self._resolve_next_change()
        warnings = self._macro_preflight_warnings(next_change)
        if warnings:
            self.hud.update_macro_status(
                MacroStatus(
                    next_change=next_change,
                    warnings=tuple(warnings),
                    queue_size=len(queue),
                )
            )
            self._pager.update(self.hud.pages())
            return

        if not next_change:
            self._pager.advance()
            return

        sequence = self._sequence_for_change(next_change)
        queue.enqueue_press_sequence(sequence)
        self._next_change_index += 1
        self.hud.update_macro_status(
            MacroStatus(
                next_change=self._resolve_next_change(),
                queue_size=len(queue),
            )
        )
        self._pager.update(self.hud.pages())

    def _sequence_for_change(self, change: SetupChange) -> Sequence[str]:
        direction = "+" if change.delta >= 0 else "-"
        steps = max(1, int(round(abs(change.delta))))
        sequence = ["F12", "F11"]
        sequence.extend(direction for _ in range(steps))
        sequence.append("F12")
        return sequence

    def _resolve_next_change(self) -> SetupChange | None:
        plan = self.hud.plan()
        if not plan or not plan.changes:
            self._next_change_index = 0
            self._last_plan_signature = ()
            return None

        signature = tuple((change.parameter, float(change.delta)) for change in plan.changes)
        if signature != self._last_plan_signature:
            self._next_change_index = 0
            self._last_plan_signature = signature

        if self._next_change_index >= len(plan.changes):
            self._next_change_index = 0
        return plan.changes[self._next_change_index]

    def _macro_preflight_warnings(self, next_change: SetupChange | None) -> List[str]:
        warnings: List[str] = []
        if next_change is None:
            warnings.append("Sin cambios pendientes")
        if not self._menu_open:
            warnings.append("Abre el menú de boxes (F12)")
        if not self._car_stopped:
            warnings.append("Detén el coche antes de aplicar")
        return warnings

    def _refresh_macro_status(self) -> None:
        next_change = self._resolve_next_change()
        queue_size = len(self._macro_queue) if self._macro_queue else 0
        warnings = tuple(self._macro_preflight_warnings(next_change))
        self.hud.update_macro_status(
            MacroStatus(
                next_change=next_change,
                warnings=warnings,
                queue_size=queue_size,
            )
        )
        self._pager.update(self.hud.pages())

    def _update_vehicle_state(self, record: TelemetryRecord) -> None:
        self._car_stopped = abs(record.speed) <= self.STOPPED_SPEED_THRESHOLD

    def _infer_menu_state(self, outgauge) -> None:
        if outgauge is None:
            return
        display = f"{outgauge.display1} {outgauge.display2}".strip().upper()
        keywords = ("F12", "PIT", "BOX", "SETUP")
        now = monotonic()
        if any(keyword in display for keyword in keywords):
            self._menu_open = True
            self._menu_open_last_seen = now
        elif self._menu_open and (now - self._menu_open_last_seen) > 3.0:
            self._menu_open = False


def _build_setup_plan(
    plan,
    car_model: str,
    decision_space=None,
    microsectors: Sequence[Microsector] | None = None,
) -> SetupPlan:
    action_recommendations = [
        rec
        for rec in plan.recommendations
        if rec.parameter and rec.delta is not None
    ]

    ordered_actions = sorted(
        action_recommendations,
        key=lambda rec: (rec.priority, -abs(rec.delta or 0.0)),
    )

    aggregated_rationales = [rec.rationale for rec in plan.recommendations if rec.rationale]
    aggregated_effects = [rec.message for rec in plan.recommendations if rec.message]

    if ordered_actions:
        changes = [
            SetupChange(
                parameter=rec.parameter or "",
                delta=float(rec.delta or 0.0),
                rationale=rec.rationale or "",
                expected_effect=rec.message,
            )
            for rec in ordered_actions
        ]
        if not aggregated_rationales:
            aggregated_rationales = [rec.rationale for rec in ordered_actions if rec.rationale]
        if not aggregated_effects:
            aggregated_effects = [rec.message for rec in ordered_actions if rec.message]
    else:
        unique_rationales = list(dict.fromkeys(aggregated_rationales))
        unique_effects = list(dict.fromkeys(aggregated_effects))
        rationale_text = "; ".join(unique_rationales or ["Optimización de objetivo Si/ΔNFR"])
        effect_text = "; ".join(unique_effects or ["Mejora equilibrada del coche"])
        changes = [
            SetupChange(
                parameter=parameter,
                delta=float(value),
                rationale=rationale_text,
                expected_effect=effect_text,
            )
            for parameter, value in plan.decision_vector.items()
            if abs(value) > 1e-6
        ]
        aggregated_rationales = unique_rationales or ["Optimización de objetivo Si/ΔNFR"]
        aggregated_effects = unique_effects or ["Mejora equilibrada del coche"]

    unique_rationales = list(dict.fromkeys(aggregated_rationales or ["Optimización de objetivo Si/ΔNFR"]))
    unique_effects = list(dict.fromkeys(aggregated_effects or ["Mejora equilibrada del coche"]))

    vector = getattr(plan, "decision_vector", {})
    clamped: List[str] = []
    if decision_space is not None:
        for variable in getattr(decision_space, "variables", ()):
            name = getattr(variable, "name", None)
            lower = getattr(variable, "lower", None)
            upper = getattr(variable, "upper", None)
            if name is None or lower is None or upper is None:
                continue
            value = vector.get(name)
            if value is None:
                continue
            if math.isclose(value, lower, abs_tol=1e-6) or math.isclose(
                value, upper, abs_tol=1e-6
            ):
                clamped.append(str(name))

    axis_target_map: Dict[str, Dict[str, float]] = {}
    axis_weight_map: Dict[str, Dict[str, float]] = {}
    axis_counts: Dict[str, int] = {}

    if microsectors:
        for microsector in microsectors:
            for goal in microsector.goals:
                family = phase_family(goal.phase)
                if family is None:
                    continue
                target_entry = axis_target_map.setdefault(
                    family,
                    {"longitudinal": 0.0, "lateral": 0.0},
                )
                target_entry["longitudinal"] += float(getattr(goal, "target_delta_nfr_long", 0.0))
                target_entry["lateral"] += float(getattr(goal, "target_delta_nfr_lat", 0.0))
                weight_entry = axis_weight_map.setdefault(
                    family,
                    {"longitudinal": 0.0, "lateral": 0.0},
                )
                weights = getattr(goal, "delta_axis_weights", {})
                weight_entry["longitudinal"] += float(weights.get("longitudinal", 0.0))
                weight_entry["lateral"] += float(weights.get("lateral", 0.0))
                axis_counts[family] = axis_counts.get(family, 0) + 1

    for phase, count in list(axis_counts.items()):
        if count <= 0:
            continue
        target_entry = axis_target_map[phase]
        weight_entry = axis_weight_map[phase]
        axis_target_map[phase] = {
            "longitudinal": target_entry["longitudinal"] / count,
            "lateral": target_entry["lateral"] / count,
        }
        axis_weight_map[phase] = {
            "longitudinal": weight_entry["longitudinal"] / count,
            "lateral": weight_entry["lateral"] / count,
        }

    return SetupPlan(
        car_model=car_model,
        session=None,
        changes=tuple(changes),
        rationales=tuple(unique_rationales),
        expected_effects=tuple(unique_effects),
        sensitivities=plan.sensitivities,
        phase_sensitivities=plan.phase_sensitivities,
        clamped_parameters=tuple(clamped),
        phase_axis_targets=axis_target_map,
        phase_axis_weights=axis_weight_map,
    )


__all__ = [
    "ActivePhase",
    "HUDPager",
    "MacroStatus",
    "OSDController",
    "TelemetryHUD",
]

