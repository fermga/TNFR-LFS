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
    OverlayManager,
    OutGaugeUDPClient,
    OutSimUDPClient,
    TelemetryFusion,
)
from ..core.epi import EPIExtractor, TelemetryRecord
from ..core.operators import DissonanceBreakdown, orchestrate_delta_metrics
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


class HUDPager:
    """Track the active HUD page and react to button clicks."""

    def __init__(self, pages: Sequence[str] | None = None) -> None:
        self._pages: Tuple[str, ...] = tuple(pages or ())
        if not self._pages:
            self._pages = ("Esperando telemetría…",) * 3
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
        car_model: str = "generic_gt",
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
        self._pages: Tuple[str, str, str] = (
            "Esperando telemetría…",
            "ΔNFR nodal en espera",
            "Plan en preparación…",
        )
        self._dirty = True
        self._plan_interval = max(0.0, float(plan_interval))
        self._time_fn = time_fn
        self._last_plan_time = -math.inf
        self._cached_plan: SetupPlan | None = None
        self._thresholds: ThresholdProfile = (
            self.recommendation_engine._resolve_context(  # type: ignore[attr-defined]
                car_model, track_name
            ).thresholds
        )

    def append(self, record: TelemetryRecord) -> None:
        self._records.append(record)
        self._dirty = True

    def pages(self) -> Tuple[str, str, str]:
        if self._dirty:
            self._recompute()
        return self._pages

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
        metrics = orchestrate_delta_metrics(
            [records],
            goal_delta,
            goal_si,
            microsectors=self._microsectors,
            phase_weights=self._thresholds.phase_weights,
        )
        resonance = analyse_modal_resonance(records)
        recommendations = self.recommendation_engine.generate(
            bundles,
            self._microsectors,
            car_model=self.car_model,
            track_name=self.track_name,
        )
        phase_hint = None
        if active:
            active_family = phase_family(active.phase)
            for recommendation in recommendations:
                if phase_family(recommendation.category) == active_family:
                    phase_hint = recommendation.message
                    break

        now = self._time_fn()
        if self._plan_interval == 0.0 or now >= self._last_plan_time + self._plan_interval:
            try:
                raw_plan = self.setup_planner.plan(
                    bundles,
                    microsectors=self._microsectors,
                    car_model=self.car_model,
                )
                self._cached_plan = _build_setup_plan(raw_plan, self.car_model)
            except Exception:
                self._cached_plan = None
            self._last_plan_time = now

        tolerance = 0.0
        if active:
            tolerance = self._thresholds.tolerance_for_phase(active.phase)

        self._pages = (
            _render_page_a(active, bundle, tolerance, metrics["dissonance_breakdown"], metrics["coupling"]),
            _render_page_b(bundle, resonance),
            _render_page_c(
                phase_hint,
                self._cached_plan,
                self._thresholds,
                active,
            ),
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
    breakdown: DissonanceBreakdown,
    coupling: float,
) -> str:
    if not active:
        return _ensure_limit(
            "Sin microsector activo\nΔNFR -- obj -- ±0.00\nAcop -- · Útil 0% · Paras 0%"
        )

    curve_label = f"Curva {active.microsector.index + 1}"
    phase_label = HUD_PHASE_LABELS.get(active.phase, active.phase.capitalize())
    current_delta = getattr(bundle, "delta_nfr", 0.0)
    goal_delta = active.goal.target_delta_nfr if active.goal else 0.0
    useful = breakdown.useful_percentage
    parasitic = breakdown.parasitic_percentage
    lines = [
        f"{curve_label} · {phase_label}",
        f"ΔNFR {current_delta:+.2f} obj {goal_delta:+.2f} ±{tolerance:.2f}",
        f"Acop {coupling:.2f} · Útil {useful:.0f}% · Paras {parasitic:.0f}%",
    ]
    return _ensure_limit("\n".join(lines))


def _render_page_b(bundle, resonance: Mapping[str, ModalAnalysis]) -> str:
    node_values = _nodal_delta_map(bundle)
    if not node_values:
        return "ΔNFR nodal\nDatos insuficientes"
    ordered = sorted(node_values.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    max_mag = max(abs(value) for _, value in ordered) or 1.0
    lines = ["ΔNFR nodal"]
    for name, value in ordered:
        bar = _bar_for_value(value, max_mag)
        lines.append(f"{name:<12}{value:+.2f} {bar}")
    dominant = _dominant_peak(resonance)
    if dominant is not None:
        axis, peak = dominant
        axis_label = NODE_AXIS_LABELS.get(axis, axis)
        lines.append(f"Modo {axis_label} {peak.frequency:.1f}Hz {peak.classification}")
    return _ensure_limit("\n".join(lines))


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
        car_model: str = "generic_gt",
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
            last_render = monotonic()
            try:
                while True:
                    outsim_packet = outsim.recv()
                    outgauge_packet = outgauge.recv()
                    if outsim_packet and outgauge_packet:
                        record = self.fusion.fuse(outsim_packet, outgauge_packet)
                        self.hud.append(record)
                        self._pager.update(self.hud.pages())
                    else:
                        sleep(self._idle_backoff)

                    event = overlay.poll_button(0.0)
                    self._pager.handle_event(event, self.layout)

                    now = monotonic()
                    if now - last_render >= self.update_period:
                        overlay.show(self._pager.current())
                        last_render = now

                    overlay.tick()
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


def _build_setup_plan(plan, car_model: str) -> SetupPlan:
    rationales = [rec.rationale for rec in plan.recommendations if rec.rationale]
    effects = [rec.message for rec in plan.recommendations if rec.message]
    if not rationales:
        rationales = ["Optimización de objetivo Si/ΔNFR"]
    if not effects:
        effects = ["Mejora equilibrada del coche"]

    unique_rationales = list(dict.fromkeys(rationales))
    unique_effects = list(dict.fromkeys(effects))
    rationale_text = "; ".join(unique_rationales)
    effect_text = "; ".join(unique_effects)

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

    return SetupPlan(
        car_model=car_model,
        session=None,
        changes=tuple(changes),
        rationales=tuple(unique_rationales),
        expected_effects=tuple(unique_effects),
        sensitivities=plan.sensitivities,
    )


__all__ = [
    "ActivePhase",
    "HUDPager",
    "OSDController",
    "TelemetryHUD",
]

