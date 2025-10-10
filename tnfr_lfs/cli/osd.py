"""Runtime HUD controller for the on-screen display (OSD)."""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from time import monotonic, sleep
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

from ..ingestion import (
    ButtonEvent,
    ButtonLayout,
    InSimClient,
    MacroQueue,
    OverlayManager,
    OutGaugePacket,
    OutGaugeUDPClient,
    OutSimPacket,
    OutSimUDPClient,
    TelemetryFusion,
)
from ..core.constants import (
    BRAKE_TEMPERATURE_MEAN_KEYS,
    BRAKE_TEMPERATURE_STD_KEYS,
    PRESSURE_MEAN_KEYS,
    PRESSURE_STD_KEYS,
    TEMPERATURE_MEAN_KEYS,
    TEMPERATURE_STD_KEYS,
    WHEEL_LABELS,
    WHEEL_SUFFIXES,
)
from ..core.epi import EPIExtractor, TelemetryRecord
from ..core.metrics import (
    AeroBalanceDrift,
    AeroBalanceDriftBin,
    BrakeHeadroom,
    CPHIReport,
    WindowMetrics,
    compute_window_metrics,
)
from ..core.operators import orchestrate_delta_metrics
from ..core.operator_detection import canonical_operator_label
from ..core.phases import PHASE_SEQUENCE, phase_family
from ..core.resonance import ModalAnalysis, ModalPeak, analyse_modal_resonance
from ..core.segmentation import (
    Goal,
    Microsector,
    detect_quiet_microsector_streaks,
    microsector_stability_metrics,
    segment_microsectors,
)
from ..exporters.setup_plan import (
    SetupChange,
    SetupPlan,
    compute_phase_axis_summary,
    phase_axis_summary_lines,
)
from ..recommender import RecommendationEngine, SetupPlanner
from ..recommender.rules import NODE_LABELS, ThresholdProfile, RuleProfileObjectives
from ..session import format_session_messages
from ..utils.numeric import _safe_float
from ..utils.sparkline import DEFAULT_SPARKLINE_BLOCKS, render_sparkline


logger = logging.getLogger(__name__)

PAYLOAD_LIMIT = OverlayManager.MAX_BUTTON_TEXT - 1
DEFAULT_UPDATE_RATE = 6.0
DEFAULT_PLAN_INTERVAL = 5.0

MOTOR_LATENCY_TARGETS_MS: Mapping[str, float] = {"entry": 250.0, "apex": 180.0, "exit": 220.0}

NODE_AXIS_LABELS = {
    "yaw": "Yaw",
    "roll": "Roll",
    "pitch": "Pitch",
}

MODAL_AXIS_SUMMARY_LABELS = {
    "yaw": "Yaw",
    "roll": "Roll",
    "pitch": "Suspension",
}

SPARKLINE_BLOCKS = DEFAULT_SPARKLINE_BLOCKS
SPARKLINE_MIDPOINT_BLOCK = SPARKLINE_BLOCKS[len(SPARKLINE_BLOCKS) // 2]

HUD_PHASE_LABELS = {
    "entry1": "Entry 1",
    "entry2": "Entry 2",
    "apex3a": "Apex 3A",
    "apex3b": "Apex 3B",
    "exit4": "Exit 4",
}

PACKING_HIGH_SPEED_THRESHOLD = 35.0
PACKING_AR_THRESHOLD = 1.25
PACKING_ASYMMETRY_GAP = 0.35


@dataclass(frozen=True)
class ActivePhase:
    """Resolved microsector/phase context for the current record."""

    microsector: Microsector
    phase: str
    goal: Optional[Goal]


@dataclass(frozen=True)
class MacroStatus:
    """State exposed in the “Apply” HUD page."""

    next_change: Optional[SetupChange] = None
    warnings: Tuple[str, ...] = ()
    queue_size: int = 0


class HUDPager:
    """Track the active HUD page and react to button clicks."""

    def __init__(self, pages: Optional[Sequence[str]] = None) -> None:
        self._pages: Tuple[str, ...] = tuple(pages or ())
        if not self._pages:
            self._pages = (
                "Waiting for telemetry…",
                "Nodal ΔNFR pending",
                "Plan in progress…",
                "Apply pending…",
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

    def handle_event(self, event: Optional[ButtonEvent], layout: ButtonLayout) -> bool:
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
        extractor: Optional[EPIExtractor] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
        setup_planner: Optional[SetupPlanner] = None,
        plan_interval: float = DEFAULT_PLAN_INTERVAL,
        time_fn: Callable[[], float] = monotonic,
        session: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._records: Deque[TelemetryRecord] = deque(maxlen=max(8, int(window)))
        self.extractor = extractor or EPIExtractor()
        self.car_model = car_model
        self.track_name = track_name
        self.recommendation_engine = recommendation_engine or RecommendationEngine(
            car_model=car_model, track_name=track_name
        )
        if session is not None:
            self.recommendation_engine.session = session
        self.setup_planner = setup_planner or SetupPlanner(
            recommendation_engine=self.recommendation_engine
        )
        self._operator_state: MutableMapping[str, Dict[str, Dict[str, object]]] = {}
        self._microsectors: Sequence[Microsector] = ()
        self._bundles: Sequence[object] = ()
        self._pages: Tuple[str, str, str, str] = (
            "Waiting for telemetry…",
            "Nodal ΔNFR pending",
            "Plan in progress…",
            "Apply pending…",
        )
        self._coherence_series: Tuple[float, ...] = ()
        self._coherence_index: float = 0.0
        self._frequency_classification: str = "no data"
        self._sense_state: Dict[str, object] = {
            "series": (),
            "memory": (),
            "average": 0.0,
        }
        self._quiet_sequences: Tuple[Tuple[int, ...], ...] = ()
        self._dirty = True
        self._plan_interval = max(0.0, float(plan_interval))
        self._time_fn: Callable[[], float] = time_fn
        self._last_plan_time = -math.inf
        self._cached_plan: Optional[SetupPlan] = None
        self._macro_status = MacroStatus()
        self._session_messages: Tuple[str, ...] = tuple(format_session_messages(session))
        context = self.recommendation_engine._resolve_context(  # type: ignore[attr-defined]
            car_model, track_name
        )
        self._thresholds: ThresholdProfile = context.thresholds
        self._hud_thresholds: Mapping[str, float] = context.thresholds.hud_thresholds
        self._profile_objectives: RuleProfileObjectives = context.objectives
        self._session_hints: Mapping[str, object] = context.session_hints
        self._lap_integrals: Tuple[float, ...] = ()
        self._integral_cov: Optional[float] = None
        self._integral_cov_lap_count: int = 0

    def append(self, record: TelemetryRecord) -> None:
        self._records.append(record)
        self._dirty = True

    def pages(self) -> Tuple[str, str, str, str]:
        if self._dirty:
            self._recompute()
        return self._pages

    def plan(self) -> Optional[SetupPlan]:
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
        context = self.recommendation_engine._resolve_context(  # type: ignore[attr-defined]
            self.car_model, self.track_name
        )
        self._thresholds = context.thresholds
        self._hud_thresholds = context.thresholds.hud_thresholds
        self._profile_objectives = context.objectives
        self._session_hints = context.session_hints
        self._session_messages = tuple(
            format_session_messages(getattr(self.recommendation_engine, "session", None))
        )

        records = list(self._records)
        if len(records) < 8:
            self._pages = (
                "Waiting for telemetry…",
                "Nodal ΔNFR pending",
                "Plan in progress…",
                "Apply pending…",
            )
            self._coherence_series = ()
            self._coherence_index = 0.0
            self._frequency_classification = "no data"
            self._sense_state = {"series": (), "memory": (), "average": 0.0}
            self._quiet_sequences = ()
            self._lap_integrals = ()
            self._integral_cov = None
            self._integral_cov_lap_count = 0
            self._dirty = False
            return

        bundles = self.extractor.extract(records)
        if not bundles:
            self._coherence_series = ()
            self._coherence_index = 0.0
            self._frequency_classification = "no data"
            self._sense_state = {"series": (), "memory": (), "average": 0.0}
            self._quiet_sequences = ()
            self._lap_integrals = ()
            self._integral_cov = None
            self._integral_cov_lap_count = 0
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
        quiet_sequences = detect_quiet_microsector_streaks(self._microsectors)
        self._quiet_sequences = tuple(tuple(sequence) for sequence in quiet_sequences)
        active = _resolve_active_phase(self._microsectors, len(records) - 1)

        bundle = bundles[-1]
        goal_delta = active.goal.target_delta_nfr if active and active.goal else 0.0
        goal_si = active.goal.target_sense_index if active and active.goal else 0.75
        metrics_state = orchestrate_delta_metrics(
            [records],
            goal_delta,
            goal_si,
            microsectors=self._microsectors,
            phase_weights=self._thresholds.phase_weights,
            operator_state=self._operator_state,
        )
        staged_bundles = tuple(metrics_state.get("bundles", ())) if metrics_state else ()
        if staged_bundles:
            bundles = staged_bundles
        self._bundles = bundles
        lap_integrals = _lap_integral_series(records, bundles)
        self._lap_integrals = tuple(lap_integrals)
        self._integral_cov_lap_count = len(lap_integrals)
        min_laps = max(
            2,
            int(round(_hud_threshold_value(self._hud_thresholds, "integral_cov_min_laps", 3.0))),
        )
        if self._integral_cov_lap_count >= min_laps and lap_integrals:
            mean_value = sum(lap_integrals) / self._integral_cov_lap_count
            if mean_value > 1e-9:
                variance = sum(
                    (value - mean_value) ** 2 for value in lap_integrals
                ) / self._integral_cov_lap_count
                self._integral_cov = math.sqrt(variance) / mean_value
            else:
                self._integral_cov = 0.0
        else:
            self._integral_cov = None
        coherence_series = metrics_state.get("coherence_index_series", ()) if metrics_state else ()
        self._coherence_series = tuple(float(value) for value in coherence_series)
        self._coherence_index = float(metrics_state.get("coherence_index", 0.0)) if metrics_state else 0.0
        frequency_classification = "no data"
        if metrics_state:
            frequency_classification = str(
                metrics_state.get(
                    "frequency_classification",
                    getattr(bundles[-1], "nu_f_classification", "no data") if bundles else "no data",
                )
            )
        self._frequency_classification = frequency_classification or "no data"
        sense_memory = metrics_state.get("sense_memory", {}) if metrics_state else {}
        if isinstance(sense_memory, Mapping):
            series = tuple(float(value) for value in sense_memory.get("series", ()))
            memory = tuple(float(value) for value in sense_memory.get("memory", ()))
            average = float(sense_memory.get("average", 0.0))
            self._sense_state = {"series": series, "memory": memory, "average": average}
        else:
            self._sense_state = {"series": (), "memory": (), "average": 0.0}
        phase_indices: Optional[Sequence[int]] = None
        if active:
            phase_indices = active.microsector.phase_samples.get(active.phase)
        window_metrics = compute_window_metrics(
            records,
            phase_indices=phase_indices,
            bundles=bundles,
            objectives=self._profile_objectives,
        )
        if not self._coherence_index:
            self._coherence_index = float(window_metrics.coherence_index)
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
            target_synchrony = float(
                getattr(active.goal, "target_phase_synchrony", target_alignment)
            )
            measured_alignment = float(window_metrics.phase_alignment)
            measured_lag = float(window_metrics.phase_lag)
            measured_synchrony = float(window_metrics.phase_synchrony_index)
            synchrony_gap = target_synchrony - measured_synchrony
            if (
                measured_alignment < target_alignment - 0.1
                or measured_alignment < 0.0
                or abs(measured_lag - target_lag) > 0.2
                or synchrony_gap > 0.08
                or measured_synchrony < 0.72
            ):
                alignment_hint = (
                    f"θ {measured_lag:+.2f}rad · Siφ {measured_alignment:+.2f}"
                    f" · Φsync {measured_synchrony:.2f}"
                    f" (obj θ {target_lag:+.2f} · Siφ {target_alignment:+.2f}"
                    f" · Φsync {target_synchrony:.2f})"
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
                coherence_series=self._coherence_series,
                coherence_index=self._coherence_index,
                frequency_classification=self._frequency_classification,
                sense_state=self._sense_state,
                quiet_sequences=self._quiet_sequences,
                hud_thresholds=self._hud_thresholds,
                integral_cov=self._integral_cov,
                integral_lap_count=self._integral_cov_lap_count,
            ),
            _render_page_b(
                bundle,
                resonance,
                self._bundles,
                self._coherence_series,
                self._frequency_classification,
                self._coherence_index,
            ),
            _render_page_c(
                phase_hint,
                self._cached_plan,
                self._thresholds,
                active,
                sense_state=self._sense_state,
                microsectors=self._microsectors,
                quiet_sequences=self._quiet_sequences,
                session_messages=self._session_messages,
                window_metrics=window_metrics,
                objectives=self._profile_objectives,
                session_hints=self._session_hints,
                bundles=self._bundles,
            ),
            _render_page_d(top_changes, self._macro_status),
        )
        self._dirty = False


def _resolve_active_phase(
    microsectors: Sequence[Microsector], sample_index: int
) -> Optional[ActivePhase]:
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
    active: Optional[ActivePhase],
    bundle,
    tolerance: float,
    window_metrics: WindowMetrics,
    bundles: Optional[Sequence[object]] = None,
    *,
    coherence_series: Optional[Sequence[float]] = None,
    coherence_index: Optional[float] = None,
    frequency_classification: str = "",
    sense_state: Optional[Mapping[str, object]] = None,
    quiet_sequences: Optional[Sequence[Sequence[int]]] = None,
    hud_thresholds: Optional[Mapping[str, float]] = None,
    integral_cov: Optional[float] = None,
    integral_lap_count: int = 0,
) -> str:
    coherence_value = (
        float(coherence_index)
        if coherence_index is not None
        else float(window_metrics.coherence_index)
    )
    coherence_line = _coherence_meter_line(coherence_value, coherence_series)
    nu_wave = _nu_wave_line(bundles)
    sense_line = _sense_state_line(sense_state)
    motor_latency_line = _motor_latency_line(window_metrics)
    classification_segment = (
        f" · ν_f {frequency_classification}" if frequency_classification else ""
    )
    entropy_line = _entropy_indicator_line(window_metrics, hud_thresholds)
    integral_line = _integral_cov_line(integral_cov, integral_lap_count, hud_thresholds)
    if not active:
        lines = [
            _truncate_line(f"No active microsector{classification_segment}"),
            _truncate_line(
                "ΔNFR -- obj -- ±0.00 "
                + _delta_threshold_meter(
                    getattr(bundle, "delta_nfr", 0.0), 0.0, tolerance
                )
            ),
        ]
        sigma_line = _truncate_line(
            f"σΔ {window_metrics.delta_nfr_std:.3f} · σΔₙ {window_metrics.nodal_delta_nfr_std:.3f}"
        )
        candidate = "\n".join((*lines, sigma_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(sigma_line)
        if sense_line:
            candidate = "\n".join((*lines, sense_line))
            if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
                lines.append(sense_line)
        if entropy_line:
            candidate = "\n".join((*lines, entropy_line))
            if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
                lines.append(entropy_line)
        gradient_entry = _truncate_line(
            (
                f"Stability {window_metrics.si:.2f}"
                f" · ∇Coupl {window_metrics.d_nfr_couple:+.2f}"
                f" · C(t) {coherence_value:.2f}"
                f" · τmot {window_metrics.motor_latency_ms:.0f}ms"
            ),
            limit=56,
        )
        candidate = "\n".join((*lines, gradient_entry))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(gradient_entry)
        if integral_line:
            candidate = "\n".join((*lines, integral_line))
            if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
                lines.append(integral_line)
        coupling_alert = _coupling_alert_line(
            window_metrics,
            longitudinal_delta=float(
                getattr(bundle, "delta_nfr_proj_longitudinal", 0.0)
            ),
            tolerance=tolerance,
        )
        if coupling_alert:
            candidate = "\n".join((*lines, coupling_alert))
            if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
                lines.append(coupling_alert)
        return _ensure_limit("\n".join(lines))

    curve_label = f"Corner {active.microsector.index + 1}"
    phase_label = HUD_PHASE_LABELS.get(active.phase, active.phase.capitalize())
    current_delta = getattr(bundle, "delta_nfr", 0.0)
    goal_delta = active.goal.target_delta_nfr if active.goal else 0.0
    spark_delta = _phase_sparkline(active.microsector, bundles, "delta_nfr")
    spark_si = _phase_sparkline(active.microsector, bundles, "sense_index")
    long_component = getattr(bundle, "delta_nfr_proj_longitudinal", 0.0)
    lat_component = getattr(bundle, "delta_nfr_proj_lateral", 0.0)
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
    gauge = _delta_threshold_meter(current_delta, goal_delta, tolerance)
    lines = [
        _truncate_line(f"{curve_label} · {phase_label}{classification_segment}"),
        _truncate_line(
            f"ΔNFR {current_delta:+.2f} obj {goal_delta:+.2f} ±{tolerance:.2f} {gauge}"
        ),
    ]
    phase_key = str(active.phase)
    phase_sigma = float(
        window_metrics.phase_delta_nfr_std.get(phase_key, window_metrics.delta_nfr_std)
    )
    phase_nodal_sigma = float(
        window_metrics.phase_nodal_delta_nfr_std.get(
            phase_key, window_metrics.nodal_delta_nfr_std
        )
    )
    sigma_line = _truncate_line(
        f"σΔ {phase_sigma:.3f} ({window_metrics.delta_nfr_std:.3f}) · σΔₙ {phase_nodal_sigma:.3f}"
    )
    candidate = "\n".join((*lines, sigma_line))
    if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
        lines.append(sigma_line)
    if sense_line:
        lines.append(sense_line)
    quiet_line: Optional[str] = None
    quiet_line_added = False
    if quiet_sequences and active is not None:
        quiet_line = _quiet_notice_line(active.microsector, quiet_sequences)
        if quiet_line:
            lines.append(quiet_line)
            quiet_line_added = True
    gradient_line = _truncate_line(_gradient_line(window_metrics))
    silence_line = None if quiet_line_added else _silence_event_meter(active.microsector)
    if silence_line:
        candidate = "\n".join((*lines, silence_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(silence_line)
    brake_line = _brake_event_meter(active.microsector)
    if brake_line:
        lines.append(brake_line)
    lines.append(gradient_line)
    if integral_line:
        candidate = "\n".join((*lines, integral_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(integral_line)
    coupling_alert = _coupling_alert_line(
        window_metrics,
        longitudinal_delta=long_component,
        tolerance=tolerance,
    )
    if coupling_alert:
        candidate = "\n".join((*lines, coupling_alert))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(coupling_alert)
    if coherence_line:
        lines.append(coherence_line)
    if nu_wave:
        lines.append(nu_wave)
    if entropy_line:
        candidate = "\n".join((*lines, entropy_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(entropy_line)
    if motor_latency_line:
        candidate = "\n".join((*lines, motor_latency_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(motor_latency_line)
    amc_value = float(getattr(window_metrics, "aero_mechanical_coherence", 0.0))
    if amc_value > 0.0:
        amc_line = _truncate_line(f"C(c/d/a) {amc_value:.2f}")
        candidate = "\n".join((*lines, amc_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(amc_line)
    drift_line = _aero_drift_line(window_metrics.aero_balance_drift)
    if drift_line:
        candidate = "\n".join((*lines, drift_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(drift_line)
    if (
        window_metrics.aero_coherence.high_speed_samples
        or window_metrics.aero_coherence.low_speed_samples
    ):
        aero_line = _truncate_line(
            "Δaero high "
            f"{window_metrics.aero_coherence.high_speed_imbalance:+.2f}"
            f" · low {window_metrics.aero_coherence.low_speed_imbalance:+.2f}"
        )
        candidate = "\n".join((*lines, aero_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(aero_line)
    front_velocity = window_metrics.suspension_velocity_front
    rear_velocity = window_metrics.suspension_velocity_rear
    damper_segments: List[str] = []
    if (
        front_velocity.compression_high_speed_percentage >= PACKING_HIGH_SPEED_THRESHOLD
        and front_velocity.ar_index >= PACKING_AR_THRESHOLD
    ):
        damper_segments.append(
            "F HS "
            f"{front_velocity.compression_high_speed_percentage:.0f}%"
            f" A/R {front_velocity.ar_index:.2f}"
        )
    if (
        rear_velocity.compression_high_speed_percentage >= PACKING_HIGH_SPEED_THRESHOLD
        and rear_velocity.ar_index >= PACKING_AR_THRESHOLD
    ):
        damper_segments.append(
            "R HS "
            f"{rear_velocity.compression_high_speed_percentage:.0f}%"
            f" A/R {rear_velocity.ar_index:.2f}"
        )
    ar_gap = abs(front_velocity.ar_index - rear_velocity.ar_index)
    if (
        ar_gap >= PACKING_ASYMMETRY_GAP
        and max(
            front_velocity.compression_high_speed_percentage,
            rear_velocity.compression_high_speed_percentage,
        )
        >= PACKING_HIGH_SPEED_THRESHOLD * 0.5
    ):
        damper_segments.append(f"ΔA/R {ar_gap:.2f}")
    if damper_segments:
        damper_line = _truncate_line(f"Dampers: {' · '.join(damper_segments)}")
        candidate = "\n".join((*lines, damper_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(damper_line)

    cphi_line = _cphi_health_line(window_metrics.cphi)
    if cphi_line:
        candidate = "\n".join((*lines, cphi_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(cphi_line)

    brake_headroom_line = _brake_headroom_line(window_metrics.brake_headroom)
    if brake_headroom_line:
        candidate = "\n".join((*lines, brake_headroom_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(brake_headroom_line)
    thermal_lines = _thermal_dispersion_lines(active.microsector)
    for thermal_line in thermal_lines:
        candidate = "\n".join((*lines, thermal_line))
        if len(candidate.encode("utf8")) <= PAYLOAD_LIMIT:
            lines.append(thermal_line)
    return _ensure_limit("\n".join(lines))


def _format_mu_metric(
    label: str,
    value: float,
    *,
    highlight: bool,
    tolerance: float,
    formatter: str,
) -> str:
    marker = "⚠️" if highlight and abs(value) > tolerance else ""
    return f"{label}{marker} {formatter.format(value)}"


def _format_mu_balance_segments(
    *,
    mu_balance: float,
    mu_sym_front: float,
    mu_sym_rear: float,
    tolerance: float,
) -> List[str]:
    tolerance = max(0.0, float(tolerance))
    display_threshold = max(0.01, tolerance * 0.5)
    metrics = [
        ("μβ", mu_balance, abs(mu_balance)),
        ("μΦF", mu_sym_front, abs(mu_sym_front)),
        ("μΦR", mu_sym_rear, abs(mu_sym_rear)),
    ]
    priority_label, priority_value, priority_severity = max(
        metrics, key=lambda item: item[2]
    )
    segments: List[str] = []
    for label, value, severity in metrics:
        if label != priority_label and label != "μβ" and severity < display_threshold:
            continue
        segments.append(
            _format_mu_metric(
                label,
                value,
                highlight=(label == priority_label),
                tolerance=tolerance,
                formatter="{:+.2f}",
            )
        )
    return segments


def _resolve_mu_symmetry(
    symmetry_map: Optional[Mapping[str, Mapping[str, float]]],
) -> Tuple[float, float]:
    if not isinstance(symmetry_map, Mapping):
        return 0.0, 0.0
    window_symmetry = symmetry_map.get("window")
    if not isinstance(window_symmetry, Mapping):
        return 0.0, 0.0
    try:
        front = float(window_symmetry.get("front", 0.0))
    except (TypeError, ValueError):
        front = 0.0
    try:
        rear = float(window_symmetry.get("rear", 0.0))
    except (TypeError, ValueError):
        rear = 0.0
    if not math.isfinite(front):
        front = 0.0
    if not math.isfinite(rear):
        rear = 0.0
    return front, rear


def _format_aero_drift_mu_segments(
    payload: AeroBalanceDriftBin,
    *,
    tolerance: float,
) -> List[str]:
    tolerance = max(0.0, float(tolerance))
    mu_balance = float(getattr(payload, "mu_balance", 0.0))
    mu_sym_front = float(getattr(payload, "mu_symmetry_front", 0.0))
    mu_sym_rear = float(getattr(payload, "mu_symmetry_rear", 0.0))
    mu_delta = float(getattr(payload, "mu_delta", 0.0))
    mu_ratio = float(getattr(payload, "mu_ratio", 0.0))
    if not math.isfinite(mu_ratio):
        mu_ratio = 0.0
    balance_severity = max(
        abs(mu_balance), abs(mu_sym_front), abs(mu_sym_rear)
    )
    ratio_deviation = abs(mu_ratio - 1.0)
    delta_severity = max(abs(mu_delta), ratio_deviation)
    balance_available = balance_severity > 0.0
    delta_available = delta_severity > 0.0
    use_balance = balance_available and (balance_severity >= delta_severity or not delta_available)
    segments: List[str] = []
    if use_balance:
        segments.extend(
            _format_mu_balance_segments(
                mu_balance=mu_balance,
                mu_sym_front=mu_sym_front,
                mu_sym_rear=mu_sym_rear,
                tolerance=tolerance,
            )
        )
    else:
        priority_label = "μΔ" if abs(mu_delta) >= ratio_deviation else "μƒ/μr"
        segments.append(
            _format_mu_metric(
                "μΔ",
                mu_delta,
                highlight=priority_label == "μΔ",
                tolerance=tolerance,
                formatter="{:+.2f}",
            )
        )
        if mu_ratio:
            segments.append(
                _format_mu_metric(
                    "μƒ/μr",
                    mu_ratio,
                    highlight=priority_label == "μƒ/μr",
                    tolerance=tolerance,
                    formatter="{:.2f}",
                )
            )
    return segments


def _aero_drift_line(aero_drift: Optional[AeroBalanceDrift]) -> Optional[str]:
    if not isinstance(aero_drift, AeroBalanceDrift):
        return None
    candidate = aero_drift.dominant_bin()
    if candidate is None:
        return None
    band_label, direction, payload = candidate
    tolerance = getattr(aero_drift, "mu_tolerance", 0.04)
    mu_segments = _format_aero_drift_mu_segments(payload, tolerance=tolerance)
    mu_block = " ".join(mu_segments) if mu_segments else "μΔ {:+.2f}".format(payload.mu_delta)
    return _truncate_line(
        "Aero drift "
        f"{band_label}: {mu_block} "
        f"rake {payload.rake_deg:+.2f}° → {direction}"
    )


def _gradient_line(window_metrics: WindowMetrics) -> str:
    if window_metrics.frequency_label:
        frequency_segment = (
            f"{window_metrics.frequency_label} · ν_exc {window_metrics.nu_exc:.2f}Hz"
        )
    else:
        frequency_segment = (
            f"ν_f {window_metrics.nu_f:.2f}Hz/ν_exc {window_metrics.nu_exc:.2f}Hz"
        )
    mu_sym_front, mu_sym_rear = _resolve_mu_symmetry(
        getattr(window_metrics, "mu_symmetry", {})
    )
    mu_tolerance = getattr(
        getattr(window_metrics, "aero_balance_drift", None), "mu_tolerance", 0.04
    )
    balance_segments = _format_mu_balance_segments(
        mu_balance=float(getattr(window_metrics, "mu_balance", 0.0)),
        mu_sym_front=mu_sym_front,
        mu_sym_rear=mu_sym_rear,
        tolerance=mu_tolerance,
    )
    balance_block = "".join(f" · {segment}" for segment in balance_segments)
    return (
        f"Stability {window_metrics.si:.2f} · ∇Coupl {window_metrics.d_nfr_couple:+.2f}"
        f" · ∇Res {window_metrics.d_nfr_res:+.2f} · ∇Flat {window_metrics.d_nfr_flat:+.2f}"
        f" · C(t) {window_metrics.coherence_index:.2f}"
        f" · τmot {window_metrics.motor_latency_ms:.0f}ms"
        f" · {frequency_segment}"
        f" · ρ {window_metrics.rho:.2f} · θ {window_metrics.phase_lag:+.2f}rad"
        f" · Siφ {window_metrics.phase_alignment:+.2f}"
        f" · Φsync {window_metrics.phase_synchrony_index:.2f}"
        f" · UDR {window_metrics.useful_dissonance_ratio:.2f}"
        f" · Support {window_metrics.support_effective:+.2f}"
        f" · Load {window_metrics.load_support_ratio:.4f}"
        f" · μF {window_metrics.mu_usage_front_ratio:.2f}"
        f" · μR {window_metrics.mu_usage_rear_ratio:.2f}"
        f"{balance_block}"
    )


def _coupling_alert_line(
    window_metrics: WindowMetrics,
    *,
    longitudinal_delta: float,
    tolerance: float,
) -> Optional[str]:
    try:
        brake_corr = float(
            getattr(window_metrics, "brake_longitudinal_correlation", 0.0)
        )
    except (TypeError, ValueError):
        brake_corr = 0.0
    try:
        throttle_corr = float(
            getattr(window_metrics, "throttle_longitudinal_correlation", 0.0)
        )
    except (TypeError, ValueError):
        throttle_corr = 0.0
    if not math.isfinite(brake_corr):
        brake_corr = 0.0
    if not math.isfinite(throttle_corr):
        throttle_corr = 0.0
    low_segments: List[str] = []
    correlation_threshold = 0.35
    if brake_corr < correlation_threshold:
        low_segments.append(f"brake {brake_corr:.2f}")
    if throttle_corr < correlation_threshold:
        low_segments.append(f"throttle {throttle_corr:.2f}")
    if not low_segments:
        return None
    try:
        longitudinal_value = abs(float(longitudinal_delta))
    except (TypeError, ValueError):
        longitudinal_value = 0.0
    if not math.isfinite(longitudinal_value):
        longitudinal_value = 0.0
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError):
        tolerance_value = 0.0
    if not math.isfinite(tolerance_value):
        tolerance_value = 0.0
    longitudinal_threshold = max(0.3, tolerance_value * 0.75)
    if longitudinal_value <= longitudinal_threshold:
        return None
    payload = "⚠️ Coupl proj ∇NFR∥ " + " · ".join(
        (*low_segments, f"|∇∥| {longitudinal_value:.2f}")
    )
    return _truncate_line(payload)


def _motor_latency_line(window_metrics: WindowMetrics) -> Optional[str]:
    global_latency = float(getattr(window_metrics, "motor_latency_ms", 0.0))
    if not math.isfinite(global_latency):
        global_latency = 0.0
    phase_map = getattr(window_metrics, "phase_motor_latency_ms", {}) or {}
    measured_segments: list[str] = []
    for phase_label, target in MOTOR_LATENCY_TARGETS_MS.items():
        value = phase_map.get(phase_label)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        measured_segments.append(f"{phase_label[0].upper()} {numeric:.0f}/{target:.0f}ms")
    target_segment = "obj " + " ".join(
        f"{phase[0].upper()}<{limit:.0f}ms" for phase, limit in MOTOR_LATENCY_TARGETS_MS.items()
    )
    segments = [f"τmot {global_latency:.0f}ms"]
    if measured_segments:
        segments.extend(measured_segments)
    segments.append(target_segment)
    return _truncate_line(" · ".join(segments))


def _cphi_health_line(report: CPHIReport) -> Optional[str]:
    if not isinstance(report, CPHIReport):
        return None
    segments: List[str] = []
    for suffix in WHEEL_SUFFIXES:
        wheel = report.get(suffix)
        if wheel is None:
            continue
        value = float(wheel.value)
        if math.isfinite(value):
            value_label = f"{value:.2f}"
        else:
            value_label = "--"
        status = report.classification(value)
        if status == "green":
            marker = "🟢"
            if report.is_optimal(value):
                marker += "+"
        elif status == "amber":
            marker = "🟠"
        elif status == "red":
            marker = "🔴"
        else:
            marker = "⬜"
        label = WHEEL_LABELS.get(suffix, suffix.upper())
        segments.append(f"{marker}{label} {value_label}")
    if not segments:
        return None
    return _truncate_line("CPHI " + " ".join(segments))


def _format_dispersion_line(
    measures: Mapping[str, object],
    mean_keys: Mapping[str, str],
    std_keys: Mapping[str, str],
    *,
    prefix: str,
    mean_format: str,
    std_format: str,
) -> Optional[str]:
    segments: List[str] = []
    for suffix in WHEEL_SUFFIXES:
        mean_key = mean_keys.get(suffix)
        std_key = std_keys.get(suffix)
        if mean_key is None or std_key is None:
            continue
        mean_raw = measures.get(mean_key)
        std_raw = measures.get(std_key)
        if mean_raw is None or std_raw is None:
            continue
        try:
            mean_value = float(mean_raw)
            std_value = float(std_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(mean_value) or not math.isfinite(std_value):
            continue
        label = WHEEL_LABELS.get(suffix, suffix.upper())
        segments.append(
            f"{label} {mean_format.format(mean_value)}±{std_format.format(std_value)}"
        )
    if not segments:
        return None
    return _truncate_line(f"{prefix} {' · '.join(segments)}")


def _thermal_dispersion_lines(microsector: Microsector) -> Tuple[str, ...]:
    measures = getattr(microsector, "filtered_measures", {}) or {}
    if not isinstance(measures, Mapping):
        measures = {}
    lines: List[str] = []
    temp_line = _format_dispersion_line(
        measures,
        TEMPERATURE_MEAN_KEYS,
        TEMPERATURE_STD_KEYS,
        prefix="T°",
        mean_format="{:.1f}",
        std_format="{:.1f}",
    )
    if temp_line:
        lines.append(temp_line)
    else:
        lines.append(_truncate_line("T° no data"))
    brake_temp_line = _format_dispersion_line(
        measures,
        BRAKE_TEMPERATURE_MEAN_KEYS,
        BRAKE_TEMPERATURE_STD_KEYS,
        prefix="Brake T°",
        mean_format="{:.1f}",
        std_format="{:.1f}",
    )
    if brake_temp_line:
        lines.append(brake_temp_line)
    else:
        lines.append(_truncate_line("Brake T° no data"))
    pressure_line = _format_dispersion_line(
        measures,
        PRESSURE_MEAN_KEYS,
        PRESSURE_STD_KEYS,
        prefix="Pbar",
        mean_format="{:.2f}",
        std_format="{:.3f}",
    )
    if pressure_line:
        lines.append(pressure_line)
    else:
        lines.append(_truncate_line("Pbar no data"))
    return tuple(lines)


def _brake_headroom_line(headroom: Optional[BrakeHeadroom]) -> Optional[str]:
    if not isinstance(headroom, BrakeHeadroom):
        return None
    temperature_available = getattr(headroom, "temperature_available", True)
    fade_available = getattr(headroom, "fade_available", True)
    fade_has_value = (
        fade_available
        and math.isfinite(headroom.fade_ratio)
        and headroom.fade_ratio > 0.0
    )
    temp_peak_has_value = (
        temperature_available
        and math.isfinite(headroom.temperature_peak)
        and headroom.temperature_peak > 0.0
    )
    temp_mean_has_value = (
        temperature_available
        and math.isfinite(headroom.temperature_mean)
        and headroom.temperature_mean > 0.0
    )
    ventilation_has_value = (
        temperature_available
        and math.isfinite(headroom.ventilation_index)
        and headroom.ventilation_index > 0.0
    )
    if (
        headroom.value <= 0.0
        and not fade_has_value
        and not temp_peak_has_value
        and not temp_mean_has_value
        and not headroom.ventilation_alert
        and not ventilation_has_value
        and temperature_available
    ):
        return None
    segments = [f"HR {headroom.value:.2f}"]
    if fade_has_value:
        fade_segment = f"fade {headroom.fade_ratio * 100:.0f}%"
        if fade_available and math.isfinite(headroom.fade_slope) and headroom.fade_slope > 0.0:
            fade_segment += f"/{headroom.fade_slope:.2f}m/s³"
        segments.append(fade_segment)
    elif not fade_available:
        segments.append("fade no data")
    if temp_peak_has_value:
        segments.append(f"T°max {headroom.temperature_peak:.0f}°C")
    elif temp_mean_has_value:
        segments.append(f"T°μ {headroom.temperature_mean:.0f}°C")
    elif not temperature_available:
        segments.append("T° no data")
    if headroom.ventilation_alert:
        segments.append(f"vent {headroom.ventilation_alert}")
    elif ventilation_has_value:
        segments.append(f"vent {headroom.ventilation_index:.2f}")
    elif not temperature_available:
        segments.append("vent no data")
    return _truncate_line("Brake " + " · ".join(segments))


def _phase_sparkline(
    microsector: Microsector,
    bundles: Optional[Sequence[object]],
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


def _delta_threshold_meter(
    current: float, target: float, tolerance: float, width: int = 9
) -> str:
    width = max(5, int(width))
    diff = float(current) - float(target)
    limit = float(tolerance)
    if not math.isfinite(limit) or limit <= 0.0:
        limit = max(0.05, abs(target) * 0.1, 0.05)
    ratio = max(-1.0, min(1.0, diff / max(limit, 1e-6)))
    pointer = int(round((ratio + 1.0) * 0.5 * (width - 1)))
    pointer = max(0, min(width - 1, pointer))
    gauge = ["-"] * width
    mid = width // 2
    if 0 <= mid < width:
        gauge[mid] = "|"
    if 0 <= pointer < width:
        gauge[pointer] = "^"
    if pointer == mid:
        gauge[mid] = "^"
    return "[" + "".join(gauge) + f"] {diff:+.2f}"


def _coherence_bar(value: float, width: int = 10) -> str:
    value = max(0.0, min(1.0, float(value)))
    width = max(4, int(width))
    filled = max(0, min(width, int(round(value * width))))
    return "█" * filled + "░" * (width - filled)


def _nu_wave_line(
    bundles: Optional[Sequence[object]], width: int = 12
) -> str:
    if not bundles:
        return ""
    values: List[float] = []
    for bundle in bundles[-width:]:
        nu_value = getattr(bundle, "nu_f_dominant", None)
        if nu_value is None or not math.isfinite(float(nu_value)):
            nu_value = getattr(bundle, "nu_f", 0.0)
        values.append(float(nu_value))
    if not values:
        return ""
    spark = render_sparkline(
        values,
        width=len(values),
        constant_block=SPARKLINE_MIDPOINT_BLOCK,
    )
    return _truncate_line(f"ν_f~{spark} {values[-1]:.2f}Hz")


def _coherence_meter_line(
    value: float,
    series: Optional[Sequence[float]],
    classification: Optional[str] = None,
) -> str:
    reference = value
    if series:
        reference = float(series[-1])
    bar = _coherence_bar(reference)
    suffix = f" · ν_f {classification}" if classification else ""
    return _truncate_line(f"C(t) {value:.2f} {bar}{suffix}")


def _sense_state_line(
    sense_state: Optional[Mapping[str, object]],
    *,
    prefix: str = "Si↺",
    width: int = 12,
) -> str:
    if not sense_state:
        return ""
    series = sense_state.get("memory") or sense_state.get("series") or ()
    try:
        values = [float(value) for value in series][-width:]
    except TypeError:
        values = []
    if not values:
        return ""
    spark = render_sparkline(
        values,
        width=len(values),
        constant_block=SPARKLINE_MIDPOINT_BLOCK,
    )
    average = sense_state.get("average", sum(values) / len(values))
    try:
        avg_value = float(average)
    except (TypeError, ValueError):
        avg_value = sum(values) / len(values)
    return _truncate_line(f"{prefix}{spark} μ {avg_value:.2f}")


def _brake_event_meter(microsector: Microsector) -> Optional[str]:
    events = getattr(microsector, "operator_events", {}) or {}
    silence_payloads = events.get("SILENCIO", ())
    micro_duration = _safe_float(getattr(microsector, "end_time", 0.0)) - _safe_float(
        getattr(microsector, "start_time", 0.0)
    )
    if silence_payloads and micro_duration > 1e-9:
        quiet_duration = sum(
            max(0.0, _safe_float(payload.get("duration")))
            for payload in silence_payloads
            if isinstance(payload, Mapping)
        )
        if quiet_duration / micro_duration >= 0.65:
            return None
    payloads: List[Dict[str, float | str]] = []
    for event_type in ("OZ", "IL"):
        for payload in events.get(event_type, ()):  # type: ignore[assignment]
            if not isinstance(payload, Mapping):
                continue
            threshold = float(payload.get("delta_nfr_threshold", 0.0) or 0.0)
            if threshold <= 0.0:
                continue
            peak = abs(float(payload.get("delta_nfr_peak", 0.0) or 0.0))
            ratio = float(payload.get("delta_nfr_ratio", 0.0) or 0.0)
            if ratio <= 0.0 and threshold > 1e-9 and peak > 0.0:
                ratio = peak / threshold
            if ratio < 1.0:
                continue
            surface_label = payload.get("surface_label")
            if not isinstance(surface_label, str) or not surface_label:
                surface = payload.get("surface")
                if isinstance(surface, Mapping):
                    label_value = surface.get("label")
                    if isinstance(label_value, str):
                        surface_label = label_value
            label = payload.get("name")
            if not isinstance(label, str) or not label:
                label = canonical_operator_label(event_type)
            payloads.append(
                {
                    "type": event_type,
                    "label": label,
                    "ratio": ratio,
                    "peak": peak,
                    "threshold": threshold,
                    "surface": surface_label or "surface",
                }
            )
    if not payloads:
        return None
    payloads.sort(key=lambda item: float(item["ratio"]), reverse=True)
    segments: List[str] = []
    for entry in payloads[:3]:
        label = entry.get("label") or canonical_operator_label(str(entry.get("type", "")))
        peak = float(entry["peak"])
        threshold = float(entry["threshold"])
        surface = entry["surface"]
        segments.append(f"{label} {peak:.2f}>{threshold:.2f} {surface}")
    remaining = len(payloads) - len(segments)
    if remaining > 0:
        segments.append(f"+{remaining}")
    return _ensure_limit("ΔNFR braking ⚠️ " + " · ".join(segments))


def _silence_event_meter(microsector: Microsector) -> Optional[str]:
    events = getattr(microsector, "operator_events", {}) or {}
    payloads = [
        payload
        for payload in events.get("SILENCIO", ())  # type: ignore[assignment]
        if isinstance(payload, Mapping)
    ]
    if not payloads:
        return None
    micro_duration = _safe_float(getattr(microsector, "end_time", 0.0)) - _safe_float(
        getattr(microsector, "start_time", 0.0)
    )
    quiet_duration = sum(max(0.0, _safe_float(payload.get("duration"))) for payload in payloads)
    if quiet_duration <= 0.0:
        return None
    coverage = 0.0
    if micro_duration > 1e-9:
        coverage = min(1.0, quiet_duration / micro_duration)
    density_values = [
        max(0.0, _safe_float(payload.get("structural_density_mean"))) for payload in payloads
    ]
    density = sum(density_values) / len(density_values) if density_values else 0.0
    load_span = max((_safe_float(payload.get("load_span")) for payload in payloads), default=0.0)
    slack = max((_safe_float(payload.get("slack")) for payload in payloads), default=0.0)
    return _ensure_limit(
        "Silence {:.0f}% ρ̄ {:.2f} load±{:.0f}N σ {:.2f}".format(
            coverage * 100.0,
            density,
            load_span,
            slack,
        )
    )


def _format_quiet_descriptor(sequence: Sequence[int]) -> str:
    if not sequence:
        return ""
    start = sequence[0] + 1
    end = sequence[-1] + 1
    if start == end:
        return f"Corner {start}"
    return f"Corners {start}-{end}"


def _quiet_notice_line(
    microsector: Microsector, sequences: Sequence[Sequence[int]]
) -> Optional[str]:
    index = getattr(microsector, "index", -1)
    for sequence in sequences:
        if index not in sequence:
            continue
        descriptor = _format_quiet_descriptor(sequence)
        coverage, slack, si_variance, epi_abs = microsector_stability_metrics(microsector)
        return _truncate_line(
            f"{descriptor} stable · Leave untouched · silence {coverage * 100.0:.0f}%"
            f" · Siσ {si_variance:.4f} · |dEPI| {epi_abs:.3f} · slack {slack:.2f}"
        )
    return None


def _quiet_summary_line(
    microsectors: Sequence[Microsector], sequences: Sequence[Sequence[int]]
) -> Optional[str]:
    if not sequences:
        return None
    descriptors: List[str] = []
    coverage_values: List[float] = []
    si_values: List[float] = []
    epi_values: List[float] = []
    for sequence in sequences:
        descriptors.append(_format_quiet_descriptor(sequence))
        for index in sequence:
            if index < 0 or index >= len(microsectors):
                continue
            coverage, _, si_variance, epi_abs = microsector_stability_metrics(
                microsectors[index]
            )
            coverage_values.append(coverage)
            si_values.append(si_variance)
            epi_values.append(epi_abs)
    if not descriptors:
        return None
    summary = f"Leave untouched: {', '.join(descriptors)}"
    if coverage_values:
        coverage_avg = sum(coverage_values) / len(coverage_values)
        si_avg = sum(si_values) / len(si_values) if si_values else 0.0
        epi_avg = sum(epi_values) / len(epi_values) if epi_values else 0.0
        summary = (
            f"{summary} · silence μ {coverage_avg * 100.0:.0f}%"
            f" · Siσ μ {si_avg:.4f} · |dEPI| μ {epi_avg:.3f}"
        )
    return _truncate_line(summary)


def _render_page_b(
    bundle,
    resonance: Mapping[str, ModalAnalysis],
    bundles: Optional[Sequence[object]] = None,
    coherence_series: Optional[Sequence[float]] = None,
    frequency_classification: str = "",
    coherence_index: Optional[float] = None,
) -> str:
    node_values = _nodal_delta_map(bundle)
    lines: List[str] = []
    frequency_label = getattr(bundle, "nu_f_label", "")
    classification = frequency_classification or getattr(
        bundle, "nu_f_classification", ""
    )
    if frequency_label or classification:
        header_parts = []
        if frequency_label:
            header_parts.append(frequency_label)
        if classification:
            header_parts.append(f"ν_f {classification}")
        lines.append(_truncate_line(" · ".join(header_parts)))
    value = (
        float(coherence_index)
        if coherence_index is not None
        else float(getattr(bundle, "coherence_index", 0.0))
    )
    coherence_line = _coherence_meter_line(value, coherence_series)
    if coherence_line:
        lines.append(coherence_line)
    nu_wave = _nu_wave_line(bundles)
    if nu_wave:
        lines.append(nu_wave)
    if not node_values:
        lines.append("ΔNFR nodal")
        lines.append("Insufficient data")
        return _ensure_limit("\n".join(lines))
    ordered = sorted(node_values.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    max_mag = max(abs(value) for _, value in ordered) or 1.0
    leader = ordered[0][0] if ordered else "--"
    lines.append(f"ΔNFR nodal · Leader → {leader}")
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


def _format_risks(parameters: Sequence[str]) -> str:
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
def _hint_float(hints: Optional[Mapping[str, object]], key: str, default: float) -> float:
    if not isinstance(hints, Mapping):
        return default
    candidate = hints.get(key)
    value = _safe_float(candidate, None)
    if value is None:
        return default
    return value


def _hud_threshold_value(
    thresholds: Optional[Mapping[str, float]], key: str, default: float
) -> float:
    if not isinstance(thresholds, Mapping):
        return default
    value = thresholds.get(key, default)
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _sense_average(
    sense_state: Optional[Mapping[str, object]],
    bundles: Optional[Sequence[object]],
) -> Optional[float]:
    if isinstance(sense_state, Mapping):
        average = _safe_float(sense_state.get("average"), None)
        if average is not None:
            return average
    values: list[float] = []
    if bundles:
        for bundle in bundles:
            value = _safe_float(getattr(bundle, "sense_index", None), None)
            if value is not None:
                values.append(value)
    if values:
        return sum(values) / len(values)
    return None


def _delta_density(bundles: Optional[Sequence[object]]) -> Optional[float]:
    if not bundles:
        return None
    series: list[tuple[float, float]] = []
    for bundle in bundles:
        timestamp = _safe_float(getattr(bundle, "timestamp", None), None)
        delta = _safe_float(getattr(bundle, "delta_nfr", None), None)
        if timestamp is None or delta is None:
            continue
        series.append((timestamp, abs(delta)))
    if len(series) < 2:
        return None
    series.sort(key=lambda entry: entry[0])
    total = 0.0
    for index, (timestamp, magnitude) in enumerate(series):
        if index + 1 < len(series):
            next_timestamp = series[index + 1][0]
            dt = next_timestamp - timestamp
        elif index > 0:
            dt = timestamp - series[index - 1][0]
        else:
            dt = 0.0
        if not math.isfinite(dt) or dt <= 0.0:
            dt = 1e-3
        total += magnitude * dt
    duration = series[-1][0] - series[0][0]
    if not math.isfinite(duration) or duration <= 0.0:
        duration = 1.0
    return total / duration


def _entropy_indicator_line(
    window_metrics: WindowMetrics, thresholds: Optional[Mapping[str, float]]
) -> Optional[str]:
    delta_entropy = _safe_float(getattr(window_metrics, "delta_nfr_entropy", None), None)
    node_entropy = _safe_float(getattr(window_metrics, "node_entropy", None), None)
    if delta_entropy is None or node_entropy is None:
        return None

    delta_green = _hud_threshold_value(thresholds, "delta_entropy_green", 0.65)
    delta_amber = _hud_threshold_value(thresholds, "delta_entropy_amber", 0.45)
    node_green = _hud_threshold_value(thresholds, "node_entropy_green", 0.6)
    node_amber = _hud_threshold_value(thresholds, "node_entropy_amber", 0.4)
    if delta_green < delta_amber:
        delta_green = delta_amber
    if node_green < node_amber:
        node_green = node_amber

    def _classify(value: float, amber: float, green: float) -> int:
        if value >= green:
            return 2
        if value >= amber:
            return 1
        return 0

    delta_rank = _classify(delta_entropy, delta_amber, delta_green)
    node_rank = _classify(node_entropy, node_amber, node_green)
    severity = min(delta_rank, node_rank)
    marker = {2: "🟢", 1: "🟠", 0: "🔴"}.get(severity, "⬜")
    coherence_value = _safe_float(getattr(window_metrics, "coherence_index", None), 0.0) or 0.0
    payload = (
        f"ENTRY Δ {marker}{delta_entropy:.2f} · nod {node_entropy:.2f}"
        f" · C(t) {coherence_value:.2f}"
    )
    return _truncate_line(payload)


def _lap_integral_series(
    records: Sequence[TelemetryRecord], bundles: Sequence[object]
) -> Tuple[float, ...]:
    if not records or not bundles:
        return ()
    limit = min(len(records), len(bundles))
    if limit < 2:
        return ()

    integrals: Dict[str, float] = {}
    order: List[str] = []
    prev_record = records[0]
    prev_bundle = bundles[0]
    prev_structural = _safe_float(getattr(prev_record, "structural_timestamp", None), None)
    prev_timestamp = _safe_float(getattr(prev_record, "timestamp", None), 0.0) or 0.0

    for index in range(1, limit):
        record = records[index]
        structural_ts = _safe_float(getattr(record, "structural_timestamp", None), None)
        timestamp = _safe_float(getattr(record, "timestamp", None), prev_timestamp) or prev_timestamp
        if structural_ts is not None and prev_structural is not None:
            dt = structural_ts - prev_structural
        else:
            dt = timestamp - prev_timestamp
        if dt < 0.0:
            dt = 0.0
        lap_label = getattr(prev_record, "lap", None)
        if lap_label is not None and dt > 0.0:
            lap_key = str(lap_label)
            magnitude = _safe_float(getattr(prev_bundle, "delta_nfr", None), 0.0) or 0.0
            magnitude = abs(magnitude)
            if magnitude > 0.0:
                if lap_key not in integrals:
                    integrals[lap_key] = 0.0
                    order.append(lap_key)
                integrals[lap_key] += magnitude * dt
        prev_record = record
        if index < len(bundles):
            prev_bundle = bundles[index]
        prev_timestamp = timestamp
        prev_structural = structural_ts if structural_ts is not None else None

    values = [integrals[key] for key in order if integrals.get(key, 0.0) > 1e-9]
    return tuple(values)


def _integral_cov_line(
    cov_value: Optional[float],
    lap_count: int,
    thresholds: Optional[Mapping[str, float]],
) -> Optional[str]:
    if cov_value is None:
        return None
    green_limit = _hud_threshold_value(thresholds, "integral_cov_green", 0.25)
    amber_limit = _hud_threshold_value(thresholds, "integral_cov_amber", 0.45)
    if amber_limit < green_limit:
        amber_limit = green_limit

    if cov_value <= green_limit:
        severity = 2
    elif cov_value <= amber_limit:
        severity = 1
    else:
        severity = 0
    marker = {2: "🟢", 1: "🟠", 0: "🔴"}.get(severity, "⬜")
    payload = f"CoV ∫|Δ| {marker}{cov_value:.2f} · n {lap_count}"
    return _truncate_line(payload)


def _format_check_entry(
    label: str,
    value: float,
    target: float,
    *,
    threshold_type: str,
) -> str:
    if threshold_type == "min":
        ok = value >= target
        comparator = "≥"
    else:
        ok = value <= target
        comparator = "≤"
    status = "✅" if ok else "⚠️"
    return f"{label}{_format_value(value)}{comparator}{_format_value(target)}{status}"


def _format_value(value: float) -> str:
    formatted = f"{value:.2f}"
    if formatted.startswith("-0") and len(formatted) >= 3:
        formatted = "-" + formatted[2:]
    elif formatted.startswith("0") and len(formatted) >= 2 and formatted[1] == ".":
        formatted = formatted[1:]
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


def _operational_checklist_line(
    bundles: Optional[Sequence[object]],
    window_metrics: Optional[WindowMetrics],
    objectives: Optional[RuleProfileObjectives],
    session_hints: Optional[Mapping[str, object]],
    *,
    sense_state: Optional[Mapping[str, object]] = None,
) -> Optional[str]:
    entries: list[str] = []

    sense_target = _safe_float(getattr(objectives, "target_sense_index", None), 0.75)
    sense_value = _sense_average(sense_state, bundles)
    if sense_value is not None and sense_target is not None:
        entries.append(_format_check_entry("Si", sense_value, sense_target, threshold_type="min"))

    delta_reference = _hint_float(session_hints, "delta_reference", 6.0)
    delta_density = _delta_density(bundles)
    if delta_density is not None and delta_reference is not None:
        entries.append(
            _format_check_entry("Δ∫", delta_density, delta_reference, threshold_type="max")
        )

    headroom_target = _safe_float(
        getattr(objectives, "target_brake_headroom", None),
        0.4,
    )
    headroom_value = None
    if window_metrics is not None:
        brake_headroom = getattr(window_metrics, "brake_headroom", None)
        headroom_value = _safe_float(getattr(brake_headroom, "value", None), None)
    if headroom_value is not None and headroom_target is not None:
        entries.append(
            _format_check_entry("Hd", headroom_value, headroom_target, threshold_type="min")
        )

    aero_target = _hint_float(session_hints, "aero_reference", 0.12)
    aero_value = None
    if window_metrics is not None:
        aero_coherence = getattr(window_metrics, "aero_coherence", None)
        aero_value = _safe_float(getattr(aero_coherence, "high_speed_imbalance", None), None)
    if aero_value is None and window_metrics is not None:
        aero_drift = getattr(window_metrics, "aero_balance_drift", None)
        if aero_drift is not None:
            high_speed = getattr(aero_drift, "high_speed", None)
            if high_speed is not None:
                aero_value = _safe_float(getattr(high_speed, "mu_delta", None), None)
    if aero_value is not None and aero_target is not None:
        entries.append(
            _format_check_entry("Δμ", abs(aero_value), aero_target, threshold_type="max")
        )

    if not entries:
        return None
    return _truncate_line("Checklist " + " ".join(entries))


def _render_page_c(
    phase_hint: Optional[str],
    plan: Optional[SetupPlan],
    thresholds: ThresholdProfile,
    active: Optional[ActivePhase],
    *,
    sense_state: Optional[Mapping[str, object]] = None,
    microsectors: Optional[Sequence[Microsector]] = None,
    quiet_sequences: Optional[Sequence[Sequence[int]]] = None,
    session_messages: Optional[Sequence[str]] = None,
    window_metrics: Optional[WindowMetrics] = None,
    objectives: Optional[RuleProfileObjectives] = None,
    session_hints: Optional[Mapping[str, object]] = None,
    bundles: Optional[Sequence[object]] = None,
) -> str:
    lines: List[str] = []
    sense_line = _sense_state_line(sense_state, prefix="Si plan ")
    if sense_line:
        lines.append(sense_line)
    checklist_line = _operational_checklist_line(
        bundles,
        window_metrics,
        objectives,
        session_hints,
        sense_state=sense_state,
    )
    if checklist_line:
        lines.append(checklist_line)
    if quiet_sequences and microsectors:
        summary_line = _quiet_summary_line(microsectors, quiet_sequences)
        if summary_line:
            lines.append(summary_line)
    if session_messages:
        for message in session_messages:
            truncated = _truncate_line(message)
            if truncated:
                lines.append(truncated)
    if active and phase_hint:
        lines.append(_truncate_line(f"Hint {phase_hint}"))
    elif active:
        tolerance = thresholds.tolerance_for_phase(active.phase)
        phase_label = HUD_PHASE_LABELS.get(active.phase, active.phase)
        lines.append(
            _truncate_line(
                f"{phase_label} target hint: {active.goal.target_delta_nfr:+.2f} ±{tolerance:.2f}"
                if active.goal
                else f"{phase_label} profile ±{tolerance:.2f}"
            )
        )
    summary_lines: Sequence[str] = ()
    post_plan_lines: List[str] = []
    if plan:
        summary_map = getattr(plan, "phase_axis_summary", {}) or {}
        suggestions = tuple(getattr(plan, "phase_axis_suggestions", ()))
        if (not summary_map or not suggestions) and (
            getattr(plan, "phase_axis_targets", None) or getattr(plan, "phase_axis_weights", None)
        ):
            computed_summary, computed_suggestions = compute_phase_axis_summary(
                getattr(plan, "phase_axis_targets", {}),
                getattr(plan, "phase_axis_weights", {}),
            )
            if not summary_map:
                summary_map = computed_summary
            if not suggestions:
                suggestions = computed_suggestions
        summary_lines = phase_axis_summary_lines(summary_map)
        for hint in suggestions:
            if hint:
                post_plan_lines.append(_truncate_line(f"→ {hint}"))
        if getattr(plan, "clamped_parameters", ()):  # compatibility with older plans
            risks = _format_risks(plan.clamped_parameters)
            if risks:
                post_plan_lines.append(_truncate_line(f"risks: {risks}"))
        dsi_line = _format_sensitivities(plan.sensitivities.get("sense_index", {}))
        if dsi_line:
            post_plan_lines.append(_truncate_line(f"dSi {dsi_line}"))
        sci_value = getattr(plan, "sci", None)
        try:
            sci_float = float(sci_value) if sci_value is not None else None
        except (TypeError, ValueError):
            sci_float = None
        if sci_float is not None:
            post_plan_lines.append(_truncate_line(f"SCI {sci_float:.3f}"))
        breakdown_mapping = (
            getattr(plan, "sci_breakdown", None)
            or getattr(plan, "objective_breakdown", None)
            or {}
        )
        if breakdown_mapping:
            order = ("sense", "delta", "udr", "bottoming", "aero", "cphi")
            parts: List[str] = []
            for key in order:
                if key in breakdown_mapping:
                    value = breakdown_mapping[key]
                    try:
                        parts.append(f"{key} {float(value):.3f}")
                    except (TypeError, ValueError):
                        parts.append(f"{key} {value}")
            for key in sorted(breakdown_mapping):
                if key in order:
                    continue
                value = breakdown_mapping[key]
                try:
                    parts.append(f"{key} {float(value):.3f}")
                except (TypeError, ValueError):
                    parts.append(f"{key} {value}")
            if parts:
                post_plan_lines.append(_truncate_line("SCI → " + " · ".join(parts)))
        aero_guidance = getattr(plan, "aero_guidance", "")
        if aero_guidance:
            post_plan_lines.append(_truncate_line(f"Aero {aero_guidance}"))
        aero_metrics = getattr(plan, "aero_metrics", {}) or {}
        amc_value = aero_metrics.get("aero_mechanical_coherence")
        if amc_value is None:
            amc_value = getattr(plan, "aero_mechanical_coherence", None)
        try:
            amc_float = float(amc_value) if amc_value is not None else None
        except (TypeError, ValueError):
            amc_float = None
        if amc_float is not None:
            post_plan_lines.append(_truncate_line(f"C(c/d/a) {amc_float:.2f}"))
        high_imbalance = aero_metrics.get("high_speed_imbalance")
        low_imbalance = aero_metrics.get("low_speed_imbalance")
        if high_imbalance is not None and low_imbalance is not None:
            post_plan_lines.append(
                _truncate_line(
                    f"Δaero high {high_imbalance:+.2f} · low {low_imbalance:+.2f}"
                )
            )
    if summary_lines:
        lines.append("ΔNFR phase map")
        for summary_line in summary_lines:
            lines.append(_truncate_line(summary_line))
    if plan and plan.changes:
        for change in plan.changes[:3]:
            delta = f"{change.delta:+.2f}"
            effect = change.expected_effect.strip()
            lines.append(_truncate_line(f"{change.parameter}: {delta} → {effect}"))
    else:
        lines.append("Plan in progress…")
    lines.extend(post_plan_lines)
    return _ensure_limit("\n".join(lines))


def _render_page_d(
    top_changes: Sequence[SetupChange],
    status: MacroStatus,
) -> str:
    lines: List[str] = ["Apply recommendations"]
    if top_changes:
        for index, change in enumerate(top_changes, start=1):
            descriptor = _truncate_line(
                f"{index}. {change.parameter}: {change.delta:+.2f}"
            )
            lines.append(descriptor)
    else:
        lines.append("No recommendations available")

    if status.next_change:
        change = status.next_change
        lines.append(
            _truncate_line(f"Next → {change.parameter} {change.delta:+.2f}")
        )
    else:
        lines.append("Next → --")

    if status.queue_size:
        lines.append(f"Macro queue {status.queue_size}")

    lines.append("APPLY NEXT (Shift+Click)")

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


def _dominant_peak(
    resonance: Mapping[str, ModalAnalysis]
) -> Optional[Tuple[str, ModalPeak]]:
    best: Optional[Tuple[str, ModalPeak]] = None
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
        layout: Optional[ButtonLayout] = None,
        fusion: Optional[TelemetryFusion] = None,
        hud: Optional[TelemetryHUD] = None,
        telemetry_buffer_size: Optional[int] = None,
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
        self._macro_queue: Optional[MacroQueue] = None
        self._menu_open = False
        self._menu_open_last_seen = 0.0
        self._car_stopped = True
        self._next_change_index = 0
        self._last_plan_signature: Tuple[Tuple[str, float], ...] = ()
        if telemetry_buffer_size is not None:
            try:
                numeric = int(telemetry_buffer_size)
            except (TypeError, ValueError):
                numeric = 0
            telemetry_buffer_size = numeric if numeric > 0 else None
        self.telemetry_buffer_size = telemetry_buffer_size

    APPLY_TRIGGER_FLAG = 0x10
    STOPPED_SPEED_THRESHOLD = 1.0

    def run(self) -> str:
        outsim: Optional[OutSimUDPClient] = None
        outgauge: Optional[OutGaugeUDPClient] = None
        insim: Optional[InSimClient] = None
        overlay: Optional[OverlayManager] = None
        last_render = 0.0
        max_buffer = self.telemetry_buffer_size
        outsim_backlog: Deque[OutSimPacket] = deque(maxlen=max_buffer)
        outgauge_backlog: Deque[OutGaugePacket] = deque(maxlen=max_buffer)
        outsim_loss_events = 0
        outgauge_loss_events = 0
        try:
            outsim = OutSimUDPClient(
                host=self.host,
                port=self.outsim_port,
                buffer_size=max_buffer,
            )
            outgauge = OutGaugeUDPClient(
                host=self.host,
                port=self.outgauge_port,
                buffer_size=max_buffer,
            )
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
                    outsim_backlog.extend(outsim.drain_ready())
                    if not outsim_backlog:
                        packet = outsim.recv()
                        if packet is not None:
                            outsim_backlog.append(packet)
                            outsim_backlog.extend(outsim.drain_ready())

                    outgauge_backlog.extend(outgauge.drain_ready())
                    if not outgauge_backlog:
                        packet = outgauge.recv()
                        if packet is not None:
                            outgauge_backlog.append(packet)
                            outgauge_backlog.extend(outgauge.drain_ready())

                    processed = False
                    while outsim_backlog and outgauge_backlog:
                        processed = True
                        outsim_packet = outsim_backlog.popleft()
                        outgauge_packet = outgauge_backlog.popleft()
                        record = self.fusion.fuse(outsim_packet, outgauge_packet)
                        self.hud.append(record)
                        self._update_vehicle_state(record)
                        self._infer_menu_state(outgauge_packet)
                        self._pager.update(self.hud.pages())
                        self._refresh_macro_status()
                    if not processed:
                        sleep(self._idle_backoff)

                    new_outsim_losses = outsim.statistics.get("loss_events", 0)
                    if new_outsim_losses > outsim_loss_events:
                        outsim_loss_events = new_outsim_losses
                        logger.warning(
                            "Packet loss detected on OutSim stream.",
                            extra={
                                "event": "osd.outsim_loss",
                                "loss_events": outsim_loss_events,
                                "host": self.host,
                                "port": self.outsim_port,
                            },
                        )
                    new_outgauge_losses = outgauge.statistics.get("loss_events", 0)
                    if new_outgauge_losses > outgauge_loss_events:
                        outgauge_loss_events = new_outgauge_losses
                        logger.warning(
                            "Packet loss detected on OutGauge stream.",
                            extra={
                                "event": "osd.outgauge_loss",
                                "loss_events": outgauge_loss_events,
                                "host": self.host,
                                "port": self.outgauge_port,
                            },
                        )

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

    def _handle_button_event(self, event: Optional[ButtonEvent]) -> None:
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

    def _resolve_next_change(self) -> Optional[SetupChange]:
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

    def _macro_preflight_warnings(self, next_change: Optional[SetupChange]) -> List[str]:
        warnings: List[str] = []
        if next_change is None:
            warnings.append("No changes pending")
        if not self._menu_open:
            warnings.append("Open the pit menu (F12)")
        if not self._car_stopped:
            warnings.append("Stop the car before applying")
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
    microsectors: Optional[Sequence[Microsector]] = None,
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
        rationale_text = "; ".join(unique_rationales or ["Sense Index objective optimisation"])
        effect_text = "; ".join(unique_effects or ["Balanced vehicle improvement"])
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
        aggregated_rationales = unique_rationales or ["Sense Index objective optimisation"]
        aggregated_effects = unique_effects or ["Balanced vehicle improvement"]

    unique_rationales = list(
        dict.fromkeys(aggregated_rationales or ["Sense Index objective optimisation"])
    )
    unique_effects = list(
        dict.fromkeys(aggregated_effects or ["Balanced vehicle improvement"])
    )

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

    summary_map, axis_suggestions = compute_phase_axis_summary(
        axis_target_map, axis_weight_map
    )

    return SetupPlan(
        car_model=car_model,
        session=None,
        sci=float(getattr(plan, "sci", 0.0)),
        changes=tuple(changes),
        rationales=tuple(unique_rationales),
        expected_effects=tuple(unique_effects),
        sensitivities=plan.sensitivities,
        phase_sensitivities=plan.phase_sensitivities,
        clamped_parameters=tuple(clamped),
        phase_axis_targets=axis_target_map,
        phase_axis_weights=axis_weight_map,
        phase_axis_summary=summary_map,
        phase_axis_suggestions=axis_suggestions,
        sci_breakdown=getattr(plan, "sci_breakdown", getattr(plan, "objective_breakdown", {})),
    )


__all__ = [
    "ActivePhase",
    "HUDPager",
    "MacroStatus",
    "OSDController",
    "TelemetryHUD",
]

