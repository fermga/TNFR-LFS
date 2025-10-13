"""Rule-based recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
import math
from statistics import mean
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    SupportsFloat,
    SupportsInt,
    Tuple,
    TYPE_CHECKING,
)

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from tnfr_lfs.core.archetypes import (
    ARCHETYPE_MEDIUM,
    DEFAULT_ARCHETYPE_PHASE_TARGETS,
    PhaseArchetypeTargets,
)
from tnfr_lfs.core.constants import (
    PRESSURE_STD_KEYS,
    TEMPERATURE_MEAN_KEYS,
    TEMPERATURE_STD_KEYS,
    WHEEL_LABELS,
    WHEEL_SUFFIXES,
)
from tnfr_lfs.core.epi_models import EPIBundle
from tnfr_lfs.core.operators import TyreBalanceControlOutput, tyre_balance_controller
from tnfr_lfs.core.operator_detection import canonical_operator_label, silence_event_payloads
from tnfr_lfs.core.phases import LEGACY_PHASE_MAP, expand_phase_alias, phase_family
from tnfr_lfs.math.conversions import _safe_float

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tnfr_lfs.ingestion.offline.profiles import AeroProfile, ProfileManager
from tnfr_lfs.core.segmentation import (
    Goal,
    Microsector,
    detect_quiet_microsector_streaks,
    microsector_stability_metrics,
)


MANUAL_REFERENCES = {
    "braking": "Basic Setup Guide · Optimal Braking [BAS-FRE]",
    "antiroll": "Advanced Setup Guide · Anti-Roll Bars [ADV-ARB]",
    "differential": "Advanced Setup Guide · Differential Configuration [ADV-DIF]",
    "curbs": "Basic Setup Guide · Kerb Usage [BAS-CUR]",
    "ride_height": "Advanced Setup Guide · Ride Heights & Load Distribution [ADV-RDH]",
    "aero": "Basic Setup Guide · Aero Balance [BAS-AER]",
    "driver": "Basic Setup Guide · Consistent Driving [BAS-DRV]",
    "tyre_balance": "Advanced Setup Guide · Pressures & Camber [ADV-TYR]",
    "dampers": "Advanced Setup Guide · Dampers [ADV-DMP]",
    "springs": "Advanced Setup Guide · Spring Stiffness [ADV-SPR]",
}

_ALIGNMENT_ALIGNMENT_GAP = 0.15
_ALIGNMENT_LAG_GAP = 0.3
_ALIGNMENT_THRESHOLD = 0.05
_LAG_THRESHOLD = 0.05
_COHERENCE_THRESHOLD = 0.05
_SYNCHRONY_THRESHOLD = 0.08


_GEOMETRY_PARAMETERS = {
    "front_camber_deg",
    "rear_camber_deg",
    "front_toe_deg",
    "rear_toe_deg",
    "caster_deg",
}


_GEOMETRY_METRIC_NODES: Mapping[str, Mapping[str, str]] = {
    "entry": {
        "phase_alignment": "suspension",
        "phase_synchrony_index": "tyres",
        "phase_lag": "suspension",
        "coherence_index": "tyres",
    },
    "apex": {
        "phase_alignment": "suspension",
        "phase_synchrony_index": "suspension",
        "phase_lag": "suspension",
        "coherence_index": "tyres",
    },
    "exit": {
        "phase_alignment": "suspension",
        "phase_synchrony_index": "tyres",
        "phase_lag": "tyres",
        "coherence_index": "tyres",
    },
}


_AXIS_FOCUS_MAP: Mapping[tuple[str, str], tuple[str, str, Tuple[str, ...]]] = {
    ("entry", "longitudinal"): (
        "Prioritize brake bias (∇NFR∥ projection)",
        "braking",
        ("brake_bias_pct",),
    ),
    ("exit", "longitudinal"): (
        "Reinforce coast locking (∇NFR∥ projection)",
        "differential",
        ("diff_coast_lock",),
    ),
    ("apex", "lateral"): (
        "Tune the anti-roll bars (∇NFR⊥ projection)",
        "antiroll",
        ("front_arb_steps", "rear_arb_steps"),
    ),
    ("entry", "lateral"): (
        "Fine-tune front toe (∇NFR⊥ projection)",
        "tyre_balance",
        ("front_toe_deg",),
    ),
    ("exit", "lateral"): (
        "Fine-tune rear toe (∇NFR⊥ projection)",
        "tyre_balance",
        ("rear_toe_deg",),
    ),
}


NODE_LABELS = {
    "tyres": "Tyres",
    "suspension": "Suspension",
    "chassis": "Chassis",
    "brakes": "Brakes",
    "transmission": "Transmission",
    "track": "Track",
    "driver": "Driver",
}


_BASE_PHASE_ACTIONS: Mapping[str, Mapping[str, Mapping[str, str]]] = {
    "entry": {
        "tyres": {
            "increase": "Increase front toe-out",
            "decrease": "Reduce front toe-out",
        },
        "brakes": {
            "increase": "Shift brake bias forward",
            "decrease": "Shift brake bias rearward",
        },
        "suspension": {
            "increase": "Stiffen front rebound",
            "decrease": "Soften front rebound",
        },
    },
    "apex": {
        "suspension": {
            "increase": "Stiffen the anti-roll bar",
            "decrease": "Soften the anti-roll bar",
        },
        "chassis": {
            "increase": "Increase front preload",
            "decrease": "Reduce front preload",
        },
        "tyres": {
            "increase": "Raise pressure in outside tyres",
            "decrease": "Lower pressure in outside tyres",
        },
    },
    "exit": {
        "transmission": {
            "increase": "Increase power locking",
            "decrease": "Reduce power locking",
        },
        "tyres": {
            "increase": "Increase rear camber",
            "decrease": "Reduce rear camber",
        },
        "suspension": {
            "increase": "Stiffen rear compression",
            "decrease": "Soften rear compression",
        },
    },
}


@dataclass(frozen=True)
class PhaseActionTemplate:
    """Template that translates gradients into actionable setup deltas."""

    metric: str
    parameter: str
    scale: float
    min_value: float
    max_value: float
    message_pattern: str
    step: float
    nodes: Tuple[str, ...] | None = None
    priority_offset: int = 0
    gradient_offset_scale: Tuple[float, float] | None = None
    gradient_offset_limit: float = 0.0
    gradient_deadband: float = 0.0


_PHASE_ACTION_ROADMAP: Mapping[str, Tuple[PhaseActionTemplate, ...]] = {
    "entry": (
        PhaseActionTemplate(
            metric="delta_nfr",
            parameter="brake_bias_pct",
            scale=-1.0,
            min_value=-4.0,
            max_value=4.0,
            message_pattern="{delta:+.1f}% brake bias forward",
            step=0.5,
            gradient_offset_scale=(60.0, 80.0),
            gradient_offset_limit=1.5,
            gradient_deadband=0.001,
        ),
        PhaseActionTemplate(
            metric="delta_nfr_proj_lateral",
            parameter="front_spring_stiffness",
            scale=-900.0,
            min_value=-35.0,
            max_value=35.0,
            message_pattern="{delta:+.1f} N/mm front spring (νf_susp · ∇NFR⊥)",
            step=0.5,
            nodes=("suspension",),
            priority_offset=-2,
        ),
        PhaseActionTemplate(
            metric="nu_f",
            parameter="front_rebound_clicks",
            scale=-0.5,
            min_value=-6.0,
            max_value=6.0,
            message_pattern="{delta:+.0f} front rebound clicks",
            step=1.0,
            nodes=("suspension",),
            priority_offset=1,
        ),
        PhaseActionTemplate(
            metric="nu_f",
            parameter="front_compression_clicks",
            scale=-0.5,
            min_value=-6.0,
            max_value=6.0,
            message_pattern="{delta:+.0f} front compression clicks",
            step=1.0,
            nodes=("suspension",),
            priority_offset=2,
        ),
        PhaseActionTemplate(
            metric="sense_index",
            parameter="front_tyre_pressure",
            scale=20.0,
            min_value=-0.6,
            max_value=0.6,
            message_pattern="{delta:+.1f} psi front axle",
            step=0.1,
            priority_offset=3,
        ),
        PhaseActionTemplate(
            metric="phase_alignment",
            parameter="front_camber_deg",
            scale=-1.1,
            min_value=-1.5,
            max_value=1.5,
            message_pattern="{delta:+.1f}° front camber",
            step=0.1,
            nodes=("suspension",),
            priority_offset=-3,
        ),
        PhaseActionTemplate(
            metric="phase_lag",
            parameter="caster_deg",
            scale=-3.5,
            min_value=-3.0,
            max_value=3.0,
            message_pattern="{delta:+.1f}° caster",
            step=0.25,
            nodes=("suspension",),
            priority_offset=-4,
        ),
        PhaseActionTemplate(
            metric="phase_synchrony_index",
            parameter="front_toe_deg",
            scale=0.4,
            min_value=-0.4,
            max_value=0.4,
            message_pattern="{delta:+.2f}° front toe",
            step=0.05,
            nodes=("tyres",),
            priority_offset=-3,
        ),
        PhaseActionTemplate(
            metric="coherence_index",
            parameter="front_toe_deg",
            scale=-0.45,
            min_value=-0.5,
            max_value=0.5,
            message_pattern="{delta:+.2f}° front toe",
            step=0.05,
            nodes=("tyres",),
            priority_offset=-2,
        ),
    ),
    "apex": (
        PhaseActionTemplate(
            metric="delta_nfr",
            parameter="front_arb_steps",
            scale=-0.6,
            min_value=-5.0,
            max_value=5.0,
            message_pattern="{delta:+.0f} front anti-roll bar steps",
            step=1.0,
        ),
        PhaseActionTemplate(
            metric="delta_nfr_proj_lateral",
            parameter="front_spring_stiffness",
            scale=-900.0,
            min_value=-35.0,
            max_value=35.0,
            message_pattern="{delta:+.1f} N/mm front spring (νf_susp · ∇NFR⊥)",
            step=0.5,
            nodes=("suspension",),
            priority_offset=-2,
        ),
        PhaseActionTemplate(
            metric="delta_nfr",
            parameter="rear_arb_steps",
            scale=0.6,
            min_value=-5.0,
            max_value=5.0,
            message_pattern="{delta:+.0f} rear anti-roll bar steps",
            step=1.0,
            priority_offset=1,
        ),
        PhaseActionTemplate(
            metric="sense_index",
            parameter="front_tyre_pressure",
            scale=18.0,
            min_value=-0.6,
            max_value=0.6,
            message_pattern="{delta:+.1f} psi outside tyres",
            step=0.1,
            priority_offset=2,
        ),
        PhaseActionTemplate(
            metric="nu_f",
            parameter="rear_rebound_clicks",
            scale=-0.5,
            min_value=-6.0,
            max_value=6.0,
            message_pattern="{delta:+.0f} rear rebound clicks",
            step=1.0,
            nodes=("suspension",),
            priority_offset=3,
        ),
        PhaseActionTemplate(
            metric="phase_alignment",
            parameter="front_camber_deg",
            scale=-0.9,
            min_value=-1.2,
            max_value=1.2,
            message_pattern="{delta:+.1f}° front camber",
            step=0.1,
            nodes=("suspension",),
            priority_offset=-3,
        ),
        PhaseActionTemplate(
            metric="phase_alignment",
            parameter="rear_camber_deg",
            scale=-0.8,
            min_value=-1.2,
            max_value=1.2,
            message_pattern="{delta:+.1f}° rear camber",
            step=0.1,
            nodes=("suspension",),
            priority_offset=-2,
        ),
        PhaseActionTemplate(
            metric="phase_lag",
            parameter="caster_deg",
            scale=-2.8,
            min_value=-2.5,
            max_value=2.5,
            message_pattern="{delta:+.1f}° caster",
            step=0.25,
            nodes=("suspension",),
            priority_offset=-4,
        ),
        PhaseActionTemplate(
            metric="coherence_index",
            parameter="front_toe_deg",
            scale=-0.4,
            min_value=-0.5,
            max_value=0.5,
            message_pattern="{delta:+.2f}° front toe",
            step=0.05,
            nodes=("tyres",),
            priority_offset=-3,
        ),
        PhaseActionTemplate(
            metric="coherence_index",
            parameter="rear_toe_deg",
            scale=-0.35,
            min_value=-0.5,
            max_value=0.5,
            message_pattern="{delta:+.2f}° rear toe",
            step=0.05,
            nodes=("tyres",),
            priority_offset=-2,
        ),
    ),
    "exit": (
        PhaseActionTemplate(
            metric="delta_nfr",
            parameter="diff_power_lock",
            scale=-2.0,
            min_value=-20.0,
            max_value=20.0,
            message_pattern="{delta:+.0f}% LSD power",
            step=5.0,
        ),
        PhaseActionTemplate(
            metric="delta_nfr_proj_lateral",
            parameter="rear_spring_stiffness",
            scale=-900.0,
            min_value=-35.0,
            max_value=35.0,
            message_pattern="{delta:+.1f} N/mm rear spring (νf_susp · ∇NFR⊥)",
            step=0.5,
            nodes=("suspension",),
            priority_offset=-2,
        ),
        PhaseActionTemplate(
            metric="delta_nfr",
            parameter="rear_ride_height",
            scale=-0.5,
            min_value=-4.0,
            max_value=4.0,
            message_pattern="{delta:+.1f} mm rear ride height",
            step=0.5,
            priority_offset=1,
        ),
        PhaseActionTemplate(
            metric="sense_index",
            parameter="rear_tyre_pressure",
            scale=20.0,
            min_value=-0.6,
            max_value=0.6,
            message_pattern="{delta:+.1f} psi rear axle",
            step=0.1,
            priority_offset=2,
        ),
        PhaseActionTemplate(
            metric="nu_f",
            parameter="diff_coast_lock",
            scale=-2.0,
            min_value=-20.0,
            max_value=20.0,
            message_pattern="{delta:+.0f}% LSD coast",
            step=5.0,
            nodes=("transmission",),
            priority_offset=3,
        ),
        PhaseActionTemplate(
            metric="sense_index",
            parameter="rear_wing_angle",
            scale=10.0,
            min_value=-3.0,
            max_value=3.0,
            message_pattern="{delta:+.1f}° rear wing",
            step=0.5,
            priority_offset=4,
        ),
        PhaseActionTemplate(
            metric="phase_alignment",
            parameter="rear_camber_deg",
            scale=-1.0,
            min_value=-1.5,
            max_value=1.5,
            message_pattern="{delta:+.1f}° rear camber",
            step=0.1,
            nodes=("suspension",),
            priority_offset=-3,
        ),
        PhaseActionTemplate(
            metric="phase_lag",
            parameter="rear_toe_deg",
            scale=-0.5,
            min_value=-0.5,
            max_value=0.5,
            message_pattern="{delta:+.2f}° rear toe",
            step=0.05,
            nodes=("tyres",),
            priority_offset=-3,
        ),
        PhaseActionTemplate(
            metric="phase_synchrony_index",
            parameter="rear_toe_deg",
            scale=-0.35,
            min_value=-0.4,
            max_value=0.4,
            message_pattern="{delta:+.2f}° rear toe",
            step=0.05,
            nodes=("tyres",),
            priority_offset=-3,
        ),
        PhaseActionTemplate(
            metric="coherence_index",
            parameter="rear_toe_deg",
            scale=-0.4,
            min_value=-0.5,
            max_value=0.5,
            message_pattern="{delta:+.2f}° rear toe",
            step=0.05,
            nodes=("tyres",),
            priority_offset=-2,
        ),
    ),
}

_SPRING_PARAMETERS = {"front_spring_stiffness", "rear_spring_stiffness"}

_OPERATOR_NODE_ACTIONS: Dict[str, Mapping[str, Mapping[str, str]]] = dict(_BASE_PHASE_ACTIONS)
for legacy, phases in LEGACY_PHASE_MAP.items():
    actions = _BASE_PHASE_ACTIONS.get(legacy)
    if not actions:
        continue
    for phase in phases:
        _OPERATOR_NODE_ACTIONS[phase] = actions


@dataclass
class Recommendation:
    """Represents an actionable recommendation."""

    category: str
    message: str
    rationale: str
    priority: int = 0
    parameter: str | None = None
    delta: float | None = None


@dataclass(frozen=True)
class PhaseTargetWindow:
    """Slip and yaw windows associated to a ΔNFR target."""

    target_delta_nfr: float
    slip_lat_window: Tuple[float, float]
    slip_long_window: Tuple[float, float]
    yaw_rate_window: Tuple[float, float]


@dataclass(frozen=True)
class ThresholdProfile:
    """Thresholds tuned per car model and circuit."""

    entry_delta_tolerance: float
    apex_delta_tolerance: float
    exit_delta_tolerance: float
    piano_delta_tolerance: float
    rho_detune_threshold: float
    phase_targets: Mapping[str, PhaseTargetWindow] = field(default_factory=dict)
    phase_weights: Mapping[str, Mapping[str, float] | float] = field(
        default_factory=dict
    )
    archetype_phase_targets: Mapping[str, Mapping[str, PhaseArchetypeTargets]] = field(
        default_factory=dict
    )
    robustness: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    hud_thresholds: Mapping[str, float] = field(default_factory=dict)

    def tolerance_for_phase(self, phase: str) -> float:
        mapping = {
            "entry": self.entry_delta_tolerance,
            "apex": self.apex_delta_tolerance,
            "exit": self.exit_delta_tolerance,
        }
        key = phase_family(phase)
        if key in mapping:
            return mapping[key]
        return mapping.get("entry", self.entry_delta_tolerance)

    def target_for_phase(self, phase: str) -> PhaseTargetWindow | None:
        direct = self.phase_targets.get(phase)
        if direct is not None:
            return direct
        key = phase_family(phase)
        return self.phase_targets.get(key)

    def weights_for_phase(self, phase: str) -> Mapping[str, float] | float:
        profile = self.phase_weights.get(phase)
        if profile is None:
            key = phase_family(phase)
            profile = self.phase_weights.get(key)
        if profile is None:
            profile = self.phase_weights.get("__default__")
        if isinstance(profile, Mapping):
            return MappingProxyType(dict(profile))
        if isinstance(profile, (int, float)):
            return float(profile)
        return MappingProxyType({})

    def archetype_targets_for(
        self, archetype: str
    ) -> Mapping[str, PhaseArchetypeTargets]:
        table = self.archetype_phase_targets.get(archetype)
        if table is None:
            table = self.archetype_phase_targets.get(ARCHETYPE_MEDIUM, {})
        return table

    def hud_threshold(self, key: str, default: float | None = None) -> float | None:
        """Return a HUD threshold override or fall back to the supplied default.

        Callers are expected to provide the baseline HUD defaults (see
        :func:`tnfr_lfs.cli.osd._hud_threshold_value`), so leaving a value
        undefined guarantees we inherit the generic thresholds instead of
        repeating them for every track.
        """
        value = self.hud_thresholds.get(key)
        if value is None:
            return default
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(numeric):
            return default
        return numeric


@dataclass(frozen=True)
class RuleProfileObjectives:
    """Minimal snapshot of profile objectives for rule evaluation."""

    target_delta_nfr: float = 0.0
    target_sense_index: float = 0.75
    target_brake_headroom: float = 0.4


@dataclass(frozen=True)
class RuleContext:
    """Context shared with the rules to build rationales."""

    car_model: str
    track_name: str
    thresholds: ThresholdProfile
    tyre_offsets: Mapping[str, float] = field(default_factory=dict)
    aero_profiles: Mapping[str, "AeroProfile"] = field(default_factory=dict)
    objectives: RuleProfileObjectives = field(default_factory=RuleProfileObjectives)
    session_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    session_hints: Mapping[str, Any] = field(default_factory=dict)

    @property
    def profile_label(self) -> str:
        return f"{self.car_model}/{self.track_name}"


class RecommendationRule(Protocol):
    """Interface implemented by recommendation rules."""

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        ...


def _freeze_phase_targets(targets: Mapping[str, PhaseTargetWindow]) -> Mapping[str, PhaseTargetWindow]:
    return MappingProxyType(dict(targets))


def _freeze_phase_weights(
    weights: Mapping[str, Mapping[str, float] | float]
) -> Mapping[str, Mapping[str, float] | float]:
    frozen: Dict[str, Mapping[str, float] | float] = {}
    for phase, profile in weights.items():
        if isinstance(profile, Mapping):
            frozen[str(phase)] = MappingProxyType(
                {
                    str(node): float(value)
                    for node, value in profile.items()
                    if isinstance(value, (int, float))
                }
            )
        elif isinstance(profile, (int, float)):
            frozen[str(phase)] = float(profile)
    return MappingProxyType(frozen)


def _freeze_metric_thresholds(payload: Mapping[str, object]) -> Mapping[str, float]:
    table: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            table[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return MappingProxyType(table) if table else MappingProxyType({})


def _build_robustness_thresholds(
    payload: Mapping[str, object] | None,
    *,
    defaults: Mapping[str, Mapping[str, float]] | None = None,
) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(payload, Mapping):
        return defaults or MappingProxyType({})
    tables: Dict[str, Mapping[str, float]] = {}
    for scope, values in payload.items():
        if not isinstance(values, Mapping):
            continue
        metrics = _freeze_metric_thresholds(values)
        if metrics:
            tables[str(scope)] = metrics
    if not tables:
        return defaults or MappingProxyType({})
    return MappingProxyType(tables)


def _coerce_session_weights(
    payload: Mapping[str, object] | None,
) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(payload, Mapping):
        return MappingProxyType({})
    weights: Dict[str, Mapping[str, float]] = {}
    for phase, profile in payload.items():
        if not isinstance(profile, Mapping):
            continue
        entry: Dict[str, float] = {}
        for node, value in profile.items():
            try:
                entry[str(node)] = float(value)
            except (TypeError, ValueError):
                continue
        if entry:
            weights[str(phase)] = MappingProxyType(entry)
    return MappingProxyType(weights)


def _coerce_session_hints(payload: Mapping[str, object] | None) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return MappingProxyType({})
    hints: Dict[str, Any] = {}
    for key, value in payload.items():
        label = str(key)
        if isinstance(value, (str, bool)):
            hints[label] = value
            continue
        if isinstance(value, (int, float)):
            hints[label] = float(value)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            entries: list[Any] = []
            for item in value:
                if isinstance(item, (str, bool)):
                    entries.append(item)
                elif isinstance(item, (int, float)):
                    entries.append(float(item))
            hints[label] = tuple(entries)
    return MappingProxyType(hints)


def _session_weight_profile(
    weights: Mapping[str, Mapping[str, float]], phase: str
) -> Mapping[str, float] | None:
    profile = weights.get(phase)
    if profile is not None:
        return profile
    key = phase_family(phase)
    return weights.get(key)


def _session_priority_scale(
    context: RuleContext, phase: str, node: str | None = None
) -> float:
    weights = getattr(context, "session_weights", {}) or {}
    if not isinstance(weights, Mapping):
        return 1.0
    profile = _session_weight_profile(weights, phase)
    if not isinstance(profile, Mapping):
        return 1.0
    factor: float | None = None
    if node is not None:
        candidate = profile.get(node)
        if isinstance(candidate, (int, float)):
            factor = float(candidate)
    if factor is None:
        candidate = profile.get("__default__")
        if isinstance(candidate, (int, float)):
            factor = float(candidate)
    if factor is None:
        return 1.0
    return max(0.4, min(2.5, factor))


def _scale_priority_value(value: int, scale: float) -> int:
    if value == 0 or not math.isfinite(scale) or abs(scale - 1.0) < 1e-9:
        return value
    scaled = int(round(value * scale))
    if value > 0:
        return max(1, scaled)
    if value < 0:
        return min(-1, scaled)
    return 0


def _apply_priority_scale(
    recommendations: Sequence[Recommendation], scale: float
) -> None:
    if not recommendations:
        return
    if abs(scale - 1.0) < 1e-9:
        return
    for recommendation in recommendations:
        recommendation.priority = _scale_priority_value(recommendation.priority, scale)


def _session_coherence_scale(context: RuleContext) -> float:
    weights = getattr(context, "session_weights", {}) or {}
    if not isinstance(weights, Mapping):
        return 1.0
    values: list[float] = []
    for profile in weights.values():
        if not isinstance(profile, Mapping):
            continue
        candidate = profile.get("__default__")
        if isinstance(candidate, (int, float)):
            values.append(float(candidate))
    if not values:
        return 1.0
    return max(0.4, min(2.5, mean(values)))


def _coerce_float(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _freeze_archetype_targets(
    targets: Mapping[str, Mapping[str, PhaseArchetypeTargets]]
) -> Mapping[str, Mapping[str, PhaseArchetypeTargets]]:
    frozen: Dict[str, Mapping[str, PhaseArchetypeTargets]] = {}
    for archetype, phases in targets.items():
        frozen[str(archetype)] = MappingProxyType(dict(phases))
    return MappingProxyType(frozen)


_BASELINE_PHASE_TARGETS = _freeze_phase_targets(
    {
        "entry": PhaseTargetWindow(
            target_delta_nfr=0.4,
            slip_lat_window=(-0.05, 0.05),
            slip_long_window=(-0.04, 0.04),
            yaw_rate_window=(-0.35, 0.35),
        ),
        "apex": PhaseTargetWindow(
            target_delta_nfr=0.2,
            slip_lat_window=(-0.04, 0.04),
            slip_long_window=(-0.03, 0.03),
            yaw_rate_window=(-0.25, 0.25),
        ),
        "exit": PhaseTargetWindow(
            target_delta_nfr=-0.1,
            slip_lat_window=(-0.06, 0.06),
            slip_long_window=(-0.05, 0.05),
            yaw_rate_window=(-0.30, 0.30),
        ),
    }
)


_BASELINE_PHASE_WEIGHTS = _freeze_phase_weights({})


_BASELINE_ARCHETYPE_TARGETS = _freeze_archetype_targets(
    DEFAULT_ARCHETYPE_PHASE_TARGETS
)


_BASELINE_ROBUSTNESS_THRESHOLDS: Mapping[str, Mapping[str, float]] = MappingProxyType({})


def _coerce_window(
    values: Sequence[object] | None, default: Tuple[float, float]
) -> Tuple[float, float]:
    if isinstance(values, Sequence) and len(values) == 2:
        try:
            return float(values[0]), float(values[1])
        except (TypeError, ValueError):
            return default
    return default


def _build_phase_targets(
    payload: Mapping[str, object] | None,
    *,
    defaults: Mapping[str, PhaseTargetWindow],
) -> Mapping[str, PhaseTargetWindow]:
    if not isinstance(payload, Mapping):
        return defaults
    targets: Dict[str, PhaseTargetWindow] = {}
    for phase, values in payload.items():
        if not isinstance(values, Mapping):
            continue
        default = defaults.get(phase, defaults["entry"])
        targets[phase] = PhaseTargetWindow(
            target_delta_nfr=float(values.get("target_delta_nfr", default.target_delta_nfr)),
            slip_lat_window=_coerce_window(
                values.get("slip_lat_window"), default.slip_lat_window
            ),
            slip_long_window=_coerce_window(
                values.get("slip_long_window"), default.slip_long_window
            ),
            yaw_rate_window=_coerce_window(
                values.get("yaw_rate_window"), default.yaw_rate_window
            ),
        )
    if not targets:
        return defaults
    return _freeze_phase_targets(targets)


def _build_phase_weights(
    payload: Mapping[str, object] | None,
    *,
    defaults: Mapping[str, Mapping[str, float] | float],
) -> Mapping[str, Mapping[str, float] | float]:
    if not isinstance(payload, Mapping):
        return defaults
    weights: Dict[str, Mapping[str, float] | float] = {}
    for phase, profile in payload.items():
        if isinstance(profile, Mapping):
            entry: Dict[str, float] = {}
            for node, value in profile.items():
                try:
                    entry[str(node)] = float(value)
                except (TypeError, ValueError):
                    continue
            if entry:
                weights[str(phase)] = MappingProxyType(entry)
        else:
            try:
                weights[str(phase)] = float(profile)
            except (TypeError, ValueError):
                continue
    if not weights:
        return defaults
    return _freeze_phase_weights(weights)


def _profile_from_payload(payload: Mapping[str, object]) -> ThresholdProfile:
    defaults = {
        "entry_delta_tolerance": 1.5,
        "apex_delta_tolerance": 1.0,
        "exit_delta_tolerance": 2.0,
        "piano_delta_tolerance": 2.5,
        "rho_detune_threshold": 0.7,
    }
    phase_targets = _build_phase_targets(
        payload.get("targets"), defaults=_BASELINE_PHASE_TARGETS
    )
    phase_weights = _build_phase_weights(
        payload.get("phase_weights"), defaults=_BASELINE_PHASE_WEIGHTS
    )
    robustness = _build_robustness_thresholds(
        payload.get("robustness"), defaults=_BASELINE_ROBUSTNESS_THRESHOLDS
    )
    hud_thresholds = _freeze_metric_thresholds(payload.get("hud", {}))
    return ThresholdProfile(
        entry_delta_tolerance=float(payload.get("entry_delta_tolerance", defaults["entry_delta_tolerance"])),
        apex_delta_tolerance=float(payload.get("apex_delta_tolerance", defaults["apex_delta_tolerance"])),
        exit_delta_tolerance=float(payload.get("exit_delta_tolerance", defaults["exit_delta_tolerance"])),
        piano_delta_tolerance=float(payload.get("piano_delta_tolerance", defaults["piano_delta_tolerance"])),
        rho_detune_threshold=float(payload.get("rho_detune_threshold", defaults["rho_detune_threshold"])),
        phase_targets=phase_targets,
        phase_weights=phase_weights,
        archetype_phase_targets=_BASELINE_ARCHETYPE_TARGETS,
        robustness=robustness,
        hud_thresholds=hud_thresholds,
    )


def _load_threshold_library_from_resource() -> Mapping[str, Mapping[str, ThresholdProfile]]:
    library: Dict[str, Dict[str, ThresholdProfile]] = {}
    try:
        resource = resources.files("tnfr_lfs.data").joinpath("threshold_profiles.toml")
    except FileNotFoundError:  # pragma: no cover - environment without package data
        return library
    if not resource.is_file():
        return library
    with resource.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, MutableMapping):
        return library
    for car_model, tracks in payload.items():
        if not isinstance(tracks, Mapping):
            continue
        track_profiles: Dict[str, ThresholdProfile] = {}
        for track_name, values in tracks.items():
            if not isinstance(values, Mapping):
                continue
            track_profiles[str(track_name)] = _profile_from_payload(values)
        if track_profiles:
            library[str(car_model)] = track_profiles
    return library


class LoadBalanceRule:
    """Suggests changes when ΔNFR deviates from the baseline."""

    threshold: float = 10.0

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        for result in results:
            if abs(result.delta_nfr) > self.threshold:
                direction = "increase" if result.delta_nfr < 0 else "decrease"
                yield Recommendation(
                    category="suspension",
                    message=(
                        f"{direction.title()} rear ride height to rebalance load "
                        f"({MANUAL_REFERENCES['ride_height']})"
                    ),
                    rationale=(
                        "ΔNFR deviated by "
                        f"{result.delta_nfr:.1f} units relative to baseline at t={result.timestamp:.2f}."
                        f" Refer to {MANUAL_REFERENCES['ride_height']} to readjust ride heights."
                    ),
                    priority=100,
                )


class BottomingPriorityRule:
    """Bias ride height vs bump adjustments using bottoming ratios."""

    def __init__(
        self,
        *,
        priority: int = 18,
        ratio_threshold: float = 0.35,
        smooth_surface_cutoff: float = 1.05,
        ride_height_delta: float = 1.0,
        bump_delta: float = 2.0,
        spring_delta: float = 8.0,
        energy_compression_threshold: float = 0.35,
        energy_spring_threshold: float = 0.85,
        density_bias: float = 0.2,
    ) -> None:
        self.priority = int(priority)
        self.ratio_threshold = float(ratio_threshold)
        self.smooth_surface_cutoff = float(smooth_surface_cutoff)
        self.ride_height_delta = float(ride_height_delta)
        self.bump_delta = float(bump_delta)
        self.spring_delta = float(spring_delta)
        self.energy_compression_threshold = float(energy_compression_threshold)
        self.energy_spring_threshold = float(energy_spring_threshold)
        self.density_bias = float(density_bias)

    def _resolve_adjustment(
        self,
        phase_category: str,
        *,
        energy: float,
        density: float,
        is_rough: bool,
        ride_param: str,
        bump_param: str,
        spring_param: str,
    ) -> Tuple[str, float, str, str]:
        if energy >= self.energy_spring_threshold:
            if phase_category == "apex":
                return spring_param, self.spring_delta, "Increase spring stiffness", "springs"
            if phase_category == "exit":
                return ride_param, self.ride_height_delta, "Raise ride height", "ride_height"
            return bump_param, self.bump_delta, "Increase bump damping", "dampers"
        if energy >= self.energy_compression_threshold:
            if phase_category == "entry":
                return bump_param, self.bump_delta, "Increase bump damping", "dampers"
            return spring_param, self.spring_delta, "Increase spring stiffness", "springs"
        if is_rough or density >= self.density_bias:
            return bump_param, self.bump_delta, "Increase bump damping", "dampers"
        return ride_param, self.ride_height_delta, "Raise ride height", "ride_height"

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors:
            return []

        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            measures = getattr(microsector, "filtered_measures", {}) or {}
            if not isinstance(measures, Mapping):
                continue
            front_ratio = float(measures.get("bottoming_ratio_front", 0.0))
            rear_ratio = float(measures.get("bottoming_ratio_rear", 0.0))
            factors = getattr(microsector, "context_factors", {}) or {}
            surface_factor = 1.0
            if isinstance(factors, Mapping):
                surface_factor = float(factors.get("surface", surface_factor))
            is_rough = surface_factor >= self.smooth_surface_cutoff
            category = phase_family(getattr(microsector, "active_phase", "apex"))
            for axle, ratio, ride_param, bump_param in (
                (
                    "front",
                    front_ratio,
                    "front_ride_height",
                    "front_compression_clicks",
                ),
                (
                    "rear",
                    rear_ratio,
                    "rear_ride_height",
                    "rear_compression_clicks",
                ),
            ):
                if ratio < self.ratio_threshold:
                    continue
                position = axle
                density = max(
                    0.0,
                    min(1.0, float(measures.get(f"bumpstop_{position}_density", ratio))),
                )
                energy = max(0.0, float(measures.get(f"bumpstop_{position}_energy", 0.0)))
                spring_param = (
                    "front_spring_stiffness" if position == "front" else "rear_spring_stiffness"
                )
                parameter, delta, focus, reference_key = self._resolve_adjustment(
                    category,
                    energy=energy,
                    density=density,
                    is_rough=is_rough,
                    ride_param=ride_param,
                    bump_param=bump_param,
                    spring_param=spring_param,
                )
                reference = MANUAL_REFERENCES[reference_key]
                surface_label = "rough" if is_rough else "smooth"
                message = (
                    f"Bottoming operator: {focus} {axle} axle in microsector {microsector.index}"
                )
                rationale = (
                    f"Bottoming index {ratio:.2f} on the {axle} axle aligns with ∇NFR∥ peaks "
                    f"in microsector {microsector.index}. Bump stop density {density:.2f} and "
                    f"energy {energy:.2f} ΔNFR. Surface {surface_label} (factor {surface_factor:.2f})"
                    f" → prioritize {focus}. {reference}."
                )
                recommendations.append(
                    Recommendation(
                        category=category,
                        message=message,
                        rationale=rationale,
                        priority=self.priority,
                        parameter=parameter,
                        delta=delta,
                    )
                )
        return recommendations


class SuspensionVelocityRule:
    """Detect damper packing and axle asymmetries using velocity histograms."""

    def __init__(
        self,
        *,
        priority: int = 18,
        packing_threshold_pct: float = 35.0,
        packing_ar_threshold: float = 1.25,
        asymmetry_gap: float = 0.35,
    ) -> None:
        self.priority = int(priority)
        self.packing_threshold_pct = max(0.0, float(packing_threshold_pct))
        self.packing_ar_threshold = max(0.0, float(packing_ar_threshold))
        self.asymmetry_gap = max(0.0, float(asymmetry_gap))

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors:
            return []

        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            measures = getattr(microsector, "filtered_measures", {}) or {}
            if not isinstance(measures, Mapping):
                continue
            front_high_pct = _safe_float(
                measures.get("suspension_velocity_front_high_speed_pct"), 0.0
            )
            rear_high_pct = _safe_float(
                measures.get("suspension_velocity_rear_high_speed_pct"), 0.0
            )
            front_rebound_pct = _safe_float(
                measures.get("suspension_velocity_front_high_speed_rebound_pct"), 0.0
            )
            rear_rebound_pct = _safe_float(
                measures.get("suspension_velocity_rear_high_speed_rebound_pct"), 0.0
            )
            front_ar = _safe_float(
                measures.get("suspension_velocity_front_ar_index"), 0.0
            )
            rear_ar = _safe_float(measures.get("suspension_velocity_rear_ar_index"), 0.0)
            phase_category = phase_family(getattr(microsector, "active_phase", "apex"))
            for axle_label, high_pct, rebound_pct, ar_index in (
                ("front", front_high_pct, front_rebound_pct, front_ar),
                ("rear", rear_high_pct, rear_rebound_pct, rear_ar),
            ):
                if high_pct < self.packing_threshold_pct:
                    continue
                if ar_index < self.packing_ar_threshold:
                    continue
                if rebound_pct >= self.packing_threshold_pct * 0.6:
                    continue
                message = (
                    f"Damper operator: relieve packing on the {axle_label} axle "
                    f"in microsector {microsector.index}"
                )
                rationale = (
                    f"HS compression {high_pct:.1f}% vs HS rebound {rebound_pct:.1f}% "
                    f"(A/R {ar_index:.2f}) exceed the {self.packing_threshold_pct:.1f}% threshold. "
                    f"Reduce rebound following {MANUAL_REFERENCES['dampers']}."
                )
                recommendations.append(
                    Recommendation(
                        category=phase_category,
                        message=message,
                        rationale=rationale,
                        priority=self.priority,
                    )
                )

            ar_gap = abs(front_ar - rear_ar)
            if ar_gap < self.asymmetry_gap:
                continue
            if max(front_high_pct, rear_high_pct) < self.packing_threshold_pct * 0.5:
                continue
            dominant = "front" if front_ar > rear_ar else "rear"
            message = (
                f"Damper operator: balance A/R on the {dominant} axle "
                f"in microsector {microsector.index}"
            )
            rationale = (
                f"A/R indices F {front_ar:.2f} · R {rear_ar:.2f} differ by {ar_gap:.2f}. "
                f"Offset compression/rebound clicks using {MANUAL_REFERENCES['dampers']}."
            )
            recommendations.append(
                Recommendation(
                    category=phase_category,
                    message=message,
                    rationale=rationale,
                    priority=self.priority + 1,
                )
            )
        return recommendations


class BrakeHeadroomRule:
    """Adjust the maximum brake force when significant headroom mismatches arise."""

    def __init__(
        self,
        *,
        priority: int = 14,
        margin: float = 0.05,
        increase_step: float = 0.02,
        decrease_step: float = 0.02,
        sustained_lock_threshold: float = 0.5,
    ) -> None:
        self.priority = int(priority)
        self.margin = max(0.0, float(margin))
        self.increase_step = float(increase_step)
        self.decrease_step = float(decrease_step)
        self.sustained_lock_threshold = max(0.0, min(1.0, float(sustained_lock_threshold)))

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None:
            return []

        target = getattr(context.objectives, "target_brake_headroom", 0.4)
        margin = self.margin
        recommendations: list[Recommendation] = []
        for microsector in microsectors:
            if not getattr(microsector, "brake_event", False):
                continue
            measures = getattr(microsector, "filtered_measures", {}) or {}
            if not isinstance(measures, Mapping):
                continue
            if "brake_headroom" not in measures:
                continue
            headroom = _safe_float(measures.get("brake_headroom"), 0.0)
            if not math.isfinite(headroom):
                continue
            deviation = headroom - target
            abs_activation = max(
                0.0,
                min(1.0, _safe_float(measures.get("brake_headroom_abs_activation"), 0.0)),
            )
            partial_locking = max(
                0.0,
                min(1.0, _safe_float(measures.get("brake_headroom_partial_locking"), 0.0)),
            )
            sustained_ratio = max(
                0.0,
                min(1.0, _safe_float(measures.get("brake_headroom_sustained_locking"), 0.0)),
            )
            peak_decel = _safe_float(measures.get("brake_headroom_peak_decel"), 0.0)
            if deviation > margin:
                delta = self.increase_step
                message = (
                    f"Brake operator: raise per-wheel maximum force in microsector "
                    f"{microsector.index}"
                )
            elif deviation < -margin:
                delta = -self.decrease_step
                if sustained_ratio >= self.sustained_lock_threshold:
                    message = (
                        f"Brake operator: relieve sustained locking in microsector "
                        f"{microsector.index}"
                    )
                else:
                    message = (
                        f"Brake operator: lower per-wheel maximum force in microsector "
                        f"{microsector.index}"
                    )
            else:
                continue
            rationale = (
                f"Brake margin μ {headroom:.2f} versus target {target:.2f} "
                f"({deviation:+.2f}). Peak decel {peak_decel:.2f}m/s², ABS μ {abs_activation:.2f}, "
                f"partial locking μ {partial_locking:.2f}."
            )
            if sustained_ratio >= self.sustained_lock_threshold:
                rationale = (
                    f"{rationale} Sustained locking μ {sustained_ratio:.2f} detected."
                )
            rationale = f"{rationale} {MANUAL_REFERENCES['braking']}"
            priority_scale = _session_priority_scale(context, "entry", "brakes")
            priority = _scale_priority_value(self.priority, priority_scale)
            recommendations.append(
                Recommendation(
                    category="entry",
                    message=message,
                    rationale=rationale,
                    priority=priority,
                    parameter="brake_max_per_wheel",
                    delta=delta,
                )
            )
        return recommendations


class StabilityIndexRule:
    """Issue recommendations when the sense index degrades."""

    threshold: float = 0.6

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        for result in results:
            if result.sense_index < self.threshold:
                yield Recommendation(
                    category="aero",
                    message=(
                        "Stabilise aero balance to recover sense index "
                        f"({MANUAL_REFERENCES['aero']})"
                    ),
                    rationale=(
                        "Sense index dropped to "
                        f"{result.sense_index:.2f} at t={result.timestamp:.2f}, below the threshold of "
                        f"{self.threshold:.2f}. Refer to {MANUAL_REFERENCES['aero']} to rebalance load."
                    ),
                    priority=110,
                )


class AeroCoherenceRule:
    """React to aero imbalance at high speed when low speed remains stable."""

    def __init__(
        self,
        *,
        high_speed_threshold: float = 0.25,
        low_speed_tolerance: float = 0.1,
        min_high_samples: int = 5,
        priority: int = 108,
        profile_name: str = "race",
        delta_step: float = 0.5,
        min_aero_mechanical: float = 0.7,
    ) -> None:
        self.high_speed_threshold = float(high_speed_threshold)
        self.low_speed_tolerance = float(low_speed_tolerance)
        self.min_high_samples = int(min_high_samples)
        self.priority = int(priority)
        self.profile_name = profile_name
        self.delta_step = float(delta_step)
        self.min_aero_mechanical = float(min_aero_mechanical)

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None:
            return []

        profiles = getattr(context, "aero_profiles", {}) or {}
        profile = profiles.get(self.profile_name) or next(iter(profiles.values()), None)
        target_high = getattr(profile, "high_speed_target", 0.0)
        target_low = getattr(profile, "low_speed_target", 0.0)

        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            measures = getattr(microsector, "filtered_measures", {}) or {}
            high_samples = float(measures.get("aero_high_samples", 0.0))
            if high_samples < self.min_high_samples:
                continue
            high_imbalance = float(measures.get("aero_high_imbalance", 0.0))
            low_imbalance = float(measures.get("aero_low_imbalance", 0.0))
            am_coherence = float(measures.get("aero_mechanical_coherence", 1.0))
            high_deviation = high_imbalance - target_high
            low_deviation = low_imbalance - target_low
            if abs(low_deviation) > self.low_speed_tolerance:
                continue
            if abs(high_deviation) < self.high_speed_threshold:
                continue
            if am_coherence > self.min_aero_mechanical:
                continue

            drift_delta = float(measures.get("aero_drift_high_mu_delta", 0.0))
            drift_tolerance = float(measures.get("aero_drift_mu_tolerance", 0.04))
            if high_deviation > 0.0 and drift_delta < -drift_tolerance:
                continue
            if high_deviation < 0.0 and drift_delta > drift_tolerance:
                continue

            if high_deviation > 0:
                delta = self.delta_step
                action = "Increase rear wing angle"
                direction = "rear axle"
            else:
                delta = -self.delta_step
                action = "Reduce rear wing angle / reinforce front load"
                direction = "front axle"

            recommendations.append(
                Recommendation(
                    category="aero",
                    message=(
                        f"High-speed microsector {microsector.index}: {action}"
                    ),
                    rationale=(
                        f"High-speed aerodynamic ΔNFR {high_imbalance:+.2f} versus the objective "
                        f"{target_high:+.2f} with stable low-speed balance ({low_imbalance:+.2f}) and "
                        f"C(c/d/a) {am_coherence:.2f}. Reinforce the {direction} load "
                        f"({MANUAL_REFERENCES['aero']})."
                    ),
                    priority=self.priority,
                    parameter="rear_wing_angle",
                    delta=delta,
                )
            )
        return recommendations


class FrontWingBalanceRule:
    """Recommend front wing adjustments when high speed balance is front-limited."""

    def __init__(
        self,
        *,
        high_speed_threshold: float = 0.18,
        min_high_samples: int = 4,
        max_aero_mechanical: float = 0.65,
        priority: int = 106,
        delta_step: float = 0.5,
        profile_name: str = "race",
    ) -> None:
        self.high_speed_threshold = float(high_speed_threshold)
        self.min_high_samples = int(min_high_samples)
        self.max_aero_mechanical = float(max_aero_mechanical)
        self.priority = int(priority)
        self.delta_step = float(delta_step)
        self.profile_name = profile_name

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None:
            return []

        profiles = getattr(context, "aero_profiles", {}) or {}
        profile = profiles.get(self.profile_name) or next(iter(profiles.values()), None)
        target_high = getattr(profile, "high_speed_target", 0.0)

        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            measures = getattr(microsector, "filtered_measures", {}) or {}
            high_samples = float(measures.get("aero_high_samples", 0.0))
            if high_samples < self.min_high_samples:
                continue
            high_imbalance = float(measures.get("aero_high_imbalance", 0.0))
            deviation = high_imbalance - target_high
            if deviation >= -self.high_speed_threshold:
                continue
            am_coherence = float(measures.get("aero_mechanical_coherence", 1.0))
            if am_coherence >= self.max_aero_mechanical:
                continue

            drift_delta = float(measures.get("aero_drift_high_mu_delta", 0.0))
            drift_tolerance = float(measures.get("aero_drift_mu_tolerance", 0.04))
            if drift_delta > -drift_tolerance:
                continue

            front_total = float(measures.get("aero_high_front_total", 0.0))
            rear_total = float(measures.get("aero_high_rear_total", 0.0))
            lat_front = float(measures.get("aero_high_front_lateral", 0.0))
            lat_rear = float(measures.get("aero_high_rear_lateral", 0.0))
            long_front = float(measures.get("aero_high_front_longitudinal", 0.0))
            long_rear = float(measures.get("aero_high_rear_longitudinal", 0.0))

            rationale = (
                f"High-speed aerodynamic ΔNFR {high_imbalance:+.2f} versus the objective "
                f"{target_high:+.2f} with C(c/d/a) {am_coherence:.2f}. F/R distribution {front_total:+.2f}/{rear_total:+.2f}. "
                f"Lateral axes {lat_front:+.2f}/{lat_rear:+.2f}, longitudinal {long_front:+.2f}/{long_rear:+.2f}. "
                f"Reinforce the front axle load ({MANUAL_REFERENCES['aero']})."
            )
            recommendations.append(
                Recommendation(
                    category="aero",
                    message=(
                        f"High-speed microsector {microsector.index}: increase front wing angle"
                    ),
                    rationale=rationale,
                    priority=self.priority,
                    parameter="front_wing_angle",
                    delta=self.delta_step,
                )
            )

        return recommendations


class CoherenceRule:
    """High-level rule that considers the average sense index across a stint."""

    min_average_si: float = 0.75

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not results:
            return []
        average_si = mean(result.sense_index for result in results)
        if average_si < self.min_average_si:
            priority = _scale_priority_value(120, _session_coherence_scale(context))
            return [
                Recommendation(
                    category="driver",
                    message=(
                        "Review driving inputs for consistency "
                        f"({MANUAL_REFERENCES['driver']})"
                    ),
                    rationale=(
                        "Average sense index across the analysed stint is "
                        f"{average_si:.2f}, below the expected threshold of {self.min_average_si:.2f}. "
                        f"Lean on {MANUAL_REFERENCES['driver']} to reinforce consistent habits."
                    ),
                    priority=priority,
                )
            ]
        return []


DEFAULT_THRESHOLD_PROFILE = ThresholdProfile(
    entry_delta_tolerance=1.5,
    apex_delta_tolerance=1.0,
    exit_delta_tolerance=2.0,
    piano_delta_tolerance=2.5,
    rho_detune_threshold=0.7,
    phase_targets=_BASELINE_PHASE_TARGETS,
    phase_weights=_BASELINE_PHASE_WEIGHTS,
    archetype_phase_targets=_BASELINE_ARCHETYPE_TARGETS,
    robustness=_BASELINE_ROBUSTNESS_THRESHOLDS,
)


_RESOURCE_THRESHOLD_LIBRARY = _load_threshold_library_from_resource()

DEFAULT_THRESHOLD_LIBRARY: Mapping[str, Mapping[str, ThresholdProfile]]

if _RESOURCE_THRESHOLD_LIBRARY:
    merged: Dict[str, Dict[str, ThresholdProfile]] = {
        "generic": {"generic": DEFAULT_THRESHOLD_PROFILE}
    }
    for car_model, tracks in _RESOURCE_THRESHOLD_LIBRARY.items():
        merged.setdefault(car_model, {}).update(tracks)
    DEFAULT_THRESHOLD_LIBRARY = merged
else:
    DEFAULT_THRESHOLD_LIBRARY = {"generic": {"generic": DEFAULT_THRESHOLD_PROFILE}}


def lookup_threshold_profile(
    car_model: str,
    track_name: str,
    library: Mapping[str, Mapping[str, ThresholdProfile]] | None = None,
) -> ThresholdProfile:
    """Resolve the closest matching threshold profile for a car/track."""

    catalogue = library or DEFAULT_THRESHOLD_LIBRARY
    for car_key in (car_model, "generic"):
        car_profiles = catalogue.get(car_key)
        if not car_profiles:
            continue
        for track_key in (track_name, "generic"):
            profile = car_profiles.get(track_key)
            if profile is not None:
                return profile
    return DEFAULT_THRESHOLD_PROFILE


def _goal_for_phase(microsector: Microsector, phase: str) -> Goal | None:
    aliases = list(expand_phase_alias(phase))
    for candidate in reversed(aliases):
        for goal in microsector.goals:
            if goal.phase == candidate:
                return goal
    for candidate in aliases:
        for goal in microsector.goals:
            if goal.phase == candidate:
                return goal
    return None


def _phase_samples(results: Sequence[EPIBundle], indices: Iterable[int]) -> List[EPIBundle]:
    return [results[i] for i in indices if 0 <= i < len(results)]


def _node_label(node: str) -> str:
    return NODE_LABELS.get(node, node)


def _node_nu_f_values(
    results: Sequence[EPIBundle], indices: Iterable[int], node: str
) -> List[float]:
    values: List[float] = []
    for index in indices:
        if 0 <= index < len(results):
            bundle = results[index]
            component = getattr(bundle, node, None)
            if component is not None and hasattr(component, "nu_f"):
                values.append(float(component.nu_f))
    return values


def _slider_action(phase: str, node: str, direction: str) -> str:
    actions = _OPERATOR_NODE_ACTIONS.get(phase, {})
    node_actions = actions.get(node, {})
    action = node_actions.get(direction)
    if action:
        return action
    node_label = _node_label(node)
    if direction == "increase":
        return f"incrementar la influencia de {node_label}"
    return f"reducir la influencia de {node_label}"


def _phase_action_templates(phase: str, metric: str, node: str | None = None) -> Sequence[PhaseActionTemplate]:
    key = phase_family(phase)
    templates = _PHASE_ACTION_ROADMAP.get(key, ())
    selected: List[PhaseActionTemplate] = []
    for template in templates:
        if template.metric != metric:
            continue
        if template.nodes and node not in template.nodes:
            continue
        selected.append(template)
    return selected


def _apply_action_template(
    template: PhaseActionTemplate, raw_value: float, *, gradient: float | None = None
) -> float | None:
    value = raw_value * template.scale
    if gradient is not None and template.gradient_offset_scale:
        try:
            gradient_value = float(gradient)
        except (TypeError, ValueError):
            gradient_value = 0.0
        else:
            if math.isfinite(gradient_value):
                deadband = template.gradient_deadband
                if abs(gradient_value) > deadband:
                    down_scale, up_scale = template.gradient_offset_scale
                    if gradient_value < 0.0:
                        offset = -abs(gradient_value) * down_scale
                    else:
                        offset = abs(gradient_value) * up_scale
                    limit = template.gradient_offset_limit
                    if limit > 0.0:
                        offset = max(-limit, min(limit, offset))
                    value += offset
    value = max(template.min_value, min(template.max_value, value))
    if template.step > 0:
        value = round(value / template.step) * template.step
        value = max(template.min_value, min(template.max_value, value))
    threshold = template.step * 0.5 if template.step > 0 else 1e-3
    if abs(value) < threshold:
        return None
    return value


def _phase_action_recommendations(
    *,
    phase: str,
    category: str,
    metric: str,
    raw_value: float,
    base_rationale: str,
    priority: int,
    reference_key: str,
    node: str | None = None,
    gradient: float | None = None,
) -> List[Recommendation]:
    recommendations: List[Recommendation] = []
    for template in _phase_action_templates(phase, metric, node):
        value = _apply_action_template(template, raw_value, gradient=gradient)
        if value is None:
            continue
        message = template.message_pattern.format(delta=value)
        rationale = (
            f"{base_rationale} Suggested action: {message}. Refer to "
            f"{MANUAL_REFERENCES[reference_key]} to apply the adjustment."
        )
        if template.parameter in _SPRING_PARAMETERS:
            rationale = (
                f"{base_rationale} Suggested action: {message}. νf_susp weights the adjustment "
                f"to smooth ∇NFR⊥ across the active G band. Refer to "
                f"{MANUAL_REFERENCES[reference_key]} to apply the adjustment."
            )
        recommendations.append(
            Recommendation(
                category=category,
                message=message,
                rationale=rationale,
                priority=priority + template.priority_offset,
                parameter=template.parameter,
                delta=float(value),
            )
        )
    return recommendations


def _merge_recommendations_by_parameter(
    existing: Mapping[str, Recommendation] | None,
    candidates: Sequence[Recommendation],
) -> Dict[str, Recommendation]:
    merged: Dict[str, Recommendation] = dict(existing or {})
    for candidate in candidates:
        parameter = candidate.parameter
        if not parameter:
            continue
        current = merged.get(parameter)
        if current is None:
            merged[parameter] = candidate
            continue
        current_delta = abs(current.delta or 0.0)
        candidate_delta = abs(candidate.delta or 0.0)
        if candidate.priority < current.priority:
            merged[parameter] = candidate
        elif candidate.priority == current.priority and candidate_delta > current_delta:
            merged[parameter] = candidate
    return merged


def _geometry_snapshot(
    goal: Goal,
    microsector: Microsector,
    samples: Sequence[EPIBundle],
    context: RuleContext,
) -> Dict[str, float]:
    filtered = getattr(microsector, "filtered_measures", {}) or {}
    measured_alignment = _safe_float(
        microsector.phase_alignment.get(
            goal.phase, getattr(goal, "measured_phase_alignment", 1.0)
        ),
        _safe_float(getattr(goal, "measured_phase_alignment", 1.0), 1.0),
    )
    if not math.isfinite(measured_alignment):
        measured_alignment = _safe_float(filtered.get("phase_alignment_window"), 1.0)
    target_alignment = _safe_float(
        getattr(goal, "target_phase_alignment", measured_alignment), measured_alignment
    )
    measured_synchrony = _safe_float(
        (microsector.phase_synchrony or {}).get(
            goal.phase, getattr(goal, "measured_phase_synchrony", 1.0)
        ),
        _safe_float(getattr(goal, "measured_phase_synchrony", 1.0), 1.0),
    )
    if not math.isfinite(measured_synchrony):
        measured_synchrony = _safe_float(filtered.get("phase_synchrony_window"), 1.0)
    target_synchrony = _safe_float(
        getattr(goal, "target_phase_synchrony", measured_synchrony), measured_synchrony
    )
    measured_lag = _safe_float(
        microsector.phase_lag.get(
            goal.phase, getattr(goal, "measured_phase_lag", 0.0)
        ),
        _safe_float(getattr(goal, "measured_phase_lag", 0.0), 0.0),
    )
    if not math.isfinite(measured_lag):
        measured_lag = _safe_float(filtered.get("phase_lag_window"), 0.0)
    target_lag = _safe_float(
        getattr(goal, "target_phase_lag", measured_lag), measured_lag
    )
    coherence_values: List[float] = []
    for bundle in samples:
        coherence_values.append(_safe_float(getattr(bundle, "coherence_index", 0.0)))
    if not coherence_values and "coherence_index" in filtered:
        coherence_values.append(_safe_float(filtered.get("coherence_index"), 0.0))
    average_coherence = mean(coherence_values) if coherence_values else 0.0
    target_coherence = _safe_float(
        getattr(goal, "target_coherence_index", context.objectives.target_sense_index),
        context.objectives.target_sense_index,
    )
    alignment_delta = target_alignment - measured_alignment
    lag_delta = measured_lag - target_lag
    synchrony_delta = target_synchrony - measured_synchrony
    coherence_delta = target_coherence - average_coherence
    return {
        "measured_alignment": measured_alignment,
        "target_alignment": target_alignment,
        "alignment_delta": alignment_delta,
        "alignment_gap": abs(alignment_delta),
        "measured_synchrony": measured_synchrony,
        "target_synchrony": target_synchrony,
        "synchrony_delta": synchrony_delta,
        "synchrony_gap": abs(synchrony_delta),
        "measured_lag": measured_lag,
        "target_lag": target_lag,
        "lag_delta": lag_delta,
        "lag_gap": abs(lag_delta),
        "average_coherence": average_coherence,
        "target_coherence": target_coherence,
        "coherence_delta": coherence_delta,
        "coherence_gap": abs(coherence_delta),
    }


def _geometry_urgency(snapshot: Mapping[str, float]) -> int:
    urgency = 0
    alignment_gap = float(snapshot.get("alignment_gap", 0.0))
    synchrony_gap = float(snapshot.get("synchrony_gap", 0.0))
    lag_gap = float(snapshot.get("lag_gap", 0.0))
    coherence_gap = float(snapshot.get("coherence_gap", 0.0))
    if alignment_gap > _ALIGNMENT_THRESHOLD:
        urgency = max(urgency, 3 if alignment_gap > _ALIGNMENT_THRESHOLD * 2 else 2)
    if synchrony_gap > _SYNCHRONY_THRESHOLD:
        urgency = max(urgency, 3 if synchrony_gap > _SYNCHRONY_THRESHOLD * 1.75 else 2)
    if lag_gap > _LAG_THRESHOLD:
        urgency = max(urgency, 3 if lag_gap > _LAG_THRESHOLD * 2 else 2)
    if coherence_gap > _COHERENCE_THRESHOLD:
        urgency = max(urgency, 3 if coherence_gap > _COHERENCE_THRESHOLD * 2 else 2)
    return urgency


def _boost_geometry_priority(
    recommendations: Sequence[Recommendation],
    base_priority: int,
    urgency: int,
) -> None:
    if urgency <= 0:
        return
    target_priority = base_priority - urgency
    for recommendation in recommendations:
        if recommendation.parameter in _GEOMETRY_PARAMETERS:
            recommendation.priority = min(recommendation.priority, target_priority)


def _alignment_snapshot(goal: Goal, microsector: Microsector | None) -> Tuple[float, float, float, float]:
    measured_lag = getattr(goal, "measured_phase_lag", 0.0)
    measured_alignment = getattr(goal, "measured_phase_alignment", 1.0)
    if microsector is not None:
        measured_lag = microsector.phase_lag.get(goal.phase, measured_lag)
        measured_alignment = microsector.phase_alignment.get(goal.phase, measured_alignment)
    target_lag = getattr(goal, "target_phase_lag", 0.0)
    target_alignment = getattr(goal, "target_phase_alignment", 1.0)
    return measured_lag, measured_alignment, target_lag, target_alignment


def _should_flip_alignment(
    measured_alignment: float,
    target_alignment: float,
    measured_lag: float,
    target_lag: float,
) -> bool:
    if measured_alignment < 0.0:
        return True
    if target_alignment - measured_alignment > _ALIGNMENT_ALIGNMENT_GAP:
        return True
    if abs(measured_lag - target_lag) > _ALIGNMENT_LAG_GAP and measured_alignment < target_alignment:
        return True
    return False


def _axis_focus_descriptor(phase: str, axis: str) -> tuple[str, str, Tuple[str, ...]] | None:
    family = phase_family(phase)
    key = (family, axis)
    if key in _AXIS_FOCUS_MAP:
        return _AXIS_FOCUS_MAP[key]
    direct = (phase, axis)
    return _AXIS_FOCUS_MAP.get(direct)
def _format_quiet_sequence(sequence: Sequence[int]) -> str:
    if not sequence:
        return ""
    start = sequence[0] + 1
    end = sequence[-1] + 1
    if start == end:
        return f"Curva {start}"
    return f"Curvas {start}-{end}"


def _quiet_recommendation_notice(
    microsectors: Sequence[Microsector], sequences: Sequence[Sequence[int]]
) -> tuple[str, str]:
    descriptors: List[str] = []
    coverage_values: List[float] = []
    slack_values: List[float] = []
    si_values: List[float] = []
    epi_values: List[float] = []
    for sequence in sequences:
        descriptors.append(_format_quiet_sequence(sequence))
        for index in sequence:
            if index < 0 or index >= len(microsectors):
                continue
            coverage, slack, si_variance, epi_abs = microsector_stability_metrics(
                microsectors[index]
            )
            coverage_values.append(coverage)
            slack_values.append(slack)
            si_values.append(si_variance)
            epi_values.append(epi_abs)
    message = f"Do not adjust: {', '.join(descriptors)}"
    if not coverage_values:
        rationale = (
            "Detected a stable sequence without notable dynamic activation."
        )
        return message, rationale
    coverage_avg = sum(coverage_values) / len(coverage_values)
    slack_avg = sum(slack_values) / len(slack_values) if slack_values else 0.0
    si_avg = sum(si_values) / len(si_values) if si_values else 0.0
    epi_avg = sum(epi_values) / len(epi_values) if epi_values else 0.0
    rationale = (
        "Detection of prolonged structural silence:"
        f" silence μ {coverage_avg * 100.0:.0f}%"
        f", slack μ {slack_avg:.2f}, Siσ μ {si_avg:.4f}, |dEPI| μ {epi_avg:.3f}."
    )
    return message, rationale


def _brake_event_summary(
    microsector: Microsector,
) -> tuple[str | None, str | None, float]:
    operator_events = getattr(microsector, "operator_events", {}) or {}
    silence_payloads = silence_event_payloads(operator_events)
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
            return None, None, 0.0
    relevant: List[Dict[str, object]] = []
    for event_type in ("OZ", "IL"):
        payloads = operator_events.get(event_type, ())
        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            threshold = _safe_float(payload.get("delta_nfr_threshold"))
            if threshold <= 0.0:
                continue
            peak = abs(_safe_float(payload.get("delta_nfr_peak")))
            ratio = _safe_float(payload.get("delta_nfr_ratio"))
            if ratio <= 0.0 and peak > 0.0 and threshold > 1e-9:
                ratio = peak / threshold
            average = _safe_float(payload.get("delta_nfr_avg"))
            surface_label: str = ""
            label_payload = payload.get("surface_label")
            if isinstance(label_payload, str):
                surface_label = label_payload
            else:
                surface_payload = payload.get("surface")
                if isinstance(surface_payload, Mapping):
                    label_value = surface_payload.get("label")
                    if isinstance(label_value, str):
                        surface_label = label_value
            label = payload.get("name")
            if not isinstance(label, str) or not label:
                label = canonical_operator_label(event_type)
            relevant.append(
                {
                    "type": event_type,
                    "label": label,
                    "ratio": ratio,
                    "threshold": threshold,
                    "peak": peak,
                    "average": average,
                    "surface": surface_label,
                }
            )
    severe = [entry for entry in relevant if entry["ratio"] >= 1.0]
    if not severe:
        return None, None, 0.0
    bias_score = 0.0
    max_ratio = 0.0
    summary_parts: List[str] = []
    for event_type in ("OZ", "IL"):
        typed = [entry for entry in severe if entry["type"] == event_type]
        if not typed:
            continue
        count = len(typed)
        worst = max(typed, key=lambda item: item["ratio"])
        max_ratio = max(max_ratio, _safe_float(worst["ratio"]))
        surface_name = worst["surface"] or "surface"
        threshold = _safe_float(worst["threshold"])
        peak = _safe_float(worst["peak"])
        label = worst.get("label") or canonical_operator_label(event_type)
        summary_parts.append(
            f"{label}×{count} ({surface_name}) ΔNFR {peak:.2f}>{threshold:.2f}"
        )
        weight = sum(_safe_float(entry["ratio"]) for entry in typed)
        if event_type == "OZ":
            bias_score += weight
        else:
            bias_score -= weight
    direction: str | None = None
    if abs(bias_score) >= 0.5:
        direction = "forward" if bias_score > 0 else "rearward"
    summary = " · ".join(summary_parts)
    return summary, direction, max_ratio


class PhaseDeltaDeviationRule:
    """Detects ΔNFR mismatches for a given phase of the corner."""

    def __init__(
        self,
        phase: str,
        operator_label: str,
        category: str,
        phase_label: str,
        priority: int,
        reference_key: str,
    ) -> None:
        self.phase = phase
        self.operator_label = operator_label
        self.category = category
        self.phase_label = phase_label
        self.priority = priority
        self.reference_key = reference_key

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None:
            return []

        tolerance = context.thresholds.tolerance_for_phase(self.phase)
        priority_scale = _session_priority_scale(context, self.phase)
        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            start_index = len(recommendations)
            goal = _goal_for_phase(microsector, self.phase)
            if goal is None:
                continue
            indices = list(microsector.phase_indices(goal.phase))
            samples = _phase_samples(results, indices)
            if not samples:
                continue
            actual_delta = mean(bundle.delta_nfr for bundle in samples)
            deviation = actual_delta - goal.target_delta_nfr
            avg_long = mean(bundle.delta_nfr_proj_longitudinal for bundle in samples)
            avg_lat = mean(bundle.delta_nfr_proj_lateral for bundle in samples)
            target_long = getattr(goal, "target_delta_nfr_long", 0.0)
            target_lat = getattr(goal, "target_delta_nfr_lat", 0.0)
            long_dev = avg_long - target_long
            lat_dev = avg_lat - target_lat
            abs_long = abs(long_dev)
            abs_lat = abs(lat_dev)
            axis_bias: str | None = None
            dominance_threshold = 1.2
            axis_delta_threshold = max(0.02, tolerance * 0.25)
            if abs_long > abs_lat * dominance_threshold and abs_long > axis_delta_threshold:
                axis_bias = "longitudinal"
            elif abs_lat > abs_long * dominance_threshold and abs_lat > axis_delta_threshold:
                axis_bias = "lateral"
            geometry_snapshot = _geometry_snapshot(goal, microsector, samples, context)
            phase_key = phase_family(self.phase)
            geometry_nodes = _GEOMETRY_METRIC_NODES.get(phase_key, {})
            lag_value = geometry_snapshot["measured_lag"]
            target_lag = geometry_snapshot["target_lag"]
            lag_gap = geometry_snapshot["lag_gap"]
            axis_weights = getattr(goal, "delta_axis_weights", {})
            weight_long = float(axis_weights.get("longitudinal", 0.5))
            weight_lat = float(axis_weights.get("lateral", 0.5))
            event_summary: str | None = None
            bias_direction: str | None = None
            worst_ratio: float = 0.0
            if phase_key == "entry":
                event_summary, bias_direction, worst_ratio = _brake_event_summary(
                    microsector
                )
            phase_summary = (
                f"{self.operator_label} applied to the {self.phase_label} phase in "
                f"microsector {microsector.index}. The ΔNFR target was "
                f"{goal.target_delta_nfr:.2f}, but the recorded average reached "
                f"{actual_delta:.2f} ({deviation:+.2f})."
            )
            if event_summary:
                phase_summary = f"{phase_summary} {event_summary}."

            geometry_actions_map: Dict[str, Recommendation] = {}
            geometry_urgency = _geometry_urgency(geometry_snapshot)

            if geometry_snapshot["alignment_gap"] > _ALIGNMENT_THRESHOLD:
                alignment_rationale = (
                    f"{phase_summary} Target Siφ {geometry_snapshot['target_alignment']:+.2f} "
                    f"versus measured {geometry_snapshot['measured_alignment']:+.2f} "
                    f"({geometry_snapshot['alignment_delta']:+.2f})."
                )
                alignment_node = geometry_nodes.get("phase_alignment")
                if alignment_node:
                    alignment_actions = _phase_action_recommendations(
                        phase=self.phase,
                        category=self.category,
                        metric="phase_alignment",
                        raw_value=geometry_snapshot["alignment_delta"],
                        base_rationale=alignment_rationale,
                        priority=self.priority - 1,
                        reference_key=self.reference_key,
                        node=alignment_node,
                    )
                    geometry_actions_map = _merge_recommendations_by_parameter(
                        geometry_actions_map, alignment_actions
                    )

            if geometry_snapshot["synchrony_gap"] > _SYNCHRONY_THRESHOLD:
                synchrony_rationale = (
                    f"{phase_summary} Target Φsync {geometry_snapshot['target_synchrony']:.2f} "
                    f"versus measured {geometry_snapshot['measured_synchrony']:.2f} "
                    f"({geometry_snapshot['synchrony_delta']:+.2f})."
                )
                synchrony_node = geometry_nodes.get("phase_synchrony_index") or geometry_nodes.get(
                    "phase_alignment"
                )
                if synchrony_node:
                    synchrony_actions = _phase_action_recommendations(
                        phase=self.phase,
                        category=self.category,
                        metric="phase_synchrony_index",
                        raw_value=geometry_snapshot["synchrony_delta"],
                        base_rationale=synchrony_rationale,
                        priority=self.priority - 1,
                        reference_key=self.reference_key,
                        node=synchrony_node,
                    )
                    geometry_actions_map = _merge_recommendations_by_parameter(
                        geometry_actions_map, synchrony_actions
                    )

            if geometry_snapshot["lag_gap"] > _LAG_THRESHOLD:
                lag_rationale = (
                    f"{phase_summary} Measured θ {geometry_snapshot['measured_lag']:+.2f}rad "
                    f"versus target {geometry_snapshot['target_lag']:+.2f}rad "
                    f"({geometry_snapshot['lag_delta']:+.2f})."
                )
                lag_node = geometry_nodes.get("phase_lag")
                if lag_node:
                    lag_actions = _phase_action_recommendations(
                        phase=self.phase,
                        category=self.category,
                        metric="phase_lag",
                        raw_value=geometry_snapshot["lag_delta"],
                        base_rationale=lag_rationale,
                        priority=self.priority - 1,
                        reference_key=self.reference_key,
                        node=lag_node,
                    )
                    geometry_actions_map = _merge_recommendations_by_parameter(
                        geometry_actions_map, lag_actions
                    )

            if geometry_snapshot["coherence_gap"] > _COHERENCE_THRESHOLD:
                coherence_rationale = (
                    f"{phase_summary} Mean C(t) {geometry_snapshot['average_coherence']:.2f} "
                    f"(target {geometry_snapshot['target_coherence']:.2f}, Δ "
                    f"{geometry_snapshot['coherence_delta']:+.2f})."
                )
                coherence_node = geometry_nodes.get("coherence_index")
                if coherence_node:
                    coherence_actions = _phase_action_recommendations(
                        phase=self.phase,
                        category=self.category,
                        metric="coherence_index",
                        raw_value=geometry_snapshot["coherence_delta"],
                        base_rationale=coherence_rationale,
                        priority=self.priority - 1,
                        reference_key=self.reference_key,
                        node=coherence_node,
                    )
                    geometry_actions_map = _merge_recommendations_by_parameter(
                        geometry_actions_map, coherence_actions
                    )

            geometry_recommendations = list(geometry_actions_map.values())
            if geometry_recommendations:
                _boost_geometry_priority(
                    geometry_recommendations, self.priority, max(geometry_urgency, 1)
                )
                recommendations.extend(geometry_recommendations)

            delta_triggered = abs(deviation) > tolerance
            if not delta_triggered and not geometry_recommendations:
                continue

            base_rationale = phase_summary
            if delta_triggered:
                base_rationale = (
                    f"{phase_summary} The tolerance defined for {context.profile_label} "
                    f"is ±{tolerance:.2f}."
                )
            spring_recommendations: List[Recommendation] = []
            if axis_bias == "lateral":
                suspension_nu_f: List[float] = []
                for bundle in samples:
                    suspension = getattr(bundle, "suspension", None)
                    if suspension is None:
                        continue
                    nu_f_value = getattr(suspension, "nu_f", None)
                    if nu_f_value is None:
                        continue
                    try:
                        suspension_nu_f.append(float(nu_f_value))
                    except (TypeError, ValueError):
                        continue
                if suspension_nu_f:
                    target_nu_f = float(getattr(goal, "nu_f_target", 0.0))
                    actual_nu_f = mean(suspension_nu_f)
                    nu_f_delta = actual_nu_f - target_nu_f
                    nu_f_tolerance = max(0.03, abs(target_nu_f) * 0.15)
                    if abs(nu_f_delta) > nu_f_tolerance:
                        spring_signal = lat_dev * nu_f_delta
                        if abs(spring_signal) > 1e-6:
                            spring_rationale = (
                                f"{base_rationale} Target νf_susp {target_nu_f:.2f} versus measured "
                                f"{actual_nu_f:.2f} ({nu_f_delta:+.2f}). Mean ∇NFR⊥ {avg_lat:.2f} "
                                f"(target {target_lat:.2f}, Δ {lat_dev:+.2f}). The spring adjustment aims "
                                "to equalise ∇NFR⊥ within the active G band."
                            )
                            spring_recommendations = _phase_action_recommendations(
                                phase=self.phase,
                                category=self.category,
                                metric="delta_nfr_proj_lateral",
                                raw_value=spring_signal,
                                base_rationale=spring_rationale,
                                priority=self.priority - 2,
                                reference_key=self.reference_key,
                                node="suspension",
                            )
                            desired_parameter = (
                                "rear_spring_stiffness" if phase_key == "exit" else "front_spring_stiffness"
                            )
                            spring_recommendations = [
                                rec
                                for rec in spring_recommendations
                                if rec.parameter == desired_parameter
                            ]
            if spring_recommendations:
                recommendations.extend(spring_recommendations)
            adjustments: List[Recommendation] = []
            if delta_triggered:
                adjustments = _phase_action_recommendations(
                    phase=self.phase,
                    category=self.category,
                    metric="delta_nfr",
                    raw_value=deviation,
                    base_rationale=base_rationale,
                    priority=self.priority,
                    reference_key=self.reference_key,
                    gradient=getattr(goal, "track_gradient", None),
                )
            if adjustments and phase_key == "entry":
                for recommendation in adjustments:
                    if recommendation.parameter != "brake_bias_pct":
                        continue
                    if recommendation.delta is not None:
                        if bias_direction == "forward":
                            recommendation.delta = abs(recommendation.delta)
                        elif bias_direction == "rearward":
                            recommendation.delta = -abs(recommendation.delta)
                        recommendation.message = (
                            f"{recommendation.delta:+.1f}% brake bias forward"
                        )
                        recommendation.rationale = (
                            f"{base_rationale} Suggested action: {recommendation.message}. "
                            f"Refer to {MANUAL_REFERENCES[self.reference_key]} to apply the adjustment."
                        )
                    if worst_ratio >= 1.2:
                        recommendation.priority = min(
                            recommendation.priority, self.priority - 3
                        )
                    elif worst_ratio >= 1.0:
                        recommendation.priority = min(
                            recommendation.priority, self.priority - 2
                        )
            if adjustments:
                _boost_geometry_priority(adjustments, self.priority, geometry_urgency)
                recommendations.extend(adjustments)

            if delta_triggered:
                direction = "increase" if deviation < 0 else "reduce"
                summary_message = (
                    f"{self.operator_label} · global ΔNFR target: {direction} ΔNFR "
                    f"in microsector {microsector.index} ({MANUAL_REFERENCES[self.reference_key]})"
                )
                summary_rationale = (
                    f"{base_rationale} Recommendation: {direction} global ΔNFR and review "
                    f"{MANUAL_REFERENCES[self.reference_key]} for the {self.phase_label} phase."
                )
                recommendations.append(
                    Recommendation(
                        category=self.category,
                        message=summary_message,
                        rationale=summary_rationale,
                        priority=self.priority + 40,
                    )
                )

            axis_focus_trigger = delta_triggered or geometry_urgency > 0
            if (
                axis_focus_trigger
                and axis_bias
                and (
                    lag_gap > 0.05
                    or (
                        axis_bias == "longitudinal" and abs_long > axis_delta_threshold
                    )
                    or (
                        axis_bias == "lateral" and abs_lat > axis_delta_threshold
                    )
                )
            ):
                descriptor = _axis_focus_descriptor(goal.phase, axis_bias)
                if descriptor is not None:
                    focus_message, focus_reference, parameters = descriptor
                    axis_label = "∥" if axis_bias == "longitudinal" else "⊥"
                    axis_delta = long_dev if axis_bias == "longitudinal" else lat_dev
                    target_value = target_long if axis_bias == "longitudinal" else target_lat
                    coherence_line = ""
                    if geometry_snapshot["coherence_gap"] > _COHERENCE_THRESHOLD:
                        coherence_line = (
                            f" Mean C(t) {geometry_snapshot['average_coherence']:.2f} "
                            f"(target {geometry_snapshot['target_coherence']:.2f})."
                        )
                        focus_rationale = (
                            f"{base_rationale} ΔNFR{axis_label} dominates ({axis_delta:+.2f} versus target "
                            f"{target_value:+.2f}). Measured θ {lag_value:+.2f}rad (target {target_lag:+.2f}). "
                            f"Target split ∥ {weight_long:.2f} · ⊥ {weight_lat:.2f}.{coherence_line}"
                        )
                    recommendations.append(
                        Recommendation(
                            category=self.category,
                            message=focus_message,
                            rationale=f"{focus_rationale} Refer to {MANUAL_REFERENCES[focus_reference]}.",
                            priority=min(self.priority - 2, self.priority - geometry_urgency),
                        )
                    )
                    for rec in recommendations[start_index:]:
                        if rec.parameter and rec.parameter in parameters:
                            rec.priority = min(rec.priority, self.priority - 1)

            target_si = getattr(goal, "target_sense_index", None)
            if target_si is not None:
                actual_si = mean(bundle.sense_index for bundle in samples)
                si_delta = target_si - actual_si
                si_tolerance = 0.01
                if abs(si_delta) > si_tolerance:
                    si_rationale = (
                        f"Sense index target was {target_si:.2f} and {actual_si:.2f} was observed "
                        f"({si_delta:+.2f})."
                    )
                    recommendations.extend(
                        _phase_action_recommendations(
                            phase=self.phase,
                            category=self.category,
                            metric="sense_index",
                            raw_value=si_delta,
                            base_rationale=f"{base_rationale} {si_rationale}",
                            priority=self.priority + 5,
                            reference_key=self.reference_key,
                        )
                    )
            _apply_priority_scale(recommendations[start_index:], priority_scale)
        return recommendations


class PhaseNodeOperatorRule:
    """Reinforce operator actions using dominant nodes and ν_f objectives."""

    def __init__(
        self,
        *,
        phase: str,
        operator_label: str,
        category: str,
        priority: int,
        reference_key: str,
    ) -> None:
        self.phase = phase
        self.operator_label = operator_label
        self.category = category
        self.priority = priority
        self.reference_key = reference_key

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None:
            return []

        priority_scale = _session_priority_scale(context, self.phase)
        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            start_index = len(recommendations)
            goal = _goal_for_phase(microsector, self.phase)
            if goal is None:
                continue
            indices = list(microsector.phase_indices(goal.phase))
            if not indices:
                continue
            samples = _phase_samples(results, indices)
            dominant_nodes = goal.dominant_nodes or microsector.dominant_nodes.get(
                goal.phase, ()
            )
            if not dominant_nodes:
                continue
            target_nu_f = float(goal.nu_f_target)
            target_si = context.objectives.target_sense_index
            tolerance_scale = max(0.5, min(1.5, 1.0 + (0.75 - target_si)))
            tolerance = max(0.05, abs(target_nu_f) * 0.2 * tolerance_scale)
            geometry_snapshot = _geometry_snapshot(goal, microsector, samples, context)
            phase_key = phase_family(self.phase)
            geometry_nodes = _GEOMETRY_METRIC_NODES.get(phase_key, {})
            measured_lag = geometry_snapshot["measured_lag"]
            measured_alignment = geometry_snapshot["measured_alignment"]
            target_lag = geometry_snapshot["target_lag"]
            target_alignment = geometry_snapshot["target_alignment"]
            flip_alignment = _should_flip_alignment(
                measured_alignment, target_alignment, measured_lag, target_lag
            )
            geometry_actions_map: Dict[str, Recommendation] = {}
            geometry_urgency = _geometry_urgency(geometry_snapshot)
            dominant_list = ", ".join(_node_label(name) for name in goal.dominant_nodes)
            geometry_context = (
                f"{self.operator_label} applied in microsector {microsector.index} "
                f"with dominant nodes {dominant_list or 'the reference nodes'}."
            )
            if (
                geometry_snapshot["alignment_gap"] > _ALIGNMENT_THRESHOLD
                and geometry_nodes.get("phase_alignment")
            ):
                alignment_rationale = (
                    f"{geometry_context} Target Siφ {geometry_snapshot['target_alignment']:+.2f} "
                    f"versus measured {geometry_snapshot['measured_alignment']:+.2f} "
                    f"({geometry_snapshot['alignment_delta']:+.2f})."
                )
                alignment_actions = _phase_action_recommendations(
                    phase=self.phase,
                    category=self.category,
                    metric="phase_alignment",
                    raw_value=geometry_snapshot["alignment_delta"],
                    base_rationale=alignment_rationale,
                    priority=self.priority - 1,
                    reference_key=self.reference_key,
                    node=geometry_nodes.get("phase_alignment"),
                )
                geometry_actions_map = _merge_recommendations_by_parameter(
                    geometry_actions_map, alignment_actions
                )
            if (
                geometry_snapshot["lag_gap"] > _LAG_THRESHOLD
                and geometry_nodes.get("phase_lag")
            ):
                lag_rationale = (
                    f"{geometry_context} Measured θ {geometry_snapshot['measured_lag']:+.2f}rad "
                    f"versus target {geometry_snapshot['target_lag']:+.2f}rad "
                    f"({geometry_snapshot['lag_delta']:+.2f})."
                )
                lag_actions = _phase_action_recommendations(
                    phase=self.phase,
                    category=self.category,
                    metric="phase_lag",
                    raw_value=geometry_snapshot["lag_delta"],
                    base_rationale=lag_rationale,
                    priority=self.priority - 1,
                    reference_key=self.reference_key,
                    node=geometry_nodes.get("phase_lag"),
                )
                geometry_actions_map = _merge_recommendations_by_parameter(
                    geometry_actions_map, lag_actions
                )
            if (
                geometry_snapshot["coherence_gap"] > _COHERENCE_THRESHOLD
                and geometry_nodes.get("coherence_index")
            ):
                coherence_rationale = (
                    f"{geometry_context} Mean C(t) {geometry_snapshot['average_coherence']:.2f} "
                    f"(target {geometry_snapshot['target_coherence']:.2f}, Δ "
                    f"{geometry_snapshot['coherence_delta']:+.2f})."
                )
                coherence_actions = _phase_action_recommendations(
                    phase=self.phase,
                    category=self.category,
                    metric="coherence_index",
                    raw_value=geometry_snapshot["coherence_delta"],
                    base_rationale=coherence_rationale,
                    priority=self.priority - 1,
                    reference_key=self.reference_key,
                    node=geometry_nodes.get("coherence_index"),
                )
                geometry_actions_map = _merge_recommendations_by_parameter(
                    geometry_actions_map, coherence_actions
                )
            geometry_recommendations = list(geometry_actions_map.values())
            if geometry_recommendations:
                _boost_geometry_priority(
                    geometry_recommendations, self.priority, max(geometry_urgency, 1)
                )
                recommendations.extend(geometry_recommendations)
            for node in dominant_nodes:
                node_values = _node_nu_f_values(results, indices, node)
                if not node_values:
                    continue
                actual_nu_f = mean(node_values)
                deviation = actual_nu_f - target_nu_f
                if abs(deviation) <= tolerance:
                    continue
                node_label = _node_label(node)
                dominant_list = ", ".join(_node_label(name) for name in goal.dominant_nodes)
                suggested_delta = -deviation if flip_alignment else deviation
                alignment_summary = (
                    f"θ {measured_lag:+.2f}rad / Siφ {measured_alignment:+.2f} "
                    f"(target θ {target_lag:+.2f}rad / Siφ {target_alignment:+.2f})."
                )
                frequency_label = ""
                classification = ""
                coherence_index = 0.0
                if indices:
                    last_bundle = results[indices[-1]]
                    frequency_label = getattr(last_bundle, "nu_f_label", "")
                    classification = getattr(last_bundle, "nu_f_classification", "")
                    coherence_index = float(getattr(last_bundle, "coherence_index", 0.0))
                classification_summary = ""
                if frequency_label:
                    display_label = frequency_label[3:] if frequency_label.startswith("ν_f ") else frequency_label
                    classification_summary = f" ν_f status: {display_label}."
                elif classification:
                    classification_summary = f" ν_f classification {classification}."
                if coherence_index > 0.0:
                    classification_summary += f" C(t) {coherence_index:.2f}."
                sense_summary = (
                    f" Profile Si target {target_si:.2f} for {context.profile_label}."
                )
                coherence_line = (
                    f" Mean C(t) {geometry_snapshot['average_coherence']:.2f} "
                    f"(target {geometry_snapshot['target_coherence']:.2f})."
                )
                base_rationale = (
                    f"{self.operator_label} applied to node {node_label} in microsector "
                    f"{microsector.index}. The target strategy highlights "
                    f"{dominant_list or 'the dominant nodes'} and sets ν_f={target_nu_f:.2f}. "
                    f"Measured mean ν_f {actual_nu_f:.2f} ({deviation:+.2f}) exceeds the "
                    f"adjusted tolerance ±{tolerance:.2f} defined for {context.profile_label}. "
                    f"{alignment_summary}{classification_summary}{sense_summary}{coherence_line}"
                )
                if flip_alignment:
                    base_rationale += " The adjustment direction is inverted to recover phase alignment."
                actions = _phase_action_recommendations(
                    phase=self.phase,
                    category=self.category,
                    metric="nu_f",
                    raw_value=suggested_delta,
                    base_rationale=base_rationale,
                    priority=self.priority,
                    reference_key=self.reference_key,
                    node=node,
                )
                if geometry_urgency > 0 and actions:
                    _boost_geometry_priority(actions, self.priority, geometry_urgency)
                if actions:
                    recommendations.extend(actions)
                    continue
                direction = "increase" if suggested_delta < 0 else "decrease"
                action_text = _slider_action(self.phase, node, direction)
                recommendations.append(
                    Recommendation(
                        category=self.category,
                        message=(
                            f"{self.operator_label} · target node {node_label}: {action_text} "
                            f"to approach ν_f {target_nu_f:.2f} "
                            f"({MANUAL_REFERENCES[self.reference_key]})"
                        ),
                        rationale=(
                            f"{base_rationale} Suggested action: {action_text}. Refer to "
                            f"{MANUAL_REFERENCES[self.reference_key]} for the adjustments."
                        ),
                        priority=self.priority,
                    )
                )
            _apply_priority_scale(recommendations[start_index:], priority_scale)
        return recommendations


class ParallelSteerRule:
    """React to Ackermann steering deviations using the aggregated index."""

    def __init__(
        self,
        priority: int = 20,
        threshold: float = 0.08,
        delta_step: float = 0.1,
        lock_step: float = 0.5,
    ) -> None:
        self.priority = int(priority)
        self.threshold = float(threshold)
        self.delta_step = float(delta_step)
        self.lock_step = float(lock_step)

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors:
            return []

        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            measures = getattr(microsector, "filtered_measures", {}) or {}
            deviation = _safe_float(measures.get("ackermann_parallel_index", 0.0), 0.0)
            if not math.isfinite(deviation):
                continue
            magnitude = abs(deviation)
            if magnitude <= self.threshold:
                continue
            budget = max(0.0, min(1.0, _safe_float(measures.get("slide_catch_budget"), 0.0)))
            yaw_ratio = max(
                0.0,
                min(1.0, _safe_float(measures.get("slide_catch_budget_yaw"), 0.0)),
            )
            steer_ratio = max(
                0.0,
                min(1.0, _safe_float(measures.get("slide_catch_budget_steer"), 0.0)),
            )
            overshoot_ratio = max(
                0.0,
                min(1.0, _safe_float(measures.get("slide_catch_budget_overshoot"), 0.0)),
            )
            if deviation < 0.0:
                steer_action = "increase"
                steer_delta = self.delta_step
                lock_delta = self.lock_step
            else:
                steer_action = "reduce"
                steer_delta = -self.delta_step
                lock_delta = -self.lock_step
            rationale_header = (
                f"Average Ackermann offset {deviation:+.3f}rad exceeds the {self.threshold:.3f} threshold."
            )
            slide_context = (
                f" Slide Catch Budget {budget:.2f} · yaw {yaw_ratio:.2f} · steering {steer_ratio:.2f} · overshoot {overshoot_ratio:.2f}."
            )
            message = (
                f"Ackermann operator: {steer_action} parallel steer in microsector {microsector.index}"
            )
            rationale = (
                f"{rationale_header}{slide_context} Adjust parallel steer to approach theoretical torque "
                f"({MANUAL_REFERENCES['tyre_balance']})."
            )
            recommendations.append(
                Recommendation(
                    category="entry",
                    message=message,
                    rationale=rationale,
                    priority=self.priority,
                    parameter="parallel_steer",
                    delta=steer_delta,
                )
            )

            if budget < 0.45:
                lock_action = "increase" if lock_delta > 0 else "reduce"
                lock_message = (
                    f"Ackermann operator: {lock_action} steering lock in microsector {microsector.index}"
                )
                lock_rationale = (
                    f"{rationale_header}{slide_context} Correction margin is limited, so adjust steering lock "
                    f"to ease slide recovery."
                )
                recommendations.append(
                    Recommendation(
                        category="entry",
                        message=lock_message,
                        rationale=lock_rationale,
                        priority=self.priority + 2,
                        parameter="steering_lock_deg",
                        delta=lock_delta,
                    )
                )
        return recommendations


class TyreBalanceRule:
    """Recommend ΔP and camber tweaks from CPHI telemetry."""

    def __init__(
        self,
        priority: int = 18,
        target_front: float = 0.82,
        target_rear: float = 0.80,
        *,
        dispersion_pressure_baseline: float = 0.5,
        dispersion_camber_baseline: float = 0.35,
        dispersion_cutoff: float = 0.05,
        max_dispersion_scale: float = 1.5,
    ) -> None:
        self.priority = priority
        self.target_front = target_front
        self.target_rear = target_rear
        self.dispersion_pressure_baseline = max(1e-6, float(dispersion_pressure_baseline))
        self.dispersion_camber_baseline = max(1e-6, float(dispersion_camber_baseline))
        self.dispersion_cutoff = max(0.0, float(dispersion_cutoff))
        self.max_dispersion_scale = max(1.0, float(max_dispersion_scale))

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None:
            return []

        def _average(values: Sequence[float]) -> float:
            return mean(values) if values else 0.0

        def _finite(value: object) -> float | None:
            numeric = _safe_float(value)
            return numeric if math.isfinite(numeric) else None

        telemetry_available = False
        for microsector in microsectors:
            measures = getattr(microsector, "filtered_measures", {}) or {}
            if not isinstance(measures, Mapping):
                continue
            if any(
                _finite(measures.get(f"cphi_{suffix}")) is not None
                for suffix in WHEEL_SUFFIXES
            ):
                telemetry_available = True
                break
        if not telemetry_available:
            return []

        controls: List[TyreBalanceControlOutput] = []
        delta_flat_values: List[float] = []
        per_wheel: Dict[str, List[float]] = {key: [] for key in WHEEL_SUFFIXES}
        cphi_values: Dict[str, List[float]] = {key: [] for key in WHEEL_SUFFIXES}
        cphi_temperature_components: Dict[str, List[float]] = {
            key: [] for key in WHEEL_SUFFIXES
        }
        cphi_gradient_components: Dict[str, List[float]] = {
            key: [] for key in WHEEL_SUFFIXES
        }
        cphi_mu_components: Dict[str, List[float]] = {key: [] for key in WHEEL_SUFFIXES}
        cphi_temp_deltas: Dict[str, List[float]] = {key: [] for key in WHEEL_SUFFIXES}
        cphi_gradient_rates: Dict[str, List[float]] = {key: [] for key in WHEEL_SUFFIXES}

        for microsector in microsectors:
            metrics = microsector.filtered_measures
            control = tyre_balance_controller(
                metrics,
                target_front=self.target_front,
                target_rear=self.target_rear,
                offsets=context.tyre_offsets,
            )
            controls.append(control)
            delta_flat_values.append(_safe_float(metrics.get("d_nfr_flat")))
            for key, value in control.per_wheel_pressure.items():
                per_wheel.setdefault(key, []).append(float(value))
            for suffix in WHEEL_SUFFIXES:
                value = metrics.get(f"cphi_{suffix}")
                if value is not None:
                    numeric = _safe_float(value)
                    if math.isfinite(numeric):
                        cphi_values.setdefault(suffix, []).append(numeric)
                temp_component = metrics.get(f"cphi_{suffix}_temperature")
                if temp_component is not None:
                    numeric = _safe_float(temp_component)
                    if math.isfinite(numeric):
                        cphi_temperature_components.setdefault(suffix, []).append(
                            numeric
                        )
                gradient_component = metrics.get(f"cphi_{suffix}_gradient")
                if gradient_component is not None:
                    numeric = _safe_float(gradient_component)
                    if math.isfinite(numeric):
                        cphi_gradient_components.setdefault(suffix, []).append(
                            numeric
                        )
                mu_component = metrics.get(f"cphi_{suffix}_mu")
                if mu_component is not None:
                    numeric = _safe_float(mu_component)
                    if math.isfinite(numeric):
                        cphi_mu_components.setdefault(suffix, []).append(numeric)
                temp_delta = metrics.get(f"cphi_{suffix}_temp_delta")
                if temp_delta is not None:
                    numeric = _safe_float(temp_delta)
                    if math.isfinite(numeric):
                        cphi_temp_deltas.setdefault(suffix, []).append(numeric)
                gradient_rate = metrics.get(f"cphi_{suffix}_gradient_rate")
                if gradient_rate is not None:
                    numeric = _safe_float(gradient_rate)
                    if math.isfinite(numeric) and numeric > 0.0:
                        cphi_gradient_rates.setdefault(suffix, []).append(numeric)

        if not controls:
            return []

        front_pressure = _average([control.pressure_delta_front for control in controls])
        rear_pressure = _average([control.pressure_delta_rear for control in controls])
        camber_front = _average([control.camber_delta_front for control in controls])
        camber_rear = _average([control.camber_delta_rear for control in controls])
        per_wheel_avg = {key: _average(values) for key, values in per_wheel.items()}

        def _aggregate(source: Mapping[str, Sequence[float]], default: float = 0.0) -> Dict[str, float]:
            return {
                key: _average(values) if values else default
                for key, values in source.items()
            }

        cphi_average = _aggregate(cphi_values, 1.0)
        cphi_minimum = {
            key: min(values) if values else 1.0
            for key, values in cphi_values.items()
        }
        cphi_temperature_average = _aggregate(cphi_temperature_components)
        cphi_gradient_average = _aggregate(cphi_gradient_components)
        cphi_mu_average = _aggregate(cphi_mu_components)
        cphi_temp_delta_average = _aggregate(cphi_temp_deltas)
        cphi_gradient_rate_average = _aggregate(cphi_gradient_rates)

        avg_delta_flat = _average(delta_flat_values)

        def _scale_delta(delta: float, dispersion: float, baseline: float) -> float:
            if abs(delta) <= 1e-6:
                return 0.0
            if dispersion <= self.dispersion_cutoff:
                return 0.0
            scale = dispersion / baseline if baseline > 1e-6 else 1.0
            scale = min(self.max_dispersion_scale, max(0.0, scale))
            adjusted = delta * scale
            if abs(adjusted) < 1e-3:
                return 0.0
            return adjusted

        recommendations: List[Recommendation] = []

        def _axle_values(keys: Sequence[str], lookup: Mapping[str, float]) -> List[float]:
            return [lookup.get(suffix, 0.0) for suffix in keys]

        def _avg(values: Sequence[float]) -> float:
            return _average([float(value) for value in values])

        front_suffixes = ("fl", "fr")
        rear_suffixes = ("rl", "rr")
        front_gradient_rates = _axle_values(front_suffixes, cphi_gradient_rate_average)
        rear_gradient_rates = _axle_values(rear_suffixes, cphi_gradient_rate_average)
        avg_front_dispersion = _avg(front_gradient_rates)
        avg_rear_dispersion = _avg(rear_gradient_rates)

        base_front_pressure = front_pressure
        base_rear_pressure = rear_pressure
        front_pressure = _scale_delta(front_pressure, avg_front_dispersion, self.dispersion_pressure_baseline)
        rear_pressure = _scale_delta(rear_pressure, avg_rear_dispersion, self.dispersion_pressure_baseline)
        cphi_threshold = 0.78

        def _select_action(cphi_value: float, temp: float, gradient: float, mu_component: float) -> str | None:
            if cphi_value >= cphi_threshold:
                return None
            dominant = max(
                (temp, "temperature"),
                (gradient, "gradient"),
                (mu_component, "mu"),
                key=lambda item: item[0],
            )[1]
            return "camber" if dominant == "gradient" else "pressure"

        front_cphi = min(_axle_values(front_suffixes, cphi_minimum))
        rear_cphi = min(_axle_values(rear_suffixes, cphi_minimum))
        front_temp_component = _avg(_axle_values(front_suffixes, cphi_temperature_average))
        rear_temp_component = _avg(_axle_values(rear_suffixes, cphi_temperature_average))
        front_gradient_component = _avg(_axle_values(front_suffixes, cphi_gradient_average))
        rear_gradient_component = _avg(_axle_values(rear_suffixes, cphi_gradient_average))
        front_mu_component = _avg(_axle_values(front_suffixes, cphi_mu_average))
        rear_mu_component = _avg(_axle_values(rear_suffixes, cphi_mu_average))
        front_temp_delta = _avg(_axle_values(front_suffixes, cphi_temp_delta_average))
        rear_temp_delta = _avg(_axle_values(rear_suffixes, cphi_temp_delta_average))

        front_action = _select_action(
            front_cphi, front_temp_component, front_gradient_component, front_mu_component
        )
        rear_action = _select_action(
            rear_cphi, rear_temp_component, rear_gradient_component, rear_mu_component
        )

        if front_action != "pressure":
            front_pressure = 0.0
        else:
            front_pressure *= max(0.0, 1.0 - front_cphi)
        if rear_action != "pressure":
            rear_pressure = 0.0
        else:
            rear_pressure *= max(0.0, 1.0 - rear_cphi)

        if front_action != "camber":
            camber_front = 0.0
        else:
            camber_front *= max(0.0, 1.0 - front_cphi)
        if rear_action != "camber":
            camber_rear = 0.0
        else:
            camber_rear *= max(0.0, 1.0 - rear_cphi)

        def _format_cphi_label(
            label: str,
            value: float,
            bias: str,
            temp_component: float,
            gradient_component: float,
            mu_component: float,
        ) -> str:
            return (
                f"{label} {value:.2f} ({bias}, T {temp_component:.2f}, "
                f"G {gradient_component:.2f}, μ {mu_component:.2f})"
            )

        front_bias_label = "outer" if front_temp_delta >= 0.0 else "inner"
        rear_bias_label = "outer" if rear_temp_delta >= 0.0 else "inner"

        front_scale = (
            front_pressure / base_front_pressure
            if abs(base_front_pressure) > 1e-6
            else 0.0
        )
        rear_scale = (
            rear_pressure / base_rear_pressure
            if abs(base_rear_pressure) > 1e-6
            else 0.0
        )
        per_wheel_scaled = {}
        for suffix, value in per_wheel_avg.items():
            scale = front_scale if suffix in {"fl", "fr"} else rear_scale
            per_wheel_scaled[suffix] = value * scale

        if abs(front_pressure) > 0.01 or abs(rear_pressure) > 0.01:
            pressure_rationale = (
                f"ΔNFR_flat {avg_delta_flat:+.2f}. "
                f"Gradient dispersion front {avg_front_dispersion:.3f} · rear {avg_rear_dispersion:.3f}. "
                f"Per-wheel ΔP {per_wheel_scaled.get('fl', 0.0):+.2f}/{per_wheel_scaled.get('fr', 0.0):+.2f}/"
                f"{per_wheel_scaled.get('rl', 0.0):+.2f}/{per_wheel_scaled.get('rr', 0.0):+.2f}."
            )
            cphi_pressure_segments: list[str] = []
            if front_action == "pressure":
                cphi_pressure_segments.append(
                    _format_cphi_label(
                        "front",
                        front_cphi,
                        front_bias_label,
                        front_temp_component,
                        front_gradient_component,
                        front_mu_component,
                    )
                )
            if rear_action == "pressure":
                cphi_pressure_segments.append(
                    _format_cphi_label(
                        "rear",
                        rear_cphi,
                        rear_bias_label,
                        rear_temp_component,
                        rear_gradient_component,
                        rear_mu_component,
                    )
                )
            if cphi_pressure_segments:
                pressure_rationale += f" CPHI {' · '.join(cphi_pressure_segments)}."
            recommendations.append(
                Recommendation(
                    category="tyres",
                    message=(
                        "Thermal operator: adjust ΔPfront "
                        f"{front_pressure:+.2f} / ΔPrear {rear_pressure:+.2f} bar "
                        f"({MANUAL_REFERENCES['tyre_balance']})"
                    ),
                    rationale=(
                        f"{pressure_rationale} Follow {MANUAL_REFERENCES['tyre_balance']} guidance "
                        "to implement the adjustments."
                    ),
                    priority=self.priority,
                    parameter="tyre_pressure",
                    delta=front_pressure,
                )
            )

        base_camber_front = camber_front
        base_camber_rear = camber_rear
        camber_front = _scale_delta(camber_front, avg_front_dispersion, self.dispersion_camber_baseline)
        camber_rear = _scale_delta(camber_rear, avg_rear_dispersion, self.dispersion_camber_baseline)

        if abs(camber_front) > 0.01 or abs(camber_rear) > 0.01:
            camber_rationale = (
                f"CPHI gradient front {front_gradient_component:.2f} · rear {rear_gradient_component:.2f}. "
                f"Gradient dispersion front {avg_front_dispersion:.3f} · rear {avg_rear_dispersion:.3f}."
            )
            cphi_camber_segments: list[str] = []
            if front_action == "camber":
                cphi_camber_segments.append(
                    _format_cphi_label(
                        "front",
                        front_cphi,
                        front_bias_label,
                        front_temp_component,
                        front_gradient_component,
                        front_mu_component,
                    )
                )
            if rear_action == "camber":
                cphi_camber_segments.append(
                    _format_cphi_label(
                        "rear",
                        rear_cphi,
                        rear_bias_label,
                        rear_temp_component,
                        rear_gradient_component,
                        rear_mu_component,
                    )
                )
            if cphi_camber_segments:
                camber_rationale += f" CPHI {' · '.join(cphi_camber_segments)}."
            recommendations.append(
                Recommendation(
                    category="tyres",
                    message=(
                        "Thermal operator: adjust camber Δfront "
                        f"{camber_front:+.2f}° / Δrear {camber_rear:+.2f}° "
                        f"({MANUAL_REFERENCES['tyre_balance']})"
                    ),
                    rationale=(
                        f"{camber_rationale} Adjust camber following {MANUAL_REFERENCES['tyre_balance']}."
                    ),
                    priority=self.priority + 1,
                    parameter="camber",
                    delta=camber_front,
                )
            )

        return recommendations


class FootprintEfficiencyRule:
    """Reduce ΔNFR guidance when the tyre footprint is saturated."""

    def __init__(
        self,
        priority: int = 16,
        *,
        threshold: float = 0.9,
        hysteresis: float = 0.1,
    ) -> None:
        self.priority = int(priority)
        self.threshold = float(threshold)
        self.hysteresis = float(hysteresis)

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None or not results:
            return []

        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            measures = getattr(microsector, "filtered_measures", {}) or {}
            if not isinstance(measures, Mapping):
                continue
            base_front_ratio = float(measures.get("mu_usage_front_ratio", 0.0))
            base_rear_ratio = float(measures.get("mu_usage_rear_ratio", 0.0))
            phase_front_ratio = float(
                measures.get("phase_mu_usage_front_ratio", base_front_ratio)
            )
            phase_rear_ratio = float(
                measures.get("phase_mu_usage_rear_ratio", base_rear_ratio)
            )
            front_ratio = phase_front_ratio if phase_front_ratio > 0.0 else base_front_ratio
            rear_ratio = phase_rear_ratio if phase_rear_ratio > 0.0 else base_rear_ratio
            if front_ratio < self.threshold and rear_ratio < self.threshold:
                continue
            goal = _goal_for_phase(microsector, getattr(microsector, "active_phase", ""))
            if goal is None:
                continue
            indices = list(microsector.phase_indices(goal.phase))
            samples = _phase_samples(results, indices)
            if not samples:
                continue
            avg_delta = mean(bundle.delta_nfr for bundle in samples)
            if not math.isfinite(avg_delta):
                continue
            tolerance = context.thresholds.tolerance_for_phase(goal.phase)
            allowable = tolerance * (1.0 + self.hysteresis)
            deviation = abs(avg_delta - goal.target_delta_nfr)
            if deviation > allowable:
                continue
            category = phase_family(goal.phase)
            for axle_label, ratio_value, symbol, reference_key in (
                ("front", front_ratio, "F", "antiroll"),
                ("rear", rear_ratio, "R", "differential"),
            ):
                if ratio_value < self.threshold:
                    continue
                reference = MANUAL_REFERENCES[reference_key]
                message = (
                    f"Footprint operator: relieve ΔNFR {axle_label} axle in microsector {microsector.index}"
                )
                rationale = (
                    f"Footprint usage μ{symbol} {ratio_value:.2f} exceeds the {self.threshold:.2f} threshold "
                    f"with mean ΔNFR {avg_delta:.2f} (target {goal.target_delta_nfr:.2f}, "
                    f"tolerance ±{tolerance:.2f}). Reduce axle load following {reference}."
                )
                recommendations.append(
                    Recommendation(
                        category=category,
                        message=message,
                        rationale=rationale,
                        priority=self.priority,
                    )
                )
        return recommendations


class CurbComplianceRule:
    """Analyses support events (pianos) against the ΔNFR target."""

    def __init__(self, priority: int) -> None:
        self.priority = priority

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None:
            return []

        tolerance = context.thresholds.piano_delta_tolerance
        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            if not microsector.support_event:
                continue
            goal = _goal_for_phase(microsector, "apex")
            if goal is None:
                continue
            indices = microsector.phase_indices(goal.phase)
            tyre_samples = [
                results[i].tyres.delta_nfr
                for i in indices
                if 0 <= i < len(results)
            ]
            if not tyre_samples:
                continue
            actual_delta = mean(tyre_samples)
            deviation = actual_delta - goal.target_delta_nfr
            if abs(deviation) <= tolerance:
                continue
            action = "release" if deviation > 0 else "load"
            recommendations.append(
                Recommendation(
                    category="pianos",
                    message=(
                        f"Kerb operator: {action} support in microsector {microsector.index}"
                        f" ({MANUAL_REFERENCES['curbs']})"
                    ),
                    rationale=(
                        "Kerbs registered a mean ΔNFR of "
                        f"{actual_delta:.2f} versus the target {goal.target_delta_nfr:.2f}. "
                        f"The deviation {deviation:+.2f} exceeds the ±{tolerance:.2f} tolerance "
                        f"configured for {context.profile_label}. "
                        f"Follow {MANUAL_REFERENCES['curbs']} to modulate support."
                    ),
                    priority=self.priority,
                )
            )
        return recommendations


class DetuneRatioRule:
    """Escalate bar/damper guidance when detune ratio collapses under load."""

    def __init__(self, priority: int = 24, resonance_threshold: float = 0.5) -> None:
        self.priority = priority
        self.resonance_threshold = resonance_threshold

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None:
            return []

        threshold = context.thresholds.rho_detune_threshold
        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            rho = float(microsector.filtered_measures.get("rho", 1.0))
            d_nfr_res = float(microsector.filtered_measures.get("d_nfr_res", 0.0))
            if rho <= 0.0 or rho >= threshold:
                continue
            if abs(d_nfr_res) <= self.resonance_threshold:
                continue
            goal = _goal_for_phase(microsector, microsector.active_phase)
            target_rho = getattr(goal, "rho_target", None) if goal else None
            target_exc = getattr(goal, "nu_exc_target", None) if goal else None
            detune_weights = getattr(goal, "detune_ratio_weights", {}) if goal else {}
            category = phase_family(microsector.active_phase)
            target_text = ""
            if target_rho:
                target_text += f" Target ρ≈{target_rho:.2f}."
            if target_exc:
                target_text += f" Reference ν_exc {target_exc:.2f}Hz."
            focus_text = ""
            if isinstance(detune_weights, Mapping):
                long_weight = float(detune_weights.get("longitudinal", 0.5))
                lat_weight = float(detune_weights.get("lateral", 0.5))
                if long_weight > lat_weight:
                    focus_text = " Longitudinal detune priority."
                elif lat_weight > long_weight:
                    focus_text = " Lateral detune priority."
            rationale = (
                f"Detune ratio ρ={rho:.2f} in microsector {microsector.index}"
                f" drops below the {threshold:.2f} threshold with ∇Res {d_nfr_res:+.2f}."
                f"{target_text}{focus_text} Review anti-roll bars and dampers"
                f" ({MANUAL_REFERENCES['antiroll']})."
            )
            message = (
                f"Modal operator · microsector {microsector.index}:"
                " adjust bars/dampers to raise ρ"
            )
            recommendations.append(
                Recommendation(
                    category=category,
                    message=message,
                    rationale=rationale,
                    priority=self.priority,
                )
            )
        return recommendations


class ShiftStabilityRule:
    """Escalate gearing guidance when apex→exit shifts destabilise the exit."""

    def __init__(
        self,
        priority: int = 28,
        *,
        stability_threshold: float = 0.75,
        gear_match_threshold: float = 0.7,
    ) -> None:
        self.priority = priority
        self.stability_threshold = max(0.0, min(1.0, stability_threshold))
        self.gear_match_threshold = max(0.0, min(1.0, gear_match_threshold))

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors:
            return []

        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            measures = getattr(microsector, "filtered_measures", {}) or {}
            try:
                stability = float(measures.get("shift_stability", 1.0))
                gear_match = float(measures.get("exit_gear_match", 1.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(stability) or not math.isfinite(gear_match):
                continue
            stability_trigger = stability < self.stability_threshold
            gear_trigger = gear_match < self.gear_match_threshold
            if not (stability_trigger or gear_trigger):
                continue
            details: List[str] = []
            if stability_trigger:
                details.append(
                    f"stability {stability:.2f} < threshold {self.stability_threshold:.2f}"
                )
            if gear_trigger:
                details.append(
                    f"gear synchrony {gear_match:.2f} < threshold {self.gear_match_threshold:.2f}"
                )
            detail_text = ", ".join(details)
            message = (
                "Transmission operator: smooth apex→exit shifts in "
                f"microsector {microsector.index}"
            )
            rationale = (
                "Transmission indicators show losses during the apex→exit transition: "
                f"{detail_text}. Adjust the final drive or gear ratios to reduce forced shifts "
                f"({MANUAL_REFERENCES['differential']})."
            )
            recommendations.append(
                Recommendation(
                    category="exit",
                    message=message,
                    rationale=rationale,
                    priority=self.priority,
                )
            )
        return recommendations


class LockingWindowRule:
    """Tighten differential guidance based on locking transition stability."""

    def __init__(
        self,
        priority: int = 27,
        *,
        on_threshold: float = 0.7,
        off_threshold: float = 0.75,
        min_transitions: int = 2,
        power_lock_step: float = 5.0,
        preload_step: float = 40.0,
    ) -> None:
        self.priority = int(priority)
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.min_transitions = max(0, int(min_transitions))
        self.power_lock_step = abs(float(power_lock_step))
        self.preload_step = abs(float(preload_step))

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors:
            return []

        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            phase = getattr(microsector, "active_phase", "")
            if phase_family(phase) != "exit":
                continue
            measures = getattr(microsector, "filtered_measures", {}) or {}
            transitions = float(measures.get("locking_window_transitions", 0.0))
            if transitions < self.min_transitions:
                continue
            on_score = float(measures.get("locking_window_score_on", 1.0))
            off_score = float(measures.get("locking_window_score_off", 1.0))
            base_score = float(measures.get("locking_window_score", 1.0))
            transition_count = int(transitions)

            if on_score < self.on_threshold:
                delta = -self.power_lock_step
                message = (
                    f"LSD operator: open power locking in microsector {microsector.index}"
                )
                rationale = (
                    f"LockingWindowScore on-throttle {on_score:.2f} < threshold "
                    f"{self.on_threshold:.2f} with {transition_count} transitions (global score {base_score:.2f}). "
                    f"Reduce power locking following {MANUAL_REFERENCES['differential']}."
                )
                recommendations.append(
                    Recommendation(
                        category="exit",
                        message=message,
                        rationale=rationale,
                        priority=self.priority,
                        parameter="diff_power_lock",
                        delta=delta,
                    )
                )

            if off_score < self.off_threshold:
                delta = -self.preload_step
                message = (
                    f"LSD operator: reduce preload in microsector {microsector.index}"
                )
                rationale = (
                    f"LockingWindowScore off-throttle {off_score:.2f} < threshold "
                    f"{self.off_threshold:.2f} with {transition_count} transitions (global score {base_score:.2f}). "
                    f"Lower differential preload per {MANUAL_REFERENCES['differential']}."
                )
                recommendations.append(
                    Recommendation(
                        category="exit",
                        message=message,
                        rationale=rationale,
                        priority=self.priority + 1,
                        parameter="diff_preload_nm",
                        delta=delta,
                    )
                )

        return recommendations


class UsefulDissonanceRule:
    """Adjust axle balance when the Useful Dissonance Ratio drifts."""

    def __init__(
        self,
        priority: int = 26,
        *,
        high_threshold: float = 0.6,
        low_threshold: float = 0.25,
    ) -> None:
        self.priority = priority
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None or not results:
            return []

        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            udr = float(microsector.filtered_measures.get("udr", 0.0))
            if udr <= 0.0:
                continue
            goal = _goal_for_phase(microsector, microsector.active_phase)
            if goal is None:
                continue
            indices = microsector.phase_indices(goal.phase)
            delta_samples = [
                results[i].delta_nfr for i in indices if 0 <= i < len(results)
            ]
            if not delta_samples:
                continue
            avg_delta = mean(delta_samples)
            tolerance = context.thresholds.tolerance_for_phase(goal.phase)
            deviation = avg_delta - goal.target_delta_nfr

            if udr >= self.high_threshold and deviation > tolerance:
                category = phase_family(goal.phase)
                message = (
                    f"UDR operator: reinforce rear axle/LSD in microsector "
                    f"{microsector.index}"
                )
                rationale = (
                    f"UDR {udr:.2f} shows dΔNFR/dt is negative under high yaw, "
                    f"yet mean ΔNFR {avg_delta:.2f} exceeds the target "
                    f"{goal.target_delta_nfr:.2f} ({deviation:+.2f}). Reinforce the rear bar "
                    f"or increase differential locking "
                    f"({MANUAL_REFERENCES['antiroll']} / {MANUAL_REFERENCES['differential']})."
                )
                recommendations.append(
                    Recommendation(
                        category=category,
                        message=message,
                        rationale=rationale,
                        priority=self.priority,
                    )
                )
                continue

            if udr <= self.low_threshold and abs(deviation) > tolerance:
                if deviation > 0:
                    axle = "front"
                    reference = MANUAL_REFERENCES["antiroll"]
                    action = "soften front bar/damping"
                    category = "entry"
                else:
                    axle = "rear"
                    reference = MANUAL_REFERENCES["differential"]
                    action = "soften rear bar or open LSD"
                    category = "exit"
                message = (
                    f"UDR operator: {action} in microsector {microsector.index}"
                )
                rationale = (
                    f"UDR {udr:.2f} suggests yaw is not reducing ΔNFR. The average "
                    f"{avg_delta:.2f} versus the target {goal.target_delta_nfr:.2f} produces "
                    f"a {deviation:+.2f} deviation; free up the {axle} axle ({reference})."
                )
                recommendations.append(
                    Recommendation(
                        category=category,
                        message=message,
                        rationale=rationale,
                        priority=self.priority + 1,
                    )
                )
        return recommendations


class RecommendationEngine:
    """Aggregate a list of rules and produce recommendations."""

    def __init__(
        self,
        rules: Sequence[RecommendationRule] | None = None,
        *,
        car_model: str | None = None,
        track_name: str | None = None,
        threshold_library: Mapping[str, Mapping[str, ThresholdProfile]] | None = None,
        profile_manager: "ProfileManager" | None = None,
    ) -> None:
        self.car_model = car_model or "generic"
        self.track_name = track_name or "generic"
        self.threshold_library = threshold_library or DEFAULT_THRESHOLD_LIBRARY
        self.profile_manager = profile_manager
        self.session: Mapping[str, Any] | None = None
        if rules is None:
            self.rules = [
                PhaseDeltaDeviationRule(
                    phase="entry",
                    operator_label="Braking operator",
                    category="entry",
                    phase_label="entry",
                    priority=10,
                    reference_key="braking",
                ),
                PhaseNodeOperatorRule(
                    phase="entry",
                    operator_label="Braking operator",
                    category="entry",
                    priority=12,
                    reference_key="braking",
                ),
                BrakeHeadroomRule(priority=14),
                PhaseDeltaDeviationRule(
                    phase="apex",
                    operator_label="Apex operator",
                    category="apex",
                    phase_label="apex",
                    priority=20,
                    reference_key="antiroll",
                ),
                PhaseNodeOperatorRule(
                    phase="apex",
                    operator_label="Apex operator",
                    category="apex",
                    priority=22,
                    reference_key="antiroll",
                ),
                ParallelSteerRule(priority=20),
                TyreBalanceRule(priority=24),
                FootprintEfficiencyRule(priority=16),
                BottomingPriorityRule(priority=18),
                SuspensionVelocityRule(priority=18),
                DetuneRatioRule(priority=24),
                ShiftStabilityRule(priority=28),
                LockingWindowRule(priority=27),
                UsefulDissonanceRule(priority=26),
                CurbComplianceRule(priority=25),
                PhaseDeltaDeviationRule(
                    phase="exit",
                    operator_label="Traction operator",
                    category="exit",
                    phase_label="exit",
                    priority=30,
                    reference_key="differential",
                ),
                PhaseNodeOperatorRule(
                    phase="exit",
                    operator_label="Traction operator",
                    category="exit",
                    priority=32,
                    reference_key="differential",
                ),
                LoadBalanceRule(),
                FrontWingBalanceRule(),
                AeroCoherenceRule(),
                StabilityIndexRule(),
                CoherenceRule(),
            ]
        else:
            self.rules = list(rules)

    def _lookup_profile(self, car_model: str, track_name: str) -> ThresholdProfile:
        return lookup_threshold_profile(car_model, track_name, self.threshold_library)

    def _resolve_context(
        self,
        car_model: str | None,
        track_name: str | None,
    ) -> RuleContext:
        resolved_car = car_model or self.car_model
        resolved_track = track_name or self.track_name
        base_profile = self._lookup_profile(resolved_car, resolved_track)
        objectives = RuleProfileObjectives()
        session_weights: Mapping[str, Mapping[str, float]] = MappingProxyType({})
        session_hints: Mapping[str, Any] = MappingProxyType({})
        if self.profile_manager is not None:
            snapshot = self.profile_manager.resolve(
                resolved_car, resolved_track, base_profile, session=getattr(self, "session", None)
            )
            profile = snapshot.thresholds
            offsets = snapshot.tyre_offsets
            aero_profiles = snapshot.aero_profiles
            session_weights = snapshot.session_weights
            session_hints = snapshot.session_hints
            profile_objectives = getattr(snapshot, "objectives", None)
            if profile_objectives is not None:
                objectives = RuleProfileObjectives(
                    target_delta_nfr=_coerce_float(
                        getattr(profile_objectives, "target_delta_nfr", 0.0), 0.0
                    ),
                    target_sense_index=_coerce_float(
                        getattr(profile_objectives, "target_sense_index", 0.75), 0.75
                    ),
                    target_brake_headroom=_coerce_float(
                        getattr(profile_objectives, "target_brake_headroom", 0.4), 0.4
                    ),
                )
        else:
            profile = base_profile
            offsets = {}
            aero_profiles = {}
            session_payload = getattr(self, "session", None)
            if isinstance(session_payload, Mapping):
                weights_payload = session_payload.get("weights")
                hints_payload = session_payload.get("hints")
                session_weights = _coerce_session_weights(weights_payload)
                session_hints = _coerce_session_hints(hints_payload)
        return RuleContext(
            car_model=resolved_car,
            track_name=resolved_track,
            thresholds=profile,
            tyre_offsets=offsets,
            aero_profiles=MappingProxyType(dict(aero_profiles)),
            objectives=objectives,
            session_weights=session_weights,
            session_hints=session_hints,
        )

    def register_plan(
        self,
        recommendations: Sequence[Recommendation],
        *,
        car_model: str | None = None,
        track_name: str | None = None,
        baseline_sense_index: float | None = None,
        baseline_delta_nfr: float | None = None,
        jacobian: Mapping[str, Mapping[str, float]] | None = None,
        phase_jacobian: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
    ) -> None:
        if self.profile_manager is None:
            return
        phases: Dict[str, float] = {}
        for recommendation in recommendations:
            phase = phase_family(recommendation.category)
            if phase not in {"entry", "apex", "exit"}:
                continue
            phases[phase] = phases.get(phase, 0.0) + 1.0
        if not phases:
            return
        resolved_car = car_model or self.car_model
        resolved_track = track_name or self.track_name
        baseline: tuple[float, float] | None = None
        if baseline_sense_index is not None and baseline_delta_nfr is not None:
            baseline = (float(baseline_sense_index), float(baseline_delta_nfr))
        self.profile_manager.register_plan(
            resolved_car,
            resolved_track,
            phases,
            baseline,
            jacobian=jacobian,
            phase_jacobian=phase_jacobian,
        )

    def register_stint_result(
        self,
        *,
        sense_index: float,
        delta_nfr: float,
        car_model: str | None = None,
        track_name: str | None = None,
    ) -> None:
        if self.profile_manager is None:
            return
        resolved_car = car_model or self.car_model
        resolved_track = track_name or self.track_name
        self.profile_manager.register_result(resolved_car, resolved_track, sense_index, delta_nfr)

    def generate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        *,
        car_model: str | None = None,
        track_name: str | None = None,
    ) -> List[Recommendation]:
        context = self._resolve_context(car_model, track_name)
        if microsectors:
            quiet_sequences = detect_quiet_microsector_streaks(microsectors)
            if quiet_sequences:
                message, rationale = _quiet_recommendation_notice(
                    microsectors, quiet_sequences
                )
                return [
                    Recommendation(
                        category="driver",
                        message=message,
                        rationale=rationale,
                        priority=-100,
                    )
                ]
        recommendations: List[Recommendation] = []
        for rule in self.rules:
            recommendations.extend(list(rule.evaluate(results, microsectors, context)))
        enumerated = list(enumerate(recommendations))
        enumerated.sort(key=lambda item: (item[1].priority, item[0]))
        unique: List[Recommendation] = []
        seen = set()
        for _, recommendation in enumerated:
            key = (recommendation.category, recommendation.message)
            if key in seen:
                continue
            seen.add(key)
            unique.append(recommendation)
        return unique
