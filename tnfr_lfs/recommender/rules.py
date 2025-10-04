"""Rule-based recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
import math
from statistics import mean
from types import MappingProxyType
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from ..core.epi_models import EPIBundle
from ..core.operators import TyreBalanceControlOutput, tyre_balance_controller
from ..core.phases import LEGACY_PHASE_MAP, expand_phase_alias, phase_family
from ..core.archetypes import (
    ARCHETYPE_MEDIUM,
    DEFAULT_ARCHETYPE_PHASE_TARGETS,
    PhaseArchetypeTargets,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..io.profiles import AeroProfile, ProfileManager
from ..core.segmentation import (
    Goal,
    Microsector,
    detect_quiet_microsector_streaks,
    microsector_stability_metrics,
)


MANUAL_REFERENCES = {
    "braking": "Basic Setup Guide · Frenada óptima [BAS-FRE]",
    "antiroll": "Advanced Setup Guide · Barras estabilizadoras [ADV-ARB]",
    "differential": "Advanced Setup Guide · Configuración de diferenciales [ADV-DIF]",
    "curbs": "Basic Setup Guide · Uso de pianos [BAS-CUR]",
    "ride_height": "Advanced Setup Guide · Alturas y reparto de carga [ADV-RDH]",
    "aero": "Basic Setup Guide · Balance aerodinámico [BAS-AER]",
    "driver": "Basic Setup Guide · Constancia de pilotaje [BAS-DRV]",
    "tyre_balance": "Advanced Setup Guide · Presiones y caídas [ADV-TYR]",
}


_ALIGNMENT_ALIGNMENT_GAP = 0.15
_ALIGNMENT_LAG_GAP = 0.3
_ALIGNMENT_THRESHOLD = 0.05
_LAG_THRESHOLD = 0.05
_COHERENCE_THRESHOLD = 0.05


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
        "phase_lag": "suspension",
        "coherence_index": "tyres",
    },
    "apex": {
        "phase_alignment": "suspension",
        "phase_lag": "suspension",
        "coherence_index": "tyres",
    },
    "exit": {
        "phase_alignment": "suspension",
        "phase_lag": "tyres",
        "coherence_index": "tyres",
    },
}


_AXIS_FOCUS_MAP: Mapping[tuple[str, str], tuple[str, str, Tuple[str, ...]]] = {
    ("entry", "longitudinal"): (
        "Prioriza el bias de frenos (ΔNFR∥)",
        "braking",
        ("brake_bias_pct",),
    ),
    ("exit", "longitudinal"): (
        "Refuerza el bloqueo de retención (ΔNFR∥)",
        "differential",
        ("diff_coast_lock",),
    ),
    ("apex", "lateral"): (
        "Ajusta las barras estabilizadoras (ΔNFR⊥)",
        "antiroll",
        ("front_arb_steps", "rear_arb_steps"),
    ),
    ("entry", "lateral"): (
        "Afina el toe delantero (ΔNFR⊥)",
        "tyre_balance",
        ("front_toe_deg",),
    ),
    ("exit", "lateral"): (
        "Afina el toe trasero (ΔNFR⊥)",
        "tyre_balance",
        ("rear_toe_deg",),
    ),
}


NODE_LABELS = {
    "tyres": "neumáticos",
    "suspension": "suspensión",
    "chassis": "chasis",
    "brakes": "frenos",
    "transmission": "transmisión",
    "track": "pista",
    "driver": "piloto",
}


_BASE_PHASE_ACTIONS: Mapping[str, Mapping[str, Mapping[str, str]]] = {
    "entry": {
        "tyres": {
            "increase": "abrir toe delantero",
            "decrease": "cerrar toe delantero",
        },
        "brakes": {
            "increase": "desplazar el balance de frenos hacia delante",
            "decrease": "desplazar el balance de frenos hacia atrás",
        },
        "suspension": {
            "increase": "endurecer el rebote delantero",
            "decrease": "ablandar el rebote delantero",
        },
    },
    "apex": {
        "suspension": {
            "increase": "endurecer la barra estabilizadora",
            "decrease": "ablandar la barra estabilizadora",
        },
        "chassis": {
            "increase": "aumentar precarga delantera",
            "decrease": "reducir precarga delantera",
        },
        "tyres": {
            "increase": "subir presión en neumáticos exteriores",
            "decrease": "bajar presión en neumáticos exteriores",
        },
    },
    "exit": {
        "transmission": {
            "increase": "cerrar el diferencial de aceleración",
            "decrease": "abrir el diferencial de aceleración",
        },
        "tyres": {
            "increase": "incrementar caída trasera",
            "decrease": "reducir caída trasera",
        },
        "suspension": {
            "increase": "endurecer compresión trasera",
            "decrease": "ablandar compresión trasera",
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


_PHASE_ACTION_ROADMAP: Mapping[str, Tuple[PhaseActionTemplate, ...]] = {
    "entry": (
        PhaseActionTemplate(
            metric="delta_nfr",
            parameter="brake_bias_pct",
            scale=-1.0,
            min_value=-4.0,
            max_value=4.0,
            message_pattern="{delta:+.1f}% bias delante",
            step=0.5,
        ),
        PhaseActionTemplate(
            metric="delta_nfr_lateral",
            parameter="front_spring_stiffness",
            scale=-900.0,
            min_value=-35.0,
            max_value=35.0,
            message_pattern="{delta:+.1f} N/mm muelle delantero (νf_susp · ΔNFR⊥)",
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
            message_pattern="{delta:+.0f} clicks rebote delante",
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
            message_pattern="{delta:+.0f} clicks compresión delante",
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
            message_pattern="{delta:+.1f} psi eje delantero",
            step=0.1,
            priority_offset=3,
        ),
        PhaseActionTemplate(
            metric="phase_alignment",
            parameter="front_camber_deg",
            scale=-1.1,
            min_value=-1.5,
            max_value=1.5,
            message_pattern="{delta:+.1f}° camber delantero",
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
            metric="coherence_index",
            parameter="front_toe_deg",
            scale=-0.45,
            min_value=-0.5,
            max_value=0.5,
            message_pattern="{delta:+.2f}° toe delantero",
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
            message_pattern="{delta:+.0f} pasos barra delantera",
            step=1.0,
        ),
        PhaseActionTemplate(
            metric="delta_nfr_lateral",
            parameter="front_spring_stiffness",
            scale=-900.0,
            min_value=-35.0,
            max_value=35.0,
            message_pattern="{delta:+.1f} N/mm muelle delantero (νf_susp · ΔNFR⊥)",
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
            message_pattern="{delta:+.0f} pasos barra trasera",
            step=1.0,
            priority_offset=1,
        ),
        PhaseActionTemplate(
            metric="sense_index",
            parameter="front_tyre_pressure",
            scale=18.0,
            min_value=-0.6,
            max_value=0.6,
            message_pattern="{delta:+.1f} psi exteriores",
            step=0.1,
            priority_offset=2,
        ),
        PhaseActionTemplate(
            metric="nu_f",
            parameter="rear_rebound_clicks",
            scale=-0.5,
            min_value=-6.0,
            max_value=6.0,
            message_pattern="{delta:+.0f} clicks rebote trasero",
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
            message_pattern="{delta:+.1f}° camber delantero",
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
            message_pattern="{delta:+.1f}° camber trasero",
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
            message_pattern="{delta:+.2f}° toe delantero",
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
            message_pattern="{delta:+.2f}° toe trasero",
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
            message_pattern="{delta:+.0f}% LSD potencia",
            step=5.0,
        ),
        PhaseActionTemplate(
            metric="delta_nfr_lateral",
            parameter="rear_spring_stiffness",
            scale=-900.0,
            min_value=-35.0,
            max_value=35.0,
            message_pattern="{delta:+.1f} N/mm muelle trasero (νf_susp · ΔNFR⊥)",
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
            message_pattern="{delta:+.1f} mm altura trasera",
            step=0.5,
            priority_offset=1,
        ),
        PhaseActionTemplate(
            metric="sense_index",
            parameter="rear_tyre_pressure",
            scale=20.0,
            min_value=-0.6,
            max_value=0.6,
            message_pattern="{delta:+.1f} psi eje trasero",
            step=0.1,
            priority_offset=2,
        ),
        PhaseActionTemplate(
            metric="nu_f",
            parameter="diff_coast_lock",
            scale=-2.0,
            min_value=-20.0,
            max_value=20.0,
            message_pattern="{delta:+.0f}% LSD retención",
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
            message_pattern="{delta:+.1f}° ala trasera",
            step=0.5,
            priority_offset=4,
        ),
        PhaseActionTemplate(
            metric="phase_alignment",
            parameter="rear_camber_deg",
            scale=-1.0,
            min_value=-1.5,
            max_value=1.5,
            message_pattern="{delta:+.1f}° camber trasero",
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
            message_pattern="{delta:+.2f}° toe trasero",
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
            message_pattern="{delta:+.2f}° toe trasero",
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


@dataclass(frozen=True)
class RuleProfileObjectives:
    """Minimal snapshot of profile objectives for rule evaluation."""

    target_delta_nfr: float = 0.0
    target_sense_index: float = 0.75


@dataclass(frozen=True)
class RuleContext:
    """Context shared with the rules to build rationales."""

    car_model: str
    track_name: str
    thresholds: ThresholdProfile
    tyre_offsets: Mapping[str, float] = field(default_factory=dict)
    aero_profiles: Mapping[str, "AeroProfile"] = field(default_factory=dict)
    objectives: RuleProfileObjectives = field(default_factory=RuleProfileObjectives)

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
    return ThresholdProfile(
        entry_delta_tolerance=float(payload.get("entry_delta_tolerance", defaults["entry_delta_tolerance"])),
        apex_delta_tolerance=float(payload.get("apex_delta_tolerance", defaults["apex_delta_tolerance"])),
        exit_delta_tolerance=float(payload.get("exit_delta_tolerance", defaults["exit_delta_tolerance"])),
        piano_delta_tolerance=float(payload.get("piano_delta_tolerance", defaults["piano_delta_tolerance"])),
        rho_detune_threshold=float(payload.get("rho_detune_threshold", defaults["rho_detune_threshold"])),
        phase_targets=phase_targets,
        phase_weights=phase_weights,
        archetype_phase_targets=_BASELINE_ARCHETYPE_TARGETS,
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
                        f"{direction.title()} Rear Ride height to rebalance load "
                        f"({MANUAL_REFERENCES['ride_height']})"
                    ),
                    rationale=(
                        "ΔNFR deviated by "
                        f"{result.delta_nfr:.1f} units relative to baseline at t={result.timestamp:.2f}."
                        f" Consulta {MANUAL_REFERENCES['ride_height']} para reajustar alturas."
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
    ) -> None:
        self.priority = int(priority)
        self.ratio_threshold = float(ratio_threshold)
        self.smooth_surface_cutoff = float(smooth_surface_cutoff)
        self.ride_height_delta = float(ride_height_delta)
        self.bump_delta = float(bump_delta)

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
                ("delantero", front_ratio, "front_ride_height", "front_compression_clicks"),
                ("trasero", rear_ratio, "rear_ride_height", "rear_compression_clicks"),
            ):
                if ratio < self.ratio_threshold:
                    continue
                if is_rough:
                    parameter = bump_param
                    delta = self.bump_delta
                    focus = "endurecer compresión"
                    reference = MANUAL_REFERENCES["antiroll"]
                else:
                    parameter = ride_param
                    delta = self.ride_height_delta
                    focus = "elevar altura"
                    reference = MANUAL_REFERENCES["ride_height"]
                surface_label = "bacheada" if is_rough else "lisa"
                message = (
                    f"Operador bottoming: {focus} {axle} en microsector {microsector.index}"
                )
                rationale = (
                    f"Índice de bottoming {ratio:.2f} en el eje {axle} coincide con picos de ΔNFR∥ "
                    f"en microsector {microsector.index}. Superficie {surface_label}"
                    f" (factor {surface_factor:.2f}) → prioriza {focus}. {reference}."
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
                        f"{self.threshold:.2f}. Consulta {MANUAL_REFERENCES['aero']} para reequilibrar la carga."
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

            if high_deviation > 0:
                delta = self.delta_step
                action = "Incrementa el ángulo del alerón trasero"
                direction = "trasera"
            else:
                delta = -self.delta_step
                action = "Reduce el ángulo del alerón trasero/refuerza la carga delantera"
                direction = "delantera"

            recommendations.append(
                Recommendation(
                    category="aero",
                    message=f"Alta velocidad microsector {microsector.index}: {action}",
                    rationale=(
                        f"ΔNFR aero alta velocidad {high_imbalance:+.2f} frente al objetivo {target_high:+.2f} "
                        f"con baja velocidad estable ({low_imbalance:+.2f}) y C(a/m) {am_coherence:.2f}. "
                        f"Refuerza carga {direction} "
                        f"({MANUAL_REFERENCES['aero']})."
                    ),
                    priority=self.priority,
                    parameter="rear_wing_angle",
                    delta=delta,
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
                        f"Apóyate en {MANUAL_REFERENCES['driver']} para reforzar hábitos consistentes."
                    ),
                    priority=120,
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


def _apply_action_template(template: PhaseActionTemplate, raw_value: float) -> float | None:
    value = raw_value * template.scale
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
) -> List[Recommendation]:
    recommendations: List[Recommendation] = []
    for template in _phase_action_templates(phase, metric, node):
        value = _apply_action_template(template, raw_value)
        if value is None:
            continue
        message = template.message_pattern.format(delta=value)
        rationale = (
            f"{base_rationale} Acción sugerida: {message}. Consulta "
            f"{MANUAL_REFERENCES[reference_key]} para aplicar el ajuste."
        )
        if template.parameter in _SPRING_PARAMETERS:
            rationale = (
                f"{base_rationale} Acción sugerida: {message}. νf_susp pondera el ajuste "
                f"para homogeneizar ΔNFR⊥ en la banda de G activa. Consulta "
                f"{MANUAL_REFERENCES[reference_key]} para aplicar el ajuste."
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
    coherence_delta = target_coherence - average_coherence
    return {
        "measured_alignment": measured_alignment,
        "target_alignment": target_alignment,
        "alignment_delta": alignment_delta,
        "alignment_gap": abs(alignment_delta),
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
    lag_gap = float(snapshot.get("lag_gap", 0.0))
    coherence_gap = float(snapshot.get("coherence_gap", 0.0))
    if alignment_gap > _ALIGNMENT_THRESHOLD:
        urgency = max(urgency, 3 if alignment_gap > _ALIGNMENT_THRESHOLD * 2 else 2)
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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


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
    message = f"No tocar: {', '.join(descriptors)}"
    if not coverage_values:
        rationale = (
            "Detectamos una secuencia estable sin activación dinámica reseñable."
        )
        return message, rationale
    coverage_avg = sum(coverage_values) / len(coverage_values)
    slack_avg = sum(slack_values) / len(slack_values) if slack_values else 0.0
    si_avg = sum(si_values) / len(si_values) if si_values else 0.0
    epi_avg = sum(epi_values) / len(epi_values) if epi_values else 0.0
    rationale = (
        "Detección de silencio estructural prolongado:"
        f" silencio μ {coverage_avg * 100.0:.0f}%"
        f", slack μ {slack_avg:.2f}, Siσ μ {si_avg:.4f}, |dEPI| μ {epi_avg:.3f}."
    )
    return message, rationale


def _brake_event_summary(
    microsector: Microsector,
) -> tuple[str | None, str | None, float]:
    operator_events = getattr(microsector, "operator_events", {}) or {}
    silence_payloads = operator_events.get("SILENCIO", ())
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
            elif isinstance(payload.get("surface"), Mapping):
                label_value = payload["surface"].get("label")  # type: ignore[index]
                if isinstance(label_value, str):
                    surface_label = label_value
            relevant.append(
                {
                    "type": event_type,
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
        label = worst["surface"] or "superficie"
        threshold = _safe_float(worst["threshold"])
        peak = _safe_float(worst["peak"])
        summary_parts.append(
            f"{event_type}×{count} ({label}) ΔNFR {peak:.2f}>{threshold:.2f}"
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
        recommendations: List[Recommendation] = []
        for microsector in microsectors:
            goal = _goal_for_phase(microsector, self.phase)
            if goal is None:
                continue
            indices = list(microsector.phase_indices(goal.phase))
            samples = _phase_samples(results, indices)
            if not samples:
                continue
            actual_delta = mean(bundle.delta_nfr for bundle in samples)
            deviation = actual_delta - goal.target_delta_nfr
            avg_long = mean(bundle.delta_nfr_longitudinal for bundle in samples)
            avg_lat = mean(bundle.delta_nfr_lateral for bundle in samples)
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
                f"{self.operator_label} aplicado sobre la fase de {self.phase_label} en "
                f"microsector {microsector.index}. El objetivo ΔNFR era "
                f"{goal.target_delta_nfr:.2f}, pero la media registrada fue "
                f"{actual_delta:.2f} ({deviation:+.2f})."
            )
            if event_summary:
                phase_summary = f"{phase_summary} {event_summary}."

            geometry_actions_map: Dict[str, Recommendation] = {}
            geometry_urgency = _geometry_urgency(geometry_snapshot)

            if geometry_snapshot["alignment_gap"] > _ALIGNMENT_THRESHOLD:
                alignment_rationale = (
                    f"{phase_summary} Siφ objetivo {geometry_snapshot['target_alignment']:+.2f} "
                    f"frente al medido {geometry_snapshot['measured_alignment']:+.2f} "
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

            if geometry_snapshot["lag_gap"] > _LAG_THRESHOLD:
                lag_rationale = (
                    f"{phase_summary} θ medido {geometry_snapshot['measured_lag']:+.2f}rad "
                    f"frente al objetivo {geometry_snapshot['target_lag']:+.2f}rad "
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
                    f"{phase_summary} C(t) medio {geometry_snapshot['average_coherence']:.2f} "
                    f"(objetivo {geometry_snapshot['target_coherence']:.2f}, Δ "
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
                    f"{phase_summary} La tolerancia definida para {context.profile_label} "
                    f"es ±{tolerance:.2f}."
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
                                f"{base_rationale} νf_susp objetivo {target_nu_f:.2f} frente al medido "
                                f"{actual_nu_f:.2f} ({nu_f_delta:+.2f}). ΔNFR⊥ medio {avg_lat:.2f} "
                                f"(objetivo {target_lat:.2f}, Δ {lat_dev:+.2f}). El ajuste del muelle busca "
                                "homogeneizar ΔNFR⊥ en la banda de G activa."
                            )
                            spring_recommendations = _phase_action_recommendations(
                                phase=self.phase,
                                category=self.category,
                                metric="delta_nfr_lateral",
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
                            f"{recommendation.delta:+.1f}% bias delante"
                        )
                        recommendation.rationale = (
                            f"{base_rationale} Acción sugerida: {recommendation.message}. "
                            f"Consulta {MANUAL_REFERENCES[self.reference_key]} para aplicar el ajuste."
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
                direction = "incrementar" if deviation < 0 else "reducir"
                summary_message = (
                    f"{self.operator_label} · objetivo ΔNFR global: {direction} ΔNFR "
                    f"en microsector {microsector.index} ({MANUAL_REFERENCES[self.reference_key]})"
                )
                summary_rationale = (
                    f"{base_rationale} Se recomienda {direction} ΔNFR global y revisar "
                    f"{MANUAL_REFERENCES[self.reference_key]} para la fase de {self.phase_label}."
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
                            f" C(t) medio {geometry_snapshot['average_coherence']:.2f} "
                            f"(objetivo {geometry_snapshot['target_coherence']:.2f})."
                        )
                    focus_rationale = (
                        f"{base_rationale} ΔNFR{axis_label} domina ({axis_delta:+.2f} frente al objetivo "
                        f"{target_value:+.2f}). θ medido {lag_value:+.2f}rad (objetivo {target_lag:+.2f}). "
                        f"Reparto objetivo ∥ {weight_long:.2f} · ⊥ {weight_lat:.2f}.{coherence_line}"
                    )
                    recommendations.append(
                        Recommendation(
                            category=self.category,
                            message=focus_message,
                            rationale=f"{focus_rationale} Consulta {MANUAL_REFERENCES[focus_reference]}.",
                            priority=min(self.priority - 2, self.priority - geometry_urgency),
                        )
                    )
                    for rec in recommendations:
                        if rec.parameter and rec.parameter in parameters:
                            rec.priority = min(rec.priority, self.priority - 1)

            target_si = getattr(goal, "target_sense_index", None)
            if target_si is not None:
                actual_si = mean(bundle.sense_index for bundle in samples)
                si_delta = target_si - actual_si
                si_tolerance = 0.01
                if abs(si_delta) > si_tolerance:
                    si_rationale = (
                        f"El objetivo de sense index era {target_si:.2f} y se observó "
                        f"{actual_si:.2f} ({si_delta:+.2f})."
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

        recommendations: List[Recommendation] = []
        for microsector in microsectors:
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
                f"{self.operator_label} aplicado en microsector {microsector.index} "
                f"con nodos dominantes {dominant_list or 'de referencia'}."
            )
            if (
                geometry_snapshot["alignment_gap"] > _ALIGNMENT_THRESHOLD
                and geometry_nodes.get("phase_alignment")
            ):
                alignment_rationale = (
                    f"{geometry_context} Siφ objetivo {geometry_snapshot['target_alignment']:+.2f} "
                    f"frente al medido {geometry_snapshot['measured_alignment']:+.2f} "
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
                    f"{geometry_context} θ medido {geometry_snapshot['measured_lag']:+.2f}rad "
                    f"frente al objetivo {geometry_snapshot['target_lag']:+.2f}rad "
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
                    f"{geometry_context} C(t) medio {geometry_snapshot['average_coherence']:.2f} "
                    f"(objetivo {geometry_snapshot['target_coherence']:.2f}, Δ "
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
                    f"(objetivo θ {target_lag:+.2f}rad / Siφ {target_alignment:+.2f})."
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
                    classification_summary = f" Estado ν_f: {display_label}."
                elif classification:
                    classification_summary = f" Clasificación ν_f {classification}."
                if coherence_index > 0.0:
                    classification_summary += f" C(t) {coherence_index:.2f}."
                sense_summary = (
                    f" Objetivo Si perfil {target_si:.2f} para {context.profile_label}."
                )
                coherence_line = (
                    f" C(t) medio {geometry_snapshot['average_coherence']:.2f} "
                    f"(objetivo {geometry_snapshot['target_coherence']:.2f})."
                )
                base_rationale = (
                    f"{self.operator_label} aplicado al nodo {node_label} en microsector "
                    f"{microsector.index}. La estrategia del objetivo destaca a "
                    f"{dominant_list or 'los nodos dominantes'} y fija ν_f={target_nu_f:.2f}. "
                    f"Se midió ν_f medio {actual_nu_f:.2f} ({deviation:+.2f}), superando la "
                    f"tolerancia ajustada ±{tolerance:.2f} definida para {context.profile_label}. "
                    f"{alignment_summary}{classification_summary}{sense_summary}{coherence_line}"
                )
                if flip_alignment:
                    base_rationale += " Se invierte el sentido del ajuste para recuperar la alineación de fase."
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
                            f"{self.operator_label} · nodo objetivo {node_label}: {action_text} "
                            f"para acercar ν_f a {target_nu_f:.2f} "
                            f"({MANUAL_REFERENCES[self.reference_key]})"
                        ),
                        rationale=(
                            f"{base_rationale} Acción sugerida: {action_text}. Consulta "
                            f"{MANUAL_REFERENCES[self.reference_key]} para los ajustes."
                        ),
                        priority=self.priority,
                    )
                )
        return recommendations


class ParallelSteerRule:
    """React to Ackermann steering deviations using the aggregated index."""

    def __init__(self, priority: int = 20, threshold: float = 0.08, delta_step: float = 0.1) -> None:
        self.priority = int(priority)
        self.threshold = float(threshold)
        self.delta_step = float(delta_step)

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
                deviation = float(measures.get("ackermann_parallel_index", 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(deviation):
                continue
            magnitude = abs(deviation)
            if magnitude <= self.threshold:
                continue
            if deviation < 0.0:
                action = "abrir toe delantero"
                delta = self.delta_step
            else:
                action = "cerrar toe delantero"
                delta = -self.delta_step
            message = (
                f"Operador Ackermann: {action} (parallel steer) en microsector {microsector.index}"
            )
            rationale = (
                f"Desfase Ackermann medio {deviation:+.3f}rad supera el umbral {self.threshold:.3f}. "
                f"Ajusta toe estático para recuperar el par teórico ({MANUAL_REFERENCES['tyre_balance']})."
            )
            recommendations.append(
                Recommendation(
                    category="entry",
                    message=message,
                    rationale=rationale,
                    priority=self.priority,
                    parameter="front_toe_deg",
                    delta=delta,
                )
            )
        return recommendations


class TyreBalanceRule:
    """Recommend ΔP and camber tweaks from tyre thermal trends."""

    def __init__(self, priority: int = 18, target_front: float = 82.0, target_rear: float = 80.0) -> None:
        self.priority = priority
        self.target_front = target_front
        self.target_rear = target_rear

    def evaluate(
        self,
        results: Sequence[EPIBundle],
        microsectors: Sequence[Microsector] | None = None,
        context: RuleContext | None = None,
    ) -> Iterable[Recommendation]:
        if not microsectors or context is None:
            return []

        def _safe_float(value: object) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        def _average(values: Sequence[float]) -> float:
            return mean(values) if values else 0.0

        controls: List[TyreBalanceControlOutput] = []
        front_temps: List[float] = []
        rear_temps: List[float] = []
        delta_flat_values: List[float] = []
        per_wheel: Dict[str, List[float]] = {key: [] for key in ("fl", "fr", "rl", "rr")}

        for microsector in microsectors:
            metrics = microsector.filtered_measures
            control = tyre_balance_controller(
                metrics,
                target_front=self.target_front,
                target_rear=self.target_rear,
                offsets=context.tyre_offsets,
            )
            controls.append(control)
            front_temps.append(
                _average(
                    [
                        _safe_float(metrics.get("tyre_temp_fl")),
                        _safe_float(metrics.get("tyre_temp_fr")),
                    ]
                )
            )
            rear_temps.append(
                _average(
                    [
                        _safe_float(metrics.get("tyre_temp_rl")),
                        _safe_float(metrics.get("tyre_temp_rr")),
                    ]
                )
            )
            delta_flat_values.append(_safe_float(metrics.get("d_nfr_flat")))
            for key, value in control.per_wheel_pressure.items():
                per_wheel.setdefault(key, []).append(float(value))

        if not controls:
            return []

        front_pressure = _average([control.pressure_delta_front for control in controls])
        rear_pressure = _average([control.pressure_delta_rear for control in controls])
        camber_front = _average([control.camber_delta_front for control in controls])
        camber_rear = _average([control.camber_delta_rear for control in controls])
        per_wheel_avg = {key: _average(values) for key, values in per_wheel.items()}

        average_front_temp = _average(front_temps)
        average_rear_temp = _average(rear_temps)
        avg_delta_flat = _average(delta_flat_values)

        recommendations: List[Recommendation] = []

        if abs(front_pressure) > 0.01 or abs(rear_pressure) > 0.01:
            pressure_rationale = (
                f"Temperaturas medias {average_front_temp:.1f}°C / {average_rear_temp:.1f}°C con ΔNFR_flat {avg_delta_flat:+.2f}. "
                f"Per-wheel ΔP {per_wheel_avg['fl']:+.2f}/{per_wheel_avg['fr']:+.2f}/"
                f"{per_wheel_avg['rl']:+.2f}/{per_wheel_avg['rr']:+.2f}."
            )
            recommendations.append(
                Recommendation(
                    category="tyres",
                    message=(
                        "Operador térmico: ajustar presiones ΔPfront "
                        f"{front_pressure:+.2f} / ΔPrear {rear_pressure:+.2f} bar "
                        f"({MANUAL_REFERENCES['tyre_balance']})"
                    ),
                    rationale=(
                        f"{pressure_rationale} Sigue las pautas de {MANUAL_REFERENCES['tyre_balance']} "
                        "para implementar los ajustes." 
                    ),
                    priority=self.priority,
                    parameter="tyre_pressure",
                    delta=front_pressure,
                )
            )

        if abs(camber_front) > 0.01 or abs(camber_rear) > 0.01:
            camber_rationale = (
                f"Gradiente térmico front {camber_front:+.2f}·dt y rear {camber_rear:+.2f}·dt derivados de dT/dt."
            )
            recommendations.append(
                Recommendation(
                    category="tyres",
                    message=(
                        "Operador térmico: camber Δfront "
                        f"{camber_front:+.2f}° / Δrear {camber_rear:+.2f}° "
                        f"({MANUAL_REFERENCES['tyre_balance']})"
                    ),
                    rationale=(
                        f"{camber_rationale} Ajusta las caídas siguiendo {MANUAL_REFERENCES['tyre_balance']}."
                    ),
                    priority=self.priority + 1,
                    parameter="camber",
                    delta=camber_front,
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
            action = "liberar" if deviation > 0 else "cargar"
            recommendations.append(
                Recommendation(
                    category="pianos",
                    message=(
                        f"Operador de pianos: {action} apoyo en microsector {microsector.index}"
                        f" ({MANUAL_REFERENCES['curbs']})"
                    ),
                    rationale=(
                        "Los pianos registraron un ΔNFR medio de "
                        f"{actual_delta:.2f} frente al objetivo {goal.target_delta_nfr:.2f}. "
                        f"El desvío {deviation:+.2f} supera la tolerancia ±{tolerance:.2f} "
                        f"configurada para {context.profile_label}. "
                        f"Sigue las pautas de {MANUAL_REFERENCES['curbs']} para modular el apoyo."
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
                target_text += f" Objetivo ρ≈{target_rho:.2f}."
            if target_exc:
                target_text += f" ν_exc ref {target_exc:.2f}Hz."
            focus_text = ""
            if isinstance(detune_weights, Mapping):
                long_weight = float(detune_weights.get("longitudinal", 0.5))
                lat_weight = float(detune_weights.get("lateral", 0.5))
                if long_weight > lat_weight:
                    focus_text = " Prioridad detune longitudinal."
                elif lat_weight > long_weight:
                    focus_text = " Prioridad detune lateral."
            rationale = (
                f"Detune ratio ρ={rho:.2f} en microsector {microsector.index}"
                f" cae por debajo del umbral {threshold:.2f} con ∇Res {d_nfr_res:+.2f}."
                f"{target_text}{focus_text} Revisa barras estabilizadoras y amortiguadores"
                f" ({MANUAL_REFERENCES['antiroll']})."
            )
            message = (
                f"Operador modal · microsector {microsector.index}:"
                " ajustar barras/amortiguadores para elevar ρ"
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
                    f"Operador UDR: reforzar eje trasero/LSD en microsector "
                    f"{microsector.index}"
                )
                rationale = (
                    f"UDR {udr:.2f} indica que dΔNFR/dt es negativo bajo alta guiñada, "
                    f"pero el ΔNFR medio {avg_delta:.2f} supera el objetivo "
                    f"{goal.target_delta_nfr:.2f} ({deviation:+.2f}). Refuerza la barra "
                    f"trasera o incrementa el bloqueo del diferencial "
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
                    axle = "delantero"
                    reference = MANUAL_REFERENCES["antiroll"]
                    action = "ablandar barra/amortiguación delantera"
                    category = "entry"
                else:
                    axle = "trasero"
                    reference = MANUAL_REFERENCES["differential"]
                    action = "ablandar barra trasera o abrir LSD"
                    category = "exit"
                message = (
                    f"Operador UDR: {action} en microsector {microsector.index}"
                )
                rationale = (
                    f"UDR {udr:.2f} sugiere que la guiñada no reduce ΔNFR. El promedio "
                    f"{avg_delta:.2f} frente al objetivo {goal.target_delta_nfr:.2f} genera "
                    f"un desvío {deviation:+.2f}; libera el eje {axle} ({reference})."
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
        if rules is None:
            self.rules = [
                PhaseDeltaDeviationRule(
                    phase="entry",
                    operator_label="Operador de frenado",
                    category="entry",
                    phase_label="entrada",
                    priority=10,
                    reference_key="braking",
                ),
                PhaseNodeOperatorRule(
                    phase="entry",
                    operator_label="Operador de frenado",
                    category="entry",
                    priority=12,
                    reference_key="braking",
                ),
                PhaseDeltaDeviationRule(
                    phase="apex",
                    operator_label="Operador de vértice",
                    category="apex",
                    phase_label="vértice",
                    priority=20,
                    reference_key="antiroll",
                ),
                PhaseNodeOperatorRule(
                    phase="apex",
                    operator_label="Operador de vértice",
                    category="apex",
                    priority=22,
                    reference_key="antiroll",
                ),
                ParallelSteerRule(priority=20),
                TyreBalanceRule(priority=24),
                BottomingPriorityRule(priority=18),
                DetuneRatioRule(priority=24),
                UsefulDissonanceRule(priority=26),
                CurbComplianceRule(priority=25),
                PhaseDeltaDeviationRule(
                    phase="exit",
                    operator_label="Operador de tracción",
                    category="exit",
                    phase_label="salida",
                    priority=30,
                    reference_key="differential",
                ),
                PhaseNodeOperatorRule(
                    phase="exit",
                    operator_label="Operador de tracción",
                    category="exit",
                    priority=32,
                    reference_key="differential",
                ),
                LoadBalanceRule(),
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
        if self.profile_manager is not None:
            snapshot = self.profile_manager.resolve(resolved_car, resolved_track, base_profile)
            profile = snapshot.thresholds
            offsets = snapshot.tyre_offsets
            aero_profiles = snapshot.aero_profiles
            profile_objectives = getattr(snapshot, "objectives", None)
            if profile_objectives is not None:
                objectives = RuleProfileObjectives(
                    target_delta_nfr=_coerce_float(
                        getattr(profile_objectives, "target_delta_nfr", 0.0), 0.0
                    ),
                    target_sense_index=_coerce_float(
                        getattr(profile_objectives, "target_sense_index", 0.75), 0.75
                    ),
                )
        else:
            profile = base_profile
            offsets = {}
            aero_profiles = {}
        return RuleContext(
            car_model=resolved_car,
            track_name=resolved_track,
            thresholds=profile,
            tyre_offsets=offsets,
            aero_profiles=MappingProxyType(dict(aero_profiles)),
            objectives=objectives,
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
