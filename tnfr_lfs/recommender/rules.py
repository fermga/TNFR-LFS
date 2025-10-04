"""Rule-based recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
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
from ..core.phases import LEGACY_PHASE_MAP, expand_phase_alias, phase_family

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..io.profiles import ProfileManager
from ..core.segmentation import Goal, Microsector


MANUAL_REFERENCES = {
    "braking": "Basic Setup Guide · Frenada óptima [BAS-FRE]",
    "antiroll": "Advanced Setup Guide · Barras estabilizadoras [ADV-ARB]",
    "differential": "Advanced Setup Guide · Configuración de diferenciales [ADV-DIF]",
    "curbs": "Basic Setup Guide · Uso de pianos [BAS-CUR]",
    "ride_height": "Advanced Setup Guide · Alturas y reparto de carga [ADV-RDH]",
    "aero": "Basic Setup Guide · Balance aerodinámico [BAS-AER]",
    "driver": "Basic Setup Guide · Constancia de pilotaje [BAS-DRV]",
}


_ALIGNMENT_ALIGNMENT_GAP = 0.15
_ALIGNMENT_LAG_GAP = 0.3


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
    ),
}

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


@dataclass(frozen=True)
class RuleContext:
    """Context shared with the rules to build rationales."""

    car_model: str
    track_name: str
    thresholds: ThresholdProfile

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
            if abs(deviation) <= tolerance:
                continue
            base_rationale = (
                f"{self.operator_label} aplicado sobre la fase de {self.phase_label} en "
                f"microsector {microsector.index}. El objetivo ΔNFR era "
                f"{goal.target_delta_nfr:.2f}, pero la media registrada fue "
                f"{actual_delta:.2f} ({deviation:+.2f}). La tolerancia definida para "
                f"{context.profile_label} es ±{tolerance:.2f}."
            )
            recommendations.extend(
                _phase_action_recommendations(
                    phase=self.phase,
                    category=self.category,
                    metric="delta_nfr",
                    raw_value=deviation,
                    base_rationale=base_rationale,
                    priority=self.priority,
                    reference_key=self.reference_key,
                )
            )

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
            dominant_nodes = goal.dominant_nodes or microsector.dominant_nodes.get(
                goal.phase, ()
            )
            if not dominant_nodes:
                continue
            target_nu_f = float(goal.nu_f_target)
            tolerance = max(0.05, abs(target_nu_f) * 0.2)
            measured_lag, measured_alignment, target_lag, target_alignment = _alignment_snapshot(
                goal, microsector
            )
            flip_alignment = _should_flip_alignment(
                measured_alignment, target_alignment, measured_lag, target_lag
            )
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
                base_rationale = (
                    f"{self.operator_label} aplicado al nodo {node_label} en microsector "
                    f"{microsector.index}. La estrategia del objetivo destaca a "
                    f"{dominant_list or 'los nodos dominantes'} y fija ν_f={target_nu_f:.2f}. "
                    f"Se midió ν_f medio {actual_nu_f:.2f} ({deviation:+.2f}), superando la "
                    f"tolerancia ±{tolerance:.2f} definida para {context.profile_label}. {alignment_summary}"
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
            category = phase_family(microsector.active_phase)
            target_text = ""
            if target_rho:
                target_text += f" Objetivo ρ≈{target_rho:.2f}."
            if target_exc:
                target_text += f" ν_exc ref {target_exc:.2f}Hz."
            rationale = (
                f"Detune ratio ρ={rho:.2f} en microsector {microsector.index}"
                f" cae por debajo del umbral {threshold:.2f} con ∇Res {d_nfr_res:+.2f}."
                f"{target_text} Revisa barras estabilizadoras y amortiguadores"
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
                DetuneRatioRule(priority=24),
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
        if self.profile_manager is not None:
            snapshot = self.profile_manager.resolve(resolved_car, resolved_track, base_profile)
            profile = snapshot.thresholds
        else:
            profile = base_profile
        return RuleContext(
            car_model=resolved_car,
            track_name=resolved_track,
            thresholds=profile,
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
