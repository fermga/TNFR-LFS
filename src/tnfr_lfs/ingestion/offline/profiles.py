"""Persistence helpers for car/track recommendation profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from tnfr_core.equations.archetypes import PhaseArchetypeTargets
from tnfr_lfs.recommender.rules import (
    DEFAULT_THRESHOLD_LIBRARY,
    DEFAULT_THRESHOLD_PROFILE,
    ThresholdProfile,
    lookup_threshold_profile,
)


GRADIENT_SMOOTHING = 0.35


def _default_aero_profiles() -> Dict[str, AeroProfile]:
    return {"race": AeroProfile(), "stint_save": AeroProfile()}


@dataclass(frozen=True)
class ProfileObjectives:
    """Target objectives tracked for a car/track profile."""

    target_delta_nfr: float = 0.0
    target_sense_index: float = 0.75


@dataclass(frozen=True)
class ProfileTolerances:
    """Phase tolerances persisted alongside a profile."""

    entry: float
    apex: float
    exit: float
    piano: float


@dataclass(frozen=True)
class StintMetrics:
    """Aggregated ΔNFR/SI metrics registered for a stint."""

    sense_index: float
    delta_nfr: float


@dataclass(frozen=True)
class AeroProfile:
    """Persistent aero balance targets for specific strategies."""

    low_speed_target: float = 0.0
    high_speed_target: float = 0.0


@dataclass
class ProfileRecord:
    """Mutable representation of a persistent profile entry."""

    car_model: str
    track_name: str
    objectives: ProfileObjectives
    tolerances: ProfileTolerances
    weights: Dict[str, Dict[str, float]]
    session_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    session_hints: Dict[str, Any] = field(default_factory=dict)
    pending_plan: Dict[str, float] = field(default_factory=dict)
    pending_reference: StintMetrics | None = None
    last_result: StintMetrics | None = None
    jacobian: Dict[str, Dict[str, float]] = field(default_factory=dict)
    phase_jacobian: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    pending_jacobian: Dict[str, Dict[str, float]] | None = None
    pending_phase_jacobian: Dict[str, Dict[str, Dict[str, float]]] | None = None
    tyre_offsets: Dict[str, float] = field(default_factory=dict)
    aero_profiles: Dict[str, AeroProfile] = field(default_factory=lambda: dict(_default_aero_profiles()))

    @classmethod
    def from_base(
        cls,
        car_model: str,
        track_name: str,
        base_profile: ThresholdProfile,
        *,
        objectives: ProfileObjectives | None = None,
        tolerances: ProfileTolerances | None = None,
        weights: Mapping[str, Mapping[str, float]] | None = None,
        aero_profiles: Mapping[str, AeroProfile] | Mapping[str, Mapping[str, float]] | None = None,
    ) -> "ProfileRecord":
        payload_weights = (
            _normalise_phase_weights(weights) if weights is not None else {}
        )
        payload_objectives = objectives or ProfileObjectives()
        payload_tolerances = tolerances or ProfileTolerances(
            entry=base_profile.entry_delta_tolerance,
            apex=base_profile.apex_delta_tolerance,
            exit=base_profile.exit_delta_tolerance,
            piano=base_profile.piano_delta_tolerance,
        )
        payload_aero = _normalise_aero_profiles(aero_profiles)
        if not payload_aero:
            payload_aero = _default_aero_profiles()
        return cls(
            car_model=car_model,
            track_name=track_name,
            objectives=payload_objectives,
            tolerances=payload_tolerances,
            weights=payload_weights,
            aero_profiles=dict(payload_aero),
        )

    def to_threshold_profile(self, base_profile: ThresholdProfile) -> ThresholdProfile:
        merged_weights: Mapping[str, Mapping[str, float] | float] = (
            base_profile.phase_weights
        )
        if self.session_weights:
            merged_weights = _merge_phase_weights(merged_weights, self.session_weights)
        merged_weights = _merge_phase_weights(merged_weights, self.weights)
        rho_threshold = base_profile.rho_detune_threshold
        hint_rho = self.session_hints.get("rho_detune_threshold")
        if isinstance(hint_rho, (int, float)):
            rho_threshold = float(hint_rho)
        else:
            try:
                rho_threshold = float(hint_rho)
            except (TypeError, ValueError):
                rho_threshold = base_profile.rho_detune_threshold
        return ThresholdProfile(
            entry_delta_tolerance=self.tolerances.entry,
            apex_delta_tolerance=self.tolerances.apex,
            exit_delta_tolerance=self.tolerances.exit,
            piano_delta_tolerance=self.tolerances.piano,
            rho_detune_threshold=rho_threshold,
            phase_targets=base_profile.phase_targets,
            phase_weights=merged_weights,
            archetype_phase_targets=base_profile.archetype_phase_targets,
            robustness=base_profile.robustness,
        )

    def apply_mutation(
        self,
        phases: Mapping[str, float],
        reference: StintMetrics,
        outcome: StintMetrics,
    ) -> None:
        si_gain = max(0.0, outcome.sense_index - reference.sense_index)
        delta_drop = max(0.0, reference.delta_nfr - outcome.delta_nfr)
        if not phases or (si_gain <= 0.0 and delta_drop <= 0.0):
            return
        total_intensity = sum(max(0.0, value) for value in phases.values())
        if total_intensity <= 0.0:
            return
        base_boost = min(0.3, max(0.02, si_gain * 0.5 + delta_drop * 0.1))
        for phase, intensity in phases.items():
            weight_factor = max(0.0, intensity) / total_intensity
            if weight_factor <= 0.0:
                continue
            profile = self.weights.setdefault(phase, {})
            current = profile.get("__default__")
            if current is None:
                session_profile = self.session_weights.get(phase, {})
                if isinstance(session_profile, Mapping):
                    current = session_profile.get("__default__")
            if current is None:
                current = 1.0
            boosted = current * (1.0 + base_boost * weight_factor)
            profile["__default__"] = round(min(boosted, 5.0), 6)

    def update_jacobian_history(
        self,
        jacobian: Mapping[str, Mapping[str, float]] | None,
        phase_jacobian: Mapping[str, Mapping[str, Mapping[str, float]]] | None,
        *,
        smoothing: float = GRADIENT_SMOOTHING,
    ) -> None:
        if jacobian:
            for metric, derivatives in jacobian.items():
                if not isinstance(derivatives, Mapping):
                    continue
                store = self.jacobian.setdefault(str(metric), {})
                for parameter, value in derivatives.items():
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    previous = store.get(str(parameter))
                    store[str(parameter)] = _smooth(previous, numeric, smoothing)
        if phase_jacobian:
            for phase, metrics in phase_jacobian.items():
                if not isinstance(metrics, Mapping):
                    continue
                phase_entry = self.phase_jacobian.setdefault(str(phase), {})
                for metric, derivatives in metrics.items():
                    if not isinstance(derivatives, Mapping):
                        continue
                    metric_entry = phase_entry.setdefault(str(metric), {})
                    for parameter, value in derivatives.items():
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            continue
                        previous = metric_entry.get(str(parameter))
                        metric_entry[str(parameter)] = _smooth(
                            previous, numeric, smoothing
                        )


@dataclass(frozen=True)
class ProfileSnapshot:
    """Resolved profile state exposed to consumers."""

    thresholds: ThresholdProfile
    objectives: ProfileObjectives
    pending_plan: Mapping[str, float]
    session_weights: Mapping[str, Mapping[str, float]]
    session_hints: Mapping[str, Any]
    last_result: StintMetrics | None
    phase_weights: Mapping[str, Mapping[str, float] | float]
    jacobian: Mapping[str, Mapping[str, float]]
    phase_jacobian: Mapping[str, Mapping[str, Mapping[str, float]]]
    tyre_offsets: Mapping[str, float]
    archetype_targets: Mapping[str, Mapping[str, PhaseArchetypeTargets]]
    aero_profiles: Mapping[str, AeroProfile]


class ProfileManager:
    """Handle persistence of car/track recommendation profiles."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        threshold_library: Mapping[str, Mapping[str, ThresholdProfile]] | None = None,
    ) -> None:
        self.path = Path(path or "profiles.toml")
        self._profiles: Dict[Tuple[str, str], ProfileRecord] = {}
        self._base_profiles: Dict[Tuple[str, str], ThresholdProfile] = {}
        self._threshold_library = threshold_library
        self._load()

    def resolve(
        self,
        car_model: str,
        track_name: str,
        base_profile: ThresholdProfile | None = None,
        *,
        session: Mapping[str, Any] | None = None,
    ) -> ProfileSnapshot:
        key = (car_model, track_name)
        if base_profile is None:
            reference = self._resolve_base_profile(car_model, track_name)
        else:
            reference = base_profile
        self._base_profiles[key] = reference
        record = self._profiles.get(key)
        if record is None:
            record = ProfileRecord.from_base(car_model, track_name, reference)
            self._profiles[key] = record
        if isinstance(session, Mapping):
            weights_payload = session.get("weights")
            hints_payload = session.get("hints")
            session_weights = (
                _normalise_phase_weights(weights_payload)
                if isinstance(weights_payload, Mapping)
                else {}
            )
            session_hints = (
                _normalise_session_hints(hints_payload)
                if isinstance(hints_payload, Mapping)
                else {}
            )
            record.session_weights = session_weights
            record.session_hints = session_hints
        elif session is not None:
            record.session_weights = {}
            record.session_hints = {}
        elif not record.session_weights:
            record.session_weights = _normalise_phase_weights(reference.phase_weights)
        thresholds = record.to_threshold_profile(reference)
        record.weights = (
            _normalise_phase_weights(record.weights) if record.weights else {}
        )
        if record.session_weights:
            record.session_weights = _normalise_phase_weights(record.session_weights)
        if record.session_hints:
            record.session_hints = _normalise_session_hints(record.session_hints)
        aero_payload = _normalise_aero_profiles(record.aero_profiles)
        if aero_payload:
            record.aero_profiles = dict(aero_payload)
        else:
            record.aero_profiles = dict(_default_aero_profiles())
        jacobian_snapshot = {
            str(metric): MappingProxyType(dict(values))
            for metric, values in record.jacobian.items()
        }
        phase_jacobian_snapshot = {
            str(phase): MappingProxyType(
                {
                    str(metric): MappingProxyType(dict(values))
                    for metric, values in metrics.items()
                }
            )
            for phase, metrics in record.phase_jacobian.items()
        }
        offsets_snapshot = MappingProxyType(
            {str(key): float(value) for key, value in record.tyre_offsets.items()}
        )
        aero_snapshot = MappingProxyType({str(mode): profile for mode, profile in record.aero_profiles.items()})
        session_weight_snapshot = MappingProxyType(
            {
                str(phase): MappingProxyType(dict(values))
                for phase, values in record.session_weights.items()
            }
        )
        session_hint_snapshot = MappingProxyType(
            {str(key): _freeze_hint_value(value) for key, value in record.session_hints.items()}
        )
        return ProfileSnapshot(
            thresholds=thresholds,
            objectives=record.objectives,
            pending_plan=MappingProxyType(dict(record.pending_plan)),
            session_weights=session_weight_snapshot,
            session_hints=session_hint_snapshot,
            last_result=record.last_result,
            phase_weights=thresholds.phase_weights,
            jacobian=MappingProxyType(jacobian_snapshot),
            phase_jacobian=MappingProxyType(phase_jacobian_snapshot),
            tyre_offsets=offsets_snapshot,
            archetype_targets=thresholds.archetype_phase_targets,
            aero_profiles=aero_snapshot,
        )

    def register_plan(
        self,
        car_model: str,
        track_name: str,
        phases: Mapping[str, float],
        baseline_metrics: Tuple[float, float] | None = None,
        *,
        jacobian: Mapping[str, Mapping[str, float]] | None = None,
        phase_jacobian: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
    ) -> None:
        if not phases:
            return
        key = (car_model, track_name)
        record = self._ensure_record(key)
        record.pending_plan = {str(phase): float(value) for phase, value in phases.items() if value}
        overall = _normalise_overall_jacobian(jacobian)
        phase_gradients = _normalise_phase_jacobian(phase_jacobian)
        record.pending_jacobian = overall or None
        record.pending_phase_jacobian = phase_gradients or None
        if baseline_metrics is not None:
            record.pending_reference = StintMetrics(
                sense_index=float(baseline_metrics[0]),
                delta_nfr=abs(float(baseline_metrics[1])),
            )
            record.last_result = record.pending_reference
        self.save()

    def register_result(
        self,
        car_model: str,
        track_name: str,
        sense_index: float,
        delta_nfr: float,
    ) -> None:
        key = (car_model, track_name)
        record = self._ensure_record(key)
        metrics = StintMetrics(sense_index=float(sense_index), delta_nfr=abs(float(delta_nfr)))
        if record.pending_plan and record.pending_reference is not None:
            if (
                metrics.sense_index > record.pending_reference.sense_index
                and metrics.delta_nfr < record.pending_reference.delta_nfr
            ):
                record.apply_mutation(record.pending_plan, record.pending_reference, metrics)
            record.update_jacobian_history(
                record.pending_jacobian or {},
                record.pending_phase_jacobian or {},
            )
            record.pending_plan = {}
            record.pending_reference = None
            record.pending_jacobian = None
            record.pending_phase_jacobian = None
        record.last_result = metrics
        self.save()

    def update_tyre_offsets(
        self,
        car_model: str,
        track_name: str,
        offsets: Mapping[str, float],
    ) -> None:
        if not offsets:
            return
        key = (car_model, track_name)
        record = self._ensure_record(key)
        record.tyre_offsets = {
            str(axis): float(value)
            for axis, value in offsets.items()
            if isinstance(value, (int, float))
        }
        self.save()

    def update_aero_profile(
        self,
        car_model: str,
        track_name: str,
        mode: str,
        *,
        low_speed_target: float | None = None,
        high_speed_target: float | None = None,
    ) -> None:
        mode_key = str(mode or "").strip()
        if not mode_key:
            return
        key = (car_model, track_name)
        record = self._ensure_record(key)
        baseline = record.aero_profiles.get(mode_key, AeroProfile())
        new_profile = AeroProfile(
            low_speed_target=
            float(low_speed_target)
            if isinstance(low_speed_target, (int, float))
            else baseline.low_speed_target,
            high_speed_target=
            float(high_speed_target)
            if isinstance(high_speed_target, (int, float))
            else baseline.high_speed_target,
        )
        record.aero_profiles[mode_key] = new_profile
        self.save()

    def gradient_history(
        self,
        car_model: str,
        track_name: str,
    ) -> tuple[Mapping[str, Mapping[str, float]], Mapping[str, Mapping[str, Mapping[str, float]]]]:
        record = self._profiles.get((car_model, track_name))
        if record is None:
            return MappingProxyType({}), MappingProxyType({})
        overall = {
            str(metric): MappingProxyType(dict(values))
            for metric, values in record.jacobian.items()
        }
        phases = {
            str(phase): MappingProxyType(
                {
                    str(metric): MappingProxyType(dict(values))
                    for metric, values in metrics.items()
                }
            )
            for phase, metrics in record.phase_jacobian.items()
        }
        return MappingProxyType(overall), MappingProxyType(phases)

    def update_objectives(
        self,
        car_model: str,
        track_name: str,
        target_delta_nfr: float,
        target_sense_index: float,
    ) -> None:
        key = (car_model, track_name)
        record = self._ensure_record(key)
        record.objectives = ProfileObjectives(
            target_delta_nfr=float(target_delta_nfr),
            target_sense_index=float(target_sense_index),
        )
        self.save()

    def save(self) -> None:
        if not self._profiles:
            if self.path.exists():
                self.path.unlink()
            return
        lines: list[str] = []
        root: Dict[str, MutableMapping[str, object]] = {}
        for (car_model, track_name), record in sorted(self._profiles.items()):
            car_entry = root.setdefault(car_model, {})
            car_entry[track_name] = record
        for car_model in sorted(root):
            tracks = root[car_model]
            for track_name in sorted(tracks):
                record = tracks[track_name]
                lines.extend(_serialise_record(car_model, track_name, record))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf8")

    def _ensure_record(self, key: Tuple[str, str]) -> ProfileRecord:
        record = self._profiles.get(key)
        if record is None:
            base = self._base_profiles.get(key)
            if base is None:
                base = self._resolve_base_profile(key[0], key[1])
            record = ProfileRecord.from_base(key[0], key[1], base)
            self._profiles[key] = record
        return record

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            payload = tomllib.load(handle)
        root = payload.get("profiles", payload)
        if not isinstance(root, Mapping):
            return
        for car_model, tracks in root.items():
            if not isinstance(tracks, Mapping):
                continue
            for track_name, spec in tracks.items():
                if not isinstance(spec, Mapping):
                    continue
                car_key = str(car_model)
                track_key = str(track_name)
                base_profile = self._resolve_base_profile(car_key, track_key)
                record = _record_from_mapping(car_key, track_key, spec, base_profile)
                self._profiles[(car_key, track_key)] = record
                self._base_profiles[(car_key, track_key)] = base_profile

    def _resolve_base_profile(self, car_model: str, track_name: str) -> ThresholdProfile:
        key = (car_model, track_name)
        cached = self._base_profiles.get(key)
        if cached is not None:
            return cached
        library = self._threshold_library or DEFAULT_THRESHOLD_LIBRARY
        profile = lookup_threshold_profile(car_model, track_name, library)
        self._base_profiles[key] = profile
        return profile


def _normalise_phase_weights(
    weights: Mapping[str, Mapping[str, float]] | Mapping[str, float]
) -> Dict[str, Dict[str, float]]:
    normalised: Dict[str, Dict[str, float]] = {}
    for phase, profile in weights.items():
        if isinstance(profile, Mapping):
            normalised[str(phase)] = {
                str(node): float(value)
                for node, value in profile.items()
                if isinstance(value, (int, float))
            }
        elif isinstance(profile, (int, float)):
            normalised[str(phase)] = {"__default__": float(profile)}
    return normalised


def _merge_phase_weights(
    base: Mapping[str, Mapping[str, float] | float],
    overrides: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Mapping[str, float] | float]:
    merged: Dict[str, Mapping[str, float] | float] = {}
    for phase, profile in base.items():
        if isinstance(profile, Mapping):
            merged[str(phase)] = {
                str(node): float(value)
                for node, value in profile.items()
                if isinstance(value, (int, float))
            }
        elif isinstance(profile, (int, float)):
            merged[str(phase)] = float(profile)
    for phase, profile in overrides.items():
        base_profile = merged.get(phase)
        updated: Dict[str, float]
        if isinstance(base_profile, Mapping):
            updated = dict(base_profile)
        else:
            updated = {}
            if isinstance(base_profile, (int, float)):
                updated["__default__"] = float(base_profile)
        for node, value in profile.items():
            updated[str(node)] = float(value)
        merged[str(phase)] = MappingProxyType(dict(updated))
    for phase, profile in list(merged.items()):
        if isinstance(profile, dict):
            merged[phase] = MappingProxyType(dict(profile))
    return MappingProxyType(dict(merged))


def _record_from_mapping(
    car_model: str,
    track_name: str,
    spec: Mapping[str, object],
    base_profile: ThresholdProfile,
) -> ProfileRecord:
    objectives = _objectives_from_spec(spec.get("objectives"))
    tolerances = _tolerances_from_spec(spec.get("tolerances"))
    weights = _weights_from_spec(spec.get("weights"))
    aero_profiles = _normalise_aero_profiles(spec.get("aero_profiles"))
    record = ProfileRecord.from_base(
        car_model,
        track_name,
        base_profile,
        objectives=objectives,
        tolerances=tolerances,
        weights=weights,
        aero_profiles=aero_profiles,
    )
    jacobian_spec = spec.get("jacobian")
    if isinstance(jacobian_spec, Mapping):
        overall_spec = jacobian_spec.get("overall")
        phase_spec = jacobian_spec.get("phases")
        overall = _normalise_overall_jacobian(overall_spec)
        phase = _normalise_phase_jacobian(phase_spec)
        if overall:
            record.jacobian = overall
        if phase:
            record.phase_jacobian = phase
    pending = spec.get("pending_plan")
    if isinstance(pending, Mapping):
        phases = pending.get("phases")
        if isinstance(phases, Mapping):
            record.pending_plan = {
                str(phase): float(value)
                for phase, value in phases.items()
                if isinstance(value, (int, float))
            }
        reference_si = pending.get("reference_sense_index")
        reference_delta = pending.get("reference_delta_nfr")
        if isinstance(reference_si, (int, float)) and isinstance(reference_delta, (int, float)):
            record.pending_reference = StintMetrics(
                sense_index=float(reference_si),
                delta_nfr=abs(float(reference_delta)),
            )
        jacobian_pending = pending.get("jacobian")
        if isinstance(jacobian_pending, Mapping):
            overall_pending = _normalise_overall_jacobian(jacobian_pending.get("overall"))
            phase_pending = _normalise_phase_jacobian(jacobian_pending.get("phases"))
            record.pending_jacobian = overall_pending or None
            record.pending_phase_jacobian = phase_pending or None
    last = spec.get("last_result")
    if isinstance(last, Mapping):
        sense_value = last.get("sense_index")
        delta_value = last.get("delta_nfr")
        if isinstance(sense_value, (int, float)) and isinstance(delta_value, (int, float)):
            record.last_result = StintMetrics(
                sense_index=float(sense_value),
                delta_nfr=abs(float(delta_value)),
            )
    offsets = spec.get("tyre_offsets")
    if isinstance(offsets, Mapping):
        record.tyre_offsets = {
            str(axis): float(value)
            for axis, value in offsets.items()
            if isinstance(value, (int, float))
        }
    session_spec = spec.get("session")
    if isinstance(session_spec, Mapping):
        session_weights = session_spec.get("weights")
        if isinstance(session_weights, Mapping):
            record.session_weights = _normalise_phase_weights(session_weights)
        session_hints = session_spec.get("hints")
        if isinstance(session_hints, Mapping):
            record.session_hints = _normalise_session_hints(session_hints)
    return record


def _objectives_from_spec(payload: object) -> ProfileObjectives:
    if not isinstance(payload, Mapping):
        return ProfileObjectives()
    delta = payload.get("target_delta_nfr")
    sense = payload.get("target_sense_index")
    return ProfileObjectives(
        target_delta_nfr=float(delta) if isinstance(delta, (int, float)) else 0.0,
        target_sense_index=float(sense) if isinstance(sense, (int, float)) else 0.75,
    )


def _tolerances_from_spec(payload: object) -> ProfileTolerances:
    if not isinstance(payload, Mapping):
        base = DEFAULT_THRESHOLD_PROFILE
        return ProfileTolerances(
            entry=base.entry_delta_tolerance,
            apex=base.apex_delta_tolerance,
            exit=base.exit_delta_tolerance,
            piano=base.piano_delta_tolerance,
        )
    base = DEFAULT_THRESHOLD_PROFILE
    entry = payload.get("entry")
    apex = payload.get("apex")
    exit_ = payload.get("exit")
    piano = payload.get("piano")
    return ProfileTolerances(
        entry=float(entry) if isinstance(entry, (int, float)) else base.entry_delta_tolerance,
        apex=float(apex) if isinstance(apex, (int, float)) else base.apex_delta_tolerance,
        exit=float(exit_) if isinstance(exit_, (int, float)) else base.exit_delta_tolerance,
        piano=float(piano) if isinstance(piano, (int, float)) else base.piano_delta_tolerance,
    )


def _weights_from_spec(payload: object) -> Mapping[str, Mapping[str, float]] | None:
    if not isinstance(payload, Mapping):
        return None
    weights: Dict[str, Dict[str, float]] = {}
    for phase, profile in payload.items():
        if not isinstance(profile, Mapping):
            continue
        weights[str(phase)] = {
            str(node): float(value)
            for node, value in profile.items()
            if isinstance(value, (int, float))
        }
    return weights


def _normalise_aero_profiles(
    payload: Mapping[str, AeroProfile] | Mapping[str, Mapping[str, float]] | None,
) -> Dict[str, AeroProfile]:
    if not isinstance(payload, Mapping):
        return {}
    profiles: Dict[str, AeroProfile] = {}
    for mode, spec in payload.items():
        if isinstance(spec, AeroProfile):
            profiles[str(mode)] = spec
            continue
        if not isinstance(spec, Mapping):
            continue
        low = spec.get("low_speed_target")
        high = spec.get("high_speed_target")
        profiles[str(mode)] = AeroProfile(
            low_speed_target=float(low) if isinstance(low, (int, float)) else 0.0,
            high_speed_target=float(high) if isinstance(high, (int, float)) else 0.0,
        )
    return profiles


def _normalise_overall_jacobian(
    payload: Mapping[str, Mapping[str, float]] | None,
) -> Dict[str, Dict[str, float]]:
    if not isinstance(payload, Mapping):
        return {}
    normalised: Dict[str, Dict[str, float]] = {}
    for metric, derivatives in payload.items():
        if not isinstance(derivatives, Mapping):
            continue
        normalised[str(metric)] = {
            str(parameter): float(value)
            for parameter, value in derivatives.items()
            if isinstance(value, (int, float))
        }
    return normalised


def _normalise_phase_jacobian(
    payload: Mapping[str, Mapping[str, Mapping[str, float]]] | None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not isinstance(payload, Mapping):
        return {}
    normalised: Dict[str, Dict[str, Dict[str, float]]] = {}
    for phase, metrics in payload.items():
        if not isinstance(metrics, Mapping):
            continue
        phase_entry: Dict[str, Dict[str, float]] = {}
        for metric, derivatives in metrics.items():
            if not isinstance(derivatives, Mapping):
                continue
            phase_entry[str(metric)] = {
                str(parameter): float(value)
                for parameter, value in derivatives.items()
                if isinstance(value, (int, float))
            }
        if phase_entry:
            normalised[str(phase)] = phase_entry
    return normalised


def _smooth(previous: float | None, new: float, smoothing: float) -> float:
    if previous is None:
        return float(new)
    return float(previous) * (1.0 - smoothing) + float(new) * smoothing


def _normalise_session_hints(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
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
    return hints


def _freeze_hint_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_hint_value(item) for item in value)
    return value


def _serialise_record(car_model: str, track_name: str, record: ProfileRecord) -> list[str]:
    lines: list[str] = []
    base_prefix = ["profiles", car_model, track_name]
    lines.append(_table_header(base_prefix + ["objectives"]))
    lines.append(
        f"target_delta_nfr = {_format_float(record.objectives.target_delta_nfr)}"
    )
    lines.append(
        f"target_sense_index = {_format_float(record.objectives.target_sense_index)}"
    )
    lines.append("")
    lines.append(_table_header(base_prefix + ["tolerances"]))
    lines.append(f"entry = {_format_float(record.tolerances.entry)}")
    lines.append(f"apex = {_format_float(record.tolerances.apex)}")
    lines.append(f"exit = {_format_float(record.tolerances.exit)}")
    lines.append(f"piano = {_format_float(record.tolerances.piano)}")
    for phase, profile in sorted(record.weights.items()):
        lines.append("")
        lines.append(_table_header(base_prefix + ["weights", phase]))
        for node, value in sorted(profile.items()):
            lines.append(f"{_format_key(node)} = {_format_float(value)}")
    if record.session_weights:
        for phase, profile in sorted(record.session_weights.items()):
            lines.append("")
            lines.append(_table_header(base_prefix + ["session", "weights", phase]))
            for node, value in sorted(profile.items()):
                lines.append(f"{_format_key(node)} = {_format_float(value)}")
    if record.session_hints:
        lines.append("")
        lines.append(_table_header(base_prefix + ["session", "hints"]))
        for key, value in sorted(record.session_hints.items()):
            formatted = _format_hint_value(value)
            if formatted is None:
                continue
            lines.append(f"{_format_key(key)} = {formatted}")
    if record.tyre_offsets:
        lines.append("")
        lines.append(_table_header(base_prefix + ["tyre_offsets"]))
        for axis, value in sorted(record.tyre_offsets.items()):
            lines.append(f"{_format_key(axis)} = {_format_float(value)}")
    if record.aero_profiles:
        for mode, profile in sorted(record.aero_profiles.items()):
            if abs(profile.low_speed_target) <= 1e-9 and abs(profile.high_speed_target) <= 1e-9:
                continue
            lines.append("")
            lines.append(_table_header(base_prefix + ["aero_profiles", mode]))
            lines.append(f"low_speed_target = {_format_float(profile.low_speed_target)}")
            lines.append(f"high_speed_target = {_format_float(profile.high_speed_target)}")
    if record.jacobian:
        for metric, derivatives in sorted(record.jacobian.items()):
            lines.append("")
            lines.append(_table_header(base_prefix + ["jacobian", "overall", metric]))
            for parameter, value in sorted(derivatives.items()):
                lines.append(f"{_format_key(parameter)} = {_format_float(value)}")
    if record.phase_jacobian:
        for phase, metrics in sorted(record.phase_jacobian.items()):
            for metric, derivatives in sorted(metrics.items()):
                lines.append("")
                lines.append(
                    _table_header(base_prefix + ["jacobian", "phases", phase, metric])
                )
                for parameter, value in sorted(derivatives.items()):
                    lines.append(f"{_format_key(parameter)} = {_format_float(value)}")
    if record.pending_plan:
        lines.append("")
        lines.append(_table_header(base_prefix + ["pending_plan"]))
        if record.pending_reference is not None:
            lines.append(
                f"reference_sense_index = {_format_float(record.pending_reference.sense_index)}"
            )
            lines.append(
                f"reference_delta_nfr = {_format_float(record.pending_reference.delta_nfr)}"
            )
        if record.pending_jacobian:
            for metric, derivatives in sorted(record.pending_jacobian.items()):
                lines.append("")
                lines.append(
                    _table_header(
                        base_prefix + ["pending_plan", "jacobian", "overall", metric]
                    )
                )
                for parameter, value in sorted(derivatives.items()):
                    lines.append(f"{_format_key(parameter)} = {_format_float(value)}")
        if record.pending_phase_jacobian:
            for phase, metrics in sorted(record.pending_phase_jacobian.items()):
                for metric, derivatives in sorted(metrics.items()):
                    lines.append("")
                    lines.append(
                        _table_header(
                            base_prefix
                            + ["pending_plan", "jacobian", "phases", phase, metric]
                        )
                    )
                    for parameter, value in sorted(derivatives.items()):
                        lines.append(f"{_format_key(parameter)} = {_format_float(value)}")
        lines.append("")
        lines.append(_table_header(base_prefix + ["pending_plan", "phases"]))
        for phase, value in sorted(record.pending_plan.items()):
            lines.append(f"{_format_key(phase)} = {_format_float(value)}")
    if record.last_result is not None:
        lines.append("")
        lines.append(_table_header(base_prefix + ["last_result"]))
        lines.append(f"sense_index = {_format_float(record.last_result.sense_index)}")
        lines.append(f"delta_nfr = {_format_float(record.last_result.delta_nfr)}")
    lines.append("")
    return lines


def _format_float(value: float) -> str:
    return ("{0:.6f}".format(float(value))).rstrip("0").rstrip(".") or "0"


def _format_hint_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _format_float(float(value))
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        formatted_items = [
            item for item in (_format_hint_value(entry) for entry in value) if item
        ]
        return f"[{', '.join(formatted_items)}]"
    return None


def _format_key(token: str) -> str:
    if token.replace("_", "").isalnum() and token == token.lower():
        return token
    escaped = token.replace("\"", "\\\"")
    return f'"{escaped}"'


def _table_header(parts: list[str]) -> str:
    tokens = [_format_key(part) for part in parts]
    return "[" + ".".join(tokens) + "]"


__all__ = [
    "ProfileManager",
    "ProfileObjectives",
    "ProfileSnapshot",
    "ProfileTolerances",
    "StintMetrics",
]

