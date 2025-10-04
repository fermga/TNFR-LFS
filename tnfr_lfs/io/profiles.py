"""Persistence helpers for car/track recommendation profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, MutableMapping, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from ..recommender.rules import (
    DEFAULT_THRESHOLD_LIBRARY,
    DEFAULT_THRESHOLD_PROFILE,
    ThresholdProfile,
    lookup_threshold_profile,
)


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


@dataclass
class ProfileRecord:
    """Mutable representation of a persistent profile entry."""

    car_model: str
    track_name: str
    objectives: ProfileObjectives
    tolerances: ProfileTolerances
    weights: Dict[str, Dict[str, float]]
    pending_plan: Dict[str, float] = field(default_factory=dict)
    pending_reference: StintMetrics | None = None
    last_result: StintMetrics | None = None

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
    ) -> "ProfileRecord":
        payload_weights = _normalise_phase_weights(
            weights if weights is not None else base_profile.phase_weights
        )
        payload_objectives = objectives or ProfileObjectives()
        payload_tolerances = tolerances or ProfileTolerances(
            entry=base_profile.entry_delta_tolerance,
            apex=base_profile.apex_delta_tolerance,
            exit=base_profile.exit_delta_tolerance,
            piano=base_profile.piano_delta_tolerance,
        )
        return cls(
            car_model=car_model,
            track_name=track_name,
            objectives=payload_objectives,
            tolerances=payload_tolerances,
            weights=payload_weights,
        )

    def to_threshold_profile(self, base_profile: ThresholdProfile) -> ThresholdProfile:
        merged_weights = _merge_phase_weights(base_profile.phase_weights, self.weights)
        return ThresholdProfile(
            entry_delta_tolerance=self.tolerances.entry,
            apex_delta_tolerance=self.tolerances.apex,
            exit_delta_tolerance=self.tolerances.exit,
            piano_delta_tolerance=self.tolerances.piano,
            phase_targets=base_profile.phase_targets,
            phase_weights=merged_weights,
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
            current = profile.get("__default__", 1.0)
            boosted = current * (1.0 + base_boost * weight_factor)
            profile["__default__"] = round(min(boosted, 5.0), 6)


@dataclass(frozen=True)
class ProfileSnapshot:
    """Resolved profile state exposed to consumers."""

    thresholds: ThresholdProfile
    objectives: ProfileObjectives
    pending_plan: Mapping[str, float]
    last_result: StintMetrics | None
    phase_weights: Mapping[str, Mapping[str, float] | float]


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
        thresholds = record.to_threshold_profile(reference)
        record.weights = _normalise_phase_weights(thresholds.phase_weights)
        return ProfileSnapshot(
            thresholds=thresholds,
            objectives=record.objectives,
            pending_plan=MappingProxyType(dict(record.pending_plan)),
            last_result=record.last_result,
            phase_weights=thresholds.phase_weights,
        )

    def register_plan(
        self,
        car_model: str,
        track_name: str,
        phases: Mapping[str, float],
        baseline_metrics: Tuple[float, float] | None = None,
    ) -> None:
        if not phases:
            return
        key = (car_model, track_name)
        record = self._ensure_record(key)
        record.pending_plan = {str(phase): float(value) for phase, value in phases.items() if value}
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
            record.pending_plan = {}
            record.pending_reference = None
        record.last_result = metrics
        self.save()

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
    record = ProfileRecord.from_base(
        car_model,
        track_name,
        base_profile,
        objectives=objectives,
        tolerances=tolerances,
        weights=weights,
    )
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
    last = spec.get("last_result")
    if isinstance(last, Mapping):
        sense_value = last.get("sense_index")
        delta_value = last.get("delta_nfr")
        if isinstance(sense_value, (int, float)) and isinstance(delta_value, (int, float)):
            record.last_result = StintMetrics(
                sense_index=float(sense_value),
                delta_nfr=abs(float(delta_value)),
            )
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

