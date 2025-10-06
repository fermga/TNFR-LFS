"""Command line entry point for TNFR × LFS."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from statistics import mean, fmean
from time import monotonic, sleep
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from ..acquisition import (
    ButtonLayout,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    InSimClient,
    OverlayManager,
    OutGaugeUDPClient,
    OutSimClient,
    OutSimUDPClient,
    TelemetryFusion,
)
from .osd import OSDController, TelemetryHUD
from ..core.epi import EPIExtractor, TelemetryRecord, NU_F_NODE_DEFAULTS
from ..core.metrics import compute_aero_coherence, resolve_aero_mechanical_coherence
from ..core.phases import replicate_phase_aliases
from ..core.resonance import analyse_modal_resonance
from ..core.operators import orchestrate_delta_metrics
from ..core.segmentation import (
    Microsector,
    detect_quiet_microsector_streaks,
    microsector_stability_metrics,
    segment_microsectors,
)
from ..analysis import SUPPORTED_LAP_METRICS, ab_compare_by_lap, compute_session_robustness
from ..exporters import (
    REPORT_ARTIFACT_FORMATS,
    build_coherence_map_payload,
    build_delta_bifurcation_payload,
    build_operator_trajectories_payload,
    exporters_registry,
    normalise_set_output_name,
    render_coherence_map,
    render_delta_bifurcation,
    render_operator_trajectories,
)
from ..exporters.setup_plan import SetupChange, SetupPlan
from ..io import logs
from ..io.profiles import ProfileManager, ProfileObjectives, ProfileSnapshot
from ..recommender import (
    Plan,
    RecommendationEngine,
    SetupPlanner,
    pareto_front,
    sweep_candidates,
)
from ..recommender.rules import ThresholdProfile
from ..session import format_session_messages
from ..track_loader import (
    Track,
    TrackConfig,
    assemble_session_weights,
    load_modifiers as load_track_modifiers,
    load_track as load_track_manifest,
    load_track_profiles,
)
from ..config_loader import (
    Car as PackCar,
    Profile as PackProfile,
    load_cars as load_pack_cars,
    load_lfs_class_overrides as load_pack_lfs_class_overrides,
    load_profiles as load_pack_profiles,
    resolve_targets as resolve_pack_targets,
)


Records = List[TelemetryRecord]
Bundles = Sequence[Any]

CONFIG_ENV_VAR = "TNFR_LFS_CONFIG"
DEFAULT_CONFIG_FILENAME = "tnfr-lfs.toml"
DEFAULT_OUTPUT_DIR = Path("out")
PROFILES_ENV_VAR = "TNFR_LFS_PROFILES"
DEFAULT_PROFILES_FILENAME = "profiles.toml"

WHEEL_SUFFIXES: Tuple[str, ...] = ("fl", "fr", "rl", "rr")
WHEEL_LABELS = MappingProxyType({
    "fl": "FL",
    "fr": "FR",
    "rl": "RL",
    "rr": "RR",
})
TEMPERATURE_MEAN_KEYS = MappingProxyType({suffix: f"tyre_temp_{suffix}" for suffix in WHEEL_SUFFIXES})
TEMPERATURE_STD_KEYS = MappingProxyType({
    suffix: f"{TEMPERATURE_MEAN_KEYS[suffix]}_std" for suffix in WHEEL_SUFFIXES
})
PRESSURE_MEAN_KEYS = MappingProxyType({suffix: f"tyre_pressure_{suffix}" for suffix in WHEEL_SUFFIXES})
PRESSURE_STD_KEYS = MappingProxyType({
    suffix: f"{PRESSURE_MEAN_KEYS[suffix]}_std" for suffix in WHEEL_SUFFIXES
})

SUPPORTED_AB_METRICS: tuple[str, ...] = tuple(sorted(SUPPORTED_LAP_METRICS))


@dataclass(frozen=True, slots=True)
class ProfilesContext:
    """Container for CLI profile settings and pack resources."""

    storage_path: Path
    pack_profiles: Mapping[str, PackProfile]


@dataclass(frozen=True, slots=True)
class TrackSelection:
    """Normalized representation of a CLI track argument."""

    name: str
    layout: str | None = None
    manifest: Track | None = None
    config: TrackConfig | None = None

    @property
    def track_profile(self) -> str | None:
        return self.config.track_profile if self.config is not None else None


@dataclass(frozen=True, slots=True)
class SetupPlanContext:
    """Container storing the artefacts required to build setup outputs."""

    plan: Plan
    planner: SetupPlanner
    engine: RecommendationEngine
    bundles: Bundles
    microsectors: Sequence[Microsector]
    metrics: Mapping[str, Any]
    delta_metric: float
    session_payload: Mapping[str, Any] | None
    track_selection: TrackSelection
    track_name: str
    profile_manager: ProfileManager
    thresholds: ThresholdProfile
    snapshot: ProfileSnapshot | None
    objectives: ProfileObjectives
    tnfr_targets: Mapping[str, Any] | None
    car_metadata: Mapping[str, Any] | None
    pack_delta: float | None
    pack_si: float | None
    lap_segments: List[Records]
    records: Records
    cars: Mapping[str, PackCar]
    track_profiles: Mapping[str, Mapping[str, Any]]
    modifiers: Mapping[tuple[str, str], Mapping[str, Any]]
    class_overrides: Mapping[str, Any]
    profiles_ctx: ProfilesContext
    pack_root: Path | None


_LAYOUT_PATTERN = re.compile(r"^([A-Z]{2,})([0-9]{1,2}[A-Z]?)$")


def _resolve_pack_root(
    namespace: argparse.Namespace | None, config: Mapping[str, Any]
) -> Path | None:
    """Determine the root directory for an optional configuration pack."""

    candidate: Path | None = None
    if namespace is not None:
        raw = getattr(namespace, "pack_root", None)
        if raw:
            candidate = Path(raw)
    if candidate is None:
        paths_cfg = config.get("paths")
        if isinstance(paths_cfg, Mapping):
            raw = paths_cfg.get("pack_root")
            if isinstance(raw, str) and raw.strip():
                candidate = Path(raw)
    if candidate is None:
        return None
    return candidate.expanduser()


def _pack_data_dir(pack_root: Path | None, name: str) -> Path | None:
    """Resolve a directory inside ``data`` or at the root of the pack."""

    if pack_root is None:
        return None
    candidates: list[Path] = []
    if name == "modifiers":
        candidates.extend(
            [
                pack_root / "modifiers" / "combos",
                pack_root / "data" / "modifiers" / "combos",
                pack_root / "modifiers",
                pack_root / "data" / "modifiers",
            ]
        )
    candidates.extend([pack_root / "data" / name, pack_root / name])
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists() and expanded.is_dir():
            return expanded
    return None


def _parse_layout_code(value: str) -> tuple[str, str] | None:
    match = _LAYOUT_PATTERN.fullmatch(value.strip().upper())
    if match is None:
        return None
    slug, suffix = match.groups()
    return slug, f"{slug}{suffix}"


def _load_pack_track_profiles(pack_root: Path | None) -> Mapping[str, Mapping[str, Any]]:
    profiles_dir = _pack_data_dir(pack_root, "track_profiles")
    return load_track_profiles(profiles_dir) if profiles_dir is not None else load_track_profiles()


def _load_pack_modifiers(pack_root: Path | None) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    modifiers_dir = _pack_data_dir(pack_root, "modifiers")
    return (
        load_track_modifiers(modifiers_dir)
        if modifiers_dir is not None
        else load_track_modifiers()
    )


def _compute_setup_plan(
    namespace: argparse.Namespace, *, config: Mapping[str, Any]
) -> SetupPlanContext:
    records = _load_records(namespace.telemetry)
    pack_root = _resolve_pack_root(namespace, config)
    profiles_ctx = _resolve_profiles_path(config, pack_root=pack_root)
    profile_manager = ProfileManager(profiles_ctx.storage_path)
    cars = _load_pack_cars(pack_root)
    track_profiles = _load_pack_track_profiles(pack_root)
    modifiers = _load_pack_modifiers(pack_root)
    class_overrides = _load_pack_lfs_class_overrides(pack_root)
    tnfr_targets = _resolve_tnfr_targets(
        namespace.car_model,
        cars,
        profiles_ctx.pack_profiles,
        overrides=class_overrides,
    )
    car_metadata = _lookup_car_metadata(namespace.car_model, cars)
    pack_delta, pack_si = _extract_target_objectives(tnfr_targets)
    track_selection = _resolve_track_argument(None, config, pack_root=pack_root)
    track_name = track_selection.name or _default_track_name(config)
    engine = RecommendationEngine(
        car_model=namespace.car_model,
        track_name=track_name,
        profile_manager=profile_manager,
    )
    session_payload = _assemble_session_payload(
        namespace.car_model,
        track_selection,
        cars=cars,
        track_profiles=track_profiles,
        modifiers=modifiers,
    )
    if session_payload is not None:
        engine.session = session_payload
    bundles, microsectors, thresholds, snapshot = _compute_insights(
        records,
        car_model=namespace.car_model,
        track_name=track_name,
        engine=engine,
        profile_manager=profile_manager,
    )
    objectives = snapshot.objectives if snapshot else ProfileObjectives()
    if snapshot is None and pack_delta is not None:
        objectives = ProfileObjectives(
            target_delta_nfr=float(pack_delta),
            target_sense_index=(
                float(pack_si)
                if pack_si is not None
                else objectives.target_sense_index
            ),
        )
    profile_manager.update_objectives(
        namespace.car_model,
        track_name,
        objectives.target_delta_nfr,
        objectives.target_sense_index,
    )
    lap_segments = _group_records_by_lap(records)
    metrics = orchestrate_delta_metrics(
        lap_segments,
        objectives.target_delta_nfr,
        objectives.target_sense_index,
        coherence_window=3,
        recursion_decay=0.4,
        microsectors=microsectors,
        phase_weights=thresholds.phase_weights,
    )
    delta_metric = _effective_delta_metric(metrics)
    engine.register_stint_result(
        sense_index=metrics.get("sense_index", 0.0),
        delta_nfr=delta_metric,
        car_model=namespace.car_model,
        track_name=track_name,
    )
    planner = SetupPlanner(recommendation_engine=engine)
    plan = planner.plan(
        bundles,
        microsectors,
        car_model=namespace.car_model,
        track_name=track_name,
    )
    engine.register_plan(
        plan.recommendations,
        car_model=namespace.car_model,
        track_name=track_name,
        baseline_sense_index=metrics.get("sense_index", 0.0),
        baseline_delta_nfr=delta_metric,
        jacobian=plan.sensitivities,
        phase_jacobian=plan.phase_sensitivities,
    )
    return SetupPlanContext(
        plan=plan,
        planner=planner,
        engine=engine,
        bundles=bundles,
        microsectors=microsectors,
        metrics=metrics,
        delta_metric=delta_metric,
        session_payload=session_payload,
        track_selection=track_selection,
        track_name=track_name,
        profile_manager=profile_manager,
        thresholds=thresholds,
        snapshot=snapshot,
        objectives=objectives,
        tnfr_targets=tnfr_targets,
        car_metadata=car_metadata,
        pack_delta=pack_delta,
        pack_si=pack_si,
        lap_segments=lap_segments,
        records=records,
        cars=cars,
        track_profiles=track_profiles,
        modifiers=modifiers,
        class_overrides=class_overrides,
        profiles_ctx=profiles_ctx,
        pack_root=pack_root,
    )


def _load_pack_lfs_class_overrides(
    pack_root: Path | None,
) -> Mapping[str, Mapping[str, Any]]:
    if pack_root is None:
        return load_pack_lfs_class_overrides()

    candidates = [
        pack_root / "data" / "lfs_class_overrides.toml",
        pack_root / "lfs_class_overrides.toml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return load_pack_lfs_class_overrides(candidate)

    return load_pack_lfs_class_overrides(candidates[0])


def _load_pack_track(pack_root: Path | None, slug: str) -> Track:
    tracks_dir = _pack_data_dir(pack_root, "tracks")
    return (
        load_track_manifest(slug, tracks_dir)
        if tracks_dir is not None
        else load_track_manifest(slug)
    )


def _resolve_track_selection(track: str, *, pack_root: Path | None) -> TrackSelection:
    candidate = track.strip()
    if not candidate:
        return TrackSelection(name="")
    parsed = _parse_layout_code(candidate)
    if parsed is None:
        return TrackSelection(name=candidate)
    slug, layout_id = parsed
    try:
        manifest = _load_pack_track(pack_root, slug)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"No se encontró el manifest de pista '{slug}' en el pack ni en los recursos."
        ) from exc
    try:
        config = manifest.configs[layout_id]
    except KeyError as exc:
        raise SystemExit(
            f"El layout '{layout_id}' no existe en '{manifest.path.name}'."
        ) from exc
    return TrackSelection(name=layout_id, layout=layout_id, manifest=manifest, config=config)


def _load_pack_profiles(pack_root: Path | None) -> Mapping[str, PackProfile]:
    """Load TNFR profiles from a pack or fall back to bundled resources."""

    profiles_dir = _pack_data_dir(pack_root, "profiles")
    if profiles_dir is not None:
        return load_pack_profiles(profiles_dir)
    return load_pack_profiles()


def _load_pack_cars(pack_root: Path | None) -> Mapping[str, PackCar]:
    """Load car metadata from a pack or from the bundled dataset."""

    cars_dir = _pack_data_dir(pack_root, "cars")
    if cars_dir is not None:
        return load_pack_cars(cars_dir)
    return load_pack_cars()


def _lookup_car_metadata(car_model: str, cars: Mapping[str, PackCar]) -> PackCar | None:
    for key in (car_model, car_model.upper(), car_model.lower()):
        car = cars.get(key)
        if car is not None:
            return car
    lowered = car_model.lower()
    for car in cars.values():
        if getattr(car, "abbrev", "").lower() == lowered:
            return car
    return None


def _serialise_pack_payload(payload: Any) -> Any:
    """Convert pack metadata into JSON-serialisable primitives."""

    if is_dataclass(payload):
        return _serialise_pack_payload(asdict(payload))
    if isinstance(payload, Mapping):
        return {str(key): _serialise_pack_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [_serialise_pack_payload(item) for item in payload]
    return payload


def _resolve_tnfr_targets(
    car_model: str,
    cars: Mapping[str, PackCar],
    profiles: Mapping[str, PackProfile],
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, Any] | None:
    """Return TNFR objectives for ``car_model`` when available."""

    car = _lookup_car_metadata(car_model, cars)
    if car is None:
        return None
    try:
        return resolve_pack_targets(car.abbrev, cars, profiles, overrides=overrides)
    except KeyError:
        return None


def _assemble_session_payload(
    car_model: str,
    selection: TrackSelection,
    *,
    cars: Mapping[str, PackCar],
    track_profiles: Mapping[str, Mapping[str, Any]],
    modifiers: Mapping[tuple[str, str], Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    track_profile = selection.track_profile
    if track_profile is None:
        return None
    car_metadata = _lookup_car_metadata(car_model, cars)
    car_profile = getattr(car_metadata, "profile", None) if car_metadata else None
    if not car_profile:
        car_profile = car_model
    if not track_profiles:
        return None
    try:
        combined = assemble_session_weights(
            car_profile,
            track_profile,
            track_profiles=track_profiles,
            modifiers=modifiers if modifiers else None,
        )
    except KeyError:
        return None
    payload: Dict[str, Any] = {
        "car_model": car_model,
        "car_profile": car_profile,
        "track_profile": track_profile,
        "weights": combined.get("weights", MappingProxyType({})),
        "hints": combined.get("hints", MappingProxyType({})),
    }
    if selection.layout:
        payload["layout"] = selection.layout
    if selection.config is not None:
        payload["layout_name"] = selection.config.name
        payload["track_length_km"] = selection.config.length_km
        payload["surface"] = selection.config.surface
    return MappingProxyType(payload)


def _extract_target_objectives(
    targets: Mapping[str, Any] | None,
) -> tuple[float | None, float | None]:
    """Extract ΔNFR/Sense Index targets from pack metadata if present."""

    if not isinstance(targets, Mapping):
        return None, None
    targets_section = targets.get("targets")
    if not isinstance(targets_section, Mapping):
        return None, None
    balance = targets_section.get("balance")
    delta_target: float | None = None
    sense_target: float | None = None
    if isinstance(balance, Mapping):
        delta_value = balance.get("delta_nfr")
        if isinstance(delta_value, (int, float)):
            delta_target = float(delta_value)
        sense_value = balance.get("sense_index")
        if isinstance(sense_value, (int, float)):
            sense_target = float(sense_value)
    return delta_target, sense_target


def _build_setup_plan_payload(
    context: SetupPlanContext,
    namespace: argparse.Namespace,
) -> Mapping[str, Any]:
    plan = context.plan
    metrics = context.metrics
    delta_metric = context.delta_metric
    bundles = context.bundles
    microsectors = context.microsectors
    session_payload = context.session_payload
    engine = context.engine
    thresholds = context.thresholds
    tnfr_targets = context.tnfr_targets
    car_metadata = context.car_metadata

    action_recommendations = [
        rec for rec in plan.recommendations if rec.parameter and rec.delta is not None
    ]

    aggregated_rationales = [rec.rationale for rec in plan.recommendations if rec.rationale]
    aggregated_effects = [rec.message for rec in plan.recommendations if rec.message]

    if action_recommendations:
        ordered_actions = sorted(
            action_recommendations,
            key=lambda rec: (rec.priority, -abs(rec.delta or 0.0)),
        )
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
        default_rationales = aggregated_rationales or ["Optimización de objetivo Si/ΔNFR"]
        default_effects = aggregated_effects or ["Mejora equilibrada del coche"]
        changes = [
            SetupChange(
                parameter=name,
                delta=value,
                rationale="; ".join(default_rationales),
                expected_effect="; ".join(default_effects),
            )
            for name, value in sorted(plan.decision_vector.items())
        ]
        aggregated_rationales = default_rationales
        aggregated_effects = default_effects

    aero = compute_aero_coherence((), plan.telemetry)

    def _rake_velocity_profile_from_bundles(
        bundles: Sequence[Any],
        *,
        low_threshold: float = 35.0,
        high_threshold: float = 50.0,
    ) -> list[tuple[float, int]]:
        bins = {
            "low": {"sum": 0.0, "count": 0},
            "medium": {"sum": 0.0, "count": 0},
            "high": {"sum": 0.0, "count": 0},
        }
        for bundle in bundles:
            transmission = getattr(bundle, "transmission", None)
            try:
                speed_value = float(getattr(transmission, "speed", 0.0))
            except (TypeError, ValueError):
                speed_value = 0.0
            if not math.isfinite(speed_value):
                speed_value = 0.0
            if speed_value <= low_threshold:
                bin_key = "low"
            elif speed_value <= high_threshold:
                bin_key = "medium"
            else:
                bin_key = "high"
            chassis = getattr(bundle, "chassis", None)
            suspension = getattr(bundle, "suspension", None)
            if chassis is None or suspension is None:
                continue
            try:
                pitch_value = float(getattr(chassis, "pitch", 0.0))
            except (TypeError, ValueError):
                pitch_value = 0.0
            if not math.isfinite(pitch_value):
                pitch_value = 0.0
            try:
                front_travel = float(getattr(suspension, "travel_front", 0.0))
            except (TypeError, ValueError):
                front_travel = 0.0
            if not math.isfinite(front_travel):
                front_travel = 0.0
            try:
                rear_travel = float(getattr(suspension, "travel_rear", 0.0))
            except (TypeError, ValueError):
                rear_travel = 0.0
            if not math.isfinite(rear_travel):
                rear_travel = 0.0
            travel_delta = rear_travel - front_travel
            if abs(front_travel) > 1e-9:
                travel_ratio = rear_travel / front_travel
            elif abs(rear_travel) > 1e-9:
                travel_ratio = math.copysign(10.0, rear_travel)
            else:
                travel_ratio = 1.0
            if not math.isfinite(travel_ratio):
                travel_ratio = 1.0
            travel_ratio = max(-10.0, min(10.0, travel_ratio))
            rake_value = pitch_value + math.atan2(travel_delta, travel_ratio)
            if not math.isfinite(rake_value):
                continue
            bin_payload = bins[bin_key]
            bin_payload["sum"] += rake_value
            bin_payload["count"] += 1
        profile: list[tuple[float, int]] = []
        for key in ("low", "medium", "high"):
            payload = bins[key]
            count = int(payload["count"])
            if count > 0:
                average = payload["sum"] / count
            else:
                average = 0.0
            profile.append((average, count))
        return profile

    suspension_deltas = [
        float(getattr(getattr(bundle, "suspension", None), "delta_nfr", 0.0))
        for bundle in plan.telemetry
    ]
    tyre_deltas = [
        float(getattr(getattr(bundle, "tyres", None), "delta_nfr", 0.0))
        for bundle in plan.telemetry
    ]
    coherence_series = [float(getattr(bundle, "coherence_index", 0.0)) for bundle in plan.telemetry]
    avg_coherence = mean(coherence_series) if coherence_series else 0.0
    ackermann_values = [
        float(getattr(bundle, "ackermann_parallel_index", 0.0))
        for bundle in plan.telemetry
    ]
    ackermann_clean = [value for value in ackermann_values if math.isfinite(value)]
    ackermann_parallel = fmean(ackermann_clean) if ackermann_clean else 0.0
    ackermann_samples = len(ackermann_clean)
    rake_velocity_profile = _rake_velocity_profile_from_bundles(plan.telemetry)
    aero_mechanical = resolve_aero_mechanical_coherence(
        avg_coherence,
        aero,
        suspension_deltas=suspension_deltas,
        tyre_deltas=tyre_deltas,
        rake_velocity_profile=rake_velocity_profile,
        ackermann_parallel_index=ackermann_parallel,
        ackermann_parallel_samples=ackermann_samples,
    )
    aero_metrics = {
        "low_speed_imbalance": aero.low_speed_imbalance,
        "high_speed_imbalance": aero.high_speed_imbalance,
        "low_speed_samples": float(aero.low_speed_samples),
        "high_speed_samples": float(aero.high_speed_samples),
        "aero_mechanical_coherence": aero_mechanical,
    }

    setup_plan = SetupPlan(
        car_model=namespace.car_model,
        session=getattr(namespace, "session", None),
        sci=plan.sci,
        changes=tuple(changes),
        rationales=tuple(aggregated_rationales),
        expected_effects=tuple(aggregated_effects),
        sensitivities=plan.sensitivities,
        clamped_parameters=tuple(),
        phase_axis_targets={},
        phase_axis_weights={},
        aero_guidance=aero.guidance,
        aero_metrics=aero_metrics,
        aero_mechanical_coherence=aero_mechanical,
        sci_breakdown=plan.sci_breakdown,
    )

    payload: Dict[str, Any] = {
        "setup_plan": setup_plan,
        "sci": plan.sci,
        "sci_breakdown": plan.sci_breakdown,
        "recommendations": plan.recommendations,
        "series": plan.telemetry,
        "sensitivities": plan.sensitivities,
        "set_output": getattr(namespace, "set_output", None),
    }
    if session_payload is not None:
        payload["session"] = session_payload
    session_messages = format_session_messages(session_payload)
    if session_messages:
        payload["session_messages"] = session_messages
    if car_metadata is not None:
        payload["car"] = _serialise_pack_payload(car_metadata)
    if tnfr_targets is not None:
        payload["tnfr_targets"] = _serialise_pack_payload(tnfr_targets)
    payload["thresholds"] = thresholds
    payload["metrics"] = metrics
    payload["delta_metric"] = delta_metric
    payload["bundles"] = bundles
    payload["microsectors"] = microsectors
    payload["engine"] = engine
    return payload


def _validated_export(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value in exporters_registry:
        return value
    return fallback


def _add_export_argument(parser: argparse.ArgumentParser, *, default: str, help_text: str) -> None:
    parser.add_argument(
        "--export",
        dest="exports",
        choices=sorted(exporters_registry.keys()),
        action="append",
        help=f"{help_text} Puede repetirse para combinar salidas.",
    )
    parser.set_defaults(exports=None, export_default=default)


def load_cli_config(path: Path | None = None) -> Dict[str, Any]:
    """Load CLI defaults from ``tnfr-lfs.toml`` style files."""

    candidates: List[Path] = []
    if path is not None:
        candidates.append(path)
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / DEFAULT_CONFIG_FILENAME)
    candidates.append(Path.home() / ".config" / DEFAULT_CONFIG_FILENAME)

    resolved_candidates: Dict[Path, None] = {}
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved in resolved_candidates:
            continue
        resolved_candidates[resolved] = None
        if not resolved.exists():
            continue
        with resolved.open("rb") as handle:
            data = tomllib.load(handle)
        data["_config_path"] = str(resolved)
        return data
    return {"_config_path": None}


def _resolve_output_dir(config: Mapping[str, Any]) -> Path:
    paths_cfg = config.get("paths")
    if isinstance(paths_cfg, Mapping):
        output_dir = paths_cfg.get("output_dir")
        if isinstance(output_dir, str):
            return Path(output_dir).expanduser()
    return DEFAULT_OUTPUT_DIR


def _resolve_profiles_path(
    config: Mapping[str, Any], *, pack_root: Path | None = None
) -> ProfilesContext:
    """Resolve the profile storage path and pack definitions."""

    env_path = os.environ.get(PROFILES_ENV_VAR)
    if env_path:
        storage = Path(env_path).expanduser()
    else:
        storage = Path(DEFAULT_PROFILES_FILENAME)
        paths_cfg = config.get("paths")
        if isinstance(paths_cfg, Mapping):
            profile_path = paths_cfg.get("profiles")
            if isinstance(profile_path, str):
                storage = Path(profile_path).expanduser()

    pack_profiles = _load_pack_profiles(pack_root)
    return ProfilesContext(storage_path=storage, pack_profiles=pack_profiles)


def _default_car_model(config: Mapping[str, Any]) -> str:
    for section in ("analyze", "suggest", "write_set"):
        section_cfg = config.get(section)
        if isinstance(section_cfg, Mapping):
            candidate = section_cfg.get("car_model")
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return "XFG"


def _default_track_name(config: Mapping[str, Any]) -> str:
    for section in ("analyze", "suggest"):
        section_cfg = config.get(section)
        if isinstance(section_cfg, Mapping):
            candidate = section_cfg.get("track")
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return "generic"


def _resolve_track_argument(
    track_value: str | None,
    config: Mapping[str, Any],
    *,
    pack_root: Path | None,
) -> TrackSelection:
    candidate = str(track_value).strip() if track_value is not None else ""
    if not candidate:
        candidate = _default_track_name(config)
    selection = _resolve_track_selection(candidate, pack_root=pack_root)
    if not selection.name:
        return TrackSelection(name=candidate)
    return selection


def _effective_delta_metric(metrics: Mapping[str, Any]) -> float:
    series = metrics.get("delta_nfr_series")
    if isinstance(series, Sequence):
        values = [abs(float(value)) for value in series if isinstance(value, (int, float))]
        if values:
            return float(fmean(values))
    value = metrics.get("delta_nfr", 0.0)
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0.0


def _format_quiet_sequence(sequence: Sequence[int]) -> str:
    if not sequence:
        return ""
    start = sequence[0] + 1
    end = sequence[-1] + 1
    if start == end:
        return f"Curva {start}"
    return f"Curvas {start}-{end}"


def _quiet_cli_notice(
    microsectors: Sequence[Microsector], sequences: Sequence[Sequence[int]]
) -> str:
    descriptors: List[str] = []
    coverage_values: List[float] = []
    si_values: List[float] = []
    epi_values: List[float] = []
    for sequence in sequences:
        descriptors.append(_format_quiet_sequence(sequence))
        for index in sequence:
            if index < 0 or index >= len(microsectors):
                continue
            coverage, _, si_variance, epi_abs = microsector_stability_metrics(
                microsectors[index]
            )
            coverage_values.append(coverage)
            si_values.append(si_variance)
            epi_values.append(epi_abs)
    message = f"No tocar: {', '.join(descriptors)}"
    if coverage_values:
        coverage_avg = sum(coverage_values) / len(coverage_values)
        si_avg = sum(si_values) / len(si_values) if si_values else 0.0
        epi_avg = sum(epi_values) / len(epi_values) if epi_values else 0.0
        message = (
            f"{message} · silencio μ {coverage_avg * 100.0:.0f}%"
            f" · Siσ μ {si_avg:.4f} · |dEPI| μ {epi_avg:.3f}"
        )
    return message


def _resolve_baseline_destination(
    namespace: argparse.Namespace, config: Mapping[str, Any]
) -> Path:
    fmt = namespace.format
    suffix = ".jsonl" if fmt == "jsonl" else ".parquet"
    output_dir_arg: Path | None = getattr(namespace, "output_dir", None)
    if output_dir_arg is not None:
        output_dir_arg = output_dir_arg.expanduser()
    auto_kwargs = {
        "car_model": _default_car_model(config),
        "track_name": _default_track_name(config),
        "output_dir": output_dir_arg,
        "suffix": suffix,
        "force": namespace.force,
    }

    output_arg: Path | None = namespace.output
    if output_arg is None:
        return logs.prepare_run_destination(**auto_kwargs)

    output_path = output_arg.expanduser()
    if output_path.exists() and output_path.is_dir():
        auto_kwargs["output_dir"] = output_path
        return logs.prepare_run_destination(**auto_kwargs)

    if output_path.suffix == "":
        auto_kwargs["output_dir"] = output_path
        return logs.prepare_run_destination(**auto_kwargs)

    if output_dir_arg is not None and not output_path.is_absolute():
        destination = output_dir_arg.expanduser() / output_path
    else:
        destination = output_path

    if destination.exists() and destination.is_dir():
        auto_kwargs["output_dir"] = destination
        return logs.prepare_run_destination(**auto_kwargs)

    if destination.exists() and not namespace.force:
        raise FileExistsError(
            f"Baseline destination {destination} already exists. Use --force to overwrite."
        )

    return destination


def _parse_lfs_cfg(cfg_path: Path) -> Dict[str, Dict[str, str]]:
    sections: Dict[str, Dict[str, str]] = {"OutSim": {}, "OutGauge": {}, "InSim": {}}
    with cfg_path.open("r", encoding="utf8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            normalized = line.replace("=", " ")
            parts = [token for token in normalized.split(" ") if token]
            if len(parts) < 3:
                continue
            prefix, key, value = parts[0], parts[1], " ".join(parts[2:])
            if prefix not in sections:
                continue
            sections[prefix][key] = value
    return sections


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


_OUTSIM_PING_SIZE = struct.calcsize("<I15f")
_OUTGAUGE_PING_SIZE = struct.calcsize("<I4s16s8s6s6sHBBfffffffIIfff16s16sI")


def _udp_ping(host: str, port: int, timeout: float, *, expected_size: int, label: str) -> Tuple[bool, str]:
    description = f"{label} {host}:{port}"
    payload = bytes(expected_size)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            try:
                sock.sendto(payload, (host, port))
            except OSError as exc:
                return False, f"No se pudo enviar ping a {description}: {exc}"
            try:
                data, addr = sock.recvfrom(expected_size)
            except socket.timeout:
                return False, f"Sin respuesta de {description} tras {timeout:.2f}s"
            except OSError as exc:
                return False, f"Error recibiendo respuesta de {description}: {exc}"
            if len(data) < expected_size:
                return False, f"Respuesta incompleta de {description}: {len(data)} bytes"
            return True, f"{label} respondió desde {addr[0]}:{addr[1]} ({len(data)} bytes)"
    except OSError as exc:
        return False, f"No se pudo crear socket UDP para {description}: {exc}"


def _outsim_ping(host: str, port: int, timeout: float) -> Tuple[bool, str]:
    return _udp_ping(host, port, timeout, expected_size=_OUTSIM_PING_SIZE, label="OutSim")


def _outgauge_ping(host: str, port: int, timeout: float) -> Tuple[bool, str]:
    return _udp_ping(host, port, timeout, expected_size=_OUTGAUGE_PING_SIZE, label="OutGauge")


def _recv_exact(sock: socket.socket, count: int, *, timeout: float, label: str) -> bytes:
    deadline = monotonic() + timeout
    data = bytearray()
    while len(data) < count:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Sin respuesta de {label}")
        sock.settimeout(remaining)
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ConnectionError(f"Conexión cerrada por {label}")
        data.extend(chunk)
    return bytes(data)


def _insim_handshake(host: str, port: int, timeout: float) -> Tuple[bool, str]:
    description = f"InSim {host}:{port}"
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            padded_name = "TNFR Diagnose".encode("utf8").ljust(16, b"\0")
            keepalive_ms = int(max(0.5, timeout) * 1000)
            packet = InSimClient.ISI_STRUCT.pack(
                InSimClient.ISI_STRUCT.size,
                InSimClient.ISP_ISI,
                0,
                0,
                padded_name,
                InSimClient.INSIM_VERSION,
                ord("!"),
                0,
                InSimClient.ISF_LOCAL,
                keepalive_ms,
            )
            sock.sendall(packet)
            header = _recv_exact(sock, 1, timeout=timeout, label=description)
            size = header[0]
            payload = _recv_exact(sock, size - 1, timeout=timeout, label=description)
            data = header + payload
            unpacked = InSimClient.VER_STRUCT.unpack(data)
            if unpacked[1] != InSimClient.ISP_VER:
                return False, f"Respuesta inesperada de {description}"
            if unpacked[4] != InSimClient.INSIM_VERSION:
                return False, f"Versión InSim incompatible en {description}"
            return True, f"InSim respondió con versión {unpacked[4]}"
    except TimeoutError as exc:
        return False, f"{exc}"
    except (OSError, ConnectionError) as exc:
        return False, f"Error en handshake con {description}: {exc}"


def _copy_to_clipboard(commands: Iterable[str]) -> bool:
    commands_list = [command.strip() for command in commands if command.strip()]
    if not commands_list:
        return False
    payload = "\n".join(commands_list)
    try:
        if sys.platform == "darwin" and shutil.which("pbcopy"):
            subprocess.run(["pbcopy"], input=payload.encode("utf8"), check=True)
            return True
        if os.name == "nt":
            completed = subprocess.run(
                "clip",
                input=payload.encode("utf-16le"),
                check=True,
                shell=True,
            )
            return completed.returncode == 0
        if shutil.which("wl-copy"):
            subprocess.run(["wl-copy"], input=payload.encode("utf8"), check=True)
            return True
        if shutil.which("xclip"):
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=payload.encode("utf8"),
                check=True,
            )
            return True
    except (OSError, subprocess.SubprocessError):
        return False
    return False


def _share_disabled_commands(commands: Iterable[str]) -> None:
    command_list = [command for command in commands if command]
    if not command_list:
        return
    print("Comandos recomendados para habilitar la telemetría:")
    for command in command_list:
        print(f"  {command}")
    if _copy_to_clipboard(command_list):
        print("Los comandos se han copiado al portapapeles.")
    else:
        print("Copia manual necesaria: no se pudo acceder al portapapeles.")


def _check_setups_directory(cfg_path: Path) -> Tuple[bool, str]:
    setups_dir = cfg_path.parent.parent / "data" / "setups"
    if not setups_dir.exists():
        return False, f"No se encontró el directorio de setups en {setups_dir}"
    if not setups_dir.is_dir():
        return False, f"La ruta de setups no es un directorio: {setups_dir}"
    try:
        with tempfile.NamedTemporaryFile(dir=setups_dir, delete=False) as handle:
            test_path = Path(handle.name)
            handle.write(b"tnfr")
    except OSError as exc:
        suggestion = (
            "Verifica permisos con `chmod u+w` o ejecuta LFS con privilegios de escritura."
        )
        return False, f"No hay permisos de escritura en {setups_dir}: {exc}. {suggestion}"
    else:
        try:
            test_path.unlink(missing_ok=True)
        except OSError:
            pass
    return True, f"Permisos de escritura confirmados en {setups_dir}"


def _phase_tolerances(
    config: Mapping[str, Any], car_model: str, track_name: str
) -> Dict[str, float]:
    tolerances: Dict[str, float] = {}
    limits_cfg = config.get("limits")
    if isinstance(limits_cfg, Mapping):
        delta_cfg = limits_cfg.get("delta_nfr")
        if isinstance(delta_cfg, Mapping):
            for phase, default_value in (
                ("entry", 1.5),
                ("apex", 1.0),
                ("exit", 2.0),
            ):
                value = delta_cfg.get(phase)
                if value is None:
                    continue
                try:
                    tolerances[phase] = float(value)
                except (TypeError, ValueError):
                    tolerances[phase] = float(default_value)
    if len(tolerances) < 3:
        engine = RecommendationEngine()
        context = engine._resolve_context(car_model, track_name)  # type: ignore[attr-defined]
        profile = context.thresholds
        baseline = {
            "entry": float(profile.entry_delta_tolerance),
            "apex": float(profile.apex_delta_tolerance),
            "exit": float(profile.exit_delta_tolerance),
        }
        baseline.update(tolerances)
        tolerances = baseline
    return tolerances


_PHASE_LABELS = {"entry": "entrada", "apex": "vértice", "exit": "salida"}
_PHASE_RECOMMENDATIONS = {
    "entry": {
        "positive": "liberar presión o adelantar el reparto para estabilizar el eje delantero",
        "negative": "ganar mordida adelantando el apoyo delantero o retrasando el reparto",
    },
    "apex": {
        "positive": "aliviar rigidez lateral (barras/altura) para evitar saturar el vértice",
        "negative": "buscar más rotación aumentando apoyo lateral o ajustando convergencias",
    },
    "exit": {
        "positive": "moderar la entrega de par o endurecer el soporte trasero para contener el sobreviraje",
        "negative": "liberar el eje trasero (altura/damper) para mejorar la tracción a la salida",
    },
}


def _microsector_goal(microsector: Microsector, phase: str):
    for goal in microsector.goals:
        if goal.phase == phase:
            return goal
    return None


def _phase_recommendation(phase: str, deviation: float) -> str:
    recommendations = _PHASE_RECOMMENDATIONS.get(phase, {})
    key = "positive" if deviation > 0 else "negative"
    return recommendations.get(key, "ajustar la puesta a punto para equilibrar ΔNFR")


def _format_window(window: Tuple[float, float]) -> str:
    return f"[{window[0]:.3f}, {window[1]:.3f}]"


def _profile_phase_templates(profile: ThresholdProfile) -> Dict[str, Dict[str, object]]:
    templates: Dict[str, Dict[str, object]] = {}
    for phase, target in profile.phase_targets.items():
        templates[phase] = {
            "target_delta_nfr": round(float(target.target_delta_nfr), 3),
            "slip_lat_window": [
                round(float(target.slip_lat_window[0]), 3),
                round(float(target.slip_lat_window[1]), 3),
            ],
            "slip_long_window": [
                round(float(target.slip_long_window[0]), 3),
                round(float(target.slip_long_window[1]), 3),
            ],
            "yaw_rate_window": [
                round(float(target.yaw_rate_window[0]), 3),
                round(float(target.yaw_rate_window[1]), 3),
            ],
        }
    return templates


def _phase_templates_from_config(
    config: Mapping[str, Any], section: str
) -> Dict[str, Dict[str, float | List[float]]]:
    section_cfg = config.get(section)
    if not isinstance(section_cfg, Mapping):
        return {}
    templates_cfg = section_cfg.get("phase_templates")
    if not isinstance(templates_cfg, Mapping):
        return {}
    templates: Dict[str, Dict[str, float | List[float]]] = {}
    for phase, payload in templates_cfg.items():
        if not isinstance(payload, Mapping):
            continue
        entry: Dict[str, float | List[float]] = {}
        target_delta = payload.get("target_delta_nfr")
        if target_delta is not None:
            try:
                entry["target_delta_nfr"] = float(target_delta)
            except (TypeError, ValueError):
                pass
        for key in ("slip_lat_window", "slip_long_window", "yaw_rate_window"):
            window = payload.get(key)
            if isinstance(window, Sequence) and len(window) == 2:
                try:
                    entry[key] = [float(window[0]), float(window[1])]
                except (TypeError, ValueError):
                    continue
        if entry:
            templates[str(phase)] = entry
    return templates


def _phase_deviation_messages(
    bundles: Bundles,
    microsectors: Sequence[Microsector],
    config: Mapping[str, Any],
    *,
    car_model: str,
    track_name: str,
    session: Mapping[str, Any] | None = None,
) -> List[str]:
    hints: Mapping[str, Any] | None = None
    if isinstance(session, Mapping):
        hints_payload = session.get("hints")
        if isinstance(hints_payload, Mapping):
            hints = hints_payload
    hint_messages: List[str] = []
    if hints:
        slip_bias = hints.get("slip_ratio_bias")
        if isinstance(slip_bias, str) and slip_bias:
            direction = "delantero" if slip_bias.lower() == "front" else "trasero"
            hint_messages.append(
                f"Hint sesión: prioriza aero {direction} (slip_ratio_bias={slip_bias})."
            )
        surface = hints.get("surface")
        if isinstance(surface, str) and surface:
            hint_messages.append(
                f"Hint sesión: superficie {surface} → ajusta amortiguación y alturas."
            )
    if not microsectors or not bundles:
        base_messages = [
            "Sin desviaciones ΔNFR↓ relevantes; no se detectaron curvas segmentadas.",
        ]
        base_messages.extend(hint_messages)
        return base_messages

    tolerances = _phase_tolerances(config, car_model, track_name)
    quiet_sequences = detect_quiet_microsector_streaks(microsectors)
    if quiet_sequences:
        messages = [_quiet_cli_notice(microsectors, quiet_sequences)]
        messages.extend(hint_messages)
        return messages
    messages: List[str] = []
    bundle_count = len(bundles)
    for microsector in microsectors:
        for phase in ("entry", "apex", "exit"):
            goal = _microsector_goal(microsector, phase)
            if goal is None:
                continue
            indices = microsector.phase_indices(phase)
            samples = [
                bundles[idx].delta_nfr
                for idx in indices
                if 0 <= idx < bundle_count
            ]
            if not samples:
                continue
            actual_delta = mean(samples)
            deviation = actual_delta - float(goal.target_delta_nfr)
            tolerance = float(tolerances.get(phase, 0.0))
            if abs(deviation) <= tolerance:
                continue
            label = _PHASE_LABELS.get(phase, phase)
            direction = "exceso" if deviation > 0 else "déficit"
            recommendation = _phase_recommendation(phase, deviation)
            messages.append(
                (
                    f"Curva {microsector.index + 1} ({label}): ΔNFR↓ medio "
                    f"{actual_delta:+.2f} vs objetivo {goal.target_delta_nfr:+.2f} "
                    f"({direction} {abs(deviation):.2f}, tolerancia ±{tolerance:.2f}). "
                    f"Sugerencia: {recommendation}."
                )
            )
    if not messages:
        messages = [
            "Sin desviaciones ΔNFR↓ relevantes por fase; mantener la referencia actual.",
        ]
        messages.extend(hint_messages)
        return messages
    messages.extend(hint_messages)
    return messages


def _sense_index_map(
    bundles: Bundles, microsectors: Sequence[Microsector]
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    bundle_count = len(bundles)
    for microsector in microsectors:
        entry: Dict[str, Any] = {
            "microsector": microsector.index,
            "label": f"Curva {microsector.index + 1}",
            "sense_index": {},
        }
        aggregate: List[float] = []
        for phase in ("entry", "apex", "exit"):
            indices = microsector.phase_indices(phase)
            samples = [
                bundles[idx].sense_index
                for idx in indices
                if 0 <= idx < bundle_count
            ]
            if samples:
                avg_value = mean(samples)
                aggregate.extend(samples)
            else:
                avg_value = 0.0
            entry["sense_index"][phase] = round(float(avg_value), 4)
        entry["sense_index"]["overall"] = round(
            float(mean(aggregate)) if aggregate else 0.0,
            4,
        )
        filtered_payload: Dict[str, Any] = {}
        symmetry_payload: Dict[str, Dict[str, float]] = {}
        samples_payload: Dict[str, Any] = {}

        def _normalise(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): _normalise(sub_value) for key, sub_value in value.items()}
            if isinstance(value, (list, tuple)):
                return [_normalise(item) for item in value]
            if isinstance(value, bool) or value is None:
                return value
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return value
            if not math.isfinite(numeric):
                return None
            return round(numeric, 4)

        def _serialise_cphi_mapping(payload: Mapping[str, Any]) -> Dict[str, Any]:
            return _normalise(payload)

        for key, value in microsector.filtered_measures.items():
            if key == "cphi" and isinstance(value, Mapping):
                filtered_payload["cphi"] = _serialise_cphi_mapping(value)
                continue
            if isinstance(key, str) and key.endswith("_samples"):
                samples_payload[key] = _normalise(value)
                continue
            if isinstance(key, str) and key.startswith("mu_symmetry"):
                parts = key.split("_")
                if len(parts) == 3:
                    phase_label = "window"
                    axle = parts[2]
                elif len(parts) >= 4:
                    phase_label = parts[2]
                    axle = parts[3]
                else:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    continue
                bucket = symmetry_payload.setdefault(phase_label, {})
                bucket[axle] = round(numeric, 4)
                continue
            if isinstance(value, bool) or value is None:
                filtered_payload[key] = value
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                filtered_payload[key] = None
                continue
            filtered_payload[key] = round(numeric, 4)
        if symmetry_payload:
            filtered_payload["mu_symmetry"] = symmetry_payload
        if "grip_rel" not in filtered_payload:
            filtered_payload["grip_rel"] = round(float(microsector.grip_rel), 4)
        entry["filtered_measures"] = filtered_payload
        if samples_payload:
            entry["sample_measures"] = samples_payload
        entry["grip_rel"] = filtered_payload.get("grip_rel", 0.0)
        entry["filtered_style_index"] = filtered_payload.get("style_index")
        occupancy_payload: Dict[str, Dict[str, float]] = {}
        for phase in ("entry", "apex", "exit"):
            phase_occupancy = microsector.window_occupancy.get(phase, {})
            occupancy_payload[phase] = {
                metric: round(float(value), 4)
                for metric, value in phase_occupancy.items()
            }
        entry["window_occupancy"] = occupancy_payload
        mutation = microsector.last_mutation
        if mutation:
            entry["mutation"] = {
                "archetype": str(mutation.get("archetype", "")),
                "mutated": bool(mutation.get("mutated", False)),
                "entropy": round(float(mutation.get("entropy", 0.0)), 4),
                "entropy_delta": round(float(mutation.get("entropy_delta", 0.0)), 4),
                "style_delta": round(float(mutation.get("style_delta", 0.0)), 4),
                "phase": mutation.get("phase"),
            }
        else:
            entry["mutation"] = None
        results.append(entry)
    return results


def _delta_breakdown_summary(bundles: Bundles) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "samples": len(bundles),
        "per_node": {
            node: {"delta_nfr_total": 0.0, "breakdown": {}}
            for node in NU_F_NODE_DEFAULTS
        },
    }
    if not bundles:
        return summary

    for bundle in bundles:
        breakdown = getattr(bundle, "delta_breakdown", {}) or {}
        for node, features in breakdown.items():
            node_entry = summary["per_node"].setdefault(
                node, {"delta_nfr_total": 0.0, "breakdown": {}}
            )
            node_total = sum(features.values())
            node_entry["delta_nfr_total"] += float(node_total)
            feature_map = node_entry["breakdown"]
            for name, value in features.items():
                feature_map[name] = feature_map.get(name, 0.0) + float(value)

    for node, node_entry in summary["per_node"].items():
        node_entry["delta_nfr_total"] = float(node_entry["delta_nfr_total"])
        node_entry["breakdown"] = {
            name: float(value) for name, value in node_entry["breakdown"].items()
        }

    return summary


def _generate_out_reports(
    records: Records,
    bundles: Bundles,
    microsectors: Sequence[Microsector],
    destination: Path,
    *,
    microsector_variability: Sequence[Mapping[str, Any]] | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifact_format: str = "json",
) -> Dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    format_key = artifact_format.lower()
    if format_key not in REPORT_ARTIFACT_FORMATS:
        raise ValueError(
            f"Formato de artefacto desconocido '{artifact_format}'. "
            f"Formatos soportados: {', '.join(REPORT_ARTIFACT_FORMATS)}"
        )
    extension_map = {"json": "json", "markdown": "md", "visual": "viz"}
    artifact_extension = extension_map[format_key]

    bundle_list = list(bundles)
    sense_map = _sense_index_map(bundle_list, microsectors)
    resonance = analyse_modal_resonance(records)
    breakdown = _delta_breakdown_summary(bundles)
    metrics = dict(metrics or {})

    def _collect_average(
        entries: Sequence[Mapping[str, Any]],
        key_map: Mapping[str, str],
    ) -> Dict[str, float | None]:
        samples: Dict[str, List[float]] = {suffix: [] for suffix in WHEEL_SUFFIXES}
        for entry in entries:
            measures = entry.get("filtered_measures", {}) if isinstance(entry, Mapping) else {}
            if not isinstance(measures, Mapping):
                continue
            for suffix, key in key_map.items():
                value = measures.get(key)
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    continue
                samples[suffix].append(numeric)
        return {
            suffix: (sum(values) / len(values) if values else None)
            for suffix, values in samples.items()
        }

    avg_temperature = _collect_average(sense_map, TEMPERATURE_MEAN_KEYS)
    avg_temperature_std = _collect_average(sense_map, TEMPERATURE_STD_KEYS)
    avg_pressure = _collect_average(sense_map, PRESSURE_MEAN_KEYS)
    avg_pressure_std = _collect_average(sense_map, PRESSURE_STD_KEYS)

    def _floatify(value: Any, *, default: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(numeric):
            return float(default)
        return numeric

    def _floatify_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                payload[str(key)] = _floatify_mapping(value)
            else:
                payload[str(key)] = _floatify(value, default=0.0)
        return payload

    def _serialise_average(mapping: Mapping[str, float | None]) -> Dict[str, float | None]:
        return {
            suffix: (float(value) if value is not None else None)
            for suffix, value in mapping.items()
        }

    def _merge_sample_indices(sequences: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
        merged: List[int] = []
        for sequence in sequences:
            for value in sequence:
                index = int(value)
                if not merged or merged[-1] != index:
                    merged.append(index)
        return tuple(merged)

    def _serialise_phase_samples_map(
        mapping: Mapping[str, Iterable[int]] | Mapping[str, Tuple[int, ...]]
    ) -> Dict[str, List[int]]:
        normalised = {
            str(phase): tuple(int(index) for index in indices)
            for phase, indices in (mapping or {}).items()
        }
        normalised = replicate_phase_aliases(
            normalised,
            combine=_merge_sample_indices,
        )
        return {
            phase: [int(index) for index in indices]
            for phase, indices in normalised.items()
        }

    def _serialise_phase_axis_map(
        mapping: Mapping[str, Mapping[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        normalised = {
            str(phase): {
                str(axis): _floatify(value)
                for axis, value in (payload or {}).items()
            }
            for phase, payload in (mapping or {}).items()
            if isinstance(payload, Mapping)
        }
        return replicate_phase_aliases(normalised)

    def _serialise_phase_float_map(mapping: Mapping[str, Any]) -> Dict[str, float]:
        normalised = {
            str(phase): _floatify(value)
            for phase, value in (mapping or {}).items()
        }
        return replicate_phase_aliases(normalised)

    sense_path = destination / "sense_index_map.json"
    resonance_path = destination / "modal_resonance.json"
    breakdown_path = destination / "delta_breakdown.json"
    occupancy_path = destination / "window_occupancy.json"
    pairwise_path = destination / "pairwise_coupling.json"
    dissonance_path = destination / "dissonance_breakdown.json"
    memory_path = destination / "sense_memory.json"
    summary_path = destination / "metrics_summary.md"
    phase_samples_path = destination / "phase_samples.json"
    phase_axis_targets_path = destination / "phase_axis_targets.json"
    phase_axis_weights_path = destination / "phase_axis_weights.json"
    phase_metrics_path = destination / "phase_metrics.json"

    occupancy_payload: List[Dict[str, Any]] = []
    for entry in sense_map:
        occupancy_payload.append(
            {
                "microsector": entry.get("microsector"),
                "label": entry.get("label"),
                "window_occupancy": {
                    phase: {
                        metric: _floatify(value, default=0.0)
                        for metric, value in (entry.get("window_occupancy", {}).get(phase, {}) or {}).items()
                    }
                    for phase in ("entry", "apex", "exit")
                },
            }
        )

    phase_samples_payload: List[Dict[str, Any]] = []
    phase_axis_targets_payload: List[Dict[str, Any]] = []
    phase_axis_weights_payload: List[Dict[str, Any]] = []
    phase_metrics_payload: List[Dict[str, Any]] = []
    for microsector in microsectors:
        label = f"Curva {microsector.index + 1}"
        samples_map = _serialise_phase_samples_map(
            getattr(microsector, "phase_samples", {}) or {}
        )
        axis_target_map = _serialise_phase_axis_map(
            getattr(microsector, "phase_axis_targets", {}) or {}
        )
        axis_weight_map = _serialise_phase_axis_map(
            getattr(microsector, "phase_axis_weights", {}) or {}
        )
        delta_map = _serialise_phase_float_map(
            getattr(microsector, "phase_delta_nfr_std", {}) or {}
        )
        nodal_map = _serialise_phase_float_map(
            getattr(microsector, "phase_nodal_delta_nfr_std", {}) or {}
        )
        phase_samples_payload.append(
            {
                "microsector": microsector.index,
                "label": label,
                "phase_samples": samples_map,
            }
        )
        phase_axis_targets_payload.append(
            {
                "microsector": microsector.index,
                "label": label,
                "phase_axis_targets": axis_target_map,
            }
        )
        phase_axis_weights_payload.append(
            {
                "microsector": microsector.index,
                "label": label,
                "phase_axis_weights": axis_weight_map,
            }
        )
        phase_metrics_payload.append(
            {
                "microsector": microsector.index,
                "label": label,
                "phase_delta_nfr_std": delta_map,
                "phase_nodal_delta_nfr_std": nodal_map,
                "delta_nfr_entropy": _floatify(
                    getattr(microsector, "delta_nfr_entropy", 0.0)
                ),
                "node_entropy": _floatify(
                    getattr(microsector, "node_entropy", 0.0)
                ),
                "phase_delta_nfr_entropy": _serialise_phase_float_map(
                    getattr(microsector, "phase_delta_nfr_entropy", {}) or {}
                ),
                "phase_node_entropy": _serialise_phase_float_map(
                    getattr(microsector, "phase_node_entropy", {}) or {}
                ),
            }
        )

    with sense_path.open("w", encoding="utf8") as handle:
        json.dump(sense_map, handle, indent=2, sort_keys=True)
    resonance_payload = {
        axis: {
            "sample_rate": analysis.sample_rate,
            "total_energy": analysis.total_energy,
            "peaks": [
                {
                    "frequency": peak.frequency,
                    "energy": peak.energy,
                    "classification": peak.classification,
                }
                for peak in analysis.peaks
            ],
        }
        for axis, analysis in resonance.items()
    }
    with resonance_path.open("w", encoding="utf8") as handle:
        json.dump(resonance_payload, handle, indent=2, sort_keys=True)
    with breakdown_path.open("w", encoding="utf8") as handle:
        json.dump(breakdown, handle, indent=2, sort_keys=True)
    with occupancy_path.open("w", encoding="utf8") as handle:
        json.dump(occupancy_payload, handle, indent=2, sort_keys=True)
    with phase_samples_path.open("w", encoding="utf8") as handle:
        json.dump(phase_samples_payload, handle, indent=2, sort_keys=True)
    with phase_axis_targets_path.open("w", encoding="utf8") as handle:
        json.dump(phase_axis_targets_payload, handle, indent=2, sort_keys=True)
    with phase_axis_weights_path.open("w", encoding="utf8") as handle:
        json.dump(phase_axis_weights_payload, handle, indent=2, sort_keys=True)
    with phase_metrics_path.open("w", encoding="utf8") as handle:
        json.dump(phase_metrics_payload, handle, indent=2, sort_keys=True)

    pairwise_payload: Dict[str, Any] = {}
    raw_pairwise = metrics.get("pairwise_coupling")
    if isinstance(raw_pairwise, Mapping):
        pairwise_payload = _floatify_mapping(raw_pairwise)
    coupling_payload = {
        "global": {
            "delta_nfr_vs_sense_index": _floatify(metrics.get("coupling")),
            "resonance_index": _floatify(metrics.get("resonance")),
            "dissonance": _floatify(metrics.get("dissonance")),
        },
        "pairwise": pairwise_payload,
    }
    with pairwise_path.open("w", encoding="utf8") as handle:
        json.dump(coupling_payload, handle, indent=2, sort_keys=True)

    dissonance_breakdown = metrics.get("dissonance_breakdown")
    if dissonance_breakdown is None:
        dissonance_payload: Dict[str, Any] = {}
    else:
        try:
            dissonance_payload = {
                key: _floatify(value)
                for key, value in asdict(dissonance_breakdown).items()
            }
        except TypeError:
            if isinstance(dissonance_breakdown, Mapping):
                dissonance_payload = {
                    key: _floatify(value) for key, value in dissonance_breakdown.items()
                }
            else:
                dissonance_payload = {"value": _floatify(dissonance_breakdown)}
    with dissonance_path.open("w", encoding="utf8") as handle:
        json.dump(dissonance_payload, handle, indent=2, sort_keys=True)

    sense_memory = metrics.get("sense_memory")
    if isinstance(sense_memory, Mapping):
        memory_payload = {
            "series": [_floatify(value) for value in sense_memory.get("series", []) if isinstance(value, (int, float))],
            "memory": [_floatify(value) for value in sense_memory.get("memory", []) if isinstance(value, (int, float))],
            "average": _floatify(sense_memory.get("average")),
            "decay": _floatify(sense_memory.get("decay")),
        }
    else:
        memory_payload = {"series": [], "memory": [], "average": 0.0, "decay": 0.0}
    with memory_path.open("w", encoding="utf8") as handle:
        json.dump(memory_payload, handle, indent=2, sort_keys=True)

    summary_lines: List[str] = [
        "# Resumen de métricas avanzadas",
        "",
        "## Resonancia modal",
    ]
    for axis, analysis in sorted(resonance_payload.items()):
        summary_lines.append(
            f"- **{axis}** · energía total {analysis['total_energy']:.3f} (Fs={analysis['sample_rate']:.1f} Hz)"
        )
        if analysis["peaks"]:
            for peak in analysis["peaks"]:
                summary_lines.append(
                    "  - "
                    f"{peak['classification']} @ {peak['frequency']:.3f} Hz "
                    f"({peak['energy']:.3f})"
                )
        else:
            summary_lines.append("  - Sin picos detectados")

    if dissonance_payload:
        summary_lines.extend(
            [
                "",
                "## Disonancia útil",
                f"- Magnitud útil: {dissonance_payload.get('useful_magnitude', 0.0):.3f}",
                f"- Eventos útiles: {int(dissonance_payload.get('useful_events', 0))}",
                f"- Magnitud parasitaria: {dissonance_payload.get('parasitic_magnitude', 0.0):.3f}",
            ]
        )

    summary_lines.extend(
        [
            "",
            "## Acoplamientos",
            f"- Acoplamiento global ΔNFR↔Si: {coupling_payload['global']['delta_nfr_vs_sense_index']:.3f}",
            f"- Índice de resonancia global: {coupling_payload['global']['resonance_index']:.3f}",
        ]
    )
    thermal_summary: List[str] = []
    temp_segments: List[str] = []
    temperature_samples_present = any(
        value is not None for value in avg_temperature.values()
    )
    for suffix in WHEEL_SUFFIXES:
        mean_value = avg_temperature.get(suffix)
        std_value = avg_temperature_std.get(suffix)
        if mean_value is None or std_value is None:
            continue
        label = WHEEL_LABELS.get(suffix, suffix.upper())
        temp_segments.append(f"{label} {mean_value:.1f}±{std_value:.1f}")
    if temp_segments:
        thermal_summary.append(
            f"- Temperatura (°C): {' · '.join(temp_segments)}"
        )
    elif not temperature_samples_present:
        thermal_summary.append(
            "- Temperatura (°C): sin datos (OutGauge omitió el bloque de neumáticos)"
        )
    pressure_segments: List[str] = []
    pressure_samples_present = any(
        value is not None for value in avg_pressure.values()
    )
    for suffix in WHEEL_SUFFIXES:
        mean_value = avg_pressure.get(suffix)
        std_value = avg_pressure_std.get(suffix)
        if mean_value is None or std_value is None:
            continue
        label = WHEEL_LABELS.get(suffix, suffix.upper())
        pressure_segments.append(f"{label} {mean_value:.2f}±{std_value:.3f}")
    if pressure_segments:
        thermal_summary.append(
            f"- Presión (bar): {' · '.join(pressure_segments)}"
        )
    elif not pressure_samples_present:
        thermal_summary.append(
            "- Presión (bar): sin datos (OutGauge omitió el bloque de neumáticos)"
        )
    if thermal_summary:
        summary_lines.extend(["", "## Dispersión térmica de neumáticos", *thermal_summary])
    brake_summary: List[str] = []
    if sense_map:
        ventilation_values: List[float] = []
        fade_values: List[float] = []
        peak_values: List[float] = []
        missing_temperature = False
        for entry in sense_map:
            measures = entry.get("filtered_measures", {}) if isinstance(entry, Mapping) else {}
            if not isinstance(measures, Mapping):
                continue
            available_flag = measures.get("brake_headroom_temperature_available")
            if available_flag is False:
                missing_temperature = True
            index_value = measures.get("brake_headroom_ventilation_index")
            if isinstance(index_value, (int, float)) and math.isfinite(index_value):
                ventilation_values.append(float(index_value))
            fade_value = measures.get("brake_headroom_fade_ratio")
            if isinstance(fade_value, (int, float)) and math.isfinite(fade_value):
                fade_values.append(float(fade_value))
            peak_value = measures.get("brake_headroom_temperature_peak")
            if isinstance(peak_value, (int, float)) and math.isfinite(peak_value):
                peak_values.append(float(peak_value))
        if ventilation_values:
            avg_vent = mean(ventilation_values)
            brake_summary.append(
                f"- Ventilación frenos: índice medio {avg_vent:.3f} (pico {max(ventilation_values):.3f})"
            )
        elif missing_temperature:
            brake_summary.append(
                "- Ventilación frenos: sin datos (requiere T° de OutGauge)"
            )
        if peak_values:
            brake_summary.append(
                f"- Temperatura máx. freno: {max(peak_values):.0f}°C"
            )
        if ventilation_values and fade_values:
            brake_summary.append(
                f"- Fade detectado: caída máxima {max(fade_values) * 100:.0f}%"
            )
    if brake_summary:
        summary_lines.extend(["", "## Frenos", *brake_summary])
    if pairwise_payload:
        summary_lines.append("- Pares analizados:")
        for domain, pairs in sorted(pairwise_payload.items()):
            summary_lines.append(f"  - {domain}:")
            for pair, value in sorted(pairs.items()):
                if isinstance(pair, str) and pair.endswith("_samples"):
                    continue
                summary_lines.append(f"    - {pair}: {value:.3f}")

    if microsector_variability:
        stability_lines: List[str] = []
        for entry in microsector_variability:
            label = str(entry.get("label", entry.get("microsector", "")))
            overall = entry.get("overall", {}) if isinstance(entry, Mapping) else {}
            if not isinstance(overall, Mapping):
                continue
            sense_stats = overall.get("sense_index", {})
            if isinstance(sense_stats, Mapping):
                stability = sense_stats.get("stability_score")
                if isinstance(stability, (int, float)) and math.isfinite(stability):
                    stability_lines.append(
                        f"- {label}: estabilidad SI {float(stability):.2f}"
                    )
        if stability_lines:
            summary_lines.extend(["", "## Estabilidad microsectores", *stability_lines])

    if memory_payload["memory"] or memory_payload["series"]:
        summary_lines.extend(
            [
                "",
                "## Memoria del índice de sensibilidad",
                f"- Promedio suavizado: {memory_payload['average']:.3f}",
                f"- Último valor de memoria: {memory_payload['memory'][-1]:.3f}",
                f"- Factor de decaimiento: {memory_payload['decay']:.3f}",
            ]
        )
    else:
        summary_lines.extend(
            [
                "",
                "## Memoria del índice de sensibilidad",
                "- No se registraron muestras para la traza de memoria.",
            ]
        )

    if occupancy_payload:
        summary_lines.extend(["", "## Ocupación de ventanas"])
        phase_totals: Dict[str, List[float]] = {"entry": [], "apex": [], "exit": []}
        for entry in occupancy_payload:
            for phase, values in entry.get("window_occupancy", {}).items():
                total = sum(_floatify(value) for value in values.values())
                phase_totals.setdefault(phase, []).append(total)
        for phase, values in sorted(phase_totals.items()):
            if not values:
                continue
            summary_lines.append(
                f"- {phase}: promedio {sum(values) / len(values):.3f} (ver window_occupancy.json para detalle)"
            )

    summary_text = "\n".join(summary_lines) + "\n"
    with summary_path.open("w", encoding="utf8") as handle:
        handle.write(summary_text)

    thermal_payload = {
        "temperature": (
            {
                "mean": _serialise_average(avg_temperature),
                "std": _serialise_average(avg_temperature_std),
            }
            if temperature_samples_present
            else None
        ),
        "pressure": (
            {
                "mean": _serialise_average(avg_pressure),
                "std": _serialise_average(avg_pressure_std),
            }
            if pressure_samples_present
            else None
        ),
    }
    thermal_path = destination / "tyre_thermal.json"
    with thermal_path.open("w", encoding="utf8") as handle:
        json.dump(thermal_payload, handle, indent=2, sort_keys=True)

    variability_data = [dict(entry) for entry in microsector_variability or ()]
    variability_path = destination / "microsector_variability.json"
    with variability_path.open("w", encoding="utf8") as handle:
        json.dump(variability_data, handle, indent=2, sort_keys=True)

    def _persist_artifact(path: Path, text: str) -> None:
        output = text if text.endswith("\n") else f"{text}\n"
        path.write_text(output, encoding="utf8")

    artifact_context = {
        "microsectors": microsectors,
        "series": bundle_list,
    }
    coherence_payload = build_coherence_map_payload(artifact_context)
    coherence_render = render_coherence_map(coherence_payload, fmt=format_key)
    coherence_path = destination / f"coherence_map.{artifact_extension}"
    _persist_artifact(coherence_path, coherence_render)

    operator_payload = build_operator_trajectories_payload(artifact_context)
    operator_render = render_operator_trajectories(operator_payload, fmt=format_key)
    operator_path = destination / f"operator_trajectories.{artifact_extension}"
    _persist_artifact(operator_path, operator_render)

    bifurcation_payload = build_delta_bifurcation_payload(artifact_context)
    bifurcation_render = render_delta_bifurcation(bifurcation_payload, fmt=format_key)
    bifurcation_path = destination / f"delta_bifurcations.{artifact_extension}"
    _persist_artifact(bifurcation_path, bifurcation_render)

    return {
        "sense_index_map": {"path": str(sense_path), "data": sense_map},
        "modal_resonance": {"path": str(resonance_path), "data": resonance_payload},
        "delta_breakdown": {"path": str(breakdown_path), "data": breakdown},
        "window_occupancy": {"path": str(occupancy_path), "data": occupancy_payload},
        "phase_samples": {"path": str(phase_samples_path), "data": phase_samples_payload},
        "phase_axis_targets": {
            "path": str(phase_axis_targets_path),
            "data": phase_axis_targets_payload,
        },
        "phase_axis_weights": {
            "path": str(phase_axis_weights_path),
            "data": phase_axis_weights_payload,
        },
        "phase_metrics": {"path": str(phase_metrics_path), "data": phase_metrics_payload},
        "tyre_thermal": {"path": str(thermal_path), "data": thermal_payload},
        "microsector_variability": {
            "path": str(variability_path),
            "data": variability_data,
        },
        "pairwise_coupling": {"path": str(pairwise_path), "data": coupling_payload},
        "dissonance_breakdown": {"path": str(dissonance_path), "data": dissonance_payload},
        "sense_memory": {"path": str(memory_path), "data": memory_payload},
        "metrics_summary": {"path": str(summary_path), "data": summary_text},
        "coherence_map": {
            "path": str(coherence_path),
            "data": coherence_payload,
            **({"rendered": coherence_render} if format_key != "json" else {}),
        },
        "operator_trajectories": {
            "path": str(operator_path),
            "data": operator_payload,
            **({"rendered": operator_render} if format_key != "json" else {}),
        },
        "delta_bifurcations": {
            "path": str(bifurcation_path),
            "data": bifurcation_payload,
            **({"rendered": bifurcation_render} if format_key != "json" else {}),
        },
    }

_TELEMETRY_DEFAULTS: Mapping[str, Any] = {
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "brake_pressure": 0.0,
    "locking": 0.0,
    "speed": 0.0,
    "yaw_rate": 0.0,
    "slip_angle": 0.0,
    "steer": 0.0,
    "throttle": 0.0,
    "gear": 0,
    "vertical_load_front": 0.0,
    "vertical_load_rear": 0.0,
    "mu_eff_front": 0.0,
    "mu_eff_rear": 0.0,
    "mu_eff_front_lateral": 0.0,
    "mu_eff_front_longitudinal": 0.0,
    "mu_eff_rear_lateral": 0.0,
    "mu_eff_rear_longitudinal": 0.0,
    "suspension_travel_front": 0.0,
    "suspension_travel_rear": 0.0,
    "suspension_velocity_front": 0.0,
    "suspension_velocity_rear": 0.0,
    "lap": None,
}


def _coerce_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data: Dict[str, Any] = dict(_TELEMETRY_DEFAULTS)
    data.update(payload)
    return data


def build_parser(config: Mapping[str, Any] | None = None) -> argparse.ArgumentParser:
    config = dict(config or {})
    parser = argparse.ArgumentParser(description="TNFR Load & Force Synthesis")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="Ruta del fichero de configuración TOML a utilizar.",
    )
    parser.add_argument(
        "--pack-root",
        dest="pack_root",
        type=Path,
        default=None,
        help=(
            "Directorio raíz de un pack TNFR × LFS con config/ y data/. "
            "Sobrescribe paths.pack_root."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser(
        "template",
        help="Generate ΔNFR, slip and yaw presets for analyze/report workflows.",
    )
    template_parser.add_argument(
        "--car",
        dest="car_model",
        default=_default_car_model(config),
        help="Car model used to resolve the preset (default: perfil actual).",
    )
    template_parser.add_argument(
        "--track",
        dest="track",
        default=_default_track_name(config),
        help="Identificador de pista usado para seleccionar el perfil (default: actual).",
    )
    template_parser.set_defaults(handler=_handle_template)

    telemetry_cfg = dict(config.get("telemetry", {}))
    osd_cfg = dict(config.get("osd", {}))
    osd_parser = subparsers.add_parser(
        "osd",
        help="Render the live ΔNFR HUD inside Live for Speed via InSim buttons.",
    )
    osd_parser.add_argument(
        "--host",
        default=str(osd_cfg.get("host", telemetry_cfg.get("host", "127.0.0.1"))),
        help="Host where the OutSim/OutGauge broadcasters are running.",
    )
    osd_parser.add_argument(
        "--outsim-port",
        type=int,
        default=int(osd_cfg.get("outsim_port", telemetry_cfg.get("outsim_port", 4123))),
        help="Port used by the OutSim UDP stream.",
    )
    osd_parser.add_argument(
        "--outgauge-port",
        type=int,
        default=int(osd_cfg.get("outgauge_port", telemetry_cfg.get("outgauge_port", 3000))),
        help="Port used by the OutGauge UDP stream.",
    )
    osd_parser.add_argument(
        "--insim-port",
        type=int,
        default=int(osd_cfg.get("insim_port", telemetry_cfg.get("insim_port", 29999))),
        help="Port used by the InSim TCP control channel.",
    )
    osd_parser.add_argument(
        "--insim-keepalive",
        type=float,
        default=float(osd_cfg.get("insim_keepalive", 5.0)),
        help="Interval in seconds between InSim keepalive packets (default: 5s).",
    )
    osd_parser.add_argument(
        "--update-rate",
        type=float,
        default=float(osd_cfg.get("update_rate", 6.0)),
        help="HUD refresh rate in Hz (default: 6).",
    )
    osd_parser.add_argument(
        "--car-model",
        default=str(osd_cfg.get("car_model", _default_car_model(config))),
        help="Car model used to resolve recommendation thresholds.",
    )
    osd_parser.add_argument(
        "--track",
        default=str(osd_cfg.get("track", _default_track_name(config))),
        help="Track identifier used to resolve recommendation thresholds.",
    )
    osd_parser.add_argument(
        "--layout-left",
        type=int,
        default=osd_cfg.get("layout_left"),
        help="Override the IS_BTN left coordinate (0-200).",
    )
    osd_parser.add_argument(
        "--layout-top",
        type=int,
        default=osd_cfg.get("layout_top"),
        help="Override the IS_BTN top coordinate (0-200).",
    )
    osd_parser.add_argument(
        "--layout-width",
        type=int,
        default=osd_cfg.get("layout_width"),
        help="Override the IS_BTN width (0-200).",
    )
    osd_parser.add_argument(
        "--layout-height",
        type=int,
        default=osd_cfg.get("layout_height"),
        help="Override the IS_BTN height (0-200).",
    )
    osd_parser.set_defaults(handler=_handle_osd)

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Validate cfg.txt telemetry configuration and UDP availability.",
    )
    diagnose_parser.add_argument(
        "cfg",
        type=Path,
        help="Ruta al fichero cfg.txt de Live for Speed.",
    )
    diagnose_parser.add_argument(
        "--timeout",
        type=float,
        default=0.25,
        help="Tiempo máximo (s) para probar cada socket UDP.",
    )
    diagnose_parser.set_defaults(handler=_handle_diagnose)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Capture telemetry from UDP clients or simulation data and persist it.",
    )
    baseline_parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Destination file for the baseline. When omitted or pointing to a directory "
            "a timestamped run is created under runs/."
        ),
    )
    baseline_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory used when auto-generating baseline runs (default: runs/)."
        ),
    )
    baseline_parser.add_argument(
        "--format",
        choices=("jsonl", "parquet"),
        default=str(telemetry_cfg.get("format", "jsonl")),
        help="Persistence format used for the baseline (default: jsonl).",
    )
    baseline_parser.add_argument(
        "--duration",
        type=float,
        default=float(telemetry_cfg.get("duration", 30.0)),
        help="Capture duration in seconds when using live UDP acquisition.",
    )
    baseline_parser.add_argument(
        "--max-samples",
        type=int,
        default=int(telemetry_cfg.get("max_samples", 10_000)),
        help="Maximum number of samples to collect from UDP clients.",
    )
    baseline_parser.add_argument(
        "--host",
        default=str(telemetry_cfg.get("host", "127.0.0.1")),
        help="Host where the OutSim/OutGauge broadcasters are running.",
    )
    baseline_parser.add_argument(
        "--outsim-port",
        type=int,
        default=int(telemetry_cfg.get("outsim_port", 4123)),
        help="Port used by the OutSim UDP stream.",
    )
    baseline_parser.add_argument(
        "--outgauge-port",
        type=int,
        default=int(telemetry_cfg.get("outgauge_port", 3000)),
        help="Port used by the OutGauge UDP stream.",
    )
    baseline_parser.add_argument(
        "--insim-port",
        type=int,
        default=int(telemetry_cfg.get("insim_port", 29999)),
        help="Port used by the InSim TCP control channel.",
    )
    baseline_parser.add_argument(
        "--timeout",
        type=float,
        default=float(telemetry_cfg.get("timeout", DEFAULT_TIMEOUT)),
        help="Polling timeout for the UDP clients.",
    )
    baseline_parser.add_argument(
        "--retries",
        type=int,
        default=int(telemetry_cfg.get("retries", DEFAULT_RETRIES)),
        help="Number of retries performed by the UDP clients while polling.",
    )
    baseline_parser.add_argument(
        "--insim-keepalive",
        type=float,
        default=float(telemetry_cfg.get("insim_keepalive", 5.0)),
        help="Interval in seconds between InSim keepalive packets.",
    )
    baseline_parser.add_argument(
        "--simulate",
        type=Path,
        help="Telemetry CSV file used in simulation mode (bypasses UDP capture).",
    )
    baseline_parser.add_argument(
        "--limit",
        type=int,
        default=(
            None
            if telemetry_cfg.get("limit") is None
            else int(telemetry_cfg.get("limit"))
        ),
        help="Optional limit of samples to persist when using simulation data.",
    )
    baseline_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    baseline_parser.add_argument(
        "--overlay",
        action="store_true",
        help="Display a Live for Speed overlay while capturing baselines.",
    )
    baseline_parser.set_defaults(handler=_handle_baseline)

    analyze_cfg = dict(config.get("analyze", {}))
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyse a telemetry baseline and export ΔNFR/Si insights."
    )
    analyze_parser.add_argument("telemetry", type=Path, help="Path to a baseline file or CSV.")
    _add_export_argument(
        analyze_parser,
        default=_validated_export(analyze_cfg.get("export"), fallback="json"),
        help_text="Exporter used to render the analysis results.",
    )
    analyze_parser.add_argument(
        "--target-delta",
        type=float,
        default=float(analyze_cfg.get("target_delta", 0.0)),
        help="Target ΔNFR objective used by the operators orchestration.",
    )
    analyze_parser.add_argument(
        "--target-si",
        type=float,
        default=float(analyze_cfg.get("target_si", 0.75)),
        help="Target sense index objective used by the operators orchestration.",
    )
    analyze_parser.add_argument(
        "--coherence-window",
        type=int,
        default=int(analyze_cfg.get("coherence_window", 3)),
        help="Window length used by the coherence operator when smoothing ΔNFR.",
    )
    analyze_parser.add_argument(
        "--recursion-decay",
        type=float,
        default=float(analyze_cfg.get("recursion_decay", 0.4)),
        help="Decay factor for the recursive operator when computing hysteresis.",
    )
    analyze_parser.set_defaults(handler=_handle_analyze)

    suggest_cfg = dict(config.get("suggest", {}))
    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Generate recommendations for a telemetry baseline using the rule engine.",
    )
    suggest_parser.add_argument("telemetry", type=Path, help="Path to a baseline file or CSV.")
    _add_export_argument(
        suggest_parser,
        default=_validated_export(suggest_cfg.get("export"), fallback="json"),
        help_text="Exporter used to render the recommendation payload.",
    )
    suggest_parser.add_argument(
        "--car-model",
        default=str(suggest_cfg.get("car_model", "generic")),
        help="Car model used to resolve the recommendation threshold profile.",
    )
    suggest_parser.add_argument(
        "--track",
        default=str(suggest_cfg.get("track", "generic")),
        help="Track identifier used to resolve the recommendation threshold profile.",
    )
    suggest_parser.set_defaults(handler=_handle_suggest)

    report_cfg = dict(config.get("report", {}))
    report_parser = subparsers.add_parser(
        "report",
        help="Generate ΔNFR and sense index reports linked to the exporter registry.",
    )
    report_parser.add_argument("telemetry", type=Path, help="Path to a baseline file or CSV.")
    _add_export_argument(
        report_parser,
        default=_validated_export(report_cfg.get("export"), fallback="json"),
        help_text="Exporter used to render the report payload.",
    )
    report_parser.add_argument(
        "--target-delta",
        type=float,
        default=float(report_cfg.get("target_delta", 0.0)),
        help="Target ΔNFR objective used by the operators orchestration.",
    )
    report_parser.add_argument(
        "--target-si",
        type=float,
        default=float(report_cfg.get("target_si", 0.75)),
        help="Target sense index objective used by the operators orchestration.",
    )
    report_parser.add_argument(
        "--coherence-window",
        type=int,
        default=int(report_cfg.get("coherence_window", 3)),
        help="Window length used by the coherence operator when smoothing ΔNFR.",
    )
    report_parser.add_argument(
        "--recursion-decay",
        type=float,
        default=float(report_cfg.get("recursion_decay", 0.4)),
        help="Decay factor for the recursive operator when computing hysteresis.",
    )
    report_parser.add_argument(
        "--report-format",
        choices=REPORT_ARTIFACT_FORMATS,
        default=str(report_cfg.get("artifact_format", "json")),
        help=(
            "Formato de los artefactos adicionales generados (json, markdown o visual)."
        ),
    )
    report_parser.set_defaults(handler=_handle_report)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compara dos stints de telemetría agregando métricas por vuelta.",
    )
    compare_parser.add_argument(
        "telemetry_a",
        type=Path,
        help="Ruta a la telemetría baseline o configuración A.",
    )
    compare_parser.add_argument(
        "telemetry_b",
        type=Path,
        help="Ruta a la telemetría variante o configuración B.",
    )
    compare_parser.add_argument(
        "--metric",
        choices=SUPPORTED_AB_METRICS,
        default="sense_index",
        help="Métrica por vuelta utilizada para la comparación (default: sense_index).",
    )
    _add_export_argument(
        compare_parser,
        default="markdown",
        help_text=(
            "Exporter usado para renderizar la comparación A/B (default: markdown)."
        ),
    )
    compare_parser.set_defaults(handler=_handle_compare)

    write_set_cfg = dict(config.get("write_set", {}))
    write_set_parser = subparsers.add_parser(
        "write-set",
        help="Create a setup plan by combining optimisation with recommendations.",
    )
    write_set_parser.add_argument(
        "telemetry", type=Path, help="Path to a baseline file or CSV containing telemetry."
    )
    _add_export_argument(
        write_set_parser,
        default=_validated_export(write_set_cfg.get("export"), fallback="markdown"),
        help_text="Exporter used to render the setup plan (default: markdown).",
    )
    write_set_parser.add_argument(
        "--car-model",
        default=str(write_set_cfg.get("car_model", _default_car_model(config))),
        help="Car model used to select the decision space for optimisation.",
    )
    write_set_parser.add_argument(
        "--session",
        default=None,
        help="Optional session label attached to the generated setup plan.",
    )
    write_set_parser.add_argument(
        "--set-output",
        default=None,
        help=(
            "Nombre base del archivo .set que se guardará bajo LFS/data/setups. "
            "Debe comenzar por el prefijo del coche."
        ),
    )
    write_set_parser.set_defaults(handler=_handle_write_set)

    pareto_parser = subparsers.add_parser(
        "pareto",
        help="Evalúa candidatos de setup y exporta el frente de Pareto resultante.",
    )
    pareto_parser.add_argument(
        "telemetry", type=Path, help="Ruta al archivo o CSV con la telemetría base."
    )
    _add_export_argument(
        pareto_parser,
        default=_validated_export(write_set_cfg.get("export"), fallback="markdown"),
        help_text="Exporter usado para representar el frente Pareto (default: markdown).",
    )
    pareto_parser.add_argument(
        "--car-model",
        default=str(write_set_cfg.get("car_model", _default_car_model(config))),
        help="Modelo de coche utilizado para resolver el espacio de decisiones.",
    )
    pareto_parser.add_argument(
        "--session",
        default=None,
        help="Etiqueta opcional de sesión que acompañará a los resultados.",
    )
    pareto_parser.add_argument(
        "--radius",
        type=int,
        default=int(write_set_cfg.get("pareto_radius", 1)),
        help=(
            "Número de pasos +/- por parámetro en el barrido de candidatos (default: 1)."
        ),
    )
    pareto_parser.set_defaults(handler=_handle_pareto)

    return parser


def _handle_template(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    car_model = str(namespace.car_model or _default_car_model(config))
    pack_root = _resolve_pack_root(namespace, config)
    track_selection = _resolve_track_argument(namespace.track, config, pack_root=pack_root)
    track_name = track_selection.name or _default_track_name(config)
    engine = RecommendationEngine(car_model=car_model, track_name=track_name)
    context = engine._resolve_context(car_model, track_name)
    profile = context.thresholds
    templates = _profile_phase_templates(profile)
    average_delta = (
        mean(entry["target_delta_nfr"] for entry in templates.values())
        if templates
        else 0.0
    )

    lines = [
        f"# Preset generado para {context.profile_label}",
        "",
        "[limits.delta_nfr]",
        f"entry = {profile.entry_delta_tolerance:.3f}",
        f"apex = {profile.apex_delta_tolerance:.3f}",
        f"exit = {profile.exit_delta_tolerance:.3f}",
        f"piano = {profile.piano_delta_tolerance:.3f}",
        "",
    ]

    for section in ("analyze", "report"):
        lines.append(f"[{section}]")
        lines.append(f"target_delta = {average_delta:.3f}")
        lines.append("")
        for phase, payload in templates.items():
            lines.append(f"[{section}.phase_templates.{phase}]")
            lines.append(f"target_delta_nfr = {payload['target_delta_nfr']:.3f}")
            lat = payload["slip_lat_window"]
            lines.append(f"slip_lat_window = [{lat[0]:.3f}, {lat[1]:.3f}]")
            lng = payload["slip_long_window"]
            lines.append(f"slip_long_window = [{lng[0]:.3f}, {lng[1]:.3f}]")
            yaw = payload["yaw_rate_window"]
            lines.append(f"yaw_rate_window = [{yaw[0]:.3f}, {yaw[1]:.3f}]")
            lines.append("")

    result = "\n".join(lines).rstrip() + "\n"
    print(result)
    return result


def _handle_osd(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    layout_defaults = ButtonLayout().clamp()
    layout = ButtonLayout(
        left=layout_defaults.left if namespace.layout_left is None else int(namespace.layout_left),
        top=layout_defaults.top if namespace.layout_top is None else int(namespace.layout_top),
        width=layout_defaults.width if namespace.layout_width is None else int(namespace.layout_width),
        height=layout_defaults.height if namespace.layout_height is None else int(namespace.layout_height),
        ucid=layout_defaults.ucid,
        inst=layout_defaults.inst,
        click_id=layout_defaults.click_id,
        style=layout_defaults.style,
        type_in=layout_defaults.type_in,
    )
    pack_root = _resolve_pack_root(namespace, config)
    profiles_ctx = _resolve_profiles_path(config, pack_root=pack_root)
    profile_manager = ProfileManager(profiles_ctx.storage_path)
    resolved_car_model = str(namespace.car_model or _default_car_model(config)).strip()
    if not resolved_car_model:
        resolved_car_model = _default_car_model(config)
    track_selection = _resolve_track_argument(namespace.track, config, pack_root=pack_root)
    resolved_track = track_selection.name or _default_track_name(config)
    cars = _load_pack_cars(pack_root)
    track_profiles = _load_pack_track_profiles(pack_root)
    modifiers = _load_pack_modifiers(pack_root)
    class_overrides = _load_pack_lfs_class_overrides(pack_root)
    tnfr_targets = _resolve_tnfr_targets(
        resolved_car_model,
        cars,
        profiles_ctx.pack_profiles,
        overrides=class_overrides,
    )
    pack_delta, pack_si = _extract_target_objectives(tnfr_targets)
    engine = RecommendationEngine(
        car_model=resolved_car_model,
        track_name=resolved_track,
        profile_manager=profile_manager,
    )
    session_payload = _assemble_session_payload(
        resolved_car_model,
        track_selection,
        cars=cars,
        track_profiles=track_profiles,
        modifiers=modifiers,
    )
    if session_payload is not None:
        engine.session = session_payload
    if pack_delta is not None or pack_si is not None:
        base_profile = engine._lookup_profile(resolved_car_model, resolved_track)
        snapshot = profile_manager.resolve(
            resolved_car_model,
            resolved_track,
            base_profile,
            session=session_payload,
        )
        profile_manager.update_objectives(
            resolved_car_model,
            resolved_track,
            pack_delta if pack_delta is not None else snapshot.objectives.target_delta_nfr,
            pack_si if pack_si is not None else snapshot.objectives.target_sense_index,
        )
    hud = TelemetryHUD(
        car_model=resolved_car_model,
        track_name=resolved_track,
        recommendation_engine=engine,
        session=session_payload,
    )
    controller = OSDController(
        host=str(namespace.host),
        outsim_port=int(namespace.outsim_port),
        outgauge_port=int(namespace.outgauge_port),
        insim_port=int(namespace.insim_port),
        insim_keepalive=float(namespace.insim_keepalive),
        update_rate=float(namespace.update_rate),
        car_model=resolved_car_model,
        track_name=resolved_track,
        layout=layout,
        hud=hud,
    )
    return controller.run()


def _handle_diagnose(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    cfg_path: Path = namespace.cfg.expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró el fichero cfg.txt en {cfg_path}")

    sections = _parse_lfs_cfg(cfg_path)
    timeout = float(namespace.timeout)
    errors: List[str] = []
    successes: List[str] = []
    commands: List[str] = []

    outsim_host = sections["OutSim"].get("IP", "127.0.0.1")
    outsim_port = _coerce_int(sections["OutSim"].get("Port"))
    outsim_mode = _coerce_int(sections["OutSim"].get("Mode") or sections["OutSim"].get("Enable"))
    if outsim_mode != 1:
        suggested_port = outsim_port if outsim_port is not None else 4123
        command = f"/outsim 1 {outsim_host} {suggested_port}"
        commands.append(command)
        errors.append(
            "OutSim Mode debe ser 1 para habilitar la telemetría. "
            f"Ejecuta `{command}` dentro de LFS."
        )
    elif outsim_port is None:
        errors.append("OutSim Port no está definido en cfg.txt")
    else:
        ok, message = _outsim_ping(outsim_host, outsim_port, timeout)
        if ok:
            successes.append(message)
        else:
            errors.append(message)

    outgauge_host = sections["OutGauge"].get("IP", outsim_host)
    outgauge_port = _coerce_int(sections["OutGauge"].get("Port"))
    outgauge_mode = _coerce_int(
        sections["OutGauge"].get("Mode") or sections["OutGauge"].get("Enable")
    )
    if outgauge_mode != 1:
        suggested_port = outgauge_port if outgauge_port is not None else 3000
        command = f"/outgauge 1 {outgauge_host} {suggested_port}"
        commands.append(command)
        errors.append(
            "OutGauge Mode debe ser 1 para habilitar la transmisión. "
            f"Ejecuta `{command}` dentro de LFS."
        )
    elif outgauge_port is None:
        errors.append("OutGauge Port no está definido en cfg.txt")
    else:
        ok, message = _outgauge_ping(outgauge_host, outgauge_port, timeout)
        if ok:
            successes.append(message)
        else:
            errors.append(message)

    insim_host = sections["InSim"].get("IP", outsim_host)
    insim_port = _coerce_int(sections["InSim"].get("Port"))
    insim_command: str | None = None
    if insim_port is not None:
        insim_command = f"/insim {insim_port}"
        ok, message = _insim_handshake(insim_host, insim_port, timeout)
        if ok:
            successes.append(message)
        else:
            commands.append(insim_command)
            errors.append(f"{message}. Ejecuta `{insim_command}` en el chat de LFS para reintentar.")
    else:
        insim_command = "/insim 29999"
        commands.append(insim_command)
        errors.append(
            "InSim Port no está definido en cfg.txt. "
            f"Ejecuta `{insim_command}` y verifica que Live for Speed confirme la conexión."
        )

    setups_ok, setups_message = _check_setups_directory(cfg_path)
    if setups_ok:
        successes.append(setups_message)
    else:
        errors.append(setups_message)

    header = f"Diagnóstico de cfg.txt en {cfg_path}"
    if errors:
        _share_disabled_commands(commands)
        summary = [header, "Estado: errores detectados"]
        summary.extend(f"- {error}" for error in errors)
        if successes:
            summary.append("Detalles adicionales:")
            summary.extend(f"  * {success}" for success in successes)
        message = "\n".join(summary)
        print(message)
        raise ValueError(message)

    summary = [header, "Estado: correcto"]
    summary.extend(f"- {success}" for success in successes)
    _share_disabled_commands(commands)
    message = "\n".join(summary)
    print(message)
    return message


def _handle_baseline(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    destination = _resolve_baseline_destination(namespace, config)

    overlay: OverlayManager | None = None
    records: Records = []
    if namespace.overlay and namespace.simulate is not None:
        raise ValueError("--overlay solo está disponible con captura en vivo (sin --simulate)")

    try:
        if namespace.overlay:
            overlay_client = InSimClient(
                host=namespace.host,
                port=namespace.insim_port,
                timeout=namespace.timeout,
                keepalive_interval=namespace.insim_keepalive,
                app_name="TNFR Baseline",
            )
            layout = ButtonLayout(left=10, top=10, width=180, height=30, click_id=7)
            overlay = OverlayManager(overlay_client, layout=layout)
            overlay.connect()
            overlay.show(
                [
                    "Capturando baseline",
                    f"Duración máx: {int(namespace.duration)} s",
                    f"Muestras objetivo: {namespace.max_samples}",
                ]
            )

        if namespace.simulate is not None:
            records = OutSimClient().ingest(namespace.simulate)
            if namespace.limit is not None:
                records = records[: namespace.limit]
        else:
            heartbeat = overlay.tick if overlay is not None else None
            records = _capture_udp_samples(
                duration=namespace.duration,
                max_samples=namespace.max_samples,
                host=namespace.host,
                outsim_port=namespace.outsim_port,
                outgauge_port=namespace.outgauge_port,
                timeout=namespace.timeout,
                retries=namespace.retries,
                heartbeat=heartbeat,
            )
    finally:
        if overlay is not None:
            overlay.close()

    if not records:
        message = "No telemetry samples captured."
        print(message)
        return message

    _persist_records(records, destination, namespace.format)
    message = (
        f"Baseline saved {len(records)} samples to {destination} "
        f"({namespace.format})."
    )
    print(message)
    return message


def _handle_analyze(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    records = _load_records(namespace.telemetry)
    car_model = _default_car_model(config)
    pack_root = _resolve_pack_root(namespace, config)
    track_selection = _resolve_track_argument(None, config, pack_root=pack_root)
    track_name = track_selection.name or _default_track_name(config)
    profiles_ctx = _resolve_profiles_path(config, pack_root=pack_root)
    profile_manager = ProfileManager(profiles_ctx.storage_path)
    cars = _load_pack_cars(pack_root)
    track_profiles = _load_pack_track_profiles(pack_root)
    modifiers = _load_pack_modifiers(pack_root)
    class_overrides = _load_pack_lfs_class_overrides(pack_root)
    tnfr_targets = _resolve_tnfr_targets(
        car_model,
        cars,
        profiles_ctx.pack_profiles,
        overrides=class_overrides,
    )
    car_metadata = _lookup_car_metadata(car_model, cars)
    pack_delta, pack_si = _extract_target_objectives(tnfr_targets)
    engine = RecommendationEngine(
        car_model=car_model,
        track_name=track_name,
        profile_manager=profile_manager,
    )
    session_payload = _assemble_session_payload(
        car_model,
        track_selection,
        cars=cars,
        track_profiles=track_profiles,
        modifiers=modifiers,
    )
    if session_payload is not None:
        engine.session = session_payload
    bundles, microsectors, thresholds, snapshot = _compute_insights(
        records,
        car_model=car_model,
        track_name=track_name,
        engine=engine,
        profile_manager=profile_manager,
    )
    analyze_cfg = dict(config.get("analyze", {}))
    default_target_delta = float(analyze_cfg.get("target_delta", 0.0))
    default_target_si = float(analyze_cfg.get("target_si", 0.75))
    objectives = snapshot.objectives if snapshot else ProfileObjectives()
    target_delta = namespace.target_delta
    target_si = namespace.target_si
    if snapshot and target_delta == default_target_delta:
        target_delta = objectives.target_delta_nfr
    elif not snapshot and target_delta == default_target_delta and pack_delta is not None:
        target_delta = pack_delta
    if snapshot and target_si == default_target_si:
        target_si = objectives.target_sense_index
    elif not snapshot and target_si == default_target_si and pack_si is not None:
        target_si = pack_si
    profile_manager.update_objectives(car_model, track_name, target_delta, target_si)
    lap_segments = _group_records_by_lap(records)
    metrics = orchestrate_delta_metrics(
        lap_segments,
        target_delta,
        target_si,
        coherence_window=namespace.coherence_window,
        recursion_decay=namespace.recursion_decay,
        microsectors=microsectors,
        phase_weights=thresholds.phase_weights,
    )
    delta_metric = _effective_delta_metric(metrics)
    engine.register_stint_result(
        sense_index=metrics.get("sense_index", 0.0),
        delta_nfr=delta_metric,
        car_model=car_model,
        track_name=track_name,
    )
    phase_messages = _phase_deviation_messages(
        bundles,
        microsectors,
        config,
        car_model=car_model,
        track_name=track_name,
        session=session_payload,
    )
    session_messages = format_session_messages(session_payload)
    if session_messages:
        phase_messages.extend(session_messages)
    reports = _generate_out_reports(
        records,
        bundles,
        microsectors,
        _resolve_output_dir(config) / namespace.telemetry.stem,
        microsector_variability=metrics.get("microsector_variability"),
        metrics=metrics,
        artifact_format=getattr(namespace, "report_format", "json"),
    )
    robustness_thresholds = getattr(thresholds, "robustness", None)
    stages_payload = metrics.get("stages") if isinstance(metrics, Mapping) else None
    reception_stage = (
        stages_payload.get("recepcion") if isinstance(stages_payload, Mapping) else None
    )
    lap_indices = (
        reception_stage.get("lap_indices") if isinstance(reception_stage, Mapping) else None
    )
    lap_metadata = metrics.get("lap_sequence") if isinstance(metrics, Mapping) else None
    robustness_metrics = compute_session_robustness(
        metrics.get("bundles") or bundles,
        lap_indices=lap_indices,
        lap_metadata=lap_metadata,
        microsectors=microsectors,
        thresholds=robustness_thresholds,
    )
    if robustness_metrics and session_payload is not None:
        session_mapping = dict(session_payload)
        session_metrics_block = session_mapping.get("metrics")
        if isinstance(session_metrics_block, Mapping):
            metrics_profile = dict(session_metrics_block)
        else:
            metrics_profile = {}
        metrics_profile["robustness"] = robustness_metrics
        session_mapping["metrics"] = metrics_profile
        session_payload = session_mapping
    phase_templates = _phase_templates_from_config(config, "analyze")
    intermediate_metrics = {
        "epi_evolution": metrics.get("epi_evolution"),
        "nodal_metrics": metrics.get("nodal_metrics"),
        "sense_memory": metrics.get("sense_memory"),
    }
    summary_metrics = {
        "delta_nfr": metrics.get("delta_nfr", 0.0),
        "sense_index": metrics.get("sense_index", 0.0),
        "dissonance": metrics.get("dissonance", 0.0),
        "coupling": metrics.get("coupling", 0.0),
        "resonance": metrics.get("resonance", 0.0),
        "dissonance_breakdown": metrics.get("dissonance_breakdown"),
        "pairwise_coupling": metrics.get("pairwise_coupling"),
        "microsector_variability": metrics.get("microsector_variability", []),
    }
    if robustness_metrics:
        summary_metrics["robustness"] = robustness_metrics
    if session_payload is not None:
        summary_metrics["session"] = session_payload
    payload: Dict[str, Any] = {
        "series": bundles,
        "microsectors": microsectors,
        "telemetry_samples": len(records),
        "metrics": summary_metrics,
        "smoothed_series": metrics.get("bundles", []),
        "intermediate_metrics": intermediate_metrics,
        "stages": metrics.get("stages"),
        "objectives": metrics.get("objectives", {}),
        "phase_messages": phase_messages,
        "reports": reports,
    }
    if session_payload is not None:
        payload["session"] = session_payload
    if session_messages:
        payload["session_messages"] = session_messages
    if car_metadata is not None:
        payload["car"] = _serialise_pack_payload(car_metadata)
    if tnfr_targets is not None:
        payload["tnfr_targets"] = _serialise_pack_payload(tnfr_targets)
    if phase_templates:
        payload["phase_templates"] = phase_templates
    return _render_payload(payload, _resolve_exports(namespace))


def _handle_suggest(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    records = _load_records(namespace.telemetry)
    pack_root = _resolve_pack_root(namespace, config)
    profiles_ctx = _resolve_profiles_path(config, pack_root=pack_root)
    profile_manager = ProfileManager(profiles_ctx.storage_path)
    cars = _load_pack_cars(pack_root)
    track_profiles = _load_pack_track_profiles(pack_root)
    modifiers = _load_pack_modifiers(pack_root)
    class_overrides = _load_pack_lfs_class_overrides(pack_root)
    tnfr_targets = _resolve_tnfr_targets(
        namespace.car_model,
        cars,
        profiles_ctx.pack_profiles,
        overrides=class_overrides,
    )
    car_metadata = _lookup_car_metadata(namespace.car_model, cars)
    pack_delta, pack_si = _extract_target_objectives(tnfr_targets)
    track_selection = _resolve_track_argument(namespace.track, config, pack_root=pack_root)
    track_name = track_selection.name or _default_track_name(config)
    engine = RecommendationEngine(
        car_model=namespace.car_model,
        track_name=track_name,
        profile_manager=profile_manager,
    )
    session_payload = _assemble_session_payload(
        namespace.car_model,
        track_selection,
        cars=cars,
        track_profiles=track_profiles,
        modifiers=modifiers,
    )
    if session_payload is not None:
        engine.session = session_payload
    bundles, microsectors, thresholds, snapshot = _compute_insights(
        records,
        car_model=namespace.car_model,
        track_name=track_name,
        engine=engine,
        profile_manager=profile_manager,
    )
    lap_segments = _group_records_by_lap(records)
    suggest_cfg = dict(config.get("suggest", {}))
    objectives = snapshot.objectives if snapshot else ProfileObjectives()
    if snapshot is None and pack_delta is not None:
        objectives = ProfileObjectives(
            target_delta_nfr=float(pack_delta),
            target_sense_index=(
                float(pack_si)
                if pack_si is not None
                else objectives.target_sense_index
            ),
        )
    target_delta = float(suggest_cfg.get("target_delta", objectives.target_delta_nfr))
    target_si = float(suggest_cfg.get("target_si", objectives.target_sense_index))
    profile_manager.update_objectives(
        namespace.car_model,
        track_name,
        target_delta,
        target_si,
    )
    metrics = orchestrate_delta_metrics(
        lap_segments,
        target_delta,
        target_si,
        coherence_window=int(suggest_cfg.get("coherence_window", 3)),
        recursion_decay=float(suggest_cfg.get("recursion_decay", 0.4)),
        microsectors=microsectors,
        phase_weights=thresholds.phase_weights,
    )
    recommendations = engine.generate(
        bundles, microsectors, car_model=namespace.car_model, track_name=track_name
    )
    delta_metric = _effective_delta_metric(metrics)
    engine.register_stint_result(
        sense_index=metrics.get("sense_index", 0.0),
        delta_nfr=delta_metric,
        car_model=namespace.car_model,
        track_name=track_name,
    )
    phase_messages = _phase_deviation_messages(
        bundles,
        microsectors,
        config,
        car_model=namespace.car_model,
        track_name=track_name,
        session=session_payload,
    )
    session_messages = format_session_messages(session_payload)
    if session_messages:
        phase_messages.extend(session_messages)
    reports = _generate_out_reports(
        records,
        bundles,
        microsectors,
        _resolve_output_dir(config) / namespace.telemetry.stem,
        metrics=metrics,
    )
    payload = {
        "series": bundles,
        "microsectors": microsectors,
        "recommendations": recommendations,
        "car_model": namespace.car_model,
        "track": track_name,
        "phase_messages": phase_messages,
        "reports": reports,
    }
    if car_metadata is not None:
        payload["car"] = _serialise_pack_payload(car_metadata)
    if tnfr_targets is not None:
        payload["tnfr_targets"] = _serialise_pack_payload(tnfr_targets)
    if metrics:
        payload["metrics"] = {
            "delta_nfr": metrics.get("delta_nfr", 0.0),
            "sense_index": metrics.get("sense_index", 0.0),
            "dissonance": metrics.get("dissonance", 0.0),
            "coupling": metrics.get("coupling", 0.0),
            "resonance": metrics.get("resonance", 0.0),
            "pairwise_coupling": metrics.get("pairwise_coupling"),
            "dissonance_breakdown": metrics.get("dissonance_breakdown"),
        }
        if session_payload is not None:
            payload["metrics"]["session"] = session_payload
    if session_payload is not None:
        payload["session"] = session_payload
    if session_messages:
        payload["session_messages"] = session_messages
    return _render_payload(payload, _resolve_exports(namespace))


def _handle_compare(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    metric = str(namespace.metric)
    pack_root = _resolve_pack_root(namespace, config)
    track_selection = _resolve_track_argument(None, config, pack_root=pack_root)
    car_model = _default_car_model(config)
    track_name = track_selection.name or _default_track_name(config)
    report_cfg = dict(config.get("report", {}))
    target_delta = float(report_cfg.get("target_delta", 0.0))
    target_si = float(report_cfg.get("target_si", 0.75))
    coherence_window = int(report_cfg.get("coherence_window", 3))
    recursion_decay = float(report_cfg.get("recursion_decay", 0.4))

    def _compute_metrics(path: Path) -> Mapping[str, Any]:
        records = _load_records(path)
        lap_segments = _group_records_by_lap(records)
        return orchestrate_delta_metrics(
            lap_segments,
            target_delta,
            target_si,
            coherence_window=coherence_window,
            recursion_decay=recursion_decay,
        )

    try:
        baseline_metrics = _compute_metrics(namespace.telemetry_a)
        variant_metrics = _compute_metrics(namespace.telemetry_b)
        abtest_result = ab_compare_by_lap(
            baseline_metrics,
            variant_metrics,
            metric=metric,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cars = _load_pack_cars(pack_root)
    track_profiles = _load_pack_track_profiles(pack_root)
    modifiers = _load_pack_modifiers(pack_root)
    session_payload = _assemble_session_payload(
        car_model,
        track_selection,
        cars=cars,
        track_profiles=track_profiles,
        modifiers=modifiers,
    )
    if isinstance(session_payload, Mapping):
        session_mapping: Dict[str, Any] = dict(session_payload)
    else:
        session_mapping = {
            "car_model": car_model,
            "track_profile": track_selection.track_profile or track_name,
        }
    session_mapping["abtest"] = abtest_result

    payload: Dict[str, Any] = {
        "metric": metric,
        "baseline": {
            "telemetry": str(namespace.telemetry_a),
            "mean": abtest_result.baseline_mean,
            "lap_means": list(abtest_result.baseline_laps),
            "lap_count": len(abtest_result.baseline_laps),
        },
        "variant": {
            "telemetry": str(namespace.telemetry_b),
            "mean": abtest_result.variant_mean,
            "lap_means": list(abtest_result.variant_laps),
            "lap_count": len(abtest_result.variant_laps),
        },
        "session": session_mapping,
    }
    session_messages = format_session_messages(session_mapping)
    if session_messages:
        payload["session_messages"] = session_messages
    return _render_payload(payload, _resolve_exports(namespace))


def _handle_report(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:

    records = _load_records(namespace.telemetry)
    car_model = _default_car_model(config)
    pack_root = _resolve_pack_root(namespace, config)
    profiles_ctx = _resolve_profiles_path(config, pack_root=pack_root)
    profile_manager = ProfileManager(profiles_ctx.storage_path)
    track_selection = _resolve_track_argument(None, config, pack_root=pack_root)
    track_name = track_selection.name or _default_track_name(config)
    cars = _load_pack_cars(pack_root)
    track_profiles = _load_pack_track_profiles(pack_root)
    modifiers = _load_pack_modifiers(pack_root)
    engine = RecommendationEngine(
        car_model=car_model,
        track_name=track_name,
        profile_manager=profile_manager,
    )
    session_payload = _assemble_session_payload(
        car_model,
        track_selection,
        cars=cars,
        track_profiles=track_profiles,
        modifiers=modifiers,
    )
    if session_payload is not None:
        engine.session = session_payload
    bundles, microsectors, thresholds, snapshot = _compute_insights(
        records,
        car_model=car_model,
        track_name=track_name,
        engine=engine,
        profile_manager=profile_manager,
    )
    report_cfg = dict(config.get("report", {}))
    default_target_delta = float(report_cfg.get("target_delta", 0.0))
    default_target_si = float(report_cfg.get("target_si", 0.75))
    objectives = snapshot.objectives if snapshot else ProfileObjectives()
    target_delta = namespace.target_delta
    target_si = namespace.target_si
    if snapshot and target_delta == default_target_delta:
        target_delta = objectives.target_delta_nfr
    if snapshot and target_si == default_target_si:
        target_si = objectives.target_sense_index
    profile_manager.update_objectives(car_model, track_name, target_delta, target_si)
    lap_segments = _group_records_by_lap(records)
    metrics = orchestrate_delta_metrics(
        lap_segments,
        target_delta,
        target_si,
        coherence_window=namespace.coherence_window,
        recursion_decay=namespace.recursion_decay,
        microsectors=microsectors,
        phase_weights=thresholds.phase_weights,
    )
    delta_metric = _effective_delta_metric(metrics)
    engine.register_stint_result(
        sense_index=metrics.get("sense_index", 0.0),
        delta_nfr=delta_metric,
        car_model=car_model,
        track_name=track_name,
    )
    reports = _generate_out_reports(
        records,
        bundles,
        microsectors,
        _resolve_output_dir(config) / namespace.telemetry.stem,
        microsector_variability=metrics.get("microsector_variability"),
        metrics=metrics,
        artifact_format=getattr(namespace, "report_format", "json"),
    )
    payload: Dict[str, Any] = {
        "objectives": metrics.get("objectives", {}),
        "delta_nfr": metrics.get("delta_nfr", 0.0),
        "sense_index": metrics.get("sense_index", 0.0),
        "dissonance": metrics.get("dissonance", 0.0),
        "coupling": metrics.get("coupling", 0.0),
        "resonance": metrics.get("resonance", 0.0),
        "recursive_trace": metrics.get("recursive_trace", []),
        "pairwise_coupling": metrics.get("pairwise_coupling", {}),
        "dissonance_breakdown": metrics.get("dissonance_breakdown"),
        "microsector_variability": metrics.get("microsector_variability", []),
        "lap_sequence": metrics.get("lap_sequence", []),
        "series": bundles if bundles else metrics.get("bundles", []),
        "reports": reports,
        "intermediate_metrics": {
            "epi_evolution": metrics.get("epi_evolution"),
            "nodal_metrics": metrics.get("nodal_metrics"),
            "sense_memory": metrics.get("sense_memory"),
        },
        "stages": metrics.get("stages"),
    }
    if session_payload is not None:
        payload["session"] = session_payload
    session_messages = format_session_messages(session_payload)
    if session_messages:
        payload["session_messages"] = session_messages
    phase_templates = _phase_templates_from_config(config, "report")
    if phase_templates:
        payload["phase_templates"] = phase_templates
    return _render_payload(payload, _resolve_exports(namespace))


def _handle_write_set(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    namespace.car_model = str(namespace.car_model or _default_car_model(config)).strip()
    if not namespace.car_model:
        raise SystemExit("Debe proporcionar un --car-model válido para generar el setup.")
    if namespace.set_output:
        namespace.set_output = normalise_set_output_name(namespace.set_output, namespace.car_model)
    context = _compute_setup_plan(namespace, config=config)
    payload = _build_setup_plan_payload(context, namespace)
    return _render_payload(payload, _resolve_exports(namespace))


def _handle_pareto(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str:
    namespace.car_model = str(namespace.car_model or _default_car_model(config)).strip()
    if not namespace.car_model:
        raise SystemExit("Debe proporcionar un --car-model válido para evaluar el frente Pareto.")
    context = _compute_setup_plan(namespace, config=config)
    payload = _build_setup_plan_payload(context, namespace)

    planner = context.planner
    space = planner._adapt_space(
        planner._space_for_car(namespace.car_model),
        namespace.car_model,
        context.track_name,
    )
    centre_vector: Mapping[str, float] = context.plan.decision_vector or space.initial_guess()

    session_weights: Mapping[str, Mapping[str, float]] | None = None
    session_hints: Mapping[str, object] | None = None
    if isinstance(context.session_payload, Mapping):
        weights_candidate = context.session_payload.get("weights")
        if isinstance(weights_candidate, Mapping):
            session_weights = weights_candidate  # type: ignore[assignment]
        hints_candidate = context.session_payload.get("hints")
        if isinstance(hints_candidate, Mapping):
            session_hints = hints_candidate  # type: ignore[assignment]

    radius = max(0, int(getattr(namespace, "radius", 1)))
    candidates = sweep_candidates(
        space,
        centre_vector,
        context.bundles,
        microsectors=context.microsectors,
        simulator=None,
        session_weights=session_weights,
        session_hints=session_hints,
        radius=radius,
        include_centre=True,
    )
    front = pareto_front(candidates)
    pareto_payload = [dict(point.as_dict()) for point in front]

    session_section: Mapping[str, Any] | None = payload.get("session")  # type: ignore[assignment]
    if isinstance(session_section, Mapping):
        updated_session = dict(session_section)
    elif isinstance(context.session_payload, Mapping):
        updated_session = dict(context.session_payload)
    else:
        updated_session = {}
    updated_session["pareto"] = pareto_payload
    payload["session"] = updated_session
    payload["pareto_points"] = pareto_payload
    payload["pareto_radius"] = radius
    return _render_payload(payload, _resolve_exports(namespace))


def _resolve_exports(namespace: argparse.Namespace) -> List[str]:
    exports = getattr(namespace, "exports", None)
    if exports:
        ordered: List[str] = []
        for name in exports:
            if name not in ordered:
                ordered.append(name)
        return ordered
    default = getattr(namespace, "export_default", None)
    if isinstance(default, str):
        return [default]
    raise ValueError("No exporter configured for this command")


def _render_payload(payload: Mapping[str, Any], exporters: Sequence[str] | str) -> str:
    if isinstance(exporters, str):
        selected = [exporters]
    else:
        ordered: List[str] = []
        for name in exporters:
            if name not in ordered:
                ordered.append(name)
        selected = ordered

    rendered_outputs: List[str] = []
    for exporter_name in selected:
        exporter = exporters_registry[exporter_name]
        rendered = exporter(dict(payload))
        print(rendered)
        rendered_outputs.append(rendered)

    return "\n\n".join(rendered_outputs)


def _group_records_by_lap(records: Records) -> List[Records]:
    if not records:
        return []
    labels: List[Any] = []
    last_label: Any = None
    for record in records:
        lap_value = getattr(record, "lap", None)
        if lap_value is not None:
            last_label = lap_value
        labels.append(last_label)
    if labels and labels[0] is None:
        first_label = next((label for label in labels if label is not None), None)
        if first_label is not None:
            labels = [label if label is not None else first_label for label in labels]
    unique_labels = {label for label in labels if label is not None}
    if len(unique_labels) <= 1:
        return [records]
    groups: List[Records] = []
    current_group: List[TelemetryRecord] = []
    current_label: Any = labels[0]
    for record, label in zip(records, labels):
        if current_group and label != current_label:
            groups.append(current_group)
            current_group = []
        current_group.append(record)
        current_label = label
    if current_group:
        groups.append(current_group)
    return groups or [records]


def _compute_insights(
    records: Records,
    *,
    car_model: str,
    track_name: str,
    engine: RecommendationEngine | None = None,
    profile_manager: ProfileManager | None = None,
) -> tuple[Bundles, Sequence[Microsector], ThresholdProfile, ProfileSnapshot | None]:
    engine = engine or RecommendationEngine(
        car_model=car_model,
        track_name=track_name,
        profile_manager=profile_manager,
    )
    base_profile = engine._lookup_profile(car_model, track_name)
    snapshot: ProfileSnapshot | None = None
    if profile_manager is not None:
        session_payload = getattr(engine, "session", None)
        snapshot = profile_manager.resolve(
            car_model, track_name, base_profile, session=session_payload
        )
        profile = snapshot.thresholds
    else:
        profile = base_profile
    if not records:
        return [], [], profile, snapshot
    extractor = EPIExtractor()
    bundles = extractor.extract(records)
    if not bundles:
        return bundles, [], profile, snapshot
    overrides = (
        snapshot.phase_weights if snapshot is not None else profile.phase_weights
    )
    microsectors = segment_microsectors(
        records,
        bundles,
        phase_weight_overrides=overrides if overrides else None,
    )
    return bundles, microsectors, profile, snapshot


def _capture_udp_samples(
    *,
    duration: float,
    max_samples: int,
    host: str,
    outsim_port: int,
    outgauge_port: int,
    timeout: float,
    retries: int,
    heartbeat: Callable[[], None] | None = None,
) -> Records:
    fusion = TelemetryFusion()
    records: Records = []
    deadline = monotonic() + max(duration, 0.0)

    with OutSimUDPClient(
        host=host, port=outsim_port, timeout=timeout, retries=retries
    ) as outsim, OutGaugeUDPClient(
        host=host, port=outgauge_port, timeout=timeout, retries=retries
    ) as outgauge:
        while len(records) < max_samples and monotonic() < deadline:
            if heartbeat is not None:
                heartbeat()
            outsim_packet = outsim.recv()
            outgauge_packet = outgauge.recv()
            if outsim_packet is None or outgauge_packet is None:
                if heartbeat is not None:
                    heartbeat()
                sleep(timeout)
                continue
            record = fusion.fuse(outsim_packet, outgauge_packet)
            records.append(record)

    return records


def _persist_records(records: Records, destination: Path, fmt: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        compress = destination.suffix in {".gz", ".gzip"}
        logs.write_run(records, destination, compress=compress)
        return

    if fmt == "parquet":
        serialised = [asdict(record) for record in records]
        try:
            import pandas as pd  # type: ignore

            frame = pd.DataFrame(serialised)
            frame.to_parquet(destination, index=False)
            return
        except ModuleNotFoundError:
            with destination.open("w", encoding="utf8") as handle:
                json.dump(serialised, handle, sort_keys=True)
            return

    raise ValueError(f"Unsupported format '{fmt}'.")


def _load_records(source: Path) -> Records:
    if not source.exists():
        raise FileNotFoundError(f"Telemetry source {source} does not exist")
    suffix = source.suffix.lower()
    name = source.name.lower()
    if suffix == ".csv":
        return OutSimClient().ingest(source)
    if name.endswith(".jsonl") or name.endswith(".jsonl.gz") or name.endswith(".jsonl.gzip"):
        return list(logs.iter_run(source))
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore

            frame = pd.read_parquet(source)
            data = frame.to_dict(orient="records")
        except ModuleNotFoundError:
            with source.open("r", encoding="utf8") as handle:
                data = json.load(handle)
        return [TelemetryRecord(**_coerce_payload(item)) for item in data]
    if suffix == ".json":
        with source.open("r", encoding="utf8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [TelemetryRecord(**_coerce_payload(item)) for item in data]
        raise ValueError(f"JSON telemetry source {source} must contain a list of samples")

    raise ValueError(f"Unsupported telemetry format: {source}")


def run_cli(args: Sequence[str] | None = None) -> str:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", dest="config_path", type=Path, default=None
    )
    config_parser.add_argument("--pack-root", dest="pack_root", type=Path, default=None)
    preliminary, remaining = config_parser.parse_known_args(args)
    remaining = list(remaining)
    if preliminary.pack_root is not None:
        remaining = ["--pack-root", str(preliminary.pack_root)] + remaining
    config = load_cli_config(preliminary.config_path)
    parser = build_parser(config)
    parser.set_defaults(config_path=preliminary.config_path)
    namespace = parser.parse_args(remaining)
    namespace.config = config
    namespace.config_path = (
        getattr(namespace, "config_path", None)
        or preliminary.config_path
        or config.get("_config_path")
    )
    handler: Callable[[argparse.Namespace, Mapping[str, Any]], str] = getattr(
        namespace, "handler"
    )
    return handler(namespace, config=config)


def main() -> None:  # pragma: no cover - thin wrapper
    run_cli()


if __name__ == "__main__":  # pragma: no cover - CLI invocation guard
    main()
