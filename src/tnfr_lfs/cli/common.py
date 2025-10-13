"""Shared helpers for TNFR × LFS command modules."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from tnfr_core.equations.epi import TelemetryRecord
from tnfr_lfs.exporters import exporters_registry
from tnfr_lfs.telemetry.track_loader import (
    Track,
    TrackConfig,
    load_modifiers as load_track_modifiers,
    load_track as load_track_manifest,
    load_track_profiles,
)
from tnfr_lfs.telemetry.config_loader import (
    Car as PackCar,
    Profile as PackProfile,
    load_cars as _load_pack_cars_dataset,
    load_profiles as _load_pack_profiles_dataset,
)
from tnfr_lfs.cli.errors import CliError
from tnfr_lfs.cli.io import Records, _load_records as _io_load_records, _load_replay_bundle

__all__ = [
    "CliError",
    "TrackSelection",
    "validated_export",
    "add_export_argument",
    "resolve_pack_root",
    "load_pack_cars",
    "load_pack_profiles",
    "load_pack_track_profiles",
    "load_pack_modifiers",
    "default_car_model",
    "default_track_name",
    "resolve_track_argument",
    "resolve_track_selection",
    "load_records",
    "resolve_cache_size",
    "_load_records_from_namespace",
    "group_records_by_lap",
    "render_payload",
    "resolve_exports",
]


_LAYOUT_PATTERN = re.compile(r"^([A-Z]{2,})([0-9]{1,2}[A-Z]?)$")

_CAR_MODEL_DEFAULT_SECTIONS: tuple[str, ...] = (
    "analyze",
    "suggest",
    "write_set",
    "report",
    "compare",
    "template",
    "osd",
    "baseline",
    "pareto",
)

_TRACK_DEFAULT_SECTIONS: tuple[str, ...] = (
    "analyze",
    "suggest",
    "report",
    "compare",
    "template",
    "osd",
)


@dataclass(frozen=True, slots=True)
class TrackSelection:
    """Normalised representation of a CLI track argument."""

    name: str
    layout: Optional[str] = None
    manifest: Optional[Track] = None
    config: Optional[TrackConfig] = None

    @property
    def track_profile(self) -> Optional[str]:
        return self.config.track_profile if self.config is not None else None


def validated_export(value: Any, *, fallback: str) -> str:
    """Return ``value`` when it matches a registered exporter, else ``fallback``."""

    if isinstance(value, str) and value in exporters_registry:
        return value
    return fallback


def add_export_argument(
    parser: argparse.ArgumentParser, *, default: str, help_text: str
) -> None:
    """Register the ``--export`` flag on ``parser`` with standard semantics."""

    parser.add_argument(
        "--export",
        dest="exports",
        choices=sorted(exporters_registry.keys()),
        action="append",
        help=f"{help_text} Repeat the flag to combine exporters.",
    )
    parser.set_defaults(exports=None, export_default=default)


def resolve_pack_root(
    namespace: Optional[argparse.Namespace], config: Mapping[str, Any]
) -> Optional[Path]:
    """Determine the root directory for an optional configuration pack."""

    candidate: Optional[Path] = None
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


def _pack_data_dir(pack_root: Optional[Path], name: str) -> Optional[Path]:
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


def load_pack_track_profiles(pack_root: Optional[Path]) -> Mapping[str, Mapping[str, Any]]:
    """Load track profile metadata either from ``pack_root`` or bundled defaults."""

    profiles_dir = _pack_data_dir(pack_root, "track_profiles")
    return load_track_profiles(profiles_dir) if profiles_dir is not None else load_track_profiles()


def load_pack_modifiers(pack_root: Optional[Path]) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    """Load modifier metadata either from ``pack_root`` or bundled defaults."""

    modifiers_dir = _pack_data_dir(pack_root, "modifiers")
    if modifiers_dir is not None:
        return load_track_modifiers(modifiers_dir)
    return load_track_modifiers()


def load_pack_cars(pack_root: Optional[Path]) -> Mapping[str, PackCar]:
    """Load car metadata either from ``pack_root`` or the bundled dataset."""

    cars_dir = _pack_data_dir(pack_root, "cars")
    if cars_dir is not None:
        return _load_pack_cars_dataset(cars_dir)
    return _load_pack_cars_dataset()


def load_pack_profiles(pack_root: Optional[Path]) -> Mapping[str, PackProfile]:
    """Load profile metadata either from ``pack_root`` or bundled defaults."""

    profiles_dir = _pack_data_dir(pack_root, "profiles")
    if profiles_dir is not None:
        return _load_pack_profiles_dataset(profiles_dir)
    return _load_pack_profiles_dataset()


def default_car_model(config: Mapping[str, Any]) -> str:
    """Return the preferred car model based on CLI configuration defaults."""

    for section in _CAR_MODEL_DEFAULT_SECTIONS:
        section_cfg = config.get(section)
        if isinstance(section_cfg, Mapping):
            candidate = section_cfg.get("car_model")
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return "XFG"


def default_track_name(config: Mapping[str, Any]) -> str:
    """Return the preferred track name based on CLI configuration defaults."""

    for section in _TRACK_DEFAULT_SECTIONS:
        section_cfg = config.get(section)
        if isinstance(section_cfg, Mapping):
            candidate = section_cfg.get("track")
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return "generic"


def _parse_layout_code(value: str) -> Optional[tuple[str, str]]:
    match = _LAYOUT_PATTERN.fullmatch(value.strip().upper())
    if match is None:
        return None
    slug, suffix = match.groups()
    return slug, f"{slug}{suffix}"


def _load_pack_track(pack_root: Optional[Path], slug: str) -> Track:
    tracks_dir = _pack_data_dir(pack_root, "tracks")
    return (
        load_track_manifest(slug, tracks_dir)
        if tracks_dir is not None
        else load_track_manifest(slug)
    )


def resolve_track_selection(track: str, *, pack_root: Optional[Path]) -> TrackSelection:
    """Resolve ``track`` into a :class:`TrackSelection` instance."""

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
        raise CliError(
            f"Track manifest '{slug}' is not available in the pack or bundled resources."
        ) from exc
    try:
        config = manifest.configs[layout_id]
    except KeyError as exc:
        raise CliError(f"Layout '{layout_id}' is not defined in '{manifest.path.name}'.") from exc
    return TrackSelection(name=layout_id, layout=layout_id, manifest=manifest, config=config)


def resolve_track_argument(
    track_value: Optional[str],
    config: Mapping[str, Any],
    *,
    pack_root: Optional[Path],
) -> TrackSelection:
    """Normalise a track CLI argument using configuration defaults when missing."""

    candidate = str(track_value).strip() if track_value is not None else ""
    if not candidate:
        candidate = default_track_name(config)
    selection = resolve_track_selection(candidate, pack_root=pack_root)
    if not selection.name:
        return TrackSelection(name=candidate)
    return selection


def load_records(source: Path) -> Records:
    """Load telemetry records from ``source`` with helpful error messages."""

    try:
        return _io_load_records(source)
    except FileNotFoundError as exc:
        raise CliError(str(exc)) from exc
    except ValueError as exc:
        raise CliError(str(exc)) from exc


def resolve_cache_size(namespace: argparse.Namespace, attribute: str) -> int | None:
    """Return the cache size stored on ``namespace.cache_options`` for ``attribute``."""

    cache_options = getattr(namespace, "cache_options", None)
    if cache_options is None:
        return None
    return getattr(cache_options, attribute, None)


def _load_records_from_namespace(
    namespace: argparse.Namespace,
) -> Tuple[Records, Path]:
    """Resolve telemetry records and path from CLI ``namespace`` arguments."""

    replay_bundle = getattr(namespace, "replay_csv_bundle", None)
    telemetry_path = getattr(namespace, "telemetry", None)
    if replay_bundle is not None:
        bundle_path = Path(replay_bundle)
        try:
            cache_size = resolve_cache_size(namespace, "telemetry_cache_size")
            records = _load_replay_bundle(bundle_path, cache_size=cache_size)
        except FileNotFoundError as exc:
            raise CliError(str(exc)) from exc
        namespace.telemetry = bundle_path
        return records, bundle_path

    if telemetry_path is None:
        raise CliError(
            "A telemetry baseline path is required unless --replay-csv-bundle is provided."
        )

    telemetry_path = Path(telemetry_path)
    namespace.telemetry = telemetry_path
    records = load_records(telemetry_path)
    return records, telemetry_path


def _unique_export_list(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def resolve_exports(namespace: argparse.Namespace) -> List[str]:
    """Return the exporters requested by ``namespace`` or raise :class:`CliError`."""

    exports = getattr(namespace, "exports", None)
    if exports:
        return _unique_export_list(exports)
    default = getattr(namespace, "export_default", None)
    if isinstance(default, str):
        return [default]
    raise CliError("No exporter configured for this command.")


_MARKDOWN_LOCALISATIONS: Mapping[str, Tuple[Tuple[str, str], ...]] = {}


def _localise_markdown(rendered: str, locale: str | None) -> str:
    if not locale:
        return rendered
    replacements = _MARKDOWN_LOCALISATIONS.get(locale)
    if not replacements:
        return rendered
    localised = rendered
    for source, target in replacements:
        localised = localised.replace(source, target)
    return localised


def render_payload(payload: Mapping[str, Any], exporters: Sequence[str] | str) -> str:
    """Render ``payload`` using the exporters specified in ``exporters``."""

    if isinstance(exporters, str):
        selected = [exporters]
    else:
        selected = _unique_export_list(exporters)

    markdown_locale: str | None = None
    if isinstance(payload, Mapping):
        raw_locale = payload.get("_markdown_locale")
        if isinstance(raw_locale, str) and raw_locale:
            markdown_locale = raw_locale

    rendered_outputs: List[str] = []
    for exporter_name in selected:
        exporter = exporters_registry[exporter_name]
        if isinstance(payload, Mapping):
            export_payload = dict(payload)
            export_payload.pop("_markdown_locale", None)
        else:
            export_payload = payload
        rendered = exporter(export_payload)
        if exporter_name == "markdown":
            rendered = _localise_markdown(rendered, markdown_locale)
        print(rendered)
        rendered_outputs.append(rendered)

    return "\n\n".join(rendered_outputs)


def group_records_by_lap(records: Records) -> List[Records]:
    """Split telemetry ``records`` into lap-based segments."""

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
    current_label: Any = None
    for record, label in zip(records, labels):
        if label is None:
            current_group.append(record)
            continue
        if current_label is None:
            current_label = label
        if label != current_label and current_group:
            groups.append(current_group[:])
            current_group = []
            current_label = label
        current_group.append(record)
    if current_group:
        groups.append(current_group)
    return groups

