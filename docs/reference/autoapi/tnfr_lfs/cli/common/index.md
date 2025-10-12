# `tnfr_lfs.cli.common` module
Shared helpers for TNFR × LFS command modules.

## Classes
### `TrackSelection`
Normalised representation of a CLI track argument.

#### Methods
- `track_profile(self) -> Optional[str]`

## Functions
- `validated_export(value: Any, *, fallback: str) -> str`
  - Return ``value`` when it matches a registered exporter, else ``fallback``.
- `add_export_argument(parser: argparse.ArgumentParser, *, default: str, help_text: str) -> None`
  - Register the ``--export`` flag on ``parser`` with standard semantics.
- `resolve_pack_root(namespace: Optional[argparse.Namespace], config: Mapping[str, Any]) -> Optional[Path]`
  - Determine the root directory for an optional configuration pack.
- `load_pack_track_profiles(pack_root: Optional[Path]) -> Mapping[str, Mapping[str, Any]]`
  - Load track profile metadata either from ``pack_root`` or bundled defaults.
- `load_pack_modifiers(pack_root: Optional[Path]) -> Mapping[tuple[str, str], Mapping[str, Any]]`
  - Load modifier metadata either from ``pack_root`` or bundled defaults.
- `load_pack_cars(pack_root: Optional[Path]) -> Mapping[str, PackCar]`
  - Load car metadata either from ``pack_root`` or the bundled dataset.
- `load_pack_profiles(pack_root: Optional[Path]) -> Mapping[str, PackProfile]`
  - Load profile metadata either from ``pack_root`` or bundled defaults.
- `default_car_model(config: Mapping[str, Any]) -> str`
  - Return the preferred car model based on CLI configuration defaults.
- `default_track_name(config: Mapping[str, Any]) -> str`
  - Return the preferred track name based on CLI configuration defaults.
- `resolve_track_selection(track: str, *, pack_root: Optional[Path]) -> TrackSelection`
  - Resolve ``track`` into a :class:`TrackSelection` instance.
- `resolve_track_argument(track_value: Optional[str], config: Mapping[str, Any], *, pack_root: Optional[Path]) -> TrackSelection`
  - Normalise a track CLI argument using configuration defaults when missing.
- `load_records(source: Path) -> Records`
  - Load telemetry records from ``source`` with helpful error messages.
- `resolve_cache_size(namespace: argparse.Namespace, attribute: str) -> int | None`
  - Return the cache size stored on ``namespace.cache_options`` for ``attribute``.
- `resolve_exports(namespace: argparse.Namespace) -> List[str]`
  - Return the exporters requested by ``namespace`` or raise :class:`CliError`.
- `render_payload(payload: Mapping[str, Any], exporters: Sequence[str] | str) -> str`
  - Render ``payload`` using the exporters specified in ``exporters``.
- `group_records_by_lap(records: Records) -> List[Records]`
  - Split telemetry ``records`` into lap-based segments.

