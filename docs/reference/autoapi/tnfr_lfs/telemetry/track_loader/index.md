# `tnfr_lfs.telemetry.track_loader` module
Helpers to resolve track manifests and recommendation profiles.

## Classes
### `TrackConfig`
Description of a specific Live for Speed layout.

### `Track`
Track manifest containing the available layouts.

## Functions
- `load_track(track_slug: str, tracks_dir: str | Path | None = None) -> Track`
  - Load a track manifest and expose its ``[config.*]`` sections.
- `load_track_profiles(profiles_dir: str | Path | None = None) -> dict[str, Mapping[str, Any]]`
  - Load threshold track profiles keyed by ``meta.id`` or filename stem.
- `load_modifiers(modifiers_dir: str | Path | None = None) -> dict[tuple[str, str], Mapping[str, Any]]`
  - Load car/track modifiers keyed by ``(car_profile, track_profile)``.
- `assemble_session_weights(car_profile: str, track_profile: str, *, track_profiles: Mapping[str, Mapping[str, Any]], modifiers: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> Mapping[str, Mapping[str, Any]]`
  - Combine a car profile and track profile into session phase weights.

