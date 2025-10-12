# `tnfr_lfs.cli.workflows` module
Command line entry point for TNFR × LFS.

## Classes
### `CaptureMetrics`

### `CaptureResult`

#### Methods
- `samples(self) -> int`

### `ProfilesContext`
Container for CLI profile settings and pack resources.

### `PackContext`
Aggregated resources sourced from an optional content pack.

### `SetupPlanContext`
Container storing the artefacts required to build setup outputs.

## Functions
- `compute_setup_plan(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> SetupPlanContext`
- `assemble_session_payload(car_model: str, selection: TrackSelection, *, cars: Mapping[str, PackCar], track_profiles: Mapping[str, Mapping[str, Any]], modifiers: Optional[Mapping[tuple[str, str], Mapping[str, Any]]]) -> Optional[Mapping[str, Any]]`
- `build_setup_plan_payload(context: SetupPlanContext, namespace: argparse.Namespace) -> Mapping[str, Any]`

## Attributes
- `logger = logging.getLogger(__name__)`
- `Bundles = Sequence[Any]`
- `DEFAULT_OUTPUT_DIR = Path('out')`
- `PROFILES_ENV_VAR = 'TNFR_LFS_PROFILES'`
- `DEFAULT_PROFILES_FILENAME = 'profiles.toml'`

