# `tnfr_lfs.telemetry.outsim_client` module
Simplified OutSim telemetry ingestion client.

The original TNFR project ingests data from an OutSim UDP stream that
encodes suspension loads, slip angles, and wheel data.  For the
purposes of this library we implement a light-weight client that reads
CSV-formatted telemetry into :class:`~tnfr_core.equations.epi.TelemetryRecord`
instances.  RAF captures produced by Live for Speed can be converted
into the same :class:`TelemetryRecord` structure via
``raf_to_telemetry_records(read_raf(...))``, making ``.raf`` files a
first-class telemetry source for the CLI while keeping this client
focused on deterministic CSV ingestion.

## Classes
### `TelemetrySchema`
Schema configuration for a telemetry dataset.

Attributes
----------
columns:
    Ordered list of expected column names.  The default schema maps
    closely to the OutSim telemetry stream.
delimiter:
    Delimiter used when parsing the telemetry file.

### `TelemetryFormatError` (ValueError)
Raised when the incoming telemetry cannot be parsed.

### `OutSimClient`
Client responsible for ingesting telemetry from different sources.

The client accepts either iterables of strings (such as file handles
or in-memory lists) or filesystem paths.  The parsing logic is kept
intentionally small to make it easy to extend with real OutSim
decoding if needed in the future.

#### Methods
- `ingest(self, source: str | Path | TextIO | Iterable[str]) -> List[TelemetryRecord]`
  - Return a list of :class:`TelemetryRecord` objects.

Parameters
----------
source:
    Either a path to a CSV file or any iterable that yields lines
    of text.  When working with RAF captures, prefer
    :func:`tnfr_lfs.telemetry.offline.raf_to_telemetry_records` in combination
    with :func:`tnfr_lfs.telemetry.offline.read_raf` before handing the
    resulting records to other consumers.

## Attributes
- `DEFAULT_SCHEMA = TelemetrySchema(columns=('timestamp', 'structural_timestamp', 'vertical_load', 'slip_ratio', 'slip_ratio_fl', 'slip_ratio_fr', 'slip_ratio_rl', 'slip_ratio_rr', 'lateral_accel', 'longitudinal_accel', 'yaw', 'pitch', 'roll', 'brake_pressure', 'locking', 'nfr', 'si', 'speed', 'yaw_rate', 'slip_angle', 'slip_angle_fl', 'slip_angle_fr', 'slip_angle_rl', 'slip_angle_rr', 'steer', 'throttle', 'gear', 'vertical_load_front', 'vertical_load_rear', 'mu_eff_front', 'mu_eff_rear', 'mu_eff_front_lateral', 'mu_eff_front_longitudinal', 'mu_eff_rear_lateral', 'mu_eff_rear_longitudinal', 'suspension_travel_front', 'suspension_travel_rear', 'suspension_velocity_front', 'suspension_velocity_rear', 'tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr', 'tyre_pressure_fl', 'tyre_pressure_fr', 'tyre_pressure_rl', 'tyre_pressure_rr', 'instantaneous_radius', 'front_track_width', 'wheelbase'))`
- `OPTIONAL_SCHEMA_COLUMNS = {'structural_timestamp', 'tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr', 'tyre_pressure_fl', 'tyre_pressure_fr', 'tyre_pressure_rl', 'tyre_pressure_rr', 'slip_ratio_fl', 'slip_ratio_fr', 'slip_ratio_rl', 'slip_ratio_rr', 'slip_angle_fl', 'slip_angle_fr', 'slip_angle_rl', 'slip_angle_rr', 'instantaneous_radius', 'front_track_width', 'wheelbase'}`
- `LEGACY_COLUMNS = ('timestamp', 'vertical_load', 'slip_ratio', 'lateral_accel', 'longitudinal_accel', 'yaw', 'pitch', 'roll', 'brake_pressure', 'locking', 'nfr', 'si')`
- `LEGACY_DEFAULTS = {'speed': math.nan, 'yaw_rate': math.nan, 'slip_angle': math.nan, 'slip_ratio_fl': math.nan, 'slip_ratio_fr': math.nan, 'slip_ratio_rl': math.nan, 'slip_ratio_rr': math.nan, 'slip_angle_fl': math.nan, 'slip_angle_fr': math.nan, 'slip_angle_rl': math.nan, 'slip_angle_rr': math.nan, 'steer': math.nan, 'throttle': math.nan, 'gear': 0, 'vertical_load_front': math.nan, 'vertical_load_rear': math.nan, 'mu_eff_front': math.nan, 'mu_eff_rear': math.nan, 'mu_eff_front_lateral': math.nan, 'mu_eff_front_longitudinal': math.nan, 'mu_eff_rear_lateral': math.nan, 'mu_eff_rear_longitudinal': math.nan, 'suspension_travel_front': math.nan, 'suspension_travel_rear': math.nan, 'suspension_velocity_front': math.nan, 'suspension_velocity_rear': math.nan, 'tyre_temp_fl': math.nan, 'tyre_temp_fr': math.nan, 'tyre_temp_rl': math.nan, 'tyre_temp_rr': math.nan, 'tyre_pressure_fl': math.nan, 'tyre_pressure_fr': math.nan, 'tyre_pressure_rl': math.nan, 'tyre_pressure_rr': math.nan, 'rpm': math.nan, 'line_deviation': math.nan, 'instantaneous_radius': math.nan, 'front_track_width': math.nan, 'wheelbase': math.nan}`

