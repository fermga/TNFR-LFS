# `tnfr_lfs.telemetry.offline.raf` module
Utilities for reading Live for Speed RAF telemetry captures.

The RAF (Replay Analyser Format) files produced by Live for Speed contain a
compact binary snapshot of a car's static configuration followed by a stream of
per-frame telemetry samples.  The helpers in this module expose the RAF
structure through small ``dataclass`` based containers and provide conversions
to :class:`~tnfr_core.epi.TelemetryRecord` instances so that callers can use
RAF recordings with the rest of the telemetry tooling.

## Classes
### `RafHeader`
Top level metadata extracted from the RAF file header.

#### Methods
- `interval_seconds(self) -> float`
  - Sampling interval for the telemetry stream in seconds.

### `RafCarStatic`
Static car information stored in the RAF header.

### `RafWheelStatic`
Static data for a single wheel.

### `RafWheelFrame`
Per-frame telemetry captured for a single wheel.

### `RafFrame`
Single telemetry sample extracted from the RAF stream.

### `RafFile`
Fully parsed RAF file.

## Functions
- `read_raf(path: Path | str) -> RafFile`
  - Parse ``path`` into a :class:`RafFile` instance.

The loader validates the RAF magic, the reported block sizes and the overall
file size before decoding the static configuration and all telemetry frames.
- `raf_to_telemetry_records(raf: RafFile) -> list[TelemetryRecord]`
  - Convert a parsed RAF file to :class:`TelemetryRecord` samples.

