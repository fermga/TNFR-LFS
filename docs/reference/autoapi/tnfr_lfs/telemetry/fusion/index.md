# `tnfr_lfs.telemetry.fusion` module
Fusion utilities for OutSim and OutGauge telemetry.

## Classes
### `FusionCalibration`
Parameter set describing how telemetry signals should be scaled.

### `TelemetryFusion`
Combine OutSim and OutGauge packets into :class:`TelemetryRecord` objects.

#### Methods
- `reset(self) -> None`
  - Clear the internal telemetry history.
- `fuse(self, outsim: OutSimPacket, outgauge: OutGaugePacket) -> TelemetryRecord`
  - Return a :class:`TelemetryRecord` derived from both UDP sources.
- `fuse_to_bundle(self, outsim: OutSimPacket, outgauge: OutGaugePacket) -> EPIBundle`
  - Return an :class:`EPIBundle` for the latest fused sample.

