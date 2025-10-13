# `tnfr_core.interfaces` module
Structural typing interfaces for telemetry and contextual helpers.

This module groups :class:`typing.Protocol` definitions that describe the
attributes consumed by the analytical helpers in :mod:`tnfr_core`.  The
protocols enable structural typing across the codebase so that third party
implementations can interoperate with the analytics layer without relying on
the concrete dataclasses used by the default ingestion pipeline.

## Classes
### `SupportsTelemetrySample` (Protocol)
Telemetry payload exposing analytics fields and optional metadata.

Implementations provide the numerical signals consumed by the
:mod:`tnfr_core` analytics along with optional descriptive metadata such
as ``car_model``, ``track_name``, and ``tyre_compound``.

### `SupportsEPINode` (Protocol)
Subsystem node payload used by EPI analytics.

### `SupportsTyresNode` (SupportsEPINode, Protocol)
Tyre subsystem payload consumed by analytics and operators.

### `SupportsSuspensionNode` (SupportsEPINode, Protocol)
Suspension subsystem payload consumed by analytics and operators.

### `SupportsChassisNode` (SupportsEPINode, Protocol)
Chassis subsystem payload consumed by analytics and operators.

### `SupportsBrakesNode` (SupportsEPINode, Protocol)
Brake subsystem payload consumed by analytics and operators.

### `SupportsTransmissionNode` (SupportsEPINode, Protocol)
Transmission subsystem payload consumed by analytics and operators.

### `SupportsTrackNode` (SupportsEPINode, Protocol)
Track condition payload consumed by analytics and operators.

### `SupportsDriverNode` (SupportsEPINode, Protocol)
Driver payload consumed by analytics and operators.

### `SupportsEPIBundle` (Protocol)
Aggregated telemetry insights required by EPI consumers.

### `SupportsContextRecord` (Protocol)
Telemetry-like payload exposing the fields required for context factors.

### `SupportsContextTyres` (Protocol)
Tyre subsystem metrics required to derive contextual surface ratios.

### `SupportsContextChassis` (Protocol)
Chassis subsystem metrics required to derive curvature and traffic cues.

### `SupportsContextTransmission` (Protocol)
Transmission subsystem metrics used as a fallback for traffic cues.

### `SupportsContextBundle` (Protocol)
Aggregate bundle exposing the nodes required by contextual helpers.

### `SupportsGoal` (Protocol)
Goal specification produced by the segmentation heuristics.

### `SupportsMicrosector` (Protocol)
Microsector abstraction consumed by operator orchestration.

#### Methods
- `phase_indices(self, phase: str) -> range`
  - Return the range of telemetry samples associated with ``phase``.

