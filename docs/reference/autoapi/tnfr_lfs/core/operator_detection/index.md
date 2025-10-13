# `tnfr_core.operator_detection` module
Detection utilities for on-track operator events.

Each detection routine analyses a windowed sequence of
:class:`~tnfr_core.interfaces.SupportsTelemetrySample` objects and yields
event descriptors when the observed behaviour exceeds the
configured thresholds.  The detectors are intentionally lightweight so that
they can be executed on every microsector without adding measurable overhead
to the orchestration pipeline.

## Classes
### `OperatorEvent`
Summary of a detected operator opportunity.

#### Methods
- `as_mapping(self) -> Mapping[str, float | str | int]`

## Functions
- `normalize_structural_operator_identifier(identifier: str) -> str`
  - Return the canonical structural identifier for ``identifier``.
- `canonical_operator_label(identifier: str) -> str`
  - Return the canonical structural label for an operator identifier.
- `silence_event_payloads(events: Mapping[str, Sequence[Mapping[str, object]] | None] | None) -> Tuple[Mapping[str, object], ...]`
  - Return all silence payloads, accepting case-insensitive identifiers.
- `detect_al(records: Sequence[SupportsTelemetrySample], *, window: int = 5, lateral_threshold: float = 1.6, load_threshold: float = 250.0) -> List[Mapping[str, float | str | int]]`
  - Detect lateral support (AL) opportunities.

The detector evaluates absolute lateral acceleration within a sliding
window and confirms that the accompanying load transfer is significant
before emitting an event.
- `detect_oz(records: Sequence[SupportsTelemetrySample], *, window: int = 5, slip_threshold: float = 0.12, yaw_threshold: float = 0.25) -> List[Mapping[str, float | str | int]]`
  - Detect oversteer (OZ) excursions.
- `detect_il(records: Sequence[SupportsTelemetrySample], *, window: int = 5, base_threshold: float = 0.35, speed_gain: float = 0.012) -> List[Mapping[str, float | str | int]]`
  - Detect ideal-line (IL) deviations with a speed dependent threshold.
- `detect_silence(records: Sequence[SupportsTelemetrySample], *, window: int = 15, load_threshold: float = 400.0, accel_threshold: float = 0.8, delta_nfr_threshold: float = 45.0, structural_window: int = 11, structural_density_threshold: float = 0.2, min_duration: float = 0.8) -> List[Mapping[str, float | str | int]]`
  - Detect structural silence intervals with low dynamic activation.

## Attributes
- `STRUCTURAL_OPERATOR_LABELS = {'AL': 'Support', 'OZ': 'Dissonance', 'IL': 'Coherence', 'SILENCE': 'Structural silence'}`

