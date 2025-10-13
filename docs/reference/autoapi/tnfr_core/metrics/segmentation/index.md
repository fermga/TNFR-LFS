# `tnfr_core.metrics.segmentation` module
Microsector segmentation utilities.

This module analyses the stream of telemetry samples together with the
corresponding :class:`~tnfr_core.operators.interfaces.SupportsEPIBundle` instances to
derive microsectors and their tactical goals.  The segmentation is
performed using three simple heuristics inspired by motorsport
engineering practice:

* **Curvature detection** – sustained lateral acceleration indicates a
  cornering event which is the base unit for microsectors.
* **Support events** – sharp vertical load increases signal the moment
  where the chassis "leans" on the tyres and generates grip.
* **ΔNFR signatures** – the average ΔNFR within the microsector is used
  to classify the underlying archetype which in turn drives the goals
  for the entry, apex and exit phases.

## Classes
### `Goal`
Operational goal associated with a microsector phase.

### `Microsector`
Segment of the lap grouped by curvature and ΔNFR behaviour.

#### Methods
- `phase_indices(self, phase: PhaseLiteral) -> range`
  - Return the range of sample indices assigned to ``phase``.

## Functions
- `segment_microsectors(records: Sequence[SupportsTelemetrySample], bundles: Sequence[SupportsEPIBundle], *, operator_state: MutableMapping[str, Mapping[str, object]] | None = None, recursion_decay: float = 0.4, mutation_thresholds: Mapping[str, float] | None = None, phase_weight_overrides: Mapping[str, Mapping[str, float] | float] | None = None) -> List[Microsector]`
  - Derive microsectors from telemetry and ΔNFR signatures.

Parameters
----------
records:
    Telemetry samples in chronological order. Each instance must implement
    :class:`~tnfr_core.operators.interfaces.SupportsTelemetrySample` and satisfy
    :class:`~tnfr_core.operators.interfaces.SupportsContextRecord` so the
    contextual weighting heuristics can access the required signals.
bundles:
    Computed :class:`~tnfr_core.operators.interfaces.SupportsEPIBundle` entries for
    the same timestamps as ``records``. Every bundle must also implement the
    :class:`~tnfr_core.operators.interfaces.SupportsContextBundle` contract.

Returns
-------
list of :class:`Microsector`
    Each microsector contains phase objectives bound to an archetype. When
    ``phase_weight_overrides`` is provided the heuristically derived
    weighting profiles for every phase are blended with the supplied
    multipliers before recomputing ΔNFR/Sense Index bundles.
- `microsector_stability_metrics(microsector: Microsector) -> Tuple[float, float, float, float]`
  - Return structural silence coverage and variance metrics for ``microsector``.
- `detect_quiet_microsector_streaks(microsectors: Sequence[Microsector], *, min_length: int = 3, coverage_threshold: float = 0.65, slack_threshold: float = 0.25, si_variance_threshold: float = 0.0025, epi_derivative_threshold: float = 0.18) -> List[Tuple[int, ...]]`
  - Return index streaks where consecutive microsectors remain quiet.

## Attributes
- `CURVATURE_THRESHOLD = 1.2`
- `MIN_SEGMENT_LENGTH = 3`
- `BRAKE_THRESHOLD = -0.35`
- `SUPPORT_THRESHOLD = 350.0`
- `PhaseLiteral = str`

