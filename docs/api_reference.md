# API Reference

## Acquisition

### `tnfr_lfs.acquisition.outsim_client.OutSimClient`

Reads OutSim-style telemetry from CSV sources and returns a list of
:class:`tnfr_lfs.core.epi.TelemetryRecord` objects.  By default the
client validates the column order and converts values to floats.

```
from tnfr_lfs.acquisition import OutSimClient

client = OutSimClient()
records = client.ingest("stint.csv")
```

## Core Analytics

### `tnfr_lfs.core.epi.EPIExtractor`

Computes :class:`tnfr_lfs.core.epi_models.EPIBundle` objects including EPI,
ΔNFR, entropy-penalised sense index values, and per-node breakdowns.
Use :func:`tnfr_lfs.core.coherence.sense_index` together with
:func:`tnfr_lfs.core.coherence.compute_node_delta_nfr` to build custom
analytics pipelines.

### Operator utilities

```
tnfr_lfs.core.operators.emission_operator(target_delta_nfr: float, target_sense_index: float) -> Dict[str, float]
tnfr_lfs.core.operators.recepcion_operator(records: Sequence[TelemetryRecord], extractor: Optional[EPIExtractor] = None) -> List[EPIBundle]
tnfr_lfs.core.operators.coherence_operator(series: Sequence[float], window: int = 3) -> List[float]
tnfr_lfs.core.operators.dissonance_operator(series: Sequence[float], target: float) -> float
tnfr_lfs.core.operators.acoplamiento_operator(series_a: Sequence[float], series_b: Sequence[float]) -> float
tnfr_lfs.core.operators.resonance_operator(series: Sequence[float]) -> float
tnfr_lfs.core.operators.recursividad_operator(series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5) -> List[float]
tnfr_lfs.core.operators.orchestrate_delta_metrics(telemetry_segments: Sequence[Sequence[TelemetryRecord]], target_delta_nfr: float, target_sense_index: float, *, coherence_window: int = 3, recursion_decay: float = 0.4) -> Mapping[str, object]
```

These functions compose an end-to-end ΔNFR/Sense Index pipeline covering
objective setting (Emisión), telemetry reception, smoothing (Coherencia),
contrast measurement (Disonancia), coupling analysis, resonance estimation,
recursive filtering, and orchestration over segmented telemetry streams.

## Recommendation Engine

### `tnfr_lfs.recommender.rules.RecommendationEngine`

Applies load balance, stability index, and coherence rules to produce a
list of :class:`tnfr_lfs.recommender.rules.Recommendation` objects.
Custom rules can be added by implementing the
:class:`tnfr_lfs.recommender.rules.RecommendationRule` protocol.

## Exporters

Two exporters are provided out of the box:

* ``json`` – produces a JSON document ready for storage or pipelines.
* ``csv`` – returns a CSV representation of the EPI series.
