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

Computes :class:`tnfr_lfs.core.epi.EPIResult` objects including EPI,
ΔNFR, and ΔSi values.  Use :func:`tnfr_lfs.core.epi.compute_coherence`
to summarise the consistency of the EPI series.

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
