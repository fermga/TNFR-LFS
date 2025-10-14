# Operator ↔ LFS signal mapping

This guide documents how every TNFR × LFS operator consumes the telemetry coming
from Live for Speed, the metrics each stage emits, and how to reproduce the
examples with the fixtures bundled in the repository. Combine the tables below
with :doc:`telemetry` when auditing captures or extending the pipeline.

## How phases are delineated

### Entry (`entry`)

Segmentation flags the entry phase when the longitudinal deceleration reaches
the minimum inside the micro-sector; the selected index marks the start of the
window and lasts until the next lateral-acceleration maximum.【F:src/tnfr_core/metrics/segmentation.py†L1369-L1407】 During this
segment the system captures the thermal and driving-style signals that populate
the recursive memory.【F:src/tnfr_core/metrics/segmentation.py†L386-L424】【F:src/tnfr_core/metrics/segmentation.py†L596-L881】

### Apex (`apex`)

The apex is detected at the lateral-acceleration peak and validated with the
support threshold (vertical load jump) to confirm the chassis is settled on the
tyres.【F:src/tnfr_core/metrics/segmentation.py†L1372-L1428】 The phase goals define the ΔNFR and Sense Index targets that must
remain stable under maximum load.【F:src/tnfr_core/metrics/segmentation.py†L275-L420】

### Exit (`exit`)

The exit begins when longitudinal acceleration turns positive again and the
curvature drops; segmentation slides the closing boundary to the sample with the
highest longitudinal acceleration and reuses the recursive memory to adapt the
recovery targets.【F:src/tnfr_core/metrics/segmentation.py†L1374-L1407】【F:src/tnfr_core/metrics/segmentation.py†L880-L998】

## Entry operators

| Operator | LFS signals consumed | Derived TNFR metrics | Configurable parameters | Typical false positives | Reproducible examples |
| --- | --- | --- | --- | --- | --- |
| [`reception_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | **OutSim**: per-wheel loads, lateral/longitudinal forces, slip angles/ratios, accelerations, yaw, suspension travel.<br>**OutGauge**: pedal inputs (throttle, brake, clutch), handbrake, gear, rpm, steering inputs, tyre temperatures and pressures, brake temperatures via `_estimate_thermal`.【F:src/tnfr_lfs/telemetry/fusion.py†L122-L220】【F:src/tnfr_lfs/telemetry/fusion.py†L430-L555】 | Converts fused `TelemetryRecord` instances into `EPIBundle` series with ΔNFR, Sense Index, and derived nodes per phase.【F:src/tnfr_core/operators/operators.py†L343-L351】 | Optional `extractor` argument to replace the default `EPIExtractor`.【F:src/tnfr_core/operators/operators.py†L343-L351】 | When OutGauge omits the extensions, temperatures/pressures persist the last finite value or `nan`, degrading the exported bundles.【F:src/tnfr_lfs/telemetry/fusion.py†L1025-L1136】【F:src/tnfr_lfs/telemetry/fusion.py†L1138-L1202】 | Reprocess `{{ fixture_link("tests/data/synthetic_stint.csv") }}` with `poetry run python tools/verify_tnfr_parity.py tests/data/synthetic_stint.csv --output parity_report.json` to inspect the generated bundles.【F:README.md†L65-L77】 |
| [`emission_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Consumes the ΔNFR/Sense Index targets calculated during segmentation (derived from OutSim/OutGauge).【F:src/tnfr_core/operators/operators.py†L336-L340】【F:src/tnfr_core/metrics/segmentation.py†L275-L420】 | Returns a `{"delta_nfr", "sense_index"}` map ready for the downstream phases.【F:src/tnfr_core/operators/operators.py†L336-L340】 | None. | When the Sense Index target leaves `[0, 1]` the value saturates and may hide upstream calibration errors.【F:src/tnfr_core/operators/operators.py†L336-L340】 | The parity report exposes the target pair for each micro-sector in `parity_report.json`.【F:README.md†L65-L77】 |
| [`recursivity_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Consumes the filtered entry series: tyre temperatures, pressures, derivatives, driving style, and phase tags, all derived from the fused record (OutSim + OutGauge).【F:src/tnfr_core/operators/operators.py†L674-L1040】【F:src/tnfr_lfs/telemetry/fusion.py†L430-L555】 | Maintains per-micro-sector state with filtered histories, thermal derivatives, and convergence flags for stint memory.【F:src/tnfr_core/operators/operators.py†L900-L1038】 | `decay`, `history`, `max_samples`, `max_time_gap`, `convergence_window`, `convergence_threshold`.【F:src/tnfr_core/operators/operators.py†L850-L1038】 | Slow sampling or gaps larger than `max_time_gap` force stint closures and may look like false mutations; flat signals reach convergence too early if `convergence_threshold` is high.【F:src/tnfr_core/operators/operators.py†L888-L1024】 | The parity report keeps the recursive trace with the metrics for each micro-sector for `{{ fixture_link("tests/data/synthetic_stint.csv") }}`.【F:README.md†L65-L77】 |
| [`recursive_filter_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) / [`recursividad_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Scalar series derived from any input channel (for example slip ratios or temperatures) pre-filtered from OutSim/OutGauge.【F:src/tnfr_core/operators/operators.py†L1352-L1380】 | Exponentially smoothed trace that captures hysteresis in the observed signal.【F:src/tnfr_core/operators/operators.py†L1352-L1366】 | `seed`, `decay` in `[0, 1)`.【F:src/tnfr_core/operators/operators.py†L1352-L1366】 | `decay` ≥ 1 removes smoothing; very low values amplify noise and create false oscillations.【F:src/tnfr_core/operators/operators.py†L1352-L1366】 | Export the filtered column from the parity JSON generated with `{{ fixture_link("tests/data/synthetic_stint.csv") }}` to compare the smoothed vs. raw values.【F:README.md†L65-L77】 |

## Apex operators

| Operator | LFS signals consumed | Derived TNFR metrics | Configurable parameters | Typical false positives | Reproducible examples |
| --- | --- | --- | --- | --- | --- |
| [`coherence_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | ΔNFR and Sense Index series computed from the fused OutSim/OutGauge payload at the micro-sector apex.【F:src/tnfr_lfs/telemetry/fusion.py†L122-L220】【F:src/tnfr_core/operators/operators.py†L343-L372】 | Sliding window with a conservative mean to smooth ΔNFR/Sense Index before comparing against the targets.【F:src/tnfr_core/operators/operators.py†L354-L372】 | Odd `window` > 0. | Short windows react to noise; long windows hide support spikes and raise false incoherence. | Plot the smoothed vs. raw series in `parity_report.json` after running the verifier on `{{ fixture_link("tests/data/synthetic_stint.csv") }}`.【F:README.md†L65-L77】 |
| [`dissonance_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Smoothed ΔNFR compared against the archetypal apex target.【F:src/tnfr_core/operators/operators.py†L354-L381】 | Mean absolute deviation that quantifies useful vs. parasitic energy.【F:src/tnfr_core/operators/operators.py†L375-L381】 | None. | Stale targets (no mutation) push the deviation up even when the line is correct. | Compare the value before and after forcing a mutation by editing the thresholds in the parity verifier.【F:README.md†L65-L77】 |
| [`dissonance_breakdown_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) & [`DissonanceBreakdown`](reference/autoapi/tnfr_core/operators/operators/index.md#classes) | Per-wheel ΔNFR, support events, yaw, and contextual multipliers extracted from the apex bundles.【F:src/tnfr_core/operators/operators.py†L383-L500】 | Splits the dissonance into useful/parasitic contributions and exposes event counts and accelerated yaw.【F:src/tnfr_core/operators/operators.py†L383-L500】 | Implicit: honours micro-sectors and contextual multipliers. | Kerbs misidentified as support inflate useful events; incomplete bundles disable yaw validation. | Inspect `useful_dissonance_ratio` in the parity report generated from `{{ fixture_link("tests/data/synthetic_stint.csv") }}`.【F:README.md†L65-L77】 |
| [`coupling_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) / [`acoplamiento_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Any synchronised pair of apex series (ΔNFR vs. Sense Index, tyres vs. suspension) extracted from the bundle.【F:src/tnfr_core/operators/operators.py†L503-L545】 | Normalised correlation per pair to evaluate nodal synchrony.【F:src/tnfr_core/operators/operators.py†L503-L545】 | `strict_length` enforces identical sample counts. | Constant series yield zero variance and reduce the apparent coupling. | Review `pairwise_coupling` in the parity JSON generated from `{{ fixture_link("tests/data/synthetic_stint.csv") }}`.【F:README.md†L65-L77】 |
| [`pairwise_coupling_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Dictionary of series per node derived from the apex bundle.【F:src/tnfr_core/operators/operators.py†L546-L566】 | Labelled coefficient map for each configured pair.【F:src/tnfr_core/operators/operators.py†L546-L566】 | Optional `pairs` to limit the combinations. | Missing nodes yield `0.0`, which may look like disconnection when the root cause is packet loss. | Regenerate the report with the RAF fixture to validate the pair list: `poetry run tnfr_lfs baseline runs/session.jsonl _fixtures/src/tnfr_lfs/resources/data/test1.raf --format jsonl`.【F:docs/cli.md†L5-L35】 |
| [`resonance_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Sense Index oscillations at the apex, fed by the fused OutSim/OutGauge payload.【F:src/tnfr_lfs/telemetry/fusion.py†L122-L220】【F:src/tnfr_core/operators/operators.py†L569-L575】 | RMS of the series to detect persistent oscillatory modes.【F:src/tnfr_core/operators/operators.py†L569-L575】 | None. | Zero-mean oscillations increase the RMS and can be confused with problematic resonance. | Compare the `resonance` metric in the parity JSON generated from `{{ fixture_link("tests/data/synthetic_stint.csv") }}`.【F:README.md†L65-L77】 |

## Exit operators

| Operator | LFS signals consumed | Derived TNFR metrics | Configurable parameters | Typical false positives | Reproducible examples |
| --- | --- | --- | --- | --- | --- |
| [`mutation_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Entropy, driving style, and phase entries produced by the recursive memory, which in turn depends on tyre temperatures/pressures derived from OutSim/OutGauge.【F:src/tnfr_core/operators/operators.py†L1242-L1340】【F:src/tnfr_core/operators/operators.py†L900-L1038】 | Selects the archetype, mutation flags, and style/entropy deltas for the current micro-sector.【F:src/tnfr_core/operators/operators.py†L1242-L1349】 | `entropy_threshold`, `entropy_increase`, `style_threshold`.【F:src/tnfr_core/operators/operators.py†L1246-L1275】 | Kerb vibrations boost entropy and may trigger unnecessary mutations when thresholds are low.【F:src/tnfr_core/operators/operators.py†L1318-L1330】 | Tweak the thresholds in the parity verifier for `{{ fixture_link("tests/data/synthetic_stint.csv") }}` and inspect the reported archetype changes.【F:README.md†L65-L77】 |
| [`orchestrate_delta_metrics`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Full telemetry segments: ΔNFR/Sense Index bundles, recursive memory, and micro-sector metadata derived from OutSim/OutGauge.【F:src/tnfr_core/operators/operators.py†L1914-L2055】【F:src/tnfr_lfs/telemetry/fusion.py†L122-L555】 | Assembles targets, smoothed series, dissonance breakdown, nodal coupling, EPI evolution, and memory snapshots.【F:src/tnfr_core/operators/operators.py†L1914-L2055】 | `coherence_window`, `recursion_decay`, `phase_weights`, `operator_state`.【F:src/tnfr_core/operators/operators.py†L1914-L2055】 | Inconsistent metadata (for example missing phase bounds) reduces variability and can hide real anomalies.【F:src/tnfr_core/operators/operators.py†L1914-L2049】 | Replay the RAF capture: `poetry run tnfr_lfs baseline runs/session.jsonl _fixtures/src/tnfr_lfs/resources/data/test1.raf --format jsonl` and review the resulting JSONL.【F:docs/cli.md†L5-L35】 |
| [`evolve_epi`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Per-node ΔNFR and natural-frequency targets derived from the fused OutSim/OutGauge telemetry and phase weights.【F:src/tnfr_core/operators/operators.py†L214-L333】 | Returns the updated EPI, its derivative, and the node contribution map feeding the exit reports.【F:src/tnfr_core/operators/operators.py†L214-L333】 | Validates `dt ≥ 0` and honours phase weights/targets when present.【F:src/tnfr_core/operators/operators.py†L214-L333】 | Large time gaps (`dt`) understate derivatives and can skew the EPI trend.【F:src/tnfr_core/operators/operators.py†L214-L333】 | The generated values live in `epi_evolution` inside the JSONL produced by `tnfr_lfs baseline` on the RAF capture.【F:docs/cli.md†L5-L35】 |

### Recommended fixtures

* **Synthetic stint CSV** – Deterministic resource to validate entry and apex
  phases without relying on Live for Speed. Run the parity verifier to
  regenerate the bundles and reports from
  {{ fixture_link("tests/data/synthetic_stint.csv") }}.【F:README.md†L65-L77】
* **RAF capture** – Exercises the full ingestion path, from the RAF parser to the
  exit operators. Replay the capture available at
  {{ fixture_link("src/tnfr_lfs/resources/data/test1.raf", "RAF capture") }} with
  [`tnfr_lfs baseline`](cli.md#baseline) to rebuild the published bundle and
  validate the telemetry dependencies.【F:docs/cli.md†L5-L35】
