# Operator mapping for TNFR × LFS pipelines

This guide summarises every operator exported via `tnfr_core.operators.__all__`,
explaining how the entry, apex, and exit phases of a microsector use telemetry
from the Live for Speed OutSim/OutGauge feeds to build TNFR metrics.  Use these
notes alongside the auto-generated API reference when extending orchestration
pipelines or porting heuristics to bespoke dashboards.

## Entry phase – establishing objectives and memory

Segmentation labels the **entry** window when sustained lateral acceleration
crosses the curvature threshold while the support detector is still idle,
marking the start of a microsector and priming the recursivity state machine.
The operator loop stores filtered thermal/style traces and runs mutation
heuristics whenever entropy spikes or the driver deviates from the expected
style envelope, ensuring fresh targets before the apex.【F:src/tnfr_core/metrics/segmentation.py†L5-L110】【F:src/tnfr_core/metrics/segmentation.py†L884-L949】

| Operator | Consumed OutSim / OutGauge signals | Derived TNFR metrics | Configurable parameters | Typical false positives | Reproducible example |
| --- | --- | --- | --- | --- | --- |
| [`emission_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Uses phase targets from segmentation (ΔNFR objective, Sense Index objective). No raw signal access; relies on prior microsector labelling. | Normalised `{"delta_nfr", "sense_index"}` objective map broadcast to downstream stages.【F:src/tnfr_core/operators/operators.py†L336-L342】 | None. | Overly aggressive targets can be hidden because Sense Index is clamped to `[0, 1]`, masking upstream calibration errors. | Feed the synthetic stint through the parity verifier to inspect the emitted objectives: `poetry run python tools/verify_tnfr_parity.py tests/data/synthetic_stint.csv --output parity_report.json`.【F:README.md†L73-L89】 |
| [`reception_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | OutSim wheel loads, slip ratios/angles, accelerations, yaw, steer inputs; OutGauge throttle/brake/clutch, engine speed – fused into `TelemetryRecord` streams.【F:src/tnfr_lfs/telemetry/fusion.py†L62-L205】 | [`EPIBundle`](reference/autoapi/tnfr_core/equations/epi/index.md#classes) series containing ΔNFR projections, Sense Index, per-node coherence scores. | Optional custom [`EPIExtractor`](reference/autoapi/tnfr_core/equations/epi/index.md#classes). | Missing OutGauge extended data can yield defaulted brake pressures/temperatures, lowering fidelity of the generated bundle. | Replay the {{ fixture_link("src/tnfr_lfs/resources/data/test1.raf", "repository RAF fixture") }} with the baseline CLI (see [CLI reference](cli.md#baseline)) to regenerate bundles: `poetry run tnfr_lfs baseline runs/session.jsonl _fixtures/src/tnfr_lfs/resources/data/test1.raf --format jsonl`.【F:docs/cli.md†L7-L68】 |
| [`recursivity_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Entry-phase filtered measures: thermal load (OutSim vertical load share), style index (Sense Index), brake temperatures/derivatives (OutSim wheel temps), timestamps, phase label.【F:src/tnfr_core/operators/operators.py†L850-L1044】 | Per-session microsector memory with exponentially filtered signals, convergence flag, stint rollover history, derivatives such as `tyre_temp_*_dt`. | `decay`, `history`, `max_samples`, `max_time_gap`, `convergence_window`, `convergence_threshold`. | Slow sampling or missing timestamps can trigger spurious stint rollovers when `max_time_gap` is exceeded; constant signals may report convergence prematurely if `convergence_threshold` is too high. | Inspect the recursivity trace emitted during the parity verification run—the JSON payload records the session state per microsector alongside the synthetic stint bundle.【F:src/tnfr_core/operators/operators.py†L996-L1044】【F:README.md†L73-L89】 |
| [`recursive_filter_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) / [`recursividad_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Any scalar telemetry (e.g., entry-phase slip ratio history); commonly applied to OutSim-derived series before comparing to targets. | Exponentially smoothed trace capturing hysteresis in the selected channel.【F:src/tnfr_core/operators/operators.py†L1352-L1372】 | `seed`, `decay`. | Choosing `decay` ≥ 1.0 removes smoothing, while very low decay amplifies noise and can resemble ringing. | Pipe the entry slip ratio column from `tests/data/synthetic_stint.csv` through the filter in a notebook to visualise hysteresis prior to apex alignment. |

## Apex phase – contextual dissonance and coupling

When the support detector fires (sharp ΔFz increase) and curvature remains high
the segmentation module locks onto the **apex**, updating the dominant goals and
feeding contextual ΔNFR into dissonance analytics.  Recursivity and mutation
outputs steer which archetype is active so the apex targets stay relevant even
if tyre entropy rises mid-corner.【F:src/tnfr_core/metrics/segmentation.py†L5-L110】【F:src/tnfr_core/metrics/segmentation.py†L884-L949】

| Operator | Consumed OutSim / OutGauge signals | Derived TNFR metrics | Configurable parameters | Typical false positives | Reproducible example |
| --- | --- | --- | --- | --- | --- |
| [`coherence_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | ΔNFR and Sense Index series computed from OutSim wheel loads, slip, yaw rate, and OutGauge throttle/brake history. | Bias-preserving moving-average smoothing of ΔNFR/Sense Index used to compute apex dissonance and coherence scores.【F:src/tnfr_core/operators/operators.py†L343-L372】 | `window` (odd, >0). | Short windows (<3) react to noise; large windows smear transient support spikes, hiding onset of understeer. | After running the parity verifier, chart the smoothed vs. raw ΔNFR in `parity_report.json` to confirm apex damping. |
| [`dissonance_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Apex ΔNFR sequence against archetype target. | Mean absolute deviation quantifying parasitic vs. useful energy at the tyre contact patch.【F:src/tnfr_core/operators/operators.py†L375-L381】 | None. | If the target profile is stale (mutation disabled) the deviation spikes even when the driver follows the intended line. | Compare the apex deviation for the synthetic stint before/after forcing a new archetype via the mutation thresholds. |
| [`dissonance_breakdown_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) & [`DissonanceBreakdown`](reference/autoapi/tnfr_core/operators/operators/index.md#classes) | OutSim tyre ΔNFR traces, support events (ΔFz), chassis yaw rate, contextual multipliers from segmentation bundles. | Split between useful/parasitic dissonance, event counts, yaw-accelerated samples, and percentage diagnostics for telemetry overlays.【F:src/tnfr_core/operators/operators.py†L383-L474】 | Implicit – honours segmentation context and contextual delta multipliers. | Mislabelled support events (e.g., kerb strikes) can inflate useful counts; missing bundles disable yaw-rate vetting. | Use `tests/data/synthetic_stint.csv` to simulate apex loads and inspect the resulting `useful_dissonance_ratio` in the parity report. |
| [`coupling_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) / [`acoplamiento_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Any paired apex series (ΔNFR vs. Sense Index, tyre vs. suspension ΔNFR). | Normalised coupling (Pearson correlation) per pair, informing synchrony dashboards.【F:src/tnfr_core/operators/operators.py†L503-L545】 | `strict_length`. | Constant or near-constant signals yield zero variance and therefore zero coupling—interpretation must account for flat traces. | Evaluate tyre–suspension coupling inside the parity report’s `pairwise_coupling` block. |
| [`pairwise_coupling_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Dictionary of apex node series (`{"tyres": ΔNFR, ...}`) extracted from OutSim bundles. | Map of labelled coupling coefficients for configured node pairs.【F:src/tnfr_core/operators/operators.py†L546-L568】 | `pairs`. | Missing node series default to zero coupling, which can look like a disconnection when telemetry simply dropped frames. | Run the parity verifier to populate the tyre/suspension/chassis pairs and validate the configured pair list. |
| [`resonance_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Apex Sense Index oscillations (OutSim slip angle, steering, yaw feedback). | RMS resonance score highlighting modal oscillations across the apex.【F:src/tnfr_core/operators/operators.py†L569-L575】 | None. | Spikes whenever slip oscillates around zero because RMS ignores sign; confirm with coupling metrics before flagging an issue. | Compute the resonance of the apex Sense Index trace from the parity run to identify oscillatory stints. |

## Exit phase – adaptations and orchestration

Exit begins when curvature relaxes and support events subside.  Segmentation
feeds the latest recursivity snapshot and mutation result back into the goal
builder so exit targets reflect tyre recovery.  The orchestration stage then
combines all smoothed series into a reportable bundle while keeping the network
memory synchronised for downstream consumers.【F:src/tnfr_core/metrics/segmentation.py†L884-L949】【F:src/tnfr_core/operators/operators.py†L1914-L2052】

| Operator | Consumed OutSim / OutGauge signals | Derived TNFR metrics | Configurable parameters | Typical false positives | Reproducible example |
| --- | --- | --- | --- | --- | --- |
| [`mutation_operator`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Exit entropy (ΔNFR variance), style index baseline, dynamic flags from brake/support detectors – all informed by OutSim loads and Sense Index history.【F:src/tnfr_core/operators/operators.py†L1242-L1351】【F:src/tnfr_core/metrics/segmentation.py†L884-L949】 | Updated archetype selection (`recuperacion`, etc.), mutation flags, entropy/style deltas for telemetry overlays. | `entropy_threshold`, `entropy_increase`, `style_threshold`. | High sensor noise during kerb exits may mimic entropy spikes and trigger unnecessary archetype swaps; tune thresholds per track. | In the parity verifier output, tweak the mutation thresholds and observe how quickly the archetype changes during the final microsectors. |
| [`orchestrate_delta_metrics`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Full telemetry segments (bundles, ΔNFR/Sense Index series, recursivity memory, microsector metadata) – ultimately sourced from OutSim/OutGauge fusion. | Aggregated ΔNFR/Sense Index means, dissonance breakdown, nodal coupling, EPI evolution, latent silence states, network memory snapshots.【F:src/tnfr_core/operators/operators.py†L1914-L2052】 | `coherence_window`, `recursion_decay`, `phase_weights`, `operator_state`. | Providing inconsistent microsector metadata (missing phase boundaries) can zero-out variability outputs even when the stint is valid. | Replay the {{ fixture_link("src/tnfr_lfs/resources/data/test1.raf", "repository RAF fixture") }} with the baseline CLI to rebuild the staged metrics bundle at `runs/session.jsonl`: `poetry run tnfr_lfs baseline runs/session.jsonl _fixtures/src/tnfr_lfs/resources/data/test1.raf --format jsonl`.【F:docs/cli.md†L7-L68】 |
| [`evolve_epi`](reference/autoapi/tnfr_core/operators/operators/index.md#functions) | Time-aligned ΔNFR by node (tyres, suspension, chassis), natural frequency targets, phase weights – derived from the fused OutSim/OutGauge telemetry and segmentation weights.【F:src/tnfr_core/operators/operators.py†L260-L318】 | Integrated / derivative Event Performance Index plus nodal evolution metadata consumed by exit reports. | None exposed; validates non-negative `dt`. | Large timestamp gaps (telemetry drop-outs) produce zero-order holds that underestimate derivatives; ensure replay timestamps are monotonic. | Inspect the EPI evolution arrays written by `orchestrate_delta_metrics` when analysing the JSONL bundle produced by the baseline run. |

### Using the fixture datasets

* **Synthetic stint CSV** – provides a deterministic stream to validate entry and
  apex analytics without Live for Speed.  The parity verifier command above
  recreates the segmentation, operator stages, and final metrics directly from
  {{ fixture_link("tests/data/synthetic_stint.csv") }}.
* **RAF capture** – exercises the full ingestion pipeline, from RAF parsing to
  OutSim/OutGauge fusion and the exit orchestration stages.  Use the bundled
  {{ fixture_link("src/tnfr_lfs/resources/data/test1.raf", "repository RAF fixture") }}
  (or your own capture) with [`tnfr_lfs baseline`](cli.md#baseline) to regenerate
  the published bundle: `poetry run tnfr_lfs baseline runs/session.jsonl
  _fixtures/src/tnfr_lfs/resources/data/test1.raf --format jsonl`.  The command
  refreshes the `runs/session.jsonl` file referenced throughout this document,
  ensuring no binary assets are copied into the published docs.【F:docs/cli.md†L7-L68】
