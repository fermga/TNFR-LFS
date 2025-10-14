# Calibration workflow

This guide walks through the end-to-end calibration tooling that tunes the TNFR × LFS structural detectors. It details the telemetry prerequisites, the supported label artefacts, how to interpret the generated metrics, and the way the resulting YAML overrides cascade through packs and runtime deployments.

## Prerequisites

The calibrator consumes the same RAF telemetry captures and Live for Speed streams as the runtime pipeline. Capture telemetry with Replay Analyzer or the bundled UDP clients following the [official RAF specification](https://en.lfsmanual.net/wiki/RAF) and the simulator telemetry guides ([OutSim](https://en.lfsmanual.net/wiki/OutSim), [OutGauge](https://en.lfsmanual.net/wiki/OutGauge)). When ingesting live data, verify that the simulator is configured as described in the :doc:`telemetry` reference so the fusion module can rebuild ΔNFR bundles.

Place the RAF captures under a dedicated directory (for example `/data/raf`) and prepare micro-sector labels that identify which structural operators were active. The calibrator reuses :mod:`tnfr_lfs.telemetry.offline.raf` to decode the captures and :class:`tnfr_core.equations.epi.EPIExtractor` plus :func:`tnfr_core.metrics.segmentation.segment_microsectors` to reconstruct the same bundles and micro-sectors used in production.【F:tools/calibrate_detectors.py†L1-L16】【F:tools/calibrate_detectors.py†L39-L47】 This keeps the evaluation aligned with the runtime behaviour documented in :doc:`operators_mapping`.

## Running the CLI

Launch the workflow with the helper shipped under ``tools/calibrate_detectors.py``. The script requires the RAF root, the labels artefact, the output directory, and the list of operators to calibrate. Optional filters limit the dataset to specific cars, classes, compounds, or tracks, and ``--operator-grid`` overrides the packaged parameter sweeps:

```bash
python tools/calibrate_detectors.py \
  --raf-root /data/raf \
  --labels labels.jsonl \
  --out output/calibration \
  --operators NAV EN SILENCE \
  --operator-grid grids.yaml
```

The command line parser mirrors these switches and aborts early when required inputs are missing, when no samples remain after filtering, or when the requested detectors lack a search space.【F:tools/calibrate_detectors.py†L1293-L1365】 During execution the tool groups labelled micro-sectors by class/car/compound combination, evaluates every requested detector across the packaged or custom parameter grid, and applies track-level cross-validation folds when ``--kfold`` is greater than one.【F:tools/calibrate_detectors.py†L1374-L1400】 Cross-reference :doc:`cli` if you need a refresher on ingesting RAF captures into JSONL runs before triggering calibrations.

GTR entries now accept both slick and road compounds for FXR, XRR, and FZR samples thanks to the expanded compatibility map, so calibrations built from mixed-tyre stints no longer raise invalid pair warnings.【F:src/tnfr_lfs/resources/tyre_compounds.py†L59-L94】

## Label artefacts and formats

The ``--labels`` argument accepts JSON, JSON Lines, TOML, YAML, and CSV artefacts. The loader normalises each format into ``LabelledMicrosector`` entries that identify the RAF capture, micro-sector index, structural operator booleans, and optional time intervals.【F:tools/calibrate_detectors.py†L398-L472】 JSON-like payloads can either provide a ``captures`` list or a flat mapping per capture; CSV rows use the ``microsector``/``microsector_index`` column alongside ``operators`` or ``label`` fields. Operator identifiers are automatically normalised to the canonical structural codes (NAV, EN, OZ, …).【F:tools/calibrate_detectors.py†L475-L520】

When interval ranges are supplied (for example to mark the precise onset and clearance of an ``OZ`` event) the calibrator clamps them to the micro-sector boundaries and uses them as positive references.【F:tools/calibrate_detectors.py†L536-L590】 Omitting intervals while keeping the boolean flag tells the tool to treat the entire micro-sector as positive for that detector.

## Interpreting precision/recall/F1 and FP/min

Each detector/parameter set combination is evaluated against the labelled micro-sectors, matching predicted event intervals to the supplied ground truth using an intersection-over-union threshold. True/false positives and false negatives are counted at the interval level so short-lived detections do not inflate results.【F:tools/calibrate_detectors.py†L968-L1053】 Precision, recall, and F1 score are derived from the confusion totals, while false positives per minute use the total labelled duration as the denominator.【F:tools/calibrate_detectors.py†L1031-L1046】 When ranking the candidates, the selector enforces the ``--fp_per_min_max`` ceiling (default 0.5) and prioritises the highest F1, then recall, then precision, breaking ties with the lowest FP/min value.【F:tools/calibrate_detectors.py†L1056-L1072】 The generated ``curves/*.csv`` files retain the entire sweep so you can audit the precision-recall trade-offs per combination.【F:tools/calibrate_detectors.py†L1164-L1202】

The Markdown report emitted as ``report.md`` summarises the best configuration per operator/class/car/compound tuple, echoing the precision/recall/F1/FP/min metrics alongside the concrete parameter choices.【F:tools/calibrate_detectors.py†L1232-L1289】 Use these artefacts to validate that the selected thresholds align with the behaviour observed in :doc:`telemetry` captures.

## YAML cascade and deployment

The calibrator serialises the winning parameter sets into ``best_params.yaml``. Each detector is materialised under its function name and nests overrides by class, car, and compound before falling back to global defaults. Class-level overrides live under ``classes/<class>/defaults`` or ``classes/<class>/compounds/<compound>``, car-specific entries under ``cars/<car>/…``, and universal compound tweaks under ``compounds/<compound>``. When none of the specific keys apply, the runtime consumes ``defaults``.【F:tools/calibrate_detectors.py†L1106-L1161】 This mirrors the loader precedence described in :doc:`detection_config_plan`, which expects calibrated payloads at ``<pack_root>/config/detection.yaml`` and resolves class/compound overrides ahead of the global defaults.【F:docs/detection_config_plan.md†L160-L208】

After reviewing the generated ``best_params.yaml`` and ``report.md``, copy the contents into the destination pack so that ``config/detection.yaml`` inherits the calibrated sections. The runtime will automatically discover the updated YAML thanks to the search order outlined in :doc:`detection_config_plan`. Share the ``curves`` and ``confusion`` CSV directories with analysts so they can trace precision/recall evolution for each detector and car class.

## Related guides

* :doc:`telemetry` – configuring Live for Speed broadcasters so captures contain the required signals.
* :doc:`operators_mapping` – understanding which operators consume the calibrated thresholds.
* :doc:`detection_config_plan` – lifecycle of detection overrides inside packs.
* :doc:`cli` – tooling to ingest RAF sessions and replay them into JSONL runs.
