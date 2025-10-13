# Test fixtures

The unit and integration suite reuses a couple of deterministic datasets to
exercise the EPI extractor, segmentation heuristics and CLI pipelines.  This
directory documents how to rebuild them if the telemetry model changes in the
future.

## Detecting similar pytest functions

The `make report-similar-tests` helper runs `tools/report_similar_tests.py` and
produces `tests/_report/similar_tests.json`. The script parses every
`tests/**/*.py` module, normalises the AST of functions that start with
`test_`, and computes a similarity score between structurally equivalent
implementations. Pairs whose similarity ratio exceeds the configured threshold
(`0.9` by default) are emitted in the JSON report. Each entry lists both test
identifiers (`path::qualified_name`) alongside the measured similarity so you
can decide whether further parametrisation would remove duplication.

The most recent snapshot shows only one matching pair around the operator
normalisation helpers, confirming that previous parametrisation work kept the
suite DRY. Re-run the command after adding new tests to ensure fresh cases do
not reintroduce duplicated logic.

## `src/tnfr_lfs/resources/data/BL1_XFG_baseline.csv`

* Canonical dataset for the quickstart walkthrough and the CLI regression
  suite.
* Mirrors the 17-sample synthetic stint under `tests/data` and stores the
  canonical copy inside the packaged resources so user-facing workflows can
  locate it via `tnfr_lfs.examples.quickstart_dataset.dataset_path()`.
* Keep the header row in sync with the expectations in
  `examples/quickstart.sh` and in
  `tnfr_lfs.examples.quickstart_dataset.dataset_columns()`; update the three
  sources together if the telemetry schema changes.
* For tutorials or manual checks you can also fall back to the RAF capture
  `src/tnfr_lfs/resources/data/test1.raf`; the CLI auto-detects it through the native RAF
  parser.

To regenerate the file you can copy the contents of
`tests/data/synthetic_stint.csv` or rerun the script shown below, saving the
result to `src/tnfr_lfs/resources/data/BL1_XFG_baseline.csv`.

## `src/tnfr_lfs/resources/data/synthetic_stint.csv`

* Seventeen-sample telemetry stint that captures two distinct cornering
  segments.
* Columns mirror the `TelemetryRecord` schema used by the acquisition layer.
* Crafted so the segmentation heuristics flag brake/support phases and the
  ΔNFR baseline produces non-zero node deltas for every subsystem.

To regenerate the file after adjusting the telemetry schema:

```python
from pathlib import Path
import csv

rows = [
    # timestamp, vertical_load, slip_ratio, lateral_accel,
    # longitudinal_accel, nfr, si
    (0.0, 5000, 0.03, 0.3, 0.2, 500, 0.88),
    (0.2, 5050, 0.03, 0.4, 0.1, 502, 0.87),
    (0.4, 5100, 0.04, 0.9, -0.1, 505, 0.86),
    (0.6, 5400, 0.05, 1.4, -0.35, 510, 0.83),
    (0.8, 5800, 0.06, 1.9, -0.55, 518, 0.79),
    (1.0, 6100, 0.07, 2.2, -0.70, 525, 0.74),
    (1.2, 5900, 0.06, 1.8, -0.45, 520, 0.77),
    (1.4, 5500, 0.05, 1.3, -0.15, 514, 0.80),
    (1.6, 5200, 0.04, 0.7, 0.15, 507, 0.83),
    (1.8, 5250, 0.04, 0.8, 0.10, 508, 0.84),
    (2.0, 5400, 0.05, 1.3, -0.20, 512, 0.82),
    (2.2, 5800, 0.06, 1.8, -0.50, 518, 0.78),
    (2.4, 6200, 0.07, 2.3, -0.75, 526, 0.72),
    (2.6, 6000, 0.07, 2.0, -0.40, 522, 0.75),
    (2.8, 5600, 0.06, 1.4, -0.05, 515, 0.78),
    (3.0, 5250, 0.05, 0.9, 0.10, 508, 0.81),
    (3.2, 5100, 0.04, 0.4, 0.20, 504, 0.84),
]

with Path("tests/data/synthetic_stint.csv").open("w", newline="", encoding="utf8") as fh:
    writer = csv.writer(fh)
    writer.writerow([
        "timestamp",
        "vertical_load",
        "slip_ratio",
        "lateral_accel",
        "longitudinal_accel",
        "nfr",
        "si",
    ])
    writer.writerows(rows)
```

## `src/tnfr_lfs/resources/data/car_track_profiles.json`

* Compact catalogue of ΔNFR tolerances scoped by car model and track.
* Loaded into `ThresholdProfile` instances so the tests can validate context
  resolution inside the recommendation engine.

To tweak tolerances create/update the dictionary and dump it with standard
`json` tools.  The current structure is:

```python
import json
from pathlib import Path

profiles = {
    "FZR": {
        "AS5": {
            "entry_delta_tolerance": 0.9,
            "apex_delta_tolerance": 0.6,
            "exit_delta_tolerance": 1.2,
            "piano_delta_tolerance": 1.5,
        }
    },
    "FO8": {
        "KY3": {
            "entry_delta_tolerance": 0.7,
            "apex_delta_tolerance": 0.5,
            "exit_delta_tolerance": 1.0,
            "piano_delta_tolerance": 1.2,
        }
    },
}

Path("tests/data/car_track_profiles.json").write_text(
    json.dumps(profiles, indent=2),
    encoding="utf8",
)
```

## Mini track pack fixtures

The `mini_track_pack` fixture synthesises a compact pack with deterministic
metadata:

* `data/tracks/AS.toml` describes the `AS3` layout and its auxiliary sections.
* `data/track_profiles/p_test_combo.toml` provides fixed weights and hints.
* `modifiers/combos/demo_profile__p_test_combo.toml` locks the scale factors and
  overrides consumed by the recommendation engine.
* `data/cars/DEMO.toml` allows the CLI to resolve the modifier from the car
  profile.

When adding test circuits or modifiers, replicate this structure inside the
temporary directory built by the fixture. Keep manifest names concise and round
floats to a few decimals so assertions remain readable. To model extra layouts,
extend the TOML file with `[config.XY#]` tables; to provide more profiles,
create additional `*.toml` files in the generated directories and expose their
identifiers as new dataclass fields returned by the fixture.

## Acceptance orchestration fixtures

The `acceptance_bundle_series`, `acceptance_records`, and
`acceptance_microsectors` fixtures exercise the nodal stage, window-occupancy
metrics, and modal coupling/resonance operators without invoking the full EPI
extractor. They feed deterministic bundles and segmentation metadata into
`orchestrate_delta_metrics`, allowing the acceptance tests to validate:

* Monotonic smoothing of the Sense Index time series.
* Per-node coupling and overall resonance derived from the smoothed series.
* ν_f weighting effects applied to the entropy-aware Sense Index calculation.
* Convergence of the recursive memory and mutation operators.

Run the scenarios with:

```bash
pytest tests/test_acceptance_pipeline.py
```
