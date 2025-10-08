# Test fixtures

The unit and integration suite reuses a couple of deterministic datasets to
exercise the EPI extractor, segmentation heuristics and CLI pipelines.  This
directory documents how to rebuild them if the telemetry model changes in the
future.

## `data/BL1_XFG_baseline.csv`

* Reference dataset for the quickstart flow and the CLI tests.
* Mirrors the 17-sample synthetic stint used under `tests/data` but lives at
  the repository root for user-facing workflows.
* Keep the same headers to preserve compatibility with the
  `examples/quickstart.sh` script and the typed functions in
  `typing_targets.quickstart_dataset`.
* When you need real telemetry for tutorials or manual verification, also rely
  on the RAF capture ``data/test1.raf`` included at the root; the CLI detects it
  automatically through the built-in RAF parser.

To regenerate the file you can copy the contents of
`tests/data/synthetic_stint.csv` or rerun the script shown below, saving the
result to `data/BL1_XFG_baseline.csv`.

## `data/synthetic_stint.csv`

* 17-sample telemetry stint that contains two distinct cornering events.
* Columns follow the `TelemetryRecord` schema used by the acquisition layer.
* Generated manually so that the segmentation heuristics detect brake and
  support events while the ΔNFR baseline produces non-zero node deltas.

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

## `data/car_track_profiles.json`

* Simplified library of ΔNFR tolerances per car model and track.
* Tests load the JSON into `ThresholdProfile` instances to validate the
  recommendation engine context resolution.

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

The `mini_track_pack` fixture synthesises a compact dataset with:

* `data/tracks/AS.toml` describing the `AS3` layout and its extras.
* `data/track_profiles/p_test_combo.toml` containing deterministic weights and hints.
* `modifiers/combos/demo_profile__p_test_combo.toml` with scale factors and hint overrides.
* `data/cars/DEMO.toml` so the CLI can resolve the modifier via the car profile.

When adding new test circuits or modifiers reuse the fixture by copying its
structure into new files under the temporary directory. Prefer short manifest
names and floats with few decimals so assertions stay readable. For additional
layouts extend the TOML with `[config.XY#]` tables; for more profiles write extra
`*.toml` files into the generated directories and return their identifiers from
the fixture as new dataclass fields.

## Acceptance orchestration fixtures

The `acceptance_bundle_series`, `acceptance_records` and
`acceptance_microsectors` fixtures exercise the nodal stage, window occupancy
metrics and modal coupling/resonance operators without relying on the full EPI
extractor.  They feed deterministic bundles and segmentation metadata to
`orchestrate_delta_metrics` so that the acceptance tests can validate:

* Monotonic smoothing of the Sense Index time series.
* Pairwise coupling per node and global resonance values derived from the
  smoothed series.
* ν_f weighting effects over the entropy-aware Sense Index calculation.
* Convergence of the recursive memory and mutation operators.

Run the scenarios with:

```bash
pytest tests/test_acceptance_pipeline.py
```
