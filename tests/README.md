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
can decide whether further parametrisation would remove duplication. The
Makefile target automatically removes `tests/_report/*.json` after printing the
summary so repeated runs keep the working tree clean. The `_report` directory is
dedicated to transient artefacts; its JSON files are intentionally left
unversioned so local experiments and the CI sweep never dirty the tree.

Use `make report-similar-tests-sweep` to execute the helper at thresholds 0.90,
0.88, and 0.85 in one go. Capture the counts printed for each threshold and
attach them to your pull request description so reviewers can see whether new
tests introduce fresh similarities. Esta barrida de tres umbrales cubre todo el
flujo: 0.90 actúa como corte duro, 0.88 destaca duplicados incipientes y 0.85
ensancha la red para auditorías exploratorias.

| Command | Umbral | Propósito |
| --- | --- | --- |
| `make report-similar-tests` | 0.90 | Hard gate local alineado con el corte obligatorio. |
| `python tools/report_similar_tests.py --threshold 0.88` | 0.88 | Aviso manual para vigilar duplicados sin frenar iteraciones. |
| `python tools/report_similar_tests.py --threshold 0.85` | 0.85 | Aviso amplio para detectar patrones candidatos a parametrizar. |

El workflow de CI [Similar tests](../.github/workflows/similar-tests.yml)
invoca el script a 0.90 dentro del job **Detect similar tests (0.90)** y falla
si encuentra pares sobre el umbral duro. Un job separado genera los informes en
0.90 y 0.85, y publica el artefacto `similar-tests-reports`. Para consultarlo,
abre la ejecución del workflow en GitHub, entra al job **Similar test reports**
y descarga el artefacto comprimido desde la sección *Artifacts*; allí encontrarás
los JSON listos para su inspección, mientras que el barrido a 0.88 queda para
ejecuciones locales con los comandos anteriores.

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

## Parametrised factories and case tables

New tests frequently rely on shared factories to keep telemetry scenarios
readable. When extending them:

* Prefer `ids=` arguments that echo the scenario name or failure message.
  Descriptive IDs keep `pytest -k` filters discoverable and aid CI triage.
* Describe the behavioural gap each new case covers (new branch, regression,
  or integration between helpers) in a one-line comment close to the table.
* Update any derived ID sequences (for example `TYRE_BALANCE_IDS`) whenever
  you append a case so parametrised runs stay in sync.

### Replay CSV bundle helpers

`tests/conftest.py` exposes a `csv_bundle` fixture that emits a temporary ZIP
bundle compatible with :class:`tnfr_lfs.ingestion.offline.ReplayCSVBundleReader`.
The builder writes a minimal `time.csv`/`speed.csv` pair unless you request
missing channels or a renamed distance column. Helper functions under
`tests/helpers/replay_bundle.py`—such as `read_reference_rows` and
`monkeypatch_row_to_record_counter`—provide reference data and instrumentation
used throughout `tests/test_replay_csv_bundle.py`.

To add a regression scenario, append an entry to the
`pytest.mark.parametrize("bundle_kwargs", ...)` table in
`tests/test_replay_csv_bundle.py` with a concise slugged `id`. Keep `bundle_kwargs`
focused on the failure condition and reuse existing expectation helpers:

```python
@pytest.mark.parametrize(
    ("bundle_kwargs", "expected"),
    [
        pytest.param({"missing": "speed"}, "missing speed.csv", id="missing-speed"),
        pytest.param({"with_distance": False}, "distance column", id="wrong-distance"),
    ],
)
def test_reader_errors(csv_bundle, bundle_kwargs, expected):
    reader = ReplayCSVBundleReader(csv_bundle(**bundle_kwargs))
    with pytest.raises(ValueError, match=expected):
        reader.to_dataframe()
```

### CLI configuration matrix

The `_CLI_CONFIG_CASES` catalogue in `tests/conftest.py` feeds the
`cli_config_case` fixture, which normalises TOML snippets and expected
configuration payloads exercised by `tests/test_cli.py` and
`tests/test_cli_io.py`. Case IDs double as `pytest` parametrisation identifiers,
so use kebab-cased slugs that describe the user-facing configuration. The
`baseline_cli_runner` fixture in the same module drives the `baseline` command
and automatically wires in the synthetic stint bundled under
`tnfr_lfs.examples.quickstart_dataset`.

When adding a new CLI regression:

```python
_CLI_CONFIG_CASES["custom-cache-horizon"] = CliConfigCase(
    toml_text="""
    [tool.tnfr_lfs.performance]
    cache_enabled = true
    telemetry_buffer_size = 96
    """,
    expected_sections={
        "performance": {
            **CacheOptions(enable_delta_cache=True, telemetry_cache_size=96).to_performance_config(),
            "telemetry_buffer_size": 96,
        }
    },
)
```

Reference the case via `@pytest.mark.parametrize("cli_config_case", [...],
ids=[...], indirect=True)` and document how it exercises
`tnfr_lfs.cli.config.load_cli_config` or `tnfr_lfs.cli.run_cli`.

### Rule scenario cases

Rule-oriented parametrisation lives in `tests/test_recommender.py`. The
`rule_scenario_factory` fixture assembles `(RuleContext, Goal, Microsector)`
tuples, while the `RuleCase` dataclass declares expected deltas for rules such
as :class:`tnfr_lfs.recommender.rules.TyreBalanceRule`,
:class:`tnfr_lfs.recommender.rules.ParallelSteerRule`, and
:class:`tnfr_lfs.recommender.rules.LockingWindowRule`. Each case description
feeds the corresponding `ids` list (for example `TYRE_BALANCE_IDS`), so new
entries must use unique, kebab-cased summaries of the scenario under test.

Additions should document which branch of the rule they target (threshold,
message rendering, suppression, …) and keep payload overrides narrow. A minimal
extension looks like:

```python
TYRE_BALANCE_CASES.append(
    RuleCase(
        description="suppresses-when-session-hints-lock-out",
        context_overrides={"session_hints": {"tyre_balance_override": "skip"}},
        expected_parameters=(),
    )
)
TYRE_BALANCE_IDS[:] = [case.description for case in TYRE_BALANCE_CASES]
```

### Synthetic window metrics

Window-metric parametrisation focuses on :func:`tnfr_lfs.metrics.window.compute_window_metrics`.
The `synthetic_window_factory` fixture in `tests/conftest.py` builds
deterministic telemetry windows, and helpers in `tests/helpers/steering.py`
(`build_parallel_window_metrics`) supply richer Ackermann/steering scenarios.
`tests/test_metrics.py` uses declarative tuples that pair factory kwargs with
expected attribute checks, naming each case through the `ids=` argument.

When extending the suite, favour inputs that toggle a new branch inside
`WindowMetrics` or its brake headroom sub-structure and express expectations via
`pytest.approx` or symbolic markers (`"isnan"`, `"lt"`, …). Example:

```python
WINDOW_CASES = [
    *WINDOW_CASES,
    (
        {"profile": "fade-recovery", "brake_pressure": 0.4},
        {"ventilation_alert": "recovered", "fade_ratio": pytest.approx(0.1)},
    ),
]
WINDOW_CASE_IDS = [*WINDOW_CASE_IDS, "fade-recovery"]
```

Cross-reference the impacted modules when committing (for instance, mention
`tnfr_lfs.metrics.window`) so reviewers can align the telemetry fixture with the
implementation branch under scrutiny.
