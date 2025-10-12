# TNFR × LFS Toolkit

TNFR × LFS operationalises the **Fractal-Resonant Nature Theory** inside the
**Live for Speed** simulator so engineering teams can capture telemetry, analyse
driver sessions and surface setup recommendations without leaving the paddock.

## Key features

- Live ΔNFR HUD and capture pipeline wired directly to Live for Speed OutSim/
  OutGauge streams.
- Automated baselines, analysis, suggestions and reporting from a single CLI.
- Extensible exporters (JSONL, Markdown, HTML) and configuration packs for
  repeatable engineering workflows.
- Benchmark suite and reproducible examples for regression testing new ideas.

## Try it now

```bash
make quickstart
```

The target runs the bundled quickstart scenario, producing JSON and Markdown
reports under `examples/out/`.

## Navigation

| Resource | Description |
| --- | --- |
| [Documentation index](docs/index.md) | MkDocs entry point for the full manual. |
| [Beginner quickstart](docs/tutorials.md) | Step-by-step guide for installation and the sample dataset. |
| [Advanced workflows](docs/advanced_workflows.md) | Robustness checks, Pareto sweeps and A/B comparisons. |
| [API reference](docs/api_reference.md) | Python API surface with module details. |
| [Examples gallery](docs/examples.md) | Overview of the automation scripts under `examples/`. |

## Quickstart

### Prerequisites

- Python 3.9 or newer on `PATH`.
- Scientific dependencies: `numpy>=1.24,<2.0` and `pandas>=1.5,<3.0`.
- Telemetry samples shipped with the repo. The quickstart resolves them
  automatically via `tnfr_lfs.examples.quickstart_dataset.dataset_path()`. Place
  custom bundles under `src/tnfr_lfs/resources/data/` or override the helper if
  you need different inputs.

### Install dependencies

```bash
# Minimal environment to run the CLI and examples
pip install .

# Full development environment (linters, mypy, pytest)
pip install .[dev]

# Optional dominant-frequency acceleration (SciPy Goertzel helper)
pip install .[spectral]

# Makefile shortcuts
make install      # base dependencies only
make dev-install  # base + development tooling
```

Prefer an editable install? Run `pip install -e .` and combine extras as
needed (for example, `pip install -e .[dev]`).

### Execute the end-to-end flow

```bash
make quickstart
```

This invokes `examples/quickstart.sh` to generate artefacts in `examples/out/`.
The script loads the packaged dataset through
`tnfr_lfs.examples.quickstart_dataset.dataset_path()`, so no manual copying is
required for the bundled samples.

Explore the [Examples gallery](docs/examples.md) for additional runnable
scripts that demonstrate ingestion, exporting and recommendation workflows.

> **API note:** Live (`OutSim`/`OutGauge`/`InSim`) and offline (RAF, Replay
> Analyzer, profiles) ingestion helpers live under the consolidated
> `tnfr_lfs.ingestion` namespace (for example
> `tnfr_lfs.ingestion.live` and `tnfr_lfs.ingestion.replay`). Use these
> modules to wire capture pipelines directly into the analytics stack.

## Telemetry overview

TNFR × LFS reads exclusively from native Live for Speed broadcasters. Enable
OutSim/OutGauge/Insim in the simulator and consult the
[telemetry guide](docs/telemetry.md) for the complete signal breakdown and brake
thermal proxy details. Developers wiring direct UDP ingest pipelines can jump to
the [`TelemetryFusion` API reference](docs/api_reference.md#tnfr_lfsingestionlivetelemetryfusion)
to see how OutSim/OutGauge packets are combined and how calibration packs slot
into the workflow.

## Benchmarks

Profile ΔNFR/ν_f caches with the optional benchmark extra or the Makefile helper.
Pair it with the new ``spectral`` extra when you want to exercise the Goertzel
accelerator for dominant frequency lookups. See the [benchmark guide](docs/benchmarks.md)
for installation steps, baseline numbers and tuning tips.

## CLI deep dives

Looking for configuration templates, Pareto sweeps or the live HUD walkthrough?
Start with the [advanced workflows](docs/advanced_workflows.md) and pair them
with the [CLI deep dive](docs/cli_deep_dive.md) alongside the primary
[CLI guide](docs/cli.md).

## Documentation

The complete documentation lives under `docs/` and is published with MkDocs.

- Install the documentation toolchain (MkDocs, plugins, and AutoAPI) with:

  ```bash
  pip install .[dev]
  ```

- [`docs/DESIGN.md`](docs/DESIGN.md): textual summary of the TNFR operations manual.
- [`docs/cli.md`](docs/cli.md): command-line interface guide.
- [`docs/setup_equivalences.md`](docs/setup_equivalences.md): link TNFR metrics to
  setup adjustments.
- [`config/plugins.toml`](config/plugins.toml): plugin discovery and profile
  templates consumed by the CLI helpers.

## Branding and relationship with the theory

- **TNFR** refers solely to the original theory: the conceptual framework that
  defines the EPI, the ΔNFR/ΔSi metrics and the resonant principles.
- **TNFR × LFS** identifies the software toolkit that implements and automates
  those principles.
- The CLI (`tnfr_lfs`), Python modules (`tnfr_lfs`) and configuration paths
  (`tnfr_lfs.toml`) retain their respective naming conventions for
  compatibility.

## Development

See `pyproject.toml` for requirements and `docs/DEVELOPMENT.md` for the full
verification flow (`ruff`, `mypy --strict`, `pytest`).

## Continuous integration

The CI workflow runs automatically for pushes to `main` and for all pull
requests. Required status checks can be configured in the repository settings
to ensure the workflow succeeds before merging.
