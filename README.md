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
| [Examples gallery](docs/examples_gallery.md) | Overview of the automation scripts under `examples/`. |

## Quickstart

### Prerequisites

- Python 3.9 or newer on `PATH`.
- Scientific dependencies: `numpy>=1.24,<2.0` and `pandas>=1.5,<3.0`.
- A telemetry dataset in `data/` (the repo ships CSV, RAF and Replay Analyzer
  bundles for testing).

### Install dependencies

```bash
# Minimal environment to run the CLI and examples
python -m pip install -r requirements.txt

# Full development environment (linters, mypy, pytest)
python -m pip install -r requirements-dev.txt

# Makefile shortcuts
make install      # base dependencies only
make dev-install  # base + development tooling
```

Prefer an editable install? Run `pip install -e .` after setting up the
requirements.

### Execute the end-to-end flow

```bash
make quickstart
```

This invokes `examples/quickstart.sh` to generate artefacts in `examples/out/`.

## Telemetry overview

TNFR × LFS reads exclusively from native Live for Speed broadcasters. Enable
OutSim/OutGauge/Insim in the simulator and consult the
[telemetry guide](docs/telemetry.md) for the complete signal breakdown and brake
thermal proxy details.

## Benchmarks

Profile ΔNFR/ν_f caches with the optional benchmark extra or the Makefile helper.
See the [benchmark guide](docs/benchmarks.md) for installation steps, baseline
numbers and tuning tips.

## CLI deep dives

Looking for configuration templates, Pareto sweeps or the live HUD walkthrough?
Start with the [advanced workflows](docs/advanced_workflows.md) and pair them
with the [CLI deep dive](docs/cli_deep_dive.md) alongside the primary
[CLI guide](docs/cli.md).

## Documentation

The complete documentation lives under `docs/` and is published with MkDocs.

- [`docs/DESIGN.md`](docs/DESIGN.md): textual summary of the TNFR operations manual.
- [`docs/cli.md`](docs/cli.md): command-line interface guide.
- [`docs/setup_equivalences.md`](docs/setup_equivalences.md): link TNFR metrics to
  setup adjustments.

## Branding and relationship with the theory

- **TNFR** refers solely to the original theory: the conceptual framework that
  defines the EPI, the ΔNFR/ΔSi metrics and the resonant principles.
- **TNFR × LFS** identifies the software toolkit that implements and automates
  those principles.
- The CLI (`tnfr-lfs`), Python modules (`tnfr_lfs`) and configuration paths
  (`tnfr-lfs.toml`) retain the hyphen for backward compatibility.

## Development

See `pyproject.toml` for requirements and `docs/DEVELOPMENT.md` for the full
verification flow (`ruff`, `mypy --strict`, `pytest`).
