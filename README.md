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

Kick the tyres with `make quickstart`, which runs the bundled scenario and
writes reports to `examples/out/`. The [installation guide](docs/index.md#installation)
covers prerequisites, dependency extras and how the dataset helper resolves the
sample telemetry before the script executes.

## Telemetry overview

Native Live for Speed broadcasters power every metric in the toolkit. The
[telemetry requirements](docs/index.md#telemetry-requirements) section walks
through the OutSim/OutGauge configuration, extended payload flags and the
fusion layer used by the CLI and HUD.

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

Find the complete manual—including CLI guides, API references and workflow
walkthroughs—in the [MkDocs portal](docs/index.md#resources).

## Branding and relationship with the theory

Read the [branding and terminology](docs/index.md#branding-and-terminology)
section to understand how the theory, toolkit and configuration namespaces stay
aligned.

## Development

See `pyproject.toml` for requirements and `docs/DEVELOPMENT.md` for the full
verification flow (`ruff`, `mypy --strict`, `pytest`).

## Continuous integration

The CI workflow runs automatically for pushes to `main` and for all pull
requests. Required status checks can be configured in the repository settings
to ensure the workflow succeeds before merging.
