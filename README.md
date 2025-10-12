# TNFR × LFS Toolkit

TNFR × LFS brings the **Fractal-Resonant Nature Theory** into the **Live for
Speed** simulator so race engineers can capture telemetry, analyse driver
sessions, and surface setup recommendations without leaving the paddock.

## Key features

- Live ΔNFR HUD and capture pipeline wired directly to Live for Speed
  OutSim/OutGauge streams.
- Automated baselines, analysis, suggestions, and reporting from a single CLI.
- Extensible exporters (JSONL, Markdown, HTML) and configuration packs for
  repeatable engineering workflows.
- Benchmark suite and reproducible examples for regression testing new ideas.

## Getting started

1. **Install the toolkit** – follow the detailed instructions in the
   [installation guide](docs/index.md#installation) to choose between base,
   developer, or extras-enabled setups.
2. **Configure telemetry** – enable the Live for Speed OutSim/OutGauge
   broadcasters as described in the
   [telemetry requirements](docs/index.md#telemetry-requirements).
3. **Run the quickstart scenario** – execute `make quickstart` to generate JSON
   and Markdown reports for the bundled dataset, then dive deeper with the
   [workflow guides](docs/index.md#resources).

## Learn more

The [documentation index](docs/index.md) hosts the complete manual, including
installation details, telemetry references, CLI walkthroughs, API docs, and
advanced workflow guides.
