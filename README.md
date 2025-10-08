# TNFR × LFS Toolkit

TNFR × LFS is the analysis and automation toolkit that operationalises the **Fractal-Resonant Telemetry Analytics for Live for Speed** proposal and the **Fractal-Resonant Nature Theory (TNFR)**. While the theory describes the conceptual foundations (symbols, structural operators and the ΔNFR/ΔSi metrics), the package provides reproducible implementations to instrument those ideas within telemetry workflows.

## Quickstart

### Prerequisites
- Python 3.9 or newer installed and available on `PATH`.
- Scientific dependencies: `numpy>=1.24,<2.0` and `pandas>=1.5,<3.0`. They are
  installed automatically when running `pip install .[dev]`, but ensure your
  environment can build/install wheels for both packages.
- A telemetry dataset under `data/`. The repository includes the synthetic CSV
  `data/BL1_XFG_baseline.csv`, the real RAF capture `data/test1.raf`, and the
  bundle exported from Replay Analyzer `data/test1.zip`, so you can rehearse
  flows with any of the supported formats.

### Dependency installation

To simplify the setup the repository publishes `pip`-compatible requirement files:

```bash
# Minimal environment to run the CLI and the examples
python -m pip install -r requirements.txt

# Full development environment (includes linters, mypy, and pytest)
python -m pip install -r requirements-dev.txt

# You can also rely on the shortcuts declared in the Makefile
make install      # installs only the base dependencies
make dev-install  # installs base dependencies + development tooling
```

If you prefer an editable environment, execute `pip install -e .` after
installing the dependencies to link the package locally.

### Execution
Run the simulated end-to-end flow with:

```bash
make quickstart
```

The target invokes `examples/quickstart.sh`, which writes artefacts in `examples/out/` from the dataset and produces JSON and Markdown reports ready to inspect.

The CLI accepts CSV, RAF, or Replay Analyzer bundles interchangeably. When working with CSV use the simulated mode (`tnfr-lfs baseline output.jsonl --simulate data/BL1_XFG_baseline.csv`); for RAF simply point the subcommand at the file (`tnfr-lfs baseline output.jsonl data/test1.raf --format jsonl`); and Replay Analyzer bundles (directory or ZIP, such as `data/test1.zip`) are ingested via `--replay-csv-bundle`. All options generate the same baseline and the remainder of the flow (`analyze`, `suggest`, `report`, `write-set`) auto-detects the format through the corresponding parser.

### Live HUD

To display the ΔNFR HUD inside Live for Speed enable the broadcasters in the simulator (`/outsim 1 127.0.0.1 4123`, `/outgauge 1 127.0.0.1 3000`, and `/insim 29999`) and execute:

```bash
tnfr-lfs osd --host 127.0.0.1 --outsim-port 4123 --outgauge-port 3000 --insim-port 29999
```

The panel rotates three pages (corner/phase status, nodal contributions, and setup plan) that fit in a 40×16 `IS_BTN` button. You can move it with `--layout-left/--layout-top` if another mod occupies the same area.

### Required telemetry

All TNFR metrics and indicators (`ΔNFR`, the nodal projections `∇NFR∥`/`∇NFR⊥`, `ν_f`, `C(t)`, and derivatives) are computed solely from the telemetry exposed by OutSim and OutGauge in Live for Speed. Ensure both broadcasters are enabled and extend the OutSim block with `OutSim Opts ff` in `cfg.txt` (or via `/outsim Opts ff` followed by `/outsim 1 …`) to include the player ID, driver inputs, and the wheel packet that contains the forces, loads, and deflections consumed by the toolkit’s telemetry fusion module.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】 The global `slip_ratio` and `slip_angle` emitted by TNFR × LFS come directly from those per-wheel channels: the decoded OutSim readings are averaged (load-weighted when available) and the historical kinematic model is used only when a packet arrives without valid samples.【F:tnfr_lfs/acquisition/fusion.py†L111-L175】【F:tnfr_lfs/acquisition/fusion.py†L242-L291】

The setup indicators (such as the Contact Patch Health Index and the thermal operator’s ΔP/Δcamber adjustments) are driven exclusively by that native telemetry: they use the per-wheel slips, forces, and loads exposed by OutSim together with the engine and pedal states transmitted by OutGauge. No external heuristics or synthetic data are applied, ensuring every recommendation comes directly from what Live for Speed broadcasts in real time.【F:tnfr_lfs/core/metrics.py†L200-L284】【F:tnfr_lfs/core/operators.py†L731-L830】

#### Key signals per metric

- **ΔNFR (nodal gradient) / ∇NFR∥/∇NFR⊥ (projections)** – combine the Fz loads (and their ΔFz variations), longitudinal and lateral accelerations, wheel forces and suspension deflections derived from the OutSim wheel packet with the engine regime, pedal position and ABS/TC lights from OutGauge to contextualise gradient distribution. Use the direct `Fz`/`ΔFz` channels when you need to evaluate absolute forces before adjusting the setup.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】
- **ν_f (natural frequency)** – requires the load distribution, slip ratios/angles, and speed/yaw rate coming from OutSim together with the driver style signal (`throttle`, `gear`) to tune the category and spectral window of each node.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L648-L710】
- **C(t) (structural coherence)** – derives from the nodal ΔNFR distribution and the ν_f bands, so it depends on the same OutSim fields, on the `mu_eff_*` adhesion coefficients computed from lateral/longitudinal accelerations, and on the ABS/TC flags delivered by OutGauge that the fusion module translates into lockup events.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】【F:tnfr_lfs/core/coherence.py†L65-L125】
- **Ventilation / brake fade** – uses the caliper/disc temperatures published by OutGauge when available; if the broadcaster sends the marker (`0 °C`) the fusion module falls back to the thermal proxy to maintain a finite series before feeding the fade indicators.【F:tnfr_lfs/acquisition/fusion.py†L248-L321】【F:tnfr_lfs/core/metrics.py†L860-L982】

### Brake thermal proxy

Live for Speed does not expose real brake temperatures in the base configuration; the extended OutGauge block (`OG_EXT_BRAKE_TEMP`) only transmits them when the car/version supports the data and the broadcaster is enabled. The TNFR × LFS thermal proxy consumes those native readings whenever they arrive (ignoring the `0 °C` markers sent by OutGauge when the sensor is disabled) and, when direct telemetry does not exist, maintains an estimated series so the pipeline continues to offer fade and ventilation indicators.【F:tnfr_lfs/acquisition/fusion.py†L248-L321】 The calculation integrates the per-wheel mechanical work `m·a·v` to inject heat into the active caliper/disc—weighted by the instantaneous normal load—and applies convective dissipation proportional to the square root of speed and to a configurable ventilation coefficient. The integration accumulates and bounds the series between ambient temperature and a 1200 °C ceiling, ensuring the analytical modules operate with finite values even without OutGauge.【F:tnfr_lfs/analysis/brake_thermal.py†L9-L122】

The proxy parameters (reference ambient temperature, effective thermal capacity, braking efficiency, convective coefficients and pedal threshold) are declared in the `[thermal.brakes]` block of `config/global.toml` and admit per-car overrides in `data/cars/*.toml`. The `mode` key controls whether the proxy runs in `auto`, `off`, or `force`, and it can also be forced at runtime via the `TNFR_LFS_BRAKE_THERMAL` environment variable.【F:config/global.toml†L13-L20】【F:tnfr_lfs/acquisition/fusion.py†L186-L321】

#### Telemetry limitations

- The proxy only fills gaps when OutGauge omits the temperature or publishes the `0 °C` marker; if the simulator transmits erratic values they are preserved as-is to avoid desynchronising the real reading with the estimate.
- In cars without brake sensors exposed by OutGauge the estimate starts from the energy dissipated in the front/rear wheels reported by OutSim, so short sampling cycles (<10 Hz) may understate instantaneous peaks.
- The estimate does not correct for OutSim/OutGauge network latency; it assumes both streams arrive synchronised within the usual Live for Speed tolerance (one or two packets).

## Documentation

The complete documentation lives under `docs/` and is published with MkDocs.

- [`docs/DESIGN.md`](docs/DESIGN.md): textual summary of the TNFR operations manual. Treat `TNFR.pdf` as a generated artefact; any canonical update must be authored in Markdown to keep diffs readable.
- [`docs/index.md`](docs/index.md): landing page for the TNFR × LFS toolkit documentation.
- [`docs/api_reference.md`](docs/api_reference.md): API reference.
- [`docs/cli.md`](docs/cli.md): command-line interface guide.
- [`docs/setup_equivalences.md`](docs/setup_equivalences.md): correspondence table linking each TNFR metric (projections `∇NFR∥`/`∇NFR⊥`, `ν_f`, `C(t)`) with concrete setup adjustments to prepare coherent sessions.

### CLI configuration template

The `tnfr-lfs.toml` file at the repository root provides a ready-to-use template with the default values documented in [`docs/cli.md`](docs/cli.md). Copy this file to `~/.config/tnfr-lfs.toml` or to your session’s working directory to customise the OutSim/OutGauge/InSim ports, the initial car/track, and the ΔNFR limits per phase before invoking the CLI.

From this release the CLI can also resolve a complete TNFR × LFS pack (car metadata and target profiles) by pointing `paths.pack_root` to the directory containing `config/global.toml`, `data/cars`, and `data/profiles`. The global `--pack-root` option lets you switch packs for a specific run without modifying the TOML file. The `analyze`, `suggest`, and `write-set` results now include the `car` and `tnfr_targets` sections, derived from that pack, so the JSON exports reflect the active TNFR context.

### `pareto` and `compare` subcommands

| Subcommand | Purpose | Recommended exporters | Example |
| --- | --- | --- | --- |
| `pareto` | Sweep the decision space around the generated plan and return the Pareto front with the dominant candidates. | `markdown` for quick inspection or `html_ext` for a navigable report with tables, histograms, and playbook suggestions. | `tnfr-lfs pareto stint.jsonl --car-model FZR --radius 2 --export html_ext > pareto.html` |
| `compare` | Compare two stints or configurations (A/B) by aggregating lap metrics and highlighting the winning variant for the selected metric. | `markdown` for engineering notes or `html_ext` when a dashboard with lap charts and session messages is required. | `tnfr-lfs compare baseline.jsonl variant.jsonl --metric sense_index --export html_ext > abtest.html` |

Both subcommands accept `--export html_ext`, which reuses the extended HTML exporter to embed the Pareto front, A/B summaries, and the playbook suggestions into a single portable file.【F:tnfr_lfs/cli/pareto.py†L23-L117】【F:tnfr_lfs/cli/compare.py†L28-L135】【F:tnfr_lfs/exporters/__init__.py†L1183-L1193】

## Branding and relationship with the theory

- **TNFR** refers solely to the original theory: the conceptual framework that defines the EPI, the ΔNFR/ΔSi metrics, and the resonant principles.
- **TNFR × LFS** identifies the software toolkit that implements and automates those principles. The name uses the "×" symbol to emphasise that the TNFR theory combines with the Load Flow Synthesis (LFS) methodology inside the package.
- The CLI (`tnfr-lfs`), Python modules (`tnfr_lfs`), and configuration paths (`tnfr-lfs.toml`) retain the hyphen for backward compatibility, but they belong to the TNFR × LFS toolkit.

## Development

See `pyproject.toml` for the development requirements and scripts, and
``docs/DEVELOPMENT.md`` for the full verification flow (``ruff``,
``mypy --strict`` and ``pytest``).
