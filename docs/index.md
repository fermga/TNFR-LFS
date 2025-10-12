# TNFR × LFS Documentation

TNFR × LFS (Fractal-Resonant Telemetry Analytics for Live for Speed) is
the lightweight Python toolkit that operationalises the canonical
Telemetry/Theory of the Fractal-Resonant Nature (TNFR) framework
alongside the Load Flow Synthesis methodology. The theory establishes
how to reason about Event Performance Indicators (EPI), ΔFz/ΔSi deltas
and Sense Index signals; the toolkit provides production-ready layers to
apply those ideas in racing telemetry pipelines:

1. **Ingest** – capture telemetry samples from OutSim-compatible
   streams via :class:`tnfr_lfs.ingestion.live.OutSimClient`.
2. **Core analytics** – extract Event Performance Indicators (EPI) and
   compute ΔFz/ΔSi deltas through :class:`tnfr_lfs.core.epi.EPIExtractor`.
3. **Recommendation engine** – map metrics to actionable setup advice
   using the rule engine in :mod:`tnfr_lfs.recommender.rules`.
4. **Exporters** – serialise analysis results to JSON or CSV with the
   functions in :mod:`tnfr_lfs.exporters`.

The project ships with a CLI (`tnfr_lfs`) as well as examples and unit
tests to illustrate the workflow. The ingestion pipeline regression
suite in `tests/test_ingestion.py` exercises the OutSim/OutGauge UDP
clients and fusion helpers end-to-end.【F:tests/test_ingestion.py†L1-L376】

## Getting started

TNFR × LFS targets Python 3.9+ environments and Live for Speed telemetry
streams. The onboarding workflow below is the definitive reference for
setting up the toolkit from scratch.

### Step 1 – Prepare your environment

1. Install **Python 3.9 or later** and ensure it is available on ``PATH``.
   Verify the interpreter with ``python --version`` or ``py --version``
   depending on your platform.
2. Create and activate a virtual environment (``python -m venv .venv`` followed
   by ``source .venv/bin/activate`` or ``.venv\Scripts\activate``) if you
   want to keep the toolkit isolated from other projects.
3. Confirm you have internet access to download dependencies. The analytics
   stack relies on ``numpy>=1.24,<2.0`` and ``pandas>=1.5,<3.0``; ``pip`` will
   resolve them automatically during installation.
4. (Optional) Copy your own telemetry into
   ``src/tnfr_lfs/resources/data/`` when you plan to override the packaged
   samples used by the quickstart workflow.

### Step 2 – Install the toolkit

Choose the install flavour that matches your needs:

- **User install** – minimal CLI and examples:

  ```bash
  pip install .
  ```

- **Developer install** – includes linting, testing, and docs extras:

  ```bash
  pip install .[dev]
  ```

- **Spectral extras** – pull in the Goertzel helper used by the benchmark
  suite:

  ```bash
  pip install .[spectral]
  ```

Make targets provide the same flows when you prefer ``make`` wrappers:

```bash
make install      # base dependencies only
make dev-install  # base + development tooling
```

Editable installs are supported with ``pip install -e .``; append extras such as
``pip install -e .[dev]`` when iterating on the codebase.

### Step 3 – Configure Live for Speed telemetry

1. Launch Live for Speed and open ``cfg.txt`` (Options → Misc → ``cfg.txt``
   or edit the file directly).
2. Enable the OutSim broadcaster with a suitable host/port pair, then append
   the extended options so TNFR × LFS receives the four-wheel payload:

   ```text
   OutSim 1 29999 1000 0.0 0.0 0.0
   OutSim Opts ff
   ```

3. Enable the OutGauge broadcaster and request the extended payload that carries
   tyre temperatures, pressures, and brake data:

   ```text
   OutGauge 1 30000 1
   OutGauge Opts OG_EXT_TYRE_TEMP|OG_EXT_TYRE_PRESS|OG_EXT_BRAKE_TEMP
   ```

   The corresponding in-game commands are ``/outsim Opts ff`` followed by
   ``/outsim 1 …`` and ``/outgauge Opts …`` before ``/outgauge 1 …`` if you
   prefer configuring from the console.
4. Restart the session so Live for Speed loads the new configuration. Refer to
   the [telemetry reference](telemetry.md) for a detailed breakdown of the
   available fields and integration tips.

All TNFR metrics (``ΔNFR``, the nodal projections ``∇NFR∥``/``∇NFR⊥``, ``ν_f``,
``C(t)`` and related indicators) are derived from these native telemetry
streams; the toolkit never fabricates missing inputs. The fusion layer reads
the OutSim/OutGauge packets produced with the configuration above so the HUD,
CLI, and exporters have access to the full data set.【F:tnfr_lfs/ingestion/fusion.py†L93-L200】【F:tnfr_lfs/ingestion/fusion.py†L594-L657】

!!! note "Brake temperature estimation"
    Live for Speed only publishes real brake temperatures when the extended OutGauge payload is enabled; otherwise the stream exposes `0 °C` placeholders. TNFR × LFS consumes those native readings whenever they arrive and seamlessly falls back to the brake thermal proxy to keep fade metrics alive, integrating brake work and convective cooling until fresh data shows up again.【F:tnfr_lfs/ingestion/fusion.py†L248-L321】【F:tnfr_lfs/ingestion/fusion.py†L1064-L1126】
    The CSV reader mirrors that philosophy by preserving optional columns as `math.nan` when OutSim leaves them out, preventing artificial estimates from leaking into the metrics pipeline.【F:tnfr_lfs/ingestion/outsim_client.py†L87-L155】 When the wheel payload is disabled the toolkit now surfaces tyre loads, slip ratios and suspension metrics as “no data” rather than fabricating zeroed values, making it obvious that the telemetry stream is incomplete.【F:tnfr_lfs/ingestion/fusion.py†L93-L200】【F:tnfr_lfs/ingestion/outsim_client.py†L87-L155】

#### Metric field checklist

- **ΔNFR (nodal gradient) and ∇NFR∥/∇NFR⊥ (gradient projections)** – rely on
  per-wheel Fz loads, their ΔFz derivatives, the longitudinal/lateral
  forces, and the suspension deflections reported by OutSim together with
  the engine regime, pedals, and ABS/TC flags provided by OutGauge to
  resolve the nodal gradient.  The ∇NFR∥/∇NFR⊥ projections are components of
  that gradient and do not replace the raw load channels; always cross-check
  recommendations against the `Fz`/`ΔFz` logs when you need absolute forces.
  【F:tnfr_lfs/ingestion/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】
- **ν_f (natural frequency)** – requires load split, slip ratios/angles,
  and yaw rate/velocity from OutSim, plus driver style signals (throttle,
  gear) resolved via OutGauge to tailor node categories and spectral
  windows.【F:tnfr_lfs/ingestion/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L648-L710】
- **C(t) (structural coherence)** – builds on the ΔNFR distribution and
  ν_f bands, leveraging the same OutSim data, the derived `mu_eff_*`
  coefficients, and the ABS/TC flags that OutGauge exposes.【F:tnfr_lfs/ingestion/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】【F:tnfr_lfs/core/coherence.py†L65-L125】
- **Ackermann / slide-catch budgets** – use only the `slip_angle_*`
  channels and `yaw_rate` broadcast by OutSim to measure parallel-steer
  deltas and slide-recovery headroom; when these signals are absent the
  toolkit surfaces the literal `"no data"` marker instead of synthetic
  values.
- **Aero balance drift** – derives rake trends exclusively from OutSim
  `pitch` plus front/rear suspension travel so the drift guidance mirrors
  native LFS telemetry even if `AeroCoherence` appears neutral.【F:tnfr_lfs/core/metrics.py†L1650-L1735】
- **Tyre temperatures/pressures** – TNFR × LFS now consumes the values
  emitted by the OutGauge extended payload when they are finite and
  positive; when the block is disabled the fusion keeps the historical
  sample or the same `"no data"` placeholder so downstream tooling does
  not fabricate temperatures.【F:tnfr_lfs/ingestion/fusion.py†L594-L657】

### Step 4 – Run the quickstart workflow

1. Open a terminal in the repository root (or the virtual environment if you
   created one).
2. Run the packaged scenario:

   ```bash
   make quickstart
   ```

3. The Make target delegates to ``examples/quickstart.sh``, which resolves the
   bundled dataset through
   ``tnfr_lfs.examples.quickstart_dataset.dataset_path()`` so JSON and Markdown
   artefacts appear under ``examples/out/`` without manual file management.
4. Inspect the generated reports with your preferred viewer and cross-reference
   the [examples gallery](examples.md) for additional automation scripts you can
   adapt to your own telemetry.

### Step 5 – Explore further

- Launch the CLI HUD with the commands in the [CLI guide](cli.md) once telemetry
  streaming is active.
- Dive into the [advanced workflows](advanced_workflows.md) to extend the
  baseline analysis with Pareto sweeps, regression comparisons, and scripting
  patterns.
- Keep the [setup equivalences](setup_equivalences.md) at hand when translating
  ΔNFR/ΔSi recommendations into garage adjustments.
## Operational checklist

Pre-stint reviews rely on a compact checklist that validates four
quantitative objectives:

1. **Average Sense Index ≥ 0.75** – the baseline target for the operator
   profile, used by the rule engine and persisted in
   `RuleProfileObjectives`.【F:tnfr_lfs/recommender/rules.py†L604-L615】
2. **ΔNFR density ≤ 6.00 kN·s⁻¹** – the default reference used to compute
   the absolute ΔNFR integral while scoring objectives.【F:tnfr_lfs/recommender/search.py†L210-L236】
3. **Brake headroom ≥ 0.40** – the minimum margin before brake-bias or
   cooling interventions become mandatory.【F:tnfr_lfs/recommender/rules.py†L604-L615】
4. **Aerodynamic Δμ ≤ 0.12** – the tolerance applied when normalising
   aero-mechanical coherence and drift alerts.【F:tnfr_lfs/core/metrics.py†L2558-L2575】

The HUD/CLI summarises these objectives under a “Checklist” line that marks
completed goals with ✅ and highlights pending work with ⚠️ using the
real-time metrics (Si average, ΔNFR integral, brake headroom, and
aerodynamic imbalance).【F:tnfr_lfs/cli/osd.py†L1477-L1567】

## Branding and terminology

- **TNFR theory** is the conceptual framework that formalises the EPI and
  ΔNFR/ΔSi metrics.
- **TNFR × LFS toolkit** identifies the software implementation that automates
  those principles inside Live for Speed.
- The CLI (``tnfr_lfs``), Python package (``tnfr_lfs``) and configuration
  tables (the ``[tool.tnfr_lfs]`` block in ``pyproject.toml``) intentionally
  retain their respective names for compatibility with existing workflows.

## Resources

- [Documentation index](index.md) – MkDocs entry point for the full manual.
- [Beginner quickstart](tutorials.md) – installation walkthrough and dataset
  primer.
- [Advanced workflows](advanced_workflows.md) – Pareto sweeps, robustness
  checks and A/B comparisons.
- [API reference](api_reference.md) – module-level documentation.
- [Examples gallery](examples.md) – automation scripts under ``examples/``.
- [CLI guide](cli.md) – command-line usage and configuration templates.
- [Setup equivalences](setup_equivalences.md) – map TNFR metrics to setup
  adjustments.
- [Preset workflow](presets.md) – shareable configurations and HUD layouts.
- [Brake thermal proxy](brake_thermal_proxy.md) – details of the brake fade
  model.
- [Design notes](DESIGN.md) – textual summary of the TNFR operations manual.
