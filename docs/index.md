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

### Quickstart overview

The [beginner quickstart](tutorials.md) now hosts the canonical onboarding guide. Use the summary below to jump straight into the relevant section:

- **Install prerequisites** – follow the environment preparation checklist in [“Install the toolkit”](tutorials.md#1-install-the-toolkit) to set up Python, create an optional virtual environment, and choose the right dependency extras.
- **Run the bundled scenario** – execute the capture pipeline described in [“Run the quickstart scenario”](tutorials.md#2-run-the-quickstart-scenario) to replay the sample telemetry and validate your installation.
- **Configure live telemetry** – when you are ready to ingest OutSim/OutGauge streams, use the [“Configure Live for Speed telemetry”](tutorials.md#3-configure-live-for-speed-telemetry) checklist to mirror the game-side configuration.
- **Review artefacts** – explore the output highlighted in [“Inspect the generated artefacts”](tutorials.md#4-inspect-the-generated-artefacts) to understand how baseline reports, setup plans, and JSON exports fit together.

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
