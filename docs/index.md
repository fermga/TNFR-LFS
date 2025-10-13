# TNFR × LFS Documentation

Treat this index as the canonical home for the extended
introduction, feature descriptions, onboarding narratives, checklists and API
map that the project README now summarises.

## Feature overview

TNFR × LFS bundles the following core capabilities for race engineers and data
scientists:

- Live ΔNFR HUD and capture pipeline wired directly to Live for Speed
  OutSim/OutGauge streams.
- Automated baselines, analysis, suggestions and reporting from a single CLI.
- Extensible exporters (JSONL, Markdown, HTML) and configuration packs for
  repeatable engineering workflows.
- Benchmark suite and reproducible examples for regression testing new ideas.
- Ingestion, analytics, recommendation and export subsystems that map the TNFR
  framework to actionable setup advice:
  1. **Ingest** – capture telemetry samples from OutSim-compatible streams via
     :class:`tnfr_lfs.ingestion.live.OutSimClient`.
  2. **Core analytics** – extract Event Performance Indicators (EPI) and
     compute ΔFz/ΔSi deltas through :class:`tnfr_core.epi.EPIExtractor`.
  3. **Recommendation engine** – map metrics to actionable setup advice using
     the rule engine in :mod:`tnfr_lfs.recommender.rules`.
  4. **Exporters** – serialise analysis results to JSON or CSV with the
     functions in :mod:`tnfr_lfs.exporters`.

The project ships with a CLI (`tnfr_lfs`) as well as examples and unit tests to
illustrate the workflow. The ingestion pipeline regression suite in
`tests/test_ingestion.py` exercises the OutSim/OutGauge UDP clients and fusion
helpers end-to-end.【F:tests/test_ingestion.py†L1-L376】

## Quickstart

!!! tip "Ready to run the toolkit?"
    Follow the [beginner quickstart tutorial](tutorials.md) for the step-by-step
    installation, sample scenario, telemetry configuration and artefact review
    walkthrough that onboards you from a fresh environment to a race-ready
    workflow.

TNFR × LFS targets Python 3.10+ environments and Live for Speed telemetry
streams. The quickstart covers dependency setup, the baseline scenario, the
simulator configuration and the report validation process without duplicating
those instructions here.

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
   aero-mechanical coherence and drift alerts.【F:tnfr_core/metrics.py†L2558-L2575】

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

## Alineación TNFR

Los operadores estructurales de TNFR cubren trece arquetipos. Los códigos
provienen de los detectores `AL`, `EN`, `IL`, `NAV`, `NUL`, `OZ`, `RA`,
`REMESH`, `SILENCE`, `THOL`, `UM`, `VAL` y `ZHIR`, que `canonical_operator_label`
traduce a sus etiquetas canónicas (Support, Reception, Coherence, Transition,
Contraction, Dissonance, Propagation, Remeshing, Structural silence,
Auto-organisation, Coupling, Amplification y Transformation).【F:src/tnfr_core/operator_detection.py†L43-L68】

La ecuación nodal extendida que guía el índice de sentido pondera cada nodo por
su aporte ΔNFR, su frecuencia natural y la entropía estructural:

```
Sense Index = 1 / (1 + Σ w_i · |ΔNFR_i| · g(ν_f_i)) - λ · H
```

El término ponderado `Σ w_i · |ΔNFR_i| · g(ν_f_i)` absorbe la dinámica nodal, la
función `g(ν_f_i)` corrige por frecuencia natural y `H` representa la entropía
Shannon calculada sobre la distribución ΔNFR.【F:docs/DESIGN.md†L41-L88】

Las nuevas métricas expuestas por el núcleo recopilan expansiones y entropías
estructurales (`support_effective`, `load_support_ratio`,
`structural_expansion_longitudinal`, `structural_contraction_longitudinal`,
`structural_expansion_lateral`, `structural_contraction_lateral`,
`delta_nfr_entropy`, `node_entropy`, `phase_delta_nfr_entropy`,
`phase_node_entropy`, `thermal_load`, `style_index`, `network_memory`, además de
`aero_balance_drift`, `slide_catch_budget` y `ackermann_parallel_index`). Estas
series alimentan los operadores `recursivity_operator` y `mutation_operator` para
mantener la memoria de red y las transiciones de arquetipo.【F:docs/DESIGN.md†L65-L89】【F:tools/tnfr_theory_audit.py†L13-L103】

El script `tools/tnfr_theory_audit.py` genera un informe actualizado de la
alineación teórica. Ejecútalo con `poetry run python tools/tnfr_theory_audit.py
--core --tests --output tests/_report/theory_impl_matrix.md` y consulta el
resultado en `tests/_report/theory_impl_matrix.md` para validar la cobertura de
los operadores y métricas expuestos.【F:tools/tnfr_theory_audit.py†L1-L67】

## Resources

- [Documentation index](index.md) – MkDocs entry point for the full manual.
- [Beginner quickstart](tutorials.md) – installation walkthrough and dataset
  primer.
- [Advanced workflows](advanced_workflows.md) – Pareto sweeps, robustness
  checks and A/B comparisons.
- [API reference](api_reference.md) – module-level documentation.
- [Operational checklist](#operational-checklist) – quantitative targets for
  stint reviews and automated rules.
- [Examples gallery](examples.md) – automation scripts under ``examples/``.
- [CLI guide](cli.md) – command-line usage and configuration templates.
- [Setup equivalences](setup_equivalences.md) – map TNFR metrics to setup
  adjustments.
- [Preset workflow](presets.md) – shareable configurations and HUD layouts.
- [Brake thermal proxy](brake_thermal_proxy.md) – details of the brake fade
  model.
- [Design notes](DESIGN.md) – textual summary of the TNFR operations manual.
