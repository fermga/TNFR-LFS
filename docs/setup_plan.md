# Setup plan workflow

The setup planner blends the optimisation-aware search module with the rule
engine to produce actionable changes that match the TNFR × LFS heuristics. The
CLI exposes the workflow through the `write-set` subcommand, which reads a
baseline capture, resolves the car/track profile and emits the consolidated plan
through the configured exporter.【F:docs/cli.md†L343-L374】 When the command is
run with a configuration pack (`--pack-root` or `paths.pack_root`), the payload
also embeds the vehicle metadata and the TNFR objectives associated with the
pack so that downstream automations inherit the context of the captured
stint.【F:docs/cli.md†L363-L369】

## Generating a plan

Run the `write-set` command with the baseline you want to iterate on. The
example below writes a JSON plan and stores it under `out/` by following the
project defaults:

```bash
tnfr_lfs write-set runs/fp1.jsonl --export json
```

Switch the exporter (`--export markdown`, `--export csv`, …) to control how the
plan is delivered. All exporters share the same schema and normalise the plan
into the corresponding format, deduplicating repeated rationales and expected
effects to make the handover concise.【F:tnfr_lfs/exporters/setup_plan.py†L105-L131】【F:docs/cli.md†L370-L384】 The Markdown exporter also exposes the empirical
Jacobian gathered during the micro-delta experiments so the change log can be
reviewed in a meeting without opening the raw JSON.【F:docs/cli.md†L379-L384】

## Anatomy of the exported payload

Plans are emitted following the `SetupPlan` dataclass. The resulting payload
captures the context of the run and the mechanical levers that should be
adjusted. Key sections include:

* **Header** – car model, optional session tag and the overall setup coherence
  index (SCI).【F:tnfr_lfs/exporters/setup_plan.py†L68-L120】【F:tnfr_lfs/exporters/setup_plan.py†L171-L204】
* **Changes** – every actionable adjustment with its delta, rationale and
  expected effect.【F:tnfr_lfs/exporters/setup_plan.py†L68-L120】
* **Sensitivity metrics** – aggregated ΔSi/Δp and Δ∫|ΔNFR|/Δp derivatives plus
  phase-specific sensitivities to help prioritise changes per corner phase.
  These metrics are exported both globally and per-phase for Sense Index and
  ΔNFR integral.【F:tnfr_lfs/exporters/setup_plan.py†L132-L184】
* **Rationale indexes** – rationale/effect collections grouped by node and by
  phase so every stakeholder can trace how a change was justified within the
  TNFR framework.【F:tnfr_lfs/exporters/setup_plan.py†L150-L184】
* **Aero guidance** – qualitative brief supported by low- and high-speed
  imbalance figures plus the aero–mechanical coherence score.【F:tnfr_lfs/exporters/setup_plan.py†L88-L118】【F:tnfr_lfs/exporters/setup_plan.py†L171-L204】
* **SCI breakdown** – optional decomposition of the setup coherence index across
  the contributing dimensions; the exporter mirrors the same mapping under the
  legacy `ics_breakdown` key for downstream compatibility.【F:tnfr_lfs/exporters/setup_plan.py†L185-L204】

Parameters that were clamped by the search and therefore not eligible for
further adjustment are listed under `clamped_parameters`, making it obvious when
an input was constrained by the rules engine.【F:tnfr_lfs/exporters/setup_plan.py†L88-L118】【F:tnfr_lfs/exporters/setup_plan.py†L171-L204】

## Phase-axis summary and suggestions

When the planner has access to phase-axis targets or weights, the exporter
produces a compact table that explains how longitudinal (∥) and lateral (⊥)
axes should evolve across entry, apex and exit. Arrows encode the direction and
emphasis of the tweak—`⇈` highlights a strong positive adjustment, `↘` depicts a
soft negative change, and `·0.00` marks negligible contributions. The summary
is accompanied by the top three textual suggestions (for example, “Entry ∥
⇊−0.20”) derived from the same ranking used for the table.【F:tnfr_lfs/exporters/setup_plan.py†L14-L78】【F:tnfr_lfs/exporters/setup_plan.py†L185-L204】 These cues help
engineers apply the nodal guidance without reading the raw numbers.

## Using the plan downstream

Because every exporter shares the same schema, the plan can be consumed by
notebooks, dashboards or documentation workflows interchangeably. JSON exports
are ideal for automation, CSV simplifies spreadsheet reviews, and Markdown
produces a shareable setup brief that mirrors the CLI handover described in the
quickstart tutorial.【F:tnfr_lfs/exporters/setup_plan.py†L105-L204】【F:docs/tutorials.md†L53-L80】 Pair the plan with the ΔNFR/Sense Index reports to validate the
impact of each change before locking in the setup.
