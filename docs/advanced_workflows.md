# Advanced workflows

Build on the quickstart by stress-testing the setup, exploring nearby
configurations and comparing alternative stints. All of the commands below assume
that you already generated a baseline with `tnfr_lfs baseline` (for example via
`examples/quickstart.sh`) and that the resulting JSONL file is available under
`examples/out/baseline.jsonl`.

## Run robustness checks

The extended HTML report exposes the ΔNFR variability, coherence map and nodal
couplings per microsector.

```bash
tnfr_lfs report examples/out/baseline.jsonl \
  --target-delta 0.5 --target-si 0.8 \
  --export html_ext > examples/out/report.html
```

Open the HTML file in a browser to review the robustness panels. Use the
operator summaries and the [setup equivalences](setup_equivalences.md) to pick
the subsystem (tyres, suspension, chassis) that should be reinforced before the
next track session.

## Explore the Pareto front

When a candidate setup looks promising, sweep the surrounding decision space to
visualise trade-offs across ΔNFR, Sense Index and coherence.

```bash
tnfr_lfs pareto examples/out/baseline.jsonl \
  --car-model XFG --radius 2 \
  --export html_ext > examples/out/pareto.html
```

The generated report highlights each lever’s contribution to the Pareto front,
making it easier to document why a particular change was approved. Capture the
rationales in the TNFR playbook or append them to the setup plan.

## Lap-by-lap A/B analysis

Compare two recorded stints to see which variant achieves the target metric.

```bash
tnfr_lfs compare baseline.jsonl variant.jsonl \
  --metric sense_index --export html_ext > examples/out/abtest.html
```

Review the stint averages and lap distributions to confirm whether the new
configuration improves the Sense Index. Reconcile the findings with the
recommendations surfaced in the [setup plan](setup_plan.md) to keep the
documentation consistent.

## Related resources

* [CLI reference](cli.md) for every command flag.
* [CLI deep dive](cli_deep_dive.md) with HUD and automation tips.
* [Setup equivalences](setup_equivalences.md) for translating ΔNFR metrics into
  mechanical adjustments.
