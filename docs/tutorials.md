# Beginner quickstart

Kick off the TNFR × LFS workflow with the bundled quickstart scenario. The
steps below cover installation, running the sample dataset and validating the
artefacts that the CLI generates. When you are ready to expand towards Pareto
sweeps or lap-by-lap comparisons, jump to the
[advanced workflows](advanced_workflows.md).

## 1. Install the toolkit

The quickstart requires Python 3.9+ and the base dependencies listed in
`requirements.txt`.

```bash
python -m pip install -r requirements.txt
# Optional: edit the code locally and keep the CLI in sync
python -m pip install -e .
```

You can also use the Makefile shorthands (`make install` or `make dev-install`).
If a command fails with `ModuleNotFoundError`, double-check that the virtual
environment is active before continuing.

!!! tip "Live for Speed telemetry"
    You do **not** need a running simulator for the quickstart. The script
    replays `data/BL1_XFG_baseline.csv` through the CLI so you can explore the
    reports offline.

## 2. Run the quickstart scenario

From the repository root run:

```bash
make quickstart
```

The helper script calls `examples/quickstart.sh`. A successful run prints the
baseline capture confirmation before exiting:

```text
{"timestamp": "2025-10-09T07:59:12.199199Z", "level": "INFO", "logger": "tnfr_lfs.cli.workflows", "message": "Baseline saved 17 samples to /workspace/TNFR-LFS/examples/out/baseline.jsonl (jsonl).", "extra": {"taskName": null, "event": "baseline.saved", "samples": 17, "destination": "/workspace/TNFR-LFS/examples/out/baseline.jsonl", "format": "jsonl", "simulate": true, "capture_metrics": {"attempts": 17, "samples": 17, "dropped_pairs": 0, "duration": 0.0, "outsim_timeouts": 0, "outgauge_timeouts": 0, "outsim_ignored_hosts": 0, "outgauge_ignored_hosts": 0}}}
Baseline saved 17 samples to /workspace/TNFR-LFS/examples/out/baseline.jsonl (jsonl).
```

!!! warning "Common pitfalls"
    * `Sample dataset not found` – make sure `data/BL1_XFG_baseline.csv` exists.
      Regenerate the repository data bundle if needed.
    * `ModuleNotFoundError: No module named 'tnfr_lfs'` – install the project
      into the active environment with `pip install -e .`.
    * Artefacts missing under `examples/out/` – check write permissions or
      delete stale files before re-running the script.

## 3. Inspect the generated artefacts

The quickstart populates `examples/out/` with JSONL, JSON and Markdown payloads:

```text
$ ls examples/out
analyze.json  baseline.jsonl  report.json  setup_plan.md  suggest.json
```

`baseline.jsonl` stores the replayed telemetry. Use `head` or your favourite
viewer to check the first sample:

```json
{"brake_pressure": 0.0, "gear": 2, "lateral_accel": 0.3, "mu_eff_front": 0.07281199941750399, "nfr": 500.0, "si": 0.88, "speed": 18.24, "steer": 0.06, "throttle": 0.95, "timestamp": 0.0, "vertical_load": 5000.0, "yaw_rate": 0.21}
```

`analyze.json` aggregates the ΔNFR and Sense Index series. The report and
suggestion files summarise the recommended setup changes, while
`setup_plan.md` distils the plan in table form:

```markdown
| Change | Adjustment | Rationale | Expected effect |
| --- | --- | --- | --- |
| caster_deg | -0.250 | Braking operator aplicado sobre la fase de entry en microsector 0... | -0.2° caster |
```

Use the Markdown output to brief engineers or paste the JSON artefacts into
notebooks for further processing.

## Next steps

Continue with the [advanced workflows](advanced_workflows.md) to learn how to
run robustness checks, sweep the Pareto front and compare laps between two
stints. Pair those guides with the [CLI reference](cli.md) for complete
command options.
