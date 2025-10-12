# Beginner quickstart

Kick off the TNFR × LFS workflow with the bundled quickstart scenario. The
steps below cover installation, running the sample dataset and validating the
artefacts that the CLI generates. When you are ready to expand towards Pareto
sweeps or lap-by-lap comparisons, jump to the
[advanced workflows](advanced_workflows.md).

## 1. Install the toolkit {#1-install-the-toolkit}

The quickstart requires Python 3.9+ and the base dependencies declared in
`pyproject.toml`.

```bash
pip install .
# Optional: edit the code locally and keep the CLI in sync
pip install -e .
```

Need the development extras? Replace the base install with
`pip install .[dev]` (or `pip install -e .[dev]` for editable workflows). You
can also use the Makefile shorthands (`make install` or `make dev-install`).
If a command fails with `ModuleNotFoundError`, double-check that the virtual
environment is active before continuing.

!!! tip "Live for Speed telemetry"
    You do **not** need a running simulator for the quickstart. The script
    replays the dataset exposed by
    `tnfr_lfs.examples.quickstart_dataset.dataset_path()` through the CLI so you
    can explore the reports offline.

!!! info "Capturing your own telemetry"
    When you're ready to ingest live UDP streams, follow the
    [`TelemetryFusion` walkthrough](reference/autoapi/tnfr_lfs/ingestion/fusion/index.md#telemetryfusion)
    to wire the OutSim and OutGauge broadcasters straight into the EPI
    extractor. The same calibration hooks and pack overrides apply to both the
    CLI and your own scripts.

## 2. Run the quickstart scenario {#2-run-the-quickstart-scenario}

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
    * `Sample dataset not found` – double-check
      `python -c "from tnfr_lfs.examples.quickstart_dataset import dataset_path; print(dataset_path())"`
      and regenerate the repository data bundle if needed.
    * `ModuleNotFoundError: No module named 'tnfr_lfs'` – install the project
      into the active environment with `pip install -e .` (or include extras
      like `pip install -e .[dev]`).
    * Artefacts missing under `examples/out/` – check write permissions or
      delete stale files before re-running the script.

## 3. Configure Live for Speed telemetry {#3-configure-live-for-speed-telemetry}

1. Launch Live for Speed and open `cfg.txt` (Options → Misc → `cfg.txt`
   or edit the file directly).
2. Enable the OutSim and OutGauge broadcasters with the extended payloads listed
   in the [telemetry configuration checklist](telemetry.md#required-simulators-settings).
   That section provides the exact `cfg.txt` entries and `/outsim`/`/outgauge`
   console commands for both host/port selection and payload flags.
3. Restart the session so Live for Speed loads the new configuration. Refer to
   the [telemetry reference](telemetry.md) for a detailed breakdown of the
   available fields and integration tips.

All TNFR metrics (`ΔNFR`, the nodal projections `∇NFR∥`/`∇NFR⊥`, `ν_f`,
`C(t)` and related indicators) are derived from these native telemetry
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
  recommendations against the `Fz`/`ΔFz` logs when you need absolute forces.【F:tnfr_lfs/ingestion/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】
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

## 4. Inspect the generated artefacts {#4-inspect-the-generated-artefacts}

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
`setup_plan.md` (see the [setup plan workflow](setup_plan.md) for a guided breakdown) distils the plan in table form:

```markdown
| Change | Adjustment | Rationale | Expected effect |
| --- | --- | --- | --- |
| caster_deg | -0.250 | Braking operator applied to the entry phase in microsector 0... | -0.2° caster |
```

Use the Markdown output to brief engineers or paste the JSON artefacts into
notebooks for further processing.

## Next steps

Continue with the [advanced workflows](advanced_workflows.md) to learn how to
run robustness checks, sweep the Pareto front and compare laps between two
stints. Pair those guides with the [setup plan workflow](setup_plan.md) and the
[CLI reference](cli.md) for complete command options.
