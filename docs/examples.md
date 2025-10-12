# Examples gallery

Discover runnable samples that exercise the TNFR × LFS pipeline without having to
reverse-engineer the repository layout. Each example lists the scenario it
covers, inputs to prepare, how to launch it, and a taste of the output you can
expect.

## Quickstart pipeline (`examples/quickstart.sh`)

**Problem it solves:** Run the full baseline → analysis → recommendations → report
loop with the bundled XF GTi telemetry so you can inspect every artefact the CLI
produces.

**Required inputs:**

- `tnfr_lfs` available on `PYTHONPATH` (e.g., `pip install -e .` from the repo
  root, or set `PYTHONPATH=src` because the package lives under
  `src/tnfr_lfs`).
- Sample telemetry resolved with
  ``python -c "from tnfr_lfs.examples.quickstart_dataset import dataset_path; print(dataset_path())"``
  (included in the repository).
- Write access to `examples/out/` for generated JSON/Markdown reports.

**How to run it:** Follow
[“Run the quickstart scenario”](tutorials.md#2-run-the-quickstart-scenario) in the
beginner tutorial. It walks through the `make quickstart` helper, the underlying
`examples/quickstart.sh` invocation, and the reference CLI log you can use to
validate your run.

> The script writes `baseline.jsonl`, `analyze.json`, `suggest.json`,
> `report.json`, `setup_plan.md`, and a text-based sense-index plot to
> `examples/out/` for further inspection.

## Telemetry ingestion and recommendations (`examples/ingest_and_recommend.py`)

**Problem it solves:** Show how to ingest telemetry records, extract event
phases, and emit natural-language setup recommendations from Python without
invoking the CLI.

**Required inputs:**

- `tnfr_lfs` importable (install the package or run with `PYTHONPATH=src` from
  the repository root because the package lives under `src/tnfr_lfs`).
- The script ships with a small legacy-format telemetry snippet; replace it with
  your own OutSim CSV if you want to analyse a full stint.

**Run it**

```bash
PYTHONPATH=src python examples/ingest_and_recommend.py
```

**Expected output**

```
[suspension] Decrease rear ride height to rebalance load (Advanced Setup Guide · Ride Heights & Load Distribution [ADV-RDH])
  ΔNFR deviated by 23.3 units relative to baseline at t=0.20. Refer to Advanced Setup Guide · Ride Heights & Load Distribution [ADV-RDH] to readjust ride heights.
[aero] Stabilise aero balance to recover sense index (Basic Setup Guide · Aero Balance [BAS-AER])
  Sense index dropped to 0.00 at t=0.10, below the threshold of 0.60. Refer to Basic Setup Guide · Aero Balance [BAS-AER] to rebalance load.
[driver] Review driving inputs for consistency (Basic Setup Guide · Consistent Driving [BAS-DRV])
  Average sense index across the analysed stint is 0.33, below the expected threshold of 0.75. Lean on Basic Setup Guide · Consistent Driving [BAS-DRV] to reinforce consistent habits.
```

> Extend the example by passing a longer telemetry capture through
> `OutSimClient().ingest(...)` or by serialising recommendations to JSON for a
> web dashboard.

## Export computed EPI to CSV (`examples/export_to_csv.py`)

**Problem it solves:** Demonstrate how to run the EPI extractor and reuse the
CSV exporter to persist calculated series for downstream analysis.

**Required inputs:**

- `tnfr_lfs` importable (install the package or run with `PYTHONPATH=src` so the
  interpreter can find `src/tnfr_lfs`).
- Inline telemetry sample with the legacy OutSim header (customise `DATA` for
  your own runs).

**Run it**

```bash
PYTHONPATH=src python examples/export_to_csv.py
```

**Expected output**

```
timestamp,epi,delta_nfr,delta_nfr_proj_longitudinal,delta_nfr_proj_lateral,sense_index
0.000,0.6220,0.000,0.000,0.000,1.000
0.100,0.6320,2.500,0.000,0.000,0.161
0.200,0.6370,5.000,0.000,0.000,0.050
```

> Swap the in-memory `StringIO` for a file handle if you want to capture the CSV
> on disk instead of printing it to stdout.
