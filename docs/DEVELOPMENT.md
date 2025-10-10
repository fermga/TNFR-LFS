# Development and verification

This repository relies on linting, typing, and testing workflows to keep the TNFR × LFS toolkit healthy. After cloning the project install the development dependencies (this includes `numpy>=1.24,<2.0` and `pandas>=1.5,<3.0`, required by the tests and analysis pipelines) via the packaged extras:

```bash
pip install .[dev]
# or, for an editable environment
pip install -e .[dev]
```

## Verification flow

Run the following commands before submitting a change. Each tool is integrated with CI and GitHub Actions:

```bash
ruff check .
mypy --strict
pytest
```

- `ruff` enforces the code-style and cleanliness rules defined in `pyproject.toml`.
- `mypy --strict` analyses `tnfr_lfs` and the rest of the declared typing targets (including the tests and examples helper). If you need to pass the paths explicitly run `mypy --strict src tests`; note that strict analysis across the whole package takes longer than the reduced subset.
- `pytest` runs the unit and integration test suite, including verification for the example datasets `src/tnfr_lfs/pack/data/test1.raf` and `src/tnfr_lfs/pack/data/test1.zip`.

### CI triggering rules

The CI workflow skips documentation-only pushes and pull requests (`docs/**` and Markdown files). Any
change touching code, configuration, tests, benchmarks, or the automation itself still runs the lint,
type-check, and test jobs. The `make quickstart` pipeline executes on merges to `main` and whenever the
`examples/` tree changes, so example scenarios stay covered without running the heavier simulation for
purely editorial branches.

## Reference dataset

The quickstart flow and the integration tests depend on the capture `src/tnfr_lfs/pack/data/test1.raf` and on the bundle `src/tnfr_lfs/pack/data/test1.zip` exported from Replay Analyzer. Both artefacts live inside the packaged resources (`tnfr_lfs._pack_resources.data_root()`) and contain real Live for Speed telemetry, serving as reference datasets for the binary ingestion flows, regression tests, and CLI tutorials.

## Performance regression benchmark

The repository ships a focused ΔNFR/ν_f benchmark under `benchmarks/`. Install the optional
extra and execute the module locally whenever a change touches the caching helpers or the
natural-frequency analysis:

```bash
pip install -e .[benchmark]
python -m benchmarks.delta_cache_benchmark
# or use the Makefile shortcut
make benchmark-delta-cache
```

The script loads `src/tnfr_lfs/pack/data/test1.zip` through `ReplayCSVBundleReader`, iterates over the telemetry,
and times ΔNFR and ν_f computations with the caches disabled versus the same workload with the
LRU layers warm.【F:benchmarks/delta_cache_benchmark.py†L1-L174】 The default configuration processes
128 samples twice per run (three measured repeats) and currently reports ≈0.0106 s per uncached pass
against ≈0.0089 s with caches enabled, a ~1.2× speed-up.【af1ae4†L1-L4】 Treat a cached mean above
0.010 s or a speed-up below ×1.1 as a regression and investigate before merging.

Increasing `--sample-limit` grows the workload linearly because each sample runs spectral analysis
for the natural-frequency estimator.【F:benchmarks/delta_cache_benchmark.py†L84-L113】 Use higher limits
when you need to stress-test longer stints, but expect minute-long runs once you reach the full
6 500-sample replay bundle; record those timings in the pull request description so reviewers can
compare them with the baseline numbers above.
