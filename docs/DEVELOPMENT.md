# Development and verification

This repository relies on linting, typing, and testing workflows to keep the TNFR × LFS toolkit healthy. After cloning the project, follow the [Step 1 installation guide](tutorials.md#1-install-the-toolkit) and choose the development extra (`pip install .[dev]` or `pip install -e .[dev]`) so the verification tools below are available.

## Verification flow

Run the following commands before submitting a change. Each tool is integrated with CI and GitHub Actions:

```bash
ruff check .
mypy --strict
pytest
lychee --config lychee.toml README.md docs/**/*.md
```

- `ruff` enforces the code-style and cleanliness rules defined in `pyproject.toml`.
- `mypy --strict` analyses `tnfr_lfs` and the rest of the declared typing targets (including the tests and examples helper). If you need to pass the paths explicitly run `mypy --strict src tests`; note that strict analysis across the whole package takes longer than the reduced subset.
- `pytest` runs the unit and integration test suite, including verification for the example datasets `src/tnfr_lfs/resources/data/test1.raf` and `src/tnfr_lfs/resources/data/test1.zip`.
- `lychee` verifies that the Markdown documentation (including the MkDocs site) references valid URLs and relative links. Install it via `cargo install lychee` or download a pre-built release binary from the [Lychee project](https://github.com/lycheeverse/lychee/releases).

### Pre-commit hooks

Install the [pre-commit](https://pre-commit.com) hooks to run Ruff, Black, and MyPy with the same
settings defined in `pyproject.toml` before every commit. With the development extra from
[Step 1](tutorials.md#1-install-the-toolkit) already installed, enable the hooks:

```bash
pre-commit install
```

Update the hook revisions periodically (or whenever a contributor bumps them in a pull request)
with:

```bash
pre-commit autoupdate
```

The CI pipeline executes the same tools, so keeping the local hooks current helps surface lint,
formatting, and typing issues early.

### Test fixtures and parametrisation

The [tests fixture catalogue](../tests/README.md) documents the deterministic
datasets, factories, and parametrised case tables exercised by the suite. Refer
to it before adding new telemetry bundles, CLI configuration cases, recommender
scenarios, or metric windows so you can reuse the shared builders and follow the
`ids=` naming guidance enforced in CI.

### CI triggering rules

The CI workflow now runs on documentation-only branches and executes a dedicated link-checking job in
addition to the lint/type/test matrix. Broken intra-site references or external URLs fail fast via the
Lychee action. The `make quickstart` pipeline executes on merges to `main` and whenever the `examples/`
tree changes, so example scenarios stay covered without running the heavier simulation for purely
editorial branches.


## Module layout

Utility helpers previously grouped under `tnfr_lfs.utils` now live in themed packages. Import from the new locations:

- `tnfr_lfs.common.immutables` for immutable container helpers.
- `tnfr_lfs.logging.config` for logging configuration helpers.
- `tnfr_lfs.math.conversions` for numeric coercion utilities.
- `tnfr_lfs.visualization.sparkline` for sparkline rendering.

The `tnfr_lfs.utils` compatibility shim has been removed. Update any remaining imports to target the themed packages directly to avoid `ModuleNotFoundError` at runtime.

## Reference dataset

The quickstart flow and the integration tests depend on the capture `src/tnfr_lfs/resources/data/test1.raf` and on the bundle `src/tnfr_lfs/resources/data/test1.zip` exported from Replay Analyzer. Both artefacts live inside the packaged resources (`tnfr_lfs.resources.data_root()`) and contain real Live for Speed telemetry, serving as reference datasets for the binary ingestion flows, regression tests, and CLI tutorials.

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

The script loads `src/tnfr_lfs/resources/data/test1.zip` through `ReplayCSVBundleReader`, iterates over the telemetry,
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
