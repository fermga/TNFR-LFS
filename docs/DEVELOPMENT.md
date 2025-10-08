# Development and verification

This repository relies on linting, typing, and testing workflows to keep the TNFR × LFS toolkit healthy. After cloning the project install the development dependencies (this includes `numpy>=1.24,<2.0` and `pandas>=1.5,<3.0`, required by the tests and analysis pipelines). You can rely on the bundled requirement files or install the package with its development extras:

```bash
pip install -r requirements-dev.txt
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
- `mypy --strict` analyses `tnfr_lfs` and the rest of the declared typing targets (for example `typing_targets`). If you need to pass the paths explicitly run `mypy --strict tnfr_lfs typing_targets`; note that strict analysis across the whole package takes longer than the reduced subset.
- `pytest` runs the unit and integration test suite, including verification for the example datasets `data/test1.raf` and `data/test1.zip`.

## Reference dataset

The quickstart flow and the integration tests depend on the capture `data/test1.raf` and on the bundle `data/test1.zip` exported from Replay Analyzer. Both artefacts contain real Live for Speed telemetry and serve as reference datasets for the binary ingestion flows, regression tests, and CLI tutorials.
