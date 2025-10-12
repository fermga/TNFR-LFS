# `tnfr_lfs.examples.quickstart_dataset` module
Helpers that expose the bundled quickstart dataset.

## Functions
- `dataset_path(root: Path | None = None) -> Path`
  - Return the path to the bundled quickstart dataset.
- `dataset_columns() -> tuple[str, ...]`
  - Column names expected by the CLI quickstart pipeline.
- `dataset_sample_count(root: Path | None = None) -> int`
  - Number of telemetry samples available in the dataset.

