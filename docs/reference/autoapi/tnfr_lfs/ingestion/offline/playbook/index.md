# `tnfr_lfs.ingestion.offline.playbook` module
Helpers to load TNFR playbook suggestions.

## Functions
- `load_playbook(source: str | Path | None = None) -> Mapping[str, tuple[str, ...]]`
  - Load the bundled TNFR playbook suggestions.

Parameters
----------
source:
    Optional filesystem path to a TOML playbook. When omitted the embedded
    resource is used.

