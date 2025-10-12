# `tnfr_lfs.ingestion.offline.logs` module
Persistence helpers for telemetry run logs.

## Classes
### `DeterministicReplayer`
Simple container that yields telemetry samples deterministically.

#### Methods
- `iter(self) -> Iterator[TelemetryRecord]`
  - Return a new iterator over the stored telemetry sequence.

## Functions
- `write_run(records: Sequence[TelemetryRecord], path: str | Path, *, compress: bool = True) -> None`
  - Persist ``records`` to ``path`` using a newline-delimited JSON format.

Parameters
----------
records:
    Sequence of :class:`~tnfr_lfs.core.epi.TelemetryRecord` objects to serialise.
path:
    Destination file.  Parent directories are created automatically.
compress:
    When ``True`` the payload is compressed using gzip.  ``iter_run`` is able
    to transparently read both compressed and uncompressed payloads.
- `prepare_run_destination(*, car_model: str, track_name: str, output_dir: str | Path | None = None, suffix: str = '.jsonl', force: bool = False) -> Path`
  - Return a timestamped run path ensuring the parent directory exists.

Parameters
----------
car_model:
    Name of the car model used for slug generation.
track_name:
    Name of the track used for slug generation.
output_dir:
    Optional base directory for the run.  Defaults to :data:`RUNS_DIR`.
suffix:
    File suffix to append to the generated name.  ``.jsonl`` by default.
force:
    When ``True`` existing files are reused; otherwise ``FileExistsError``
    is raised on collisions.
- `iter_run(path: str | Path) -> Iterator[TelemetryRecord]`
  - Yield telemetry samples previously persisted with :func:`write_run`.

## Attributes
- `RUNS_DIR = Path('runs')`

