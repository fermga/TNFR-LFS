# `tnfr_lfs.ingestion.offline.replay_csv_bundle` module
Utilities for reading CSV bundles exported from Live for Speed replays.

## Classes
### `ReplayCSVBundleReader`
Reader for CSV bundles exported by the LFS replay analyser.

#### Methods
- `clear_cache(self) -> None`
  - Invalidate cached dataframe and record representations.
- `to_dataframe(self, copy: bool = True) -> pd.DataFrame`
  - Return the bundle contents as a merged :class:`~pandas.DataFrame`.

Parameters
----------
copy:
    If ``True`` (the default) return a defensive copy of the cached
    frame.  Callers that only need read-only access can pass
    ``copy=False`` to reuse the cached instance without incurring the
    overhead of duplicating the underlying data.
- `to_records(self, copy: bool = True) -> list[TelemetryRecord]`
  - Convert the bundle contents into :class:`TelemetryRecord` samples.

Parameters
----------
copy:
    If ``True`` (the default) return a copy of the cached records so
    callers can safely mutate the result.  Pass ``copy=False`` to obtain
    a mutable list backed by the cache, which forces the records to be
    recomputed to avoid exposing shared mutable state.

