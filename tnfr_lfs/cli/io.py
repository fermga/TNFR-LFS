"""Telemetry and configuration helpers for the TNFR × LFS CLI."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from ..acquisition import OutSimClient
from ..io import ReplayCSVBundleReader, logs, raf_to_telemetry_records, read_raf
from ..core.epi import TelemetryRecord

CONFIG_ENV_VAR = "TNFR_LFS_CONFIG"
DEFAULT_CONFIG_FILENAME = "tnfr-lfs.toml"

Records = List[TelemetryRecord]

_PARQUET_DEPENDENCY_MESSAGE = (
    "Reading Parquet telemetry requires the 'pandas' package and a compatible engine "
    "(install 'pyarrow' or 'fastparquet')."
)


def load_cli_config(path: Path | None = None) -> Dict[str, Any]:
    """Load CLI defaults from ``tnfr-lfs.toml`` style files."""

    candidates: List[Path] = []
    if path is not None:
        candidates.append(path)
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / DEFAULT_CONFIG_FILENAME)
    candidates.append(Path.home() / ".config" / DEFAULT_CONFIG_FILENAME)

    resolved_candidates: Dict[Path, None] = {}
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved in resolved_candidates:
            continue
        resolved_candidates[resolved] = None
        if not resolved.exists():
            continue
        with resolved.open("rb") as handle:
            data = tomllib.load(handle)
        data["_config_path"] = str(resolved)
        return data
    return {"_config_path": None}


def _persist_records(records: Records, destination: Path, fmt: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        compress = destination.suffix in {".gz", ".gzip"}
        logs.write_run(records, destination, compress=compress)
        return

    if fmt == "parquet":
        serialised = [asdict(record) for record in records]
        try:
            import pandas as pd  # type: ignore

            frame = pd.DataFrame(serialised)
            frame.to_parquet(destination, index=False)
            return
        except ModuleNotFoundError:
            with destination.open("w", encoding="utf8") as handle:
                json.dump(serialised, handle, sort_keys=True)
            return

    raise ValueError(f"Unsupported format '{fmt}'.")


def _load_replay_bundle(source: Path) -> Records:
    if not source.exists():
        raise FileNotFoundError(f"Replay CSV bundle {source} does not exist")
    reader = ReplayCSVBundleReader(source)
    return reader.to_records()


def _load_records_from_namespace(
    namespace: argparse.Namespace,
) -> Tuple[Records, Path]:
    replay_bundle = getattr(namespace, "replay_csv_bundle", None)
    telemetry_path = getattr(namespace, "telemetry", None)
    if replay_bundle is not None:
        bundle_path = Path(replay_bundle)
        records = _load_replay_bundle(bundle_path)
        namespace.telemetry = bundle_path
        return records, bundle_path
    if telemetry_path is None:
        raise SystemExit(
            "A telemetry baseline path is required unless --replay-csv-bundle is provided."
        )
    records = _load_records(telemetry_path)
    return records, telemetry_path


def _coerce_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    data = dict(payload)
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = list(value)
    return data


def _load_records(source: Path) -> Records:
    if not source.exists():
        raise FileNotFoundError(f"Telemetry source {source} does not exist")
    suffix = source.suffix.lower()
    name = source.name.lower()
    if suffix == ".csv":
        return OutSimClient().ingest(source)
    if suffix == ".raf":
        return raf_to_telemetry_records(read_raf(source))
    if name.endswith(".jsonl") or name.endswith(".jsonl.gz") or name.endswith(".jsonl.gzip"):
        return list(logs.iter_run(source))
    if suffix == ".parquet":
        data = _load_parquet_records(source)
        return [TelemetryRecord(**_coerce_payload(item)) for item in data]
    if suffix == ".json":
        data = _load_json_records(source)
        return [TelemetryRecord(**_coerce_payload(item)) for item in data]

    raise ValueError(f"Unsupported telemetry format: {source}")


def _load_parquet_records(source: Path) -> List[Mapping[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        if _is_probably_json_document(source):
            return _load_json_records(source)
        raise RuntimeError(_PARQUET_DEPENDENCY_MESSAGE) from exc

    try:
        frame = pd.read_parquet(source)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        if _is_probably_json_document(source):
            return _load_json_records(source)
        raise RuntimeError(_PARQUET_DEPENDENCY_MESSAGE) from exc

    return frame.to_dict(orient="records")


def _load_json_records(source: Path) -> List[Mapping[str, Any]]:
    with source.open("r", encoding="utf8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    raise ValueError(f"JSON telemetry source {source} must contain a list of samples")


def _is_probably_json_document(source: Path) -> bool:
    with source.open("rb") as handle:
        while True:
            byte = handle.read(1)
            if not byte:
                return False
            if byte in b" \t\r\n":
                continue
            return byte in {b"{", b"["}
