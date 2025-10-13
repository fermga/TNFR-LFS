"""Telemetry and configuration helpers for the TNFR × LFS CLI."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tnfr_lfs.configuration import load_project_config
from tnfr_core.operators.cache_settings import CacheOptions
from tnfr_lfs.ingestion import OutSimClient
from tnfr_lfs.ingestion.offline import (
    ReplayCSVBundleReader,
    iter_run,
    raf_to_telemetry_records,
    read_raf,
    write_run,
)
from tnfr_core.equations.epi import TelemetryRecord
from tnfr_lfs.cli.errors import CliError

CONFIG_ENV_VAR = "TNFR_LFS_CONFIG"
PROJECT_CONFIG_FILENAME = "pyproject.toml"

Records = List[TelemetryRecord]

_PARQUET_DEPENDENCY_MESSAGE = (
    "Reading Parquet telemetry requires the 'pandas' package and a compatible engine "
    "(install 'pyarrow' or 'fastparquet')."
)


def _normalise_cli_config(payload: Mapping[str, Any], source: Path) -> dict[str, Any]:
    data = {str(key): value for key, value in payload.items()}
    telemetry_cfg = data.get("telemetry")
    if isinstance(telemetry_cfg, Mapping) and "core" not in data:
        data["core"] = dict(telemetry_cfg)

    cache_options = CacheOptions.from_config(data)
    performance_cfg_raw = data.get("performance")
    legacy_cache_raw = data.get("cache")
    if isinstance(performance_cfg_raw, Mapping):
        performance_cfg = dict(performance_cfg_raw)
    else:
        performance_cfg = {}
    if isinstance(performance_cfg_raw, Mapping) or isinstance(legacy_cache_raw, Mapping):
        performance_cfg.update(cache_options.to_performance_config())
        data["performance"] = performance_cfg

    data["_config_path"] = str(source.expanduser().resolve())
    return data


def _iter_unique_paths(candidates: List[Path]) -> List[Path]:
    seen: Dict[Path, None] = {}
    ordered: List[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen[resolved] = None
        ordered.append(resolved)
    return ordered


def _pyproject_candidates(base: Path) -> List[Path]:
    base = base.expanduser()
    if base.name == PROJECT_CONFIG_FILENAME:
        return [base]
    if base.suffix:
        return []
    return [base / PROJECT_CONFIG_FILENAME]


def load_cli_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load CLI defaults from ``pyproject.toml`` files."""

    env_config = os.environ.get(CONFIG_ENV_VAR)
    env_path = Path(env_config) if env_config else None

    explicit_bases: List[Path] = []
    if path is not None:
        explicit_bases.append(path)
    if env_path is not None:
        explicit_bases.append(env_path)

    for base in explicit_bases:
        resolved = base.expanduser()
        pyproject_paths = _pyproject_candidates(resolved)
        for candidate in _iter_unique_paths(pyproject_paths):
            loaded = load_project_config(candidate)
            if not loaded:
                continue
            payload, resolved_candidate = loaded
            return _normalise_cli_config(payload, resolved_candidate)

    for candidate in _iter_unique_paths(_pyproject_candidates(Path.cwd())):
        loaded = load_project_config(candidate)
        if not loaded:
            continue
        payload, resolved = loaded
        return _normalise_cli_config(payload, resolved)

    return {"_config_path": None}


def _persist_records(records: Records, destination: Path, fmt: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        compress = destination.suffix in {".gz", ".gzip"}
        write_run(records, destination, compress=compress)
        return

    if fmt == "parquet":
        serialised = [asdict(record) for record in records]
        try:
            import pandas as pd  # type: ignore

            frame = pd.DataFrame(serialised)
            frame.to_parquet(destination, index=False)
            return
        except ModuleNotFoundError as exc:
            raise CliError(
                _PARQUET_DEPENDENCY_MESSAGE,
                category="usage",
                context={"format": fmt, "destination": str(destination)},
            ) from exc
        except (ImportError, ValueError) as exc:
            raise CliError(
                _PARQUET_DEPENDENCY_MESSAGE,
                category="usage",
                context={"format": fmt, "destination": str(destination)},
            ) from exc

    raise CliError(
        f"Unsupported format '{fmt}'.",
        category="usage",
        context={"format": fmt, "destination": str(destination)},
    )


def _load_replay_bundle(source: Path, *, cache_size: int | None = None) -> Records:
    if not source.exists():
        raise CliError(
            f"Replay CSV bundle {source} does not exist",
            category="not_found",
            context={"path": str(source), "kind": "replay_csv_bundle"},
        )
    size = 1 if cache_size is None else int(cache_size)
    reader = ReplayCSVBundleReader(source, cache_size=size)
    return reader.to_records()


def _coerce_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    data = dict(payload)
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = list(value)
    return data


def _load_records(source: Path) -> Records:
    if not source.exists():
        raise CliError(
            f"Telemetry source {source} does not exist",
            category="not_found",
            context={"path": str(source)},
        )
    suffix = source.suffix.lower()
    name = source.name.lower()
    if suffix == ".csv":
        return OutSimClient().ingest(source)
    if suffix == ".raf":
        return raf_to_telemetry_records(read_raf(source))
    if name.endswith(".jsonl") or name.endswith(".jsonl.gz") or name.endswith(".jsonl.gzip"):
        return list(iter_run(source))
    if suffix == ".parquet":
        data = _load_parquet_records(source)
        return [TelemetryRecord(**_coerce_payload(item)) for item in data]
    if suffix == ".json":
        data = _load_json_records(source)
        return [TelemetryRecord(**_coerce_payload(item)) for item in data]

    raise CliError(
        f"Unsupported telemetry format: {source}",
        category="usage",
        context={"path": str(source), "suffix": suffix, "name": name},
    )


def _load_parquet_records(source: Path) -> List[Mapping[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        if _is_probably_json_document(source):
            return _load_json_records(source)
        raise CliError(
            _PARQUET_DEPENDENCY_MESSAGE,
            category="usage",
            context={"path": str(source), "format": "parquet"},
        ) from exc

    try:
        frame = pd.read_parquet(source)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        if _is_probably_json_document(source):
            return _load_json_records(source)
        raise CliError(
            _PARQUET_DEPENDENCY_MESSAGE,
            category="usage",
            context={"path": str(source), "format": "parquet"},
        ) from exc

    return frame.to_dict(orient="records")


def _load_json_records(source: Path) -> List[Mapping[str, Any]]:
    with source.open("r", encoding="utf8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    raise CliError(
        f"JSON telemetry source {source} must contain a list of samples",
        category="usage",
        context={"path": str(source)},
    )


def _is_probably_json_document(source: Path) -> bool:
    with source.open("rb") as handle:
        while True:
            byte = handle.read(1)
            if not byte:
                return False
            if byte in b" \t\r\n":
                continue
            return byte in {b"{", b"["}
