"""Telemetry and configuration helpers for the TNFR × LFS CLI."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from ..core.cache_settings import CacheOptions
from ..ingestion import OutSimClient
from ..ingestion.offline import (
    ReplayCSVBundleReader,
    iter_run,
    raf_to_telemetry_records,
    read_raf,
    write_run,
)
from ..core.epi import TelemetryRecord
from .errors import CliError

CONFIG_ENV_VAR = "TNFR_LFS_CONFIG"
DEFAULT_CONFIG_FILENAME = "tnfr_lfs.toml"

Records = List[TelemetryRecord]

_PARQUET_DEPENDENCY_MESSAGE = (
    "Reading Parquet telemetry requires the 'pandas' package and a compatible engine "
    "(install 'pyarrow' or 'fastparquet')."
)


def load_cli_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load CLI defaults from ``tnfr_lfs.toml`` style files."""

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
        if isinstance(data, dict):
            telemetry_cfg = data.get("telemetry")
            if isinstance(telemetry_cfg, Mapping) and "core" not in data:
                data["core"] = dict(telemetry_cfg)
            cache_options = CacheOptions.from_config(data)
            performance_cfg_raw = data.get("performance")
            legacy_cache_raw = data.get("cache")
            should_update_performance = isinstance(performance_cfg_raw, Mapping) or isinstance(
                legacy_cache_raw, Mapping
            )
            if should_update_performance:
                performance_cfg: dict[str, Any]
                if isinstance(performance_cfg_raw, Mapping):
                    performance_cfg = dict(performance_cfg_raw)
                else:
                    performance_cfg = {}
                performance_cfg.update(
                    {
                        "cache_enabled": cache_options.enable_delta_cache,
                        "max_cache_size": cache_options.max_cache_size,
                        "nu_f_cache_size": cache_options.nu_f_cache_size,
                        "telemetry_cache_size": cache_options.telemetry_cache_size,
                        "recommender_cache_size": cache_options.recommender_cache_size,
                    }
                )
                data["performance"] = performance_cfg
        data["_config_path"] = str(resolved)
        return data
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
