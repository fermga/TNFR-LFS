from __future__ import annotations

import argparse
import builtins
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from tnfr_lfs.cache_settings import CacheOptions
from tnfr_lfs.cli import common as cli_common
from tnfr_lfs.cli import io as cli_io
from tnfr_lfs.cli.common import CliError


def test_load_records_from_namespace_prefers_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"dummy")
    expected = [object()]
    monkeypatch.setattr(
        cli_common, "_load_replay_bundle", lambda path, **_: expected
    )
    namespace = argparse.Namespace(replay_csv_bundle=bundle_path, telemetry=None)

    records, resolved = cli_common._load_records_from_namespace(namespace)

    assert records is expected
    assert resolved == bundle_path
    assert namespace.telemetry == bundle_path


def test_load_records_from_namespace_threads_cache_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"dummy")
    captured: dict[str, int | None] = {}

    def _fake_loader(path: Path, *, cache_size: int | None = None):
        captured["path"] = path
        captured["cache_size"] = cache_size
        return []

    monkeypatch.setattr(cli_common, "_load_replay_bundle", _fake_loader)
    namespace = argparse.Namespace(replay_csv_bundle=bundle_path, telemetry=None)
    namespace.cache_options = CacheOptions(
        enable_delta_cache=True, nu_f_cache_size=128, telemetry_cache_size=0
    )

    cli_common._load_records_from_namespace(namespace)

    assert captured["path"] == bundle_path
    assert captured["cache_size"] == 0


def test_load_records_from_namespace_requires_path() -> None:
    namespace = argparse.Namespace(replay_csv_bundle=None, telemetry=None)
    with pytest.raises(CliError):
        cli_common._load_records_from_namespace(namespace)


def test_load_records_uses_outsim_for_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    telemetry_path = tmp_path / "stint.csv"
    telemetry_path.write_text("timestamp")
    captured: dict[str, Path] = {}

    class DummyClient:
        def ingest(self, source: Path):
            captured["source"] = source
            return ["ok"]

    monkeypatch.setattr(cli_io, "OutSimClient", lambda: DummyClient())

    records = cli_io._load_records(telemetry_path)

    assert records == ["ok"]
    assert captured["source"] == telemetry_path


def test_load_records_converts_json_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    telemetry_path = tmp_path / "session.json"
    telemetry_path.write_text(json.dumps([{"timestamp": 1.5, "gear": 3}]))

    class DummyRecord:
        def __init__(self, **payload: float):
            self.payload = payload

    monkeypatch.setattr(cli_io, "TelemetryRecord", DummyRecord)

    records = cli_io._load_records(telemetry_path)

    assert len(records) == 1
    assert isinstance(records[0], DummyRecord)
    assert records[0].payload["timestamp"] == 1.5
    assert records[0].payload["gear"] == 3


def test_load_records_parquet_requires_pandas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    telemetry_path = tmp_path / "baseline.parquet"
    telemetry_path.write_bytes(b"PAR1\x00\x00\x00PAR1")

    original = sys.modules.pop("pandas", None)
    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "pandas":
            raise ModuleNotFoundError("No module named 'pandas'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        with pytest.raises(RuntimeError, match="requires the 'pandas' package"):
            cli_io._load_records(telemetry_path)
    finally:
        if original is not None:
            sys.modules["pandas"] = original


def test_load_records_parquet_requires_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    telemetry_path = tmp_path / "baseline.parquet"
    telemetry_path.write_bytes(b"PAR1\x00\x00\x00PAR1")

    dummy = types.ModuleType("pandas")

    def read_parquet(path: Path) -> None:  # type: ignore[no-untyped-def]
        raise ImportError("Unable to find a usable engine")

    dummy.read_parquet = read_parquet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas", dummy)

    with pytest.raises(RuntimeError, match="compatible engine"):
        cli_io._load_records(telemetry_path)


@dataclass
class _DummyRecord:
    value: int


def test_persist_records_parquet_requires_pandas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "baseline.parquet"
    records = [_DummyRecord(1)]

    original = sys.modules.pop("pandas", None)
    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "pandas":
            raise ModuleNotFoundError("No module named 'pandas'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        with pytest.raises(RuntimeError, match="requires the 'pandas' package"):
            cli_io._persist_records(records, destination, "parquet")
    finally:
        if original is not None:
            sys.modules["pandas"] = original

    assert not destination.exists()


def test_persist_records_parquet_requires_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "baseline.parquet"
    records = [_DummyRecord(1)]

    original = sys.modules.pop("pandas", None)

    class DummyFrame:
        def __init__(self, data):  # type: ignore[no-untyped-def]
            self.data = data

        def to_parquet(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise ImportError("Unable to find a usable engine")

    dummy = types.ModuleType("pandas")
    dummy.DataFrame = lambda data: DummyFrame(data)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas", dummy)

    try:
        with pytest.raises(RuntimeError, match="compatible engine"):
            cli_io._persist_records(records, destination, "parquet")
    finally:
        if original is not None:
            sys.modules["pandas"] = original
        else:
            sys.modules.pop("pandas", None)

    assert not destination.exists()

