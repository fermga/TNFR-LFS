from __future__ import annotations

import argparse
import builtins
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import pytest

pytest.importorskip("numpy")

from tnfr_lfs.core.cache_settings import CacheOptions
from tnfr_lfs.cli import common as cli_common
from tnfr_lfs.cli import io as cli_io
from tnfr_lfs.cli.common import CliError


def test_load_cli_config_promotes_legacy_cache(tmp_path: Path) -> None:
    config_path = tmp_path / "tnfr_lfs.toml"
    config_path.write_text(
        dedent(
            """
            [cache]
            cache_enabled = "yes"
            nu_f_cache_size = "48"
            """
        ),
        encoding="utf8",
    )

    config = cli_io.load_cli_config(config_path)

    assert config["_config_path"] == str(config_path.resolve())
    performance_cfg = config["performance"]
    assert performance_cfg["cache_enabled"] is True
    assert performance_cfg["max_cache_size"] == 48
    assert performance_cfg["nu_f_cache_size"] == 48
    assert performance_cfg["telemetry_cache_size"] == 48
    assert performance_cfg["recommender_cache_size"] == 48


def test_load_cli_config_normalises_performance_section(tmp_path: Path) -> None:
    config_path = tmp_path / "tnfr_lfs.toml"
    config_path.write_text(
        dedent(
            """
            [performance]
            cache_enabled = "no"
            max_cache_size = "12"
            telemetry_buffer_size = 42
            """
        ),
        encoding="utf8",
    )

    config = cli_io.load_cli_config(config_path)

    assert config["performance"]["cache_enabled"] is False
    assert config["performance"]["max_cache_size"] == 0
    assert config["performance"]["nu_f_cache_size"] == 0
    assert config["performance"]["telemetry_cache_size"] == 0
    assert config["performance"]["recommender_cache_size"] == 0
    assert config["performance"]["telemetry_buffer_size"] == 42


def test_resolve_cache_size_returns_none_without_options() -> None:
    namespace = argparse.Namespace()

    assert cli_common.resolve_cache_size(namespace, "telemetry_cache_size") is None


def test_resolve_cache_size_handles_missing_attribute() -> None:
    namespace = argparse.Namespace()
    namespace.cache_options = types.SimpleNamespace()

    assert cli_common.resolve_cache_size(namespace, "telemetry_cache_size") is None


def test_resolve_cache_size_returns_configured_value() -> None:
    namespace = argparse.Namespace()
    namespace.cache_options = CacheOptions(
        enable_delta_cache=True,
        nu_f_cache_size=32,
        telemetry_cache_size=16,
    )

    assert cli_common.resolve_cache_size(namespace, "telemetry_cache_size") == 16


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


def _build_load_parquet_args(tmp_path: Path) -> tuple[tuple[Path, ...], dict[str, object], Path | None]:
    telemetry_path = tmp_path / "baseline.parquet"
    telemetry_path.write_bytes(b"PAR1\x00\x00\x00PAR1")
    return (telemetry_path,), {}, None


def _build_persist_parquet_args(
    tmp_path: Path,
) -> tuple[tuple[list[_DummyRecord], Path, str], dict[str, object], Path | None]:
    destination = tmp_path / "baseline.parquet"
    records = [_DummyRecord(1)]
    return (records, destination, "parquet"), {}, destination


@pytest.mark.parametrize(
    ("target", "arguments_builder"),
    [
        (cli_io._load_records, _build_load_parquet_args),
        (cli_io._persist_records, _build_persist_parquet_args),
    ],
    ids=["load", "persist"],
)
def test_parquet_operations_require_pandas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target,
    arguments_builder,
) -> None:
    args, kwargs, destination = arguments_builder(tmp_path)

    original = sys.modules.pop("pandas", None)
    original_import = builtins.__import__

    def fake_import(name: str, *f_args, **f_kwargs):  # type: ignore[no-untyped-def]
        if name == "pandas":
            raise ModuleNotFoundError("No module named 'pandas'")
        return original_import(name, *f_args, **f_kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        with pytest.raises(RuntimeError, match="requires the 'pandas' package"):
            target(*args, **kwargs)
    finally:
        if original is not None:
            sys.modules["pandas"] = original
        else:
            sys.modules.pop("pandas", None)

    if destination is not None:
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

