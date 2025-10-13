from __future__ import annotations

import argparse
import builtins
import json
import sys
import types
from typing import Callable
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from tnfr_lfs.core.cache_settings import CacheOptions
from tnfr_lfs.cli import common as cli_common
from tnfr_lfs.cli import io as cli_io
from tnfr_lfs.cli.common import CliError
from tests.conftest import write_pyproject
from tests.helpers import (
    DummyRecord,
    build_load_parquet_args,
    build_persist_parquet_args,
    pandas_engine_failure,
)


@pytest.mark.parametrize(
    "cli_config_case",
    [
        pytest.param("cache-section", id="cache-section"),
        pytest.param("normalised-performance", id="normalised-performance"),
        pytest.param("explicit-sections", id="explicit-sections"),
        pytest.param("prefers-top-level", id="prefers-top-level"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "explicit_path",
    [pytest.param(False, id="via-chdir"), pytest.param(True, id="explicit-path")],
)
def test_load_cli_config_from_pyproject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cli_config_case: tuple[
        str, dict[str, object], Callable[[Path], None] | None
    ],
    explicit_path: bool,
) -> None:
    toml_text, expected_sections, setup = cli_config_case
    pyproject_path = write_pyproject(tmp_path, toml_text)
    if callable(setup):
        setup(tmp_path)

    if explicit_path:
        config = cli_io.load_cli_config(pyproject_path)
    else:
        monkeypatch.chdir(tmp_path)
        config = cli_io.load_cli_config()

    assert config["_config_path"] == str(pyproject_path.resolve())
    for section, expected in expected_sections.items():
        assert config[section] == expected


@pytest.mark.parametrize(
    ("namespace_factory", "expected"),
    [
        (lambda: argparse.Namespace(), None),
        (lambda: argparse.Namespace(cache_options=types.SimpleNamespace()), None),
        (
            lambda: argparse.Namespace(
                cache_options=CacheOptions(
                    enable_delta_cache=True,
                    nu_f_cache_size=32,
                    telemetry_cache_size=16,
                )
            ),
            16,
        ),
    ],
)
def test_resolve_cache_size(
    namespace_factory: Callable[[], argparse.Namespace], expected: int | None
) -> None:
    namespace = namespace_factory()

    assert (
        cli_common.resolve_cache_size(namespace, "telemetry_cache_size") == expected
    )


@pytest.mark.parametrize(
    ("cache_options", "expected_cache_size"),
    [
        pytest.param(None, None, id="returns-expected-object"),
        pytest.param(
            CacheOptions(
                enable_delta_cache=True,
                nu_f_cache_size=128,
                telemetry_cache_size=0,
            ),
            0,
            id="threads-cache-size",
        ),
    ],
)
def test_load_records_from_namespace_prefers_replay_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cache_options: CacheOptions | None,
    expected_cache_size: int | None,
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"dummy")
    expected = [object()]
    captured: dict[str, Path | int | None] = {}

    def _fake_loader(path: Path, *, cache_size: int | None = None):
        captured["path"] = path
        captured["cache_size"] = cache_size
        return expected

    monkeypatch.setattr(cli_common, "_load_replay_bundle", _fake_loader)
    namespace = argparse.Namespace(replay_csv_bundle=bundle_path, telemetry=None)
    if cache_options is not None:
        namespace.cache_options = cache_options

    records, resolved = cli_common._load_records_from_namespace(namespace)

    assert records is expected
    assert resolved == bundle_path
    assert namespace.telemetry == bundle_path
    assert captured["path"] == bundle_path
    assert captured["cache_size"] == expected_cache_size


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


@pytest.mark.parametrize(
    ("target", "arguments_builder", "mode"),
    [
        (cli_io._load_records, build_load_parquet_args, "load"),
        (cli_io._persist_records, build_persist_parquet_args, "persist"),
    ],
    ids=["load", "persist"],
)
def test_parquet_engine_requires_compatible_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target,
    arguments_builder,
    mode: str,
) -> None:
    args, kwargs, destination = arguments_builder(tmp_path)

    pandas_engine_failure(monkeypatch, mode)

    with pytest.raises(RuntimeError, match="compatible engine"):
        target(*args, **kwargs)

    if destination is not None:
        assert not destination.exists()


@pytest.mark.parametrize(
    ("target", "arguments_builder"),
    [
        (cli_io._load_records, build_load_parquet_args),
        (cli_io._persist_records, build_persist_parquet_args),
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

