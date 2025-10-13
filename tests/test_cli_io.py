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
from tests.helpers import DummyRecord, build_load_parquet_args, build_persist_parquet_args


@pytest.mark.parametrize(
    ("toml_text", "expected_sections"),
    [
        (
            """
            [tool.tnfr_lfs.cache]
            cache_enabled = "yes"
            nu_f_cache_size = "48"
            """,
            {
                "performance": CacheOptions(
                    enable_delta_cache=True,
                    nu_f_cache_size=48,
                    telemetry_cache_size=48,
                    recommender_cache_size=48,
                ).to_performance_config(),
            },
        ),
        (
            """
            [tool.tnfr_lfs.performance]
            cache_enabled = "no"
            max_cache_size = "12"
            telemetry_buffer_size = 42
            """,
            {
                "performance": {
                    **CacheOptions(
                        enable_delta_cache=False,
                        nu_f_cache_size=0,
                        telemetry_cache_size=0,
                        recommender_cache_size=0,
                    ).to_performance_config(),
                    "telemetry_buffer_size": 42,
                },
            },
        ),
        (
            """
            [tool.tnfr_lfs.core]
            host = "192.0.2.1"

            [tool.tnfr_lfs.performance]
            cache_enabled = true
            max_cache_size = 48
            """,
            {
                "core": {"host": "192.0.2.1"},
                "performance": CacheOptions(
                    enable_delta_cache=True,
                    nu_f_cache_size=48,
                    telemetry_cache_size=48,
                    recommender_cache_size=48,
                ).to_performance_config(),
            },
        ),
        (
            """
            [tool.tnfr_lfs.logging]
            level = "warning"
            format = "text"
            output = "stdout"
            """,
            {
                "__setup__": lambda root: (
                    (root / "secondary").mkdir(),
                    write_pyproject(root / "secondary", ""),
                ),
                "logging": {
                    "level": "warning",
                    "format": "text",
                    "output": "stdout",
                },
            },
        ),
    ],
    ids=[
        "cache-section",
        "normalised-performance",
        "explicit-sections",
        "prefers-top-level",
    ],
)
def test_load_cli_config_from_pyproject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    toml_text: str,
    expected_sections: dict[str, object],
) -> None:
    pyproject_path = write_pyproject(tmp_path, toml_text)
    sections = dict(expected_sections)
    setup = sections.pop("__setup__", None)
    if callable(setup):
        setup(tmp_path)

    monkeypatch.chdir(tmp_path)
    config = cli_io.load_cli_config()

    assert config["_config_path"] == str(pyproject_path.resolve())
    for section, expected in sections.items():
        assert config[section] == expected


@pytest.mark.parametrize(
    ("toml_text", "expected_sections"),
    [
        (
            """
            [tool.tnfr_lfs.logging]
            level = "debug"
            """,
            {"logging": {"level": "debug"}},
        ),
    ],
)
def test_load_cli_config_accepts_explicit_pyproject_path(
    tmp_path: Path, toml_text: str, expected_sections: dict[str, object]
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    pyproject_path = write_pyproject(config_dir, toml_text)

    config = cli_io.load_cli_config(pyproject_path)

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


def test_persist_records_parquet_requires_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "baseline.parquet"
    records = [DummyRecord(1)]

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

