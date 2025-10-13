from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import dataclass
from typing import Any, Callable
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from tnfr_core.cache_settings import CacheOptions
from tnfr_lfs.cli import common as cli_common
from tnfr_lfs.cli import io as cli_io
from tnfr_lfs.cli.common import CliError
from tests.conftest import write_pyproject
from tests.helpers import DummyRecord, build_load_parquet_args, build_persist_parquet_args


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


@dataclass(frozen=True)
class CacheCase:
    label: str
    options: CacheOptions | None
    expected_size: int | None


@dataclass(frozen=True)
class LoadRecordsCase:
    label: str
    setup: Callable[
        [pytest.MonkeyPatch, Path, CacheCase],
        tuple[argparse.Namespace, Path | None, dict[str, Any]],
    ]
    verify: Callable[[object, argparse.Namespace, Path | None, dict[str, Any], CacheCase], None] | None = None
    expect_exception: type[Exception] | None = None


def _setup_replay_bundle_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache_case: CacheCase
) -> tuple[argparse.Namespace, Path, dict[str, Any]]:
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"dummy")
    expected_records: list[object] = [object()]
    captured: dict[str, Path | int | None] = {}

    def _fake_loader(path: Path, *, cache_size: int | None = None):
        captured["path"] = path
        captured["cache_size"] = cache_size
        return expected_records

    monkeypatch.setattr(cli_common, "_load_replay_bundle", _fake_loader)
    namespace = argparse.Namespace(replay_csv_bundle=bundle_path, telemetry=None)
    if cache_case.options is not None:
        namespace.cache_options = cache_case.options

    return namespace, bundle_path, {"captured": captured, "expected_records": expected_records}


def _verify_replay_bundle_case(
    records: object,
    namespace: argparse.Namespace,
    expected_path: Path | None,
    context: dict[str, Any],
    cache_case: CacheCase,
) -> None:
    assert isinstance(records, list)
    assert records is context["expected_records"]
    assert expected_path is not None
    captured = context["captured"]
    assert captured["path"] == expected_path
    assert captured["cache_size"] == cache_case.expected_size
    assert namespace.telemetry == expected_path


def _setup_csv_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache_case: CacheCase
) -> tuple[argparse.Namespace, Path, dict[str, Any]]:
    telemetry_path = tmp_path / "stint.csv"
    telemetry_path.write_text("timestamp")
    captured: dict[str, Path] = {}

    class DummyClient:
        def ingest(self, source: Path):
            captured["source"] = source
            return ["ok"]

    monkeypatch.setattr(cli_io, "OutSimClient", lambda: DummyClient())
    namespace = argparse.Namespace(replay_csv_bundle=None, telemetry=telemetry_path)
    if cache_case.options is not None:
        namespace.cache_options = cache_case.options

    return namespace, telemetry_path, {"captured": captured}


def _verify_csv_case(
    records: object,
    namespace: argparse.Namespace,
    expected_path: Path | None,
    context: dict[str, Any],
    _cache_case: CacheCase,
) -> None:
    assert expected_path is not None
    assert records == ["ok"]
    assert namespace.telemetry == expected_path
    assert context["captured"]["source"] == expected_path


def _setup_json_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache_case: CacheCase
) -> tuple[argparse.Namespace, Path, dict[str, Any]]:
    telemetry_path = tmp_path / "session.json"
    telemetry_path.write_text(json.dumps([{"timestamp": 1.5, "gear": 3}]))

    class DummyRecord:
        def __init__(self, **payload: float):
            self.payload = payload

    monkeypatch.setattr(cli_io, "TelemetryRecord", DummyRecord)
    namespace = argparse.Namespace(replay_csv_bundle=None, telemetry=telemetry_path)
    if cache_case.options is not None:
        namespace.cache_options = cache_case.options

    return namespace, telemetry_path, {"record_type": DummyRecord}


def _verify_json_case(
    records: object,
    namespace: argparse.Namespace,
    expected_path: Path | None,
    context: dict[str, Any],
    _cache_case: CacheCase,
) -> None:
    assert expected_path is not None
    assert namespace.telemetry == expected_path
    assert isinstance(records, list)
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, context["record_type"])
    assert record.payload["timestamp"] == 1.5
    assert record.payload["gear"] == 3


def _setup_missing_path_case(
    _monkeypatch: pytest.MonkeyPatch, _tmp_path: Path, cache_case: CacheCase
) -> tuple[argparse.Namespace, None, dict[str, Any]]:
    namespace = argparse.Namespace(replay_csv_bundle=None, telemetry=None)
    if cache_case.options is not None:
        namespace.cache_options = cache_case.options
    return namespace, None, {}


CACHE_CASES = [
    pytest.param(CacheCase("cache-none", None, None), id="cache-none"),
    pytest.param(
        CacheCase(
            "cache-threads",
            CacheOptions(
                enable_delta_cache=True,
                nu_f_cache_size=128,
                telemetry_cache_size=0,
            ),
            0,
        ),
        id="cache-threads",
    ),
]


LOAD_RECORDS_CASES = [
    pytest.param(
        LoadRecordsCase(
            label="replay-bundle",
            setup=_setup_replay_bundle_case,
            verify=_verify_replay_bundle_case,
        ),
        id="replay-bundle",
    ),
    pytest.param(
        LoadRecordsCase(
            label="csv",
            setup=_setup_csv_case,
            verify=_verify_csv_case,
        ),
        id="csv",
    ),
    pytest.param(
        LoadRecordsCase(
            label="json",
            setup=_setup_json_case,
            verify=_verify_json_case,
        ),
        id="json",
    ),
    pytest.param(
        LoadRecordsCase(
            label="missing-path",
            setup=_setup_missing_path_case,
            expect_exception=CliError,
        ),
        id="missing-path",
    ),
]


@pytest.mark.parametrize("cache_case", CACHE_CASES)
@pytest.mark.parametrize("case", LOAD_RECORDS_CASES)
def test_load_records_from_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cache_case: CacheCase,
    case: LoadRecordsCase,
) -> None:
    namespace, expected_path, context = case.setup(monkeypatch, tmp_path, cache_case)

    if case.expect_exception is not None:
        with pytest.raises(case.expect_exception):
            cli_common._load_records_from_namespace(namespace)
        return

    records, resolved_path = cli_common._load_records_from_namespace(namespace)

    assert resolved_path == expected_path
    if case.verify is not None:
        case.verify(records, namespace, expected_path, context, cache_case)



def _assert_parquet_requirement_failure(
    target: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    expected_match: str,
    destination: Path | None,
) -> None:
    with pytest.raises(RuntimeError, match=expected_match):
        target(*args, **kwargs)
    if destination is not None:
        assert not destination.exists()

@pytest.mark.parametrize(
    ("target", "arguments_builder", "mode"),
    [
        pytest.param(
            cli_io._load_records,
            build_load_parquet_args,
            "load",
            id="load",
        ),
        pytest.param(
            cli_io._persist_records,
            build_persist_parquet_args,
            "persist",
            id="persist",
        ),
    ],
)
@pytest.mark.parametrize(
    ("simulate", "expected_match"),
    [
        pytest.param(
            "parquet_engine_failure",
            "compatible engine",
            id="engine-incompatibility",
        ),
        pytest.param(
            "pandas_absence",
            "requires the 'pandas' package",
            id="missing-pandas",
        ),
    ],
)
def test_parquet_requirements(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    target,
    arguments_builder,
    mode: str,
    simulate: str,
    expected_match: str,
) -> None:
    args, kwargs, destination = arguments_builder(tmp_path)

    simulator = request.getfixturevalue(simulate)
    simulator(mode)

    _assert_parquet_requirement_failure(
        target,
        args,
        kwargs,
        expected_match=expected_match,
        destination=destination,
    )

