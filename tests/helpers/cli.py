"""CLI-related test helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import pytest

from tnfr_lfs.cli import run_cli as _run_cli
from tnfr_lfs.cli import workflows as workflows_module


@contextmanager
def instrument_prepare_pack_context(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, int]]:
    """Patch ``_prepare_pack_context`` to track how often it is invoked."""

    calls = {"count": 0}
    original = workflows_module._prepare_pack_context

    def _instrumented_prepare(
        namespace: Any,
        config: Mapping[str, Any] | None,
        *,
        car_model: str,
    ) -> Any:
        calls["count"] += 1
        return original(namespace, config, car_model=car_model)

    monkeypatch.setattr(workflows_module, "_prepare_pack_context", _instrumented_prepare)
    try:
        yield calls
    finally:
        setattr(workflows_module, "_prepare_pack_context", original)


def run_cli_in_tmp(
    args: Sequence[str] | Iterable[str],
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str] | None = None,
    capture_output: bool = False,
) -> str | tuple[str, pytest.CaptureResult[str]]:
    """Execute ``run_cli`` from within ``tmp_path``.

    Parameters
    ----------
    args:
        Command-line arguments to pass to :func:`tnfr_lfs.cli.run_cli`.
    tmp_path:
        Temporary directory that should be treated as the working directory for
        the duration of the command invocation.
    monkeypatch:
        ``pytest`` fixture used to swap the process working directory.
    capsys:
        Optional ``pytest`` capturing fixture. Required when ``capture_output``
        is ``True``.
    capture_output:
        When ``True`` the return value includes the captured stdout/stderr via
        ``capsys.readouterr()``.
    """

    monkeypatch.chdir(tmp_path)
    result = _run_cli(list(args))

    if capture_output:
        if capsys is None:
            raise ValueError("capture_output=True requires providing the capsys fixture")
        return result, capsys.readouterr()

    return result


@dataclass
class DummyRecord:
    """Simple stand-in record used by parquet CLI tests."""

    value: int


def build_load_parquet_args(
    tmp_path: Path,
) -> tuple[tuple[Path, ...], dict[str, object], Path | None]:
    """Construct arguments for invoking ``_load_records`` with parquet data."""

    telemetry_path = tmp_path / "baseline.parquet"
    telemetry_path.write_bytes(b"PAR1\x00\x00\x00PAR1")
    return (telemetry_path,), {}, None


def build_persist_parquet_args(
    tmp_path: Path,
) -> tuple[tuple[list[DummyRecord], Path, str], dict[str, object], Path | None]:
    """Construct arguments for invoking ``_persist_records`` with parquet data."""

    destination = tmp_path / "baseline.parquet"
    records = [DummyRecord(1)]
    return (records, destination, "parquet"), {}, destination
