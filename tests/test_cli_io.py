from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from tnfr_lfs.cli import io as cli_io


def test_load_records_from_namespace_prefers_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"dummy")
    expected = [object()]
    monkeypatch.setattr(cli_io, "_load_replay_bundle", lambda path: expected)
    namespace = argparse.Namespace(replay_csv_bundle=bundle_path, telemetry=None)

    records, resolved = cli_io._load_records_from_namespace(namespace)

    assert records is expected
    assert resolved == bundle_path
    assert namespace.telemetry == bundle_path


def test_load_records_from_namespace_requires_path() -> None:
    namespace = argparse.Namespace(replay_csv_bundle=None, telemetry=None)
    with pytest.raises(SystemExit):
        cli_io._load_records_from_namespace(namespace)


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

