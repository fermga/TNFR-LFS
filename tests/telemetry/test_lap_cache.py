"""Tests for the on-disk lap cache (pickle-based, no compression)."""

from __future__ import annotations

import os
import pickle

import pandas as pd
import pytest

from lfs_telemetry.telemetry import lap_cache


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect the cache directory to a per-test tmp folder."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    # macOS / Linux fallbacks too
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        lap_cache, "cache_dir",
        lambda: _ensured(tmp_path / "cache"),
    )
    yield


def _ensured(p):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _csv(tmp_path, name="lap.csv", body="a,b\n1,2\n3,4\n"):
    p = tmp_path / name
    p.write_text(body)
    return p


def _frames():
    raw = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    enriched = raw.assign(c=[10, 20])
    return raw, enriched


def test_save_then_load_roundtrip(tmp_path):
    src = _csv(tmp_path)
    raw, enriched = _frames()
    lap_cache.save(src, raw, enriched)

    loaded = lap_cache.load(src)
    assert loaded is not None
    raw2, enriched2 = loaded
    pd.testing.assert_frame_equal(raw2, raw)
    pd.testing.assert_frame_equal(enriched2, enriched)


def test_miss_when_file_missing(tmp_path):
    assert lap_cache.load(tmp_path / "nope.csv") is None


def test_invalidates_on_size_change(tmp_path):
    src = _csv(tmp_path)
    raw, enriched = _frames()
    lap_cache.save(src, raw, enriched)
    assert lap_cache.load(src) is not None

    src.write_text("a,b\n1,2\n3,4\n5,6\n")  # different bytes/size
    assert lap_cache.load(src) is None


def test_invalidates_on_mtime_change(tmp_path):
    src = _csv(tmp_path)
    raw, enriched = _frames()
    lap_cache.save(src, raw, enriched)

    # Same size but different mtime → different key.
    new_mtime = src.stat().st_mtime_ns + 10_000_000_000  # +10s
    os.utime(src, ns=(new_mtime, new_mtime))
    assert lap_cache.load(src) is None


def test_corrupt_entry_is_dropped(tmp_path):
    src = _csv(tmp_path)
    raw, enriched = _frames()
    lap_cache.save(src, raw, enriched)
    cache_file = next(lap_cache.cache_dir().glob("*.pkl"))
    cache_file.write_bytes(b"not a pickle")

    assert lap_cache.load(src) is None
    assert not cache_file.exists()  # dropped


def test_format_version_mismatch_invalidates(tmp_path, monkeypatch):
    src = _csv(tmp_path)
    raw, enriched = _frames()
    lap_cache.save(src, raw, enriched)
    cache_file = next(lap_cache.cache_dir().glob("*.pkl"))

    # Hand-craft a payload with wrong format version.
    with open(cache_file, "wb") as fp:
        pickle.dump(
            {"format": 999, "raw": raw, "enriched": enriched},
            fp, protocol=pickle.HIGHEST_PROTOCOL,
        )
    assert lap_cache.load(src) is None


def test_clear_removes_entries(tmp_path):
    src1 = _csv(tmp_path, name="a.csv")
    src2 = _csv(tmp_path, name="b.csv", body="a,b\n9,8\n")
    raw, enriched = _frames()
    lap_cache.save(src1, raw, enriched)
    lap_cache.save(src2, raw, enriched)

    n = lap_cache.clear()
    assert n == 2
    assert lap_cache.load(src1) is None
    assert lap_cache.load(src2) is None


def test_lap_telemetry_uses_cache(tmp_path, monkeypatch):
    """End-to-end: second LapTelemetry.from_csv hits the disk cache."""
    from lfs_telemetry.telemetry import derived
    from lfs_telemetry.telemetry.lap import LapTelemetry

    # Minimal CSV the loader accepts.
    csv = tmp_path / "stint.csv"
    csv.write_text(
        "# schema=v1\n"
        "t_s,car,player_id\n"
        "0.0,FBM,1\n"
        "0.05,FBM,1\n"
    )

    calls = {"n": 0}
    real = derived.enrich_dataframe

    def counting(df, spec):
        calls["n"] += 1
        return real(df, spec)

    monkeypatch.setattr(derived, "enrich_dataframe", counting)
    # The lap module imported the symbol directly, patch there too.
    from lfs_telemetry.telemetry import lap as lap_mod
    monkeypatch.setattr(lap_mod, "enrich_dataframe", counting)

    lap1 = LapTelemetry.from_csv(csv)
    _ = lap1.enriched
    assert calls["n"] == 1

    # Second load should pull (raw, enriched) from disk → no enrich call.
    lap2 = LapTelemetry.from_csv(csv)
    _ = lap2.enriched
    assert calls["n"] == 1
