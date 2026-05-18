"""Tests for the LFS folder bulk-import helpers."""
from __future__ import annotations

from pathlib import Path

from lfs_telemetry import lfs_config
from lfs_telemetry.telemetry import observables

# ---------------------------------------------------------------------------
# find_lfs_car_info_bins
# ---------------------------------------------------------------------------


def test_find_lfs_car_info_bins_returns_empty_when_no_data_dir(tmp_path):
    # No data/ subfolder.
    assert lfs_config.find_lfs_car_info_bins(tmp_path) == []


def test_find_lfs_car_info_bins_lists_only_matching_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "FBM_CAR_info.bin").write_bytes(b"x")
    (data / "XRG_CAR_info.bin").write_bytes(b"x")
    (data / "readme.txt").write_text("ignore me")
    (data / "FBM.set").write_bytes(b"x")
    (data / "subdir").mkdir()

    found = lfs_config.find_lfs_car_info_bins(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["FBM_CAR_info.bin", "XRG_CAR_info.bin"]


def test_find_lfs_car_info_bins_case_insensitive(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "FBM_CAR_info.bin").write_bytes(b"x")
    (data / "fz5_car_info.bin").write_bytes(b"x")

    found = lfs_config.find_lfs_car_info_bins(tmp_path)
    assert len(found) == 2


# ---------------------------------------------------------------------------
# import_car_info_bins_from_lfs
# ---------------------------------------------------------------------------


def test_import_car_info_bins_from_lfs_invokes_import_per_file(
    tmp_path, monkeypatch,
):
    data = tmp_path / "data"
    data.mkdir()
    (data / "FBM_CAR_info.bin").write_bytes(b"x")
    (data / "XRG_CAR_info.bin").write_bytes(b"x")

    user_dir = tmp_path / "user_cars"

    calls: list[Path] = []

    def fake_import(src, *, target_key=None):
        calls.append(Path(src))
        dst = user_dir / Path(src).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
        return dst, object()

    monkeypatch.setattr(observables, "import_car_info_bin", fake_import)

    imported, failed = observables.import_car_info_bins_from_lfs(tmp_path)

    assert failed == []
    assert {Path(p).name for p in calls} == {
        "FBM_CAR_info.bin", "XRG_CAR_info.bin",
    }
    keys = sorted(k for k, _ in imported)
    assert keys == ["FBM", "XRG"]


def test_import_car_info_bins_from_lfs_collects_failures(
    tmp_path, monkeypatch,
):
    data = tmp_path / "data"
    data.mkdir()
    (data / "FBM_CAR_info.bin").write_bytes(b"x")
    (data / "BROKEN_CAR_info.bin").write_bytes(b"x")

    def fake_import(src, *, target_key=None):
        if "BROKEN" in str(src).upper():
            raise ValueError("not a valid export")
        return Path(src), object()

    monkeypatch.setattr(observables, "import_car_info_bin", fake_import)

    imported, failed = observables.import_car_info_bins_from_lfs(tmp_path)

    assert len(imported) == 1
    assert imported[0][0] == "FBM"
    assert len(failed) == 1
    assert failed[0][0].name == "BROKEN_CAR_info.bin"
    assert "not a valid export" in failed[0][1]


def test_import_car_info_bins_from_lfs_returns_empty_for_empty_dir(tmp_path):
    (tmp_path / "data").mkdir()
    imported, failed = observables.import_car_info_bins_from_lfs(tmp_path)
    assert imported == []
    assert failed == []
