"""Unit tests for :mod:`lfs_telemetry.lfs_paths`."""
from __future__ import annotations

from pathlib import Path

import pytest

from lfs_telemetry import lfs_paths


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Replace ``lfs_paths._settings`` with a fresh per-test QSettings.

    Qt caches QSettings backends globally so simply changing
    ``QSettings.setPath`` between tests is not enough — the in-memory
    keys from a previous test leak into the next one. Patching the
    factory guarantees each test starts with an empty store.
    """
    from PySide6.QtCore import QSettings
    ini_path = tmp_path / "lfs-test.ini"
    monkeypatch.setattr(
        lfs_paths, "_settings",
        lambda: QSettings(str(ini_path), QSettings.Format.IniFormat),
    )
    yield


def _make_valid_lfs(tmp_path) -> Path:
    """Create a folder that looks like an LFS install."""
    root = tmp_path / "LFS"
    root.mkdir()
    (root / "LFS.exe").write_bytes(b"")
    (root / "cfg.txt").write_text("OutSim Mode 0\n", encoding="latin-1")
    return root


def test_is_valid_lfs_dir_accepts_folder_with_lfs_exe(tmp_path):
    root = _make_valid_lfs(tmp_path)
    assert lfs_paths.is_valid_lfs_dir(root)


def test_is_valid_lfs_dir_rejects_missing_markers(tmp_path):
    empty = tmp_path / "nope"
    empty.mkdir()
    assert not lfs_paths.is_valid_lfs_dir(empty)


def test_is_valid_lfs_dir_rejects_none():
    assert not lfs_paths.is_valid_lfs_dir(None)


def test_set_get_roundtrip(tmp_path):
    root = _make_valid_lfs(tmp_path)
    lfs_paths.set_lfs_dir(root)
    assert lfs_paths.get_lfs_dir() == root


def test_get_returns_none_when_saved_path_invalid(tmp_path):
    lfs_paths.set_lfs_dir(tmp_path / "ghost")
    assert lfs_paths.get_lfs_dir() is None


def test_forget_lfs_dir(tmp_path):
    root = _make_valid_lfs(tmp_path)
    lfs_paths.set_lfs_dir(root)
    lfs_paths.forget_lfs_dir()
    assert lfs_paths.get_lfs_dir() is None


def test_first_run_complete_roundtrip():
    assert not lfs_paths.first_run_complete()
    lfs_paths.mark_first_run_complete()
    assert lfs_paths.first_run_complete()


def test_autodetect_prefers_saved_path(tmp_path, monkeypatch):
    saved = _make_valid_lfs(tmp_path)
    lfs_paths.set_lfs_dir(saved)
    # Pretend none of the static candidates exist.
    monkeypatch.setattr(lfs_paths, "_STATIC_CANDIDATES", ())
    monkeypatch.setattr(lfs_paths, "_registry_lfs_dir", lambda: None)
    assert lfs_paths.autodetect_lfs_dir() == saved


def test_autodetect_falls_back_to_static_candidates(tmp_path, monkeypatch):
    candidate = _make_valid_lfs(tmp_path)
    monkeypatch.setattr(
        lfs_paths, "_STATIC_CANDIDATES", (tmp_path / "ghost", candidate),
    )
    monkeypatch.setattr(lfs_paths, "_registry_lfs_dir", lambda: None)
    assert lfs_paths.autodetect_lfs_dir() == candidate


def test_autodetect_candidates_dedup(tmp_path, monkeypatch):
    root = _make_valid_lfs(tmp_path)
    lfs_paths.set_lfs_dir(root)
    monkeypatch.setattr(lfs_paths, "_STATIC_CANDIDATES", (root, root))
    monkeypatch.setattr(lfs_paths, "_registry_lfs_dir", lambda: root)
    cands = lfs_paths.autodetect_candidates()
    assert cands.count(root) == 1


def test_path_helpers(tmp_path):
    root = _make_valid_lfs(tmp_path)
    assert lfs_paths.lfs_exe(root) == root / "LFS.exe"
    assert lfs_paths.cfg_path(root) == root / "cfg.txt"
    assert lfs_paths.data_dir(root) == root / "data"
    assert lfs_paths.veh_dir(root) == root / "data" / "veh"
    assert lfs_paths.setups_dir(root) == root / "data" / "setups"
    assert lfs_paths.setups_dir(root, "xrt") == root / "data" / "setups" / "XRT"
    assert (lfs_paths.car_info_bin_path(root, "fbm")
            == root / "data" / "FBM_CAR_info.bin")
