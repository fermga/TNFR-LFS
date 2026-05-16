"""Tests for :mod:`lfs_telemetry.studio.widgets.racing_line_loader`."""

from __future__ import annotations

from pathlib import Path

from lfs_telemetry.studio.widgets.racing_line_loader import (
    find_racing_line_for_track,
    load_racing_line,
)


def test_missing_file_returns_empty(tmp_path: Path):
    line = load_racing_line(tmp_path / "does_not_exist.csv")
    assert line.is_empty
    assert line.points == []


def test_load_basic_csv(tmp_path: Path):
    f = tmp_path / "TEST_racing.csv"
    f.write_text(
        "x_center_m,y_center_m\n0,0\n10,0\n10,5\n0,5\n",
        encoding="utf-8",
    )
    line = load_racing_line(f)
    assert not line.is_empty
    assert len(line.points) == 4
    assert line.bbox == (0.0, 0.0, 10.0, 5.0)


def test_skip_invalid_rows(tmp_path: Path):
    f = tmp_path / "TEST_racing.csv"
    f.write_text(
        "x_center_m,y_center_m\n0,0\nbad,row\n5,5\n",
        encoding="utf-8",
    )
    line = load_racing_line(f)
    assert len(line.points) == 2


def test_find_by_track_name(tmp_path: Path):
    (tmp_path / "BL1_racing.csv").write_text(
        "x_center_m,y_center_m\n1,1\n2,2\n", encoding="utf-8",
    )
    line = find_racing_line_for_track(tmp_path, "BL1")
    assert not line.is_empty
    miss = find_racing_line_for_track(tmp_path, "ZZ9")
    assert miss.is_empty
    none_track = find_racing_line_for_track(tmp_path, None)
    assert none_track.is_empty
