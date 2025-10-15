from __future__ import annotations

from collections.abc import Sequence

import pytest

from tnfr_lfs.analysis.robustness import _lap_groups, compute_session_robustness

from tests.helpers import build_balanced_bundle


@pytest.fixture
def multi_lap_session() -> tuple[Sequence, Sequence[int]]:
    bundles = [
        build_balanced_bundle(0.0, 1.0, 0.80),
        build_balanced_bundle(0.1, 2.0, 0.82),
        build_balanced_bundle(0.2, 3.0, 0.78),
        build_balanced_bundle(0.3, 1.5, 0.90),
        build_balanced_bundle(0.4, 1.5, 0.90),
        build_balanced_bundle(0.5, 1.5, 0.90),
    ]
    lap_indices = [0, 0, 0, 1, 1, 1]
    return bundles, lap_indices


def test_multi_lap_cov_computation(multi_lap_session: tuple[Sequence, Sequence[int]]) -> None:
    bundles, lap_indices = multi_lap_session
    thresholds = {"lap": {"delta_nfr": 0.5, "sense_index": 0.25}}
    metadata = [
        {"index": 0, "label": "Qualifying"},
        {"index": 1, "label": "Lap 2"},
    ]

    robustness = compute_session_robustness(
        bundles,
        lap_indices=lap_indices,
        lap_metadata=metadata,
        thresholds=thresholds,
    )

    assert "laps" in robustness
    entries = {entry["label"]: entry for entry in robustness["laps"]}
    assert set(entries) == {"Qualifying", "Lap 2"}

    first_lap = entries["Qualifying"]
    assert first_lap["samples"] == 3
    delta_summary = first_lap["delta_nfr"]
    assert delta_summary["samples"] == 3
    assert delta_summary["coefficient_of_variation"] == pytest.approx(0.408248, rel=1e-6)
    assert delta_summary["ok"] is True

    si_summary = first_lap["sense_index"]
    assert si_summary["coefficient_of_variation"] == pytest.approx(0.020412, abs=5e-7)

    second_lap = entries["Lap 2"]
    assert second_lap["samples"] == 3
    assert second_lap["delta_nfr"]["coefficient_of_variation"] == pytest.approx(0.0)
    assert second_lap["sense_index"]["coefficient_of_variation"] == pytest.approx(0.0)
    assert second_lap["delta_nfr"]["ok"] is True
    assert second_lap["sense_index"]["ok"] is True


def test_lap_groups_with_metadata() -> None:
    bundles = [object() for _ in range(5)]
    lap_indices = [0, 0, 1, 2, 2]
    metadata = [
        {"index": 0, "label": "Warmup"},
        {"index": 2, "label": "Final"},
    ]

    groups = _lap_groups(bundles, lap_indices, metadata)

    assert groups == [
        (0, "Warmup", [0, 1]),
        (2, "Final", [3, 4]),
        (1, "Lap 2", [2]),
    ]


def test_lap_groups_without_metadata() -> None:
    bundles = [object() for _ in range(4)]
    lap_indices = [0, 0, 1, 1]

    groups = _lap_groups(bundles, lap_indices, [])

    assert groups == [
        (0, "Lap 1", [0, 1]),
        (1, "Lap 2", [2, 3]),
    ]
