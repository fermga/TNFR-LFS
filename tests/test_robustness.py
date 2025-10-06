from __future__ import annotations

from collections.abc import Sequence

import pytest

from tnfr_lfs.analysis.robustness import compute_session_robustness

from test_search import _build_bundle


@pytest.fixture
def multi_lap_session() -> tuple[Sequence, Sequence[int]]:
    bundles = [
        _build_bundle(0.0, 1.0, 0.80),
        _build_bundle(0.1, 2.0, 0.82),
        _build_bundle(0.2, 3.0, 0.78),
        _build_bundle(0.3, 1.5, 0.90),
        _build_bundle(0.4, 1.5, 0.90),
        _build_bundle(0.5, 1.5, 0.90),
    ]
    lap_indices = [0, 0, 0, 1, 1, 1]
    return bundles, lap_indices


def test_multi_lap_cov_computation(multi_lap_session: tuple[Sequence, Sequence[int]]) -> None:
    bundles, lap_indices = multi_lap_session
    thresholds = {"lap": {"delta_nfr": 0.5, "sense_index": 0.25}}
    metadata = [{"index": 0, "label": "Clasificación"}]

    robustness = compute_session_robustness(
        bundles,
        lap_indices=lap_indices,
        lap_metadata=metadata,
        thresholds=thresholds,
    )

    assert "laps" in robustness
    entries = {entry["label"]: entry for entry in robustness["laps"]}
    assert set(entries) == {"Clasificación", "Vuelta 2"}

    first_lap = entries["Clasificación"]
    assert first_lap["samples"] == 3
    delta_summary = first_lap["delta_nfr"]
    assert delta_summary["samples"] == 3
    assert delta_summary["coefficient_of_variation"] == pytest.approx(0.408248, rel=1e-6)
    assert delta_summary["ok"] is True

    si_summary = first_lap["sense_index"]
    assert si_summary["coefficient_of_variation"] == pytest.approx(0.020412, abs=5e-7)

    second_lap = entries["Vuelta 2"]
    assert second_lap["samples"] == 3
    assert second_lap["delta_nfr"]["coefficient_of_variation"] == pytest.approx(0.0)
    assert second_lap["sense_index"]["coefficient_of_variation"] == pytest.approx(0.0)
    assert second_lap["delta_nfr"]["ok"] is True
    assert second_lap["sense_index"]["ok"] is True
