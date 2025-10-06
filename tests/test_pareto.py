from __future__ import annotations

from tnfr_lfs.recommender.pareto import ParetoPoint, pareto_front


def test_pareto_front_filters_known_optima() -> None:
    candidates = [
        ParetoPoint(
            decision_vector={"label": "Setup A"},
            score=0.82,
            breakdown={"lap_time": 100.0, "tyre_wear": 8.0},
            objective_breakdown={},
        ),
        ParetoPoint(
            decision_vector={"label": "Setup B"},
            score=0.95,
            breakdown={"lap_time": 99.0, "tyre_wear": 7.0},
            objective_breakdown={},
        ),
        ParetoPoint(
            decision_vector={"label": "Setup C"},
            score=0.90,
            breakdown={"lap_time": 101.0, "tyre_wear": 6.0},
            objective_breakdown={},
        ),
        ParetoPoint(
            decision_vector={"label": "Setup D"},
            score=0.70,
            breakdown={"lap_time": 105.0, "tyre_wear": 9.0},
            objective_breakdown={},
        ),
    ]

    front = pareto_front(candidates)
    labels = [point.decision_vector["label"] for point in front]

    assert labels == ["Setup B", "Setup C"]
    assert all(entry in front for entry in candidates[1:3])
    assert candidates[0] not in front
    assert candidates[3] not in front
