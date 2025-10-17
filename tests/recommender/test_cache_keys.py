from __future__ import annotations

from typing import Mapping, Sequence

from tnfr_core.runtime.shared import LRUCache

from tnfr_lfs.recommender.search import (
    DecisionSpace,
    DecisionVariable,
    SetupPlanner,
    evaluate_candidate,
    objective_score,
)
from tests.helpers.epi import build_balanced_bundle


def _demo_space() -> DecisionSpace:
    return DecisionSpace(
        car_model="demo",
        variables=(
            DecisionVariable("front_ride_height", -5.0, 5.0, 1.0),
            DecisionVariable("rear_ride_height", -5.0, 5.0, 1.0),
            DecisionVariable("rear_wing_angle", -3.0, 3.0, 1.0),
        ),
    )


def _baseline_samples() -> list:
    return [
        build_balanced_bundle(0.0, delta_nfr=4.0, si=0.6),
        build_balanced_bundle(0.1, delta_nfr=-3.0, si=0.58),
    ]


def test_evaluate_candidate_reuses_cache_for_permuted_vectors() -> None:
    space = _demo_space()
    baseline = _baseline_samples()
    cache: LRUCache[
        tuple[tuple[str, float], ...],
        tuple[float, tuple, Mapping[str, float], Mapping[str, float]],
    ] = LRUCache(maxsize=8)

    call_count = 0

    def simulator(vector: Mapping[str, float], _: Sequence) -> Sequence:
        nonlocal call_count
        call_count += 1
        return baseline

    first = evaluate_candidate(
        space,
        {
            "front_ride_height": 1.0,
            "rear_ride_height": -1.0,
            "rear_wing_angle": 0.5,
        },
        baseline,
        simulator=simulator,
        cache=cache,
    )

    permuted = dict(
        [
            ("rear_wing_angle", 0.5),
            ("rear_ride_height", -1.0),
            ("front_ride_height", 1.0),
        ]
    )
    second = evaluate_candidate(
        space,
        permuted,
        baseline,
        simulator=simulator,
        cache=cache,
    )

    assert call_count == 1
    assert second.score == first.score


def test_compute_sensitivities_reuses_cache_for_permuted_vectors() -> None:
    space = _demo_space()
    planner = SetupPlanner(decision_library={space.car_model: space})
    baseline = _baseline_samples()
    telemetry = list(baseline)
    cache: LRUCache[
        tuple[tuple[str, float], ...],
        tuple[float, tuple, Mapping[str, float]],
    ] = LRUCache(maxsize=16)

    call_count = 0

    def simulator(vector: Mapping[str, float], _: Sequence) -> Sequence:
        nonlocal call_count
        call_count += 1
        return baseline

    base_vector = space.clamp(
        {
            "front_ride_height": 0.0,
            "rear_ride_height": 0.0,
            "rear_wing_angle": 0.0,
        }
    )
    score = objective_score(telemetry)

    planner._compute_sensitivities(
        vector=base_vector,
        telemetry=telemetry,
        baseline=baseline,
        microsectors=None,
        simulator=simulator,
        space=space,
        cache=cache,
        score=score,
        session_weights=None,
        session_hints=None,
    )

    baseline_calls = call_count
    assert baseline_calls > 0

    permuted_vector = dict(reversed(base_vector.items()))

    planner._compute_sensitivities(
        vector=permuted_vector,
        telemetry=telemetry,
        baseline=baseline,
        microsectors=None,
        simulator=simulator,
        space=space,
        cache=cache,
        score=score,
        session_weights=None,
        session_hints=None,
    )

    assert call_count == baseline_calls
