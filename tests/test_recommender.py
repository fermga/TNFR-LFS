from tnfr_lfs.core.epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    EPIBundle,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)
from tnfr_lfs.core.segmentation import Goal, Microsector
from tnfr_lfs.recommender.rules import RecommendationEngine


def test_recommendation_engine_detects_anomalies():
    def build_nodes(delta_nfr: float, sense_index: float):
        return dict(
            tyres=TyresNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            suspension=SuspensionNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            chassis=ChassisNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            brakes=BrakesNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            transmission=TransmissionNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            track=TrackNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            driver=DriverNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
        )

    results = [
        EPIBundle(timestamp=0.0, epi=0.4, delta_nfr=15.0, sense_index=0.55, **build_nodes(15.0, 0.55)),
        EPIBundle(timestamp=0.1, epi=0.5, delta_nfr=-12.0, sense_index=0.50, **build_nodes(-12.0, 0.50)),
        EPIBundle(timestamp=0.2, epi=0.6, delta_nfr=2.0, sense_index=0.90, **build_nodes(2.0, 0.90)),
    ]
    engine = RecommendationEngine()
    recommendations = engine.generate(results)
    categories = {recommendation.category for recommendation in recommendations}
    assert {"suspension", "aero", "driver"} <= categories
    messages = [recommendation.message for recommendation in recommendations]
    assert any("Ride" in message for message in messages)
    assert any("sense index" in message.lower() for message in messages)


def test_phase_specific_rules_triggered_with_microsectors():
    def build_bundle(timestamp: float, delta_nfr: float, sense_index: float, tyre_delta: float) -> EPIBundle:
        tyre_node = TyresNode(delta_nfr=tyre_delta, sense_index=sense_index)
        return EPIBundle(
            timestamp=timestamp,
            epi=0.5,
            delta_nfr=delta_nfr,
            sense_index=sense_index,
            tyres=tyre_node,
            suspension=SuspensionNode(delta_nfr=delta_nfr / 2, sense_index=sense_index),
            chassis=ChassisNode(delta_nfr=delta_nfr / 2, sense_index=sense_index),
            brakes=BrakesNode(delta_nfr=delta_nfr / 2, sense_index=sense_index),
            transmission=TransmissionNode(delta_nfr=delta_nfr / 2, sense_index=sense_index),
            track=TrackNode(delta_nfr=delta_nfr / 2, sense_index=sense_index),
            driver=DriverNode(delta_nfr=delta_nfr / 2, sense_index=sense_index),
        )

    entry_target = 1.0
    apex_target = 0.5
    exit_target = -0.2
    microsector = Microsector(
        index=3,
        start_time=0.0,
        end_time=0.6,
        curvature=1.8,
        brake_event=True,
        support_event=True,
        delta_nfr_signature=0.0,
        goals=(
            Goal(
                phase="entry",
                archetype="apoyo",
                description="",
                target_delta_nfr=entry_target,
                target_sense_index=0.9,
            ),
            Goal(
                phase="apex",
                archetype="apoyo",
                description="",
                target_delta_nfr=apex_target,
                target_sense_index=0.9,
            ),
            Goal(
                phase="exit",
                archetype="apoyo",
                description="",
                target_delta_nfr=exit_target,
                target_sense_index=0.9,
            ),
        ),
        phase_boundaries={
            "entry": (0, 2),
            "apex": (2, 4),
            "exit": (4, 6),
        },
    )

    results = [
        build_bundle(0.0, 4.0, 0.92, tyre_delta=6.0),
        build_bundle(0.1, 4.2, 0.91, tyre_delta=6.2),
        build_bundle(0.2, 2.8, 0.93, tyre_delta=6.6),
        build_bundle(0.3, 2.6, 0.94, tyre_delta=6.5),
        build_bundle(0.4, -3.0, 0.95, tyre_delta=6.0),
        build_bundle(0.5, -3.2, 0.95, tyre_delta=5.8),
    ]

    engine = RecommendationEngine()
    recommendations = engine.generate(results, [microsector])

    assert len(recommendations) >= 4
    categories = [recommendation.category for recommendation in recommendations[:4]]
    assert categories == ["entry", "apex", "pianos", "exit"]

    entry_rationale = next(
        recommendation.rationale for recommendation in recommendations if recommendation.category == "entry"
    )
    assert "ΔNFR" in entry_rationale
    assert f"{entry_target:.2f}" in entry_rationale

    piano_message = next(
        recommendation.message for recommendation in recommendations if recommendation.category == "pianos"
    )
    assert "Operador de pianos" in piano_message
