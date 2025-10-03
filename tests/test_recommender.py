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
