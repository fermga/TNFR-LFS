from tnfr_lfs.core.epi import EPIResult
from tnfr_lfs.recommender.rules import RecommendationEngine


def test_recommendation_engine_detects_anomalies():
    results = [
        EPIResult(timestamp=0.0, epi=0.4, delta_nfr=15.0, delta_si=0.02),
        EPIResult(timestamp=0.1, epi=0.5, delta_nfr=-12.0, delta_si=0.1),
        EPIResult(timestamp=0.2, epi=0.6, delta_nfr=2.0, delta_si=-0.12),
    ]
    engine = RecommendationEngine()
    recommendations = engine.generate(results)
    categories = {recommendation.category for recommendation in recommendations}
    assert {"suspension", "aero", "driver"} <= categories
    messages = [recommendation.message for recommendation in recommendations]
    assert any("Ride" in message for message in messages)
    assert any("Wing" in message for message in messages)
