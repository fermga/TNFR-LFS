"""Example showing how to ingest telemetry and generate recommendations."""

from __future__ import annotations

from tempfile import NamedTemporaryFile

from tnfr_lfs.ingestion.live import OutSimClient
from tnfr_lfs.core import EPIExtractor, segment_microsectors
from tnfr_lfs.recommender import RecommendationEngine


def load_sample_records() -> str:
    """Return a path to a generated telemetry CSV file."""

    data = """timestamp,vertical_load,slip_ratio,lateral_accel,longitudinal_accel,yaw,pitch,roll,brake_pressure,locking,nfr,si\n"""
    data += "\n".join(
        [
            "0.00,6500,0.05,1.2,0.4,0.01,0.0,0.0,0.20,0.0,520,0.82",
            "0.10,6900,0.02,1.0,0.5,0.02,0.0,0.0,0.18,0.0,540,0.81",
            "0.20,7200,0.08,1.1,0.6,0.04,0.0,0.0,0.22,0.0,565,0.75",
        ]
    )
    with NamedTemporaryFile("w", delete=False, suffix=".csv") as handle:
        handle.write(data)
        return handle.name


def main() -> None:
    telemetry_path = load_sample_records()
    client = OutSimClient()
    records = client.ingest(telemetry_path)
    extractor = EPIExtractor()
    results = extractor.extract(records)
    microsectors = segment_microsectors(records, results)
    engine = RecommendationEngine()
    for recommendation in engine.generate(results, microsectors):
        print(f"[{recommendation.category}] {recommendation.message}\n  {recommendation.rationale}")


if __name__ == "__main__":
    main()
