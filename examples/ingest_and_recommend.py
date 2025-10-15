"""Example showing how to ingest telemetry and generate recommendations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from tnfr_lfs.telemetry.live import OutSimClient
from tnfr_core import EPIExtractor, segment_microsectors
from tnfr_lfs.recommender import RecommendationEngine
from tnfr_lfs.exporters import build_operator_trajectories_payload


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "telemetry_path",
        nargs="?",
        help="Path to the telemetry CSV file (defaults to bundled sample data).",
    )
    parser.add_argument(
        "--emit-ops-json",
        metavar="PATH",
        dest="emit_ops_json",
        help="Write operator events to PATH as a JSON file.",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = _build_parser().parse_args()

    telemetry_path = args.telemetry_path or load_sample_records()
    client = OutSimClient()
    records = client.ingest(telemetry_path)
    extractor = EPIExtractor()
    results = extractor.extract(records)
    microsectors = segment_microsectors(
        records,
        results,
        baseline=getattr(extractor, "baseline_record", None),
    )
    engine = RecommendationEngine()
    for recommendation in engine.generate(results, microsectors):
        print(f"[{recommendation.category}] {recommendation.message}\n  {recommendation.rationale}")

    if args.emit_ops_json:
        payload_context = {
            "series": results,
            "microsectors": microsectors,
        }
        payload = build_operator_trajectories_payload(payload_context)
        destination = Path(args.emit_ops_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf8",
        )
        print(f"Operator events written to {destination}")


if __name__ == "__main__":
    main()
