"""Command line entry point for TNFR-LFS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ..acquisition import OutSimClient
from ..core.epi import EPIExtractor
from ..exporters import exporters_registry
from ..recommender import RecommendationEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TNFR Load & Force Synthesis")
    parser.add_argument("telemetry", type=Path, help="Path to the telemetry CSV file")
    parser.add_argument(
        "--export",
        choices=sorted(exporters_registry.keys()),
        default="json",
        help="Exporter used to render the output",
    )
    return parser


def run_cli(args: argparse.Namespace | None = None) -> str:
    parser = build_parser()
    namespace = parser.parse_args(args=args) if args is not None else parser.parse_args()

    client = OutSimClient()
    records = client.ingest(namespace.telemetry)
    extractor = EPIExtractor()
    results = extractor.extract(records)
    engine = RecommendationEngine()
    recommendations = engine.generate(results)

    payload: Dict[str, Any] = {
        "series": results,
        "recommendations": [recommendation.__dict__ for recommendation in recommendations],
    }
    exporter = exporters_registry[namespace.export]
    output = exporter(payload)
    print(output)
    return output


def main() -> None:  # pragma: no cover - thin wrapper
    run_cli()


if __name__ == "__main__":  # pragma: no cover - CLI invocation guard
    main()
