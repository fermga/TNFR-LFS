"""Command line entry point for TNFR-LFS."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Callable, Dict, List, Mapping, Sequence

from ..acquisition import (
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    OutGaugeUDPClient,
    OutSimClient,
    OutSimUDPClient,
    TelemetryFusion,
)
from ..core.epi import EPIExtractor, TelemetryRecord
from ..core.operators import orchestrate_delta_metrics
from ..core.segmentation import Microsector, segment_microsectors
from ..exporters import exporters_registry
from ..exporters.setup_plan import SetupChange, SetupPlan
from ..recommender import RecommendationEngine, SetupPlanner


Records = List[TelemetryRecord]
Bundles = Sequence[Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TNFR Load & Force Synthesis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # baseline
    # ------------------------------------------------------------------
    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Capture telemetry from UDP clients or simulation data and persist it.",
    )
    baseline_parser.add_argument("output", type=Path, help="Destination file for the baseline")
    baseline_parser.add_argument(
        "--format",
        choices=("jsonl", "parquet"),
        default="jsonl",
        help="Persistence format used for the baseline (default: jsonl).",
    )
    baseline_parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Capture duration in seconds when using live UDP acquisition.",
    )
    baseline_parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Maximum number of samples to collect from UDP clients.",
    )
    baseline_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host where the OutSim/OutGauge broadcasters are running.",
    )
    baseline_parser.add_argument(
        "--outsim-port",
        type=int,
        default=4123,
        help="Port used by the OutSim UDP stream.",
    )
    baseline_parser.add_argument(
        "--outgauge-port",
        type=int,
        default=3000,
        help="Port used by the OutGauge UDP stream.",
    )
    baseline_parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Polling timeout for the UDP clients.",
    )
    baseline_parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Number of retries performed by the UDP clients while polling.",
    )
    baseline_parser.add_argument(
        "--simulate",
        type=Path,
        help="Telemetry CSV file used in simulation mode (bypasses UDP capture).",
    )
    baseline_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of samples to persist when using simulation data.",
    )
    baseline_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    baseline_parser.set_defaults(handler=_handle_baseline)

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyse a telemetry baseline and export ΔNFR/Si insights."
    )
    analyze_parser.add_argument("telemetry", type=Path, help="Path to a baseline file or CSV.")
    analyze_parser.add_argument(
        "--export",
        choices=sorted(exporters_registry.keys()),
        default="json",
        help="Exporter used to render the analysis results.",
    )
    analyze_parser.add_argument(
        "--target-delta",
        type=float,
        default=0.0,
        help="Target ΔNFR objective used by the operators orchestration.",
    )
    analyze_parser.add_argument(
        "--target-si",
        type=float,
        default=0.75,
        help="Target sense index objective used by the operators orchestration.",
    )
    analyze_parser.add_argument(
        "--coherence-window",
        type=int,
        default=3,
        help="Window length used by the coherence operator when smoothing ΔNFR.",
    )
    analyze_parser.add_argument(
        "--recursion-decay",
        type=float,
        default=0.4,
        help="Decay factor for the recursive operator when computing hysteresis.",
    )
    analyze_parser.set_defaults(handler=_handle_analyze)

    # ------------------------------------------------------------------
    # suggest
    # ------------------------------------------------------------------
    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Generate recommendations for a telemetry baseline using the rule engine.",
    )
    suggest_parser.add_argument("telemetry", type=Path, help="Path to a baseline file or CSV.")
    suggest_parser.add_argument(
        "--export",
        choices=sorted(exporters_registry.keys()),
        default="json",
        help="Exporter used to render the recommendation payload.",
    )
    suggest_parser.add_argument(
        "--car-model",
        default="generic",
        help="Car model used to resolve the recommendation threshold profile.",
    )
    suggest_parser.add_argument(
        "--track",
        default="generic",
        help="Track identifier used to resolve the recommendation threshold profile.",
    )
    suggest_parser.set_defaults(handler=_handle_suggest)

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    report_parser = subparsers.add_parser(
        "report",
        help="Generate ΔNFR and sense index reports linked to the exporter registry.",
    )
    report_parser.add_argument("telemetry", type=Path, help="Path to a baseline file or CSV.")
    report_parser.add_argument(
        "--export",
        choices=sorted(exporters_registry.keys()),
        default="json",
        help="Exporter used to render the report payload.",
    )
    report_parser.add_argument(
        "--target-delta",
        type=float,
        default=0.0,
        help="Target ΔNFR objective used by the operators orchestration.",
    )
    report_parser.add_argument(
        "--target-si",
        type=float,
        default=0.75,
        help="Target sense index objective used by the operators orchestration.",
    )
    report_parser.add_argument(
        "--coherence-window",
        type=int,
        default=3,
        help="Window length used by the coherence operator when smoothing ΔNFR.",
    )
    report_parser.add_argument(
        "--recursion-decay",
        type=float,
        default=0.4,
        help="Decay factor for the recursive operator when computing hysteresis.",
    )
    report_parser.set_defaults(handler=_handle_report)

    # ------------------------------------------------------------------
    # write-set
    # ------------------------------------------------------------------
    write_set_parser = subparsers.add_parser(
        "write-set",
        help="Create a setup plan by combining optimisation with recommendations.",
    )
    write_set_parser.add_argument(
        "telemetry", type=Path, help="Path to a baseline file or CSV containing telemetry."
    )
    write_set_parser.add_argument(
        "--export",
        choices=sorted(exporters_registry.keys()),
        default="markdown",
        help="Exporter used to render the setup plan (default: markdown).",
    )
    write_set_parser.add_argument(
        "--car-model",
        default="generic_gt",
        help="Car model used to select the decision space for optimisation.",
    )
    write_set_parser.add_argument(
        "--session",
        default=None,
        help="Optional session label attached to the generated setup plan.",
    )
    write_set_parser.set_defaults(handler=_handle_write_set)

    return parser


def _handle_baseline(namespace: argparse.Namespace) -> str:
    if namespace.output.exists() and not namespace.force:
        raise FileExistsError(
            f"Baseline destination {namespace.output} already exists. Use --force to overwrite."
        )

    if namespace.simulate is not None:
        records = OutSimClient().ingest(namespace.simulate)
        if namespace.limit is not None:
            records = records[: namespace.limit]
    else:
        records = _capture_udp_samples(
            duration=namespace.duration,
            max_samples=namespace.max_samples,
            host=namespace.host,
            outsim_port=namespace.outsim_port,
            outgauge_port=namespace.outgauge_port,
            timeout=namespace.timeout,
            retries=namespace.retries,
        )

    if not records:
        message = "No telemetry samples captured."
        print(message)
        return message

    _persist_records(records, namespace.output, namespace.format)
    message = (
        f"Baseline saved {len(records)} samples to {namespace.output} "
        f"({namespace.format})."
    )
    print(message)
    return message


def _handle_analyze(namespace: argparse.Namespace) -> str:
    records = _load_records(namespace.telemetry)
    bundles, microsectors = _compute_insights(records)
    metrics = orchestrate_delta_metrics(
        [records],
        namespace.target_delta,
        namespace.target_si,
        coherence_window=namespace.coherence_window,
        recursion_decay=namespace.recursion_decay,
    )
    payload: Dict[str, Any] = {
        "series": bundles,
        "microsectors": microsectors,
        "telemetry_samples": len(records),
        "metrics": {key: value for key, value in metrics.items() if key != "bundles"},
        "smoothed_series": metrics.get("bundles", []),
    }
    return _render_payload(payload, namespace.export)


def _handle_suggest(namespace: argparse.Namespace) -> str:
    records = _load_records(namespace.telemetry)
    bundles, microsectors = _compute_insights(records)
    engine = RecommendationEngine()
    recommendations = engine.generate(
        bundles, microsectors, car_model=namespace.car_model, track_name=namespace.track
    )
    payload = {
        "series": bundles,
        "microsectors": microsectors,
        "recommendations": recommendations,
        "car_model": namespace.car_model,
        "track": namespace.track,
    }
    return _render_payload(payload, namespace.export)


def _handle_report(namespace: argparse.Namespace) -> str:
    records = _load_records(namespace.telemetry)
    metrics = orchestrate_delta_metrics(
        [records],
        namespace.target_delta,
        namespace.target_si,
        coherence_window=namespace.coherence_window,
        recursion_decay=namespace.recursion_decay,
    )
    payload: Dict[str, Any] = {
        "objectives": metrics.get("objectives", {}),
        "delta_nfr": metrics.get("delta_nfr", 0.0),
        "sense_index": metrics.get("sense_index", 0.0),
        "dissonance": metrics.get("dissonance", 0.0),
        "coupling": metrics.get("coupling", 0.0),
        "resonance": metrics.get("resonance", 0.0),
        "recursive_trace": metrics.get("recursive_trace", []),
        "series": metrics.get("bundles", []),
    }
    return _render_payload(payload, namespace.export)


def _handle_write_set(namespace: argparse.Namespace) -> str:
    records = _load_records(namespace.telemetry)
    bundles, microsectors = _compute_insights(records)
    planner = SetupPlanner()
    plan = planner.plan(bundles, microsectors, car_model=namespace.car_model)

    aggregated_rationales = [rec.rationale for rec in plan.recommendations if rec.rationale]
    aggregated_effects = [rec.message for rec in plan.recommendations if rec.message]
    if not aggregated_rationales:
        aggregated_rationales = ["Optimización de objetivo Si/ΔNFR"]
    if not aggregated_effects:
        aggregated_effects = ["Mejora equilibrada del coche"]

    changes = [
        SetupChange(
            parameter=name,
            delta=value,
            rationale="; ".join(aggregated_rationales),
            expected_effect="; ".join(aggregated_effects),
        )
        for name, value in sorted(plan.decision_vector.items())
    ]

    setup_plan = SetupPlan(
        car_model=namespace.car_model,
        session=namespace.session,
        changes=tuple(changes),
        rationales=tuple(aggregated_rationales),
        expected_effects=tuple(aggregated_effects),
    )

    payload = {
        "setup_plan": setup_plan,
        "objective_value": plan.objective_value,
        "recommendations": plan.recommendations,
        "series": plan.telemetry,
    }
    return _render_payload(payload, namespace.export)


def _render_payload(payload: Mapping[str, Any], exporter_name: str) -> str:
    exporter = exporters_registry[exporter_name]
    rendered = exporter(dict(payload))
    print(rendered)
    return rendered


def _compute_insights(records: Records) -> tuple[Bundles, Sequence[Microsector]]:
    if not records:
        return [], []
    extractor = EPIExtractor()
    bundles = extractor.extract(records)
    if not bundles:
        return bundles, []
    microsectors = segment_microsectors(records, bundles)
    return bundles, microsectors


def _capture_udp_samples(
    *,
    duration: float,
    max_samples: int,
    host: str,
    outsim_port: int,
    outgauge_port: int,
    timeout: float,
    retries: int,
) -> Records:
    fusion = TelemetryFusion()
    records: Records = []
    deadline = monotonic() + max(duration, 0.0)

    with OutSimUDPClient(
        host=host, port=outsim_port, timeout=timeout, retries=retries
    ) as outsim, OutGaugeUDPClient(
        host=host, port=outgauge_port, timeout=timeout, retries=retries
    ) as outgauge:
        while len(records) < max_samples and monotonic() < deadline:
            outsim_packet = outsim.recv()
            outgauge_packet = outgauge.recv()
            if outsim_packet is None or outgauge_packet is None:
                sleep(timeout)
                continue
            record = fusion.fuse(outsim_packet, outgauge_packet)
            records.append(record)

    return records


def _persist_records(records: Records, destination: Path, fmt: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with destination.open("w", encoding="utf8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), sort_keys=True))
                handle.write("\n")
        return

    if fmt == "parquet":
        serialised = [asdict(record) for record in records]
        try:
            import pandas as pd  # type: ignore

            frame = pd.DataFrame(serialised)
            frame.to_parquet(destination, index=False)
            return
        except ModuleNotFoundError:
            with destination.open("w", encoding="utf8") as handle:
                json.dump(serialised, handle, sort_keys=True)
            return

    raise ValueError(f"Unsupported format '{fmt}'.")


def _load_records(source: Path) -> Records:
    if not source.exists():
        raise FileNotFoundError(f"Telemetry source {source} does not exist")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return OutSimClient().ingest(source)
    if suffix == ".jsonl":
        records: Records = []
        with source.open("r", encoding="utf8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                records.append(TelemetryRecord(**payload))
        return records
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore

            frame = pd.read_parquet(source)
            data = frame.to_dict(orient="records")
        except ModuleNotFoundError:
            with source.open("r", encoding="utf8") as handle:
                data = json.load(handle)
        return [TelemetryRecord(**item) for item in data]
    if suffix == ".json":
        with source.open("r", encoding="utf8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [TelemetryRecord(**item) for item in data]
        raise ValueError(f"JSON telemetry source {source} must contain a list of samples")

    raise ValueError(f"Unsupported telemetry format: {source}")


def run_cli(args: Sequence[str] | None = None) -> str:
    parser = build_parser()
    namespace = parser.parse_args(args=args)
    handler: Callable[[argparse.Namespace], str] = getattr(namespace, "handler")
    return handler(namespace)


def main() -> None:  # pragma: no cover - thin wrapper
    run_cli()


if __name__ == "__main__":  # pragma: no cover - CLI invocation guard
    main()
