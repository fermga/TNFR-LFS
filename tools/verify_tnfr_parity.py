"""Verify parity between the local TNFR engine and the canonical extra.

The script loads telemetry samples, runs both engines, and emits a JSON report
summarising the absolute metrics together with their relative differences.  It
expects the optional ``tnfr`` extra to be installed so that the canonical engine
is available.

Usage examples::

    # Verify parity for the default synthetic stint and write the report to JSON
    poetry run python tools/verify_tnfr_parity.py tests/data/synthetic_stint.csv \
        --output parity_report.json

    # Override tolerances for a stricter parity check
    poetry run python tools/verify_tnfr_parity.py data/custom.csv \
        --epi-rel-tol 1e-6 --nu-f-rel-tol 1e-5
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path
try:  # pragma: no cover - python < 3.10 fallback
    from types import UnionType
except ImportError:  # pragma: no cover - fallback for older interpreters
    UnionType = None  # type: ignore[assignment]

from typing import Any, Iterable, Mapping, Union, get_args, get_origin

import numpy as np

_DEFAULT_REL_TOLERANCES: Mapping[str, float] = {
    "epi": 5e-5,
    "delta_nfr": 5e-5,
    "sense_index": 5e-5,
    "nu_f": 5e-4,
    "phase_alignment": 1e-3,
}

_NU_F_NODES: tuple[str, ...] = (
    "tyres",
    "suspension",
    "chassis",
    "brakes",
    "transmission",
    "track",
    "driver",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_path",
        type=Path,
        help="Ruta al fichero con las muestras de telemetría (CSV, JSON o JSONL).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Escribe el informe JSON en la ruta indicada (stdout por defecto).",
    )
    parser.add_argument(
        "--epi-rel-tol",
        type=float,
        default=_DEFAULT_REL_TOLERANCES["epi"],
        help="Tolerancia relativa para la métrica EPI (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--delta-nfr-rel-tol",
        type=float,
        default=_DEFAULT_REL_TOLERANCES["delta_nfr"],
        help="Tolerancia relativa para ΔNFR (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--sense-index-rel-tol",
        type=float,
        default=_DEFAULT_REL_TOLERANCES["sense_index"],
        help="Tolerancia relativa para Sense Index (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--nu-f-rel-tol",
        type=float,
        default=_DEFAULT_REL_TOLERANCES["nu_f"],
        help="Tolerancia relativa para ν_f (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--phase-alignment-rel-tol",
        type=float,
        default=_DEFAULT_REL_TOLERANCES["phase_alignment"],
        help="Tolerancia relativa para la alineación de fases (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Devuelve código de salida 1 si alguna métrica supera su tolerancia.",
    )
    return parser.parse_args()


def _clear_tnfr_core_modules() -> None:
    for name in list(sys.modules):
        if name == "tnfr_core" or name.startswith("tnfr_core."):
            sys.modules.pop(name, None)


def _load_tnfr_core(canonical_flag: str) -> Any:
    original = os.getenv("TNFR_CANONICAL")
    try:
        os.environ["TNFR_CANONICAL"] = canonical_flag
        _clear_tnfr_core_modules()
        return importlib.import_module("tnfr_core")
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "No se pudo importar tnfr_core; instale el extra canónico 'tnfr'."
        ) from exc
    finally:
        if original is None:
            os.environ.pop("TNFR_CANONICAL", None)
        else:
            os.environ["TNFR_CANONICAL"] = original


def _base_type(annotation: Any) -> Any:
    origin = get_origin(annotation)
    union_types = (Union,) + ((UnionType,) if UnionType is not None else ())
    if origin in union_types:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if args:
            return _base_type(args[0])
        return float
    if origin is None:
        return annotation
    return origin


def _convert_value(value: str, target_type: Any) -> Any:
    if value == "" or value is None:
        return None
    target = _base_type(target_type)
    if target in {float, int}:
        numeric = float(value)
        if target is int:
            return int(round(numeric))
        return float(numeric)
    return value


def _load_samples(
    data_path: Path, field_types: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not data_path.exists():  # pragma: no cover - CLI guard
        raise FileNotFoundError(f"No se encontró el fichero de datos: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        samples: list[dict[str, Any]] = []
        with data_path.open(encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                entry = {}
                for key in row:
                    if key not in field_types:
                        continue
                    value = _convert_value(row[key], field_types.get(key, float))
                    if value is None:
                        continue
                    entry[key] = value
                samples.append(entry)
        return samples

    if suffix == ".jsonl":
        samples = []
        with data_path.open(encoding="utf8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                entry = {}
                for key in record:
                    if key not in field_types:
                        continue
                    value = record.get(key)
                    if value is None:
                        continue
                    entry[key] = value
                samples.append(entry)
        return samples

    if suffix == ".json":
        payload = json.loads(data_path.read_text(encoding="utf8"))
        if not isinstance(payload, list):  # pragma: no cover - guard rail
            raise ValueError("El JSON debe contener una lista de muestras.")
        result: list[dict[str, Any]] = []
        for sample in payload:
            entry: dict[str, Any] = {}
            for key, value in sample.items():
                if key not in field_types or value is None:
                    continue
                entry[key] = value
            result.append(entry)
        return result

    raise ValueError(
        "Formato de datos no soportado. Use CSV, JSON o JSONL con cabeceras válidas."
    )


def _summarise_outputs(
    bundles: Iterable[Any], microsectors: Iterable[Any]
) -> dict[str, Any]:
    bundle_list = list(bundles)
    microsector_list = list(microsectors)
    epi = np.array([float(bundle.epi) for bundle in bundle_list], dtype=float)
    delta_nfr = np.array(
        [float(bundle.delta_nfr) for bundle in bundle_list], dtype=float
    )
    sense_index = np.array(
        [float(bundle.sense_index) for bundle in bundle_list], dtype=float
    )
    nu_f_matrix = np.array(
        [
            [float(getattr(bundle, node).nu_f) for node in _NU_F_NODES]
            for bundle in bundle_list
        ],
        dtype=float,
    )
    alignment_totals: dict[str, float] = {}
    alignment_counts: dict[str, int] = {}
    active_phases: list[str] = []
    for sector in microsector_list:
        active_phases.append(str(sector.active_phase))
        for phase, value in getattr(sector, "phase_alignment", {}).items():
            phase_key = str(phase)
            alignment_totals[phase_key] = alignment_totals.get(phase_key, 0.0) + float(
                value
            )
            alignment_counts[phase_key] = alignment_counts.get(phase_key, 0) + 1
    phase_alignment = {
        phase: alignment_totals[phase] / alignment_counts[phase]
        for phase in alignment_totals
        if alignment_counts[phase] > 0
    }
    return {
        "epi": epi,
        "delta_nfr": delta_nfr,
        "sense_index": sense_index,
        "nu_f": nu_f_matrix,
        "phase_alignment": phase_alignment,
        "active_phases": tuple(active_phases),
    }


def _relative_difference(local: np.ndarray, canonical: np.ndarray) -> np.ndarray:
    diff = np.abs(local - canonical)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(canonical) > 0.0, diff / np.abs(canonical), diff)
    return rel


def _summarise_array(
    metric: str,
    local: np.ndarray,
    canonical: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    rel = _relative_difference(local, canonical)
    abs_diff = np.abs(local - canonical)
    max_rel = float(rel.max()) if rel.size else 0.0
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    return {
        "metric": metric,
        "local": local.tolist(),
        "canonical": canonical.tolist(),
        "absolute_difference": abs_diff.tolist(),
        "relative_difference": rel.tolist(),
        "max_absolute_difference": max_abs,
        "max_relative_difference": max_rel,
        "within_tolerance": max_rel <= tolerance,
        "tolerance": tolerance,
    }


def _summarise_phase_alignment(
    local: Mapping[str, float],
    canonical: Mapping[str, float],
    tolerance: float,
) -> dict[str, Any]:
    all_phases = sorted({*local.keys(), *canonical.keys()})
    diff_report = {}
    max_rel = 0.0
    for phase in all_phases:
        lhs = float(local.get(phase, 0.0))
        rhs = float(canonical.get(phase, 0.0))
        abs_diff = abs(lhs - rhs)
        rel = abs_diff / abs(rhs) if rhs else abs_diff
        diff_report[phase] = {
            "local": lhs,
            "canonical": rhs,
            "absolute_difference": abs_diff,
            "relative_difference": rel,
        }
        max_rel = max(max_rel, rel)
    return {
        "metric": "phase_alignment",
        "values": diff_report,
        "max_relative_difference": max_rel,
        "within_tolerance": max_rel <= tolerance,
        "tolerance": tolerance,
    }


def main() -> int:
    args = _parse_args()

    fallback_core = _load_tnfr_core("0")
    field_types = getattr(fallback_core.TelemetryRecord, "__annotations__", {})
    samples = _load_samples(args.data_path, field_types)

    fallback_records = [fallback_core.TelemetryRecord(**sample) for sample in samples]
    fallback_bundles = fallback_core.EPIExtractor().extract(fallback_records)
    fallback_micro = fallback_core.segment_microsectors(
        fallback_records, fallback_bundles
    )
    fallback_summary = _summarise_outputs(fallback_bundles, fallback_micro)

    canonical_core = _load_tnfr_core("1")
    canonical_records = [
        canonical_core.TelemetryRecord(**sample) for sample in samples
    ]
    canonical_bundles = canonical_core.EPIExtractor().extract(canonical_records)
    canonical_micro = canonical_core.segment_microsectors(
        canonical_records, canonical_bundles
    )
    canonical_summary = _summarise_outputs(canonical_bundles, canonical_micro)

    epi_report = _summarise_array(
        "epi",
        fallback_summary["epi"],
        canonical_summary["epi"],
        args.epi_rel_tol,
    )
    delta_report = _summarise_array(
        "delta_nfr",
        fallback_summary["delta_nfr"],
        canonical_summary["delta_nfr"],
        args.delta_nfr_rel_tol,
    )
    sense_report = _summarise_array(
        "sense_index",
        fallback_summary["sense_index"],
        canonical_summary["sense_index"],
        args.sense_index_rel_tol,
    )
    nu_f_report = _summarise_array(
        "nu_f",
        fallback_summary["nu_f"],
        canonical_summary["nu_f"],
        args.nu_f_rel_tol,
    )
    phase_report = _summarise_phase_alignment(
        fallback_summary["phase_alignment"],
        canonical_summary["phase_alignment"],
        args.phase_alignment_rel_tol,
    )

    active_match = (
        fallback_summary["active_phases"] == canonical_summary["active_phases"]
    )

    report = {
        "data_path": str(args.data_path),
        "sample_count": len(samples),
        "metrics": {
            "epi": epi_report,
            "delta_nfr": delta_report,
            "sense_index": sense_report,
            "nu_f": nu_f_report,
            "phase_alignment": phase_report,
        },
        "active_phases": {
            "local": list(fallback_summary["active_phases"]),
            "canonical": list(canonical_summary["active_phases"]),
            "match": active_match,
        },
    }

    overall_pass = all(
        metric_report["within_tolerance"]
        for metric_report in report["metrics"].values()
    ) and active_match
    report["status"] = "pass" if overall_pass else "fail"

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf8")
    else:
        print(output)

    if args.strict and not overall_pass:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
