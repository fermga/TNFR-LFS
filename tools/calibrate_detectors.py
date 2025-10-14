"""Calibration workflow for structural operator detectors.

The command line interface exposed by this module orchestrates the end-to-end
calibration process used by TNFR × LFS deployments.  It consumes labelled RAF
captures, derives TNFR telemetry features, performs parameter sweeps for the
detectors under ``tnfr_core.operators.operator_detection`` and materialises the
resulting overrides together with human-readable reports.

The workflow intentionally relies on the existing telemetry helpers so it
remains aligned with the runtime pipeline: ``read_raf`` and
``raf_to_telemetry_records`` extract :class:`~tnfr_core.equations.epi.TelemetryRecord`
sequences, :class:`~tnfr_core.equations.epi.EPIExtractor` reconstructs ΔNFR
bundles and :func:`tnfr_core.metrics.segmentation.segment_microsectors` provides
microsector level aggregates that the detectors inspect.  Labels supplied by
the caller are aligned against the generated microsectors to evaluate the
detector outcomes and determine the best configuration per operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import argparse
import csv
import json
import logging
import math
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

import yaml

from tnfr_core.equations.epi import EPIExtractor, TelemetryRecord
from tnfr_core.metrics.segmentation import Microsector, segment_microsectors
from tnfr_core.operators import operator_detection
from tnfr_core.operators.operator_detection import (
    canonical_operator_label,
    normalize_structural_operator_identifier,
)

from tnfr_lfs.telemetry.offline.raf import raf_to_telemetry_records, read_raf


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelledMicrosector:
    """Label payload resolved from the ``--labels`` artefact."""

    capture_id: str
    raf_path: Path
    track: str | None
    car: str | None
    car_class: str | None
    compound: str | None
    microsector_index: int
    operators: Mapping[str, bool]


@dataclass(frozen=True)
class MicrosectorSample:
    """Telemetry slice paired with operator ground truth."""

    capture_id: str
    microsector_index: int
    track: str
    car: str
    car_class: str | None
    compound: str | None
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    records: Sequence[TelemetryRecord]
    labels: Mapping[str, bool]

    @property
    def duration_seconds(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))


@dataclass(frozen=True)
class ParameterSet:
    """Candidate parameter payload for a detector."""

    identifier: str
    parameters: Mapping[str, object]


@dataclass
class EvaluationResult:
    """Aggregated metrics for a detector/parameter set combination."""

    operator_id: str
    combination: tuple[str, str, str]
    parameter_set: ParameterSet
    precision: float
    recall: float
    f1: float
    fp_per_minute: float
    support: int
    tp: int
    fp: int
    tn: int
    fn: int
    duration_minutes: float


def _normalise_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalise_operator_labels(payload: Any) -> dict[str, bool]:
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        entries: dict[str, bool] = {}
        for key, value in payload.items():
            if key is None:
                continue
            identifier = normalize_structural_operator_identifier(str(key))
            if isinstance(value, str):
                token = value.strip().lower()
                entries[identifier] = token not in {"", "0", "false", "no"}
            else:
                entries[identifier] = bool(value)
        return entries
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        entries: dict[str, bool] = {}
        for item in payload:
            if item is None:
                continue
            identifier = normalize_structural_operator_identifier(str(item))
            entries[identifier] = True
        return entries
    if isinstance(payload, str):
        tokens = [token.strip() for token in payload.split(",")]
        return {
            normalize_structural_operator_identifier(token): True
            for token in tokens
            if token
        }
    return {}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_lines(path: Path) -> list[Any]:
    entries: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Invalid JSON entry on line {line_number} of '{path}': {exc.msg}"
                ) from exc
            if isinstance(payload, Mapping) and "captures" in payload:
                captures = payload.get("captures")
                if isinstance(captures, Sequence) and not isinstance(
                    captures, (str, bytes)
                ):
                    entries.extend(captures)
                else:
                    entries.append(payload)
            else:
                entries.append(payload)
    return entries


def _load_toml(path: Path) -> Any:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_labels(path: Path, *, raf_root: Path) -> list[LabelledMicrosector]:
    """Parse the ``--labels`` artefact into :class:`LabelledMicrosector` entries."""

    loader_map: dict[str, Callable[[Path], Any]] = {
        ".json": _load_json,
        ".jsonl": _load_json_lines,
        ".toml": _load_toml,
        ".tml": _load_toml,
        ".yaml": _load_yaml,
        ".yml": _load_yaml,
        ".csv": None,
    }
    suffix = path.suffix.lower()
    if suffix not in loader_map:
        raise ValueError(f"Unsupported label format: {path.suffix}")
    if suffix == ".csv":
        return list(_load_labels_csv(path, raf_root=raf_root))
    loader = loader_map[suffix]
    assert loader is not None
    data = loader(path)
    return list(_load_labels_mapping(data, raf_root=raf_root))


def _load_labels_csv(path: Path, *, raf_root: Path) -> Iterator[LabelledMicrosector]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                micro_index = int(str(row.get("microsector") or row.get("microsector_index")))
            except (TypeError, ValueError):
                LOGGER.debug("Skipping CSV row without a valid microsector index: %s", row)
                continue
            operators = _normalise_operator_labels(row.get("operators") or row.get("operator"))
            if not operators and row.get("label") is not None:
                operators = {
                    normalize_structural_operator_identifier(str(row.get("operator"))): bool(
                        str(row.get("label")).strip() not in {"", "0", "false", "False"}
                    )
                }
            raf_token = row.get("raf") or row.get("path") or row.get("capture")
            if not raf_token:
                LOGGER.debug("CSV row missing RAF reference: %s", row)
                continue
            raf_path = (raf_root / str(raf_token)).resolve()
            capture_id = str(row.get("capture_id") or row.get("capture") or raf_path.stem)
            yield LabelledMicrosector(
                capture_id=capture_id,
                raf_path=raf_path,
                track=_normalise_identifier(row.get("track") or row.get("track_name")),
                car=_normalise_identifier(row.get("car") or row.get("car_model")),
                car_class=_normalise_identifier(row.get("car_class") or row.get("class")),
                compound=_normalise_identifier(
                    row.get("compound") or row.get("tyre") or row.get("tyre_compound")
                ),
                microsector_index=micro_index,
                operators=operators,
            )


def _load_labels_mapping(payload: Any, *, raf_root: Path) -> Iterator[LabelledMicrosector]:
    if isinstance(payload, Mapping):
        if "captures" in payload and isinstance(payload["captures"], Sequence):
            captures = payload["captures"]
        else:
            captures = [payload]
    elif isinstance(payload, Sequence):
        captures = payload
    else:
        raise ValueError("Labels payload must be a mapping or sequence")

    for entry in captures:
        if not isinstance(entry, Mapping):
            continue
        raf_token = entry.get("raf") or entry.get("path") or entry.get("capture")
        if not raf_token:
            LOGGER.debug("Skipping capture without RAF reference: %s", entry)
            continue
        raf_path = (raf_root / str(raf_token)).resolve()
        capture_id = str(entry.get("id") or entry.get("capture_id") or raf_path.stem)
        track = _normalise_identifier(entry.get("track") or entry.get("track_name"))
        car = _normalise_identifier(entry.get("car") or entry.get("car_model"))
        car_class = _normalise_identifier(entry.get("car_class") or entry.get("class"))
        compound = _normalise_identifier(
            entry.get("compound") or entry.get("tyre") or entry.get("tyre_compound")
        )
        micro_payload = entry.get("microsectors") or entry.get("labels")
        if isinstance(micro_payload, Mapping):
            iterator = micro_payload.items()
        elif isinstance(micro_payload, Sequence):
            iterator = enumerate(micro_payload)
        else:
            LOGGER.debug("Capture missing microsector labels: %s", entry)
            continue
        for micro_index, micro_entry in iterator:
            index_value = micro_index
            operators = {}
            if isinstance(micro_entry, Mapping):
                index_value = micro_entry.get("index", micro_index)
                operators = _normalise_operator_labels(
                    micro_entry.get("operators")
                    or micro_entry.get("labels")
                    or micro_entry.get("events")
                )
            elif isinstance(micro_entry, Sequence) and not isinstance(micro_entry, (str, bytes)):
                operators = _normalise_operator_labels(micro_entry)
            else:
                operators = _normalise_operator_labels(micro_entry)
            try:
                microsector_index = int(index_value)
            except (TypeError, ValueError):
                LOGGER.debug("Skipping microsector with invalid index: %s", micro_entry)
                continue
            yield LabelledMicrosector(
                capture_id=capture_id,
                raf_path=raf_path,
                track=track,
                car=car,
                car_class=car_class,
                compound=compound,
                microsector_index=microsector_index,
                operators=operators,
            )


def _microsector_span(microsector: Microsector) -> tuple[int, int]:
    indices: list[int] = []
    for sequence in microsector.phase_samples.values():
        indices.extend(int(value) for value in sequence)
    if not indices:
        raise ValueError("Microsector does not expose phase samples")
    start = min(indices)
    end = max(indices)
    return start, end


def _build_microsector_dataset(
    labels: Sequence[LabelledMicrosector],
    *,
    raf_root: Path,
) -> list[MicrosectorSample]:
    grouped: MutableMapping[Path, list[LabelledMicrosector]] = {}
    for entry in labels:
        grouped.setdefault(entry.raf_path, []).append(entry)

    extractor = EPIExtractor()
    dataset: list[MicrosectorSample] = []

    for raf_path, micro_labels in grouped.items():
        try:
            raf_file = read_raf(raf_path)
        except OSError as exc:
            LOGGER.warning("Failed to read RAF '%s': %s", raf_path, exc)
            continue
        metadata: dict[str, object] = {}
        compound = next((label.compound for label in micro_labels if label.compound), None)
        if compound:
            metadata["tyre_compound"] = compound
        try:
            records = raf_to_telemetry_records(raf_file, metadata=metadata)
        except Exception as exc:  # pragma: no cover - defensive catch
            LOGGER.warning("Failed to convert RAF '%s' to telemetry records: %s", raf_path, exc)
            continue
        if not records:
            LOGGER.debug("RAF '%s' produced no telemetry records", raf_path)
            continue
        try:
            bundles = extractor.extract(list(records))
        except Exception as exc:  # pragma: no cover - defensive catch
            LOGGER.warning("Failed to extract EPI bundles for '%s': %s", raf_path, exc)
            continue
        try:
            microsectors = segment_microsectors(records, bundles)
        except Exception as exc:  # pragma: no cover - defensive catch
            LOGGER.warning("Microsector segmentation failed for '%s': %s", raf_path, exc)
            continue
        if not microsectors:
            LOGGER.debug("RAF '%s' yielded no microsectors", raf_path)
            continue
        micro_map: dict[int, Microsector] = {micro.index: micro for micro in microsectors}
        for label in micro_labels:
            micro = micro_map.get(label.microsector_index)
            if micro is None:
                LOGGER.warning(
                    "Capture '%s' missing microsector %s in RAF '%s'",
                    label.capture_id,
                    label.microsector_index,
                    raf_path,
                )
                continue
            try:
                start_index, end_index = _microsector_span(micro)
            except ValueError as exc:
                LOGGER.warning(
                    "Microsector %s in '%s' is missing sample indices: %s",
                    label.microsector_index,
                    raf_path,
                    exc,
                )
                continue
            track = label.track or raf_file.header.track_name or "__unknown__"
            car_model = label.car or raf_file.header.car_model or "__unknown__"
            car_class = label.car_class
            compound = label.compound
            dataset.append(
                MicrosectorSample(
                    capture_id=label.capture_id,
                    microsector_index=label.microsector_index,
                    track=str(track),
                    car=str(car_model),
                    car_class=str(car_class) if car_class else None,
                    compound=str(compound) if compound else None,
                    start_index=start_index,
                    end_index=end_index,
                    start_time=float(micro.start_time),
                    end_time=float(micro.end_time),
                    records=records,
                    labels=dict(label.operators),
                )
            )
    return dataset


def _filter_samples(
    samples: Sequence[MicrosectorSample],
    *,
    cars: set[str] | None,
    compounds: set[str] | None,
    tracks: set[str] | None,
) -> list[MicrosectorSample]:
    def include(sample: MicrosectorSample) -> bool:
        if cars and sample.car not in cars:
            return False
        if compounds and (sample.compound or "") not in compounds:
            return False
        if tracks and sample.track not in tracks:
            return False
        return True

    filtered = [sample for sample in samples if include(sample)]
    if not filtered:
        LOGGER.warning("Filters removed every microsector sample; returning the original set")
        return list(samples)
    return filtered


def _load_operator_grid(path: Path) -> dict[str, list[ParameterSet]]:
    loader_map: dict[str, Callable[[Path], Any]] = {
        ".json": _load_json,
        ".toml": _load_toml,
        ".tml": _load_toml,
        ".yaml": _load_yaml,
        ".yml": _load_yaml,
    }
    suffix = path.suffix.lower()
    if suffix not in loader_map:
        raise ValueError(f"Unsupported operator search space format: {path.suffix}")
    data = loader_map[suffix](path)
    if not isinstance(data, Mapping):
        raise ValueError("Operator search space must be a mapping")
    payload = data.get("operators") if "operators" in data else data

    operator_grid: dict[str, list[ParameterSet]] = {}
    for key, value in payload.items():
        identifier = normalize_structural_operator_identifier(str(key))
        param_sets: list[ParameterSet] = []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, entry in enumerate(value):
                if not isinstance(entry, Mapping):
                    continue
                param_sets.append(
                    ParameterSet(
                        identifier=f"{identifier}-{index}",
                        parameters=dict(entry),
                    )
                )
        elif isinstance(value, Mapping):
            keys = sorted(value.keys())
            grids: list[list[tuple[str, object]]] = []
            for param_name in keys:
                options = value[param_name]
                if isinstance(options, Mapping) and "values" in options:
                    options = options["values"]
                if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
                    grids.append([(param_name, option) for option in options])
                else:
                    grids.append([(param_name, options)])
            for index, combination in enumerate(product(*grids)):
                params = {name: entry for name, entry in combination}
                param_sets.append(
                    ParameterSet(
                        identifier=f"{identifier}-{index}",
                        parameters=params,
                    )
                )
        else:
            raise ValueError(
                f"Operator '{key}' search space must be a mapping or sequence"
            )
        if not param_sets:
            raise ValueError(f"Operator '{key}' has no parameter sets to evaluate")
        operator_grid[identifier] = param_sets
    return operator_grid


def _detector_callable(operator_id: str) -> Callable[..., Sequence[Mapping[str, object]]]:
    normalised = normalize_structural_operator_identifier(operator_id)
    func_name = f"detect_{normalised.lower()}"
    detector = getattr(operator_detection, func_name, None)
    if detector is None:
        raise ValueError(f"Unknown detector for operator '{operator_id}'")
    return detector


def _group_combinations(samples: Sequence[MicrosectorSample]) -> dict[tuple[str, str, str], list[MicrosectorSample]]:
    grouped: dict[tuple[str, str, str], list[MicrosectorSample]] = {}
    for sample in samples:
        car_class = sample.car_class or "__default__"
        car_model = sample.car or "__unknown__"
        compound = sample.compound or "__default__"
        grouped.setdefault((car_class, car_model, compound), []).append(sample)
    return grouped


def _build_folds(samples: Sequence[MicrosectorSample], kfold: int) -> dict[str, int]:
    tracks = sorted({sample.track for sample in samples})
    if not tracks:
        return {}
    folds = max(1, min(kfold, len(tracks)))
    assignments: dict[str, int] = {}
    for index, track in enumerate(tracks):
        assignments[track] = index % folds
    return assignments


def _evaluate_detector(
    operator_id: str,
    samples: Sequence[MicrosectorSample],
    parameter_set: ParameterSet,
    *,
    fold_assignments: Mapping[str, int],
) -> EvaluationResult | None:
    if not samples:
        return None
    detector = _detector_callable(operator_id)
    tp = fp = tn = fn = 0
    total_duration = 0.0
    support = 0

    if fold_assignments:
        fold_indices = sorted(set(fold_assignments.values()))
    else:
        fold_indices = [0]

    for fold_index in fold_indices:
        for sample in samples:
            sample_fold = fold_assignments.get(sample.track, 0)
            if sample_fold != fold_index:
                continue
            label = sample.labels.get(operator_id)
            if label is None:
                continue
            truth = bool(label)
            window = list(sample.records[sample.start_index : sample.end_index + 1])
            try:
                events = detector(window, **parameter_set.parameters)
            except Exception as exc:
                LOGGER.debug(
                    "Detector '%s' failed for sample %s/%s: %s",
                    operator_id,
                    sample.capture_id,
                    sample.microsector_index,
                    exc,
                )
                events = ()
            prediction = bool(events)
            support += 1
            if truth and prediction:
                tp += 1
            elif truth and not prediction:
                fn += 1
            elif not truth and prediction:
                fp += 1
            else:
                tn += 1
            total_duration += sample.duration_seconds

    if support == 0:
        return None

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    duration_minutes = total_duration / 60.0 if total_duration > 0 else 0.0
    if duration_minutes <= 0.0:
        fp_per_minute = math.inf if fp > 0 else 0.0
    else:
        fp_per_minute = fp / duration_minutes
    return EvaluationResult(
        operator_id=operator_id,
        combination=(samples[0].car_class or "__default__", samples[0].car, samples[0].compound or "__default__"),
        parameter_set=parameter_set,
        precision=precision,
        recall=recall,
        f1=f1,
        fp_per_minute=fp_per_minute,
        support=support,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        duration_minutes=duration_minutes,
    )


def _select_best(
    results: Sequence[EvaluationResult],
    *,
    fp_per_minute_max: float,
) -> EvaluationResult | None:
    if not results:
        return None
    eligible = [result for result in results if result.fp_per_minute <= fp_per_minute_max]
    candidates = eligible or list(results)
    return max(
        candidates,
        key=lambda entry: (
            round(entry.f1, 6),
            round(entry.recall, 6),
            round(entry.precision, 6),
            -round(entry.fp_per_minute, 6),
        ),
    )


def _update_nested_mapping(
    root: MutableMapping[str, Any],
    path: Sequence[str],
    operator_id: str,
    params: Mapping[str, object],
) -> None:
    cursor: MutableMapping[str, Any] = root
    for key in path:
        cursor = cursor.setdefault(key, {})  # type: ignore[assignment]
    operators = cursor.setdefault("operators", {})
    if isinstance(operators, MutableMapping):
        operators[operator_id] = dict(params)
    else:  # pragma: no cover - defensive
        cursor["operators"] = {operator_id: dict(params)}


def _materialise_best_params(
    selections: Sequence[EvaluationResult],
    *,
    output_path: Path,
) -> None:
    payload: dict[str, Any] = {}
    for selection in selections:
        operator_id = selection.operator_id
        car_class, car_model, compound = selection.combination
        if car_class and car_class != "__default__":
            _update_nested_mapping(
                payload,
                ("classes", car_class, "compounds", compound),
                operator_id,
                selection.parameter_set.parameters,
            )
        if car_model and car_model != "__unknown__":
            _update_nested_mapping(
                payload,
                ("cars", car_model, "compounds", compound),
                operator_id,
                selection.parameter_set.parameters,
            )
        if not car_class or car_class == "__default__":
            if not car_model or car_model == "__unknown__":
                _update_nested_mapping(
                    payload,
                    ("defaults",),
                    operator_id,
                    selection.parameter_set.parameters,
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


def _write_curves_csv(
    results: Sequence[EvaluationResult],
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_operator: dict[str, list[EvaluationResult]] = {}
    for result in results:
        by_operator.setdefault(result.operator_id, []).append(result)
    for operator_id, entries in by_operator.items():
        path = output_dir / f"{operator_id.lower()}_curves.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "car_class",
                    "car",
                    "compound",
                    "parameter_set",
                    "precision",
                    "recall",
                    "f1",
                    "fp_per_minute",
                    "support",
                ]
            )
            for entry in sorted(
                entries,
                key=lambda item: (
                    item.combination,
                    item.parameter_set.identifier,
                ),
            ):
                writer.writerow(
                    [
                        entry.combination[0],
                        entry.combination[1],
                        entry.combination[2],
                        entry.parameter_set.identifier,
                        f"{entry.precision:.6f}",
                        f"{entry.recall:.6f}",
                        f"{entry.f1:.6f}",
                        f"{entry.fp_per_minute:.6f}",
                        entry.support,
                    ]
                )


def _write_confusion_csv(
    selections: Sequence[EvaluationResult],
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_operator: dict[str, list[EvaluationResult]] = {}
    for result in selections:
        by_operator.setdefault(result.operator_id, []).append(result)
    for operator_id, entries in by_operator.items():
        path = output_dir / f"{operator_id.lower()}_confusion.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "car_class",
                    "car",
                    "compound",
                    "parameter_set",
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                    "duration_minutes",
                ]
            )
            for entry in sorted(entries, key=lambda item: item.combination):
                writer.writerow(
                    [
                        entry.combination[0],
                        entry.combination[1],
                        entry.combination[2],
                        entry.parameter_set.identifier,
                        entry.tp,
                        entry.fp,
                        entry.tn,
                        entry.fn,
                        f"{entry.duration_minutes:.6f}",
                    ]
                )


def _write_report(
    selections: Sequence[EvaluationResult],
    *,
    output_path: Path,
) -> None:
    lines: list[str] = ["# Detector calibration report", ""]
    by_operator: dict[str, list[EvaluationResult]] = {}
    for result in selections:
        by_operator.setdefault(result.operator_id, []).append(result)
    for operator_id, entries in sorted(by_operator.items()):
        label = canonical_operator_label(operator_id)
        lines.append(f"## {operator_id} · {label}")
        lines.append("")
        lines.append(
            "| Car class | Car | Compound | Precision | Recall | F1 | FP/min | Params | Support |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for entry in sorted(entries, key=lambda item: item.combination):
            params = ", ".join(
                f"{key}={value}" for key, value in sorted(entry.parameter_set.parameters.items())
            )
            lines.append(
                "| {car_class} | {car} | {compound} | {precision:.3f} | {recall:.3f} | {f1:.3f} | "
                "{fp_per_minute:.3f} | {params} | {support} |".format(
                    car_class=entry.combination[0],
                    car=entry.combination[1],
                    compound=entry.combination[2],
                    precision=entry.precision,
                    recall=entry.recall,
                    f1=entry.f1,
                    fp_per_minute=entry.fp_per_minute,
                    params=params or "—",
                    support=entry.support,
                )
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def calibrate_detectors(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raf_root = Path(args.raf_root).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()

    LOGGER.info("Loading labelled microsectors from %s", labels_path)
    label_entries = _load_labels(labels_path, raf_root=raf_root)
    if not label_entries:
        raise SystemExit("No labelled microsectors were found in the provided artefact")

    LOGGER.info("Building telemetry dataset from %d labelled microsectors", len(label_entries))
    samples = _build_microsector_dataset(label_entries, raf_root=raf_root)
    if not samples:
        raise SystemExit("No microsector samples were produced from the supplied RAF captures")

    cars = {value.strip() for value in (args.cars or []) if value.strip()}
    compounds = {value.strip() for value in (args.compounds or []) if value.strip()}
    tracks = {value.strip() for value in (args.tracks or []) if value.strip()}
    if cars or compounds or tracks:
        samples = _filter_samples(samples, cars=cars or None, compounds=compounds or None, tracks=tracks or None)

    LOGGER.info("Loading operator search space from %s", args.operators)
    operator_grid = _load_operator_grid(Path(args.operators))

    LOGGER.info("Evaluating detectors across %d microsectors", len(samples))
    grouped = _group_combinations(samples)
    fold_assignments = _build_folds(samples, args.kfold)

    all_results: list[EvaluationResult] = []
    best_selections: list[EvaluationResult] = []

    for combination, combo_samples in sorted(grouped.items()):
        LOGGER.info(
            "Processing combination class=%s car=%s compound=%s (%d samples)",
            combination[0],
            combination[1],
            combination[2],
            len(combo_samples),
        )
        operators_in_combo = sorted({normalize_structural_operator_identifier(key) for sample in combo_samples for key in sample.labels.keys()})
        for operator_id in operators_in_combo:
            if operator_id not in operator_grid:
                LOGGER.warning("No parameter grid defined for operator '%s'; skipping", operator_id)
                continue
            candidates: list[EvaluationResult] = []
            for parameter_set in operator_grid[operator_id]:
                result = _evaluate_detector(
                    operator_id,
                    combo_samples,
                    parameter_set,
                    fold_assignments=fold_assignments,
                )
                if result is None:
                    continue
                candidates.append(result)
                all_results.append(result)
            if not candidates:
                LOGGER.warning(
                    "No successful evaluations for operator '%s' under combination %s",
                    operator_id,
                    combination,
                )
                continue
            selection = _select_best(candidates, fp_per_minute_max=args.fp_per_min_max)
            if selection:
                best_selections.append(selection)

    if not best_selections:
        raise SystemExit("No detector selections satisfied the evaluation constraints")

    best_params_path = output_dir / "best_params.yaml"
    LOGGER.info("Writing best parameter overrides to %s", best_params_path)
    _materialise_best_params(best_selections, output_path=best_params_path)

    curves_dir = output_dir / "curves"
    LOGGER.info("Writing calibration curves to %s", curves_dir)
    _write_curves_csv(all_results, output_dir=curves_dir)

    confusion_dir = output_dir / "confusion"
    LOGGER.info("Writing confusion matrices to %s", confusion_dir)
    _write_confusion_csv(best_selections, output_dir=confusion_dir)

    report_path = output_dir / "report.md"
    LOGGER.info("Writing summary report to %s", report_path)
    _write_report(best_selections, output_path=report_path)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate TNFR operator detectors")
    parser.add_argument("--raf-root", required=True, help="Directory containing RAF telemetry captures")
    parser.add_argument("--labels", required=True, help="Path to the labelled microsector artefact")
    parser.add_argument("--operators", required=True, help="Parameter grid describing detector search spaces")
    parser.add_argument("--out", required=True, help="Output directory for calibration artefacts")
    parser.add_argument("--cars", nargs="*", help="Optional car filter (one or more identifiers)")
    parser.add_argument("--compounds", nargs="*", help="Optional compound filter (one or more identifiers)")
    parser.add_argument("--tracks", nargs="*", help="Optional track filter (one or more identifiers)")
    parser.add_argument("--kfold", type=int, default=5, help="Number of cross-validation folds (grouped by track)")
    parser.add_argument(
        "--fp_per_min_max",
        type=float,
        default=0.5,
        help="Maximum allowed false positives per minute when ranking configurations",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    calibrate_detectors(args)


if __name__ == "__main__":
    main()

