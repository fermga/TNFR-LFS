"""Audit TNFR theory operators and variable references across the repository.

Usage examples::

    # Use default core (src/tnfr_core + src/tnfr_lfs/core) and tests directories
    poetry run python tools/tnfr_theory_audit.py --core --tests

    # Extend the search to an additional directory
    poetry run python tools/tnfr_theory_audit.py --core examples --tests docs/tests
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_CORE_DIRECTORIES: tuple[Path, ...] = (
    Path("src/tnfr_core"),
    Path("src/tnfr_lfs/core"),
)

DEFAULT_TEST_DIRECTORIES: tuple[Path, ...] = (Path("tests"),)

OPERATOR_DETECTION_MODULE = Path("src/tnfr_core/operators/operator_detection.py")

THEORY_OPERATORS: Mapping[str, Mapping[str, object]] = {
    "emission_operator": {
        "description": "Normalise ΔNFR and sense index targets.",
        "tokens": ("emission_operator",),
    },
    "reception_operator": {
        "description": "Convert raw telemetry into EPI bundles.",
        "tokens": ("reception_operator",),
    },
    "coherence_operator": {
        "description": "Smooth numeric series while preserving averages.",
        "tokens": ("coherence_operator",),
    },
    "plugin_coherence_operator": {
        "description": "Push coherence outputs into a plugin instance.",
        "tokens": ("plugin_coherence_operator",),
    },
    "dissonance_operator": {
        "description": "Compute mean absolute deviation for ΔNFR series.",
        "tokens": ("dissonance_operator",),
    },
    "dissonance_breakdown_operator": {
        "description": "Classify useful versus parasitic dissonance events.",
        "tokens": ("dissonance_breakdown_operator",),
    },
    "coupling_operator": {
        "description": "Return normalised coupling between two series.",
        "tokens": ("coupling_operator",),
    },
    "acoplamiento_operator": {
        "description": "Compatibility alias for the coupling operator.",
        "tokens": ("acoplamiento_operator",),
    },
    "pairwise_coupling_operator": {
        "description": "Compute coupling across node pairs.",
        "tokens": ("pairwise_coupling_operator",),
    },
    "resonance_operator": {
        "description": "Estimate RMS resonance for a series.",
        "tokens": ("resonance_operator",),
    },
    "recursivity_operator": {
        "description": "Maintain recursive thermal/style state per microsector.",
        "tokens": ("recursivity_operator",),
    },
    "recursive_filter_operator": {
        "description": "Apply a recursive filter capturing hysteresis.",
        "tokens": ("recursive_filter_operator",),
    },
    "recursividad_operator": {
        "description": "Compatibility alias for recursive filter operator.",
        "tokens": ("recursividad_operator",),
    },
    "mutation_operator": {
        "description": "Mutate archetypes when entropy or style drifts.",
        "tokens": ("mutation_operator",),
    },
    "tyre_balance_controller": {
        "description": "Compute ΔP and camber adjustments from tyre metrics.",
        "tokens": ("tyre_balance_controller",),
    },
    "orchestrate_delta_metrics": {
        "description": "Aggregate ΔNFR and Si metrics across segments.",
        "tokens": ("orchestrate_delta_metrics",),
    },
    "evolve_epi": {
        "description": "Integrate the Event Performance Index over time.",
        "tokens": ("evolve_epi",),
    },
}

THEORY_VARIABLES: Mapping[str, Mapping[str, object]] = {
    "delta_nfr": {
        "description": "ΔNFR gradient magnitude.",
        "tokens": ("delta_nfr",),
    },
    "delta_nfr_flat": {
        "description": "Reference ΔNFR used by recursivity filters.",
        "tokens": ("delta_nfr_flat",),
    },
    "sense_index": {
        "description": "Entropy-penalised Sense Index.",
        "tokens": ("sense_index",),
    },
    "nu_f": {
        "description": "Natural frequency classification.",
        "tokens": ("nu_f", "nu_f_"),
    },
    "coherence_index": {
        "description": "Structural coherence metric C(t).",
        "tokens": ("coherence_index",),
    },
    "support_effective": {
        "description": "Structurally weighted support metric.",
        "tokens": ("support_effective",),
    },
    "load_support_ratio": {
        "description": "Normalised load support ratio.",
        "tokens": ("load_support_ratio",),
    },
    "structural_expansion_longitudinal": {
        "description": "Longitudinal structural expansion indicator.",
        "tokens": ("structural_expansion_longitudinal",),
    },
    "structural_contraction_longitudinal": {
        "description": "Longitudinal structural contraction indicator.",
        "tokens": ("structural_contraction_longitudinal",),
    },
    "structural_expansion_lateral": {
        "description": "Lateral structural expansion indicator.",
        "tokens": ("structural_expansion_lateral",),
    },
    "structural_contraction_lateral": {
        "description": "Lateral structural contraction indicator.",
        "tokens": ("structural_contraction_lateral",),
    },
    "ackermann_parallel_index": {
        "description": "Ackermann alignment index.",
        "tokens": ("ackermann_parallel_index",),
    },
    "slide_catch_budget": {
        "description": "Slide recovery budget metric.",
        "tokens": ("slide_catch_budget",),
    },
    "aero_balance_drift": {
        "description": "Average aerodynamic balance drift.",
        "tokens": ("aero_balance_drift",),
    },
    "delta_nfr_entropy": {
        "description": "Entropy of ΔNFR distribution across phases.",
        "tokens": ("delta_nfr_entropy",),
    },
    "node_entropy": {
        "description": "Node-level ΔNFR entropy.",
        "tokens": ("node_entropy",),
    },
    "phase_delta_nfr_entropy": {
        "description": "Phase-level ΔNFR entropy map.",
        "tokens": ("phase_delta_nfr_entropy",),
    },
    "phase_node_entropy": {
        "description": "Phase-level node entropy map.",
        "tokens": ("phase_node_entropy",),
    },
    "thermal_load": {
        "description": "Thermal load tracked by recursivity state.",
        "tokens": ("thermal_load",),
    },
    "style_index": {
        "description": "Driving style index tracked by recursivity state.",
        "tokens": ("style_index",),
    },
    "network_memory": {
        "description": "Recursivity network memory snapshot.",
        "tokens": ("network_memory",),
    },
}


@dataclass(frozen=True)
class TheoryEntry:
    category: str
    identifier: str
    description: str
    tokens: tuple[str, ...]
    status: str | None = None


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_python_files(base_paths: Sequence[Path]) -> Iterable[Path]:
    for base in base_paths:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.is_file():
                yield path


def _collect_matches(base_paths: Sequence[Path], tokens: Sequence[str]) -> list[str]:
    if not tokens:
        return []
    hits: set[str] = set()
    repo_root = _resolve_repo_root()
    for file_path in _iter_python_files(base_paths):
        try:
            contents = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for token in tokens:
            if token and token in contents:
                relative = file_path.relative_to(repo_root).as_posix()
                hits.add(relative)
                break
    return sorted(hits)


def _format_hits(matches: Sequence[str]) -> str:
    if not matches:
        return "✗"
    if len(matches) <= 3:
        return "<br />".join(matches)
    visible = "<br />".join(matches[:3])
    return f"{visible}<br />(+{len(matches) - 3} more)"


def _build_detector_entries(repo_root: Path) -> list[TheoryEntry]:
    module_path = repo_root / OPERATOR_DETECTION_MODULE
    if not module_path.exists():
        return []

    try:
        source = module_path.read_text(encoding="utf-8")
    except OSError:
        return []

    tree = ast.parse(source)

    labels: dict[str, str] = {}
    detector_functions: set[str] = set()

    def _constant_str(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    class _DetectorVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
            if node.name.startswith("detect_"):
                detector_functions.add(node.name)
            self.generic_visit(node)

        def _extract_mapping(self, value: ast.AST) -> None:
            if not isinstance(value, ast.Dict):
                return
            for key_node, value_node in zip(value.keys, value.values):
                key = _constant_str(key_node)
                label = _constant_str(value_node)
                if key is None:
                    continue
                labels[key] = label or ""

        def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "STRUCTURAL_OPERATOR_LABELS":
                    self._extract_mapping(node.value)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # type: ignore[override]
            target = node.target
            if isinstance(target, ast.Name) and target.id == "STRUCTURAL_OPERATOR_LABELS":
                if node.value is not None:
                    self._extract_mapping(node.value)
            self.generic_visit(node)

    _DetectorVisitor().visit(tree)

    entries: list[TheoryEntry] = []
    for identifier, description in sorted(labels.items()):
        function_name = f"detect_{identifier.lower()}"
        implemented = function_name in detector_functions
        tokens: tuple[str, ...] = (function_name,) if implemented else ()
        status = None if implemented else "Pending"
        entries.append(
            TheoryEntry(
                category="Detector",
                identifier=identifier,
                description=description or "",
                tokens=tokens,
                status=status,
            )
        )

    return entries


def _build_entries(repo_root: Path) -> list[TheoryEntry]:
    entries: list[TheoryEntry] = []
    for identifier, payload in THEORY_OPERATORS.items():
        entries.append(
            TheoryEntry(
                category="Operator",
                identifier=identifier,
                description=str(payload["description"]),
                tokens=tuple(str(token) for token in payload.get("tokens", ())),
            )
        )
    entries.extend(_build_detector_entries(repo_root))
    for identifier, payload in THEORY_VARIABLES.items():
        entries.append(
            TheoryEntry(
                category="Variable",
                identifier=identifier,
                description=str(payload["description"]),
                tokens=tuple(str(token) for token in payload.get("tokens", ())),
            )
        )
    entries.sort(key=lambda entry: (entry.category, entry.identifier))
    return entries


def generate_report(
    core_paths: Sequence[Path], tests_paths: Sequence[Path], output_path: Path
) -> None:
    repo_root = _resolve_repo_root()
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()

    entries = _build_entries(repo_root)
    lines = ["# TNFR theory implementation matrix", ""]
    lines.append("| Category | Identifier | Description | Core references | Tests references |")
    lines.append("| --- | --- | --- | --- | --- |")

    for entry in entries:
        core_hits = _collect_matches(core_paths, entry.tokens)
        tests_hits = _collect_matches(tests_paths, entry.tokens)
        if entry.status:
            core_cell = entry.status
            tests_cell = entry.status
        else:
            core_cell = _format_hits(core_hits)
            tests_cell = _format_hits(tests_hits)
        lines.append(
            "| {category} | `{identifier}` | {description} | {core_hits} | {tests_hits} |".format(
                category=entry.category,
                identifier=entry.identifier,
                description=entry.description,
                core_hits=core_cell,
                tests_hits=tests_cell,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    relative_output = output_path.relative_to(repo_root)
    print(f"TNFR theory implementation matrix written to {relative_output}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit TNFR theory coverage across the repository.")
    parser.add_argument(
        "--core",
        nargs="*",
        metavar="PATH",
        help=(
            "Scan one or more core directories for theory references (default: "
            "src/tnfr_core and src/tnfr_lfs/core)."
        ),
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        metavar="PATH",
        help="Scan one or more test directories for theory references (default: tests).",
    )
    parser.add_argument(
        "--output",
        default=str(Path("tests/_report/theory_impl_matrix.md")),
        help="Path to the Markdown report (default: tests/_report/theory_impl_matrix.md).",
    )
    return parser.parse_args(argv)


def _resolve_requested_paths(
    requested: Sequence[str] | None,
    default_directories: Sequence[Path],
    repo_root: Path,
) -> list[Path]:
    if requested is None:
        return []
    if not requested:
        candidates = list(default_directories)
    else:
        candidates = [Path(path) for path in requested]

    resolved: list[Path] = []
    for candidate in candidates:
        absolute = candidate if candidate.is_absolute() else repo_root / candidate
        if absolute not in resolved:
            resolved.append(absolute)
    return resolved


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = _resolve_repo_root()

    default_core_paths = [repo_root / path for path in DEFAULT_CORE_DIRECTORIES]
    default_test_paths = [repo_root / path for path in DEFAULT_TEST_DIRECTORIES]

    if args.core is None and args.tests is None:
        core_paths = default_core_paths
        test_paths = default_test_paths
    else:
        core_paths = _resolve_requested_paths(args.core, DEFAULT_CORE_DIRECTORIES, repo_root)
        test_paths = _resolve_requested_paths(args.tests, DEFAULT_TEST_DIRECTORIES, repo_root)

    output_path = Path(args.output)
    generate_report(core_paths, test_paths, output_path)


if __name__ == "__main__":
    main()
