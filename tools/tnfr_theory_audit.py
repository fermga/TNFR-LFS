"""Audit TNFR theory operators and variable references across the repository.

Usage example::

    poetry run python tools/tnfr_theory_audit.py --core --tests --output tests/_report/theory_impl_matrix.md
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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


def _build_entries() -> list[TheoryEntry]:
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


def generate_report(include_core: bool, include_tests: bool, output_path: Path) -> None:
    repo_root = _resolve_repo_root()
    core_paths = [repo_root / "src" / "tnfr_lfs" / "core"] if include_core else []
    tests_paths = [repo_root / "tests"] if include_tests else []
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()

    entries = _build_entries()
    lines = ["# TNFR theory implementation matrix", ""]
    lines.append("| Category | Identifier | Description | Core references | Tests references |")
    lines.append("| --- | --- | --- | --- | --- |")

    for entry in entries:
        core_hits = _collect_matches(core_paths, entry.tokens)
        tests_hits = _collect_matches(tests_paths, entry.tokens)
        lines.append(
            "| {category} | `{identifier}` | {description} | {core_hits} | {tests_hits} |".format(
                category=entry.category,
                identifier=entry.identifier,
                description=entry.description,
                core_hits=_format_hits(core_hits),
                tests_hits=_format_hits(tests_hits),
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
        action="store_true",
        help="Scan src/tnfr_core for theory references.",
    )
    parser.add_argument(
        "--tests",
        action="store_true",
        help="Scan tests/ for theory references.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("tests/_report/theory_impl_matrix.md")),
        help="Path to the Markdown report (default: tests/_report/theory_impl_matrix.md).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    include_core = args.core
    include_tests = args.tests
    if not include_core and not include_tests:
        include_core = include_tests = True
    output_path = Path(args.output)
    generate_report(include_core, include_tests, output_path)


if __name__ == "__main__":
    main()
