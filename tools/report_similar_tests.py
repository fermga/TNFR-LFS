"""Report structurally similar pytest tests.

This utility walks through ``tests/**/*.py``, extracts ``test_*`` functions,
normalises their abstract syntax tree (AST) and compares the resulting
structures.  It outputs a JSON report under ``tests/_report`` with the most
similar pairs.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Iterator, List, Sequence


@dataclass
class TestFunction:
    file_path: Path
    qualified_name: str
    normalized_dump: str
    tokens: tuple[str, ...]
    token_set: frozenset[str]
    structure_signature: str

    @property
    def identifier(self) -> str:
        return f"{self.file_path.as_posix()}::{self.qualified_name}"


class ASTNormalizer(ast.NodeTransformer):
    """Normalise AST nodes to reduce noise when comparing tests."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        new_node = ast.Name(id="VAR", ctx=type(node.ctx)())
        return ast.copy_location(new_node, node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        new_node = ast.Attribute(
            value=self.visit(node.value),
            attr="ATTR",
            ctx=type(node.ctx)(),
        )
        return ast.copy_location(new_node, node)

    def visit_arg(self, node: ast.arg) -> ast.AST:  # type: ignore[override]
        new_node = ast.arg(arg="ARG", annotation=None, type_comment=None)
        return ast.copy_location(new_node, node)

    def visit_keyword(self, node: ast.keyword) -> ast.AST:  # type: ignore[override]
        new_node = ast.keyword(arg="KWARG" if node.arg is not None else None, value=self.visit(node.value))
        return ast.copy_location(new_node, node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # type: ignore[override]
        value = self._normalise_constant(node.value)
        new_node = ast.Constant(value=value)
        return ast.copy_location(new_node, node)

    def visit_alias(self, node: ast.alias) -> ast.AST:  # type: ignore[override]
        new_node = ast.alias(name="ALIAS", asname="ALIAS" if node.asname else None)
        return ast.copy_location(new_node, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        node.name = "test_function"
        node.decorator_list = [self.visit(deco) for deco in node.decorator_list]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        node.name = "test_function"
        node.decorator_list = [self.visit(deco) for deco in node.decorator_list]
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        node.func = self.visit(node.func)
        return node

    def _normalise_constant(self, value: object) -> str:
        if value is None:
            return "CONST_NONE"
        if isinstance(value, bool):
            return "CONST_BOOL"
        if isinstance(value, (int, float, complex)):
            return "CONST_NUMBER"
        if isinstance(value, str):
            return "CONST_STRING"
        if isinstance(value, (bytes, bytearray)):
            return "CONST_BYTES"
        if isinstance(value, (list, tuple, set, frozenset)):
            return "CONST_SEQUENCE"
        if isinstance(value, dict):
            return "CONST_MAPPING"
        return "CONST_OTHER"


def iter_test_functions(node: ast.AST, parents: Sequence[str] | None = None) -> Iterator[tuple[str, ast.AST]]:
    parents = list(parents or [])
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
            yield ("::".join(parents + [child.name]), child)
        elif isinstance(child, ast.ClassDef):
            yield from iter_test_functions(child, parents + [child.name])
        else:
            yield from iter_test_functions(child, parents)


def remove_docstring(node: ast.AST) -> None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return
    if node.body and isinstance(node.body[0], ast.Expr):
        expr = node.body[0]
        if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
            node.body = node.body[1:]


def collect_tests(tests_root: Path) -> List[TestFunction]:
    normalizer = ASTNormalizer()
    collected: List[TestFunction] = []

    for path in sorted(tests_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            source = path.read_text(encoding="utf8")
            module = ast.parse(source, filename=str(path))
        except (UnicodeDecodeError, SyntaxError):
            continue

        for qualified_name, fn_node in iter_test_functions(module):
            fn_copy = copy.deepcopy(fn_node)
            remove_docstring(fn_copy)
            normalized_node = normalizer.visit(fn_copy)
            ast.fix_missing_locations(normalized_node)
            normalized_dump = ast.dump(normalized_node, annotate_fields=False, include_attributes=False)
            tokens = tuple(token for token in normalized_dump.replace("(", " ").replace(")", " ").replace(",", " ").split() if token)
            structure_signature = " ".join(type(node).__name__ for node in ast.walk(normalized_node))
            collected.append(
                TestFunction(
                    file_path=path.relative_to(tests_root.parent),
                    qualified_name=qualified_name,
                    normalized_dump=normalized_dump,
                    tokens=tokens,
                    token_set=frozenset(tokens),
                    structure_signature=structure_signature,
                )
            )
    unique: dict[str, TestFunction] = {}
    for test in collected:
        unique[test.identifier] = test
    return list(unique.values())


def compute_similarities(tests: Sequence[TestFunction], threshold: float) -> List[dict[str, object]]:
    grouped: dict[str, List[TestFunction]] = defaultdict(list)
    for test in tests:
        grouped[test.structure_signature].append(test)

    matches: List[dict[str, object]] = []
    for candidates in grouped.values():
        if len(candidates) < 2:
            continue
        for idx, test_a in enumerate(candidates):
            len_a = len(test_a.tokens)
            set_a = test_a.token_set
            for test_b in candidates[idx + 1 :]:
                len_b = len(test_b.tokens)
                if not len_a or not len_b:
                    continue
                min_len = min(len_a, len_b)
                max_len = max(len_a, len_b)
            if max_len == 0:
                continue
            length_gap = 1 - (min_len / max_len)
            if length_gap > 0.3:
                continue
            set_b = test_b.token_set
            if not set_a or not set_b:
                continue
            overlap = len(set_a & set_b)
            overlap_ratio = overlap / min(len(set_a), len(set_b))
            if overlap_ratio < 0.5:
                continue
            max_possible = (2 * min_len) / (len_a + len_b)
            if max_possible < threshold:
                continue
            ratio = SequenceMatcher(None, test_a.tokens, test_b.tokens, autojunk=False).ratio()
            if ratio >= threshold and test_a.identifier != test_b.identifier:
                matches.append(
                    {
                        "test_a": {
                            "file": test_a.file_path.as_posix(),
                            "name": test_a.qualified_name,
                        },
                        "test_b": {
                            "file": test_b.file_path.as_posix(),
                            "name": test_b.qualified_name,
                        },
                        "similarity": round(ratio, 4),
                    }
                )
    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return matches


def generate_report(threshold: float, output_path: Path) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    tests_root = repo_root / "tests"
    tests = collect_tests(tests_root)
    similarities = compute_similarities(tests, threshold)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "total_tests": len(tests),
        "similar_pairs": similarities,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Minimum similarity ratio (0-1) to report. Defaults to 0.9.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/_report/similar_tests.json"),
        help="Destination JSON file for the report.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate_report(args.threshold, args.output)
    print(f"Collected {report['total_tests']} tests.")
    print(f"Found {len(report['similar_pairs'])} similar pairs with threshold {args.threshold}.")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
