#!/usr/bin/env python3
"""Generate lightweight AutoAPI-style Markdown documentation.

This script walks the ``tnfr_lfs`` package under ``src`` and emits Markdown
pages that loosely mirror MkDocs AutoAPI output.  It is intentionally minimal –
we only need enough structure to unblock documentation builds in environments
where the real AutoAPI plugin is unavailable.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
import shutil
import textwrap
from typing import Iterable, List, Optional

PACKAGE_NAME = "tnfr_lfs"
SRC_ROOT = Path("src") / PACKAGE_NAME
DOCS_ROOT = Path("docs/reference/autoapi")


@dataclasses.dataclass
class FunctionDoc:
    name: str
    signature: str
    docstring: str


@dataclasses.dataclass
class ClassDoc:
    name: str
    bases: List[str]
    docstring: str
    methods: List[FunctionDoc]


@dataclasses.dataclass
class ModuleDoc:
    name: str
    path: Path
    docstring: str
    functions: List[FunctionDoc]
    classes: List[ClassDoc]
    attributes: List[str]
    is_package: bool


def _format_arguments(args: ast.arguments) -> str:
    parts: List[str] = []

    def _fmt_arg(arg: ast.arg, default: Optional[ast.AST], annotation: Optional[ast.AST]) -> str:
        text = arg.arg
        if annotation is not None:
            text += f": {ast.unparse(annotation)}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        return text

    pos_defaults = list(args.defaults)
    pos_args = args.posonlyargs + args.args
    default_start = len(pos_args) - len(pos_defaults)
    for index, arg in enumerate(pos_args):
        default = None
        if index >= default_start:
            default = pos_defaults[index - default_start]
        annotation = arg.annotation
        parts.append(_fmt_arg(arg, default, annotation))

    if args.vararg is not None:
        vararg = args.vararg.arg
        if args.vararg.annotation is not None:
            vararg += f": {ast.unparse(args.vararg.annotation)}"
        parts.append("*" + vararg)

    if args.kwonlyargs:
        if args.vararg is None:
            parts.append("*")
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            parts.append(_fmt_arg(arg, default, arg.annotation))

    if args.kwarg is not None:
        kwarg = args.kwarg.arg
        if args.kwarg.annotation is not None:
            kwarg += f": {ast.unparse(args.kwarg.annotation)}"
        parts.append("**" + kwarg)

    return ", ".join(parts)


def _format_signature(node: ast.FunctionDef) -> str:
    args = _format_arguments(node.args)
    signature = f"({args})"
    if node.returns is not None:
        signature += f" -> {ast.unparse(node.returns)}"
    return signature


def _extract_functions(nodes: Iterable[ast.AST]) -> List[FunctionDoc]:
    functions: List[FunctionDoc] = []
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            doc = ast.get_docstring(node) or ""
            functions.append(
                FunctionDoc(
                    name=node.name,
                    signature=_format_signature(node),
                    docstring=doc,
                )
            )
    return functions


def _extract_classes(nodes: Iterable[ast.AST]) -> List[ClassDoc]:
    classes: List[ClassDoc] = []
    for node in nodes:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            bases = [ast.unparse(base) for base in node.bases]
            doc = ast.get_docstring(node) or ""
            methods = _extract_functions(node.body)
            classes.append(ClassDoc(name=node.name, bases=bases, docstring=doc, methods=methods))
    return classes


def _extract_attributes(nodes: Iterable[ast.AST]) -> List[str]:
    attrs: List[str] = []
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets: List[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            else:
                targets = [node.target]
                value = node.value
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    try:
                        rendered = ast.unparse(value)
                    except Exception:
                        rendered = "..."
                    attrs.append(f"{target.id} = {rendered}")
    return attrs


def parse_module(path: Path) -> ModuleDoc:
    module_path = path.relative_to(SRC_ROOT)
    if path.name == "__init__.py":
        parts = module_path.parts[:-1]
    else:
        parts = module_path.parts
    module_name = ".".join((PACKAGE_NAME, *[part[:-3] if part.endswith(".py") else part for part in parts]))

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    functions = _extract_functions(tree.body)
    classes = _extract_classes(tree.body)
    attributes = _extract_attributes(tree.body)
    is_package = path.name == "__init__.py"
    return ModuleDoc(
        name=module_name,
        path=module_path,
        docstring=docstring,
        functions=functions,
        classes=classes,
        attributes=attributes,
        is_package=is_package,
    )


def discover_modules() -> List[ModuleDoc]:
    modules: List[ModuleDoc] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.name == "__main__.py":
            continue
        modules.append(parse_module(path))
    return modules


def ensure_output_dir() -> None:
    if DOCS_ROOT.exists():
        shutil.rmtree(DOCS_ROOT)
    DOCS_ROOT.mkdir(parents=True)


def _module_output_dir(module: ModuleDoc) -> Path:
    parts = module.name.split(".")
    return DOCS_ROOT.joinpath(*parts)


def _render_markdown(module: ModuleDoc, children: List[str]) -> str:
    lines: List[str] = []
    title = f"`{module.name}` package" if module.is_package else f"`{module.name}` module"
    lines.append(f"# {title}\n")
    if module.docstring:
        lines.append(textwrap.dedent(module.docstring).strip())
        lines.append("")

    if module.is_package and children:
        lines.append("## Submodules\n")
        for child in children:
            module_parts = module.name.split(".")
            child_parts = child.split(".")
            relative_parts = child_parts[len(module_parts):]
            rel_path = "/".join(relative_parts + ["index.md"]) if relative_parts else "index.md"
            lines.append(f"- [`{child}`]({rel_path})")
        lines.append("")

    if module.classes:
        lines.append("## Classes\n")
        for cls in module.classes:
            header = f"### `{cls.name}`"
            if cls.bases:
                header += " (" + ", ".join(cls.bases) + ")"
            lines.append(header)
            if cls.docstring:
                lines.append(textwrap.dedent(cls.docstring).strip())
            if cls.methods:
                lines.append("")
                lines.append("#### Methods")
                for method in cls.methods:
                    lines.append(f"- `{method.name}{method.signature}`")
                    if method.docstring:
                        lines.append(f"  - {textwrap.dedent(method.docstring).strip()}")
            lines.append("")
    if module.functions:
        lines.append("## Functions\n")
        for func in module.functions:
            lines.append(f"- `{func.name}{func.signature}`")
            if func.docstring:
                lines.append(f"  - {textwrap.dedent(func.docstring).strip()}")
        lines.append("")
    if module.attributes:
        lines.append("## Attributes\n")
        for attr in module.attributes:
            lines.append(f"- `{attr}`")
        lines.append("")

    return "\n".join(line.rstrip() for line in lines if line is not None)


def _collect_children(modules: List[ModuleDoc]) -> dict[str, List[str]]:
    children: dict[str, List[str]] = {module.name: [] for module in modules}
    for module in modules:
        if module.name == PACKAGE_NAME:
            continue
        parent_name = ".".join(module.name.split(".")[:-1])
        if parent_name in children:
            children[parent_name].append(module.name)
    for name, values in children.items():
        values.sort()
    return children


def write_module_pages(modules: List[ModuleDoc]) -> None:
    children = _collect_children(modules)
    for module in modules:
        output_dir = _module_output_dir(module)
        output_dir.mkdir(parents=True, exist_ok=True)
        content = _render_markdown(module, children.get(module.name, []))
        (output_dir / "index.md").write_text(content + "\n", encoding="utf-8")


def write_root_index(modules: List[ModuleDoc]) -> None:
    top_level = [module for module in modules if module.name.count(".") == 1]
    lines = ["# API index\n"]
    lines.append("This index was generated from the source tree and mirrors the"
                 " ``tnfr_lfs`` package layout.\n")
    lines.append("- [`tnfr_lfs`](tnfr_lfs/index.md)\n")
    lines.append("## Top-level modules\n")
    for module in sorted(top_level, key=lambda m: m.name):
        rel_path = "/".join(module.name.split(".")) + "/index.md"
        lines.append(f"- [`{module.name}`]({rel_path})")
    DOCS_ROOT.joinpath("index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not SRC_ROOT.exists():
        raise SystemExit(f"Could not find source package under {SRC_ROOT!r}")
    ensure_output_dir()
    modules = discover_modules()
    write_module_pages(modules)
    write_root_index(modules)
    print(f"Generated documentation for {len(modules)} modules under {DOCS_ROOT}")


if __name__ == "__main__":
    main()
