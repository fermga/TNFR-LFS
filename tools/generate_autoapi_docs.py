#!/usr/bin/env python3
"""Generate lightweight AutoAPI-style Markdown documentation.

This script walks the configured packages under ``src`` and emits Markdown
pages that loosely mirror MkDocs AutoAPI output.  It is intentionally minimal –
we only need enough structure to unblock documentation builds in environments
where the real AutoAPI plugin is unavailable.
"""
from __future__ import annotations

import ast
import dataclasses
import fnmatch
from pathlib import Path
import shutil
import textwrap
from typing import Dict, Iterable, List, Optional

import yaml

MKDOCS_CONFIG = Path("mkdocs.yml")
SRC_BASE = Path("src")
DOCS_ROOT = Path("docs/reference/autoapi")


def load_autoapi_config() -> tuple[List[str], List[str]]:
    """Read the MkDocs configuration and extract AutoAPI settings."""

    default_packages = ["tnfr_lfs"]
    if not MKDOCS_CONFIG.exists():
        return default_packages, []

    data = yaml.safe_load(MKDOCS_CONFIG.read_text(encoding="utf-8")) or {}
    packages: List[str] = []
    ignores: List[str] = []
    seen_packages: set[str] = set()
    plugin_config: Optional[dict] = None
    for plugin in data.get("plugins", []):
        if not isinstance(plugin, dict):
            continue
        if "autoapi" in plugin:
            plugin_config = plugin["autoapi"] or {}
            break
        if "mkdocs-autoapi" in plugin:
            plugin_config = plugin["mkdocs-autoapi"] or {}
            break

    if plugin_config is None:
        return default_packages, []

    candidate = plugin_config.get("packages", [])
    if isinstance(candidate, list):
        for item in candidate:
            if isinstance(item, str):
                normalized = str(item)
                if normalized not in seen_packages:
                    seen_packages.add(normalized)
                    packages.append(normalized)

    ignore_candidate = plugin_config.get("autoapi_ignore", [])
    if isinstance(ignore_candidate, list):
        for item in ignore_candidate:
            if isinstance(item, str):
                ignores.append(str(item))

    return (packages or default_packages, ignores)


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
    package: str
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


def parse_module(package: str, src_root: Path, path: Path) -> ModuleDoc:
    module_path = path.relative_to(src_root)
    if path.name == "__init__.py":
        parts = module_path.parts[:-1]
    else:
        parts = module_path.parts
    module_name = ".".join((package, *[part[:-3] if part.endswith(".py") else part for part in parts]))

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    functions = _extract_functions(tree.body)
    classes = _extract_classes(tree.body)
    attributes = _extract_attributes(tree.body)
    is_package = path.name == "__init__.py"
    return ModuleDoc(
        package=package,
        name=module_name,
        path=module_path,
        docstring=docstring,
        functions=functions,
        classes=classes,
        attributes=attributes,
        is_package=is_package,
    )


def discover_modules(package: str) -> List[ModuleDoc]:
    modules: List[ModuleDoc] = []
    src_root = SRC_BASE / package
    if not src_root.exists():
        raise SystemExit(f"Could not find source package under {src_root!r}")
    for path in sorted(src_root.rglob("*.py")):
        if path.name == "__main__.py":
            continue
        modules.append(parse_module(package, src_root, path))
    return modules


def ensure_output_dir() -> None:
    if DOCS_ROOT.exists():
        shutil.rmtree(DOCS_ROOT)
    DOCS_ROOT.mkdir(parents=True)


def _module_output_dir(module: ModuleDoc) -> Path:
    parts = module.name.split(".")
    return DOCS_ROOT.joinpath(*parts)


def _should_ignore(module_name: str, patterns: List[str]) -> bool:
    dotted_name = module_name
    path_name = module_name.replace(".", "/")
    for pattern in patterns:
        if fnmatch.fnmatch(dotted_name, pattern) or fnmatch.fnmatch(path_name, pattern):
            return True
    return False


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


def write_root_index(package_modules: Dict[str, List[ModuleDoc]]) -> None:
    package_names = sorted(package_modules)
    lines: List[str] = ["# API index", ""]
    if package_names:
        package_list = ", ".join(f"``{name}``" for name in package_names)
        lines.append(
            "This index was generated from the source tree and mirrors the "
            f"{package_list} package layout."
        )
        lines.append("")
        for package in package_names:
            lines.append(f"- [`{package}`]({package}/index.md)")
        lines.append("")
    lines.append("## Top-level modules")
    lines.append("")
    for package in package_names:
        top_level = [module for module in package_modules[package] if module.name.count(".") == 1]
        if not top_level:
            continue
        lines.append(f"### `{package}`")
        lines.append("")
        for module in sorted(top_level, key=lambda m: m.name):
            rel_path = "/".join(module.name.split(".")) + "/index.md"
            lines.append(f"- [`{module.name}`]({rel_path})")
        lines.append("")
    DOCS_ROOT.joinpath("index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    packages, ignore_patterns = load_autoapi_config()
    ensure_output_dir()
    package_modules: Dict[str, List[ModuleDoc]] = {}
    total_modules = 0
    for package in packages:
        modules = [
            module
            for module in discover_modules(package)
            if not _should_ignore(module.name, ignore_patterns)
        ]
        package_modules[package] = modules
        write_module_pages(modules)
        total_modules += len(modules)
    write_root_index(package_modules)
    print(
        f"Generated documentation for {total_modules} modules across {len(package_modules)} packages under {DOCS_ROOT}"
    )


if __name__ == "__main__":
    main()
