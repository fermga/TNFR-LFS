"""Scan documentation Markdown files for non-English (Spanish) markers."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

ACCENTED_CHARACTERS = set("áéíóúÁÉÍÓÚñÑüÜ¡¿")
SPANISH_STOP_WORDS: Sequence[str] = (
    "aunque",
    "como",
    "donde",
    "entonces",
    "entre",
    "esta",
    "este",
    "esto",
    "hasta",
    "mientras",
    "para",
    "porque",
    "pero",
    "segun",
    "según",
    "siempre",
    "sin",
    "sobre",
    "tambien",
    "también",
)

WORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in SPANISH_STOP_WORDS) + r")\b",
    re.IGNORECASE,
)
def strip_inline_code(line: str, in_span: bool) -> tuple[str, bool]:
    """Remove inline code spans while tracking multi-line fences."""

    output: list[str] = []
    index = 0
    while index < len(line):
        character = line[index]
        if character == "`":
            in_span = not in_span
            index += 1
            continue
        if not in_span:
            output.append(character)
        index += 1
    return "".join(output), in_span


@dataclass
class Finding:
    file: Path
    line_number: int
    line: str
    matches: tuple[str, ...]
    accents: tuple[str, ...]

    def format(self) -> str:
        components: list[str] = []
        if self.matches:
            unique_words = ", ".join(sorted(set(self.matches)))
            components.append(f"stop words: {unique_words}")
        if self.accents:
            unique_accents = " ".join(sorted(set(self.accents)))
            components.append(f"accented characters: {unique_accents}")
        details = "; ".join(components)
        snippet = self.line.strip()
        return f"{self.file}:{self.line_number}: {details}\n    {snippet}"


def iter_markdown_files() -> Iterable[Path]:
    yield REPO_ROOT / "README.md"
    yield from (REPO_ROOT / "docs").rglob("*.md")
    yield REPO_ROOT / "tests" / "README.md"


def sanitise_line(line: str, in_inline_span: bool) -> tuple[str, bool]:
    """Remove inline code spans so examples do not trigger the detector."""

    return strip_inline_code(line, in_inline_span)


def scan_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    findings: list[Finding] = []
    in_code_block = False
    in_inline_span = False
    for index, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            in_inline_span = False
            continue
        if in_code_block:
            continue

        analysed_line, in_inline_span = sanitise_line(raw_line, in_inline_span)
        accents = tuple(sorted({ch for ch in analysed_line if ch in ACCENTED_CHARACTERS}))
        words = tuple(sorted({match.group(0).lower() for match in WORD_PATTERN.finditer(analysed_line)}))
        if accents or words:
            findings.append(
                Finding(
                    file=path.relative_to(REPO_ROOT),
                    line_number=index,
                    line=raw_line,
                    matches=words,
                    accents=accents,
                )
            )
    return findings


def main() -> int:
    all_findings: list[Finding] = []
    for markdown_file in iter_markdown_files():
        all_findings.extend(scan_file(markdown_file))

    if all_findings:
        print("Detected potential non-English documentation content:", file=sys.stderr)
        for finding in all_findings:
            print(finding.format(), file=sys.stderr)
        return 1

    print("Documentation language check passed (English-only content detected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
