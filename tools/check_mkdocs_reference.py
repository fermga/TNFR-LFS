#!/usr/bin/env python3
"""Validate MkDocs configuration against generated AutoAPI docs."""
from __future__ import annotations

import sys
from pathlib import Path

MKDOCS_FILE = Path("mkdocs.yml")
API_REFERENCE = Path("docs/api_reference.md")
AUTOAPI_INDEX = Path("docs/reference/autoapi/index.md")


def nav_contains_reference() -> bool:
    return "reference/autoapi/index.md" in MKDOCS_FILE.read_text(encoding="utf-8")


def extract_links_from_api_reference() -> list[str]:
    links: list[str] = []
    for line in API_REFERENCE.read_text(encoding="utf-8").splitlines():
        if "reference/autoapi/" in line:
            start = line.find("reference/autoapi/")
            end = line.find(")", start)
            if end != -1:
                links.append(line[start:end])
    return links


def main() -> int:
    if not AUTOAPI_INDEX.exists():
        print("AutoAPI index is missing")
        return 1
    if not nav_contains_reference():
        print("MkDocs navigation does not include reference/autoapi/index.md")
        return 1
    missing = [link for link in extract_links_from_api_reference() if not Path("docs", link).exists()]
    if missing:
        print("Missing referenced pages:")
        for link in missing:
            print(f"  - {link}")
        return 1
    print("MkDocs reference navigation validated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
