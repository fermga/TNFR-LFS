from __future__ import annotations

import argparse

import pytest

pytest.importorskip("numpy")

from tnfr_lfs.cli.parser import build_parser
from tnfr_lfs.cli import workflows


@pytest.fixture()
def parser() -> argparse.ArgumentParser:
    return build_parser({})


def _subparser_choices(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):  # pragma: no branch
            return action.choices
    raise AssertionError("Subparser action not found")


def test_build_parser_registers_core_commands(parser: argparse.ArgumentParser) -> None:
    choices = _subparser_choices(parser)
    expected = {"template", "analyze", "suggest", "report", "write-set", "compare", "pareto"}
    assert expected.issubset(set(choices))


@pytest.mark.parametrize(
    "argv, handler, expected_car, expected_track",
    [
        (["template"], workflows._handle_template, "XFG", "generic"),
        (["suggest"], workflows._handle_suggest, "generic", "generic"),
    ],
)
def test_command_defaults(
    parser: argparse.ArgumentParser,
    argv: list[str],
    handler: object,
    expected_car: str,
    expected_track: str,
) -> None:
    namespace = parser.parse_args(argv)
    assert namespace.handler is handler
    assert namespace.car_model == expected_car
    assert namespace.track == expected_track


def test_write_set_default_export(parser: argparse.ArgumentParser) -> None:
    namespace = parser.parse_args(["write-set"])
    assert namespace.export_default == "markdown"

