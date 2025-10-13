from __future__ import annotations

import argparse

import pytest

pytest.importorskip("numpy")

from tnfr_lfs.cli.parser import build_parser
from tnfr_lfs.cli import workflows
from tnfr_lfs.cli import compare as compare_command


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


@pytest.mark.parametrize(
    "argv, handler",
    [
        (["report"], workflows._handle_report),
        (
            ["compare", "baseline.raf", "variant.raf"],
            compare_command.handle,
        ),
    ],
)
def test_report_and_compare_inherit_report_defaults(
    argv: list[str], handler: object
) -> None:
    config = {"report": {"car_model": "FXR", "track": "BL1"}}
    parser = build_parser(config)
    namespace = parser.parse_args(argv)
    assert namespace.handler is handler
    assert workflows._default_car_model(config) == "FXR"
    assert workflows._default_track_name(config) == "BL1"


def test_compare_specific_defaults_are_used() -> None:
    config = {"compare": {"car_model": "RB4", "track": "SO1"}}
    assert workflows._default_car_model(config) == "RB4"
    assert workflows._default_track_name(config) == "SO1"


@pytest.mark.parametrize(
    "argv, expected_exit_code, stdout_snippets, stderr_snippets",
    [
        pytest.param(
            ["--help"],
            0,
            ["TNFR × LFS", "diagnose"],
            [],
            id="help",
        ),
        pytest.param(
            ["unknown"],
            2,
            [],
            ["invalid choice: 'unknown'"],
            id="unknown-command",
        ),
    ],
)
def test_cli_entrypoint_outcomes(
    cli_runner,
    argv: list[str],
    expected_exit_code: int,
    stdout_snippets: list[str],
    stderr_snippets: list[str],
) -> None:
    result = cli_runner(argv)

    assert result.exit_code == expected_exit_code
    for snippet in stdout_snippets:
        assert snippet in result.stdout
    for snippet in stderr_snippets:
        assert snippet in result.stderr

