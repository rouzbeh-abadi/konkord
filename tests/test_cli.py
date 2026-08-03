"""Smoke tests for the command surface.

These assert the CLI is wired up, not that any phase works. Real behaviour is
tested in the phase that introduces it.
"""

from typer.testing import CliRunner

from konkord import __version__
from konkord.cli import app

runner = CliRunner()

COMMANDS = ["run", "check", "judge", "label", "calibrate", "report"]


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.stdout


def test_no_args_shows_help_without_crashing() -> None:
    result = runner.invoke(app, [])
    assert "Usage:" in result.stdout


def test_unimplemented_commands_fail_loudly() -> None:
    """A stub must exit non-zero, never pretend to succeed."""
    result = runner.invoke(app, ["check", "--suite", "suites/python_codegen.yaml"])
    assert result.exit_code == 1
    assert "not implemented" in result.stderr
