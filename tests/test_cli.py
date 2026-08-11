"""Smoke tests for the command surface.

These assert the CLI is wired up. Each command's real behaviour is tested in the
module that implements it.
"""

from typer.testing import CliRunner

from konkord import __version__
from konkord.cli import app

runner = CliRunner()

COMMANDS = ["run", "judge", "label", "calibrate", "report"]


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


def test_nothing_advertised_is_a_stub() -> None:
    """`--help` must not list a command that only exists to say it does not work.

    A stub in the help text advertises capability the tool does not have, which
    is the failure mode this project exists to argue against.
    """
    for command in COMMANDS:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, command
        assert "not implemented" not in result.stdout.lower(), command
