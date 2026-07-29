"""Typer entrypoint.

The command surface is fixed here in phase 1 so it stops moving; each later phase
replaces one stub with a real implementation.
"""

from pathlib import Path
from typing import Annotated

import typer

from konkord import __version__

app = typer.Typer(
    name="konkord",
    help="Rank LLMs on a task suite, and measure whether the automated judge can be trusted.",
    no_args_is_help=True,
    add_completion=False,
)

SuiteOption = Annotated[
    Path,
    typer.Option(
        "--suite",
        help="Path to a task suite YAML file.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
]


def _not_implemented(command: str, phase: int) -> None:
    """Fail loudly for a command whose phase has not landed yet."""
    typer.secho(
        f"konkord {command}: not implemented yet (build phase {phase}).",
        err=True,
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=1)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"konkord {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Konkord — LLM eval harness with judge calibration."""


@app.command()
def run(
    suite: SuiteOption,
    models: Annotated[
        str,
        typer.Option("--models", help="Comma-separated model identifiers to generate with."),
    ],
) -> None:
    """Generate one output per (task x model), concurrently and resumably."""
    _not_implemented("run", phase=3)


@app.command()
def check(suite: SuiteOption) -> None:
    """Run the suite's deterministic checks against each generation."""
    _not_implemented("check", phase=4)


@app.command()
def judge(
    suite: SuiteOption,
    judge_model: Annotated[
        str,
        typer.Option("--judge", help="Judge model; must be a different provider family."),
    ],
) -> None:
    """Score every model pair with an LLM judge, in both orderings."""
    _not_implemented("judge", phase=5)


@app.command()
def label(
    suite: SuiteOption,
    n: Annotated[
        int,
        typer.Option("--n", min=1, help="Number of comparisons to sample for labelling."),
    ] = 100,
) -> None:
    """Launch the local blind labeller."""
    _not_implemented("label", phase=6)


@app.command()
def calibrate() -> None:
    """Compare human labels against judge verdicts."""
    _not_implemented("calibrate", phase=7)


@app.command()
def report() -> None:
    """Aggregate everything into results.json."""
    _not_implemented("report", phase=7)
