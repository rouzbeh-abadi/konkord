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
    db: Annotated[
        Path,
        typer.Option("--db", help="DuckDB results file."),
    ] = Path("konkord.duckdb"),
    cache_dir: Annotated[
        Path,
        typer.Option("--cache", help="Directory for the response cache."),
    ] = Path(".konkord_cache"),
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", min=1, help="Maximum model calls in flight."),
    ] = 8,
    max_attempts: Annotated[
        int,
        typer.Option("--max-attempts", min=1, help="Attempts per call before giving up."),
    ] = 4,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", min=1, help="Output token cap per call."),
    ] = 4096,
) -> None:
    """Generate one output per (task x model), concurrently and resumably."""
    # Imported here so `konkord --help` does not pay for litellm's import.
    import asyncio

    from konkord.cache import ResponseCache
    from konkord.litellm_provider import LiteLLMCompleter
    from konkord.runner import Outcome, RunConfig, run_suite
    from konkord.store import ResultStore
    from konkord.suites import SuiteError, load_suite

    model_list = _parse_models(models)
    try:
        loaded = load_suite(suite)
    except SuiteError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"{loaded.name}: {len(loaded.tasks)} tasks x {len(model_list)} models "
        f"= {len(loaded.tasks) * len(model_list)} generations"
    )

    def report(generation: object, outcome: Outcome) -> None:
        marker = {"generated": "+", "cached": "=", "failed": "!"}[str(outcome)]
        task_id = getattr(generation, "task_id", "?")
        model = getattr(generation, "model", "?")
        typer.echo(f"  {marker} {task_id} / {model}")

    with ResponseCache(cache_dir) as cache, ResultStore(db) as store:
        summary = asyncio.run(
            run_suite(
                suite=loaded,
                models=model_list,
                completer=LiteLLMCompleter(),
                cache=cache,
                store=store,
                config=RunConfig(
                    concurrency=concurrency,
                    max_attempts=max_attempts,
                    max_tokens=max_tokens,
                ),
                on_result=report,
            )
        )

    typer.echo(
        f"\ngenerated {summary.generated}, cached {summary.cached}, "
        f"failed {summary.failed}, already present {summary.skipped}"
    )
    typer.echo(f"cost ${summary.cost_usd:.4f}")
    if summary.models_without_pricing:
        typer.secho(
            "warning: no pricing data for "
            f"{', '.join(summary.models_without_pricing)} — their cost is reported as 0",
            err=True,
            fg=typer.colors.YELLOW,
        )
    if summary.failed:
        raise typer.Exit(code=1)


def _parse_models(raw: str) -> list[str]:
    """Split --models, dropping blanks and duplicates but keeping order."""
    seen = [name.strip() for name in raw.split(",") if name.strip()]
    unique = list(dict.fromkeys(seen))
    if not unique:
        typer.secho("--models requires at least one model name", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    return unique


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
