"""Typer entrypoint.

The command surface is fixed here in phase 1 so it stops moving; each later phase
replaces one stub with a real implementation.
"""

import sys
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


def _load_env() -> None:
    """Read `.env` from the working directory, if there is one.

    Values already exported win, so a shell export overrides the file rather
    than the other way round. That keeps a one-off `OPENAI_API_KEY=... konkord
    run` doing what it looks like it does.
    """
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env", override=False)


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
    """Konkord: LLM eval harness with judge calibration."""
    _load_env()


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
            f"{', '.join(summary.models_without_pricing)}; their cost is reported as 0",
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
    models: Annotated[
        str,
        typer.Option("--models", help="Comma-separated models to rank against each other."),
    ],
    db: Annotated[Path, typer.Option("--db", help="DuckDB results file.")] = Path("konkord.duckdb"),
    cache_dir: Annotated[
        Path, typer.Option("--cache", help="Directory for the response cache.")
    ] = Path(".konkord_cache"),
    concurrency: Annotated[
        int, typer.Option("--concurrency", min=1, help="Maximum judge calls in flight.")
    ] = 8,
    max_attempts: Annotated[
        int, typer.Option("--max-attempts", min=1, help="Attempts per call before giving up.")
    ] = 4,
) -> None:
    """Score every model pair with an LLM judge, in both orderings."""
    import asyncio

    from konkord.cache import ResponseCache
    from konkord.judge import JudgeConfig, JudgeError, judge_suite
    from konkord.litellm_provider import LiteLLMCompleter
    from konkord.store import ResultStore
    from konkord.suites import SuiteError, load_suite

    model_list = _parse_models(models)
    try:
        loaded = load_suite(suite)
    except SuiteError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    with ResponseCache(cache_dir) as cache, ResultStore(db) as store:
        try:
            summary = asyncio.run(
                judge_suite(
                    suite=loaded,
                    models=model_list,
                    judge_model=judge_model,
                    completer=LiteLLMCompleter(),
                    cache=cache,
                    store=store,
                    config=JudgeConfig(concurrency=concurrency, max_attempts=max_attempts),
                )
            )
        except JudgeError as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

    typer.echo(
        f"judged {summary.judged}, already present {summary.skipped}, "
        f"unjudgeable {summary.unjudgeable}"
    )
    typer.echo(f"cost ${summary.cost_usd:.4f}")
    typer.echo(
        f"order-flip rate {summary.flip_rate:.1%} "
        f"({summary.flips} of {summary.pairs} pairs disagreed between orderings)"
    )
    if summary.parse_failures or summary.call_failures:
        typer.secho(
            f"{summary.parse_failures} unparseable verdicts, "
            f"{summary.call_failures} failed calls; see the judge_failures table",
            err=True,
            fg=typer.colors.YELLOW,
        )


@app.command()
def label(
    suite: SuiteOption,
    n: Annotated[
        int,
        typer.Option("--n", min=1, help="Number of comparisons to sample for labelling."),
    ] = 100,
    db: Annotated[Path, typer.Option("--db", help="DuckDB results file.")] = Path("konkord.duckdb"),
    seed: Annotated[
        int, typer.Option("--seed", help="Sampling seed; the same seed resumes the same queue.")
    ] = 0,
    port: Annotated[int, typer.Option("--port", help="Port for the local app.")] = 8501,
) -> None:
    """Launch the local blind labeller."""
    import importlib.util
    import os
    import subprocess

    if importlib.util.find_spec("streamlit") is None:
        typer.secho(
            "the labeller needs Streamlit: install it with `pip install 'konkord[label]'`",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    from konkord.labeling import app as labeler

    environment = dict(os.environ)
    environment.update(
        {
            "KONKORD_DB": str(db.resolve()),
            "KONKORD_SUITE_PATH": str(suite.resolve()),
            "KONKORD_LABEL_N": str(n),
            "KONKORD_LABEL_SEED": str(seed),
        }
    )
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(labeler.__file__).resolve()),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    typer.echo(f"labelling {suite.name} on http://localhost:{port}, Ctrl-C to stop")
    raise typer.Exit(code=subprocess.call(command, env=environment))


@app.command()
def calibrate(
    suite: SuiteOption,
    db: Annotated[Path, typer.Option("--db", help="DuckDB results file.")] = Path("konkord.duckdb"),
    gallery: Annotated[
        int, typer.Option("--gallery", min=0, help="Disagreements to print in full.")
    ] = 10,
) -> None:
    """Compare human labels against judge verdicts."""
    from konkord.calibrate import calibrate as run_calibration
    from konkord.store import ResultStore
    from konkord.suites import SuiteError, load_suite

    try:
        loaded = load_suite(suite)
    except SuiteError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    with ResultStore(db) as store:
        result = run_calibration(
            store.comparisons(loaded.name, "judge"),
            store.comparisons(loaded.name, "human"),
            store.generations(loaded.name),
        )

    if not result.labelled:
        typer.secho(
            "no human labels yet, so there is nothing to calibrate against. "
            "Run `konkord label` first.",
            err=True,
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    typer.echo(f"judge: {', '.join(result.judge_models) or 'unknown'}")
    typer.echo(f"human labels: {result.labelled}")
    typer.echo(f"agreement: {result.agreement:.1%} ({result.agreed}/{result.labelled})")
    typer.echo(f"Cohen's kappa: {result.kappa:.3f}")

    for title, rows in (
        ("by task", result.by_task),
        ("by model pair", result.by_model_pair),
        ("by answer-length quartile", result.by_length_quartile),
    ):
        typer.echo(f"\n{title}:")
        for row in rows:
            typer.echo(f"  {row.key:<40} {row.rate:>6.1%}  (n={row.labelled})")

    if result.disagreements and gallery:
        typer.echo(f"\nfailure gallery ({len(result.disagreements)} disagreements):")
        for item in result.disagreements[:gallery]:
            typer.echo(
                f"\n  {item.task_id}: {item.model_a} vs {item.model_b}\n"
                f"    judge said {item.judge_winner or 'tie'}, "
                f"human said {item.human_winner or 'tie'}\n"
                f"    judge rationale: {(item.judge_rationale or '').strip()[:400]}"
            )


@app.command()
def report(
    suite: SuiteOption,
    db: Annotated[Path, typer.Option("--db", help="DuckDB results file.")] = Path("konkord.duckdb"),
    out: Annotated[Path, typer.Option("--out", help="Where to write the report.")] = Path(
        "results.json"
    ),
    resamples: Annotated[
        int, typer.Option("--resamples", min=1000, help="Bootstrap resamples.")
    ] = 1000,
    seed: Annotated[int, typer.Option("--seed", help="Bootstrap seed.")] = 0,
) -> None:
    """Aggregate everything into results.json."""
    from konkord import report as reporting
    from konkord.calibrate import calibrate as run_calibration
    from konkord.store import ResultStore
    from konkord.suites import SuiteError, load_suite

    try:
        loaded = load_suite(suite)
    except SuiteError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    with ResultStore(db) as store:
        generations = store.generations(loaded.name)
        judge_rows = store.comparisons(loaded.name, "judge")
        human_rows = store.comparisons(loaded.name, "human")

    if not judge_rows:
        typer.secho(
            "no judge comparisons yet, so there is nothing to rank. Run `konkord judge` first.",
            err=True,
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    built = reporting.build(
        suite=loaded,
        generations=generations,
        judge_rows=judge_rows,
        calibration=run_calibration(judge_rows, human_rows, generations),
        resamples=resamples,
        seed=seed,
    )
    reporting.write(built, out)

    typer.echo(f"wrote {out}")
    for entry in built.models:
        typer.echo(
            f"  [{entry.rank_group}] {entry.model:<28} rating {entry.rating:>6.3f}  "
            f"win {entry.win_rate:>5.1%} "
            f"[{entry.ci_low:.1%}, {entry.ci_high:.1%}]"
        )
    typer.echo("  models sharing a bracket number are tied; do not order them")

    block = built.calibration
    if block.human_labels:
        typer.echo(
            f"\ncalibration: {block.agreement:.1%} agreement, "
            f"kappa {block.kappa:.3f}, over {block.human_labels} human labels"
        )
    else:
        typer.secho(
            "\nwarning: no human labels, so this ranking ships uncalibrated "
            "and should not be published",
            err=True,
            fg=typer.colors.YELLOW,
        )
