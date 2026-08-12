"""Assembling results.json, the single artefact the site reads.

The calibration block is not an appendix. It ships in the same file as the
ranking, at the top level, because a ranking without it is a claim with no
evidence behind it. Anything consuming this file can therefore see how much the
ordering is worth before it renders a single row.

Models whose confidence intervals overlap share a `rank_group`. A renderer must
show them as tied, and the field exists so that rule is data rather than a note
in a style guide.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from statistics import median

from pydantic import BaseModel, ConfigDict

from konkord.calibrate import Breakdown, Calibration
from konkord.judge import resolve
from konkord.models import Comparison, Generation, Suite
from konkord.stats import (
    PairOutcome,
    bootstrap_win_rate_intervals,
    bradley_terry,
    tie_groups,
    win_rates,
)


class ReportError(Exception):
    """The stored results cannot be aggregated into one honest report."""


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelReport(Frozen):
    model: str
    rating: float
    win_rate: float
    ci_low: float
    ci_high: float
    rank_group: int
    comparisons: int
    generations: int
    failed_generations: int
    cost_usd: float
    median_latency_ms: int


class BreakdownReport(Frozen):
    key: str
    labelled: int
    agreed: int
    rate: float


class DisagreementReport(Frozen):
    """One entry in the failure gallery, with the judge's own reasoning."""

    task_id: str
    model_a: str
    model_b: str
    judge_winner: str | None
    human_winner: str | None
    judge_rationale: str | None


class CalibrationReport(Frozen):
    judge_models: tuple[str, ...]
    human_labels: int
    agreement: float
    kappa: float
    order_flip_rate: float
    by_task: tuple[BreakdownReport, ...]
    by_model_pair: tuple[BreakdownReport, ...]
    by_length_quartile: tuple[BreakdownReport, ...]
    failure_gallery: tuple[DisagreementReport, ...]


class AnswerReport(Frozen):
    """One model's answer to one task, as shown on the browse page."""

    model: str
    output: str
    error: str | None
    tokens_out: int
    cost_usd: float
    latency_ms: int


class JudgementReport(Frozen):
    """What the judge concluded about one pair on one task, and why.

    `rationales` holds one entry per presentation order, so a reader can see
    both halves of a flipped pair rather than a summary that hides the flip.
    """

    model_a: str
    model_b: str
    winner: str | None
    flipped: bool
    rationales: tuple[str, ...]


class TaskReport(Frozen):
    task_id: str
    prompt: str
    answers: tuple[AnswerReport, ...]
    judgements: tuple[JudgementReport, ...]


class Report(Frozen):
    suite: str
    models: tuple[ModelReport, ...]
    calibration: CalibrationReport
    tasks: tuple[TaskReport, ...]
    bootstrap_resamples: int
    bootstrap_seed: int
    #: The judge's system prompt, taken from the verdicts rather than recomposed
    #: from the suite file, so what is published is what actually ran. `None`
    #: when nothing has been judged, or when the verdicts predate the field.
    judge_prompt: str | None = None
    #: The suite's answer format, for renderers deciding how to show an answer.
    answer_language: str | None = None


def resolved_outcomes(judge_rows: Sequence[Comparison]) -> list[PairOutcome]:
    """Collapse each pair's two orderings into one outcome for the ranking.

    A pair the judge flipped on becomes a tie, so position bias reduces the
    separation between models instead of silently picking a winner.
    """
    grouped: dict[tuple[str, str, str], list[Comparison]] = {}
    for row in judge_rows:
        grouped.setdefault(row.pair_key, []).append(row)

    outcomes: list[PairOutcome] = []
    for group in grouped.values():
        first = group[0]
        verdict = resolve(group)
        winner = None if verdict == "tie" else (first.model_a if verdict == "a" else first.model_b)
        outcomes.append(PairOutcome(model_a=first.model_a, model_b=first.model_b, winner=winner))
    return outcomes


def published_prompt(judge_rows: Sequence[Comparison]) -> str | None:
    """The one prompt every verdict here was produced under.

    Returns `None` when there is nothing to publish: no verdicts, or verdicts
    written before they carried their prompt. Publishing the suite file's
    current rubric instead would show a reader a prompt that may never have run,
    which is the exact failure the published-prompt page exists to prevent.

    Raises:
        ReportError: if the verdicts were produced under more than one prompt.
    """
    distinct = {row.judge_prompt for row in judge_rows}
    if not distinct or distinct == {None}:
        return None
    prompts = sorted(p for p in distinct if p is not None)
    if len(distinct) > 1:
        raise ReportError(
            f"these verdicts were produced under {len(distinct)} different judge prompts. "
            f"A rating fitted across them means neither standard. Re-judge the suite from "
            f"scratch so every verdict shares one rubric."
        )
    return prompts[0]


def flip_rate(judge_rows: Sequence[Comparison]) -> float:
    """Share of pairs where the two presentation orders disagreed."""
    grouped: dict[tuple[str, str, str], list[Comparison]] = {}
    for row in judge_rows:
        grouped.setdefault(row.pair_key, []).append(row)
    if not grouped:
        return 0.0
    flipped = sum(1 for group in grouped.values() if len({c.winner_model for c in group}) != 1)
    return flipped / len(grouped)


def per_task(
    suite: Suite,
    generations: Sequence[Generation],
    judge_rows: Sequence[Comparison],
) -> list[TaskReport]:
    """The per-task detail the browse page renders.

    Aggregates alone cannot answer "what did this model actually write", which
    is the question the browse page exists to answer and the reason anyone
    shares a leaderboard rather than just reading the top line.
    """
    by_task: dict[str, list[Generation]] = {}
    for generation in generations:
        by_task.setdefault(generation.task_id, []).append(generation)

    pairs: dict[tuple[str, str, str], list[Comparison]] = {}
    for row in judge_rows:
        pairs.setdefault(row.pair_key, []).append(row)

    reports: list[TaskReport] = []
    for task in suite.tasks:
        answers = tuple(
            AnswerReport(
                model=g.model,
                output=g.output,
                error=g.error,
                tokens_out=g.tokens_out,
                cost_usd=g.cost_usd,
                latency_ms=g.latency_ms,
            )
            for g in sorted(by_task.get(task.id, []), key=lambda g: g.model)
        )
        judgements = []
        for key, group in sorted(pairs.items()):
            if key[0] != task.id:
                continue
            verdict = resolve(group)
            first = group[0]
            winner = (
                None if verdict == "tie" else (first.model_a if verdict == "a" else first.model_b)
            )
            judgements.append(
                JudgementReport(
                    model_a=first.model_a,
                    model_b=first.model_b,
                    winner=winner,
                    flipped=len({c.winner_model for c in group}) != 1,
                    rationales=tuple(c.rationale or "" for c in group),
                )
            )
        reports.append(
            TaskReport(
                task_id=task.id,
                prompt=task.prompt,
                answers=answers,
                judgements=tuple(judgements),
            )
        )
    return reports


def build(
    *,
    suite: Suite,
    generations: Sequence[Generation],
    judge_rows: Sequence[Comparison],
    calibration: Calibration,
    resamples: int = 1000,
    seed: int = 0,
) -> Report:
    """Aggregate everything into one report.

    Raises:
        ReportError: if the verdicts do not share a single judge prompt.
    """
    outcomes = resolved_outcomes(judge_rows)
    ratings = bradley_terry(outcomes)
    rates = win_rates(outcomes)
    intervals = bootstrap_win_rate_intervals(outcomes, resamples=resamples, seed=seed)
    groups = tie_groups(ratings, intervals)

    faced: dict[str, int] = {}
    for outcome in outcomes:
        for name in (outcome.model_a, outcome.model_b):
            faced[name] = faced.get(name, 0) + 1

    models = []
    for name in sorted(ratings, key=lambda n: (groups[n], -ratings[n], n)):
        mine = [g for g in generations if g.model == name]
        succeeded = [g for g in mine if g.error is None]
        low, high = intervals.get(name, (0.0, 0.0))
        models.append(
            ModelReport(
                model=name,
                rating=ratings[name],
                win_rate=rates.get(name, 0.0),
                ci_low=low,
                ci_high=high,
                rank_group=groups[name],
                comparisons=faced.get(name, 0),
                generations=len(mine),
                failed_generations=len(mine) - len(succeeded),
                cost_usd=sum(g.cost_usd for g in mine),
                median_latency_ms=int(median([g.latency_ms for g in succeeded]))
                if succeeded
                else 0,
            )
        )

    return Report(
        suite=suite.name,
        models=tuple(models),
        calibration=CalibrationReport(
            judge_models=calibration.judge_models,
            human_labels=calibration.labelled,
            agreement=calibration.agreement,
            kappa=calibration.kappa,
            order_flip_rate=flip_rate(judge_rows),
            by_task=_breakdowns(calibration.by_task),
            by_model_pair=_breakdowns(calibration.by_model_pair),
            by_length_quartile=_breakdowns(calibration.by_length_quartile),
            failure_gallery=tuple(
                DisagreementReport(
                    task_id=d.task_id,
                    model_a=d.model_a,
                    model_b=d.model_b,
                    judge_winner=d.judge_winner,
                    human_winner=d.human_winner,
                    judge_rationale=d.judge_rationale,
                )
                for d in calibration.disagreements
            ),
        ),
        tasks=tuple(per_task(suite, generations, judge_rows)),
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
        judge_prompt=published_prompt(judge_rows),
        answer_language=suite.answer_language,
    )


def write(report: Report, path: Path) -> None:
    path.write_text(json.dumps(report.model_dump(), indent=2) + "\n", encoding="utf-8")


def _breakdowns(items: Sequence[Breakdown]) -> tuple[BreakdownReport, ...]:
    return tuple(
        BreakdownReport(
            key=item.key,
            labelled=item.labelled,
            agreed=item.agreed,
            rate=item.rate,
        )
        for item in items
    )
