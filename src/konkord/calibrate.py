"""Judge versus human agreement: the number the whole project exists to produce.

A leaderboard built on an automated judge is only worth what the judge is worth.
This module joins the human labels to the judge verdicts on the same comparisons
and reports how often they agreed, corrected for chance.

Two details decide whether the number means anything:

* The join is on `pair_key`, which is orientation independent, and the
  comparison is between winning model names rather than the positional `a`/`b`
  field. Comparing the raw field would score a judge and a human who agree
  perfectly as disagreeing whenever the labeller happened to see the pair the
  other way round.
* The judge side is resolved from both orderings first. A judge that flipped on
  a pair has no opinion about it, and recording one ordering as its verdict
  would credit it with a view it did not hold.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from konkord.judge import resolve
from konkord.models import Comparison, Generation

#: Chance-corrected agreement is undefined when both raters always say the same
#: thing. Convention here: perfect agreement scores 1.0, anything else 0.0.
_DEGENERATE_KAPPA = (1.0, 0.0)


@dataclass(frozen=True, slots=True)
class AlignedPair:
    """One comparison both the judge and a human have an opinion about."""

    task_id: str
    model_a: str
    model_b: str
    judge_winner: str | None
    human_winner: str | None
    judge_rationale: str | None
    answer_chars: int

    @property
    def agreed(self) -> bool:
        return self.judge_winner == self.human_winner

    @property
    def judge_label(self) -> str:
        return self.judge_winner or "tie"

    @property
    def human_label(self) -> str:
        return self.human_winner or "tie"


@dataclass(frozen=True, slots=True)
class Breakdown:
    """Agreement within one slice of the labelled sample."""

    key: str
    labelled: int
    agreed: int

    @property
    def rate(self) -> float:
        return self.agreed / self.labelled if self.labelled else 0.0


@dataclass(frozen=True, slots=True)
class Calibration:
    """Everything `konkord calibrate` reports."""

    labelled: int
    agreed: int
    kappa: float
    judge_models: tuple[str, ...]
    by_task: tuple[Breakdown, ...]
    by_model_pair: tuple[Breakdown, ...]
    by_length_quartile: tuple[Breakdown, ...]
    disagreements: tuple[AlignedPair, ...]

    @property
    def agreement(self) -> float:
        return self.agreed / self.labelled if self.labelled else 0.0


def align(
    judge_rows: Sequence[Comparison],
    human_rows: Sequence[Comparison],
    generations: Iterable[Generation],
) -> list[AlignedPair]:
    """Pair up the judge's resolved verdict with the human label for each pair."""
    lengths = {(g.task_id, g.model): len(g.output) for g in generations}

    judged: dict[tuple[str, str, str], list[Comparison]] = {}
    for row in judge_rows:
        judged.setdefault(row.pair_key, []).append(row)

    aligned: list[AlignedPair] = []
    for human in human_rows:
        group = judged.get(human.pair_key)
        if not group:
            continue
        verdict = resolve(group)
        first = group[0]
        judge_winner = (
            None if verdict == "tie" else (first.model_a if verdict == "a" else first.model_b)
        )
        answer_chars = lengths.get((human.task_id, human.model_a), 0) + lengths.get(
            (human.task_id, human.model_b), 0
        )
        aligned.append(
            AlignedPair(
                task_id=human.task_id,
                model_a=human.model_a,
                model_b=human.model_b,
                judge_winner=judge_winner,
                human_winner=human.winner_model,
                judge_rationale=first.rationale,
                answer_chars=answer_chars // 2,
            )
        )
    return aligned


def cohens_kappa(pairs: Sequence[AlignedPair]) -> float:
    """Chance-corrected agreement between the judge and the human.

    Raw agreement flatters a judge on a suite where one model dominates: always
    naming the favourite scores well without the judge reading anything. Kappa
    subtracts the agreement two raters would reach by guessing with the same
    marginal frequencies.
    """
    if not pairs:
        return 0.0
    labels = sorted({p.judge_label for p in pairs} | {p.human_label for p in pairs})
    total = len(pairs)

    observed = sum(1 for p in pairs if p.agreed) / total
    expected = sum(
        (sum(1 for p in pairs if p.judge_label == label) / total)
        * (sum(1 for p in pairs if p.human_label == label) / total)
        for label in labels
    )
    if expected >= 1.0:
        return _DEGENERATE_KAPPA[0] if observed >= 1.0 else _DEGENERATE_KAPPA[1]
    return (observed - expected) / (1.0 - expected)


def by_task(pairs: Sequence[AlignedPair]) -> list[Breakdown]:
    return _group(pairs, lambda p: p.task_id)


def by_model_pair(pairs: Sequence[AlignedPair]) -> list[Breakdown]:
    return _group(pairs, lambda p: f"{p.model_a} vs {p.model_b}")


def by_length_quartile(pairs: Sequence[AlignedPair]) -> list[Breakdown]:
    """Agreement split by how long the answers were.

    This is what surfaces verbosity bias. A judge that agrees with the human on
    short answers and diverges on long ones is rewarding length, and the ranking
    it produces measures wordiness rather than quality.
    """
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda p: (p.answer_chars, p.task_id, p.model_a))
    size = len(ordered)
    quartiles: list[Breakdown] = []
    for q in range(4):
        chunk = ordered[q * size // 4 : (q + 1) * size // 4]
        if not chunk:
            continue
        quartiles.append(
            Breakdown(
                key=f"Q{q + 1} ({chunk[0].answer_chars}-{chunk[-1].answer_chars} chars)",
                labelled=len(chunk),
                agreed=sum(1 for p in chunk if p.agreed),
            )
        )
    return quartiles


def calibrate(
    judge_rows: Sequence[Comparison],
    human_rows: Sequence[Comparison],
    generations: Iterable[Generation],
) -> Calibration:
    """Compare human labels against judge verdicts on the same comparisons."""
    pairs = align(judge_rows, human_rows, generations)
    return Calibration(
        labelled=len(pairs),
        agreed=sum(1 for p in pairs if p.agreed),
        kappa=cohens_kappa(pairs),
        judge_models=tuple(sorted({r.judge_model for r in judge_rows if r.judge_model})),
        by_task=tuple(by_task(pairs)),
        by_model_pair=tuple(by_model_pair(pairs)),
        by_length_quartile=tuple(by_length_quartile(pairs)),
        disagreements=tuple(p for p in pairs if not p.agreed),
    )


def _group(
    pairs: Sequence[AlignedPair],
    key: Callable[[AlignedPair], str],
) -> list[Breakdown]:
    buckets: dict[str, list[AlignedPair]] = {}
    for pair in pairs:
        buckets.setdefault(key(pair), []).append(pair)
    return [
        Breakdown(
            key=name,
            labelled=len(group),
            agreed=sum(1 for p in group if p.agreed),
        )
        for name, group in sorted(buckets.items())
    ]
