"""Choosing which comparisons a person is asked to label.

The sample decides what the calibration number is computed from, so it has to be
spread rather than convenient. Concentrating labels on one task or one model pair
would produce an agreement rate that describes that corner and nothing else.

Sampling is deterministic given `(seed, candidates)`. That is what makes a
labelling session resumable: closing the app and reopening it reproduces the same
queue, minus whatever has already been labelled.
"""

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from konkord.models import Comparison, Order, Verdict


@dataclass(frozen=True, slots=True)
class LabelItem:
    """One comparison to put in front of a person, in a chosen orientation."""

    task_id: str
    model_a: str
    model_b: str
    order: Order

    @property
    def first_model(self) -> str:
        """Shown as Answer 1. Never displayed — only used to fetch the text."""
        return self.model_a if self.order == "ab" else self.model_b

    @property
    def second_model(self) -> str:
        return self.model_b if self.order == "ab" else self.model_a

    @property
    def pair_key(self) -> tuple[str, str, str]:
        return (self.task_id, self.model_a, self.model_b)


def candidates(comparisons: Iterable[Comparison]) -> list[tuple[str, str, str]]:
    """Distinct (task, model_a, model_b) triples the judge has an opinion on.

    Sorted, so the sample depends only on the seed and the set — not on the order
    rows happened to come back from the database.
    """
    return sorted({c.pair_key for c in comparisons})


def stratified_sample(
    pool: Sequence[tuple[str, str, str]],
    *,
    n: int,
    seed: int,
    exclude: Iterable[tuple[str, str, str]] = (),
) -> list[LabelItem]:
    """Up to `n` comparisons, spread evenly across tasks and model pairs.

    Round-robins across tasks, so task counts differ by at most one however the
    pool is shaped. Within a task, pairs are shuffled, which spreads the model
    pairs without needing a second explicit pass.

    Orientation is randomised per item: a labeller who always saw the same model
    first would acquire the position bias the judge is being measured for.
    """
    skip = set(exclude)
    rng = random.Random(seed)

    by_task: dict[str, list[tuple[str, str, str]]] = {}
    for triple in sorted(pool):
        if triple in skip:
            continue
        by_task.setdefault(triple[0], []).append(triple)

    task_ids = sorted(by_task)
    rng.shuffle(task_ids)
    for task_id in task_ids:
        rng.shuffle(by_task[task_id])

    chosen: list[tuple[str, str, str]] = []
    while len(chosen) < n:
        drained = True
        for task_id in task_ids:
            queue = by_task[task_id]
            if not queue:
                continue
            drained = False
            chosen.append(queue.pop())
            if len(chosen) == n:
                break
        if drained:
            break

    return [
        LabelItem(
            task_id=task_id,
            model_a=model_a,
            model_b=model_b,
            order=("ab" if rng.random() < 0.5 else "ba"),
        )
        for task_id, model_a, model_b in chosen
    ]


def already_labelled(comparisons: Iterable[Comparison]) -> set[tuple[str, str, str]]:
    """Pairs a person has already judged, in whichever orientation they saw."""
    return {c.pair_key for c in comparisons if c.source == "human"}


def to_verdict(item: LabelItem, position: str) -> Verdict:
    """Turn "Answer 1 won" into a statement about a model.

    Lives here rather than in the app so it can be tested without Streamlit:
    getting it wrong would invert half the human labels, and inverted labels
    would look like judge disagreement rather than a bug.
    """
    if position == "tie":
        return "tie"
    won = item.first_model if position == "first" else item.second_model
    return "a" if won == item.model_a else "b"
