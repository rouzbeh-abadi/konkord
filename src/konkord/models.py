"""Typed contracts shared by every stage of the pipeline.

Only these models cross a module boundary — no bare dicts, no tuples standing in
for records. Every model is frozen and rejects unknown fields, so a typo in a
suite file or a renamed column fails at construction rather than silently
producing a wrong leaderboard.

Sequence fields are tuples rather than lists: `frozen=True` blocks attribute
assignment but would still let a caller mutate a list in place, which is exactly
the kind of shared-state bug that is invisible in a results table.
"""

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Task ids appear in file names, DuckDB keys and URLs, so they are kept to a
#: conservative slug rather than arbitrary text.
TASK_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"

Verdict = Literal["a", "b", "tie"]
"""Which of the two presented answers won. Positional, not a model name."""

Order = Literal["ab", "ba"]
"""Which answer was shown first — the position-bias control."""

Source = Literal["judge", "human"]
"""Who produced a verdict."""


class Frozen(BaseModel):
    """Base for every contract in this module."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Task(Frozen):
    """A single prompt, with its checks already resolved against suite defaults."""

    id: str = Field(pattern=TASK_ID_PATTERN)
    prompt: str = Field(min_length=1)
    context: str | None = None
    checks: tuple[str, ...] = ()
    reference: str | None = None


class Suite(Frozen):
    """A named collection of tasks, as loaded from a YAML file."""

    name: str = Field(min_length=1)
    context: str | None = None
    tasks: tuple[Task, ...] = Field(min_length=1)

    @field_validator("tasks")
    @classmethod
    def _reject_duplicate_ids(cls, tasks: tuple[Task, ...]) -> tuple[Task, ...]:
        counts = Counter(task.id for task in tasks)
        duplicates = sorted(task_id for task_id, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate task ids: {', '.join(duplicates)}")
        return tasks

    def task(self, task_id: str) -> Task:
        """Look up a task by id, raising `KeyError` if the suite has no such task."""
        for candidate in self.tasks:
            if candidate.id == task_id:
                return candidate
        raise KeyError(task_id)


class Generation(Frozen):
    """One model's answer to one task, plus what it cost to obtain."""

    task_id: str
    model: str
    output: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    error: str | None = None


class CheckResult(Frozen):
    """The outcome of one deterministic check against one generation."""

    task_id: str
    model: str
    check: str
    passed: bool
    detail: str | None = None


class Comparison(Frozen):
    """One pairwise verdict, from either the judge or a human.

    `model_a` / `model_b` record who was compared; `order` records who was shown
    first. Both are needed: the judge sees positions, the leaderboard needs
    identities, and the flip-rate diagnostic needs to know which was which.
    """

    task_id: str
    model_a: str
    model_b: str
    order: Order
    winner: Verdict
    source: Source
    rationale: str | None = None
    judge_model: str | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> "Comparison":
        if self.model_a == self.model_b:
            raise ValueError(f"a model cannot be compared with itself: {self.model_a}")
        if self.source == "judge" and self.judge_model is None:
            raise ValueError("judge comparisons must record judge_model")
        if self.source == "human" and self.judge_model is not None:
            raise ValueError("human comparisons must not record judge_model")
        return self

    @property
    def pair_key(self) -> tuple[str, str, str]:
        """Orientation-independent identity of this comparison.

        `calibrate` joins human labels to judge verdicts on this key, so it must
        not depend on which model happened to be called `a`. Compare
        `winner_model`, never `winner`, across two comparisons.
        """
        first, second = sorted((self.model_a, self.model_b))
        return (self.task_id, first, second)

    @property
    def winner_model(self) -> str | None:
        """The winning model's name, or `None` for a tie."""
        if self.winner == "tie":
            return None
        return self.model_a if self.winner == "a" else self.model_b


class JudgeFailure(Frozen):
    """A judge response that could not be parsed, kept rather than discarded.

    An unparseable verdict is never coerced into a winner. It is recorded here
    with the raw response so the failure is auditable — a judge that produces
    many of these is itself a finding.
    """

    task_id: str
    model_a: str
    model_b: str
    order: Order
    judge_model: str
    reason: str
    raw: str
