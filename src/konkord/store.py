"""DuckDB result store — one file, no server.

Rows are keyed by (suite, task_id, model). Task ids are only unique within a
suite, so the suite name is part of the key here even though it is not part of
`Generation`: the suite is the partition, not a property of one answer.

Writes are idempotent. Re-running a suite overwrites the matching row rather
than accumulating duplicates, which is what makes `konkord run` safe to repeat.
"""

from pathlib import Path
from types import TracebackType
from typing import Self, cast

import duckdb

from konkord.models import (
    Comparison,
    Generation,
    JudgeFailure,
    Order,
    Source,
    Verdict,
)

_SCHEMA = (
    """
CREATE TABLE IF NOT EXISTS generations (
    suite      VARCHAR NOT NULL,
    task_id    VARCHAR NOT NULL,
    model      VARCHAR NOT NULL,
    output     VARCHAR NOT NULL,
    tokens_in  BIGINT  NOT NULL,
    tokens_out BIGINT  NOT NULL,
    cost_usd   DOUBLE  NOT NULL,
    latency_ms BIGINT  NOT NULL,
    error      VARCHAR,
    PRIMARY KEY (suite, task_id, model)
)
""",
    # One row per presentation order, not per pair: the flip-rate diagnostic
    # needs both orderings kept apart, and collapsing them here would destroy
    # the evidence that the judge is position-biased.
    """
CREATE TABLE IF NOT EXISTS comparisons (
    suite       VARCHAR NOT NULL,
    task_id     VARCHAR NOT NULL,
    model_a     VARCHAR NOT NULL,
    model_b     VARCHAR NOT NULL,
    "order"     VARCHAR NOT NULL,
    winner      VARCHAR NOT NULL,
    source      VARCHAR NOT NULL,
    rationale   VARCHAR,
    judge_model VARCHAR,
    PRIMARY KEY (suite, task_id, model_a, model_b, "order", source)
)
""",
    """
CREATE TABLE IF NOT EXISTS judge_failures (
    suite       VARCHAR NOT NULL,
    task_id     VARCHAR NOT NULL,
    model_a     VARCHAR NOT NULL,
    model_b     VARCHAR NOT NULL,
    "order"     VARCHAR NOT NULL,
    judge_model VARCHAR NOT NULL,
    reason      VARCHAR NOT NULL,
    raw         VARCHAR NOT NULL,
    PRIMARY KEY (suite, task_id, model_a, model_b, "order")
)
""",
)


class ResultStore:
    """Every stage's view of what has already been produced."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(path))
        for statement in _SCHEMA:
            self._connection.execute(statement)

    def record(self, suite: str, generation: Generation) -> None:
        """Insert or replace one generation."""
        self._connection.execute(
            """
            INSERT OR REPLACE INTO generations
                (suite, task_id, model, output, tokens_in, tokens_out,
                 cost_usd, latency_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                suite,
                generation.task_id,
                generation.model,
                generation.output,
                generation.tokens_in,
                generation.tokens_out,
                generation.cost_usd,
                generation.latency_ms,
                generation.error,
            ],
        )

    def completed(self, suite: str) -> set[tuple[str, str]]:
        """The (task_id, model) pairs already recorded, successful or not.

        A recorded failure counts as done: a permanent error is a result, and
        re-running the suite should not silently retry it.
        """
        rows = self._connection.execute(
            "SELECT task_id, model FROM generations WHERE suite = ?", [suite]
        ).fetchall()
        return {(str(task_id), str(model)) for task_id, model in rows}

    def generations(self, suite: str) -> list[Generation]:
        """Every generation recorded for a suite, ordered for stable output."""
        rows = self._connection.execute(
            """
            SELECT task_id, model, output, tokens_in, tokens_out,
                   cost_usd, latency_ms, error
            FROM generations WHERE suite = ?
            ORDER BY task_id, model
            """,
            [suite],
        ).fetchall()
        return [
            Generation(
                task_id=str(row[0]),
                model=str(row[1]),
                output=str(row[2]),
                tokens_in=int(row[3]),
                tokens_out=int(row[4]),
                cost_usd=float(row[5]),
                latency_ms=int(row[6]),
                error=None if row[7] is None else str(row[7]),
            )
            for row in rows
        ]

    def record_comparison(self, suite: str, comparison: Comparison) -> None:
        """Insert or replace one comparison, in one presentation order."""
        self._connection.execute(
            """
            INSERT OR REPLACE INTO comparisons
                (suite, task_id, model_a, model_b, "order", winner,
                 source, rationale, judge_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                suite,
                comparison.task_id,
                comparison.model_a,
                comparison.model_b,
                comparison.order,
                comparison.winner,
                comparison.source,
                comparison.rationale,
                comparison.judge_model,
            ],
        )

    def comparisons(self, suite: str, source: Source | None = None) -> list[Comparison]:
        """Recorded comparisons, optionally narrowed to judge or human."""
        clause = "" if source is None else " AND source = ?"
        params: list[str] = [suite] if source is None else [suite, source]
        rows = self._connection.execute(
            f"""
            SELECT task_id, model_a, model_b, "order", winner, source,
                   rationale, judge_model
            FROM comparisons WHERE suite = ?{clause}
            ORDER BY task_id, model_a, model_b, "order"
            """,
            params,
        ).fetchall()
        return [
            Comparison(
                task_id=str(row[0]),
                model_a=str(row[1]),
                model_b=str(row[2]),
                order=cast("Order", str(row[3])),
                winner=cast("Verdict", str(row[4])),
                source=cast("Source", str(row[5])),
                rationale=None if row[6] is None else str(row[6]),
                judge_model=None if row[7] is None else str(row[7]),
            )
            for row in rows
        ]

    def compared(self, suite: str, source: Source) -> set[tuple[str, str, str, str]]:
        """The (task_id, model_a, model_b, order) tuples already recorded."""
        rows = self._connection.execute(
            'SELECT task_id, model_a, model_b, "order" FROM comparisons '
            "WHERE suite = ? AND source = ?",
            [suite, source],
        ).fetchall()
        return {(str(a), str(b), str(c), str(d)) for a, b, c, d in rows}

    def record_judge_failure(self, suite: str, item: JudgeFailure) -> None:
        """Keep an unparseable judge response as evidence."""
        self._connection.execute(
            """
            INSERT OR REPLACE INTO judge_failures
                (suite, task_id, model_a, model_b, "order", judge_model, reason, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                suite,
                item.task_id,
                item.model_a,
                item.model_b,
                item.order,
                item.judge_model,
                item.reason,
                item.raw,
            ],
        )

    def judge_failures(self, suite: str) -> list[JudgeFailure]:
        rows = self._connection.execute(
            """
            SELECT task_id, model_a, model_b, "order", judge_model, reason, raw
            FROM judge_failures WHERE suite = ?
            ORDER BY task_id, model_a, model_b, "order"
            """,
            [suite],
        ).fetchall()
        return [
            JudgeFailure(
                task_id=str(row[0]),
                model_a=str(row[1]),
                model_b=str(row[2]),
                order=cast("Order", str(row[3])),
                judge_model=str(row[4]),
                reason=str(row[5]),
                raw=str(row[6]),
            )
            for row in rows
        ]

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
