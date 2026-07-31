"""DuckDB result store — one file, no server.

Rows are keyed by (suite, task_id, model). Task ids are only unique within a
suite, so the suite name is part of the key here even though it is not part of
`Generation`: the suite is the partition, not a property of one answer.

Writes are idempotent. Re-running a suite overwrites the matching row rather
than accumulating duplicates, which is what makes `konkord run` safe to repeat.
"""

from pathlib import Path
from types import TracebackType
from typing import Self

import duckdb

from konkord.models import Generation

_SCHEMA = """
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
"""


class ResultStore:
    """Every stage's view of what has already been produced."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(path))
        self._connection.execute(_SCHEMA)

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
