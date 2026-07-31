"""Persistence: round-tripping, idempotent writes, and suite isolation."""

from pathlib import Path

from konkord.models import Generation
from konkord.store import ResultStore


def generation(task_id: str = "t1", model: str = "alpha", **overrides: object) -> Generation:
    fields: dict[str, object] = {
        "task_id": task_id,
        "model": model,
        "output": "answer",
        "tokens_in": 10,
        "tokens_out": 20,
        "cost_usd": 0.5,
        "latency_ms": 300,
    }
    fields.update(overrides)
    return Generation(**fields)  # type: ignore[arg-type]


class TestRoundTrip:
    def test_record_then_read(self, tmp_path: Path) -> None:
        with ResultStore(tmp_path / "r.duckdb") as store:
            store.record("suite", generation())
            assert store.generations("suite") == [generation()]

    def test_error_survives_the_round_trip(self, tmp_path: Path) -> None:
        with ResultStore(tmp_path / "r.duckdb") as store:
            store.record("suite", generation(error="boom", output=""))
            assert store.generations("suite")[0].error == "boom"

    def test_results_are_ordered(self, tmp_path: Path) -> None:
        with ResultStore(tmp_path / "r.duckdb") as store:
            store.record("suite", generation("t2", "beta"))
            store.record("suite", generation("t1", "alpha"))
            assert [(g.task_id, g.model) for g in store.generations("suite")] == [
                ("t1", "alpha"),
                ("t2", "beta"),
            ]

    def test_survives_reopening(self, tmp_path: Path) -> None:
        path = tmp_path / "r.duckdb"
        with ResultStore(path) as store:
            store.record("suite", generation())
        with ResultStore(path) as reopened:
            assert len(reopened.generations("suite")) == 1


class TestIdempotence:
    def test_recording_twice_replaces_rather_than_duplicates(self, tmp_path: Path) -> None:
        """`konkord run` is safe to repeat; repeating must not double the rows."""
        with ResultStore(tmp_path / "r.duckdb") as store:
            store.record("suite", generation(output="first"))
            store.record("suite", generation(output="second"))
            rows = store.generations("suite")
            assert len(rows) == 1
            assert rows[0].output == "second"


class TestScoping:
    def test_suites_do_not_collide(self, tmp_path: Path) -> None:
        """Task ids are unique per suite, so the suite has to be part of the key."""
        with ResultStore(tmp_path / "r.duckdb") as store:
            store.record("one", generation(output="from one"))
            store.record("two", generation(output="from two"))
            assert store.generations("one")[0].output == "from one"
            assert store.generations("two")[0].output == "from two"

    def test_completed_pairs(self, tmp_path: Path) -> None:
        with ResultStore(tmp_path / "r.duckdb") as store:
            store.record("suite", generation("t1", "alpha"))
            store.record("suite", generation("t2", "beta"))
            assert store.completed("suite") == {("t1", "alpha"), ("t2", "beta")}
            assert store.completed("other") == set()

    def test_recorded_failures_count_as_completed(self, tmp_path: Path) -> None:
        """A permanent failure is a result; re-running must not silently retry it."""
        with ResultStore(tmp_path / "r.duckdb") as store:
            store.record("suite", generation(error="denied", output=""))
            assert store.completed("suite") == {("t1", "alpha")}
