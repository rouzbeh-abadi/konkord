"""Runner behaviour, with a fake completer — these tests never touch a network.

The properties under test are the ones that make `konkord run` safe to repeat:
resumability, bounded concurrency, retry only where retrying helps, and failures
recorded rather than raised.
"""

import asyncio
from pathlib import Path

import pytest

from konkord.cache import ResponseCache
from konkord.models import Suite, Task
from konkord.providers import (
    CompletionRequest,
    CompletionResponse,
    PermanentError,
    TransientError,
)
from konkord.runner import RunConfig, RunSummary, run_suite
from konkord.store import ResultStore

# No real backoff: these tests assert on retry counts, not on wall-clock waits.
FAST = RunConfig(concurrency=4, max_attempts=3, initial_backoff_s=0.0, max_backoff_s=0.0)

SUITE = Suite(
    name="demo",
    tasks=(
        Task(id="t1", prompt="first", context="ctx"),
        Task(id="t2", prompt="second", context="ctx"),
    ),
)
MODELS = ("alpha", "beta")


class FakeCompleter:
    """Records calls, optionally raising a scripted sequence of failures."""

    def __init__(
        self,
        failures: dict[str, list[Exception]] | None = None,
        cost_usd: float = 0.01,
        cost_known: bool = True,
    ) -> None:
        self.requests: list[CompletionRequest] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._failures = {model: list(queue) for model, queue in (failures or {}).items()}
        self._cost_usd = cost_usd
        self._cost_known = cost_known

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(0.005)  # long enough for overlap to be observable
            self.requests.append(request)
            queue = self._failures.get(request.model)
            if queue:
                raise queue.pop(0)
            return CompletionResponse(
                text=f"{request.model} answers {request.prompt}",
                tokens_in=7,
                tokens_out=11,
                cost_usd=self._cost_usd,
                latency_ms=42,
                cost_known=self._cost_known,
            )
        finally:
            self._in_flight -= 1

    def calls_for(self, model: str) -> int:
        return sum(1 for request in self.requests if request.model == model)


async def execute(
    tmp_path: Path,
    completer: FakeCompleter,
    *,
    config: RunConfig = FAST,
    db: str = "r.duckdb",
    cache_dir: str = "cache",
    models: tuple[str, ...] = MODELS,
) -> RunSummary:
    with (
        ResponseCache(tmp_path / cache_dir) as cache,
        ResultStore(tmp_path / db) as store,
    ):
        return await run_suite(
            suite=SUITE,
            models=models,
            completer=completer,
            cache=cache,
            store=store,
            config=config,
        )


class TestHappyPath:
    async def test_generates_every_pair(self, tmp_path: Path) -> None:
        completer = FakeCompleter()
        summary = await execute(tmp_path, completer)
        assert summary.generated == 4
        assert summary.attempted == 4
        assert len(completer.requests) == 4

    async def test_records_every_generation(self, tmp_path: Path) -> None:
        await execute(tmp_path, FakeCompleter())
        with ResultStore(tmp_path / "r.duckdb") as store:
            rows = store.generations("demo")
        assert {(g.task_id, g.model) for g in rows} == {
            ("t1", "alpha"),
            ("t1", "beta"),
            ("t2", "alpha"),
            ("t2", "beta"),
        }

    async def test_passes_task_context_to_the_model(self, tmp_path: Path) -> None:
        completer = FakeCompleter()
        await execute(tmp_path, completer)
        assert all(request.context == "ctx" for request in completer.requests)

    async def test_cost_is_summed(self, tmp_path: Path) -> None:
        summary = await execute(tmp_path, FakeCompleter(cost_usd=0.25))
        assert summary.cost_usd == pytest.approx(1.0)

    async def test_unpriced_models_are_surfaced(self, tmp_path: Path) -> None:
        """A model litellm cannot price must not silently report as free."""
        summary = await execute(tmp_path, FakeCompleter(cost_known=False))
        assert summary.models_without_pricing == ("alpha", "beta")


class TestResumability:
    async def test_second_run_skips_pairs_already_stored(self, tmp_path: Path) -> None:
        await execute(tmp_path, FakeCompleter())
        second = FakeCompleter()
        summary = await execute(tmp_path, second)
        assert summary.skipped == 4
        assert summary.attempted == 0
        assert second.requests == []

    async def test_cache_serves_a_fresh_store(self, tmp_path: Path) -> None:
        """Losing the DuckDB file must not mean paying for the calls again."""
        await execute(tmp_path, FakeCompleter())
        second = FakeCompleter()
        summary = await execute(tmp_path, second, db="fresh.duckdb")
        assert summary.cached == 4
        assert second.requests == []

    async def test_cached_replay_keeps_the_original_latency(self, tmp_path: Path) -> None:
        await execute(tmp_path, FakeCompleter())
        await execute(tmp_path, FakeCompleter(), db="fresh.duckdb")
        with ResultStore(tmp_path / "fresh.duckdb") as store:
            assert {g.latency_ms for g in store.generations("demo")} == {42}

    async def test_adding_a_model_only_generates_the_new_pairs(self, tmp_path: Path) -> None:
        await execute(tmp_path, FakeCompleter(), models=("alpha",))
        second = FakeCompleter()
        summary = await execute(tmp_path, second, models=("alpha", "beta"))
        assert summary.skipped == 2
        assert second.calls_for("beta") == 2
        assert second.calls_for("alpha") == 0


class TestRetry:
    async def test_transient_failure_is_retried_then_succeeds(self, tmp_path: Path) -> None:
        completer = FakeCompleter({"alpha": [TransientError("rate limited")]})
        summary = await execute(tmp_path, completer)
        assert summary.generated == 4
        assert summary.failed == 0
        # One scripted failure, claimed by whichever alpha task got there first:
        # two base calls plus the one retry it forced.
        assert completer.calls_for("alpha") == 3

    async def test_permanent_failure_is_not_retried(self, tmp_path: Path) -> None:
        completer = FakeCompleter({"alpha": [PermanentError("bad key")] * 4})
        summary = await execute(tmp_path, completer)
        assert summary.failed == 2
        assert completer.calls_for("alpha") == 2  # one attempt per task, no retries

    async def test_exhausted_retries_are_recorded_not_raised(self, tmp_path: Path) -> None:
        completer = FakeCompleter({"alpha": [TransientError("down")] * 20})
        summary = await execute(tmp_path, completer)
        assert summary.failed == 2
        assert completer.calls_for("alpha") == 2 * FAST.max_attempts

    async def test_a_failing_model_does_not_stop_the_others(self, tmp_path: Path) -> None:
        completer = FakeCompleter({"alpha": [PermanentError("bad key")] * 4})
        summary = await execute(tmp_path, completer)
        assert summary.generated == 2  # beta finished normally

    async def test_failures_are_stored_with_the_reason(self, tmp_path: Path) -> None:
        await execute(tmp_path, FakeCompleter({"alpha": [PermanentError("bad key")] * 4}))
        with ResultStore(tmp_path / "r.duckdb") as store:
            failures = [g for g in store.generations("demo") if g.error is not None]
        assert len(failures) == 2
        assert all("bad key" in str(g.error) for g in failures)
        assert all(g.output == "" for g in failures)

    async def test_failures_are_not_cached(self, tmp_path: Path) -> None:
        """A failure must be retryable after the cause is fixed."""
        await execute(tmp_path, FakeCompleter({"alpha": [PermanentError("bad key")] * 4}))
        recovered = FakeCompleter()
        summary = await execute(tmp_path, recovered, db="fresh.duckdb")
        assert recovered.calls_for("alpha") == 2
        assert summary.generated == 2


class TestConcurrency:
    async def test_calls_are_bounded_by_the_semaphore(self, tmp_path: Path) -> None:
        completer = FakeCompleter()
        await execute(
            tmp_path,
            completer,
            config=RunConfig(concurrency=2, initial_backoff_s=0.0, max_backoff_s=0.0),
        )
        assert completer.max_in_flight <= 2

    async def test_concurrency_is_actually_used(self, tmp_path: Path) -> None:
        completer = FakeCompleter()
        await execute(tmp_path, completer)
        assert completer.max_in_flight > 1
