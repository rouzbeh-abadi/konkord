"""Generating one output per (task x model), concurrently and resumably.

Three properties hold this together:

* **Resumable.** Every generation is written to the store the moment it lands,
  so an interrupted run resumes where it stopped instead of starting over.
  A pair already in the store is not regenerated.
* **Bounded.** At most `concurrency` calls are in flight. A cache hit does not
  consume a slot, since it costs no quota and should not queue behind one.
* **Non-fatal.** A model that fails permanently produces a `Generation` with
  `error` set. The run continues; nothing is raised past this module.
"""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from konkord.cache import ResponseCache, cache_key
from konkord.models import Generation, Suite, Task
from konkord.providers import (
    Completer,
    CompletionError,
    CompletionRequest,
    CompletionResponse,
)
from konkord.retrying import RetryPolicy, call_with_retry
from konkord.store import ResultStore


class Outcome(StrEnum):
    """How one (task, model) pair was satisfied."""

    GENERATED = "generated"
    CACHED = "cached"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RunConfig:
    concurrency: int = 8
    max_attempts: int = 4
    max_tokens: int = 4096
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 30.0

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=self.max_attempts,
            initial_backoff_s=self.initial_backoff_s,
            max_backoff_s=self.max_backoff_s,
        )


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What a run did, in the terms a person asks about afterwards."""

    generated: int
    cached: int
    failed: int
    skipped: int
    cost_usd: float
    models_without_pricing: tuple[str, ...]

    @property
    def attempted(self) -> int:
        return self.generated + self.cached + self.failed


@dataclass(frozen=True, slots=True)
class _Unit:
    generation: Generation
    outcome: Outcome
    cost_known: bool


ResultCallback = Callable[[Generation, Outcome], None]


async def run_suite(
    *,
    suite: Suite,
    models: Sequence[str],
    completer: Completer,
    cache: ResponseCache,
    store: ResultStore,
    config: RunConfig | None = None,
    on_result: ResultCallback | None = None,
) -> RunSummary:
    """Generate every missing (task x model) output for a suite."""
    config = config or RunConfig()
    already = store.completed(suite.name)
    pending = [
        (task, model) for task in suite.tasks for model in models if (task.id, model) not in already
    ]
    skipped = len(suite.tasks) * len(models) - len(pending)

    semaphore = asyncio.Semaphore(config.concurrency)
    units = await asyncio.gather(
        *(
            _generate(
                suite=suite,
                task=task,
                model=model,
                completer=completer,
                cache=cache,
                store=store,
                config=config,
                semaphore=semaphore,
                on_result=on_result,
            )
            for task, model in pending
        )
    )
    return _summarise(units, skipped=skipped)


async def _generate(
    *,
    suite: Suite,
    task: Task,
    model: str,
    completer: Completer,
    cache: ResponseCache,
    store: ResultStore,
    config: RunConfig,
    semaphore: asyncio.Semaphore,
    on_result: ResultCallback | None,
) -> _Unit:
    request = CompletionRequest(
        model=model,
        prompt=task.prompt,
        context=task.context,
        max_tokens=config.max_tokens,
    )
    key = cache_key(request)

    # Checked before the semaphore: a cache hit spends no quota and should not
    # wait behind a call that does.
    cached = cache.get(key)
    if cached is not None:
        return _finish(suite, task, model, cached, Outcome.CACHED, store, on_result)

    async with semaphore:
        try:
            response = await call_with_retry(completer, request, config.retry_policy)
        except CompletionError as exc:
            failure = Generation(
                task_id=task.id,
                model=model,
                output="",
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                latency_ms=0,
                error=str(exc),
            )
            store.record(suite.name, failure)
            if on_result is not None:
                on_result(failure, Outcome.FAILED)
            return _Unit(failure, Outcome.FAILED, cost_known=True)

    cache.set(key, response)
    return _finish(suite, task, model, response, Outcome.GENERATED, store, on_result)


def _finish(
    suite: Suite,
    task: Task,
    model: str,
    response: CompletionResponse,
    outcome: Outcome,
    store: ResultStore,
    on_result: ResultCallback | None,
) -> _Unit:
    generation = Generation(
        task_id=task.id,
        model=model,
        output=response.text,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
    )
    store.record(suite.name, generation)
    if on_result is not None:
        on_result(generation, outcome)
    return _Unit(generation, outcome, cost_known=response.cost_known)


def _summarise(units: Sequence[_Unit], *, skipped: int) -> RunSummary:
    unpriced = sorted({u.generation.model for u in units if not u.cost_known})
    return RunSummary(
        generated=sum(1 for u in units if u.outcome is Outcome.GENERATED),
        cached=sum(1 for u in units if u.outcome is Outcome.CACHED),
        failed=sum(1 for u in units if u.outcome is Outcome.FAILED),
        skipped=skipped,
        cost_usd=sum(u.generation.cost_usd for u in units),
        models_without_pricing=tuple(unpriced),
    )
