"""One retry policy, shared by every stage that calls a model.

Retries are for `TransientError` only. A `PermanentError` is returned to the
caller on the first attempt so it can be recorded rather than hammered.
"""

from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from konkord.providers import (
    Completer,
    CompletionRequest,
    CompletionResponse,
    TransientError,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 4
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 30.0


async def call_with_retry(
    completer: Completer,
    request: CompletionRequest,
    policy: RetryPolicy,
) -> CompletionResponse:
    """Call the model, retrying only what is worth retrying."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(policy.max_attempts),
        wait=wait_exponential_jitter(initial=policy.initial_backoff_s, max=policy.max_backoff_s),
        retry=retry_if_exception_type(TransientError),
        reraise=True,
    ):
        with attempt:
            return await completer.complete(request)
    raise AssertionError("tenacity always either returns or reraises")  # pragma: no cover
