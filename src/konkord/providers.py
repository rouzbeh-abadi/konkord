"""The seam between the harness and whatever actually calls a model.

Deliberately free of heavy imports: everything here is data and protocol, so
tests can supply their own `Completer` and never touch the network. The real
implementation lives in `litellm_provider`.

Failures are classified at this boundary, not at the call site. `TransientError`
is worth retrying; `PermanentError` is not and gets recorded on the generation.
Anything a provider raises that we do not recognise is treated as permanent — an
unknown failure is not something to hammer with retries.
"""

from dataclasses import dataclass
from typing import Protocol


class CompletionError(Exception):
    """A model call failed."""


class TransientError(CompletionError):
    """Failed for a reason that may not recur: rate limit, timeout, 5xx."""


class PermanentError(CompletionError):
    """Failed for a reason that will recur: bad credentials, bad request."""


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """Everything that determines an output, and therefore the cache key."""

    model: str
    prompt: str
    context: str | None = None
    max_tokens: int = 4096


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    """One model's answer, with what it cost to obtain.

    `latency_ms` is measured by the completer and carried through the cache, so a
    replayed run reports the latency of the call that actually happened rather
    than the microseconds a cache hit took.

    `cost_known` is false when the provider could not price the call. The cost is
    then reported as 0.0, which would silently understate a leaderboard — the
    runner surfaces the affected models instead of letting that pass quietly.
    """

    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    cost_known: bool = True


class Completer(Protocol):
    """Anything that can turn a request into a response."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Call the model.

        Raises:
            TransientError: worth retrying.
            PermanentError: not worth retrying.
        """
        ...
