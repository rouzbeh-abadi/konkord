"""The real `Completer`, backed by litellm.

Kept apart from `providers` so the protocol stays importable without paying for
litellm's import, and so nothing in the test suite can reach a network by
accident.
"""

import time
from typing import Any

import litellm

from konkord.providers import (
    CompletionRequest,
    CompletionResponse,
    PermanentError,
    TransientError,
)

#: Retrying these has a real chance of succeeding.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.Timeout,
)

#: Retrying these just spends money and time on the same failure.
PERMANENT_ERRORS: tuple[type[BaseException], ...] = (
    litellm.AuthenticationError,
    litellm.BadRequestError,
    litellm.ContentPolicyViolationError,
    litellm.ContextWindowExceededError,
    litellm.NotFoundError,
    litellm.PermissionDeniedError,
    litellm.UnprocessableEntityError,
)


class LiteLLMCompleter:
    """Calls a model through litellm, one provider-neutral interface."""

    def __init__(self, timeout_s: float = 120.0) -> None:
        self._timeout_s = timeout_s

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        messages: list[dict[str, str]] = []
        if request.context:
            messages.append({"role": "system", "content": request.context})
        messages.append({"role": "user", "content": request.prompt})

        started = time.perf_counter()
        try:
            raw = await litellm.acompletion(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                timeout=self._timeout_s,
            )
        except TRANSIENT_ERRORS as exc:
            raise TransientError(f"{type(exc).__name__}: {exc}") from exc
        except PERMANENT_ERRORS as exc:
            raise PermanentError(f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:  # unrecognised failures are recorded, never retried
            raise PermanentError(f"unclassified {type(exc).__name__}: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        cost, cost_known = _price(raw)
        return CompletionResponse(
            text=_text(raw),
            tokens_in=_usage(raw, "prompt_tokens"),
            tokens_out=_usage(raw, "completion_tokens"),
            cost_usd=cost,
            latency_ms=latency_ms,
            cost_known=cost_known,
            truncated=_truncated(raw),
        )


def _text(raw: Any) -> str:
    """Pull the answer out, tolerating a provider that returns no content."""
    choices = getattr(raw, "choices", None)
    if not choices:
        return ""
    content = getattr(choices[0].message, "content", None)
    return content if isinstance(content, str) else ""


def _truncated(raw: Any) -> bool:
    """Whether the provider stopped at the token cap rather than finishing."""
    choices = getattr(raw, "choices", None)
    if not choices:
        return False
    return getattr(choices[0], "finish_reason", None) == "length"


def _usage(raw: Any, field: str) -> int:
    usage = getattr(raw, "usage", None)
    value = getattr(usage, field, None) if usage is not None else None
    return value if isinstance(value, int) and value >= 0 else 0


def _price(raw: Any) -> tuple[float, bool]:
    """Cost in USD, and whether it is actually known.

    Two sources, in order of trustworthiness:

    1. `_hidden_params["response_cost"]`, which litellm attaches during the call
       using the provider's own figure where one is returned. That is exact.
    2. `completion_cost()`, a lookup against litellm's price table. An
       approximation, and not populated for every model string: routed names
       like `openrouter/openai/gpt-4o-mini` are missing from it even when the
       call itself priced fine.

    Preferring the attached value is what stops a whole leaderboard of routed
    models reporting as free.
    """
    hidden = getattr(raw, "_hidden_params", None)
    if isinstance(hidden, dict):
        attached = hidden.get("response_cost")
        if isinstance(attached, (int, float)) and not isinstance(attached, bool) and attached >= 0:
            return float(attached), True

    try:
        cost = litellm.completion_cost(completion_response=raw)
    except Exception:  # noqa: BLE001 - litellm raises assorted types for unknown models
        return 0.0, False
    if not isinstance(cost, (int, float)) or cost < 0:
        return 0.0, False
    return float(cost), True
