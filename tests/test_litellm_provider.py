"""Provider adapter tests.

Every other test in the suite substitutes a fake completer at the protocol
boundary, which is what keeps the suite offline and fast. That leaves this
module as the only place the litellm adapter itself is exercised, so it covers
the two jobs nothing else can: turning an untyped provider response into a typed
one, and putting each kind of failure in the right bucket.

The provider is stubbed rather than called. These tests still never touch a
network.
"""

from typing import Any

import litellm
import pytest

from konkord.litellm_provider import (
    PERMANENT_ERRORS,
    TRANSIENT_ERRORS,
    LiteLLMCompleter,
    _price,
    _text,
    _usage,
)
from konkord.providers import CompletionRequest, PermanentError, TransientError


class Message:
    def __init__(self, content: object) -> None:
        self.content = content


class Choice:
    def __init__(self, content: object) -> None:
        self.message = Message(content)


class Usage:
    def __init__(self, prompt_tokens: object, completion_tokens: object) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class Response:
    """The shape litellm returns, with only the fields the adapter reads."""

    def __init__(
        self,
        content: object = "an answer",
        prompt_tokens: object = 11,
        completion_tokens: object = 7,
    ) -> None:
        self.choices = [Choice(content)]
        self.usage = Usage(prompt_tokens, completion_tokens)


class Bare:
    """A response missing everything the adapter hopes for."""


def error(kind: type[BaseException]) -> BaseException:
    return kind(message="stubbed", llm_provider="stub", model="stub-1")  # type: ignore[call-arg]


class TestText:
    def test_extracts_the_message_content(self) -> None:
        assert _text(Response(content="hello")) == "hello"

    def test_missing_choices_is_empty(self) -> None:
        assert _text(Bare()) == ""

    def test_empty_choices_is_empty(self) -> None:
        response = Response()
        response.choices = []
        assert _text(response) == ""

    def test_null_content_is_empty(self) -> None:
        """Providers return a null content on a filtered or empty completion."""
        assert _text(Response(content=None)) == ""

    def test_non_string_content_is_empty(self) -> None:
        assert _text(Response(content=[{"type": "text"}])) == ""


class TestUsage:
    def test_reads_both_counts(self) -> None:
        response = Response(prompt_tokens=11, completion_tokens=7)
        assert _usage(response, "prompt_tokens") == 11
        assert _usage(response, "completion_tokens") == 7

    def test_missing_usage_is_zero(self) -> None:
        assert _usage(Bare(), "prompt_tokens") == 0

    def test_null_count_is_zero(self) -> None:
        assert _usage(Response(prompt_tokens=None), "prompt_tokens") == 0

    def test_non_integer_count_is_zero(self) -> None:
        assert _usage(Response(prompt_tokens="11"), "prompt_tokens") == 0

    def test_negative_count_is_zero(self) -> None:
        """A negative token count is nonsense; a zero at least does not corrupt sums."""
        assert _usage(Response(prompt_tokens=-5), "prompt_tokens") == 0


class TestPrice:
    def test_known_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0042)
        assert _price(Response()) == (0.0042, True)

    def test_integer_cost_becomes_a_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0)
        cost, known = _price(Response())
        assert isinstance(cost, float)
        assert known

    def test_unpriceable_model_is_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The case that would otherwise report a paid model as free."""

        def explode(**_: object) -> float:
            raise ValueError("model not in the pricing map")

        monkeypatch.setattr(litellm, "completion_cost", explode)
        assert _price(Response()) == (0.0, False)

    def test_negative_cost_is_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "completion_cost", lambda **_: -1.0)
        assert _price(Response()) == (0.0, False)

    def test_non_numeric_cost_is_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "completion_cost", lambda **_: None)
        assert _price(Response()) == (0.0, False)


class TestClassification:
    def test_rate_limits_and_timeouts_are_transient(self) -> None:
        assert litellm.RateLimitError in TRANSIENT_ERRORS
        assert litellm.Timeout in TRANSIENT_ERRORS

    def test_credentials_and_bad_requests_are_permanent(self) -> None:
        assert litellm.AuthenticationError in PERMANENT_ERRORS
        assert litellm.BadRequestError in PERMANENT_ERRORS

    def test_the_buckets_do_not_overlap(self) -> None:
        assert not set(TRANSIENT_ERRORS) & set(PERMANENT_ERRORS)


class TestComplete:
    async def test_returns_a_typed_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def stub(**_: object) -> Response:
            return Response(content="the answer", prompt_tokens=11, completion_tokens=7)

        monkeypatch.setattr(litellm, "acompletion", stub)
        monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.5)

        result = await LiteLLMCompleter().complete(
            CompletionRequest(model="stub-1", prompt="ask", context="be brief")
        )
        assert result.text == "the answer"
        assert (result.tokens_in, result.tokens_out) == (11, 7)
        assert result.cost_usd == 0.5
        assert result.cost_known
        assert result.latency_ms >= 0

    async def test_context_becomes_a_system_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        async def stub(**kwargs: Any) -> Response:
            seen.update(kwargs)
            return Response()

        monkeypatch.setattr(litellm, "acompletion", stub)
        monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0)

        await LiteLLMCompleter().complete(
            CompletionRequest(model="stub-1", prompt="ask", context="be brief", max_tokens=64)
        )
        assert seen["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "ask"},
        ]
        assert seen["model"] == "stub-1"
        assert seen["max_tokens"] == 64

    async def test_no_context_sends_no_system_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        async def stub(**kwargs: Any) -> Response:
            seen.update(kwargs)
            return Response()

        monkeypatch.setattr(litellm, "acompletion", stub)
        monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0)

        await LiteLLMCompleter().complete(CompletionRequest(model="stub-1", prompt="ask"))
        assert seen["messages"] == [{"role": "user", "content": "ask"}]

    async def test_transient_provider_error_is_retryable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def stub(**_: object) -> Response:
            raise error(litellm.RateLimitError)

        monkeypatch.setattr(litellm, "acompletion", stub)
        with pytest.raises(TransientError, match="RateLimitError"):
            await LiteLLMCompleter().complete(CompletionRequest(model="stub-1", prompt="ask"))

    async def test_permanent_provider_error_is_not_retryable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def stub(**_: object) -> Response:
            raise error(litellm.AuthenticationError)

        monkeypatch.setattr(litellm, "acompletion", stub)
        with pytest.raises(PermanentError, match="AuthenticationError"):
            await LiteLLMCompleter().complete(CompletionRequest(model="stub-1", prompt="ask"))

    async def test_unrecognised_error_is_permanent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unknown failure is recorded, not hammered with retries."""

        async def stub(**_: object) -> Response:
            raise RuntimeError("something new")

        monkeypatch.setattr(litellm, "acompletion", stub)
        with pytest.raises(PermanentError, match="unclassified RuntimeError"):
            await LiteLLMCompleter().complete(CompletionRequest(model="stub-1", prompt="ask"))
