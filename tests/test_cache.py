"""Cache key stability, and round-tripping responses."""

from pathlib import Path

from konkord.cache import ResponseCache, cache_key
from konkord.providers import CompletionRequest, CompletionResponse

REQUEST = CompletionRequest(model="alpha", prompt="p", context="c", max_tokens=100)
RESPONSE = CompletionResponse(
    text="answer", tokens_in=3, tokens_out=5, cost_usd=0.02, latency_ms=1234
)


class TestKey:
    def test_equal_requests_hash_equal(self) -> None:
        assert cache_key(REQUEST) == cache_key(
            CompletionRequest(model="alpha", prompt="p", context="c", max_tokens=100)
        )

    def test_model_changes_the_key(self) -> None:
        assert cache_key(REQUEST) != cache_key(
            CompletionRequest(model="beta", prompt="p", context="c", max_tokens=100)
        )

    def test_prompt_changes_the_key(self) -> None:
        assert cache_key(REQUEST) != cache_key(
            CompletionRequest(model="alpha", prompt="q", context="c", max_tokens=100)
        )

    def test_context_changes_the_key(self) -> None:
        """Context is part of what the model saw, so it must be part of the key."""
        assert cache_key(REQUEST) != cache_key(
            CompletionRequest(model="alpha", prompt="p", context="other", max_tokens=100)
        )

    def test_token_cap_changes_the_key(self) -> None:
        assert cache_key(REQUEST) != cache_key(
            CompletionRequest(model="alpha", prompt="p", context="c", max_tokens=200)
        )


class TestStorage:
    def test_round_trip_preserves_cost_and_latency(self, tmp_path: Path) -> None:
        """A replayed run must report the original call's numbers, not the cache's."""
        with ResponseCache(tmp_path / "cache") as cache:
            cache.set("k", RESPONSE)
            assert cache.get("k") == RESPONSE

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        with ResponseCache(tmp_path / "cache") as cache:
            assert cache.get("absent") is None

    def test_entry_written_in_an_older_layout_is_a_miss(self, tmp_path: Path) -> None:
        """A stale entry must not crash a long run."""
        with ResponseCache(tmp_path / "cache") as cache:
            cache._cache.set("k", {"text": "only"})
            assert cache.get("k") is None

    def test_survives_reopening(self, tmp_path: Path) -> None:
        path = tmp_path / "cache"
        with ResponseCache(path) as cache:
            cache.set("k", RESPONSE)
        with ResponseCache(path) as reopened:
            assert reopened.get("k") == RESPONSE
