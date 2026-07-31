"""Disk cache for model responses, so a run is cheap to repeat.

The key is a hash of everything that determines the output — model, prompt,
context, token cap — plus a version tag. Bump `CACHE_VERSION` whenever the
meaning of a cached entry changes; that retires every old entry without anyone
having to remember to delete a directory.

Cached entries carry the original latency and cost, so replaying a run reports
what the calls actually took and cost rather than what the cache lookup did.
"""

import hashlib
import json
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from diskcache import Cache

from konkord.providers import CompletionRequest, CompletionResponse

#: Part of every cache key. Bump on any change to what an entry means.
CACHE_VERSION = 1


def cache_key(request: CompletionRequest) -> str:
    """Stable hash of a request. Equal requests hash equal across processes."""
    payload = {
        "version": CACHE_VERSION,
        "model": request.model,
        "prompt": request.prompt,
        "context": request.context,
        "max_tokens": request.max_tokens,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """A `diskcache.Cache` that only ever holds `CompletionResponse` values."""

    def __init__(self, path: Path) -> None:
        self._cache = Cache(str(path))

    def get(self, key: str) -> CompletionResponse | None:
        stored: Any = self._cache.get(key)
        if not isinstance(stored, dict):
            return None
        try:
            return CompletionResponse(**stored)
        except TypeError:
            # An entry written by an older layout that shares this version tag.
            # Treat it as a miss rather than crashing a long run.
            return None

    def set(self, key: str, response: CompletionResponse) -> None:
        self._cache.set(
            key,
            {
                "text": response.text,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
                "cost_known": response.cost_known,
            },
        )

    def close(self) -> None:
        self._cache.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
