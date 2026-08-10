"""Pairwise LLM-as-judge, with the bias controls that make it worth trusting.

Four things here are deliberate, and each one exists to stop a specific way the
number could come out wrong:

* **Blind.** The judge sees "Answer 1" and "Answer 2" and never a model name.
  `build_prompt` is the only place answers are laid out, so blinding is a
  property of one function rather than a convention spread across the module.
* **Both orderings.** Every pair is judged twice, once each way round, as two
  separate calls. Disagreement between the two is position bias, and a pair that
  flips is recorded as a tie rather than resolved by whichever call came first.
* **Different provider family.** A judge from the same family as a ranked model
  gets a self-preference discount it has no way to correct for, so it is refused
  outright.
* **Strict parsing.** A response without a well-formed verdict token is retried
  once and then recorded as a failure. It is never coerced into a winner.
"""

import asyncio
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from konkord.cache import ResponseCache, cache_key
from konkord.models import (
    Comparison,
    Generation,
    JudgeFailure,
    Order,
    Suite,
    Task,
    Verdict,
)
from konkord.providers import (
    Completer,
    CompletionError,
    CompletionRequest,
    CompletionResponse,
)
from konkord.retrying import RetryPolicy, call_with_retry
from konkord.store import ResultStore

#: Where the verdict must appear, and in what form.
_VERDICT_PATTERN = re.compile(r"^\s*VERDICT:\s*(1|2|TIE)\s*$", re.IGNORECASE | re.MULTILINE)

#: Positional, not identity: "first" is whichever answer was shown first.
Position = Literal["first", "second", "tie"]

JUDGE_SYSTEM = """\
You are grading two candidate answers to the same programming task.

Judge in this order:
1. Correctness. Does the answer actually do what the task asked, including the
   edge cases the task names? A subtly wrong answer loses to a plainly correct
   one, however well written.
2. Idiomatic quality. Given equal correctness, prefer the answer a competent
   reviewer would rather maintain.

Explicitly ignore: answer length, formatting, and how many comments there are.
A longer answer is not a better answer. Do not reward verbosity.

Respond with at most three sentences of rationale, then a final line in exactly
this form and nothing after it:

VERDICT: 1
VERDICT: 2
VERDICT: TIE

Use TIE only when the answers are genuinely of equal merit."""

_STRICT_REMINDER = """\

Your previous response did not end with a valid verdict line. Respond again,
ending with exactly one of `VERDICT: 1`, `VERDICT: 2`, or `VERDICT: TIE`."""

#: Prefixes litellm accepts without an explicit `provider/` segment.
_BARE_MODEL_FAMILIES: tuple[tuple[str, str], ...] = (
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("chatgpt", "openai"),
    ("openai", "openai"),
    ("claude", "anthropic"),
    ("anthropic", "anthropic"),
    ("gemini", "google"),
    ("google", "google"),
    ("command", "cohere"),
    ("cohere", "cohere"),
    ("mistral", "mistral"),
    ("codestral", "mistral"),
    ("llama", "meta"),
    ("meta", "meta"),
    ("deepseek", "deepseek"),
    ("qwen", "alibaba"),
    ("grok", "xai"),
)

#: Prefixes that route to somebody else's model rather than serving their own.
#:
#: The family is whatever they route *to*. `openrouter/anthropic/claude-opus-5`
#: and a direct `anthropic/claude-opus-5` are the same model from the same
#: vendor, and treating the router as a family of its own would let a judge
#: rank its own family through an aggregator. That is a silent hole in the
#: self-preference control rather than a cosmetic naming issue.
_ROUTERS: frozenset[str] = frozenset(
    {"openrouter", "litellm_proxy", "bedrock", "vertex_ai", "azure", "azure_ai"}
)

#: Vendor slugs that differ between routers and litellm's own prefixes.
_FAMILY_ALIASES: dict[str, str] = {
    "mistralai": "mistral",
    "meta-llama": "meta",
    "meta_llama": "meta",
    "x-ai": "xai",
    "qwen": "alibaba",
    "alibaba": "alibaba",
    "google-vertex": "google",
    # litellm reaches Google AI Studio as `gemini/`, routers list it as `google/`.
    "gemini": "google",
    "googleai": "google",
}


class JudgeError(Exception):
    """The judging configuration is unusable."""


def provider_family(model: str) -> str:
    """Best-effort provider family for a litellm model string.

    Routing prefixes are peeled off first, so `openrouter/anthropic/claude-opus-5`,
    `vertex_ai/claude-opus-5` and a bare `claude-opus-5` all resolve to
    `anthropic`. Without that, a judge reached through an aggregator would look
    like a different family from the same model reached directly, and the
    self-preference check would pass something it exists to refuse.

    An unrecognised name falls back to itself, which is the conservative
    direction: it may refuse a judge unnecessarily, but it will not wave one
    through by pretending two vendors are different.
    """
    name = model.strip().lower()

    # Peel routers, bounded so a pathological name cannot loop.
    for _ in range(4):
        head, separator, rest = name.partition("/")
        if not separator or head not in _ROUTERS:
            break
        name = rest

    head, separator, _ = name.partition("/")
    if separator:
        return _FAMILY_ALIASES.get(head, head)
    for prefix, family in _BARE_MODEL_FAMILIES:
        if name.startswith(prefix):
            return family
    return _FAMILY_ALIASES.get(name, name)


def check_judge_independence(judge_model: str, ranked: Iterable[str]) -> None:
    """Refuse a judge drawn from the same family as anything it would rank.

    Raises:
        JudgeError: naming the models that share the judge's family.
    """
    family = provider_family(judge_model)
    clashes = sorted({m for m in ranked if provider_family(m) == family})
    if clashes:
        raise JudgeError(
            f"judge {judge_model!r} shares provider family {family!r} with "
            f"{', '.join(clashes)}; a judge cannot rank its own family without "
            f"a self-preference bias this tool cannot correct for"
        )


@dataclass(frozen=True, slots=True)
class Matchup:
    """One pair on one task, in one presentation order."""

    task: Task
    model_a: str
    model_b: str
    order: Order

    @property
    def first_model(self) -> str:
        return self.model_a if self.order == "ab" else self.model_b

    @property
    def second_model(self) -> str:
        return self.model_b if self.order == "ab" else self.model_a


def build_prompt(matchup: Matchup, answers: dict[str, str], *, strict: bool = False) -> str:
    """Lay out one blind comparison.

    The only place answers meet the prompt, and therefore the only place model
    identity could leak. It does not.
    """
    context = f"\n\nContext for the task:\n{matchup.task.context}" if matchup.task.context else ""
    reminder = _STRICT_REMINDER if strict else ""
    return (
        f"Task:\n{matchup.task.prompt}{context}\n\n"
        f"--- Answer 1 ---\n{answers[matchup.first_model]}\n\n"
        f"--- Answer 2 ---\n{answers[matchup.second_model]}\n"
        f"{reminder}"
    )


def parse_verdict(text: str) -> Position | None:
    """Extract the verdict, or `None` if the response does not carry one.

    Takes the last well-formed verdict line: a judge that reasons out loud may
    mention the format before committing to it.
    """
    matches = _VERDICT_PATTERN.findall(text)
    if not matches:
        return None
    token = matches[-1].upper()
    if token == "1":
        return "first"
    if token == "2":
        return "second"
    return "tie"


def to_comparison(
    matchup: Matchup,
    position: Position,
    *,
    judge_model: str,
    rationale: str,
) -> Comparison:
    """Turn a positional verdict into one about model identity.

    This is where position becomes identity, and it has to account for the
    ordering: in a `ba` matchup, "Answer 1" was `model_b`.
    """
    if position == "tie":
        winner: Verdict = "tie"
    else:
        won = matchup.first_model if position == "first" else matchup.second_model
        winner = "a" if won == matchup.model_a else "b"
    return Comparison(
        task_id=matchup.task.id,
        model_a=matchup.model_a,
        model_b=matchup.model_b,
        order=matchup.order,
        winner=winner,
        source="judge",
        rationale=rationale,
        judge_model=judge_model,
    )


def resolve(comparisons: Sequence[Comparison]) -> Verdict:
    """Collapse both orderings of one pair into a single verdict.

    Agreement stands; disagreement is position bias and resolves to a tie. A
    single ordering is not enough evidence to call a winner, so it also ties.
    """
    winners = {c.winner_model for c in comparisons}
    if len(comparisons) < 2 or len(winners) != 1:
        return "tie"
    only = next(iter(winners))
    if only is None:
        return "tie"
    return "a" if only == comparisons[0].model_a else "b"


def flipped(comparisons: Sequence[Comparison]) -> bool:
    """True when the two orderings disagreed. This is the position-bias diagnostic."""
    if len(comparisons) < 2:
        return False
    return len({c.winner_model for c in comparisons}) != 1


def judgeable(generations: Iterable[Generation]) -> dict[tuple[str, str], str]:
    """Answers worth comparing, keyed by (task_id, model).

    A generation that errored or came back empty has nothing to judge; including
    it would hand every pairing to whichever model did produce output.
    """
    return {
        (g.task_id, g.model): g.output for g in generations if g.error is None and g.output.strip()
    }


def matchups(tasks: Sequence[Task], models: Sequence[str]) -> list[Matchup]:
    """Every (task, unordered pair, ordering), with pairs canonically sorted."""
    ordered = sorted(set(models))
    return [
        Matchup(task=task, model_a=first, model_b=second, order=order)
        for task in tasks
        for index, first in enumerate(ordered)
        for second in ordered[index + 1 :]
        for order in ("ab", "ba")
    ]


def failure(matchup: Matchup, judge_model: str, reason: str, raw: str) -> JudgeFailure:
    return JudgeFailure(
        task_id=matchup.task.id,
        model_a=matchup.model_a,
        model_b=matchup.model_b,
        order=matchup.order,
        judge_model=judge_model,
        reason=reason,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    concurrency: int = 8
    max_attempts: int = 4
    #: Generous on purpose. A reasoning judge spends most of its budget before
    #: emitting any visible text, and a cap that starves it returns an empty
    #: response that looks like a refusal. 1024 lost 72% of verdicts in testing.
    max_tokens: int = 8192
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
class JudgeSummary:
    """What a judging pass did, and how much to trust it."""

    judged: int
    skipped: int
    unjudgeable: int
    parse_failures: int
    call_failures: int
    pairs: int
    flips: int
    cost_usd: float

    @property
    def flip_rate(self) -> float:
        """Share of pairs where the two orderings disagreed.

        The headline reliability diagnostic: a high flip rate means the judge is
        deciding on position rather than content, and the ranking built on it is
        not worth much.
        """
        return self.flips / self.pairs if self.pairs else 0.0


async def judge_suite(
    *,
    suite: Suite,
    models: Sequence[str],
    judge_model: str,
    completer: Completer,
    cache: ResponseCache,
    store: ResultStore,
    config: JudgeConfig | None = None,
) -> JudgeSummary:
    """Judge every model pair on every task, in both orderings.

    Raises:
        JudgeError: if the judge shares a provider family with a ranked model.
    """
    config = config or JudgeConfig()
    check_judge_independence(judge_model, models)

    answers = judgeable(store.generations(suite.name))
    already = store.compared(suite.name, "judge")

    candidates = matchups(suite.tasks, models)
    runnable = [
        m
        for m in candidates
        if (m.task.id, m.model_a) in answers and (m.task.id, m.model_b) in answers
    ]
    unjudgeable = len(candidates) - len(runnable)
    pending = [m for m in runnable if (m.task.id, m.model_a, m.model_b, m.order) not in already]

    semaphore = asyncio.Semaphore(config.concurrency)
    outcomes = await asyncio.gather(
        *(
            _judge_one(
                matchup=m,
                answers=answers,
                judge_model=judge_model,
                completer=completer,
                cache=cache,
                store=store,
                suite_name=suite.name,
                config=config,
                semaphore=semaphore,
            )
            for m in pending
        )
    )

    stored = store.comparisons(suite.name, "judge")
    by_pair: dict[tuple[str, str, str], list[Comparison]] = {}
    for comparison in stored:
        by_pair.setdefault(comparison.pair_key, []).append(comparison)

    return JudgeSummary(
        judged=sum(1 for kind, _ in outcomes if kind == "judged"),
        skipped=len(runnable) - len(pending),
        unjudgeable=unjudgeable,
        parse_failures=sum(1 for kind, _ in outcomes if kind == "unparseable"),
        call_failures=sum(1 for kind, _ in outcomes if kind == "call-failed"),
        pairs=len(by_pair),
        flips=sum(1 for group in by_pair.values() if flipped(group)),
        cost_usd=sum(cost for _, cost in outcomes),
    )


async def _judge_one(
    *,
    matchup: Matchup,
    answers: dict[tuple[str, str], str],
    judge_model: str,
    completer: Completer,
    cache: ResponseCache,
    store: ResultStore,
    suite_name: str,
    config: JudgeConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[str, float]:
    text = {
        matchup.model_a: answers[(matchup.task.id, matchup.model_a)],
        matchup.model_b: answers[(matchup.task.id, matchup.model_b)],
    }
    cost = 0.0
    raw = ""

    # One retry on an unparseable verdict, with an explicit reminder. The
    # reminder changes the prompt, so it is a distinct cache entry rather than a
    # second read of the response that already failed to parse.
    for strict in (False, True):
        request = CompletionRequest(
            model=judge_model,
            prompt=build_prompt(matchup, text, strict=strict),
            context=JUDGE_SYSTEM,
            max_tokens=config.max_tokens,
        )
        try:
            response, spent = await _ask(request, completer, cache, config, semaphore)
        except CompletionError as exc:
            store.record_judge_failure(
                suite_name, failure(matchup, judge_model, f"call failed: {exc}", raw)
            )
            return "call-failed", cost
        cost += spent
        raw = response.text

        truncated = response.truncated
        position = parse_verdict(raw)
        if position is not None:
            store.record_comparison(
                suite_name,
                to_comparison(
                    matchup,
                    position,
                    judge_model=judge_model,
                    rationale=raw.strip(),
                ),
            )
            return "judged", cost

    reason = (
        "response hit the token cap before a verdict; raise --max-tokens"
        if truncated
        else "no verdict token in response"
    )
    store.record_judge_failure(suite_name, failure(matchup, judge_model, reason, raw))
    return "unparseable", cost


async def _ask(
    request: CompletionRequest,
    completer: Completer,
    cache: ResponseCache,
    config: JudgeConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[CompletionResponse, float]:
    """Cache-or-call. Returns the response and what this attempt actually cost."""
    key = cache_key(request)
    cached = cache.get(key)
    if cached is not None:
        return cached, 0.0
    async with semaphore:
        response = await call_with_retry(completer, request, config.retry_policy)
    cache.set(key, response)
    return response, response.cost_usd
