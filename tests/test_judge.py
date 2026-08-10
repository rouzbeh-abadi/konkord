"""Judge tests.

The bias controls are the point of this module, so most of these assert a
property the leaderboard depends on rather than an implementation detail:
blinding, position handling, dual-ordering resolution, and refusal to coerce an
unparseable verdict into a winner.
"""

from pathlib import Path

import pytest

from konkord.cache import ResponseCache
from konkord.judge import (
    JUDGE_SYSTEM,
    JudgeConfig,
    JudgeError,
    JudgeSummary,
    Matchup,
    build_prompt,
    check_judge_independence,
    flipped,
    judge_suite,
    judgeable,
    matchups,
    parse_verdict,
    provider_family,
    resolve,
    to_comparison,
)
from konkord.models import Comparison, Generation, Suite, Task
from konkord.providers import CompletionRequest, CompletionResponse, PermanentError
from konkord.store import ResultStore

FAST = JudgeConfig(concurrency=4, max_attempts=2, initial_backoff_s=0.0, max_backoff_s=0.0)

TASK = Task(id="t1", prompt="Write a thing.", context="Python 3.12.")
SUITE = Suite(name="demo", tasks=(TASK, Task(id="t2", prompt="Write another.")))
ANSWERS = {"alpha": "SENTINEL_ONE_BODY", "beta": "SENTINEL_TWO_BODY"}
SENTINELS = ANSWERS


def matchup(order: str = "ab") -> Matchup:
    return Matchup(task=TASK, model_a="alpha", model_b="beta", order=order)  # type: ignore[arg-type]


def comparison(order: str, winner: str) -> Comparison:
    return Comparison(
        task_id="t1",
        model_a="alpha",
        model_b="beta",
        order=order,  # type: ignore[arg-type]
        winner=winner,  # type: ignore[arg-type]
        source="judge",
        judge_model="referee",
    )


class TestProviderFamily:
    def test_explicit_prefix_wins(self) -> None:
        assert provider_family("anthropic/claude-opus-5") == "anthropic"

    def test_bare_names_are_recognised(self) -> None:
        assert provider_family("claude-opus-5") == "anthropic"
        assert provider_family("gpt-5") == "openai"
        assert provider_family("gemini-2.5-pro") == "google"

    def test_prefixed_and_bare_agree(self) -> None:
        assert provider_family("claude-opus-5") == provider_family("anthropic/claude-opus-5")

    def test_unknown_name_falls_back_to_itself(self) -> None:
        """Better to look like its own family than to be wrongly grouped."""
        assert provider_family("some-new-model") == "some-new-model"


class TestRoutedModels:
    """A router is not a family. Treating it as one is a hole, not a cosmetic bug.

    Reaching a model through OpenRouter must resolve to the vendor behind it,
    or a judge could rank its own family simply by being routed differently.
    """

    def test_openrouter_resolves_to_the_vendor_behind_it(self) -> None:
        assert provider_family("openrouter/anthropic/claude-opus-5") == "anthropic"
        assert provider_family("openrouter/openai/gpt-5") == "openai"
        assert provider_family("openrouter/google/gemini-2.5-pro") == "google"

    def test_routed_and_direct_are_the_same_family(self) -> None:
        assert provider_family("openrouter/anthropic/claude-opus-5") == provider_family(
            "anthropic/claude-opus-5"
        )
        assert provider_family("openrouter/openai/gpt-5") == provider_family("gpt-5")

    def test_vendor_slugs_are_normalised(self) -> None:
        assert provider_family("openrouter/meta-llama/llama-3.3-70b") == "meta"
        assert provider_family("openrouter/mistralai/mistral-large") == "mistral"
        assert provider_family("openrouter/x-ai/grok-4") == "xai"

    def test_gemini_and_google_prefixes_agree(self) -> None:
        assert provider_family("gemini/gemini-2.5-pro") == provider_family(
            "openrouter/google/gemini-2.5-pro"
        )

    def test_other_hosts_are_peeled_too(self) -> None:
        assert provider_family("vertex_ai/claude-opus-5") == "anthropic"
        assert provider_family("bedrock/anthropic.claude-3") == "anthropic"
        assert provider_family("azure/gpt-5") == "openai"

    def test_a_routed_judge_cannot_rank_its_own_family(self) -> None:
        """The hole this fix closes."""
        with pytest.raises(JudgeError, match="self-preference"):
            check_judge_independence(
                "openrouter/anthropic/claude-opus-5", ["anthropic/claude-haiku-4-5"]
            )

    def test_a_routed_judge_from_a_different_family_is_allowed(self) -> None:
        check_judge_independence(
            "openrouter/mistralai/mistral-large",
            ["anthropic/claude-opus-5", "openai/gpt-5"],
        )


class TestJudgeIndependence:
    def test_different_family_is_allowed(self) -> None:
        check_judge_independence("gpt-5", ["claude-opus-5", "gemini-2.5-pro"])

    def test_same_family_is_refused(self) -> None:
        with pytest.raises(JudgeError, match="self-preference"):
            check_judge_independence("claude-opus-5", ["claude-haiku-4-5", "gpt-5"])

    def test_prefix_does_not_disguise_the_family(self) -> None:
        with pytest.raises(JudgeError):
            check_judge_independence("anthropic/claude-opus-5", ["claude-haiku-4-5"])

    def test_error_names_the_offending_models(self) -> None:
        with pytest.raises(JudgeError, match="claude-haiku-4-5"):
            check_judge_independence("claude-opus-5", ["claude-haiku-4-5"])


class TestBlinding:
    def test_prompt_never_names_a_model(self) -> None:
        """The whole judging design rests on this."""
        prompt = build_prompt(matchup(), ANSWERS)
        assert "alpha" not in prompt.lower()
        assert "beta" not in prompt.lower()

    def test_prompt_uses_neutral_labels(self) -> None:
        prompt = build_prompt(matchup(), ANSWERS)
        assert "Answer 1" in prompt
        assert "Answer 2" in prompt

    def test_both_answers_are_present(self) -> None:
        prompt = build_prompt(matchup(), ANSWERS)
        assert "SENTINEL_ONE_BODY" in prompt
        assert "SENTINEL_TWO_BODY" in prompt

    def test_ordering_swaps_which_answer_is_first(self) -> None:
        forward = build_prompt(matchup("ab"), ANSWERS)
        reverse = build_prompt(matchup("ba"), ANSWERS)
        assert forward.index("SENTINEL_ONE_BODY") < forward.index("SENTINEL_TWO_BODY")
        assert reverse.index("SENTINEL_TWO_BODY") < reverse.index("SENTINEL_ONE_BODY")

    def test_task_context_is_included(self) -> None:
        assert "Python 3.12." in build_prompt(matchup(), ANSWERS)

    def test_strict_variant_adds_a_reminder(self) -> None:
        assert "did not end with a valid verdict" in build_prompt(matchup(), ANSWERS, strict=True)

    def test_system_prompt_forbids_rewarding_length(self) -> None:
        assert "Do not reward verbosity" in JUDGE_SYSTEM


class TestParseVerdict:
    def test_reads_each_token(self) -> None:
        assert parse_verdict("why\nVERDICT: 1") == "first"
        assert parse_verdict("why\nVERDICT: 2") == "second"
        assert parse_verdict("why\nVERDICT: TIE") == "tie"

    def test_is_case_insensitive(self) -> None:
        assert parse_verdict("verdict: tie") == "tie"

    def test_tolerates_surrounding_whitespace(self) -> None:
        assert parse_verdict("reasoning\n  VERDICT:  2  \n") == "second"

    def test_last_verdict_wins(self) -> None:
        """A judge may quote the format before committing to it."""
        assert parse_verdict("e.g. VERDICT: 1\nActually\nVERDICT: 2") == "second"

    def test_missing_verdict_is_none(self) -> None:
        assert parse_verdict("Answer 1 is better, obviously.") is None

    def test_invalid_token_is_none(self) -> None:
        assert parse_verdict("VERDICT: 3") is None
        assert parse_verdict("VERDICT: maybe") is None

    def test_inline_mention_is_not_a_verdict(self) -> None:
        """The token has to be on its own line, not buried in prose."""
        assert parse_verdict("I would say VERDICT: 1 is the format.") is None


class TestPositionToIdentity:
    def test_forward_ordering(self) -> None:
        assert (
            to_comparison(matchup("ab"), "first", judge_model="referee", rationale="").winner_model
            == "alpha"
        )

    def test_reversed_ordering(self) -> None:
        """In a `ba` matchup, Answer 1 was model_b. That is the trap this guards."""
        assert (
            to_comparison(matchup("ba"), "first", judge_model="referee", rationale="").winner_model
            == "beta"
        )

    def test_tie_has_no_winner(self) -> None:
        assert (
            to_comparison(matchup(), "tie", judge_model="referee", rationale="").winner_model
            is None
        )


class TestResolution:
    def test_agreement_stands(self) -> None:
        both = [comparison("ab", "a"), comparison("ba", "a")]
        assert resolve(both) == "a"
        assert not flipped(both)

    def test_disagreement_becomes_a_tie(self) -> None:
        """Position bias, not a winner."""
        both = [comparison("ab", "a"), comparison("ba", "b")]
        assert resolve(both) == "tie"
        assert flipped(both)

    def test_two_ties_stay_a_tie(self) -> None:
        both = [comparison("ab", "tie"), comparison("ba", "tie")]
        assert resolve(both) == "tie"
        assert not flipped(both)

    def test_a_single_ordering_is_not_enough(self) -> None:
        assert resolve([comparison("ab", "a")]) == "tie"


class TestSelection:
    def test_errored_and_empty_generations_are_excluded(self) -> None:
        pool = [
            Generation(
                task_id="t1",
                model="ok",
                output="text",
                tokens_in=1,
                tokens_out=1,
                cost_usd=0.0,
                latency_ms=1,
            ),
            Generation(
                task_id="t1",
                model="broken",
                output="",
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                latency_ms=0,
                error="boom",
            ),
            Generation(
                task_id="t1",
                model="blank",
                output="   ",
                tokens_in=1,
                tokens_out=1,
                cost_usd=0.0,
                latency_ms=1,
            ),
        ]
        assert set(judgeable(pool)) == {("t1", "ok")}

    def test_every_pair_in_both_orderings(self) -> None:
        found = matchups([TASK], ["gamma", "alpha", "beta"])
        assert len(found) == 3 * 2  # three pairs, two orderings each
        assert {(m.model_a, m.model_b) for m in found} == {
            ("alpha", "beta"),
            ("alpha", "gamma"),
            ("beta", "gamma"),
        }

    def test_pairs_are_canonically_sorted(self) -> None:
        """Stable identity: the pair is the same however the models were listed."""
        assert [(m.model_a, m.model_b) for m in matchups([TASK], ["beta", "alpha"])] == [
            ("alpha", "beta"),
            ("alpha", "beta"),
        ]


class FakeJudge:
    """Returns scripted verdict text, keyed by presentation order."""

    def __init__(self, by_order: dict[str, str] | None = None, fail: bool = False) -> None:
        self.requests: list[CompletionRequest] = []
        self._by_order = by_order or {"ab": "VERDICT: 1", "ba": "VERDICT: 2"}
        self._fail = fail

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if self._fail:
            raise PermanentError("judge unavailable")
        order = (
            "ab"
            if request.prompt.index("SENTINEL_ONE_BODY") < request.prompt.index("SENTINEL_TWO_BODY")
            else "ba"
        )
        return CompletionResponse(
            text=f"because reasons\n{self._by_order[order]}",
            tokens_in=5,
            tokens_out=5,
            cost_usd=0.01,
            latency_ms=3,
        )


async def seed(store: ResultStore, *, models: tuple[str, ...] = ("alpha", "beta")) -> None:
    for task in SUITE.tasks:
        for model in models:
            store.record(
                SUITE.name,
                Generation(
                    task_id=task.id,
                    model=model,
                    output=SENTINELS[model],
                    tokens_in=1,
                    tokens_out=1,
                    cost_usd=0.0,
                    latency_ms=1,
                ),
            )


async def judge_run(tmp_path: Path, completer: FakeJudge, *, db: str = "r.duckdb") -> JudgeSummary:
    with ResponseCache(tmp_path / "cache") as cache, ResultStore(tmp_path / db) as store:
        await seed(store)
        return await judge_suite(
            suite=SUITE,
            models=["alpha", "beta"],
            judge_model="referee",
            completer=completer,
            cache=cache,
            store=store,
            config=FAST,
        )


class TestJudgeSuite:
    async def test_judges_every_matchup(self, tmp_path: Path) -> None:
        completer = FakeJudge({"ab": "VERDICT: 1", "ba": "VERDICT: 2"})
        summary = await judge_run(tmp_path, completer)
        assert summary.judged == 4  # two tasks x one pair x two orderings
        assert summary.pairs == 2

    async def test_consistent_judge_shows_no_flips(self, tmp_path: Path) -> None:
        """`1` then `2` both mean alpha won, once ordering is accounted for."""
        summary = await judge_run(tmp_path, FakeJudge({"ab": "VERDICT: 1", "ba": "VERDICT: 2"}))
        assert summary.flips == 0
        assert summary.flip_rate == 0.0

    async def test_position_biased_judge_is_caught(self, tmp_path: Path) -> None:
        """Always picking Answer 1 means the two orderings disagree every time."""
        summary = await judge_run(tmp_path, FakeJudge({"ab": "VERDICT: 1", "ba": "VERDICT: 1"}))
        assert summary.flips == summary.pairs
        assert summary.flip_rate == 1.0

    async def test_prompts_never_leak_model_names(self, tmp_path: Path) -> None:
        completer = FakeJudge()
        await judge_run(tmp_path, completer)
        for request in completer.requests:
            assert "alpha" not in request.prompt.lower()
            assert "beta" not in request.prompt.lower()

    async def test_judge_system_prompt_is_sent(self, tmp_path: Path) -> None:
        completer = FakeJudge()
        await judge_run(tmp_path, completer)
        assert all(r.context == JUDGE_SYSTEM for r in completer.requests)

    async def test_same_family_judge_is_refused_before_any_call(self, tmp_path: Path) -> None:
        completer = FakeJudge()
        with (
            ResponseCache(tmp_path / "c") as cache,
            ResultStore(tmp_path / "r") as store,
            pytest.raises(JudgeError),
        ):
            await judge_suite(
                suite=SUITE,
                models=["claude-opus-5", "gpt-5"],
                judge_model="claude-haiku-4-5",
                completer=completer,
                cache=cache,
                store=store,
                config=FAST,
            )
        assert completer.requests == []

    async def test_second_pass_skips_stored_comparisons(self, tmp_path: Path) -> None:
        await judge_run(tmp_path, FakeJudge())
        second = FakeJudge()
        summary = await judge_run(tmp_path, second)
        assert summary.skipped == 4
        assert second.requests == []

    async def test_unparseable_verdict_is_retried_once_then_recorded(self, tmp_path: Path) -> None:
        completer = FakeJudge({"ab": "no verdict here", "ba": "none here either"})
        summary = await judge_run(tmp_path, completer)
        assert summary.judged == 0
        assert summary.parse_failures == 4
        assert len(completer.requests) == 8  # one retry each, never a third try

    async def test_unparseable_verdict_is_never_coerced(self, tmp_path: Path) -> None:
        with ResponseCache(tmp_path / "c") as cache, ResultStore(tmp_path / "r") as store:
            await seed(store)
            await judge_suite(
                suite=SUITE,
                models=["alpha", "beta"],
                judge_model="referee",
                completer=FakeJudge({"ab": "nothing", "ba": "nothing"}),
                cache=cache,
                store=store,
                config=FAST,
            )
            assert store.comparisons(SUITE.name, "judge") == []
            failures = store.judge_failures(SUITE.name)
        assert len(failures) == 4
        assert all("no verdict token" in f.reason for f in failures)

    async def test_call_failure_is_recorded_not_raised(self, tmp_path: Path) -> None:
        summary = await judge_run(tmp_path, FakeJudge(fail=True))
        assert summary.call_failures == 4
        assert summary.judged == 0

    async def test_pairs_without_two_answers_are_unjudgeable(self, tmp_path: Path) -> None:
        with ResponseCache(tmp_path / "c") as cache, ResultStore(tmp_path / "r") as store:
            store.record(
                SUITE.name,
                Generation(
                    task_id="t1",
                    model="alpha",
                    output="SENTINEL_ONE_BODY",
                    tokens_in=1,
                    tokens_out=1,
                    cost_usd=0.0,
                    latency_ms=1,
                ),
            )
            store.record(
                SUITE.name,
                Generation(
                    task_id="t1",
                    model="beta",
                    output="",
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    latency_ms=0,
                    error="failed",
                ),
            )
            summary = await judge_suite(
                suite=SUITE,
                models=["alpha", "beta"],
                judge_model="referee",
                completer=FakeJudge(),
                cache=cache,
                store=store,
                config=FAST,
            )
        assert summary.judged == 0
        assert summary.unjudgeable == 4
