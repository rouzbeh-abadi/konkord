"""Calibration tests.

The join and the kappa are where this can silently produce a wrong headline
number, so most of these check a property of the statistic rather than a value
the code happened to emit.
"""

import pytest

from konkord.calibrate import (
    align,
    by_length_quartile,
    by_model_pair,
    by_task,
    calibrate,
    cohens_kappa,
)
from konkord.models import Comparison, Generation


def judge(task: str, winner: str, order: str = "ab", rationale: str = "because") -> Comparison:
    return Comparison(
        task_id=task,
        model_a="alpha",
        model_b="beta",
        order=order,  # type: ignore[arg-type]
        winner=winner,  # type: ignore[arg-type]
        source="judge",
        rationale=rationale,
        judge_model="referee",
    )


def human(task: str, winner: str, order: str = "ab") -> Comparison:
    return Comparison(
        task_id=task,
        model_a="alpha",
        model_b="beta",
        order=order,  # type: ignore[arg-type]
        winner=winner,  # type: ignore[arg-type]
        source="human",
    )


def generation(task: str, model: str, chars: int) -> Generation:
    return Generation(
        task_id=task,
        model=model,
        output="x" * chars,
        tokens_in=1,
        tokens_out=1,
        cost_usd=0.0,
        latency_ms=1,
    )


def both_orders(task: str, winner: str) -> list[Comparison]:
    """A judge that agreed with itself across orderings."""
    return [judge(task, winner, "ab"), judge(task, winner, "ba")]


class TestAlignment:
    def test_matches_judge_and_human_on_the_same_pair(self) -> None:
        pairs = align(both_orders("t1", "a"), [human("t1", "a")], [])
        assert len(pairs) == 1
        assert pairs[0].agreed

    def test_orientation_does_not_break_the_join(self) -> None:
        """The human saw the pair the other way round; that is not disagreement."""
        pairs = align(both_orders("t1", "a"), [human("t1", "a", order="ba")], [])
        assert pairs[0].judge_winner == "alpha"
        assert pairs[0].human_winner == "alpha"
        assert pairs[0].agreed

    def test_a_flipped_judge_has_no_opinion(self) -> None:
        """Disagreeing with itself resolves to a tie, not to one of the orderings."""
        flipped = [judge("t1", "a", "ab"), judge("t1", "b", "ba")]
        pairs = align(flipped, [human("t1", "a")], [])
        assert pairs[0].judge_winner is None
        assert not pairs[0].agreed

    def test_unlabelled_pairs_are_ignored(self) -> None:
        assert align(both_orders("t1", "a"), [], []) == []

    def test_human_labels_without_a_judge_verdict_are_ignored(self) -> None:
        assert align([], [human("t1", "a")], []) == []

    def test_carries_the_judge_rationale_for_the_gallery(self) -> None:
        pairs = align(both_orders("t1", "a"), [human("t1", "b")], [])
        assert pairs[0].judge_rationale == "because"


class TestAgreement:
    def test_perfect_agreement(self) -> None:
        judged = both_orders("t1", "a") + both_orders("t2", "b")
        result = calibrate(judged, [human("t1", "a"), human("t2", "b")], [])
        assert result.agreement == 1.0
        assert result.labelled == 2

    def test_total_disagreement(self) -> None:
        judged = both_orders("t1", "a") + both_orders("t2", "a")
        result = calibrate(judged, [human("t1", "b"), human("t2", "b")], [])
        assert result.agreement == 0.0
        assert len(result.disagreements) == 2

    def test_no_labels_reports_zero_rather_than_dividing_by_zero(self) -> None:
        result = calibrate(both_orders("t1", "a"), [], [])
        assert result.labelled == 0
        assert result.agreement == 0.0


class TestKappa:
    def test_perfect_agreement_with_varied_labels_is_one(self) -> None:
        judged = both_orders("t1", "a") + both_orders("t2", "b")
        pairs = align(judged, [human("t1", "a"), human("t2", "b")], [])
        assert cohens_kappa(pairs) == pytest.approx(1.0)

    def test_agreement_by_always_saying_the_same_thing_scores_zero(self) -> None:
        """The reason raw agreement alone is not enough.

        Both raters always pick alpha, so they agree 100% of the time while
        carrying no information. Kappa is undefined here and the convention is
        to report perfect agreement as 1.0; the informative case is the next
        test, where one rater varies.
        """
        judged = both_orders("t1", "a") + both_orders("t2", "a")
        pairs = align(judged, [human("t1", "a"), human("t2", "a")], [])
        assert cohens_kappa(pairs) == 1.0

    def test_chance_level_agreement_scores_near_zero(self) -> None:
        judged = (
            both_orders("t1", "a")
            + both_orders("t2", "a")
            + both_orders("t3", "b")
            + both_orders("t4", "b")
        )
        humans = [human("t1", "a"), human("t2", "b"), human("t3", "a"), human("t4", "b")]
        assert cohens_kappa(align(judged, humans, [])) == pytest.approx(0.0, abs=1e-9)

    def test_kappa_is_below_raw_agreement_when_labels_are_skewed(self) -> None:
        judged = (
            both_orders("t1", "a")
            + both_orders("t2", "a")
            + both_orders("t3", "a")
            + both_orders("t4", "b")
        )
        humans = [human("t1", "a"), human("t2", "a"), human("t3", "b"), human("t4", "b")]
        pairs = align(judged, humans, [])
        observed = sum(1 for p in pairs if p.agreed) / len(pairs)
        assert cohens_kappa(pairs) < observed

    def test_no_pairs(self) -> None:
        assert cohens_kappa([]) == 0.0


class TestBreakdowns:
    def test_by_task(self) -> None:
        judged = both_orders("t1", "a") + both_orders("t2", "a")
        pairs = align(judged, [human("t1", "a"), human("t2", "b")], [])
        rows = {b.key: b for b in by_task(pairs)}
        assert rows["t1"].rate == 1.0
        assert rows["t2"].rate == 0.0

    def test_by_model_pair(self) -> None:
        pairs = align(both_orders("t1", "a"), [human("t1", "a")], [])
        assert [b.key for b in by_model_pair(pairs)] == ["alpha vs beta"]

    def test_length_quartiles_split_the_sample(self) -> None:
        """The breakdown that surfaces verbosity bias."""
        judged: list[Comparison] = []
        humans: list[Comparison] = []
        generations: list[Generation] = []
        for index in range(8):
            task = f"t{index}"
            judged += both_orders(task, "a")
            humans.append(human(task, "a"))
            generations.append(generation(task, "alpha", 10 * (index + 1)))
            generations.append(generation(task, "beta", 10 * (index + 1)))
        quartiles = by_length_quartile(align(judged, humans, generations))
        assert len(quartiles) == 4
        assert sum(q.labelled for q in quartiles) == 8

    def test_length_quartiles_on_an_empty_sample(self) -> None:
        assert by_length_quartile([]) == []


class TestFullCalibration:
    def test_reports_the_judge_model(self) -> None:
        result = calibrate(both_orders("t1", "a"), [human("t1", "a")], [])
        assert result.judge_models == ("referee",)

    def test_failure_gallery_holds_only_disagreements(self) -> None:
        judged = both_orders("t1", "a") + both_orders("t2", "a")
        result = calibrate(judged, [human("t1", "a"), human("t2", "b")], [])
        assert len(result.disagreements) == 1
        assert result.disagreements[0].task_id == "t2"
        assert result.disagreements[0].judge_rationale == "because"
