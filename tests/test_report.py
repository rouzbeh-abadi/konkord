"""Report assembly.

The properties worth pinning are the ones a reader of results.json relies on:
the calibration block travels with the ranking, flipped pairs do not become
wins, and overlapping intervals are marked as tied.
"""

import json
from pathlib import Path

from konkord.calibrate import calibrate
from konkord.models import Comparison, Generation
from konkord.report import Report, build, flip_rate, resolved_outcomes, write


def judge(task: str, winner: str, order: str, a: str = "alpha", b: str = "beta") -> Comparison:
    return Comparison(
        task_id=task,
        model_a=a,
        model_b=b,
        order=order,  # type: ignore[arg-type]
        winner=winner,  # type: ignore[arg-type]
        source="judge",
        rationale="reasons",
        judge_model="referee",
    )


def agreed(task: str, winner: str, a: str = "alpha", b: str = "beta") -> list[Comparison]:
    return [judge(task, winner, "ab", a, b), judge(task, winner, "ba", a, b)]


def generation(task: str, model: str, latency: int = 100, cost: float = 0.01) -> Generation:
    return Generation(
        task_id=task,
        model=model,
        output="answer",
        tokens_in=1,
        tokens_out=1,
        cost_usd=cost,
        latency_ms=latency,
    )


class TestOutcomeResolution:
    def test_consistent_pair_yields_a_winner(self) -> None:
        outcomes = resolved_outcomes(agreed("t1", "a"))
        assert len(outcomes) == 1
        assert outcomes[0].winner == "alpha"

    def test_flipped_pair_yields_a_tie(self) -> None:
        """Position bias must reduce separation, not manufacture a win."""
        flipped = [judge("t1", "a", "ab"), judge("t1", "b", "ba")]
        assert resolved_outcomes(flipped)[0].winner is None

    def test_flip_rate(self) -> None:
        rows = [*agreed("t1", "a"), judge("t2", "a", "ab"), judge("t2", "b", "ba")]
        assert flip_rate(rows) == 0.5

    def test_flip_rate_with_no_rows(self) -> None:
        assert flip_rate([]) == 0.0


def sample_report(resamples: int = 1000) -> Report:
    judge_rows = agreed("t1", "a") + agreed("t2", "a") + agreed("t3", "a")
    human_rows = [
        Comparison(
            task_id="t1",
            model_a="alpha",
            model_b="beta",
            order="ab",
            winner="a",
            source="human",
        )
    ]
    generations = [
        generation("t1", "alpha", latency=100),
        generation("t1", "beta", latency=300),
        generation("t2", "alpha", latency=200),
        generation("t2", "beta", latency=400),
    ]
    return build(
        suite="demo",
        generations=generations,
        judge_rows=judge_rows,
        calibration=calibrate(judge_rows, human_rows, generations),
        resamples=resamples,
        seed=1,
    )


class TestReport:
    def test_ranks_the_stronger_model_first(self) -> None:
        assert sample_report().models[0].model == "alpha"

    def test_every_model_carries_an_interval(self) -> None:
        for entry in sample_report().models:
            assert entry.ci_low <= entry.win_rate <= entry.ci_high

    def test_costs_and_latency_are_per_model(self) -> None:
        entries = {m.model: m for m in sample_report().models}
        assert entries["alpha"].median_latency_ms == 150
        assert entries["beta"].median_latency_ms == 350
        assert entries["alpha"].cost_usd == 0.02

    def test_calibration_travels_with_the_ranking(self) -> None:
        """A ranking without its calibration number is a claim with no evidence."""
        block = sample_report().calibration
        assert block.human_labels == 1
        assert block.agreement == 1.0
        assert block.judge_models == ("referee",)

    def test_records_the_bootstrap_settings(self) -> None:
        report = sample_report(resamples=1000)
        assert report.bootstrap_resamples == 1000
        assert report.bootstrap_seed == 1

    def test_tied_models_share_a_rank_group(self) -> None:
        rows = [*agreed("t1", "a"), judge("t2", "b", "ab"), judge("t2", "b", "ba")]
        report = build(
            suite="demo",
            generations=[],
            judge_rows=rows,
            calibration=calibrate(rows, [], []),
            resamples=1000,
            seed=2,
        )
        assert len({m.rank_group for m in report.models}) == 1

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        write(sample_report(), path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["suite"] == "demo"
        assert "calibration" in loaded
        assert loaded["models"][0]["model"] == "alpha"

    def test_uncalibrated_report_is_still_produced_but_empty(self) -> None:
        """Report must run before labelling; it just reports zero labels."""
        rows = agreed("t1", "a")
        report = build(
            suite="demo",
            generations=[],
            judge_rows=rows,
            calibration=calibrate(rows, [], []),
        )
        assert report.calibration.human_labels == 0
        assert report.calibration.failure_gallery == ()
