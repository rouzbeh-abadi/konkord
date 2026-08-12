"""Labelling sample tests.

The sample decides what the calibration number describes, so the properties that
matter are spread, determinism (which is what makes a session resumable), and the
positional-to-identity mapping. Inverting that would flip half the human labels
and look like judge disagreement rather than a bug.
"""

from collections import Counter
from pathlib import Path

from konkord.labeling.sampling import (
    LabelItem,
    already_labelled,
    candidates,
    stratified_sample,
    to_verdict,
)
from konkord.models import Comparison

POOL = [
    (task, "alpha", model_b)
    for task in ("t1", "t2", "t3", "t4")
    for model_b in ("beta", "gamma", "delta")
]


def comparison(task_id: str, source: str = "judge", **overrides: object) -> Comparison:
    fields: dict[str, object] = {
        "task_id": task_id,
        "model_a": "alpha",
        "model_b": "beta",
        "order": "ab",
        "winner": "a",
        "source": source,
        "judge_model": "referee" if source == "judge" else None,
    }
    fields.update(overrides)
    return Comparison(**fields)  # type: ignore[arg-type]


class TestCandidates:
    def test_deduplicates_across_orderings(self) -> None:
        """Both orderings of a pair are one thing to label, not two."""
        both = [comparison("t1", order="ab"), comparison("t1", order="ba")]
        assert candidates(both) == [("t1", "alpha", "beta")]

    def test_result_is_sorted(self) -> None:
        found = candidates([comparison("t2"), comparison("t1")])
        assert found == sorted(found)


class TestSampleShape:
    def test_respects_the_requested_size(self) -> None:
        assert len(stratified_sample(POOL, n=5, seed=1)) == 5

    def test_cannot_exceed_the_pool(self) -> None:
        assert len(stratified_sample(POOL, n=999, seed=1)) == len(POOL)

    def test_no_duplicates(self) -> None:
        sample = stratified_sample(POOL, n=len(POOL), seed=3)
        assert len({item.pair_key for item in sample}) == len(POOL)

    def test_empty_pool_is_empty_sample(self) -> None:
        assert stratified_sample([], n=10, seed=1) == []


class TestStratification:
    def test_tasks_are_evenly_covered(self) -> None:
        """Labels concentrated on one task would describe that task, not the suite."""
        counts = Counter(item.task_id for item in stratified_sample(POOL, n=8, seed=7))
        assert max(counts.values()) - min(counts.values()) <= 1
        assert len(counts) == 4

    def test_partial_sample_touches_every_task_first(self) -> None:
        sample = stratified_sample(POOL, n=4, seed=11)
        assert len({item.task_id for item in sample}) == 4

    def test_model_pairs_are_spread(self) -> None:
        counts = Counter(item.model_b for item in stratified_sample(POOL, n=12, seed=5))
        assert set(counts) == {"beta", "gamma", "delta"}


class TestDeterminism:
    def test_same_seed_gives_the_same_queue(self) -> None:
        """This is what makes closing and reopening the labeller safe."""
        assert stratified_sample(POOL, n=6, seed=42) == stratified_sample(POOL, n=6, seed=42)

    def test_different_seeds_differ(self) -> None:
        assert stratified_sample(POOL, n=6, seed=1) != stratified_sample(POOL, n=6, seed=2)

    def test_orientation_is_randomised(self) -> None:
        """A labeller who always saw the same model first would learn the bias."""
        orders = {item.order for item in stratified_sample(POOL, n=len(POOL), seed=4)}
        assert orders == {"ab", "ba"}


class TestResumption:
    def test_labelled_pairs_are_excluded(self) -> None:
        done = {("t1", "alpha", "beta"), ("t2", "alpha", "gamma")}
        sample = stratified_sample(POOL, n=len(POOL), seed=1, exclude=done)
        assert done.isdisjoint({item.pair_key for item in sample})
        assert len(sample) == len(POOL) - 2

    def test_already_labelled_reads_only_human_rows(self) -> None:
        rows = [
            comparison("t1", source="judge"),
            comparison("t2", source="human", judge_model=None),
        ]
        assert already_labelled(rows) == {("t2", "alpha", "beta")}


class TestVerdictMapping:
    def test_forward_ordering(self) -> None:
        item = LabelItem(task_id="t1", model_a="alpha", model_b="beta", order="ab")
        assert to_verdict(item, "first") == "a"
        assert to_verdict(item, "second") == "b"

    def test_reversed_ordering_inverts_the_positions(self) -> None:
        """In a `ba` item, Answer 1 was model_b."""
        item = LabelItem(task_id="t1", model_a="alpha", model_b="beta", order="ba")
        assert to_verdict(item, "first") == "b"
        assert to_verdict(item, "second") == "a"

    def test_tie_is_orientation_independent(self) -> None:
        for order in ("ab", "ba"):
            item = LabelItem(task_id="t1", model_a="alpha", model_b="beta", order=order)
            assert to_verdict(item, "tie") == "tie"


class TestAppIsImportSafe:
    """The labeller module must not do anything when imported.

    `konkord label` resolves the app by path precisely so it never imports it,
    but a module that runs a Streamlit app at import time is a trap for anything
    else that touches it. This failed for real: the CLI imported the module to
    read `__file__`, which executed the app before its environment was set.
    """

    def test_importing_the_app_does_not_run_it(self) -> None:
        import importlib

        module = importlib.import_module("konkord.labeling.app")
        assert hasattr(module, "main")

    def test_the_cli_never_imports_the_app(self) -> None:
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "src/konkord/cli.py").read_text()
        assert "from konkord.labeling import app" not in source
        assert "import konkord.labeling.app" not in source


class TestLabellerIsDomainNeutral:
    """The labeller shows answers the way its suite says they should be read."""

    def source(self) -> str:
        return (Path(__file__).resolve().parents[1] / "src/konkord/labeling/app.py").read_text(
            encoding="utf-8"
        )

    def test_no_language_is_hard_coded(self) -> None:
        """A suite of prose rendered as Python is punishing to label a hundred of."""
        assert 'language="python"' not in self.source()
        assert "suite.answer_language" in self.source()

    def test_the_labeller_is_shown_the_same_rubric_as_the_judge(self) -> None:
        """Otherwise agreement measures the gap between two unstated standards."""
        assert "suite.rubric" in self.source()
