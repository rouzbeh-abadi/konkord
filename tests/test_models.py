"""Contract tests for the shared models.

The comparison-identity tests matter most: `calibrate` joins human labels to
judge verdicts on `pair_key` and compares `winner_model`, so a regression here
would silently corrupt the headline number rather than raise.
"""

import pytest
from pydantic import ValidationError

from konkord.models import Comparison, Generation, Suite, Task


def _task(task_id: str = "t1") -> Task:
    return Task(id=task_id, prompt="do the thing")


def _comparison(**overrides: object) -> Comparison:
    fields: dict[str, object] = {
        "task_id": "t1",
        "model_a": "alpha",
        "model_b": "beta",
        "order": "ab",
        "winner": "a",
        "source": "judge",
        "judge_model": "referee",
    }
    fields.update(overrides)
    return Comparison(**fields)  # type: ignore[arg-type]


class TestStrictness:
    def test_models_are_frozen(self) -> None:
        task = _task()
        with pytest.raises(ValidationError):
            task.id = "other"

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Task(id="t1", prompt="p", tags=["nope"])  # type: ignore[call-arg]

    def test_task_id_must_be_a_slug(self) -> None:
        with pytest.raises(ValidationError):
            _task("Not A Slug")

    def test_negative_cost_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Generation(
                task_id="t1",
                model="alpha",
                output="x",
                tokens_in=1,
                tokens_out=1,
                cost_usd=-0.01,
                latency_ms=5,
            )


class TestSuite:
    def test_duplicate_task_ids_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate task ids: t1"):
            Suite(name="s", tasks=(_task("t1"), _task("t1")))

    def test_empty_suite_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Suite(name="s", tasks=())

    def test_task_lookup(self) -> None:
        suite = Suite(name="s", tasks=(_task("t1"), _task("t2")))
        assert suite.task("t2").id == "t2"
        with pytest.raises(KeyError):
            suite.task("absent")


class TestComparisonIdentity:
    def test_pair_key_ignores_orientation(self) -> None:
        """The same two models on the same task are one comparison, either way round."""
        forward = _comparison(model_a="alpha", model_b="beta")
        reversed_ = _comparison(model_a="beta", model_b="alpha")
        assert forward.pair_key == reversed_.pair_key

    def test_pair_key_separates_tasks(self) -> None:
        assert _comparison(task_id="t1").pair_key != _comparison(task_id="t2").pair_key

    def test_winner_model_is_orientation_independent(self) -> None:
        """Judge and human can disagree on layout and still agree on the winner."""
        judge = _comparison(model_a="alpha", model_b="beta", winner="a")
        human = _comparison(
            model_a="beta",
            model_b="alpha",
            winner="b",
            source="human",
            judge_model=None,
        )
        assert judge.winner_model == human.winner_model == "alpha"

    def test_tie_has_no_winning_model(self) -> None:
        assert _comparison(winner="tie").winner_model is None


class TestComparisonInvariants:
    def test_model_cannot_face_itself(self) -> None:
        with pytest.raises(ValidationError, match="cannot be compared with itself"):
            _comparison(model_b="alpha")

    def test_judge_verdict_requires_judge_model(self) -> None:
        with pytest.raises(ValidationError, match="must record judge_model"):
            _comparison(judge_model=None)

    def test_human_verdict_forbids_judge_model(self) -> None:
        with pytest.raises(ValidationError, match="must not record judge_model"):
            _comparison(source="human", judge_model="referee")
