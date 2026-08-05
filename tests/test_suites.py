"""Loader tests, including that the suite shipped in the repo actually parses."""

from pathlib import Path

import pytest

from konkord.suites import SuiteError, load_suite

REPO_SUITE = Path(__file__).resolve().parents[1] / "suites" / "python_codegen.yaml"

VALID = """
name: demo
context: Suite-level context.
checks_default: [compiles, ruff_clean]
tasks:
  - id: alpha-01
    prompt: First prompt.
  - id: beta-01
    prompt: Second prompt.
    context: Task-level context.
    checks: [mypy_clean]
    reference: The gold answer.
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoading:
    def test_loads_a_valid_suite(self, tmp_path: Path) -> None:
        suite = load_suite(write(tmp_path, VALID))
        assert suite.name == "demo"
        assert [task.id for task in suite.tasks] == ["alpha-01", "beta-01"]

    def test_repo_suite_parses(self) -> None:
        """Guards against the shipped suite drifting out of sync with the schema."""
        suite = load_suite(REPO_SUITE)
        assert suite.name == "python_codegen"
        assert len(suite.tasks) == 25


class TestDefaults:
    def test_task_inherits_suite_context(self, tmp_path: Path) -> None:
        suite = load_suite(write(tmp_path, VALID))
        assert suite.task("alpha-01").context == "Suite-level context."

    def test_task_context_overrides_suite_context(self, tmp_path: Path) -> None:
        suite = load_suite(write(tmp_path, VALID))
        assert suite.task("beta-01").context == "Task-level context."

    def test_checks_are_the_union_of_defaults_and_task_checks(self, tmp_path: Path) -> None:
        """A task adds to the suite floor; it cannot opt out of it."""
        suite = load_suite(write(tmp_path, VALID))
        assert suite.task("alpha-01").checks == ("compiles", "ruff_clean")
        assert suite.task("beta-01").checks == ("compiles", "ruff_clean", "mypy_clean")

    def test_checks_are_deduplicated_in_order(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            """
            name: demo
            checks_default: [compiles, ruff_clean]
            tasks:
              - id: alpha-01
                prompt: p
                checks: [ruff_clean, mypy_clean]
            """,
        )
        assert load_suite(path).task("alpha-01").checks == (
            "compiles",
            "ruff_clean",
            "mypy_clean",
        )


class TestErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteError, match="cannot read suite file"):
            load_suite(tmp_path / "absent.yaml")

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteError, match="invalid YAML"):
            load_suite(write(tmp_path, "name: [unclosed\n"))

    def test_empty_file(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteError, match="suite file is empty"):
            load_suite(write(tmp_path, ""))

    def test_top_level_must_be_a_mapping(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteError, match="expected a mapping"):
            load_suite(write(tmp_path, "- just\n- a list\n"))

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        """A typo'd key must fail loudly rather than being silently ignored."""
        with pytest.raises(SuiteError, match="checks_defaults"):
            load_suite(write(tmp_path, "name: demo\nchecks_defaults: [a]\ntasks: []\n"))

    def test_error_names_the_offending_task_field(self, tmp_path: Path) -> None:
        path = write(tmp_path, "name: demo\ntasks:\n  - id: Bad Id\n    prompt: p\n")
        with pytest.raises(SuiteError, match=r"tasks\.0\.id"):
            load_suite(path)

    def test_duplicate_ids_are_reported(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "name: demo\ntasks:\n  - id: a-1\n    prompt: p\n  - id: a-1\n    prompt: q\n",
        )
        with pytest.raises(SuiteError, match="duplicate task ids: a-1"):
            load_suite(path)

    def test_suite_with_no_tasks_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteError, match="tasks"):
            load_suite(write(tmp_path, "name: demo\ntasks: []\n"))
