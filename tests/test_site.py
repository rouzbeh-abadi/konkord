"""The site is a static file set, so what can be tested is that it stays honest.

The methodology page publishes the judge prompt, and a published prompt that has
quietly drifted from the one the tool sends is worse than no prompt at all. The
page therefore holds no copy of its own: it renders whatever the run recorded.
These tests fail if a copy reappears.
"""

import json
from pathlib import Path

import pytest

from konkord.judge import JUDGE_FRAME

SITE = Path(__file__).resolve().parents[1] / "site"
PAGES = ("index.html", "methodology.html", "browse.html")


class TestPages:
    @pytest.mark.parametrize("page", PAGES)
    def test_page_exists(self, page: str) -> None:
        assert (SITE / page).is_file()

    @pytest.mark.parametrize("page", PAGES)
    def test_page_loads_the_shared_assets(self, page: str) -> None:
        html = (SITE / page).read_text(encoding="utf-8")
        assert "style.css" in html
        assert "app.js" in html

    @pytest.mark.parametrize("page", PAGES)
    def test_every_page_links_to_the_others(self, page: str) -> None:
        html = (SITE / page).read_text(encoding="utf-8")
        for target in ("./", "methodology.html", "browse.html"):
            assert target in html


class TestPublishedPrompt:
    def test_methodology_reserves_a_slot_for_the_prompt(self) -> None:
        html = (SITE / "methodology.html").read_text(encoding="utf-8")
        assert 'id="judge-prompt"' in html

    def test_the_page_keeps_no_copy_of_the_prompt(self) -> None:
        """A second copy is a copy that can drift. The run is the only source."""
        html = (SITE / "methodology.html").read_text(encoding="utf-8")
        for line in JUDGE_FRAME.splitlines():
            stripped = line.strip()
            if len(stripped) > 20 and "{rubric}" not in stripped:
                assert stripped not in html, f"methodology.html has its own copy of: {stripped!r}"

    def test_the_prompt_is_rendered_from_the_run(self) -> None:
        html = (SITE / "methodology.html").read_text(encoding="utf-8")
        app = (SITE / "app.js").read_text(encoding="utf-8")
        assert "renderJudgePrompt" in html
        assert "data.judge_prompt" in app

    def test_the_prompt_is_never_rendered_as_markup(self) -> None:
        """A published prompt is the wrong place to start interpreting HTML."""
        app = (SITE / "app.js").read_text(encoding="utf-8")
        body = app[app.index("function renderJudgePrompt") :]
        body = body[: body.index("\nfunction ")]
        assert "textContent" in body
        assert "innerHTML" not in body


class TestHonesty:
    def test_leaderboard_page_reserves_a_slot_for_the_calibration_block(self) -> None:
        html = (SITE / "index.html").read_text(encoding="utf-8")
        assert 'id="calibration"' in html
        assert html.index('id="calibration"') < html.index('id="board"')

    def test_app_marks_a_zero_label_report_as_uncalibrated(self) -> None:
        app = (SITE / "app.js").read_text(encoding="utf-8")
        assert "human_labels === 0" in app
        assert "uncalibrated" in app

    def test_app_refuses_to_order_models_inside_a_rank_group(self) -> None:
        app = (SITE / "app.js").read_text(encoding="utf-8")
        assert "rank_group" in app
        assert "tied" in app.lower()


class TestPublishedRuns:
    """runs.json is the index a static host cannot generate, so it can drift."""

    def runs(self) -> list[dict[str, str]]:
        index = json.loads((SITE / "runs.json").read_text(encoding="utf-8"))
        runs = index["runs"]
        assert isinstance(runs, list)
        return runs

    def test_every_listed_run_has_its_file(self) -> None:
        for run in self.runs():
            assert (SITE / run["file"]).is_file(), f"{run['file']} is listed but missing"

    def test_every_results_file_is_listed(self) -> None:
        """A published file nothing links to is a run nobody can reach."""
        listed = {run["file"] for run in self.runs()}
        on_disk = {p.name for p in SITE.glob("results*.json")}
        assert on_disk == listed, f"unlisted: {on_disk - listed}, missing: {listed - on_disk}"

    def test_each_run_names_the_suite_its_file_contains(self) -> None:
        """Mislabelling which suite a run describes would misattribute its calibration."""
        for run in self.runs():
            report = json.loads((SITE / run["file"]).read_text(encoding="utf-8"))
            assert report["suite"] == run["suite"]

    def test_each_run_carries_a_title(self) -> None:
        for run in self.runs():
            assert run.get("title", "").strip()

    @pytest.mark.parametrize("page", PAGES)
    def test_every_page_offers_the_run_picker(self, page: str) -> None:
        html = (SITE / page).read_text(encoding="utf-8")
        assert 'id="runs"' in html
        assert "renderRunPicker" in html

    def test_the_picker_selects_by_query_parameter(self) -> None:
        """Real links, so a particular run can be shared or opened in a new tab."""
        app = (SITE / "app.js").read_text(encoding="utf-8")
        assert "URLSearchParams" in app
        assert "?suite=" in app
