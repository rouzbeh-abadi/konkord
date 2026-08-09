"""The site is a static file set, so what can be tested is that it stays honest.

The methodology page publishes the judge prompt verbatim, and a published prompt
that has quietly drifted from the one the tool sends is worse than no prompt at
all. That is the drift this module fails on.
"""

from pathlib import Path

import pytest

from konkord.judge import JUDGE_SYSTEM

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
    def test_methodology_publishes_the_exact_judge_prompt(self) -> None:
        """A published prompt that has drifted is worse than none at all."""
        html = (SITE / "methodology.html").read_text(encoding="utf-8")
        assert JUDGE_SYSTEM.strip() in html

    def test_the_prompt_needs_no_html_escaping(self) -> None:
        """Guards the test above: escaping would make it silently pass or fail."""
        assert not set("<>&") & set(JUDGE_SYSTEM)


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
