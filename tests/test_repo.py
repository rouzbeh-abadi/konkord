"""Repository files that only CI would otherwise validate.

A workflow file GitHub cannot parse does not fail loudly in a useful place. It
produces a run named after its own path, reporting "no jobs were run", on every
push, until somebody reads the email. That happened here: seven files were
written with a stray closing tag on the last line, and two of them were
workflows. So it is tested rather than trusted.

The stray-markup check walks the tree instead of naming files, because the
version of it that named files would not have caught the file it was written in.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
TEMPLATES = sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml"))

#: Markup that means a file was written by pasting rather than by an editor.
#: Split so this module does not trip its own check.
STRAY_MARKUP = ("</" + "content>", "</" + "invoke>", "</" + "antml:invoke>")

SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".konkord_cache", ".wrangler"}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".json", ".html", ".css", ".js", ".jsonc"}


def text_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix in TEXT_SUFFIXES
        and not SKIP_DIRS & set(path.relative_to(ROOT).parts)
    )


class TestNoStrayMarkup:
    def test_the_tree_has_files_to_check(self) -> None:
        """Guards the test below from passing because it found nothing."""
        assert len(text_files()) > 20

    def test_no_file_carries_a_paste_artefact(self) -> None:
        """One of these shipped in two workflow files and broke both."""
        offenders = [
            f"{path.relative_to(ROOT)}: {marker}"
            for path in text_files()
            for marker in STRAY_MARKUP
            if marker in path.read_text(encoding="utf-8", errors="ignore")
        ]
        assert not offenders, "stray markup: " + ", ".join(offenders)


@pytest.mark.parametrize("path", WORKFLOWS + TEMPLATES, ids=lambda p: p.name)
def test_github_config_parses_as_yaml(path: Path) -> None:
    assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)


class TestWorkflows:
    def test_there_are_some(self) -> None:
        assert WORKFLOWS

    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_declares_a_trigger_and_a_job(self, path: Path) -> None:
        """`on` parses as the boolean True under YAML 1.1, which GitHub accepts."""
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert doc.get("on") or doc.get(True), f"{path.name} declares no trigger"
        assert doc.get("jobs"), f"{path.name} declares no jobs"
