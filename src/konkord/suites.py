"""Reading task suites from YAML.

The file format and the domain model are deliberately separate types. The file
carries defaults that have to be resolved (`checks_default`, suite-level
`context`); `Suite` and `Task` carry only resolved values, so no stage after
this one has to know that defaults ever existed.

Check resolution is a **union**: a task receives the suite defaults plus
whatever it names itself. A task cannot opt out of a default. That is the point
if some tasks could silently skip the compile check, models would be compared
on unequal terms and the leaderboard would be quietly wrong.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from konkord.models import Suite, Task


class SuiteError(Exception):
    """A suite file is missing, unreadable, or does not describe a valid suite."""


class _TaskEntry(BaseModel):
    """One `tasks:` entry exactly as it appears in the file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    prompt: str
    context: str | None = None
    checks: tuple[str, ...] = ()
    reference: str | None = None


class _SuiteFile(BaseModel):
    """The whole file, before defaults are resolved."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    context: str | None = None
    #: Required, with no default, unlike everywhere else in this file. A rubric
    #: is what every number downstream means, and a suite that inherited one
    #: silently would be graded on a standard its author never read. The cost of
    #: getting it wrong is invisible, so the cost of omitting it is not.
    rubric: str = Field(min_length=1)
    answer_language: str | None = None
    checks_default: tuple[str, ...] = ()
    tasks: tuple[_TaskEntry, ...] = Field(min_length=1)


def load_suite(path: Path) -> Suite:
    """Load and validate a suite file.

    Raises:
        SuiteError: with a message naming the file and the offending field.
    """
    text = _read(path)
    raw = _parse_yaml(text, path)
    spec = _validate(raw, path)
    return _resolve(spec, path)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SuiteError(f"{path}: cannot read suite file ({exc.strerror})") from exc


def _parse_yaml(text: str, path: Path) -> object:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SuiteError(f"{path}: invalid YAML ({exc})") from exc


def _validate(raw: object, path: Path) -> _SuiteFile:
    if raw is None:
        raise SuiteError(f"{path}: suite file is empty")
    if not isinstance(raw, dict):
        raise SuiteError(f"{path}: expected a mapping at the top level, found {type(raw).__name__}")
    try:
        return _SuiteFile.model_validate(raw)
    except ValidationError as exc:
        raise SuiteError(_describe(exc, path)) from exc


def _resolve(spec: _SuiteFile, path: Path) -> Suite:
    """Apply suite-level defaults, then hand off to the domain model's own checks."""
    tasks: list[Task] = []
    for index, entry in enumerate(spec.tasks):
        try:
            tasks.append(
                Task(
                    id=entry.id,
                    prompt=entry.prompt,
                    context=entry.context if entry.context is not None else spec.context,
                    checks=_merge_checks(spec.checks_default, entry.checks),
                    reference=entry.reference,
                )
            )
        except ValidationError as exc:
            # Locations are relative to the Task, so re-root them at the file.
            raise SuiteError(_describe(exc, path, prefix=f"tasks.{index}")) from exc
    try:
        return Suite(
            name=spec.name,
            context=spec.context,
            rubric=spec.rubric.strip(),
            answer_language=spec.answer_language,
            tasks=tuple(tasks),
        )
    except ValidationError as exc:
        raise SuiteError(_describe(exc, path)) from exc


def _merge_checks(defaults: tuple[str, ...], extra: tuple[str, ...]) -> tuple[str, ...]:
    """Union of suite defaults and task-level checks, order-preserving and deduplicated."""
    return tuple(dict.fromkeys(defaults + extra))


def _describe(exc: ValidationError, path: Path, prefix: str = "") -> str:
    """Turn a pydantic error into something a suite author can act on."""
    lines = [f"{path}: invalid suite"]
    for error in exc.errors():
        parts = [str(part) for part in error["loc"]]
        if prefix:
            parts.insert(0, prefix)
        location = ".".join(part for part in parts if part) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)
