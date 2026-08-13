# Contributing

Thanks for looking. This is a small project with a narrow claim, so the most useful contributions
are ones that make the claim harder to fool.

## Setup

```bash
git clone https://github.com/rouzbeh-abadi/konkord
cd konkord
uv sync --all-extras
```

Everything CI checks, in one line:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

`mypy` runs in strict mode over `src` and `tests`, and the test suite never touches the network:
provider calls go through the `Completer` protocol in `providers.py`, and tests substitute a fake.
If a change makes a test need a real API key, the seam is in the wrong place.

## What a good change looks like

- **One concern per pull request.** A bug fix and a refactor in the same diff take twice as long to
  review and cannot be reverted separately.
- **A test that fails before the change.** For a bug fix, the test should reproduce the bug.
- **A reason, not just a description.** The comments in this codebase explain why a decision was
  made where the code cannot; the commit message should do the same.

Numbers reported to a user must be the numbers that were measured. A rounding that turns a real cost
into `$0.0000`, a stale row left in a failure table, or a verdict inferred from a response that did
not contain one are all bugs here, even when nothing crashes.

## Adding a task to a suite

Suites are YAML, one file per suite, in `suites/`. A task is worth adding if a competent developer
could get it wrong in a specific way, and the mistake shows in the code rather than in a comment
about the code. Ambiguous prompts produce ambiguous comparisons, and those are the ones the judge
and the human split on.

```yaml
tasks:
  - id: interval-merge-01
    checks: [mypy_clean]
    prompt: |
      Write a function that merges overlapping closed integer intervals ...
```

`checks` are declared per task and resolved as a union with the suite's `checks_default`, so a task
can add a check but never opt out of one: every model is compared on the same terms. Nothing
executes them yet; see the note at the end of the README.

## Writing a rubric for a new suite

`rubric` is required and has no default, because it decides what every number downstream means. Write
the criteria in priority order and stop: the response format and the instruction to ignore length are
the tool's, not yours, and a rubric mentioning the verdict token is refused.

Two things worth knowing before you write one:

- **A human is going to read it too.** The labeller shows the rubric beside the answers, so it has to
  be something a person can apply consistently at item seventy. "Prefer the better answer" is not.
- **Changing it invalidates the verdicts under it.** Every verdict records the prompt that produced
  it, and the tool refuses to add verdicts under a second one rather than fitting one rating across
  two standards. Settle the rubric before spending money on a run.

Set `answer_language` to a syntax name for code suites, or leave it out for prose so the labeller and
the browse page render paragraphs as paragraphs.

## Reporting a judge failure

If the judge produced something Konkord could not parse, the row is already in the `judge_failures`
table with the raw response attached. Paste that row into the issue. A judge that fails on a whole
class of input is more interesting than one that fails once.

## What is out of scope

Sandboxed execution of model answers is planned and deliberately absent rather than half-built. A
pull request that shells out to run untrusted model output without isolation will not be merged.

## Licence

By contributing you agree that your contribution is licensed under the MIT licence, as in
[LICENSE](LICENSE).
