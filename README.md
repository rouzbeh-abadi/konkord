# Konkord

Rank LLMs on a task suite, and measure whether the automated judge can be trusted.

The usual pipeline (run models, have an LLM grade the outputs, publish a leaderboard) is
commodity. Konkord adds one step: the operator hand-labels a sample of comparisons **blind**, and
every published ranking ships with the judge-versus-human agreement rate attached.

If the judge agrees with the human 85% of the time, the leaderboard is credible. If it agrees 55%
of the time, that is itself the finding.

> **Status: pre-alpha.** Everything except `check` works. `check` exits non-zero and says which
> build phase it belongs to. No leaderboard has been published from this yet; the numbers below
> are the ones the tool produces, not results.

## Quickstart

Not on PyPI yet, so clone it:

```bash
git clone https://github.com/rouzbeh-abadi/konkord
cd konkord
uv sync --all-extras
```

Set credentials for whichever providers you are ranking. Konkord calls models through
[litellm](https://docs.litellm.ai/docs/providers), so the usual environment variables apply:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

Then the whole pipeline. Substitute your own models; the judge must come from a provider family
that is not being ranked:

```bash
SUITE=suites/python_codegen.yaml
MODELS=gpt-5,claude-opus-5,gemini-2.5-pro

uv run konkord run       --suite $SUITE --models $MODELS
uv run konkord judge     --suite $SUITE --models $MODELS --judge mistral/mistral-large-latest
uv run konkord label     --suite $SUITE --n 100
uv run konkord calibrate --suite $SUITE
uv run konkord report    --suite $SUITE --out results.json
```

`run` and `judge` cost money and take a few minutes. Both are resumable and cache every response,
so re-running them is free. `label` opens a local app and is the part that needs you: budget a
couple of hours for 100 comparisons. `calibrate` has nothing to say until those labels exist.

## Commands

| Command | Does | Status |
|---|---|---|
| `konkord run` | Generate one output per (task × model) | ✅ |
| `konkord judge` | Pairwise LLM-as-judge, both orderings | ✅ |
| `konkord label` | Local blind labeller | ✅ |
| `konkord calibrate` | Judge-versus-human agreement and kappa | ✅ |
| `konkord report` | Aggregate into `results.json` | ✅ |
| `konkord check` | Deterministic graders in a sandbox | phase 4 |

## Running a suite

```bash
konkord run --suite suites/python_codegen.yaml --models gpt-5,claude-opus-5,gemini-2.5-pro
```

Model names are whatever [litellm](https://docs.litellm.ai/docs/providers) accepts; provider
credentials come from the usual environment variables. The command is idempotent and resumable:
a `(task, model)` pair already in the results file is not regenerated, and responses are cached on
disk, so a repeat run costs nothing. Permanent failures are recorded against the generation rather
than aborting the run.

## Judging

```bash
konkord judge --suite suites/python_codegen.yaml \
  --models gpt-5,claude-opus-5,gemini-2.5-pro --judge <a model from a fourth family>
```

Every model pair is judged twice, once in each presentation order, as two separate calls. The
judge sees "Answer 1" and "Answer 2" and never a model name. If the two orderings disagree the pair
is recorded as a tie, and the **order-flip rate** is printed: a high flip rate means the judge is
deciding on position rather than content, and the ranking built on it is not worth much.

A judge from the same provider family as any ranked model is refused outright, because self-preference
bias is not something this tool can correct for. Verdicts that cannot be parsed are retried once and
then recorded in a `judge_failures` table, never coerced into a winner.

## Labelling

```bash
pip install 'konkord[label]'
konkord label --suite suites/python_codegen.yaml --n 100
```

Opens a local Streamlit app showing sampled comparisons side by side, with no model identity and no
judge verdict visible. Seeing the judge's opinion first would measure suggestibility rather than
agreement. The sample is stratified across tasks and model pairs, orientation is randomised per
item, and every label is written immediately, so the session is resumable with the same `--seed`.

## Calibrating and reporting

```bash
konkord calibrate --suite suites/python_codegen.yaml
konkord report --suite suites/python_codegen.yaml --out results.json
```

`calibrate` joins the human labels to the judge verdicts on the same comparisons and reports raw
agreement, Cohen's kappa, breakdowns by task, by model pair and by answer-length quartile, plus a
failure gallery of every disagreement with the judge's own rationale beside it. The length-quartile
breakdown is what surfaces verbosity bias.

`report` writes `results.json`: Bradley-Terry ratings fitted over all pairwise comparisons rather
than raw win counts, win rates with bootstrap 95% confidence intervals, per-model cost and median
latency, and the calibration block. Models whose intervals overlap share a `rank_group` and must be
rendered as tied; presenting an order within a group asserts a difference the data does not support.

A report produced before any labelling still runs, and says plainly that it is uncalibrated.

## Development

```bash
uv sync
uv run konkord --help
```

Lint, type-check and test, the same four commands CI runs:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

## License

MIT. Built by [Rouzbeh Abadi](https://github.com/rouzbeh-abadi).
