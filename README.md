# Konkord

Rank LLMs on a task suite, and measure whether the automated judge can be trusted.

The usual pipeline — run models, have an LLM grade the outputs, publish a leaderboard — is
commodity. Konkord adds one step: the operator hand-labels a sample of comparisons **blind**, and
every published ranking ships with the judge-versus-human agreement rate attached.

If the judge agrees with the human 85% of the time, the leaderboard is credible. If it agrees 55%
of the time, that is itself the finding.

> **Status: pre-alpha.** The command surface below is fixed. `run`, `judge` and `label` work;
> the rest exit non-zero and say which build phase they belong to.

## Commands

| Command | Does | Status |
|---|---|---|
| `konkord run` | Generate one output per (task × model) | ✅ |
| `konkord judge` | Pairwise LLM-as-judge, both orderings | ✅ |
| `konkord label` | Local blind labeller | ✅ |
| `konkord check` | Deterministic graders in a sandbox | phase 4 |
| `konkord calibrate` | Judge-versus-human agreement and kappa | phase 7 |
| `konkord report` | Aggregate into `results.json` | phase 7 |

## Running a suite

```bash
konkord run --suite suites/python_codegen.yaml --models gpt-5,claude-opus-5,gemini-2.5-pro
```

Model names are whatever [litellm](https://docs.litellm.ai/docs/providers) accepts; provider
credentials come from the usual environment variables. The command is idempotent and resumable —
a `(task, model)` pair already in the results file is not regenerated, and responses are cached on
disk, so a repeat run costs nothing. Permanent failures are recorded against the generation rather
than aborting the run.

## Judging

```bash
konkord judge --suite suites/python_codegen.yaml \
  --models gpt-5,claude-opus-5,gemini-2.5-pro --judge <a model from a fourth family>
```

Every model pair is judged twice — once in each presentation order — as two separate calls. The
judge sees "Answer 1" and "Answer 2" and never a model name. If the two orderings disagree the pair
is recorded as a tie, and the **order-flip rate** is printed: a high flip rate means the judge is
deciding on position rather than content, and the ranking built on it is not worth much.

A judge from the same provider family as any ranked model is refused outright — self-preference
bias is not something this tool can correct for. Verdicts that cannot be parsed are retried once and
then recorded in a `judge_failures` table, never coerced into a winner.

## Labelling

```bash
pip install 'konkord[label]'
konkord label --suite suites/python_codegen.yaml --n 100
```

Opens a local Streamlit app showing sampled comparisons side by side, with no model identity and no
judge verdict visible — seeing the judge's opinion first would measure suggestibility rather than
agreement. The sample is stratified across tasks and model pairs, orientation is randomised per
item, and every label is written immediately, so the session is resumable with the same `--seed`.

## Development

```bash
uv sync
uv run konkord --help
```

Lint, type-check and test — the same three commands CI runs:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

## License

MIT
