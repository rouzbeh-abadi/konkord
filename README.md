# Konkord

Rank LLMs on a task suite, and measure whether the automated judge can be trusted.

The usual pipeline — run models, have an LLM grade the outputs, publish a leaderboard — is
commodity. Konkord adds one step: the operator hand-labels a sample of comparisons **blind**, and
every published ranking ships with the judge-versus-human agreement rate attached.

If the judge agrees with the human 85% of the time, the leaderboard is credible. If it agrees 55%
of the time, that is itself the finding.

> **Status: pre-alpha.** The command surface below is fixed. `run` works; the rest exit
> non-zero and say which build phase they belong to.

## Commands

| Command | Does | Status |
|---|---|---|
| `konkord run` | Generate one output per (task × model) | ✅ |
| `konkord check` | Deterministic graders in a sandbox | phase 4 |
| `konkord judge` | Pairwise LLM-as-judge, both orderings | phase 5 |
| `konkord label` | Local blind labeller | phase 6 |
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
