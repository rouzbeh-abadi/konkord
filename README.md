# Konkord

Rank LLMs on a task suite, and measure whether the automated judge can be trusted.

The usual pipeline — run models, have an LLM grade the outputs, publish a leaderboard — is
commodity. Konkord adds one step: the operator hand-labels a sample of comparisons **blind**, and
every published ranking ships with the judge-versus-human agreement rate attached.

If the judge agrees with the human 85% of the time, the leaderboard is credible. If it agrees 55%
of the time, that is itself the finding.

> **Status: pre-alpha.** The command surface below is fixed, but only the scaffolding is
> implemented. Unimplemented commands exit non-zero and say which build phase they belong to.

## Commands

| Command | Does | Phase |
|---|---|---|
| `konkord run` | Generate one output per (task × model) | 3 |
| `konkord check` | Deterministic graders in a sandbox | 4 |
| `konkord judge` | Pairwise LLM-as-judge, both orderings | 5 |
| `konkord label` | Local blind labeller | 6 |
| `konkord calibrate` | Judge-versus-human agreement and kappa | 7 |
| `konkord report` | Aggregate into `results.json` | 7 |

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
