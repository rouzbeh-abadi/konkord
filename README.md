# Konkord

Rank LLMs on a task suite, and measure whether the automated judge can be trusted.

The usual pipeline (run models, have an LLM grade the outputs, publish a leaderboard) is commodity.
Konkord adds one step: the operator hand-labels a sample of comparisons **blind**, and every
published ranking ships with the judge-versus-human agreement rate attached.

## What the first run found

Three models answered 25 hand-written Python tasks. A fourth model judged every pair blind, in both
presentation orders. Then a human labelled 75 of those comparisons blind, seeing neither the model
names nor the judge's opinion.

**The judge agreed with the human on 61.3% of them.** That is 46 of 75, 95% CI 50.0% to 71.5%.
Cohen's kappa was 0.370, where 0 is chance and 1 is perfect.

Split by matchup, the number moves a long way:

| Matchup | Agreement |
|---|---|
| gpt-5.4 vs gemini-2.5-flash-lite | 76.0% (19/25) |
| claude-haiku-4.5 vs gemini-2.5-flash-lite | 56.0% (14/25) |
| gpt-5.4 vs claude-haiku-4.5 | 52.0% (13/25) |

The judge could tell a strong model from a weak one. Between the two strong models it was at chance:
52% on a two-way call is a coin flip.

None of that is visible on the leaderboard itself. The ranking separates all three models with
non-overlapping bootstrap intervals, and it is drawn correctly. But a confidence interval only
measures how much the judge's verdicts would wobble under resampling. It says nothing about whether
those verdicts are right. Judging more pairs would have tightened the interval around the same
error.

Two tasks reached 0% agreement, meaning the judge and the human disagreed on every labelled
comparison for them. The judge also changed its answer on 20% of pairs when the two answers were
swapped, deciding on position rather than content. Answer length showed no clean gradient, so this
run gives no evidence of verbosity bias either way.

The ranking, the judge prompt, every model's answer, and all 150 judged pairs with rationales:
**[konkord.deadpixelstudio.io](https://konkord.deadpixelstudio.io)**.

## Quickstart

Not on PyPI yet, so clone it:

```bash
git clone https://github.com/rouzbeh-abadi/konkord
cd konkord
uv sync --all-extras
```

Set credentials for whichever providers you are ranking. Copy the template and fill in the keys
you actually have:

```bash
cp .env.example .env
```

`.env` is gitignored and read at startup. Anything already exported in your shell wins over it, so
a one-off `OPENAI_API_KEY=... konkord run` still does what it looks like. Plain exports work too if
you would rather not keep a file.

**[OpenRouter](https://openrouter.ai) reaches many vendors with one key.** Set `OPENROUTER_API_KEY`
and name models as `openrouter/anthropic/claude-opus-5`. Konkord resolves the vendor *behind* the
router, so an OpenRouter judge is still refused when it shares a provider family with a model being
ranked. Reaching the same model through a router does not make it a different family.

Set a spending cap on each key in the provider's dashboard. Storage hygiene stops a key leaking; a
cap stops a leak or a runaway loop from being expensive.

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

`run` and `judge` cost money and take a few minutes. The run above came to well under a dollar for
75 answers and 150 judgements. Both stages are resumable and cache every response, so re-running
them is free. `label` opens a local app and is the part that needs you: budget an hour or two for
75 to 100 comparisons. `calibrate` has nothing to say until those labels exist.

## Commands

| Command | Does |
|---|---|
| `konkord run` | Generate one output per (task x model) |
| `konkord judge` | Pairwise LLM-as-judge, both orderings |
| `konkord label` | Local blind labeller |
| `konkord calibrate` | Judge-versus-human agreement and kappa |
| `konkord report` | Aggregate into `results.json` |

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

A judge from the same provider family as any ranked model is refused outright, because
self-preference bias is not something this tool can correct for. Routing prefixes are peeled off
first, so `openrouter/anthropic/claude-opus-5`, `vertex_ai/claude-opus-5` and a bare
`claude-opus-5` all count as the same family. Verdicts that cannot be parsed are retried once and
then recorded in a `judge_failures` table, never coerced into a winner.

Give the judge room to answer. The default output cap is 8192 tokens because a reasoning judge on a
tight cap spends the whole budget thinking and returns nothing: at 1024 tokens, a test run lost 72%
of its verdicts to truncation. A verdict lost that way is recorded with the cap named as the reason.

## Labelling

```bash
konkord label --suite suites/python_codegen.yaml --n 100
```

Opens a local Streamlit app showing sampled comparisons side by side, with no model identity and no
judge verdict visible. Seeing the judge's opinion first would measure suggestibility rather than
agreement. The sample is stratified across tasks and model pairs, orientation is randomised per
item, and every label is written immediately, so the session is resumable with the same `--seed`.

This is the slow part and there is no way around it. The whole claim rests on these labels being
yours.

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

## Site

`site/` is a static page set: a leaderboard, a methodology page, and a per-task browser, published
at [konkord.deadpixelstudio.io](https://konkord.deadpixelstudio.io). It reads `results.json` from
its own directory and computes nothing at load time, so there is no build step and no server.
`wrangler.jsonc` holds the deployment config; every push to `main` redeploys.

```bash
konkord report --suite suites/python_codegen.yaml --out site/results.json
python3 -m http.server 8787 --directory site
```

Without that file the pages render an empty state rather than a broken one. With a report that has
no human labels, the leaderboard states plainly that it is uncalibrated instead of quietly showing a
ranking. The methodology page publishes the judge prompt verbatim, and a test fails if that copy
drifts from the prompt the tool actually sends.

## What this does not do

Nothing here is executed. There is no sandbox, so no answer is compiled, linted or run, and every
judgement on a published page is a reading of the code rather than a test of it. Suites can declare
deterministic checks and the loader validates them, but no runner consumes them yet.

## Development

```bash
uv sync
uv run konkord --help
```

Lint, type-check and test, the same four commands CI runs:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

[CONTRIBUTING.md](CONTRIBUTING.md) covers the bar for a change and how to add a task to a suite.

## License

MIT. Built by [Rouzbeh Abadi](https://github.com/rouzbeh-abadi).
</content>
</invoke>
