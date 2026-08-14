# Konkord

Rank LLMs on a task suite, and measure whether the automated judge can be trusted.

The usual pipeline (run models, have an LLM grade the outputs, publish a leaderboard) is commodity.
Konkord adds one step: the operator hand-labels a sample of comparisons **blind**, and every
published ranking ships with the judge-versus-human agreement rate attached.

Why it is worth the extra step: a leaderboard's confidence intervals measure how much the judge's
verdicts wobble under resampling, not whether those verdicts are right. Judging more pairs only
tightens the interval around the same error. Two published runs, same three models and the same
judge, show what that hides:

| | Support replies | Python codegen |
|---|---|---|
| Agreement with a human | 46.7% | 61.3% |
| ...on pairs the judge answered consistently | **81.4%** | **76.7%** |
| Order-flip rate | 42.7% | 20.0% |

Read the headline alone and the judge looks worse at prose. It isn't. When it commits to a winner it
agrees slightly *more* on prose; the gap is entirely that it contradicts itself twice as often, and a
pair it contradicts itself on is recorded as no winner at all. A harness that judged each pair once
would never see this. It would take whichever verdict arrived first on those flipped pairs and
publish a confident ranking built on coin flips.

Both runs, the judge prompt each was produced under, every answer and every verdict:
**[konkord.deadpixelstudio.io](https://konkord.deadpixelstudio.io)**.

## Any domain, one standard at a time

A suite says what "better" means. Code, summarisation, translation, extraction: write the criteria
into the file and every stage follows.

```yaml
name: support_replies
rubric: |
  1. Accuracy. Does the reply answer what was actually asked, without inventing
     policy the ticket does not support?
  2. Tone. Given equal accuracy, prefer the reply a frustrated customer would
     rather receive.
answer_language: null      # prose, so render it as prose rather than as code
tasks:
  - id: refund-window-01
    prompt: |
      A customer writes in 40 days after purchase asking for a refund ...
```

What a suite cannot change is how a verdict comes back. The response format, and the instruction to
ignore length and never reward verbosity, live in the tool's own frame. A rubric that tried to
redefine them would break verdict parsing and delete the bias controls in the same move, so one
mentioning the verdict token is refused outright.

**The rubric is the only field with no default.** Every number this tool produces means whatever the
rubric said, and a suite that inherited one silently would be graded against a standard its author
never read. So it has to be stated.

Three things follow from that, and they are the reason this is more than a settings field:

- **The judge and the human are shown the same criteria.** The labeller displays the rubric. Measuring
  a stated standard against an unstated one reports the gap between two rubrics as a fault in the judge.
- **Every verdict records the prompt that produced it.** Editing a rubric and re-judging half a suite
  is refused rather than averaged, because one rating fitted across two standards means neither.
- **The site publishes the prompt that ran**, read back out of the verdicts rather than recomposed from
  the suite file. A published prompt that has drifted from the one actually sent is worse than none.

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
NAME=python_codegen
MODELS=gpt-5,claude-opus-5,gemini-2.5-pro

uv run konkord run       --suite $SUITE --models $MODELS
uv run konkord judge     --suite $SUITE --models $MODELS --judge mistral/mistral-large-latest
uv run konkord label     --suite $SUITE --n 100
uv run konkord calibrate --suite $SUITE
uv run konkord report    --suite $SUITE --out site/results.$NAME.json
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
| `konkord report` | Aggregate into one results file |

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
konkord report --suite suites/python_codegen.yaml --out site/results.python_codegen.json
```

`calibrate` joins the human labels to the judge verdicts on the same comparisons and reports raw
agreement, Cohen's kappa, breakdowns by task, by model pair and by answer-length quartile, plus a
failure gallery of every disagreement with the judge's own rationale beside it. The length-quartile
breakdown is what surfaces verbosity bias.

`report` writes a results file: Bradley-Terry ratings fitted over all pairwise comparisons rather
than raw win counts, win rates with bootstrap 95% confidence intervals, per-model cost and median
latency, and the calibration block. Models whose intervals overlap share a `rank_group` and must be
rendered as tied; presenting an order within a group asserts a difference the data does not support.

A report produced before any labelling still runs, and says plainly that it is uncalibrated.

## Site

`site/` is a static page set: a leaderboard, a methodology page, and a per-task browser, published
at [konkord.deadpixelstudio.io](https://konkord.deadpixelstudio.io). It computes nothing at load
time, so there is no build step and no server. `wrangler.jsonc` holds the deployment config, and
`.github/workflows/deploy.yml` uploads the directory whenever a push to `main` touches it.
Regenerating a report is therefore the whole publishing step.

One results file per suite, indexed by `runs.json`, which a static host cannot generate for itself:

```bash
konkord report --suite suites/python_codegen.yaml --out site/results.python_codegen.json
python3 -m http.server 8787 --directory site
```

Adding a run means adding a line to `runs.json`; a test fails if the index and the files on disk
disagree in either direction. Every page shows one run at a time, selected with `?suite=`, because
an agreement rate describes the suite and the judge that produced it and means nothing averaged
across two of them.

Without a results file the pages render an empty state rather than a broken one. With a report that
has no human labels, the leaderboard states plainly that it is uncalibrated instead of quietly
showing a ranking. The methodology page publishes the judge prompt read out of the run itself, so it
cannot show a prompt that never ran.

## What this does not do

Nothing here is executed. There is no sandbox, so no answer is compiled, linted or run, and every
judgement on a published page is a reading of the answer rather than a test of it. Suites can declare
deterministic checks and the loader carries them through, but nothing yet checks that a check name
means anything, and no runner consumes them.

Calibration is per suite and per judge, and it does not travel with the tool. The agreement rate
published here describes this suite judged by this judge. Running your own suite means labelling your
own sample; that cost is the reason the number means anything.

One labeller also cannot tell you where the ceiling is. On code, "better" is close to objective. On
softer domains two competent people may agree with each other only 70% of the time, and a judge at
65% there is near ceiling rather than broken. Reading agreement against a human-versus-human baseline
needs a second labeller, which this does not yet do.

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
