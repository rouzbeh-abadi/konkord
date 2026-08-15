# Konkord

A pairwise LLM-as-judge evaluation harness that measures the reliability of its own judge.

Ranking language models usually works like this: run every model over a set of tasks, ask a second
model to decide which of two answers is better, count the wins, publish a leaderboard. The weak point
is the middle step. That judge is itself a language model with its own preferences, and nothing in
the pipeline establishes whether its verdicts match what a competent person would have said. The
leaderboard inherits that uncertainty and reports none of it.

Konkord runs the same pipeline and adds one measurement. The operator hand-labels a sample of the
same comparisons the judge scored, without seeing which model wrote which answer or what the judge
concluded. How often the two agree, corrected for chance with Cohen's kappa, is computed and shipped
in the same file as the ranking.

**That agreement rate is the product.** A high one means the ordering is worth reading. A low one
does not invalidate the run: it is the result, and it is information the usual pipeline cannot
produce at all, because a judge scored against nothing external has no way to be found wrong. Either
way the reader is told what the ranking is worth rather than left to assume it.

The same reasoning applies to the judge's other failure modes, so each one is measured rather than
hoped away: whether it prefers whichever answer it sees first, whether it favours its own vendor,
whether it was even asked the same question throughout. Those are the controls listed below, and what
they produce is published with every run.

## Pipeline

| Stage | Command | Output |
|---|---|---|
| Generation | `konkord run` | one response per (task, model), cached and resumable |
| Judging | `konkord judge` | a pairwise verdict per (task, model pair, presentation order) |
| Labelling | `konkord label` | human verdicts on a stratified sample of the same pairs |
| Calibration | `konkord calibrate` | judge-versus-human agreement and Cohen's kappa |
| Reporting | `konkord report` | Bradley-Terry ratings, bootstrap intervals, calibration block |

Ratings are fitted with Bradley-Terry over all pairwise outcomes rather than counted as raw wins, so
a model is not rewarded for an easy schedule. Win rates carry 95% percentile bootstrap intervals over
a seeded resample. Models with overlapping intervals share a `rank_group` and are rendered as tied.

## What it adds over a standard judge harness

A bootstrap interval quantifies sampling variance in the judge's verdicts. It does not quantify
whether those verdicts are correct. Nothing internal to the judge can establish that, so the judge is
measured against an external reference and the result is published with the ranking:

- **Human calibration.** The operator labels a sample of the same comparisons with model identity and
  the judge's verdict both hidden. Reported as raw agreement and Cohen's kappa, which corrects for the
  agreement two raters reach by chance. Seeing the judge's verdict first would measure suggestibility,
  so it is never shown.
- **Position-bias control.** Every pair is judged twice, once in each presentation order, as two
  independent calls. Disagreement between the two orderings is position bias, not an opinion: the pair
  resolves to a tie rather than to whichever call returned first, and the **order-flip rate** is
  published. A harness that judges each pair once cannot detect this and will rank on it regardless.
- **Provider-family independence.** A judge sharing a provider family with any ranked model is refused
  rather than adjusted for. Routing prefixes are resolved first, so a model reached through an
  aggregator is not mistaken for a different vendor.
- **Recorded rubric.** The suite supplies the judging criteria; the tool owns the response format and
  the instruction to ignore length. The composed prompt is stored with every verdict, so a rubric
  changed mid-suite is refused instead of averaged, and the published prompt is the one that ran.
- **No silent coercion.** A response with no parseable verdict is retried once, then recorded in a
  `judge_failures` table with the raw text. It is never resolved into a winner.

## Published runs

Two calibrated runs at **[konkord.deadpixelstudio.io](https://konkord.deadpixelstudio.io)**: Python
code generation and customer support replies, over the same three models with the same judge. Each
carries its agreement rate, kappa, order-flip rate, the judge prompt it was produced under, and every
answer and rationale behind it.

Their headline agreement rates differ substantially, and the difference is accounted for entirely by
the order-flip rate rather than by the judge choosing wrongly: restricted to pairs where the judge
gave the same verdict in both orderings, agreement is comparable across the two domains and higher
than either headline. The flip rate is the reliability signal; the headline conflates it with
accuracy.

## Suites and rubrics

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

## Install

```bash
pip install 'konkord[label]'
```

Python 3.12 or newer. The `label` extra pulls in Streamlit, which only the labeller needs; plain
`pip install konkord` gives you everything except that one command.

This installs the harness. Task suites are yours to write, and the two in this repository are worked
examples rather than a bundled benchmark, so clone it if you would rather start from one than from a
blank file:

```bash
git clone https://github.com/rouzbeh-abadi/konkord
cd konkord
uv sync --all-extras
```

## Quickstart

Set credentials for whichever providers you are ranking, as ordinary environment variables:

```bash
export OPENROUTER_API_KEY=...
```

A `.env` file in the working directory is read at startup if there is one, and
[.env.example](https://github.com/rouzbeh-abadi/konkord/blob/main/.env.example) lists the names
Konkord looks for. Anything already exported in your shell wins over the file, so a one-off
`OPENAI_API_KEY=... konkord run` does what it looks like.

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

konkord run       --suite $SUITE --models $MODELS
konkord judge     --suite $SUITE --models $MODELS --judge mistral/mistral-large-latest
konkord label     --suite $SUITE --n 100
konkord calibrate --suite $SUITE
konkord report    --suite $SUITE --out results.$NAME.json
```

`run` and `judge` cost money and take a few minutes. The run above came to well under a dollar for
75 answers and 150 judgements. Both stages are resumable and cache every response, so re-running
them is free. `label` opens a local app and is the part that needs you: budget an hour or two for
75 to 100 comparisons. `calibrate` has nothing to say until those labels exist.

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
konkord report --suite suites/python_codegen.yaml --out results.python_codegen.json
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

[CONTRIBUTING.md](https://github.com/rouzbeh-abadi/konkord/blob/main/CONTRIBUTING.md) covers the bar for a change and how to add a task to a suite.

## License

MIT. Built by [Rouzbeh Abadi](https://github.com/rouzbeh-abadi).
