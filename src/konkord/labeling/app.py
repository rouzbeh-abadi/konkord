"""The local blind labeller.

Run through `konkord label`, which sets the environment variables below and
starts Streamlit. Not imported by anything else, since Streamlit executes this file
top to bottom on every interaction.

Two invariants make the labels worth anything, and both live in `_render`:

* **No model identity.** The operator sees "Answer 1" and "Answer 2". Nothing on
  the page names a model.
* **No judge verdict.** The judge's opinion is in the same database and is
  deliberately never read here. Seeing it first would anchor the human, and the
  agreement rate would then measure suggestibility rather than agreement.

The task prompt *is* shown, because you cannot grade an answer without the question.

Every label is written the moment it is submitted, so closing the tab loses
nothing.
"""

import os
from pathlib import Path

import streamlit as st

from konkord.labeling.sampling import (
    LabelItem,
    already_labelled,
    candidates,
    stratified_sample,
    to_verdict,
)
from konkord.models import Comparison, Suite, Verdict
from konkord.store import ResultStore
from konkord.suites import load_suite

ENV_DB = "KONKORD_DB"
ENV_SUITE_PATH = "KONKORD_SUITE_PATH"
ENV_N = "KONKORD_LABEL_N"
ENV_SEED = "KONKORD_LABEL_SEED"

_SHORTCUTS = """
<script>
const doc = window.parent.document;
doc.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase();
  const wanted = {'1': '1', '2': '2', '3': '3', 't': '3'}[key];
  if (!wanted) return;
  const button = [...doc.querySelectorAll('button')]
    .find((b) => b.innerText.trim().startsWith(wanted));
  if (button) button.click();
});
</script>
"""


def queue_for(store: ResultStore, suite: str, n: int, seed: int) -> list[LabelItem]:
    """The remaining sample: deterministic, minus anything already labelled."""
    return stratified_sample(
        candidates(store.comparisons(suite, "judge")),
        n=n,
        seed=seed,
        exclude=already_labelled(store.comparisons(suite, "human")),
    )


def record(store: ResultStore, suite: str, item: LabelItem, choice: Verdict) -> None:
    store.record_comparison(
        suite,
        Comparison(
            task_id=item.task_id,
            model_a=item.model_a,
            model_b=item.model_b,
            order=item.order,
            winner=choice,
            source="human",
        ),
    )


def _answer(suite: Suite, text: str) -> None:
    """Show one answer the way its suite says answers should be read.

    Code wants a monospace block with highlighting; prose read as code is
    punishing to label a hundred of. `st.text` rather than `st.markdown` for the
    prose case: model output is full of characters markdown would eat, and a
    labeller has to see what the model actually wrote.
    """
    if suite.answer_language:
        st.code(text, language=suite.answer_language)
    else:
        st.text(text)


def _render(suite: Suite, store: ResultStore, total: int, seed: int) -> None:
    queue = queue_for(store, suite.name, total, seed)
    labelled = len(store.comparisons(suite.name, "human"))

    st.caption(f"{suite.name}: {labelled} labelled, {len(queue)} left in this sample")
    if not queue:
        st.success("Nothing left to label in this sample.")
        return

    item = queue[0]
    st.progress(labelled / max(labelled + len(queue), 1))

    with st.expander(f"Task: {item.task_id}", expanded=True):
        st.write(suite.task(item.task_id).prompt)

    # The judge is told what "better" means for this suite, so the labeller is
    # told the same thing. Measuring a stated standard against an unstated one
    # would report the gap between two rubrics as a fault in the judge.
    with st.expander("What counts as better here", expanded=False):
        st.text(suite.rubric)

    answers = {
        (g.task_id, g.model): g.output for g in store.generations(suite.name) if g.error is None
    }
    left, right = st.columns(2)
    with left:
        st.subheader("Answer 1")
        _answer(suite, answers.get((item.task_id, item.first_model), ""))
    with right:
        st.subheader("Answer 2")
        _answer(suite, answers.get((item.task_id, item.second_model), ""))

    one, two, three = st.columns(3)
    choice: str | None = None
    if one.button("1. Answer 1 is better", use_container_width=True):
        choice = "first"
    if two.button("2. Answer 2 is better", use_container_width=True):
        choice = "second"
    if three.button("3. Tie", use_container_width=True):
        choice = "tie"

    st.components.v1.html(_SHORTCUTS, height=0)
    st.caption("Keys: 1 / 2 / 3 (T also ties)")

    if choice is not None:
        record(store, suite.name, item, to_verdict(item, choice))
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Konkord blind labelling", layout="wide")
    suite = load_suite(Path(os.environ[ENV_SUITE_PATH]))
    with ResultStore(Path(os.environ.get(ENV_DB, "konkord.duckdb"))) as store:
        _render(
            suite,
            store,
            total=int(os.environ.get(ENV_N, "100")),
            seed=int(os.environ.get(ENV_SEED, "0")),
        )


# Guarded so importing this module is side-effect free. Streamlit executes the
# script as `__main__`, so the app still starts when run properly.
if __name__ == "__main__":
    main()
