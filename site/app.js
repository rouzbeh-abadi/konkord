/**
 * Shared rendering for the three pages.
 *
 * The site is a static file set: `konkord report` produces one results file per
 * suite and nothing is computed here at load time. The one rule this file enforces is the
 * one the data encodes: models sharing a rank_group are never ordered against
 * each other, because their confidence intervals overlap and any ordering would
 * assert a difference the evidence does not support.
 */

const PERCENT = (value) => `${(value * 100).toFixed(1)}%`;

async function fetchJSON(file) {
  try {
    const response = await fetch(file, { cache: "no-cache" });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * The published runs, newest first.
 *
 * A static host cannot list a directory, so runs.json is the index. Every page
 * shows one run at a time, chosen by `?suite=`, because a calibration number
 * describes the suite and the judge that produced it and means nothing averaged
 * across two of them.
 */
async function loadRuns() {
  const doc = await fetchJSON("runs.json");
  const runs = doc && Array.isArray(doc.runs) ? doc.runs.filter((r) => r && r.file) : [];
  return runs.length ? runs : null;
}

/** Load whichever run the URL asks for, falling back to the first listed. */
async function loadCurrentRun() {
  const runs = await loadRuns();
  if (!runs) return { runs: [], run: null, data: null };
  const wanted = new URLSearchParams(window.location.search).get("suite");
  const run = runs.find((r) => r.suite === wanted) || runs[0];
  return { runs, run, data: await fetchJSON(run.file) };
}

/**
 * Tabs across the published runs.
 *
 * Real links rather than click handlers, so a particular run can be linked to
 * and opened in a new tab. Hidden when there is only one run, since a picker
 * with one option is furniture.
 */
function renderRunPicker(target, runs, current) {
  if (!target || !runs || runs.length < 2) return;
  const page = window.location.pathname.split("/").pop() || "index.html";
  const wrap = el("nav", "runs");
  wrap.setAttribute("aria-label", "Published runs");
  for (const run of runs) {
    const link = el("a", "", run.title || run.suite);
    link.href = `${page}?suite=${encodeURIComponent(run.suite)}`;
    if (run.suite === current.suite) link.setAttribute("aria-current", "page");
    if (run.blurb) link.title = run.blurb;
    wrap.append(link);
  }
  target.replaceChildren(wrap);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function noData(target, what) {
  target.replaceChildren();
  const box = el("div", "empty");
  box.append(
    `No results yet. ${what} appears once a run has been judged and `,
    Object.assign(el("code"), { textContent: "konkord report --out site/results.<suite>.json" }),
    " has written the file."
  );
  target.append(box);
}

/** The headline: what the ranking is worth, stated before the ranking itself. */
function renderCalibration(target, data) {
  const c = data.calibration;
  const box = el("div", "calibration");
  box.append(el("div", "label", "CALIBRATION"));

  const headline = el("p", "headline");
  if (c.human_labels === 0) {
    box.classList.add("warn");
    headline.append(
      "This ranking is ",
      Object.assign(el("strong"), { textContent: "uncalibrated" }),
      ". No blind human labels have been collected, so there is no evidence the judge agrees with a person."
    );
    box.append(headline);
    box.append(
      el(
        "p",
        "detail",
        "Treat the ordering below as unverified. The whole point of this project is that a leaderboard without an agreement rate is a claim with nothing behind it."
      )
    );
  } else {
    headline.append(
      "The judge agreed with a human on ",
      Object.assign(el("strong"), { textContent: PERCENT(c.agreement) }),
      ` of ${c.human_labels} blind comparisons.`
    );
    box.append(headline);
    box.append(
      el(
        "p",
        "detail",
        `Cohen's kappa ${c.kappa.toFixed(3)}, which corrects for the agreement two raters would reach by chance. ` +
          `Order-flip rate ${PERCENT(c.order_flip_rate)}: the share of pairs where the judge changed its mind when the two answers were swapped. ` +
          `Judge: ${c.judge_models.join(", ") || "unknown"}.`
      )
    );
  }
  target.append(box);
}

function intervalCell(entry) {
  const cell = el("div", "interval");
  cell.append(el("div", "track"));
  const bar = el("div", "bar");
  bar.style.left = `${entry.ci_low * 100}%`;
  bar.style.width = `${Math.max((entry.ci_high - entry.ci_low) * 100, 0.6)}%`;
  cell.append(bar);
  const point = el("div", "point");
  point.style.left = `${entry.win_rate * 100}%`;
  cell.append(point);
  cell.append(el("div", "caption", `${PERCENT(entry.ci_low)} to ${PERCENT(entry.ci_high)}`));
  return cell;
}

function renderBoard(target, data) {
  const board = el("div", "board");
  const head = el("div", "row head");
  for (const label of ["", "MODEL", "RATING", "WIN RATE, 95% INTERVAL", "COST", "MEDIAN"]) {
    head.append(el("div", "", label));
  }
  board.append(head);

  let previousGroup = null;
  let groupWrap = null;
  for (const entry of data.models) {
    if (entry.rank_group !== previousGroup) {
      groupWrap = el("div", "group");
      board.append(groupWrap);
      previousGroup = entry.rank_group;
    }
    const row = el("div", "row");
    row.append(el("div", "rank", String(entry.rank_group + 1)));
    row.append(el("div", "model", entry.model));
    row.append(el("div", "num", entry.rating.toFixed(3)));
    row.append(intervalCell(entry));
    row.append(el("div", "num", `$${entry.cost_usd.toFixed(2)}`));
    row.append(el("div", "num", `${entry.median_latency_ms} ms`));
    groupWrap.append(row);
  }

  const tied = new Set(data.models.map((m) => m.rank_group)).size !== data.models.length;
  target.append(board);
  if (tied) {
    target.append(
      el(
        "div",
        "tie-note",
        "MODELS SHARING A RANK NUMBER ARE TIED. THEIR INTERVALS OVERLAP, SO THE DATA DOES NOT SUPPORT ORDERING THEM."
      )
    );
  }
}

/**
 * The judge prompt, taken from the run rather than kept in step with it.
 *
 * `textContent` rather than innerHTML: the prompt is data from the results file,
 * and a published prompt is exactly the wrong place to start interpreting
 * markup. A run whose verdicts predate recorded prompts says so instead of
 * showing a prompt that may never have been sent.
 */
function renderJudgePrompt(target, data) {
  if (!target) return;
  if (data && data.judge_prompt) {
    target.textContent = data.judge_prompt;
    return;
  }
  target.replaceChildren(
    el(
      "span",
      "muted",
      data
        ? "These verdicts were recorded before the prompt was stored with them, so it cannot be published here."
        : "The prompt appears once a run has been judged and reported."
    )
  );
}

function renderFooter() {
  const footer = el("footer", "bottom");
  const wrap = el("div", "wrap");
  wrap.append(el("span", "", "KONKORD"));
  const links = el("span");
  const repo = el("a", "", "GITHUB");
  repo.href = "https://github.com/rouzbeh-abadi/konkord";
  const studio = el("a", "", "DEADPIXEL STUDIO");
  studio.href = "https://deadpixelstudio.io";
  links.append(repo, "  ·  ", studio);
  wrap.append(links);
  footer.append(wrap);
  document.body.append(footer);
}
