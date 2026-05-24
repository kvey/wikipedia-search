# Evaluation Report — Wikipedia Research Agent

*Compiled 2026-05-22. Sources: `JOURNAL.md`, the archived eval runs in
`eval_results/`, the retroactive entity-coverage aggregation, and the build
history across the project's working sessions.*

This document traces **how the evaluation methodology evolved**, presents the
**current eval suite and its results**, and folds in the **retroactive
`entity_coverage` aggregation** that was just produced. It's both a narrative of
the path taken and a snapshot of where the agent stands.

---

## 1. What we're evaluating

A Wikipedia research agent built **directly on the Anthropic API** (no agent
framework, per the project brief), exposing a single `search_wikipedia(query)`
tool backed by the MediaWiki API. Every run emits a structured JSON **trace**
(`traces/`) capturing the question, each model turn, every tool result, and the
final answer — which is what later made retroactive grading possible.

The interesting question was never "can it answer trivia" — it's **whether the
answer is actually grounded in what the agent retrieved**, or just recalled from
the model's own parametric memory. The whole eval methodology grew around making
that distinction legible.

---

## 2. The path we've taken

Drawn from `JOURNAL.md` and the session history, in order:

1. **Build the agent on the raw API.** Tool-calling loop for `search_wikipedia`,
   MediaWiki backend, JSON traces, a `docs/DESIGN.md`, and `.env` loading.
2. **Force tool use.** The first runs showed **Opus skipping the tool call
   entirely** and answering from its own knowledge. We forced **at least one
   tool call** so the agent behaves like a research agent, not a closed book.
3. **First evals: simple confirmations**, then **multiple models** (opus,
   sonnet, haiku) side by side.
4. **Counterfactual cases.** `false_premise` (the question asserts something
   untrue), `unanswerable` (the subject doesn't exist), `contrastive` (one detail
   changed) — to test robustness, not just recall.
5. **Visualization + archival.** matplotlib dashboards, with every run preserved
   under a `timestamp_git-hash` directory so history never clobbers.
6. **Everything passed → make it harder.** Added **multi-hop** (needs several
   queries), **ambiguous / disambiguation**, **comparison**, and **computation**
   cases.
7. **Multi-dimensional grading.** A single keyword boolean hid too much, so
   `grade()` scores four independent 0–1 dimensions — **answer**, **search**,
   **grounding** (LLM judge), **calibration** — averaged into an overall score.
8. **Manual trace review** to sanity-check the automated scores.
9. **Parallelism within a safe rate limit** — a bounded thread pool capping
   in-flight requests.
10. **Trace analysis → the grounding problem.** Measuring query lengths and
    checking whether required facts actually appeared in the retrieved text
    revealed the models **often produce correct answers without grounding them in
    search**. To quantify it, we added an **ablation**: run every case **with and
    without tools in the same run** and compare.
11. **A rate-limit wrinkle.** A small set of rate-limit failures dragged Sonnet's
    scores down — we weren't jittering retries. The harness now isolates
    transient infra errors (excluding them from stats) and uses a shared,
    jittered cooldown.
12. **`entity_coverage` (this work).** An additional grounding metric that checks
    the entities asserted in the final answer against the search results — and,
    because traces hold the complete history, **computable retroactively** over
    runs graded before the metric existed.

---

## 3. The current eval suite

- **37 cases** across **8 categories**: `factual`, `false_premise`,
  `unanswerable`, `contrastive`, `multi_hop`, `computation`, `comparison`,
  `disambiguation`.
- **Four scored dimensions** (averaged → overall; a case passes at
  `overall ≥ 0.7`):

  | Dimension | Measures | How |
  |---|---|---|
  | `answer` | Factually right? | Fraction of `must_include` / `must_include_any` / `must_exclude` constraints met. Deterministic. |
  | `search` | How much did it search? | Tool-call count vs the case's `[min, max]` window. Under- and over-searching both penalized. |
  | `grounding` | Supported by what Wikipedia returned? | **LLM judge** rates the answer's claims against the retrieved passages (0–1 + rationale). |
  | `calibration` | Qualifies when it can't find info? | `expect_refusal` cases must hedge; answerable cases penalize false abstention. Deterministic. |

- **One auxiliary metric, `entity_coverage`** — reported but **not in the overall
  mean** (see §5).
- **Ablation arms** (with-tools vs closed-book), **parallel execution** with
  rate-limit handling, full **tracing**, and **archived dashboards**.

---

## 4. Results — the eval series

Three archived runs in `eval_results/`:

### Run 1 — `2026-05-22T18-22-42` · 14 cases · pass/fail only
Early sanity check before multi-dimensional grading. **All three models passed
14/14.** This is exactly the "everything passes immediately" moment that motivated
harder cases and richer scoring.

### Run 2 — `2026-05-22T18-52-16` · 37 cases · 4 dimensions · LLM judge
| Model | Overall | answer | search | grounding | calibration | Passed |
|---|---|---|---|---|---|---|
| opus | 0.926 | 1.00 | 0.932 | **0.773** | 1.00 | 37/37 |
| sonnet | 0.947 | 0.986 | 0.986 | **0.842** | 0.973 | 36/37 |
| haiku | 0.962 | 0.973 | 1.00 | **0.903** | 0.973 | 36/37 |

**`grounding` is the weakest dimension for every model** — the headline finding.
Answers are nearly always correct, but the judge sees claims that aren't fully
supported by the retrieved text (i.e. recalled, not retrieved).

### Run 3 — `2026-05-22T19-11-51` · 37 cases · **with ablation**
| Model | Arm | Overall | answer | search | grounding |
|---|---|---|---|---|---|
| opus | tools | 0.910 | 0.959 | 0.905 | 0.801 |
| opus | **no tools** | 0.480 | **0.973** | 0.0 | 0.0 |
| sonnet | tools | 0.788 | 0.824 | 0.824 | 0.692 |
| sonnet | **no tools** | 0.486 | **0.973** | 0.0 | 0.0 |
| haiku | tools | 0.954 | 0.959 | 0.986 | 0.899 |
| haiku | **no tools** | 0.484 | **0.964** | 0.0 | 0.0 |

Two things stand out:
- **The ablation confirms the grounding problem.** Closed-book overall collapses
  to ~0.48 — but only because `search`/`grounding` go to 0 by construction. The
  **`answer` dimension barely moves (~0.97 even with no tools)**: the models
  already *knew* most of these facts. Retrieval is improving grounding, not
  raw correctness, on this dataset.
- **Sonnet's with-tools run dipped (0.788, 30/37).** This is the rate-limit /
  no-jitter episode from the journal showing up in the numbers, not a true
  capability regression — addressed since via transient-error isolation and a
  jittered cooldown.

---

## 5. The retroactive `entity_coverage` aggregation

**What it is.** A deterministic complement to the LLM grounding judge: **spaCy
NER** extracts the named entities the answer asserts (people, places, orgs,
dates, quantities) and each is checked for a literal match in the retrieved text.
An entity present in *no* passage is a mechanical hallucination signal. It's
cheap, reproducible (fixed model weights, no API call), and **kept out of the
overall mean** so it never moves existing scores.

**Why retroactive.** It needs only the answer and the retrieved text — both saved
in every trace — so `evals/regrade_traces.py` scores it over the **entire run
history** with no agent re-run and no API calls.

**Results** (archived under
`eval_results/entity_coverage_2026-05-22T19-37-10_07b1e15-dirty/`, as
`entity_coverage_retro.{png,json}`):

| Scope | Mean `entity_coverage` |
|---|---|
| **All retrieval-bearing traces (323)** | **0.808** |
| claude-haiku-4-5 (n=99) | 0.877 |
| claude-sonnet-4-6 (n=82) | 0.780 |
| claude-opus-4-7 (n=142) | 0.776 |

*110 closed-book traces are excluded — with no retrieval they score 0.0 by
construction, and folding them in (the naïve mean was 0.599) understates how
grounded the real answers were.* The distribution skews toward 1.0 with a tail of
lower-grounded answers — that tail is where to look (`--min-score 1.0`).

**Read the per-model numbers carefully.** Haiku scoring *highest* is most likely
a **verbosity effect**: shorter answers assert fewer entities, so fewer chances to
include an ungrounded one. So this is "of the entities each model chose to state,
what fraction were in the sources" — a *precision* signal that pairs with the LLM
judge, not a standalone verdict on quality.

**Design decisions taken** (recorded for posterity):
- **Standalone, not in the mean** — so "don't change the existing metrics" holds
  literally; re-running an old eval reproduces identical dimension/overall scores.
- **Extractor: regex → spaCy.** The first cut used hand-rolled regex; it needed
  constant tuning (sentence-boundary merges, citation-footer words). Switched to
  spaCy NER for reliability while staying deterministic and API-free.
- **Closed-book correction** to the headline mean (above).

**Known limitations.**
- Literal matching is blind to **paraphrase/aliases** ("US" vs "United States")
  and **number formatting** ("300,000 km/s" vs "299,792 km/s") — by design, a
  precision signal rather than a verdict.
- The three existing **per-run dashboards can't be honestly backfilled**:
  `results.json` stores each answer but **not the retrieved text**, and has no
  link back to a trace, so case→trace mapping would be guesswork. The retroactive
  chart is therefore rendered straight from the self-contained traces. *(A clean
  future fix: have the harness write a `run_id`/trace link into `results.json`.)*

---

## 6. What the evals revealed

- **Grounding is the consistent weak point** across models — answers are right,
  but not always traceable to the sources. This is the main quality gap the suite
  surfaces.
- **Ablation shows retrieval lifts grounding, not raw correctness** on this
  dataset: the models already know most of these facts.
- **Two complementary grounding lenses now exist** — the LLM judge (holistic
  claim support) and `entity_coverage` (mechanical, auditable, retroactive).
- **Infra matters for fair comparison**: untreated rate limits made a model look
  worse than it is; transient errors are now isolated from the stats.

---

## 7. Reproducing

```bash
# Full eval sweep (3 models, 4 dimensions, LLM grounding judge), archived + charted
uv run evals/run_evals.py

# With the closed-book ablation arm
uv run evals/run_evals.py --ablation

# Cheap variant: deterministic grounding proxy, no judge API calls
uv run evals/run_evals.py --no-grounding

# Retroactive entity_coverage over all saved traces, with a chart.
# Write each aggregation into its own timestamped dir so runs aren't overwritten:
out="eval_results/entity_coverage_$(date +%Y-%m-%dT%H-%M-%S)" && mkdir -p "$out"
uv run evals/regrade_traces.py --chart "$out/entity_coverage_retro.png" \
                               --json  "$out/entity_coverage_retro.json"
```

See `docs/DESIGN.md` for the grading internals and `README.md` for setup.

---

## 8. Post-tool-call improvement — query fan-out + per-article detail fetch

§6 named **grounding** as the consistent weak point. This iteration acts on it by
changing the *tool surface* the model calls, then re-measures with the same suite,
judge, and threshold so the before/after is honest.

### What changed (and why, from the traces)

Mining the 433 archived traces surfaced two mechanical causes of un-grounded
answers:

1. **One query, one shot.** `search_wikipedia` took a single `query`. When it
   missed, the model re-queried *sequentially* across turns. *"Who won the Nobel
   Prize in Mathematics in 2000?"* never converged and **hit `MAX_TURNS` with no
   answer** (twice).
2. **Thin extracts.** Only the top hit got a lead extract; hits #2–4 were
   snippet-only. *"longest river on the continent where the Sahara Desert is
   located?"* had the answer (the Nile) sitting in a non-top hit's snippet but
   never in the extracted text, forcing an extra turn.

Two changes, kept inside the tool layer (model-agnostic, so they apply across all
three models):

- **`search_wikipedia(queries[])` — query fan-out.** Accepts several phrasings in
  one call, runs each, and merges/dedupes hits by title (best rank wins), now
  extracting the **top 2** merged articles instead of only #1.
- **`get_article(title, section?)` — per-article detail fetch.** Pulls the fuller
  article body (or a named section) when a snippet looks relevant but the lead
  extract lacked the specific fact.

The `search` dimension was redefined to count **search *steps*** (`search_wikipedia`
calls), not raw tool calls: a fan-out call is one step and `get_article` is
excluded — so multi-hop cases still demand genuine chaining and fan-out width is
never penalized. `get_article` output still flows into the grounding judge and
`entity_coverage`.

### Results — before vs after

Baseline = Run 2 (§4, 37 cases, 4 dimensions, LLM judge `claude-sonnet-4-6`).
After = archived run `eval_results/2026-05-22T20-06-24_073937c-dirty/`, identical
suite/judge/threshold.

**`grounding` (the targeted dimension):**

| Model | before | after | Δ |
|---|---|---|---|
| opus | 0.773 | **0.903** | **+0.130** |
| sonnet | 0.842 | **0.884** | +0.042 |
| haiku | 0.903 | **0.953** | +0.050 |

Grounding rose for **every model**, most for **opus** — which was the weakest and
the model that "reaches for tools less often." It's now the largest single
improvement to the headline weak spot.

**All dimensions:**

| Model | overall | answer | search | grounding | calibration | Passed |
|---|---|---|---|---|---|---|
| opus | 0.926 → **0.952** | 1.00 → 1.00 | 0.932 → 0.932 | 0.773 → **0.903** | 1.00 → 0.973 | 37/37 → **37/37** |
| sonnet | 0.947 → 0.941 | 0.986 → 0.986 | 0.986 → 0.919 | 0.842 → **0.884** | 0.973 → 0.973 | 36/37 → **37/37** |
| haiku | 0.962 → **0.981** | 0.973 → 0.986 | 1.00 → 0.986 | 0.903 → **0.953** | 0.973 → 1.00 | 36/37 → **37/37** |

- **All three models now pass 37/37** (sonnet and haiku each cleared a prior miss).
- **No `MAX_TURNS` failures** — the Nobel-Mathematics dead-end now resolves in a
  single 4-phrasing fan-out.
- **`search` dips for sonnet/haiku** by design: a well-phrased fan-out resolves
  some `multi_hop` cases in one step, scoring `search = 0.5` (it didn't take two
  steps) while still passing on strong grounding. This is the deliberate "fan-out
  isn't multi-hop chaining" trade-off, not a regression in search behavior.

### `entity_coverage` — and a metric fix it forced

Re-running the retroactive aggregation on the new traces exposed a flaw in the
metric itself: the **single most-flagged "uncovered entity" was the literal token
`Wikipedia`** (111 of 116 traces). That's the citation footer ("Sources: Wikipedia
articles …"), not a hallucination — and "Wikipedia" never appears in the retrieved
article *text*, so it was counted as ungrounded every time, depressing the score
with pure boilerplate.

Fix (in `grade_entity_coverage`): strip a trailing `Sources:/Citations:/References:`
footer and stoplist the token `Wikipedia` before NER. The narrow footer regex
leaves prose like "Sources of the Nile are…" untouched. After the fix, **zero**
traces flag `Wikipedia`, and the remaining sub-1.0 flags are genuine — date/number
formatting (`January 1643`, `7,088 kilometers` vs `7,088 km`) and true recall (the
Nobel answer naming 2000 Fields medalists never retrieved).

Effect of the fix alone, on the same new traces: mean **0.684 → 0.802** (+0.118).

Apples-to-apples with the *fixed* metric (pre-change traces vs post-change traces,
retrieval-bearing only) isolates the **agent** change:

| Model | old agent | new agent | Δ |
|---|---|---|---|
| opus | 0.712 | 0.740 | +0.028 |
| sonnet | 0.758 | 0.825 | +0.067 |
| haiku | 0.858 | 0.851 | −0.007 |
| **overall** | **0.769** (n=323) | **0.802** (n=116) | **+0.033** |

A modest lift for opus/sonnet and flat for haiku (already high, and the verbosity
effect from §5 still applies). The bigger story here is the **metric fix**: the
`Wikipedia` boilerplate was masking the true grounding level.

> Note: the archived sweep dashboard shows the *pre-fix* `entity_coverage`
> (0.63/0.71/0.73); the numbers above are the same traces re-scored with the fixed
> metric.

### Net read

The change does what §6 asked for: it **lifts grounding across all models** (the
named weak point), eliminates the only hard failures (`MAX_TURNS`), and takes every
model to a clean 37/37 — at the cost of a small, intentional `search`-dimension dip
where fan-out short-circuits a multi-hop chain. `answer` is unchanged (the models
already knew the facts; §4's ablation finding still holds — this improves
*grounding*, not raw correctness). Caveats: a single post-change sweep carries LLM-
judge variance, and `search`'s new meaning makes its column not directly comparable
to pre-change runs.
