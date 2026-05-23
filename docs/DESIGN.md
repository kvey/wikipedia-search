# Design: Wikipedia Agent & Eval Harness

## 1. Goal & constraints

Build a small agent that answers factual questions by searching Wikipedia, plus
an eval harness to measure how well it does.

Hard constraints from the brief:

- **Use the Anthropic API directly** — no agent framework (LangChain, etc.).
- **Tool calling** for a single tool: `search_wikipedia(query: str)`.
- **Multi-turn** tool invocation — the agent can call the tool repeatedly within
  one question.
- **Python**, run with **uv**, API key supplied via env var.
- Separate the **entrypoint**, **Wikipedia wrapper**, and **prompt** into their
  own files; a separate **evals** directory with an entrypoint that invokes the
  agent.

These constraints favor a thin, legible implementation over a feature-rich one.
The design optimizes for *readability and testability* — someone reviewing it
should be able to trace the entire control flow in a few minutes.

## 2. Architecture

```
                    question (str)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │  agent/main.py : run_agent()      │
        │  manual multi-turn tool-use loop  │
        └──────────────────────────────────┘
            │  Messages API          ▲ tool_result
            ▼  (tools=[search_…])    │
        ┌───────────────┐     ┌──────────────────────┐
        │  Anthropic API │     │ agent/wikipedia.py    │
        │  (Claude)      │     │ search_wikipedia()    │
        └───────────────┘     │  → MediaWiki API      │
                              └──────────────────────┘

  agent/prompt.py  — system prompt, model id, tool schema (data, no logic)
  agent/trace.py   — per-run JSON trace, written to traces/ (one file per run)
  evals/run_evals.py — imports run_agent(), grades answers, prints report
```

### Module responsibilities

| File | Responsibility | Why separate |
|---|---|---|
| `agent/prompt.py` | System prompt, model id, JSON tool schema | Prompt is the highest-churn artifact; isolating it lets us tune wording and the tool surface without touching control flow, and lets evals import the same constants. |
| `agent/wikipedia.py` | MediaWiki API calls behind `search_wikipedia(query)` | The only side-effecting I/O. Pure-ish function with no Anthropic dependency, so it can be tested standalone (and was — it needs no API key). |
| `agent/main.py` | The agentic loop (`run_agent`) + CLI | Orchestration only. Exposes a library function so callers (the CLI, the evals) reuse one code path. |
| `agent/trace.py` | `RunTrace` — collects events and writes one JSON file per run | Observability isolated from orchestration; the loop calls `record_turn` / `record_tool_result` / `write` and stays readable. |
| `agent/env.py` | Loads `<repo>/.env` (git-ignored) at agent import | Keeps `ANTHROPIC_API_KEY` out of shell history and source control; real env vars still override (`override=False`). |
| `evals/cases.py` | The dataset | Data, not code — easy to extend. |
| `evals/grading.py` | The four scoring dimensions + the LLM grounding judge | Grading is the highest-churn part of an eval; isolating it keeps the scoring rubric in one place and `run_evals.py` focused on orchestration and reporting. |
| `evals/run_evals.py` | Run cases through `run_agent`, grade, report | The eval entrypoint the brief asked for. |

## 3. The agent loop

`run_agent` is a standard manual agentic loop (`agent/main.py`):

1. Seed `messages` with the user question.
2. Call `client.messages.create(...)` with the system prompt and `tools`.
3. Append the assistant turn (including any `tool_use` blocks) to `messages`.
4. If `stop_reason != "tool_use"`, extract the text and return.
5. Otherwise, run every requested tool, append a `user` turn carrying one
   `tool_result` per `tool_use` (matched by `tool_use_id`), and loop.

A `MAX_TURNS` bound (10) guarantees termination even if the model loops.

**Forcing the first search.** Opus 4.7 reaches for tools less often than prior
models and will answer confident questions ("capital of Australia") from its own
knowledge without searching. For a Wikipedia-*grounded* agent that defeats the
purpose, so the first turn passes
`tool_choice={"type": "tool", "name": "search_wikipedia"}`, guaranteeing at least
one search. Subsequent turns use the default auto choice so the model can stop
and answer once it has what it needs. This is controlled by `force_first_tool`
(default `True`) and is why `expect_tool_use` in the evals is a meaningful,
satisfiable assertion.

**Why a manual loop instead of the SDK tool runner?** The SDK's beta tool runner
would remove this boilerplate, but the manual loop is the right call here:

- It is the explicit, framework-free implementation the brief asks for.
- It gives the eval harness a natural seam to capture a **trace** of every tool
  call (name, input, output) — used both for grading ("did it actually search?")
  and for debugging.
- It is the pattern most worth understanding; nothing is hidden.

**Return shape.** `run_agent` returns an `AgentResult` dataclass
(`answer`, `tool_calls`, `turns`) rather than a bare string. The trace is what
makes the agent *evaluable* — the harness asserts on tool use, not just text.

### Tool dispatch

`TOOL_IMPLEMENTATIONS` maps the schema's tool name to its Python function and is
called with `**tool_input`. Adding a second tool is: write the function, add a
schema entry in `prompt.py`, add one line to the map. Tool exceptions are caught
and returned to the model as an error string (with `is_error` semantics in
spirit), so a transient failure becomes something Claude can react to rather than
a crash.

### Tracing

Every run is captured as a single JSON file in `traces/` (override with
`AGENT_TRACE_DIR`). `agent/trace.py` defines `RunTrace`, which the loop feeds via
`record_turn` (per model turn: text, tool calls, token usage) and
`record_tool_result` (per executed tool). On completion it writes
`<timestamp>-<run_id>.json` with the question, final answer, stop reason, per-turn
detail, aggregate token usage, and elapsed time.

Keeping this in its own module means the loop gains full observability with three
call sites and no inline serialization logic. The CLI also prints tool calls,
their arguments, and each tool's result to stderr as the run progresses, so the
terminal shows the agent's reasoning trail while the JSON trace persists it.

## 4. The Wikipedia wrapper

`search_wikipedia` makes two MediaWiki API calls:

1. `list=search` → top *N* article titles + snippets.
2. `prop=extracts` (`exintro`, `explaintext`) → a plain-text lead extract of the
   best match.

It returns a single formatted string combining both. **Rationale:** returning
the lead extract alongside the hit list means the common case ("what is X?") is
answerable from a *single* tool call, while the snippet list still supports
follow-up searches for multi-hop questions. This trades a slightly larger tool
result for fewer round trips.

Implementation notes: a shared `requests.Session` with a descriptive
`User-Agent` (MediaWiki etiquette), a timeout on every call, and graceful
degradation — network errors and empty results return explanatory strings rather
than raising, keeping the agent loop robust.

## 5. The eval harness

`evals/run_evals.py` imports `run_agent` directly (in-process) rather than
shelling out to the CLI — it's faster, lets the harness reuse one Anthropic
client across cases, and gives structured access to the tool-call trace.

**Grading is multi-dimensional** (`evals/grading.py`). A single "did the answer
contain the right keywords" boolean hides most of what makes a *grounded research
agent* good or bad — an answer can be correct but pulled from the model's own
memory, or it can search wastefully, or it can confidently fabricate when it
should have said "I can't find this." So `grade()` scores each answer along four
independent dimensions, each **0-1**, and averages them into an overall score:

| Dimension | What it measures | How it's scored |
|---|---|---|
| `answer` | Is the answer factually right? | Fraction of the case's `must_include` / `must_include_any` / `must_exclude` keyword constraints satisfied. Deterministic. |
| `search` | *How much did it search?* | Whether the tool-call count lands in the case's `[min_searches, max_searches]` window. Under-searching (answering from memory) scales toward 0; wasteful over-searching is penalized more gently. |
| `grounding` | *How well is the answer supported by what Wikipedia returned?* | An **LLM judge** rates the answer's claims against the concatenated retrieved passages (0-1 + a one-sentence rationale). |
| `calibration` | *Does it qualify when it can't find information?* | Cases that set `expect_refusal` must hedge ("no such element", "could not find …"); answerable cases instead penalize a *false* abstention. Deterministic (a hedging lexicon). |

A case **passes** when its overall mean meets `--threshold` (default 0.7).

One **auxiliary** metric rides alongside but is deliberately **excluded from the
overall mean** (kept in `AUX_METRICS`, not `DIMENSIONS`), so it never moves any
existing score:

| Metric | What it measures | How it's scored |
|---|---|---|
| `entity_coverage` *(aux)* | *Do the entities the answer asserts appear in the sources?* | spaCy NER extracts the answer's named entities (people, places, orgs, dates, quantities); the score is the fraction that literally appear in the retrieved text. Deterministic (fixed model weights, no API call). |

It's a cheap, reproducible complement to the LLM `grounding` judge: where the
judge reasons holistically about claim support, `entity_coverage` is a blunt,
auditable check for entities pulled from parametric memory that never appeared in
any retrieved passage. Its trade-off is the mirror of that bluntness — literal
matching is blind to paraphrase and aliases ("US" vs "United States") and to
number formatting ("300,000 km/s" vs "299,792 km/s"), so it's a precision signal,
not a verdict. Because it needs only the answer and the retrieved text — both in
every trace — it can be recomputed retroactively over the whole run history with
no agent re-run (`evals/regrade_traces.py`); that's also how it was backfilled
across runs graded before the metric existed.

The keyword fields driving the `answer` dimension live on each `EvalCase`:

- `must_include` — substrings the answer must contain (case-insensitive).
- `must_include_any` — at least one must appear. Used where acceptable phrasing
  varies (corrections and refusals), so a single required substring is too brittle.
- `must_exclude` — substrings that must *not* appear (catches a common wrong
  answer, e.g. "Sydney is the capital").

Search expectations live there too: `expect_tool_use` / `min_searches` (multi-hop
and comparison cases require ≥2; everything else ≥1) and an optional
`max_searches`. `expect_refusal` (default-on for the `unanswerable` category)
drives the `calibration` dimension.

**Why an LLM judge for grounding but not the rest.** Three of the four dimensions
are cheap, deterministic, and reproducible — substring matching and counting,
with no judge-model variance, which is exactly right for short factual answers
with unambiguous keys. Grounding is the one dimension substring matching can't
capture: "are *these specific claims* supported by *this retrieved text*" is a
semantic judgment, and it's precisely where a correct-but-ungrounded answer
(recalled from parametric memory) hides. So that one dimension spends an API call
on a judge (a mid-size model — `claude-sonnet-4-6` by default, `--judge-model` to
change). The judge is forced to return a structured `{score, rationale}` via a
tool call, and any judge error degrades gracefully to a deterministic proxy
(do the answer's required facts literally appear in the retrieved text?).
`--no-grounding` uses that proxy for everyone, making the whole suite API-judge-free
and fully reproducible when desired.

Cases carry a `category` tag. Beyond plain `factual` recall, the suite includes
**counterfactual** cases that probe robustness: `false_premise` (the question
embeds a wrong fact and the agent should correct it), `unanswerable` (the subject
does not exist and the agent should say it cannot find an answer rather than
hallucinate), `contrastive` (a near-duplicate of a factual case with one detail
changed), `multi_hop`, `computation`, `disambiguation`, and `comparison`.

The harness prints a per-case line with the overall score and a per-dimension
breakdown (plus reasons / the judge's rationale for any case below threshold),
then two comparison matrices — **mean score by dimension** and cases passed by
category — and a one-page dashboard PNG. It **exits non-zero if any case fails**
(overall < threshold, for any model) so it drops into CI unchanged. The
dimensions are what make a model's failure *legible*: a model can ace `answer`
while quietly losing points on `grounding` (recalling rather than retrieving) or
`search` (skipping the second hop), and the matrix surfaces that directly.

### Multi-model comparison

`run_agent` takes a `model` argument, so the same loop, prompt, and grader run
unchanged across models — the only thing that varies is the model id. By default
the harness sweeps three (`opus` → `claude-opus-4-7`, `sonnet` →
`claude-sonnet-4-6`, `haiku` → `claude-haiku-4-5`; friendly names live in
`agent/prompt.py:MODELS`) and prints a comparison matrix: overall and
per-category pass counts plus average latency per case. `--models` selects a
subset by friendly name or full id.

Holding the harness fixed and varying only the model is the point — it makes the
counterfactual categories a capability probe. Smaller/faster models tend to pass
plain `factual` recall but more readily echo a `false_premise` or fabricate an
answer to an `unanswerable` question, which the matrix surfaces directly. Per-run
JSON traces (tagged with their model) let you diff *why* a given model failed a
case.

## 6. Trade-offs & what I deliberately left out

- **No streaming.** Answers are short; non-streaming with a modest `max_tokens`
  keeps the loop simple and well under SDK HTTP timeouts.
- **No prompt caching.** The system prompt and tool schema are small; caching
  would add complexity for negligible savings at this scale.
- **Bounded-concurrency evals.** Every (model, case) pair runs in a
  `ThreadPoolExecutor` (the Anthropic SDK is sync/blocking on HTTP, so threads
  give real I/O parallelism). `--concurrency` (default 4) caps the workers and
  therefore the number of requests in flight — the rate-limit lever: with each
  request taking a few seconds, a small worker count keeps the effective rate
  well within tier limits. Beneath it, the client's exponential-backoff retry
  (`--max-retries`, default 6, honoring `retry-after`) absorbs any 429s, so the
  two together stay safe without a separate token bucket. A per-case error is
  caught and recorded as a zero-scored failure so one bad case can't sink the
  run, and results are slotted back in case order so the report stays
  deterministic regardless of completion order.
- **English Wikipedia only**, fixed result/extract sizes. Exposed as function
  args, not yet surfaced to the model.
- **`sys.path` bootstrap** in both entrypoints so `uv run agent/main.py …` and
  `python -m agent.main` both work without packaging the project.

## 7. Possible extensions

- Weighted dimensions — the overall score is an unweighted mean of the four;
  some uses would weight `answer`/`grounding` above `search`. The roll-up in
  `grade()` is the single seam for that.
- LLM judging for the `answer` dimension too (open-ended questions without a
  clean keyword key), reusing the structured-output judge pattern in
  `grading.py`.
- More tools: `get_article(title)` for full-text, or section-level fetches.
- Parallel eval execution across cases/models (currently sequential); the
  comparison already reports per-case latency, token reporting could join it.
- Citation extraction — the agent is prompted to cite article titles; a stricter
  eval could assert the cited title appears in the tool-call trace (a precise
  complement to the semantic `grounding` judge).
```
