# Wikipedia Agent

A small research agent built **directly on the Anthropic Messages API** (no agent
framework), plus an eval harness. Claude answers factual questions by calling a
single tool, `search_wikipedia(query)`, which hits the
[MediaWiki API](https://www.mediawiki.org/wiki/API:Main_page). The agent supports
**multi-turn tool use** — Claude can search, read results, and search again before
answering.

## Layout

```
agent/
  main.py        # entrypoint + the manual multi-turn tool-use loop (run_agent)
  wikipedia.py   # MediaWiki API wrapper exposing search_wikipedia(query)
  prompt.py      # system prompt, model id, and the tool schema
  trace.py       # per-run JSON trace logging
evals/
  run_evals.py       # eval entrypoint — imports run_agent from agent/main.py
  grading.py         # four scoring dimensions, LLM grounding judge + entity_coverage
  regrade_traces.py  # retroactively score entity_coverage over saved traces/
  cases.py           # the evaluation dataset
traces/              # JSON trace files, one per run (git-ignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an Anthropic API key. Copy the
example env file and fill in your key:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

The agent loads `<repo>/.env` automatically at startup (via `python-dotenv`), so
no `export` is needed. `.env` is git-ignored; an exported `ANTHROPIC_API_KEY` in
the shell still takes precedence over the file. `uv` resolves and installs
dependencies automatically on first run — including spaCy and the pinned
`en_core_web_sm` model (declared as dependencies, so no separate
`spacy download` step is needed) used by the `entity_coverage` metric.

## Run the agent

```bash
uv run agent/main.py "What is the capital of Australia?"
```

Progress (tool calls, turn count) is printed to stderr; the final answer goes to
stdout.

## Traces

Every run writes a structured JSON trace to `traces/` (one file per run, named
`<timestamp>-<run_id>.json`). A trace captures the question, each model turn
(text, tool calls, per-turn token usage), every tool result, the final answer,
aggregate token usage, and elapsed time.

Change the directory with `AGENT_TRACE_DIR`:

```bash
AGENT_TRACE_DIR=/tmp/agent-traces uv run agent/main.py "Who created the periodic table?"
```

Tracing is on by default; `run_agent(..., trace=False)` disables it
programmatically. Eval runs also emit one trace per case.

## Run the evals

```bash
uv run evals/run_evals.py                       # compare opus, sonnet, haiku
uv run evals/run_evals.py --models opus         # a single model
uv run evals/run_evals.py --models opus,sonnet  # a subset
uv run evals/run_evals.py --filter australia    # subset of cases by name
uv run evals/run_evals.py --verbose             # also print passing answers
uv run evals/run_evals.py --no-grounding        # skip the LLM judge (proxy only)
uv run evals/run_evals.py --threshold 0.8       # raise the pass bar
uv run evals/run_evals.py --judge-model opus    # judge with a different model
uv run evals/run_evals.py --concurrency 8       # more parallelism
```

Cases run **concurrently** across all model×case pairs. `--concurrency` (default
4) caps how many run at once — i.e. the number of API requests in flight, which
is the rate-limit lever: a few seconds per request times a small worker count
keeps the effective rate well within tier limits. Beneath it, the Anthropic
client retries 429/5xx with exponential backoff (`--max-retries`, default 6),
so the two together stay safe without a separate token bucket. Raise
`--concurrency` if your tier allows; lower it if you see 429s. Results are
collected in case order, so the report is deterministic regardless of finish
order.

The harness runs each case in `evals/cases.py` through `run_agent` and grades the
answer along **four dimensions of correctness**, each scored 0-1 and averaged
into an overall score (a case passes when the mean meets `--threshold`, default
0.7):

- **`answer`** — is the answer factually right? (keyword matching)
- **`search`** — *how much did it search?* The tool-call count should land in the
  case's expected window; under-searching (answering from memory) and wasteful
  over-searching both lose points.
- **`grounding`** — *how well is the answer supported by what Wikipedia returned?*
  An LLM judge rates the answer's claims against the retrieved passages (0-1 +
  rationale), catching answers that are correct but recalled rather than
  retrieved. Use `--no-grounding` for a deterministic proxy (no extra API calls).
- **`calibration`** — *does it qualify when it can't find information?* Cases where
  the honest answer is "it doesn't exist / I can't find it" must hedge; answerable
  cases instead penalize a false abstention.

Alongside those, one **auxiliary** metric is reported but **not folded into the
overall mean** (so it never moves the headline score):

- **`entity_coverage`** — *do the entities the answer asserts actually appear in the
  retrieved text?* spaCy NER pulls the answer's named entities (people, places,
  orgs, dates, quantities) and each is checked for a literal match in the sources;
  a named entity or year present in *no* passage is a mechanical hallucination
  signal. It's deterministic (fixed model weights, no API call) and the cheap,
  reproducible complement to the LLM `grounding` judge — its blind spot is
  paraphrase/aliases ("US" vs "United States"), so it's a precision-oriented check,
  not a replacement. In the dashboard and comparison matrix it's flagged with a `*`.

Because it needs only the final answer and the retrieved text — both saved in every
trace — it can be computed **retroactively** over already-recorded runs, with no
agent re-run and no API calls:

```bash
uv run evals/regrade_traces.py                  # score entity_coverage over traces/
uv run evals/regrade_traces.py --min-score 1.0  # only show answers with a gap
uv run evals/regrade_traces.py --json out.json  # also write a machine report
uv run evals/regrade_traces.py --chart out.png  # render a PNG (mean by model + dist)
```

Closed-book (no-retrieval) traces score 0.0 by construction, so the headline mean
and the chart are computed over **retrieval-bearing** traces only and report the
closed-book count separately.

It prints a per-case breakdown (overall + per-dimension scores plus the aux line,
with reasons / the judge's rationale for anything below threshold) and exits
non-zero if any case fails, so it works in CI.

By default it **sweeps three models** — `opus` (`claude-opus-4-7`), `sonnet`
(`claude-sonnet-4-6`), and `haiku` (`claude-haiku-4-5`) — and prints two
side-by-side matrices (**mean score by dimension** and cases passed by category)
plus average latency per case. `--models` and `--judge-model` take friendly names
or full model ids. The dimensions make divergence legible: a model can ace
`answer` while quietly losing points on `grounding` (recalling rather than
retrieving) or `search` (skipping a second hop).

The dataset mixes plain `factual` recall with **counterfactual** cases that test
robustness: `false_premise` (the question states something untrue and the agent
should correct it), `unanswerable` (the subject doesn't exist and the agent should
say so rather than make something up), and `contrastive` (a factual case with one
detail changed). See `docs/DESIGN.md` §5 for the grading fields.

## Notes

- Model: `claude-opus-4-7`. The loop is a standard manual agentic loop — append
  the assistant turn, run requested tools, feed `tool_result` blocks back, repeat
  until `stop_reason != "tool_use"` (bounded by `MAX_TURNS`).
- The Wikipedia wrapper returns the top search hits plus a plain-text lead extract
  of the best match, giving Claude enough context to answer in one search where
  possible.
# wikipedia-search
