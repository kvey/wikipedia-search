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
  run_evals.py   # eval entrypoint — imports run_agent from agent/main.py
  cases.py       # the evaluation dataset
traces/          # JSON trace files, one per run (git-ignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

`uv` resolves and installs dependencies automatically on first run.

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
```

The harness runs each case in `evals/cases.py` through `run_agent`, grades the
answer with keyword matching (and asserts the agent actually searched Wikipedia),
and prints a pass/fail report with a final score and a per-category breakdown. It
exits non-zero if any case fails, so it works in CI.

By default it **sweeps three models** — `opus` (`claude-opus-4-7`), `sonnet`
(`claude-sonnet-4-6`), and `haiku` (`claude-haiku-4-5`) — and prints a side-by-side
comparison: overall and per-category pass counts plus average latency per case.
`--models` takes friendly names or full model ids. The counterfactual categories
are where the models tend to diverge (smaller models more readily echo a false
premise or fabricate an answer).

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
