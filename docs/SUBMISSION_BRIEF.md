# Submission Brief

This is the short written companion to the five-minute slide deck in `slides.md`.
The longer audit trail remains in `docs/EVAL_REPORT.md`.

## Approach and design rationale

I built a Wikipedia research agent directly on the Anthropic API instead of using
an agent framework. The goal was to keep the control loop small enough to inspect:
the model can call Wikipedia tools, the loop records each model turn and tool
result, and the final answer is emitted with a complete JSON trace under
`traces/`.

The central design choice was observability. Early runs showed that a strong model
could answer many questions from memory without using search. That is not a
research agent, so I forced at least one search call and treated every trace as a
first-class artifact. This later paid off because new metrics could be computed
retroactively without rerunning the agent.

## Eval design

The eval suite grew from simple factual checks into 37 cases across eight
categories: factual, false premise, unanswerable, contrastive, multi-hop,
computation, comparison, and disambiguation. Each case is scored across four
dimensions and passes when the average score is at least 0.7:

- `answer`: deterministic constraints over required and forbidden answer content.
- `search`: whether the agent searched an appropriate amount for the case.
- `grounding`: an LLM judge compares claims in the answer with retrieved
  Wikipedia passages.
- `calibration`: the agent should hedge or refuse when the premise is false or
  the answer is not supported.

I also added `entity_coverage` as an auxiliary grounding metric. It extracts
named entities from the final answer and checks whether those entities appear in
retrieved text. It is not included in the overall score because it is a precision
signal with known limitations around aliases and number formatting.

## Learnings

The first important result was that answer correctness was too easy. In the
37-case sweep, all models scored very high on `answer`, but `grounding` was the
weakest dimension for every model: Opus 0.773, Sonnet 0.842, and Haiku 0.903.

The ablation made the failure mode clear. Running the same cases without tools
dropped overall scores to about 0.48 because search and grounding became zero by
construction, but answer scores stayed around 0.97. The models already knew many
facts. Retrieval was not mainly improving raw correctness; it was improving
whether the answer could be justified from evidence.

Infrastructure also affected interpretation. A small rate-limit issue made one
Sonnet run look worse than it was, so the harness now isolates transient errors
and uses jittered cooldowns.

## Iterations

Trace review pointed to two mechanical grounding gaps. First, `search_wikipedia`
accepted only one query, so a missed query forced slow sequential retries. Second,
only the top search hit received a lead extract, so relevant facts in lower hits
often required another turn.

I changed the tool surface rather than the prompt. `search_wikipedia` now accepts
multiple query phrasings, fans them out, merges and dedupes hits by title, and
extracts the top two merged articles. I also added `get_article(title, section?)`
for deeper article fetches when snippets are too thin. The search score now counts
search steps rather than every tool call, so a fan-out is one search step and
article detail fetches do not inflate the score.

The targeted metric improved across all models on the same suite and threshold:
Opus grounding rose from 0.773 to 0.903, Sonnet from 0.842 to 0.884, and Haiku
from 0.903 to 0.953. All three models reached 37/37 passing cases, with no
max-turn failures.

The entity-coverage metric also needed iteration. Regrading exposed that the most
common uncovered entity was the word "Wikipedia" from citation boilerplate, not a
hallucination. I fixed the metric by stripping citation footers and stoplisting
that token before NER. On the same traces, mean entity coverage moved from 0.684
to 0.802, and the remaining flags are more meaningful.

## Current caveats

The grounding judge has variance because it is itself model-based. Entity coverage
is deterministic, but literal matching misses aliases and equivalent number/date
formats. The evals also contain many facts that frontier models already know, so a
future suite should include fresher or less memorized questions. Finally, the
archived `results.json` files should link directly to traces so dashboards can be
recomputed cleanly after new retroactive metrics are added.

Approximate time spent: about 2 hours total.
