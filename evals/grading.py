"""Multi-dimensional grading for the Wikipedia agent.

A single boolean "did the answer contain the right keywords" hides most of what
makes a *grounded research agent* good or bad. This module scores each answer
along **four independent dimensions of correctness**, each a 0-1 score, then
averages them into an overall score:

- ``answer`` — keyword correctness. Fraction of the case's `must_include`,
  `must_include_any`, and `must_exclude` constraints that hold. Deterministic.
- ``search`` — *how much it searched*. Whether the tool-call count lands in the
  case's `[min_searches, max_searches]` window: under-searching (answering from
  parametric memory) and wasteful over-searching are both penalized.
- ``grounding`` — *how well the answer is supported by what Wikipedia returned*.
  An LLM judge rates the answer's factual claims against the concatenated
  retrieved passages (0-1 + a short rationale). This is the one dimension that
  cannot be done well with substring matching: it catches answers that are
  *correct* but recalled from the model's own memory rather than the sources.
- ``calibration`` — *qualifying when it can't find information*. For cases that
  set `expect_refusal`, the answer must explicitly hedge ("no such element",
  "could not find …"). For answerable cases it's the mirror image: a confident,
  unhedged answer scores 1.0 and a *false* abstention is penalized.

Three of the four dimensions are cheap, deterministic, and reproducible (in the
spirit of the keyword grader they extend); only ``grounding`` spends an API
call, and it can be turned off with `--no-grounding` to fall back to a
deterministic proxy.

On top of those four scored dimensions there is one **auxiliary** metric:

- ``entity_coverage`` — a deterministic, retroactive complement to ``grounding``.
  It uses spaCy NER to pull the entities the answer actually asserts (people,
  places, orgs, dates, quantities) and checks each one literally appears in the
  retrieved text. A named entity or year that shows up in *no* source passage is
  a mechanical hallucination signal. Unlike the LLM ``grounding`` judge it is
  cheap, fully reproducible (fixed model weights, no API call), and computable
  after the fact from a saved trace — see ``evals/regrade_traces.py``. It is
  reported alongside the dimensions but is **not folded into the overall mean**
  (see ``AUX_METRICS`` vs ``DIMENSIONS``), so it never moves the headline score.
  Its blind spot is paraphrase/aliases ("US" vs "United States"); it is a
  precision-oriented complement to the judge, not a replacement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import anthropic

from evals.cases import EvalCase

# Dimensions in display order. Kept here so the harness, JSON summary, and
# charts all agree on the set and ordering.
DIMENSIONS = ["answer", "search", "grounding", "calibration"]

# Auxiliary metrics: reported alongside the dimensions but deliberately kept
# OUT of the overall mean, so adding them never moves any existing score. Kept
# as its own list (rather than appended to DIMENSIONS) precisely so the harness,
# JSON summary, and charts can treat them separately.
AUX_METRICS = ["entity_coverage"]

# Default model for the LLM grounding judge. A mid-size model is plenty for
# "is this claim supported by this passage" and keeps judging cheap; override
# with --judge-model.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

# Hedging vocabulary for the calibration dimension. `REFUSAL_MARKERS` are the
# phrasings that *count as* an appropriate "I can't answer this" for cases that
# expect a refusal. `FALSE_ABSTENTION_MARKERS` is a deliberately narrow subset —
# "could not find" style admissions of defeat — that should *not* appear on an
# answerable question. The narrow set avoids false positives on `false_premise`
# corrections, which legitimately say "no"/"never"/"not" while still answering.
REFUSAL_MARKERS = [
    "could not find",
    "couldn't find",
    "cannot find",
    "can't find",
    "did not find",
    "unable to find",
    "no information",
    "not enough information",
    "does not exist",
    "doesn't exist",
    "no such",
    "not a real",
    "not an element",
    "fictional",
    "there is no",
    "there's no",
    "was not awarded",
    "no record",
    "i could not",
    "i couldn't",
]
FALSE_ABSTENTION_MARKERS = [
    "could not find",
    "couldn't find",
    "cannot find",
    "can't find",
    "unable to find",
    "no information",
    "not enough information",
    "i don't have enough",
    "i do not have enough",
]


@dataclass
class DimensionScore:
    """One dimension's 0-1 score plus human-readable notes on why."""

    name: str
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class CaseGrade:
    """All dimension scores for one answer, plus the rolled-up overall score.

    ``aux`` holds auxiliary metrics (e.g. ``entity_coverage``) that are reported
    but intentionally excluded from ``overall`` — see ``AUX_METRICS``.
    """

    dimensions: dict[str, DimensionScore]
    overall: float
    passed: bool
    aux: dict[str, DimensionScore] = field(default_factory=dict)

    def score(self, name: str) -> float:
        return self.dimensions[name].score

    def aux_score(self, name: str) -> float:
        return self.aux[name].score


# --------------------------------------------------------------------------- #
# Deterministic dimensions
# --------------------------------------------------------------------------- #
def grade_answer(case: EvalCase, answer: str) -> DimensionScore:
    """Keyword correctness as a fraction of satisfied constraints (0-1)."""
    lower = answer.lower()
    reasons: list[str] = []
    satisfied = 0
    total = 0

    for needle in case.must_include:
        total += 1
        if needle.lower() in lower:
            satisfied += 1
        else:
            reasons.append(f"missing expected text: {needle!r}")

    if case.must_include_any:
        total += 1
        if any(n.lower() in lower for n in case.must_include_any):
            satisfied += 1
        else:
            reasons.append(f"missing any of: {case.must_include_any!r}")

    for needle in case.must_exclude:
        total += 1
        if needle.lower() not in lower:
            satisfied += 1
        else:
            reasons.append(f"contains forbidden text: {needle!r}")

    score = 1.0 if total == 0 else satisfied / total
    return DimensionScore("answer", score, reasons)


def grade_search(case: EvalCase, n_tool_calls: int) -> DimensionScore:
    """Score whether the search count lands in the expected window.

    Under-searching scales linearly toward zero (0 searches when ≥1 is expected
    is a hard 0); over-searching past `max_searches` is penalized more gently,
    since a wasteful extra lookup is less serious than answering ungrounded.
    """
    lo = case.min_searches or 0
    hi = case.max_searches
    reasons: list[str] = []

    if lo and n_tool_calls < lo:
        score = n_tool_calls / lo
        reasons.append(
            f"under-searched: {n_tool_calls} of ≥{lo} expected search(es)"
        )
    elif hi is not None and n_tool_calls > hi:
        score = max(0.0, 1.0 - 0.25 * (n_tool_calls - hi))
        reasons.append(
            f"over-searched: {n_tool_calls} searches, expected ≤{hi}"
        )
    else:
        score = 1.0
    return DimensionScore("search", score, reasons)


def grade_calibration(case: EvalCase, answer: str) -> DimensionScore:
    """Score appropriate hedging (or its absence) about findability."""
    lower = answer.lower()
    if case.expect_refusal:
        hedged = any(m in lower for m in REFUSAL_MARKERS)
        if hedged:
            return DimensionScore("calibration", 1.0)
        return DimensionScore(
            "calibration",
            0.0,
            ["expected the answer to qualify that it can't find / doesn't exist"],
        )

    # Answerable case: a confident answer is good; admitting defeat is the fault.
    abstained = any(m in lower for m in FALSE_ABSTENTION_MARKERS)
    if abstained:
        return DimensionScore(
            "calibration",
            0.0,
            ["false abstention: claimed it couldn't find an answerable fact"],
        )
    return DimensionScore("calibration", 1.0)


# --------------------------------------------------------------------------- #
# Grounding dimension (LLM judge, with a deterministic fallback)
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM = """\
You are a strict grading judge for a Wikipedia research agent. You are given a \
question, the agent's final answer, and the exact Wikipedia text the agent \
retrieved. Rate ONLY how well the answer's factual claims are *grounded in the \
retrieved text* — not whether the answer is correct in general, and not its \
style.

Scoring guide (0.0-1.0):
- 1.0  Every substantive claim in the answer is directly supported by the \
retrieved text.
- 0.5  Partially supported: some claims are backed by the text, others are not \
present in it (likely recalled from memory).
- 0.0  The key claims are absent from or contradicted by the retrieved text.

Special cases:
- If the answer correctly states that the information could not be found or that \
the subject does not exist, and the retrieved text indeed does not contain it, \
that is well grounded → score near 1.0.
- If no text was retrieved, the answer cannot be grounded → score 0.0.

Call `report_grounding` with your score and a one-sentence rationale."""

_GROUNDING_TOOL = [
    {
        "name": "report_grounding",
        "description": "Report how well the answer is grounded in the retrieved text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "description": "Grounding score from 0.0 to 1.0.",
                },
                "rationale": {
                    "type": "string",
                    "description": "One sentence explaining the score.",
                },
            },
            "required": ["score", "rationale"],
        },
    }
]


def _deterministic_grounding(
    case: EvalCase, answer: str, retrieved: str
) -> DimensionScore:
    """Fallback proxy: are the answer's required facts present in the sources?

    Used when the LLM judge is disabled. Checks that the case's `must_include`
    answer keywords (the substantive facts) actually appear in the retrieved
    text — i.e. the answer is traceable to a source rather than recalled.
    """
    if not retrieved.strip():
        return DimensionScore("grounding", 0.0, ["no text retrieved to ground on"])
    terms = case.must_include or case.must_include_any
    if not terms:
        # No positive fact to trace (e.g. refusals); treat as grounded.
        return DimensionScore("grounding", 1.0, ["no positive fact to trace"])
    low = retrieved.lower()
    present = [t for t in terms if t.lower() in low]
    score = len(present) / len(terms)
    reasons = []
    if score < 1.0:
        missing = [t for t in terms if t.lower() not in low]
        reasons.append(f"fact(s) not found in retrieved text: {missing!r}")
    return DimensionScore("grounding", score, reasons)


def grade_grounding(
    case: EvalCase,
    answer: str,
    retrieved: str,
    *,
    client: anthropic.Anthropic | None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> DimensionScore:
    """Score how well the answer is supported by the retrieved Wikipedia text.

    Uses an LLM judge when `client` is provided; otherwise falls back to the
    deterministic proxy. Any judge error degrades gracefully to the proxy so a
    transient API hiccup never crashes a run.
    """
    if client is None:
        return _deterministic_grounding(case, answer, retrieved)
    if not retrieved.strip():
        return DimensionScore("grounding", 0.0, ["no text retrieved to ground on"])

    user = (
        f"Question:\n{case.question}\n\n"
        f"Agent's answer:\n{answer}\n\n"
        f"Retrieved Wikipedia text:\n{retrieved}"
    )
    try:
        resp = client.messages.create(
            model=judge_model,
            max_tokens=512,
            system=_JUDGE_SYSTEM,
            tools=_GROUNDING_TOOL,
            tool_choice={"type": "tool", "name": "report_grounding"},
            messages=[{"role": "user", "content": user}],
        )
        block = next(b for b in resp.content if b.type == "tool_use")
        score = float(block.input["score"])
        rationale = str(block.input.get("rationale", "")).strip()
    except (anthropic.APIError, StopIteration, KeyError, ValueError, TypeError) as exc:
        proxy = _deterministic_grounding(case, answer, retrieved)
        proxy.reasons.insert(0, f"(judge unavailable, used proxy: {exc})")
        return proxy

    score = max(0.0, min(1.0, score))
    return DimensionScore("grounding", score, [rationale] if rationale else [])


# --------------------------------------------------------------------------- #
# Auxiliary metric: entity coverage (deterministic, retroactive)
# --------------------------------------------------------------------------- #
# spaCy NER does the entity extraction. It's deterministic (fixed model
# weights) and API-free, so the metric stays cheap and reproducible, but it's
# far more reliable than hand-rolled regex at deciding what *is* an entity —
# multi-word names, nationalities, dates and quantities — and at where one ends.
# The model is loaded once and cached; loading is deferred to first use so
# importing this module (e.g. for the four scored dimensions alone) stays cheap.
_SPACY_MODEL = "en_core_web_sm"
_NLP = None

# Entity labels worth grounding: named things plus dates/quantities. The chatty
# numeric labels spaCy emits for bare counts and ranks (CARDINAL "two", ORDINAL
# "first") are excluded — they're rarely substantive claims and their surface
# form rarely matches the source verbatim, which would only add false misses.
_GROUNDED_LABELS = frozenset({
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY",
    "QUANTITY",
})

# Markdown decoration to strip before extraction so "**Canberra**" parses cleanly.
_MARKDOWN_RE = re.compile(r"[*_`#>]+|\[|\]|\((?:https?://)[^)]*\)")


def _nlp():
    """Return the cached spaCy pipeline, loading it on first use."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load(_SPACY_MODEL)
    return _NLP


def _extract_entities(text: str) -> list[str]:
    """Pull the substantive entities an answer asserts, via spaCy NER.

    Deterministic and API-free. Keeps named entities plus dates/quantities (see
    ``_GROUNDED_LABELS``); over-extraction is harmless — an entity that's in the
    sources just scores as covered — so the point is simply to surface the
    *uncovered* ones. Returns entity surface forms, order-preserved and deduped.
    """
    clean = _MARKDOWN_RE.sub(" ", text)
    seen: dict[str, None] = {}  # preserve order, dedupe case-insensitively
    for ent in _nlp()(clean).ents:
        if ent.label_ not in _GROUNDED_LABELS:
            continue
        # Collapse internal whitespace (a span can run across a newline into the
        # citation footer) and trim surrounding punctuation before matching.
        surface = " ".join(ent.text.split()).strip(".,;:!?\"'()[]")
        if surface:
            seen.setdefault(surface, None)
    return list(seen)


def grade_entity_coverage(answer: str, retrieved: str) -> DimensionScore:
    """Auxiliary metric: are the answer's entities present in the sources?

    Extracts the answer's named entities (via spaCy NER) and checks each appears
    (case-insensitive substring) in the concatenated retrieved text. Score is
    the fraction covered. Self-contained — takes only the answer and the
    retrieved text — so the same function grades a live run and a saved trace.
    """
    if not retrieved.strip():
        return DimensionScore("entity_coverage", 0.0, ["no text retrieved to ground on"])
    entities = _extract_entities(answer)
    if not entities:
        return DimensionScore("entity_coverage", 1.0, ["no entities to verify"])

    low = retrieved.lower()
    missing = [e for e in entities if e.lower() not in low]
    score = 1.0 - len(missing) / len(entities)
    reasons = []
    if missing:
        reasons.append(f"entit(ies) not found in retrieved text: {missing!r}")
    return DimensionScore("entity_coverage", score, reasons)


# --------------------------------------------------------------------------- #
# Roll-up
# --------------------------------------------------------------------------- #
def grade(
    case: EvalCase,
    answer: str,
    tool_calls: list[dict],
    *,
    client: anthropic.Anthropic | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = 0.7,
) -> CaseGrade:
    """Grade one answer across all dimensions and roll up to an overall score.

    `tool_calls` are the agent's executed tool calls (each a {name, input,
    output} dict); their outputs are concatenated into the text the grounding
    judge sees. The overall score is the unweighted mean of the four dimensions;
    a case "passes" when that mean meets `threshold`.
    """
    retrieved = "\n\n".join(c.get("output", "") for c in tool_calls)
    dims = {
        "answer": grade_answer(case, answer),
        "search": grade_search(case, len(tool_calls)),
        "grounding": grade_grounding(
            case, answer, retrieved, client=client, judge_model=judge_model
        ),
        "calibration": grade_calibration(case, answer),
    }
    # Auxiliary metrics are computed here but kept out of the overall mean.
    aux = {"entity_coverage": grade_entity_coverage(answer, retrieved)}
    overall = sum(d.score for d in dims.values()) / len(dims)
    return CaseGrade(
        dimensions=dims, overall=overall, passed=overall >= threshold, aux=aux
    )
