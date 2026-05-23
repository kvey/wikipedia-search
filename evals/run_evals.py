"""Eval harness entrypoint.

Runs each case in `cases.CASES` through the agent (importing `run_agent` from
agent/main.py), grades the answers along several dimensions (see
`evals/grading.py`), and prints a report. Can sweep multiple models and print a
side-by-side comparison. Cases run concurrently under a bounded thread pool whose
size (`--concurrency`) caps the number of in-flight requests — the rate-limit
lever. Beneath it sit two safety nets: the SDK's per-request retry/backoff, and a
case-level retry (`--case-retries`) that shares a cooldown gate so rate-limit
retries don't stampede the same limit. A case that still fails on a transient
error (rate limit / timeout / 5xx) is flagged as errored and excluded from the
stats — never scored as a 0.0 model failure — so a contended run can't make a
model look like it regressed.

Each run is archived to its own directory under `eval_results/` (named with an
ISO timestamp and the current git hash, so past runs are preserved), containing
the raw results as JSON plus matplotlib charts of performance.

Usage:
    export ANTHROPIC_API_KEY=...
    uv run evals/run_evals.py                       # compare opus, sonnet, haiku
    uv run evals/run_evals.py --models opus         # a single model
    uv run evals/run_evals.py --models opus,sonnet  # a subset
    uv run evals/run_evals.py --filter australia    # subset of cases by name
    uv run evals/run_evals.py --verbose             # also print passing answers
    uv run evals/run_evals.py --concurrency 8       # more parallelism
    uv run evals/run_evals.py --output-dir out      # where to archive runs
    uv run evals/run_evals.py --no-save             # don't write any artifacts
    uv run evals/run_evals.py --ablation            # also run closed-book (no
                                                    # tools) and compare
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Make the repo root importable whether run as a script or a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from agent.main import run_agent
from agent.prompt import MODELS
from evals.cases import CASES, EvalCase
from evals.grading import (
    AUX_METRICS,
    DEFAULT_JUDGE_MODEL,
    DIMENSIONS,
    CaseGrade,
    DimensionScore,
    grade,
)


@dataclass(frozen=True)
class Series:
    """One arm of a run: a model evaluated either with or without retrieval.

    A run evaluates every case across one or more series. With `--ablation` each
    model contributes two series — a `tools` arm and a closed-book `no-tools`
    arm — so the same questions are answered both with and without Wikipedia in
    a single invocation. `label` is the key results are stored and charted under.
    """

    model: str
    use_tools: bool = True

    @property
    def label(self) -> str:
        return self.model if self.use_tools else f"{self.model} (no tools)"


@dataclass
class CaseResult:
    case: EvalCase
    grade: CaseGrade
    answer: str
    n_tool_calls: int
    seconds: float
    use_tools: bool = True
    # True when the run ended in a transient infrastructure error (rate limit,
    # timeout, 5xx) that survived all retries. Such a result is *not* a model
    # quality signal, so it is excluded from every aggregate rather than counted
    # as a 0.0 failure — otherwise a contended run makes a model look like it
    # regressed (which is exactly what a stampede of 429s did to one series).
    errored: bool = False

    @property
    def passed(self) -> bool:
        return self.grade.passed


def run_case(
    case: EvalCase,
    client: anthropic.Anthropic,
    model: str,
    *,
    use_tools: bool,
    judge_client: anthropic.Anthropic | None,
    judge_model: str,
    threshold: float,
) -> CaseResult:
    start = time.monotonic()
    result = run_agent(case.question, client=client, model=model, use_tools=use_tools)
    elapsed = time.monotonic() - start
    case_grade = grade(
        case,
        result.answer,
        result.tool_calls,
        client=judge_client,
        judge_model=judge_model,
        threshold=threshold,
    )
    return CaseResult(
        case=case,
        grade=case_grade,
        answer=result.answer,
        n_tool_calls=len(result.tool_calls),
        seconds=elapsed,
        use_tools=use_tools,
    )


def _dims_line(res: CaseResult) -> str:
    """One-line per-dimension score summary, e.g. 'answer=1.00 search=0.50 …'."""
    return "  ".join(f"{d}={res.grade.score(d):.2f}" for d in DIMENSIONS)


def _aux_line(res: CaseResult) -> str:
    """One-line auxiliary-metric summary (reported, not in the overall mean)."""
    return "  ".join(f"{m}={res.grade.aux_score(m):.2f}" for m in AUX_METRICS)


# API failures that reflect capacity/transport, not model quality. We retry
# these, and if they survive we mark the result `errored` so it is excluded from
# the stats instead of scored as a model failure.
_TRANSIENT_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,  # 5xx, including 529 overloaded
)
# Status codes worth a retry even if the SDK surfaced a generic APIStatusError.
_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def _is_transient(exc: BaseException) -> bool:
    """True if `exc` is a retryable infrastructure error, not a real failure."""
    if isinstance(exc, _TRANSIENT_ERRORS):
        return True
    return getattr(exc, "status_code", None) in _TRANSIENT_STATUS


class RateLimitGate:
    """Shared cooldown so a rate-limit hit on one worker pauses all of them.

    The failure we actually saw was a *concurrent-connection* rate limit, which
    per-request retries can't fix alone: every worker backs off and retries in
    lockstep, re-colliding on the same limit. When any worker trips this gate,
    all workers wait out a short, jittered cooldown before their next attempt —
    thinning the herd instead of stampeding it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until = 0.0

    def wait(self) -> None:
        """Block until any active cooldown has elapsed."""
        while True:
            with self._lock:
                remaining = self._until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.5))

    def trip(self, seconds: float) -> None:
        """Open a cooldown of at least `seconds` for every worker."""
        with self._lock:
            self._until = max(self._until, time.monotonic() + seconds)


def _error_result(
    case: EvalCase,
    exc: Exception,
    use_tools: bool = True,
    *,
    errored: bool = False,
) -> CaseResult:
    """Wrap an unrecoverable per-case error as a zero-scored failed result.

    Lets one case that errors out fail in isolation instead of aborting the
    whole parallel run. When `errored` is set (a transient infra error that
    survived retries) the result is flagged so aggregates skip it rather than
    treating it as a model failure.
    """
    dims = {d: DimensionScore(d, 0.0, [f"run error: {exc}"]) for d in DIMENSIONS}
    aux = {m: DimensionScore(m, 0.0, [f"run error: {exc}"]) for m in AUX_METRICS}
    return CaseResult(
        case=case,
        grade=CaseGrade(dimensions=dims, overall=0.0, passed=False, aux=aux),
        answer=f"(run error: {exc})",
        n_tool_calls=0,
        seconds=0.0,
        use_tools=use_tools,
        errored=errored,
    )


def run_all(
    series: list[Series],
    cases: list[EvalCase],
    client: anthropic.Anthropic,
    *,
    judge_client: anthropic.Anthropic | None,
    judge_model: str,
    threshold: float,
    concurrency: int,
    case_retries: int = 4,
    backoff_base: float = 2.0,
    backoff_cap: float = 30.0,
) -> dict[str, list[CaseResult]]:
    """Run every (series, case) pair concurrently and collect ordered results.

    A `series` is a (model, use_tools) arm; with `--ablation` each model appears
    as two series, so the same cases are evaluated both with and without
    retrieval in one run. Results are keyed by `series.label`.

    Concurrency is bounded by a thread pool of `concurrency` workers, so at most
    that many requests are ever in flight — the primary rate-limit lever. Two
    safety nets sit beneath it: the SDK's own per-request backoff (see `main`),
    and a case-level retry here that re-runs the whole case (agent + judge) on a
    transient error the SDK gave up on. Case retries share a `RateLimitGate`, so
    a rate-limit hit on one worker pauses all of them — the fix for the
    concurrent-connection limit that a per-request retry can't clear alone. A
    case that still fails after `case_retries` is flagged `errored` and dropped
    from the stats rather than scored as a model failure.

    Per-case results are slotted back into per-series lists in their original
    case order, so the downstream report is deterministic regardless of the
    order in which futures complete.
    """
    by_label: dict[str, list[CaseResult | None]] = {
        s.label: [None] * len(cases) for s in series
    }
    tasks = [
        (s, i, case) for s in series for i, case in enumerate(cases)
    ]
    total = len(tasks)
    lock = threading.Lock()
    done = 0
    gate = RateLimitGate()

    def work(task: tuple[Series, int, EvalCase]) -> tuple[Series, int, CaseResult]:
        s, idx, case = task
        last_exc: Exception | None = None
        for attempt in range(1, case_retries + 1):
            gate.wait()  # honor any cooldown another worker opened
            try:
                return s, idx, run_case(
                    case,
                    client,
                    s.model,
                    use_tools=s.use_tools,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    threshold=threshold,
                )
            except Exception as exc:  # isolate a failed case; don't sink the run
                last_exc = exc
                if not _is_transient(exc) or attempt == case_retries:
                    break
                # Full-jitter exponential backoff, applied to every worker via
                # the shared gate so retries don't stampede the same limit.
                window = min(backoff_base * 2 ** (attempt - 1), backoff_cap)
                cooldown = random.uniform(0, window)
                gate.trip(cooldown)
                with lock:
                    print(
                        f"  retry {attempt}/{case_retries - 1}  {s.label} :: "
                        f"{case.name} after {type(exc).__name__} "
                        f"(cooldown ~{cooldown:.1f}s)",
                        file=sys.stderr,
                    )
        # Exhausted retries or hit a non-transient error.
        errored = last_exc is not None and _is_transient(last_exc)
        res = _error_result(case, last_exc, use_tools=s.use_tools, errored=errored)
        return s, idx, res

    print(
        f"Running {total} evaluation(s) with concurrency {concurrency} "
        f"(≤{concurrency} requests in flight)…",
        file=sys.stderr,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, t) for t in tasks]
        for fut in as_completed(futures):
            s, idx, res = fut.result()
            by_label[s.label][idx] = res
            with lock:
                done += 1
                status = "ERROR" if res.errored else ("PASS" if res.passed else "FAIL")
                print(
                    f"  [{done}/{total}] {status}  {s.label} :: {res.case.name} "
                    f"(overall={res.grade.overall:.2f}, {res.seconds:.1f}s)",
                    file=sys.stderr,
                )

    # All slots are now filled; drop the Optional for the return type.
    return {lbl: [r for r in lst if r is not None] for lbl, lst in by_label.items()}


def print_model_report(model: str, results: list[CaseResult], verbose: bool) -> None:
    """Print the per-case report for one model from already-collected results."""
    print(f"\n{'#' * 60}\n# Model: {model}\n{'#' * 60}\n")
    for res in results:
        case = res.case
        status = "ERROR" if res.errored else ("PASS" if res.passed else "FAIL")
        print(
            f"[{status}] {case.name} ({case.category}, "
            f"overall={res.grade.overall:.2f}, "
            f"{res.n_tool_calls} search(es), {res.seconds:.1f}s)"
        )
        print(f"        {_dims_line(res)}")
        print(f"        aux: {_aux_line(res)}  (not in overall)")
        # Surface why any below-perfect dimension lost points.
        if verbose or not res.passed:
            for dim in DIMENSIONS:
                ds = res.grade.dimensions[dim]
                for reason in ds.reasons:
                    print(f"        - [{dim}] {reason}")
            for m in AUX_METRICS:
                for reason in res.grade.aux[m].reasons:
                    print(f"        - [{m}] {reason}")
        if verbose or not res.passed:
            print(f"        answer: {res.answer}\n")

    scored = _scored(results)
    n_err = len(results) - len(scored)
    passed = sum(r.passed for r in scored)
    mean = _mean([r.grade.overall for r in scored])
    err_note = f"  ·  {n_err} excluded (infra error)" if n_err else ""
    print(
        f"\n  {model}: {passed}/{len(scored)} passed  ·  "
        f"mean score {mean:.2f}{err_note}"
    )


def _categories(cases: list[EvalCase]) -> list[str]:
    seen: list[str] = []
    for c in cases:
        if c.category not in seen:
            seen.append(c.category)
    return seen


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _scored(results: list[CaseResult]) -> list[CaseResult]:
    """Drop transient-error runs; only genuinely graded cases feed the stats."""
    return [r for r in results if not r.errored]


def print_comparison(
    results_by_model: dict[str, list[CaseResult]],
    cases: list[EvalCase],
) -> None:
    """Print per-dimension and per-category score matrices across models."""
    name_w = max(len(m) for m in results_by_model) + 2

    def _matrix(title: str, cols: list[str], cell: callable) -> None:
        header = "model".ljust(name_w) + "".join(c.center(14) for c in cols)
        print(f"\n{'=' * len(header)}")
        print(title)
        print("=" * len(header))
        print(header)
        for model, results in results_by_model.items():
            cells = [cell(model, results, c) for c in cols]
            print(model.ljust(name_w) + "".join(c.center(14) for c in cells))

    # 1) Per-dimension mean scores (the headline: where each model is strong/weak).
    # Transient-error runs are excluded so a contended series isn't dragged down.
    def dim_cell(model: str, results: list[CaseResult], col: str) -> str:
        results = _scored(results)
        if col == "overall":
            return f"{_mean([r.grade.overall for r in results]):.2f}"
        if col in AUX_METRICS:
            return f"{_mean([r.grade.aux_score(col) for r in results]):.2f}"
        return f"{_mean([r.grade.score(col) for r in results]):.2f}"

    _matrix(
        "MODEL COMPARISON — mean score by dimension  (* = aux, not in overall)",
        ["overall", *DIMENSIONS, *AUX_METRICS],
        dim_cell,
    )

    # 2) Per-category pass counts (cases passing the overall threshold).
    categories = _categories(cases)

    def cat_cell(model: str, results: list[CaseResult], col: str) -> str:
        results = _scored(results)
        if col == "overall":
            return f"{sum(r.passed for r in results)}/{len(results)}"
        cat = [r for r in results if r.case.category == col]
        return f"{sum(r.passed for r in cat)}/{len(cat)}"

    _matrix(
        "MODEL COMPARISON — cases passed by category",
        ["overall", *categories],
        cat_cell,
    )

    # Average latency per model is a useful tiebreaker (errored runs excluded:
    # their seconds are 0.0 and would understate latency).
    print()
    for model, results in results_by_model.items():
        scored = _scored(results)
        avg = sum(r.seconds for r in scored) / len(scored) if scored else 0.0
        n_err = len(results) - len(scored)
        err_note = f"  ({n_err} infra error excluded)" if n_err else ""
        print(f"  {model}: avg {avg:.1f}s/case{err_note}")


def print_ablation(
    series: list[Series],
    results_by_model: dict[str, list[CaseResult]],
) -> None:
    """Print the with-tools vs without-tools comparison for each ablated model.

    The headline is the `answer` dimension (keyword correctness): how much the
    score *drops* when retrieval is removed is the retrieval lift. A small drop
    means the model already knew the fact from parametric memory; a large drop
    means the answer genuinely depended on what Wikipedia returned. `overall`
    falls further in the no-tools arm by construction (its `search`/`grounding`
    dimensions are necessarily ~0), so `answer` is the fair head-to-head.
    """
    models: list[str] = []
    for s in series:
        if s.model not in models:
            models.append(s.model)
    pairs = [
        m
        for m in models
        if Series(m, True).label in results_by_model
        and Series(m, False).label in results_by_model
    ]
    if not pairs:
        return

    def mean_answer(results: list[CaseResult]) -> float:
        return _mean([r.grade.score("answer") for r in _scored(results)])

    cols = ["answer w/ tools", "answer w/o", "Δ (lift)", "overall w/", "overall w/o"]
    name_w = max(len(m) for m in pairs) + 2
    header = "model".ljust(name_w) + "".join(c.center(16) for c in cols)
    print(f"\n{'=' * len(header)}")
    print("ABLATION — retrieval lift (with tools vs closed-book)")
    print("=" * len(header))
    print(header)
    for m in pairs:
        tools = results_by_model[Series(m, True).label]
        closed = results_by_model[Series(m, False).label]
        a_t, a_n = mean_answer(tools), mean_answer(closed)
        o_t = _mean([r.grade.overall for r in _scored(tools)])
        o_n = _mean([r.grade.overall for r in _scored(closed)])
        cells = [
            f"{a_t:.2f}",
            f"{a_n:.2f}",
            f"{a_t - a_n:+.2f}",
            f"{o_t:.2f}",
            f"{o_n:.2f}",
        ]
        print(m.ljust(name_w) + "".join(c.center(16) for c in cells))


def _git_hash() -> str:
    """Return the short HEAD hash, suffixed '-dirty' if the tree has changes.

    Falls back to 'nogit' when we're not inside a git repo.
    """
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "nogit"


def _summarize(
    results_by_model: dict[str, list[CaseResult]],
    cases: list[EvalCase],
) -> dict:
    """Build the JSON-serializable summary that's saved and charted."""
    categories = _categories(cases)
    models: dict[str, dict] = {}
    for model, results in results_by_model.items():
        # Stats are computed over genuinely graded cases; transient-error runs
        # are reported as a count but never folded into means or pass rates.
        scored = _scored(results)
        per_category = {}
        for cat in categories:
            cat_results = [r for r in scored if r.case.category == cat]
            per_category[cat] = {
                "passed": sum(r.passed for r in cat_results),
                "total": len(cat_results),
            }
        dimension_means = {
            dim: _mean([r.grade.score(dim) for r in scored]) for dim in DIMENSIONS
        }
        aux_means = {
            m: _mean([r.grade.aux_score(m) for r in scored]) for m in AUX_METRICS
        }
        models[model] = {
            "use_tools": results[0].use_tools if results else True,
            "passed": sum(r.passed for r in scored),
            "total": len(scored),
            "errored": len(results) - len(scored),
            "mean_score": _mean([r.grade.overall for r in scored]),
            "avg_seconds": (
                sum(r.seconds for r in scored) / len(scored) if scored else 0.0
            ),
            "dimensions": dimension_means,
            "aux": aux_means,
            "categories": per_category,
            "cases": [
                {
                    "name": r.case.name,
                    "category": r.case.category,
                    "passed": r.passed,
                    "errored": r.errored,
                    "overall": round(r.grade.overall, 3),
                    "dimensions": {
                        dim: {
                            "score": round(r.grade.dimensions[dim].score, 3),
                            "reasons": r.grade.dimensions[dim].reasons,
                        }
                        for dim in DIMENSIONS
                    },
                    "aux": {
                        m: {
                            "score": round(r.grade.aux[m].score, 3),
                            "reasons": r.grade.aux[m].reasons,
                        }
                        for m in AUX_METRICS
                    },
                    "n_tool_calls": r.n_tool_calls,
                    "seconds": round(r.seconds, 3),
                    "answer": r.answer,
                }
                for r in results
            ],
        }
    return {
        "categories": categories,
        "dimensions": DIMENSIONS,
        "aux_metrics": AUX_METRICS,
        "models": models,
    }


def save_run(
    output_dir: str,
    results_by_model: dict[str, list[CaseResult]],
    cases: list[EvalCase],
    args_filter: str | None,
    *,
    grounding: str = "",
    threshold: float = 0.7,
) -> Path:
    """Archive a run to a fresh, timestamped directory and return its path.

    The directory is named `<iso-timestamp>_<git-hash>` so concurrent and
    historical runs never clobber each other.
    """
    now = datetime.now()
    git_hash = _git_hash()
    run_dir = Path(output_dir) / f"{now.strftime('%Y-%m-%dT%H-%M-%S')}_{git_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = _summarize(results_by_model, cases)
    payload = {
        "timestamp": now.isoformat(timespec="seconds"),
        "git_hash": git_hash,
        "n_cases": len(cases),
        "filter": args_filter,
        "grounding": grounding,
        "threshold": threshold,
        **summary,
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2))

    render_visualizations(summary, run_dir, subtitle=f"{payload['timestamp']}  ·  {git_hash}")
    return run_dir


def render_visualizations(summary: dict, run_dir: Path, subtitle: str) -> None:
    """Render a one-page performance dashboard PNG from a run summary."""
    # Headless backend: no display needed on CI or servers.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(summary["models"])
    categories = summary["categories"]
    dimensions = summary.get("dimensions", [])
    aux_metrics = summary.get("aux_metrics", [])

    # Distinguish ablation arms: no-tools (closed-book) series are drawn hatched.
    use_tools = {m: summary["models"][m].get("use_tools", True) for m in models}
    has_ablation = any(use_tools.values()) and not all(use_tools.values())

    def hatch(m: str) -> str:
        return "" if use_tools[m] else "//"

    def pass_rate(d: dict) -> float:
        return 100.0 * d["passed"] / d["total"] if d["total"] else 0.0

    fig, ((ax_overall, ax_dim), (ax_cat, ax_lat)) = plt.subplots(
        2, 2, figsize=(15, 10)
    )
    title = f"Wikipedia agent evals\n{subtitle}"
    if has_ablation:
        title += "   ·   hatched = no tools (closed-book)"
    fig.suptitle(title, fontsize=13)

    # 1) Overall mean score per model (0-1).
    overall = [summary["models"][m]["mean_score"] for m in models]
    bars = ax_overall.bar(models, overall, color="#4c72b0")
    for bar, m in zip(bars, models):
        bar.set_hatch(hatch(m))
    ax_overall.set_title("Overall mean score")
    ax_overall.set_ylabel("mean score (0-1)")
    ax_overall.set_ylim(0, 1.05)
    ax_overall.set_xticklabels(models, rotation=20, ha="right")
    for bar, v in zip(bars, overall):
        ax_overall.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2) Mean score by dimension, grouped by model — the new headline view.
    # Auxiliary metrics are appended with a trailing '*' so they're visible but
    # clearly flagged as not contributing to the overall mean.
    labels = list(dimensions) + [f"{a}*" for a in aux_metrics]
    n = len(labels)
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        md = summary["models"][m]
        scores = [md["dimensions"][d] for d in dimensions] + [
            md.get("aux", {}).get(a, 0.0) for a in aux_metrics
        ]
        offsets = [x + i * width for x in range(n)]
        ax_dim.bar(offsets, scores, width=width, label=m, hatch=hatch(m))
    dim_title = "Mean score by dimension"
    if aux_metrics:
        dim_title += "   (* = aux, not in overall)"
    ax_dim.set_title(dim_title)
    ax_dim.set_ylabel("mean score (0-1)")
    ax_dim.set_ylim(0, 1.05)
    ax_dim.set_xticks([x + width * (len(models) - 1) / 2 for x in range(n)])
    ax_dim.set_xticklabels(labels, rotation=20, ha="right")
    ax_dim.legend(fontsize=8)

    # 3) Per-category pass rate, grouped by model.
    n = len(categories)
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        rates = [pass_rate(summary["models"][m]["categories"][c]) for c in categories]
        offsets = [x + i * width for x in range(n)]
        ax_cat.bar(offsets, rates, width=width, label=m, hatch=hatch(m))
    ax_cat.set_title("Pass rate by category")
    ax_cat.set_ylabel("% passed")
    ax_cat.set_ylim(0, 105)
    ax_cat.set_xticks([x + width * (len(models) - 1) / 2 for x in range(n)])
    ax_cat.set_xticklabels(categories, rotation=20, ha="right")
    ax_cat.legend(fontsize=8)

    # 4) Average latency per model (tiebreaker when scores match).
    latency = [summary["models"][m]["avg_seconds"] for m in models]
    bars = ax_lat.bar(models, latency, color="#dd8452")
    for bar, m in zip(bars, models):
        bar.set_hatch(hatch(m))
    ax_lat.set_title("Avg latency per case")
    ax_lat.set_ylabel("seconds")
    ax_lat.set_xticklabels(models, rotation=20, ha="right")
    for bar, v in zip(bars, latency):
        ax_lat.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for ax in (ax_overall, ax_dim, ax_cat, ax_lat):
        ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(run_dir / "dashboard.png", dpi=120)
    plt.close(fig)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run Wikipedia agent evals.")
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Comma-separated models to compare. Accepts friendly names "
        f"({', '.join(MODELS)}) or full model ids. Default: all three.",
    )
    parser.add_argument(
        "--filter", help="Only run cases whose name contains this substring."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print each agent's full answer."
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory to archive each run under. Default: eval_results.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing results JSON and charts to disk.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Also run every case closed-book (tool disabled) for each model, "
        "in the same run, so the dashboard and report compare answers with vs "
        "without retrieval. The no-tools arm is a diagnostic baseline and is "
        "excluded from the pass/fail CI gate.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Model used by the LLM grounding judge. Accepts a friendly name "
        f"({', '.join(MODELS)}) or a full id. Default: {DEFAULT_JUDGE_MODEL}.",
    )
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="Skip the LLM grounding judge; use the deterministic proxy instead "
        "(no extra API calls).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Overall mean score a case must reach to count as passed (and for "
        "the CI exit code). Default: 0.7.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max evaluations (model×case pairs) running at once, i.e. the cap "
        "on requests in flight — the rate-limit lever. Default: 4. Raise it if "
        "your tier allows; lower it if you see 429s.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Max retries the Anthropic client makes per request. The SDK backs "
        "off exponentially (honoring retry-after) on 429/5xx, absorbing rate-"
        "limit responses. Default: 6.",
    )
    parser.add_argument(
        "--case-retries",
        type=int,
        default=4,
        help="Max attempts per case (agent+judge) if the SDK's own retries are "
        "exhausted by a transient error. Attempts share a cooldown gate so "
        "rate-limit retries don't stampede. Default: 4.",
    )
    args = parser.parse_args(argv)

    if args.case_retries < 1:
        print("Error: --case-retries must be at least 1.", file=sys.stderr)
        return 2

    if args.concurrency < 1:
        print("Error: --concurrency must be at least 1.", file=sys.stderr)
        return 2

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    # Resolve friendly names to model ids; pass through anything unrecognized.
    models = [MODELS.get(m.strip(), m.strip()) for m in args.models.split(",") if m.strip()]
    if not models:
        print("No models specified.", file=sys.stderr)
        return 2

    cases = CASES
    if args.filter:
        cases = [c for c in cases if args.filter in c.name]
    if not cases:
        print("No matching eval cases.", file=sys.stderr)
        return 2

    # Expand models into series: a tools arm always, plus a closed-book no-tools
    # arm per model under --ablation. Ordered tools-then-no-tools within each
    # model so the grouped charts place each model's two arms side by side.
    series: list[Series] = []
    for model in models:
        series.append(Series(model, use_tools=True))
        if args.ablation:
            series.append(Series(model, use_tools=False))

    # Bound concurrency to the work actually available — no point spinning up
    # more workers than there are (series, case) pairs.
    concurrency = min(args.concurrency, len(series) * len(cases))
    # The SDK's retry/backoff is the safety net beneath the concurrency cap.
    client = anthropic.Anthropic(max_retries=args.max_retries)
    judge_model = MODELS.get(args.judge_model, args.judge_model)
    # The judge reuses the same client; None disables it (proxy fallback).
    judge_client = None if args.no_grounding else client
    grounding_note = (
        "deterministic proxy" if args.no_grounding else f"LLM judge ({judge_model})"
    )
    ablation_note = "  ·  ablation: with vs without tools" if args.ablation else ""
    print(
        f"Running {len(cases)} case(s) across {len(series)} series "
        f"({', '.join(s.label for s in series)})\n"
        f"Grounding: {grounding_note}  ·  pass threshold: {args.threshold:.2f}"
        f"  ·  concurrency: {concurrency}{ablation_note}"
    )

    results_by_model = run_all(
        series,
        cases,
        client,
        judge_client=judge_client,
        judge_model=judge_model,
        threshold=args.threshold,
        concurrency=concurrency,
        case_retries=args.case_retries,
    )

    for s in series:
        print_model_report(s.label, results_by_model[s.label], args.verbose)

    if len(series) > 1:
        print_comparison(results_by_model, cases)

    if args.ablation:
        print_ablation(series, results_by_model)

    if not args.no_save:
        run_dir = save_run(
            args.output_dir,
            results_by_model,
            cases,
            args.filter,
            grounding=grounding_note,
            threshold=args.threshold,
        )
        print(f"\nArchived results + charts to {run_dir}/")

    # Surface transient infra errors loudly: they're excluded from the scores,
    # so without this note a contended run would look clean despite missing
    # data. Exit code 3 keeps CI from reading partial results as a clean pass.
    errored = [
        (lbl, r.case.name)
        for lbl, rs in results_by_model.items()
        for r in rs
        if r.errored
    ]
    if errored:
        print(
            f"\n⚠️  {len(errored)} case(s) excluded after transient errors "
            f"(rate limit / timeout / 5xx) survived retries:",
            file=sys.stderr,
        )
        for lbl, name in errored:
            print(f"     - {lbl} :: {name}", file=sys.stderr)
        print(
            "   These are NOT model failures and were dropped from the stats. "
            "Re-run (optionally with lower --concurrency or higher "
            "--case-retries) for complete coverage.",
            file=sys.stderr,
        )

    # Exit non-zero if any genuinely graded case failed — but only on the tools
    # arm(s). The closed-book baseline is a diagnostic, not a quality gate, so
    # its (necessarily low) scores never fail CI. Errored cases are excluded
    # from the gate (they aren't quality signals) but force exit code 3 so a
    # run with missing data is never mistaken for a clean pass.
    gated = [s.label for s in series if s.use_tools]
    all_passed = all(
        r.passed for lbl in gated for r in _scored(results_by_model[lbl])
    )
    if errored:
        return 3
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
