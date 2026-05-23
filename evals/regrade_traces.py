"""Retroactively score recorded agent runs for entity coverage.

Every agent run is archived as a self-contained JSON trace under `traces/`
(see `agent/trace.py`), holding the final `answer` and every tool call's full
`output` (the retrieved Wikipedia text). That's exactly the two things the
`entity_coverage` metric needs — so we can compute it *after the fact* over the
entire run history, with no agent re-run and no API calls.

This is the cheap, deterministic complement to the live LLM `grounding` judge:
it flags answers whose named entities / numbers never appear in any retrieved
passage — a mechanical hallucination signal — across runs that were graded long
before this metric existed.

Usage:
    uv run evals/regrade_traces.py                     # score every trace
    uv run evals/regrade_traces.py --traces-dir traces # explicit dir
    uv run evals/regrade_traces.py --min-score 1.0     # only flag <1.0 (likely
                                                       # ungrounded entities)
    uv run evals/regrade_traces.py --json out.json     # also write a report
    uv run evals/regrade_traces.py --chart out.png     # render a PNG dashboard
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Make the repo root importable whether run as a script or a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.trace import DEFAULT_TRACE_DIR
from evals.grading import grade_entity_coverage


def _retrieved_text(trace: dict) -> str:
    """Concatenate tool-call outputs exactly as the live grader does."""
    return "\n\n".join(c.get("output", "") for c in trace.get("tool_calls", []))


def score_trace(trace: dict) -> dict:
    """Score one loaded trace dict for entity coverage."""
    answer = trace.get("answer") or ""
    retrieved = _retrieved_text(trace)
    ds = grade_entity_coverage(answer, retrieved)
    return {
        "run_id": trace.get("run_id"),
        "model": trace.get("model"),
        "question": trace.get("question", ""),
        "n_tool_calls": len(trace.get("tool_calls", [])),
        "score": round(ds.score, 3),
        "reasons": ds.reasons,
    }


def load_traces(traces_dir: str) -> list[tuple[str, dict]]:
    """Load every `*.json` trace in `traces_dir`, sorted by filename (≈ time)."""
    out: list[tuple[str, dict]] = []
    for path in sorted(glob.glob(os.path.join(traces_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                out.append((os.path.basename(path), json.load(f)))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skipping unreadable trace {path}: {exc}", file=sys.stderr)
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def render_chart(scored: list[dict], path: str) -> None:
    """Render a small dashboard of retroactive entity_coverage to `path`.

    Closed-book traces (no retrieval) score 0.0 by construction, so they're
    excluded from the means and shown only as a separate count — folding them in
    would understate how grounded the *retrieval-bearing* answers actually were.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    retr = [s for s in scored if s["n_tool_calls"] > 0]
    models = sorted({s["model"] for s in retr})
    by_model = {m: [s["score"] for s in retr if s["model"] == m] for m in models}

    fig, (ax_bar, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Retroactive entity_coverage over {len(retr)} retrieval-bearing trace(s)"
        f"   ·   {len(scored) - len(retr)} closed-book excluded",
        fontsize=12,
    )

    # 1) Mean entity_coverage per model (with per-bar trace counts).
    means = [_mean(by_model[m]) for m in models]
    bars = ax_bar.bar(models, means, color="#4c72b0")
    ax_bar.set_title("Mean entity_coverage by model")
    ax_bar.set_ylabel("mean entity_coverage (0-1)")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_xticklabels(models, rotation=20, ha="right")
    for bar, m, v in zip(bars, models, means):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.01,
            f"{v:.2f}\n(n={len(by_model[m])})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 2) Distribution of scores across retrieval-bearing traces.
    ax_hist.hist(
        [s["score"] for s in retr],
        bins=[i / 10 for i in range(11)],
        color="#dd8452",
        edgecolor="white",
    )
    ax_hist.set_title("Distribution of entity_coverage")
    ax_hist.set_xlabel("entity_coverage")
    ax_hist.set_ylabel("# traces")
    ax_hist.set_xlim(0, 1)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Retroactively score recorded traces for entity coverage."
    )
    parser.add_argument(
        "--traces-dir",
        default=DEFAULT_TRACE_DIR,
        help=f"Directory of trace JSON files (default: {DEFAULT_TRACE_DIR}).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Only print traces scoring strictly below this (e.g. 1.0 to surface "
        "answers with an uncovered entity). Aggregates still cover all traces.",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Also write the full per-trace report as JSON to PATH.",
    )
    parser.add_argument(
        "--chart",
        metavar="PATH",
        default=None,
        help="Render a PNG dashboard (mean by model + score distribution) to PATH.",
    )
    args = parser.parse_args(argv)

    traces = load_traces(args.traces_dir)
    if not traces:
        print(f"No traces found in {args.traces_dir!r}.", file=sys.stderr)
        return 1

    scored = [score_trace(t) for _, t in traces]
    retr = [s for s in scored if s["n_tool_calls"] > 0]
    no_retrieval = len(scored) - len(retr)
    # Headline mean is over retrieval-bearing traces; closed-book runs score 0.0
    # by construction and would only understate how grounded real answers were.
    mean = _mean([s["score"] for s in retr])
    models = sorted({s["model"] for s in retr})
    by_model = {
        m: _mean([s["score"] for s in retr if s["model"] == m]) for m in models
    }

    shown = scored
    if args.min_score is not None:
        shown = [s for s in scored if s["score"] < args.min_score]

    qw = 50
    print(f"\nEntity coverage over {len(scored)} trace(s) in {args.traces_dir!r}")
    print("=" * 96)
    print(f"{'run_id':14} {'score':>6} {'tools':>5}  question")
    print("-" * 96)
    for s in shown:
        q = s["question"].replace("\n", " ")
        q = q[: qw - 1] + "…" if len(q) > qw else q
        print(f"{(s['run_id'] or '?'):14} {s['score']:>6.2f} {s['n_tool_calls']:>5}  {q}")
        for reason in s["reasons"]:
            print(f"{'':14}        - {reason}")
    if args.min_score is not None:
        print(f"\n  shown {len(shown)}/{len(scored)} below {args.min_score}")

    print("-" * 96)
    print(
        f"  mean entity_coverage: {mean:.3f}  "
        f"(over {len(retr)} retrieval-bearing traces)"
    )
    for m in models:
        print(f"    - {m}: {by_model[m]:.3f}")
    if no_retrieval:
        print(
            f"  note: {no_retrieval} closed-book trace(s) excluded "
            f"(no retrieval → 0.0 by definition)"
        )

    if args.json:
        payload = {
            "traces_dir": args.traces_dir,
            "n_traces": len(scored),
            "n_retrieval_bearing": len(retr),
            "no_retrieval": no_retrieval,
            "mean_entity_coverage": round(mean, 4),
            "mean_by_model": {m: round(by_model[m], 4) for m in models},
            "traces": scored,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  wrote report to {args.json}")

    if args.chart:
        render_chart(scored, args.chart)
        print(f"  wrote chart to {args.chart}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
