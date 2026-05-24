"""Run tracing for the Wikipedia agent.

Each agent run is captured as a single structured JSON file written to a trace
directory (default `traces/`, override with the `AGENT_TRACE_DIR` env var). A
trace records the question, every model turn (text, tool calls, token usage),
the resulting tool results, and the final answer — enough to reconstruct and
debug a run after the fact.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone

DEFAULT_TRACE_DIR = os.environ.get("AGENT_TRACE_DIR", "traces")


class RunTrace:
    """Accumulates events for one agent run, then writes them as JSON."""

    def __init__(self, question: str, model: str):
        self.run_id = uuid.uuid4().hex[:12]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._start = time.monotonic()
        self.question = question
        self.model = model
        self.turns: list[dict] = []
        self.tool_calls: list[dict] = []
        self.answer: str | None = None
        self.stop_reason: str | None = None
        self.input_tokens = 0
        self.output_tokens = 0

    def record_turn(self, turn: int, response) -> None:
        """Capture one assistant turn from a Messages API response."""
        text = "".join(b.text for b in response.content if b.type == "text")
        tool_uses = [
            {"id": b.id, "name": b.name, "input": b.input}
            for b in response.content
            if b.type == "tool_use"
        ]
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        self.input_tokens += in_tok
        self.output_tokens += out_tok

        self.turns.append(
            {
                "turn": turn,
                "stop_reason": response.stop_reason,
                "text": text,
                "tool_uses": tool_uses,
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
            }
        )

    def record_tool_result(self, name: str, tool_input: dict, output: str) -> None:
        """Capture the result of executing one tool call."""
        self.tool_calls.append(
            {"name": name, "input": tool_input, "output": output}
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "model": self.model,
            "question": self.question,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "turns_taken": len(self.turns),
            "tool_call_count": len(self.tool_calls),
            # Search *steps* (search_wikipedia calls) vs total tool calls: with
            # query fan-out and get_article drill-downs these now differ.
            "search_count": sum(
                1 for c in self.tool_calls if c.get("name") == "search_wikipedia"
            ),
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "elapsed_seconds": round(time.monotonic() - self._start, 3),
            "turns": self.turns,
            "tool_calls": self.tool_calls,
        }

    def write(self, trace_dir: str = DEFAULT_TRACE_DIR) -> str:
        """Write the trace to `<trace_dir>/<timestamp>-<run_id>.json`."""
        os.makedirs(trace_dir, exist_ok=True)
        stamp = self.started_at.replace(":", "").replace("-", "").split(".")[0]
        path = os.path.join(trace_dir, f"{stamp}-{self.run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path
