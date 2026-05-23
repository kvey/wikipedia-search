"""Entrypoint for the Wikipedia agent.

Runs a manual multi-turn tool-use loop directly against the Anthropic Messages
API (no agent framework). Claude can call `search_wikipedia` as many times as it
needs before producing a final answer.

Usage:
    export ANTHROPIC_API_KEY=...
    uv run agent/main.py "What is the capital of Australia?"

Or import `run_agent` from another module (e.g. the eval harness).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

# Make the repo root importable whether this file is run as a script
# (`uv run agent/main.py`) or as a module (`python -m agent.main`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from agent.env import load_env
from agent.prompt import MODEL, NO_TOOLS_SYSTEM_PROMPT, SYSTEM_PROMPT, TOOLS
from agent.trace import DEFAULT_TRACE_DIR, RunTrace
from agent.wikipedia import search_wikipedia

# Load <repo>/.env at import time so the CLI and the eval harness (which imports
# this module) both pick up ANTHROPIC_API_KEY without an explicit export.
load_env()

# Maps tool names from the schema to their Python implementations.
TOOL_IMPLEMENTATIONS = {
    "search_wikipedia": search_wikipedia,
}

MAX_TURNS = 10  # safety bound on the tool-use loop


@dataclass
class AgentResult:
    """The outcome of a single agent run."""

    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    turns: int = 0
    trace_path: str | None = None


def _run_tool(name: str, tool_input: dict) -> str:
    """Dispatch a tool call to its implementation, capturing errors as strings."""
    impl = TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        return f"Error: unknown tool {name!r}"
    try:
        return impl(**tool_input)
    except Exception as exc:  # surface failures to the model rather than crashing
        return f"Error running {name}: {exc}"


def run_agent(
    question: str,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
    verbose: bool = False,
    trace: bool = True,
    trace_dir: str = DEFAULT_TRACE_DIR,
    on_tool_call: Callable[[dict], None] | None = None,
    force_first_tool: bool = True,
    use_tools: bool = True,
) -> AgentResult:
    """Answer a question, using Wikipedia search as needed.

    Args:
        question: The user's question.
        client: An optional Anthropic client (one is created if omitted).
        model: The Claude model id to use (default `agent.prompt.MODEL`).
        verbose: If True, print the loop's progress to stderr.
        trace: If True, write a JSON trace of the run to `trace_dir`.
        trace_dir: Directory for trace files (default from AGENT_TRACE_DIR).
        on_tool_call: Optional callback invoked after each tool executes, with a
            {"name", "input", "output"} dict. Used by the CLI to print activity.
        force_first_tool: If True, require a `search_wikipedia` call on the first
            turn so the agent always grounds its answer in Wikipedia. Subsequent
            turns use the default auto tool choice so the model can stop and
            answer. (Opus 4.7 otherwise often answers confident questions from
            its own knowledge without searching.) Ignored when `use_tools` is
            False.
        use_tools: If True (default), expose the `search_wikipedia` tool. If
            False, run closed-book: no tool is offered and the model must answer
            from its own parametric knowledge. This is the eval harness's
            ablation baseline — same question, retrieval removed — and it swaps
            in `NO_TOOLS_SYSTEM_PROMPT` so the prompt never references a tool the
            model doesn't have.

    Returns:
        An AgentResult with the final answer, a trace of tool calls, and the
        path to the written JSON trace (if tracing is enabled).
    """
    client = client or anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": question}]
    tool_calls: list[dict] = []
    run_trace = RunTrace(question=question, model=model)

    def finish(answer: str, turns: int, stop_reason: str | None) -> AgentResult:
        run_trace.answer = answer
        run_trace.stop_reason = stop_reason
        path = run_trace.write(trace_dir) if trace else None
        if trace and verbose:
            print(f"[agent] trace written to {path}", file=sys.stderr)
        return AgentResult(
            answer=answer, tool_calls=tool_calls, turns=turns, trace_path=path
        )

    system_prompt = SYSTEM_PROMPT if use_tools else NO_TOOLS_SYSTEM_PROMPT

    for turn in range(1, MAX_TURNS + 1):
        create_kwargs = {
            "model": model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages,
        }
        if use_tools:
            create_kwargs["tools"] = TOOLS
            # Force a search on the first turn so every answer is grounded in
            # Wikipedia; let later turns choose freely so the model can finish.
            if turn == 1 and force_first_tool:
                create_kwargs["tool_choice"] = {
                    "type": "tool",
                    "name": "search_wikipedia",
                }

        response = client.messages.create(**create_kwargs)
        run_trace.record_turn(turn, response)

        # Append the assistant turn (including any tool_use blocks) to history.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            if verbose:
                print(f"[agent] finished in {turn} turn(s)", file=sys.stderr)
            return finish(answer.strip(), turn, response.stop_reason)

        # Execute every tool the model requested this turn.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output = _run_tool(block.name, block.input)
            call_record = {"name": block.name, "input": block.input, "output": output}
            tool_calls.append(call_record)
            run_trace.record_tool_result(block.name, block.input, output)
            if on_tool_call is not None:
                on_tool_call(call_record)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return finish(
        "(stopped: reached max tool-use turns without a final answer)",
        MAX_TURNS,
        "max_turns",
    )


def main(argv: list[str]) -> int:
    if not argv:
        print('Usage: uv run agent/main.py "your question"', file=sys.stderr)
        return 2
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    question = " ".join(argv)

    def print_tool_call(call: dict) -> None:
        """Print each tool use, its arguments, and its result as it executes."""
        args = ", ".join(f"{k}={v!r}" for k, v in call["input"].items())
        print(f"\n→ tool call: {call['name']}({args})", file=sys.stderr)
        output = call["output"]
        # Keep the terminal readable; the full result is in the JSON trace.
        preview = output if len(output) <= 800 else output[:800] + "\n  …(truncated)"
        indented = "\n".join(f"  {line}" for line in preview.splitlines())
        print(f"← result:\n{indented}", file=sys.stderr)

    result = run_agent(question, verbose=True, on_tool_call=print_tool_call)

    if result.tool_calls:
        print(f"\n[{len(result.tool_calls)} Wikipedia search(es)]", file=sys.stderr)
    print("\n=== Answer ===", file=sys.stderr)
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
