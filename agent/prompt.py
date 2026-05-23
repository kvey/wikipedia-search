"""System prompt and tool schema for the Wikipedia agent.

Kept separate from the agent loop so the prompt and tool surface can be edited
and evaluated independently of the orchestration code.
"""

MODEL = "claude-opus-4-7"

# Friendly aliases for the models the eval harness can compare. Maps a short
# name to the exact model id passed to the Messages API.
MODELS = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}

SYSTEM_PROMPT = """\
You are a research assistant that answers factual questions using Wikipedia.

Guidelines:
- When a question depends on facts you are not certain of, call the \
`search_wikipedia` tool before answering. Prefer searching over guessing.
- You may call the tool multiple times to follow up, refine a query, or check a \
related article before you have enough to answer.
- Base your answer on the search results. If the results do not contain the \
answer, say so plainly rather than inventing details.
- Answer concisely and directly. Cite the Wikipedia article title(s) you relied \
on.\
"""

# Closed-book counterpart used by the eval harness's `--ablation` arm: the same
# task with no search tool available, so the model must answer from its own
# parametric knowledge. Kept parallel to SYSTEM_PROMPT but without any reference
# to a tool (which would only confuse a model that has none), so the ablation
# isolates a single variable — the presence of retrieval.
NO_TOOLS_SYSTEM_PROMPT = """\
You are a research assistant that answers factual questions.

Guidelines:
- You do not have access to any search tools. Answer from your own knowledge.
- Answer concisely and directly.
- If you are not confident of a fact, say so plainly rather than inventing \
details.\
"""

# Tool schema passed to the Messages API. The name and parameters match
# `agent.wikipedia.search_wikipedia`.
TOOLS = [
    {
        "name": "search_wikipedia",
        "description": (
            "Search English Wikipedia for a query and return the top matching "
            "article titles, snippets, and a plain-text extract of the best "
            "match. Use this to look up facts, people, places, events, and "
            "definitions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'capital of Australia'.",
                }
            },
            "required": ["query"],
        },
    }
]
