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
- `search_wikipedia` accepts several queries at once. When the entity is \
ambiguous or could be phrased multiple ways, pass several phrasings in a single \
call (e.g. ["Nobel Prize Mathematics 2000", "Fields Medal 2000", "Abel \
Prize"]) — different wordings surface different articles. Prefer one wide search \
over many narrow ones.
- If a result's title or snippet looks like it holds the answer but the \
returned extract doesn't contain the specific fact, call `get_article` on that \
exact title to read more of it (optionally a section) before searching again.
- You may call the tools multiple times to follow up or check a related \
article, but prefer one fan-out search plus a targeted `get_article` over many \
repeated searches.
- Base your answer on the retrieved text. If it does not contain the answer, \
say so plainly rather than inventing details.
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

# Tool schemas passed to the Messages API. Names and parameters match the
# implementations in `agent.wikipedia`.
TOOLS = [
    {
        "name": "search_wikipedia",
        "description": (
            "Search English Wikipedia and return the top matching article "
            "titles, snippets, and lead extracts of the best matches. Accepts "
            "one or more queries: pass several differently-phrased queries in a "
            "single call to cast a wider net (results are merged and "
            "deduplicated). Use this to look up facts, people, places, events, "
            "and definitions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": (
                        "One or more search queries. Provide several phrasings "
                        "of the same information need in one call (e.g. "
                        "['capital of Australia', 'Canberra']); prefer this over "
                        "issuing separate sequential searches."
                    ),
                }
            },
            "required": ["queries"],
        },
    },
    {
        "name": "get_article",
        "description": (
            "Fetch the fuller plain-text body of ONE Wikipedia article by its "
            "exact title. Use after search_wikipedia when a result's snippet "
            "looks relevant but the lead extract didn't contain the specific "
            "fact you need (e.g. a detail buried in the body, or a result that "
            "wasn't the top hit). Optionally pass a section heading to fetch "
            "just that section."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Exact article title from a search result.",
                },
                "section": {
                    "type": "string",
                    "description": (
                        "Optional section heading to fetch just that section, "
                        "e.g. 'Early life'."
                    ),
                },
            },
            "required": ["title"],
        },
    },
]
