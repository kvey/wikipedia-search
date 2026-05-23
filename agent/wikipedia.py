"""Thin wrapper around the Wikipedia MediaWiki API.

Exposes a single `search_wikipedia(query)` function that the agent calls as a
tool. It performs a full-text search and returns the top matching articles
along with a plain-text extract of the best match, so Claude has enough context
to answer without a second round trip.
"""

from __future__ import annotations

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
# The MediaWiki API etiquette guidelines ask for a descriptive User-Agent.
USER_AGENT = "wikipedia-agent/0.1 (Anthropic take-home; contact: agent@example.com)"

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _get(params: dict, timeout: float = 15.0) -> dict:
    """Issue a GET against the MediaWiki API and return the parsed JSON."""
    params = {**params, "format": "json"}
    resp = _session.get(API_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _search_titles(query: str, limit: int) -> list[dict]:
    """Run a full-text search, returning a list of {title, snippet} dicts."""
    data = _get(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            # Strip HTML markup out of the returned snippets.
            "srprop": "snippet",
        }
    )
    results = []
    for hit in data.get("query", {}).get("search", []):
        snippet = (
            hit.get("snippet", "")
            .replace('<span class="searchmatch">', "")
            .replace("</span>", "")
        )
        results.append({"title": hit["title"], "snippet": snippet})
    return results


def _get_extract(title: str, chars: int) -> str:
    """Fetch the intro plain-text extract for a given article title."""
    data = _get(
        {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "explaintext": 1,  # plain text, no HTML
            "exintro": 1,  # only the lead section
            "exchars": chars,  # cap the length
            "redirects": 1,  # follow redirects
        }
    )
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "extract" in page:
            return page["extract"]
    return ""


def search_wikipedia(query: str, results: int = 4, extract_chars: int = 1500) -> str:
    """Search Wikipedia and return a formatted summary of the top results.

    Args:
        query: The search query.
        results: How many article titles/snippets to return.
        extract_chars: Max characters of the lead extract for the top result.

    Returns:
        A human-readable string with the top hits and the lead extract of the
        best match. Returns an explanatory message if nothing is found or the
        API call fails.
    """
    try:
        hits = _search_titles(query, results)
    except requests.RequestException as exc:
        return f"Wikipedia search failed: {exc}"

    if not hits:
        return f"No Wikipedia articles found for query: {query!r}"

    lines = [f"Top {len(hits)} Wikipedia results for {query!r}:\n"]
    for i, hit in enumerate(hits, 1):
        lines.append(f"{i}. {hit['title']}")
        if hit["snippet"]:
            lines.append(f"   {hit['snippet']}")

    try:
        extract = _get_extract(hits[0]["title"], extract_chars)
    except requests.RequestException as exc:
        extract = f"(failed to fetch extract: {exc})"

    if extract:
        lines.append(f"\nExtract from '{hits[0]['title']}':\n{extract}")

    return "\n".join(lines)
