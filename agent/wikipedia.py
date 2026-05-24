"""Thin wrapper around the Wikipedia MediaWiki API.

Exposes two tools the agent calls:

- ``search_wikipedia(queries)`` — runs *one or more* full-text searches and
  returns a merged, deduplicated set of top articles with lead extracts for the
  best few. Accepting several phrasings in a single call lets the agent cast a
  wider net up front (different wordings surface different articles) instead of
  re-querying turn after turn.
- ``get_article(title)`` — fetches a fuller plain-text body (or a single
  section) of one article, for when a result looks relevant but its snippet or
  lead extract didn't contain the specific fact needed.
"""

from __future__ import annotations

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
# The MediaWiki API etiquette guidelines ask for a descriptive User-Agent.
USER_AGENT = "wikipedia-agent/0.1 (Anthropic take-home; contact: agent@example.com)"

# Cap on how many distinct phrasings a single search_wikipedia call will run, so
# a model that returns a long `queries` list can't fan out into a flood of API
# requests.
MAX_QUERIES = 5

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


def _get_full_extract(title: str) -> str:
    """Fetch the full plain-text body of an article (intro + sections).

    Unlike `_get_extract` this drops `exintro`, so the whole article body comes
    back, with section headings rendered as plain lines (`exsectionformat`).
    `exchars` is deliberately omitted — it forces intro-only behavior — so the
    caller truncates the (possibly section-sliced) text instead.
    """
    data = _get(
        {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "explaintext": 1,  # plain text, no HTML
            "exsectionformat": "plain",  # section headers as plain text lines
            "redirects": 1,  # follow redirects
        }
    )
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "extract" in page:
            return page["extract"]
    return ""


def _truncate(text: str, max_chars: int) -> str:
    """Trim `text` to at most `max_chars`, preferring a paragraph boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer to break at the last blank line so we don't end mid-paragraph; only
    # do so if it doesn't throw away too much of the budget.
    para = cut.rfind("\n\n")
    if para >= max_chars * 0.6:
        cut = cut[:para]
    return cut.rstrip() + "\n…(truncated)"


def search_wikipedia(
    queries: str | list[str],
    results_per_query: int = 4,
    extract_chars: int = 1200,
    extract_top_n: int = 2,
) -> str:
    """Search Wikipedia with one or more queries and return a merged summary.

    Args:
        queries: A single query string, or a list of differently-phrased
            queries. Each is searched and the hits are merged into one ranked
            list, deduplicated by article title (an article's best/lowest rank
            across the queries wins). Passing several phrasings in one call casts
            a wider net than searching one phrasing at a time.
        results_per_query: Title/snippet hits to request per query.
        extract_chars: Max characters of the lead extract per surfaced article.
        extract_top_n: How many of the merged top articles get a lead extract
            (not just the single best match) — so a fact living in the #2 or #3
            hit is visible without a follow-up search.

    Returns:
        A human-readable string with the merged top hits (with provenance) and
        lead extracts for the best `extract_top_n`. Returns an explanatory
        message if nothing is found or every query failed.
    """
    # Normalize to a clean, deduped, bounded list of query strings.
    if isinstance(queries, str):
        queries = [queries]
    seen_q: dict[str, str] = {}  # lowercase -> original, preserves order + dedupes
    for q in queries:
        q = (q or "").strip()
        if q and q.lower() not in seen_q:
            seen_q[q.lower()] = q
    query_list = list(seen_q.values())[:MAX_QUERIES]
    if not query_list:
        return "No search query provided."

    # Run each query, merging hits by title. `best_rank` is the lowest (best)
    # position the title reached in any query; `found_by` records which queries
    # surfaced it (a title found by several phrasings is a stronger signal).
    merged: dict[str, dict] = {}
    errors: list[str] = []
    for q in query_list:
        try:
            hits = _search_titles(q, results_per_query)
        except requests.RequestException as exc:
            errors.append(f"{q!r}: {exc}")
            continue
        for rank, hit in enumerate(hits, 1):
            title = hit["title"]
            entry = merged.get(title)
            if entry is None:
                merged[title] = {
                    "title": title,
                    "snippet": hit["snippet"],
                    "best_rank": rank,
                    "found_by": [q],
                }
            else:
                entry["best_rank"] = min(entry["best_rank"], rank)
                entry["found_by"].append(q)
                if not entry["snippet"] and hit["snippet"]:
                    entry["snippet"] = hit["snippet"]

    if not merged:
        if errors:
            return "Wikipedia search failed for all queries: " + "; ".join(errors)
        return f"No Wikipedia articles found for: {query_list!r}"

    # Best rank first; break ties toward titles surfaced by more phrasings.
    ranked = sorted(merged.values(), key=lambda e: (e["best_rank"], -len(e["found_by"])))

    queries_note = ", ".join(repr(q) for q in query_list)
    lines = [f"Searched {len(query_list)} query/queries: {queries_note}\n"]
    if errors:
        lines.append(f"(note: {len(errors)} query/queries failed: {'; '.join(errors)})\n")
    lines.append("Merged top results:")
    for i, entry in enumerate(ranked, 1):
        found = ", ".join(repr(q) for q in entry["found_by"])
        lines.append(f"{i}. {entry['title']}  [found by: {found}]")
        if entry["snippet"]:
            lines.append(f"   {entry['snippet']}")

    for entry in ranked[: max(1, extract_top_n)]:
        try:
            extract = _get_extract(entry["title"], extract_chars)
        except requests.RequestException as exc:
            extract = f"(failed to fetch extract: {exc})"
        if extract:
            lines.append(f"\nExtract from '{entry['title']}':\n{extract}")

    return "\n".join(lines)


def _slice_section(text: str, section: str) -> str | None:
    """Return the lines of `text` under a heading matching `section`.

    `_get_full_extract` renders section headings as their own plain lines, so a
    section runs from its heading to the next short heading-like line. Matching
    is case-insensitive and forgiving of surrounding whitespace. Returns None if
    no heading matches.
    """
    lines = text.splitlines()
    target = section.strip().lower()
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == target:
            start = i
            break
    if start is None:
        return None
    body = [lines[start]]
    for line in lines[start + 1 :]:
        stripped = line.strip()
        # A short, non-empty line with no sentence punctuation looks like the
        # next section heading — stop there.
        if stripped and len(stripped) < 60 and not stripped.endswith((".", ":", "?", "!")):
            break
        body.append(line)
    return "\n".join(body).strip()


def get_article(title: str, section: str | None = None, max_chars: int = 6000) -> str:
    """Fetch a fuller plain-text extract of one Wikipedia article.

    Use after `search_wikipedia` when a result's title or snippet looks like it
    holds the answer but the lead extract didn't contain the specific fact.

    Args:
        title: Exact article title (as returned by `search_wikipedia`).
        section: Optional section heading; when given, only that section's text
            is returned (falling back to the full article if it isn't found).
        max_chars: Hard cap on returned characters.

    Returns:
        The article (or section) plain text, or an explanatory message on
        failure / empty result.
    """
    title = (title or "").strip()
    if not title:
        return "No article title provided."
    try:
        text = _get_full_extract(title)
    except requests.RequestException as exc:
        return f"Failed to fetch article {title!r}: {exc}"
    if not text:
        return f"No Wikipedia article content found for {title!r}."

    if section:
        sliced = _slice_section(text, section)
        if sliced:
            return (
                f"Article '{title}', section {section!r}:\n"
                f"{_truncate(sliced, max_chars)}"
            )
        return (
            f"(section {section!r} not found in '{title}'; returning the full "
            f"article)\n\nArticle '{title}':\n{_truncate(text, max_chars)}"
        )

    return f"Article '{title}':\n{_truncate(text, max_chars)}"
