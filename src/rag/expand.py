"""Query expansion via a local Ollama LLM.

This step runs *before* retrieval. It asks the LLM for a small number of
alternative queries / keyword phrases that are then fed to the FTS backend
(benefiting most from lexical variety). The original query is always retained
as the first element of the returned list.

Design goals:
- Isolated and unit-testable (no module-level heavy imports).
- Fail-safe: any error, empty/invalid output, or timeout silently falls back
  to ``[query]``.
- Optimised for small Qwen models (0.6B-3B): directive prompt, temperature ~0,
  no streaming, strict line-based parsing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    EXPANSION_NUM_QUERIES,
    EXPANSION_TEMPERATURE,
    EXPANSION_TIMEOUT,
)


EXPANSION_PROMPT = """\
You expand a user question into alternative search queries for a knowledge base.
Rules:
- Output ONE alternative query per line.
- Each line must be a short keyword phrase or rephrased question.
- No numbering, no bullets, no quotes, no preamble, no explanation.
- Maximum {n} lines.
- Do NOT repeat the original question.

Question: {query}
Alternatives:
"""


def _default_call_ollama(prompt: str) -> str:
    """Default LLM call using requests to the /api/generate endpoint."""
    import requests  # imported lazily so the module is import-safe in tests

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": EXPANSION_TEMPERATURE},
        },
        timeout=EXPANSION_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "") or ""


def _parse_alternatives(raw: str, limit: int) -> List[str]:
    """Parse raw LLM output into a de-duplicated list of alternatives.

    Strips markdown fences, bullets, numbering and surrounding whitespace.
    Preserves order of first appearance. Caps at ``limit`` items.
    """
    text = raw or ""
    # Strip markdown code fences if present
    if "```" in text:
        parts = text.split("```")
        # take the first fenced block that is non-empty
        for part in parts[1:]:
            cleaned = part
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            if cleaned:
                text = cleaned
                break

    results: List[str] = []
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip common bullet / numbering prefixes
        for prefix in ("- ", "* ", "+ ", "• "):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        # strip leading "N." / "N)" numbering
        if len(line) > 2 and line[0].isdigit() and line[1] in ".)":
            line = line[2:].strip()
        # strip surrounding quotes
        if len(line) >= 2 and line[0] in "\"'" and line[-1] == line[0]:
            line = line[1:-1].strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(line)
        if len(results) >= limit:
            break
    return results


def expand_query(
    query: str,
    generate_fn: Optional[Callable[[str], str]] = None,
    num_queries: int = EXPANSION_NUM_QUERIES,
) -> List[str]:
    """Expand a query into alternatives, always retaining the original first.

    Args:
        query: The original user query.
        generate_fn: Callable mapping a prompt string to raw LLM text. Defaults
            to ``_default_call_ollama``. Inject a stub in tests.
        num_queries: Max number of alternatives to request/keep.

    Returns:
        ``[query, alt1, alt2, ...]``. On any failure (empty output, timeout,
        exception, invalid parse) returns ``[query]``.
    """
    if query is None:
        return []
    if not query.strip():
        return [query]

    call = generate_fn if generate_fn is not None else _default_call_ollama
    prompt = EXPANSION_PROMPT.format(query=query, n=num_queries)

    try:
        raw = call(prompt)
    except Exception:
        # timeout, connection error, JSON error, etc. -> silent fallback
        return [query]

    alternatives = _parse_alternatives(raw, num_queries)

    # Drop any alternative that is (case-insensitively) identical to original
    original_lower = query.strip().lower()
    alternatives = [a for a in alternatives if a.lower() != original_lower]

    if not alternatives:
        return [query]

    return [query, *alternatives]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Expand a query via Ollama.")
    parser.add_argument("query", help="Query to expand.")
    args = parser.parse_args()
    for q in expand_query(args.query):
        print(q)
