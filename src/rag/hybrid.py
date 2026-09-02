"""Hybrid retrieval: concurrent FAISS + FTS with Reciprocal Rank Fusion.

Flow:
1. Expand the query via ``rag.expand.expand_query`` (original always retained).
2. Run FAISS (on the original query) and FTS (on the expanded queries)
   concurrently using ``ThreadPoolExecutor``.
3. Fuse the two rankings with independent RRF:
       score(d) = sum( 1 / (RRF_K + rank(d)) )
   where ``rank`` is the 0-based position in each backend's result list.
4. Deduplicate by stable chunk ``uid``, sort by descending score, return top-K.

Robustness:
- A failure in one backend must NOT discard results from the other. Each backend
  runs inside a try/except and returns ``[]`` on failure.
- If both backends fail (or return nothing), an empty list is returned; the
  caller (``rag.query.retrieve``) falls back to FAISS-only retrieval.

The FAISS/FTS/expand backends are injectable for unit testing so tests never
touch Ollama, embeddings, or the filesystem.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RRF_K, HYBRID_TOP_K, FTS_DB_PATH


# Type aliases for injected backends
ExpandFn = Callable[[str], List[str]]
RetrieveFn = Callable[[str], List[Dict[str, Any]]]
SearchFtsFn = Callable[..., List[Dict[str, Any]]]


def rrf_fuse(
    rankings: Sequence[Sequence[Dict[str, Any]]],
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: Iterable of ranked lists. Each item is a dict with a ``uid``.
        k: RRF constant (default 60). score(d) = sum( 1 / (k + rank(d)) ).

    Returns:
        Deduplicated (by uid), score-sorted list of chunk dicts. Each result has
        its ``rrf_score`` field set. Original chunk metadata is preserved.
    """
    scores: Dict[str, float] = {}
    best: Dict[str, Dict[str, Any]] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            uid = doc.get("uid")
            if uid is None:
                # synthesise a uid from source + chunk_id for legacy chunks
                uid = f"{doc.get('source', '')}::{doc.get('chunk_id', '')}"
            contribution = 1.0 / (k + rank)
            scores[uid] = scores.get(uid, 0.0) + contribution
            if uid not in best:
                best[uid] = dict(doc)
                best[uid]["uid"] = uid

    fused = []
    for uid, score in scores.items():
        doc = dict(best[uid])
        doc["rrf_score"] = score
        fused.append(doc)

    fused.sort(key=lambda d: d["rrf_score"], reverse=True)
    return fused


def _ensure_uid(chunk: Dict[str, Any]) -> str:
    """Get or synthesise a stable uid for a chunk dict."""
    uid = chunk.get("uid")
    if uid is not None:
        return uid
    return f"{chunk.get('source', '')}::{chunk.get('chunk_id', '')}"


def _run_faiss(query: str, faiss_fn: Optional[RetrieveFn]) -> List[Dict[str, Any]]:
    """Run the FAISS backend, returning [] on any failure."""
    if faiss_fn is None:
        try:
            from rag.query import retrieve_faiss as _retrieve_faiss
            faiss_fn = _retrieve_faiss
        except Exception:
            return []
    try:
        results = faiss_fn(query) or []
        # Ensure every chunk has a uid for dedup
        for r in results:
            if "uid" not in r:
                r["uid"] = _ensure_uid(r)
        return results
    except Exception:
        return []


def _run_fts(
    queries: List[str],
    fts_fn: Optional[SearchFtsFn],
    top_k: int,
    db_path: Optional[str | Path],
) -> List[Dict[str, Any]]:
    """Run the FTS backend across expanded queries, returning [] on failure."""
    if fts_fn is None:
        try:
            from rag.fts import search_fts as _search_fts
            fts_fn = _search_fts
        except Exception:
            return []
    seen: Dict[str, Dict[str, Any]] = {}
    combined: List[Dict[str, Any]] = []
    for q in queries:
        try:
            hits = fts_fn(q, top_k=top_k, db_path=db_path) if db_path is not None else fts_fn(q, top_k=top_k)
        except TypeError:
            # backend doesn't accept db_path
            try:
                hits = fts_fn(q, top_k=top_k)
            except Exception:
                hits = []
        except Exception:
            hits = []
        for hit in hits:
            uid = hit.get("uid") or _ensure_uid(hit)
            if uid in seen:
                continue
            seen[uid] = {**hit, "uid": uid}
            combined.append({**hit, "uid": uid})
    return combined


def retrieve_hybrid(
    query: str,
    expand_fn: Optional[ExpandFn] = None,
    faiss_fn: Optional[RetrieveFn] = None,
    fts_fn: Optional[SearchFtsFn] = None,
    top_k: Optional[int] = None,
    rrf_k: int = RRF_K,
    db_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Hybrid retrieval: expand -> concurrent FAISS+FTS -> RRF fusion.

    Args:
        query: Original user query.
        expand_fn: Query-expansion callable. Defaults to ``rag.expand.expand_query``.
        faiss_fn: FAISS retrieval callable ``(query) -> list[chunk]``.
        fts_fn: FTS search callable ``(query, top_k, db_path?) -> list[hit]``.
        top_k: Final number of fused results. Defaults to ``config.HYBRID_TOP_K``.
        rrf_k: RRF denominator constant.
        db_path: Optional FTS database path override.

    Returns:
        Fused, deduplicated, score-sorted list of chunk dicts (same shape as the
        existing ``retrieve`` plus an ``rrf_score`` field). Empty list if both
        backends fail/return nothing.
    """
    if top_k is None:
        top_k = HYBRID_TOP_K

    if db_path is None:
        db_path = str(Path(__file__).parent.parent / FTS_DB_PATH)

    # Step 1: expand the query (always retains original first)
    if expand_fn is None:
        try:
            from rag.expand import expand_query as _expand_query
            expand_fn = _expand_query
        except Exception:
            expand_fn = lambda q: [q]
    try:
        expanded = expand_fn(query)
    except Exception:
        expanded = [query]
    if not expanded:
        expanded = [query]

    # FTS benefits most from keyword variety; FAISS semantic search is robust
    # on the original query.
    faiss_query = expanded[0]
    fts_queries = expanded[1:] if len(expanded) > 1 else expanded

    # Step 2: run both backends concurrently
    faiss_results: List[Dict[str, Any]] = []
    fts_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_faiss = executor.submit(_run_faiss, faiss_query, faiss_fn)
        future_fts = executor.submit(_run_fts, fts_queries, fts_fn, top_k, db_path)
        # Each future already swallows its own exceptions -> returns []
        faiss_results = future_faiss.result()
        fts_results = future_fts.result()

    # Step 3: RRF fusion + dedup + top-k
    fused = rrf_fuse([faiss_results, fts_results], k=rrf_k)
    return fused[:top_k]


if __name__ == "__main__":
    # Manual smoke test with stub backends
    fake_faiss = lambda q: [
        {"uid": "a::0", "source": "a", "chunk_id": 0, "text": "faiss top1"},
        {"uid": "b::0", "source": "b", "chunk_id": 0, "text": "faiss top2"},
    ]
    fake_fts = lambda q, top_k=5, db_path=None: [
        {"uid": "b::0", "source": "b", "chunk_id": 0, "text": "fts top1"},
        {"uid": "c::0", "source": "c", "chunk_id": 0, "text": "fts top2"},
    ]
    fake_expand = lambda q: [q, "alt1", "alt2"]
    out = retrieve_hybrid("test", expand_fn=fake_expand, faiss_fn=fake_faiss, fts_fn=fake_fts)
    for o in out:
        print(o["uid"], round(o["rrf_score"], 4))
