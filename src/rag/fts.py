"""Persistent SQLite FTS5 full-text index over existing chunks.

This is a *secondary* index that complements (does not replace) FAISS. It is
built/updated alongside the FAISS index during ``build-index`` and reuses the
same ``chunks`` list produced by ``rag.chunk.chunk_documents``. The on-disk
artifact is ``chunks.sqlite`` (path from ``config.FTS_DB_PATH``) located next to
``index.faiss`` / ``chunks.pkl`` in the ``src/`` directory.

Each row stores the chunk text plus its stable ``uid`` (and source/chunk_id
metadata) so fused results can be deduplicated by uid and correlated back to
the in-memory chunk dicts.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FTS_DB_PATH, TOP_K


def _resolve_db_path(db_path: Optional[str | Path] = None) -> Path:
    """Resolve the FTS database path relative to the src/ directory."""
    if db_path is not None:
        return Path(db_path)
    src_dir = Path(__file__).parent.parent
    return src_dir / FTS_DB_PATH


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def build_fts(chunks: Sequence[Dict[str, Any]], db_path: Optional[str | Path] = None) -> None:
    """(Re)build the FTS5 table from a list of chunk dicts.

    Drops any existing ``chunks_fts`` table first, so this is the canonical
    rebuild path used by ``rag.build_index.build_index``.
    """
    if not chunks:
        return

    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = _connect(path)
    try:
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text,
                uid UNINDEXED,
                source UNINDEXED,
                chunk_id UNINDEXED
            )
            """
        )
        rows = []
        for c in chunks:
            uid = c.get("uid")
            if uid is None:
                uid = f"{c.get('source', '')}::{c.get('chunk_id', '')}"
            rows.append((
                c.get("text", ""),
                uid,
                c.get("source", ""),
                c.get("chunk_id", 0),
            ))
        conn.executemany(
            "INSERT INTO chunks_fts (text, uid, source, chunk_id) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def fts_exists(db_path: Optional[str | Path] = None) -> bool:
    """Return True if the FTS table exists in the database."""
    path = _resolve_db_path(db_path)
    if not path.exists():
        return False
    conn = _connect(path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        return cur.fetchone() is not None
    except Exception:
        return False
    finally:
        conn.close()


def load_fts(db_path: Optional[str | Path] = None) -> bool:
    """Verify the FTS index is usable. Returns True if available, else False."""
    try:
        return fts_exists(db_path)
    except Exception:
        return False


def _sanitize_fts_query(query: str) -> str:
    """Escape an FTS5 MATCH query so control characters are treated as terms.

    FTS5 treats characters like ``*`` ``"`` ``(`` ``)`` ``OR`` ``AND`` ``NOT``
    specially. To do a robust substring/keyword search we quote tokens and join
    them with OR, which matches any of the terms.
    """
    if not query:
        return ""
    # Replace control characters with spaces, then split into tokens
    cleaned = []
    for ch in query:
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    tokens = [t for t in "".join(cleaned).split() if t]
    if not tokens:
        return ""
    # double-quote each token to escape FTS5 syntax
    return " OR ".join(f'"{t}"' for t in tokens)


def search_fts(
    query: str,
    top_k: int = TOP_K,
    db_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Search the FTS5 index. Returns BM25-ranked chunks with 0-based ``rank``.

    Each returned dict has: ``uid``, ``source``, ``chunk_id``, ``text``, and
    ``rank`` (0 = best). On any error returns an empty list (callers must treat
    a failed backend as empty results, not raise).
    """
    if not query or not query.strip():
        return []

    path = _resolve_db_path(db_path)
    if not path.exists():
        return []

    match_query = _sanitize_fts_query(query)
    if not match_query:
        return []

    conn = _connect(path)
    try:
        cur = conn.execute(
            """
            SELECT uid, source, chunk_id, text, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match_query, top_k),
        )
        rows = cur.fetchall()
    except Exception:
        # FTS MATCH can error on edge-case input; fall back to LIKE search.
        try:
            like_pattern = f"%{query}%"
            cur = conn.execute(
                """
                SELECT uid, source, chunk_id, text
                FROM chunks_fts
                WHERE text LIKE ?
                LIMIT ?
                """,
                (like_pattern, top_k),
            )
            rows = cur.fetchall()
            # LIKE rows lack a score; synthesize rank by row order
            return [
                {
                    "uid": r[0],
                    "source": r[1],
                    "chunk_id": r[2],
                    "text": r[3],
                    "rank": i,
                }
                for i, r in enumerate(rows)
            ]
        except Exception:
            return []
    finally:
        conn.close()

    return [
        {
            "uid": r[0],
            "source": r[1],
            "chunk_id": r[2],
            "text": r[3],
            "rank": i,  # 0-based rank for RRF
        }
        for i, r in enumerate(rows)
    ]


if __name__ == "__main__":
    # Quick manual sanity check
    sample = [
        {"text": "Company vacation policy allows 20 days.", "source": "docs/leave.md", "chunk_id": 0, "uid": "docs/leave.md::0"},
        {"text": "Sick leave requires a doctor's note.", "source": "docs/leave.md", "chunk_id": 1, "uid": "docs/leave.md::1"},
    ]
    db = Path(__file__).parent.parent / "chunks_test.sqlite"
    build_fts(sample, db)
    print("exists:", fts_exists(db))
    for hit in search_fts("vacation", db_path=db):
        print(hit)
    db.unlink(missing_ok=True)
