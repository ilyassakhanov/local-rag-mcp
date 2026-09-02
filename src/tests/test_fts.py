"""Tests for rag.fts: persistent SQLite FTS5 build + search."""

import sqlite3
from pathlib import Path

from rag.fts import build_fts, search_fts, fts_exists, load_fts


class TestBuildFts:
    def test_creates_fts5_table(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
            )
            assert cur.fetchone() is not None
            # verify FTS5 specifically
            cur = conn.execute("SELECT name FROM pragma_module_list() WHERE name='fts5'")
            assert cur.fetchone() is not None
        finally:
            conn.close()

    def test_inserts_all_chunks(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        try:
            cur = conn.execute("SELECT COUNT(*) FROM chunks_fts")
            assert cur.fetchone()[0] == len(sample_chunks)
        finally:
            conn.close()

    def test_rebuild_replaces_existing(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        # rebuild with fewer chunks
        smaller = sample_chunks[:1]
        build_fts(smaller, tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        try:
            cur = conn.execute("SELECT COUNT(*) FROM chunks_fts")
            assert cur.fetchone()[0] == 1
        finally:
            conn.close()

    def test_empty_chunks_noop(self, tmp_db):
        build_fts([], tmp_db)
        # file may or may not be created, but table must not exist
        assert not fts_exists(tmp_db)

    def test_synthesises_uid_if_missing(self, tmp_db):
        chunks = [{"text": "hello", "source": "a.md", "chunk_id": 2}]
        build_fts(chunks, tmp_db)
        hits = search_fts("hello", top_k=5, db_path=tmp_db)
        assert len(hits) == 1
        assert hits[0]["uid"] == "a.md::2"


class TestFtsExists:
    def test_missing_db_returns_false(self, tmp_db):
        assert fts_exists(tmp_db) is False

    def test_existing_db_returns_true(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        assert fts_exists(tmp_db) is True

    def test_load_fts_false_when_missing(self, tmp_db):
        assert load_fts(tmp_db) is False

    def test_load_fts_true_when_built(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        assert load_fts(tmp_db) is True


class TestSearchFts:
    def test_returns_stable_uid(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        hits = search_fts("vacation", top_k=5, db_path=tmp_db)
        assert len(hits) >= 1
        assert all("uid" in h for h in hits)
        assert all("::" in h["uid"] for h in hits)

    def test_returns_metadata(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        hits = search_fts("sick", top_k=5, db_path=tmp_db)
        assert len(hits) >= 1
        h = hits[0]
        assert "source" in h and "chunk_id" in h and "text" in h

    def test_returns_rank_field(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        hits = search_fts("leave", top_k=5, db_path=tmp_db)
        assert len(hits) >= 1
        for i, h in enumerate(hits):
            assert h["rank"] == i

    def test_bm25_ordering(self, tmp_db):
        chunks = [
            {"text": "vacation vacation vacation policy", "source": "a", "chunk_id": 0, "uid": "a::0"},
            {"text": "sick leave policy", "source": "b", "chunk_id": 0, "uid": "b::0"},
            {"text": "remote work guideline", "source": "c", "chunk_id": 0, "uid": "c::0"},
        ]
        build_fts(chunks, tmp_db)
        hits = search_fts("vacation", top_k=3, db_path=tmp_db)
        # The chunk with the most "vacation" occurrences must rank first.
        assert hits[0]["uid"] == "a::0"

    def test_top_k_limit(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        hits = search_fts("policy", top_k=1, db_path=tmp_db)
        assert len(hits) == 1

    def test_empty_query_returns_empty(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        assert search_fts("", top_k=5, db_path=tmp_db) == []
        assert search_fts("   ", top_k=5, db_path=tmp_db) == []

    def test_missing_db_returns_empty(self, tmp_db):
        assert search_fts("anything", top_k=5, db_path=tmp_db) == []

    def test_special_chars_sanitized(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        # Query with FTS5 control characters; must not raise.
        hits = search_fts("vacation*) OR \"", top_k=5, db_path=tmp_db)
        # Should not raise; may or may not return results
        assert isinstance(hits, list)

    def test_multiple_terms(self, tmp_db, sample_chunks):
        build_fts(sample_chunks, tmp_db)
        hits = search_fts("vacation policy", top_k=5, db_path=tmp_db)
        assert isinstance(hits, list)
        if hits:
            assert "vacation" in hits[0]["text"].lower() or "policy" in hits[0]["text"].lower()
