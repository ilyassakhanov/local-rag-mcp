"""Tests for rag.hybrid.retrieve_hybrid: concurrency, partial failure, fusion."""

import threading
import time

from rag.hybrid import retrieve_hybrid


class TestRetrieveHybrid:
    def test_both_backends_called(self):
        calls = {"faiss": 0, "fts": 0}

        def faiss_fn(q):
            calls["faiss"] += 1
            return [{"uid": "a::0", "source": "a", "chunk_id": 0, "text": "x"}]

        def fts_fn(q, top_k=5, db_path=None):
            calls["fts"] += 1
            return [{"uid": "b::0", "source": "b", "chunk_id": 0, "text": "y"}]

        expand_fn = lambda q: [q, "alt1"]
        out = retrieve_hybrid(
            "q", expand_fn=expand_fn, faiss_fn=faiss_fn, fts_fn=fts_fn
        )
        assert calls["faiss"] == 1
        assert calls["fts"] == 1  # FTS called with expanded queries
        assert len(out) >= 2

    def test_faiss_runs_on_original_query(self):
        captured = {}

        def faiss_fn(q):
            captured["faiss_query"] = q
            return []

        fts_fn = lambda q, top_k=5, db_path=None: []
        expand_fn = lambda q: [q, "alt1", "alt2"]
        retrieve_hybrid("ORIGINAL", expand_fn=expand_fn, faiss_fn=faiss_fn, fts_fn=fts_fn)
        assert captured["faiss_query"] == "ORIGINAL"

    def test_fts_runs_on_expanded_queries(self):
        captured = []

        def fts_fn(q, top_k=5, db_path=None):
            captured.append(q)
            return []

        faiss_fn = lambda q: []
        expand_fn = lambda q: [q, "alt1", "alt2"]
        retrieve_hybrid(
            "orig", expand_fn=expand_fn, faiss_fn=faiss_fn, fts_fn=fts_fn
        )
        # FTS should receive the expanded alternatives (not the original)
        assert "alt1" in captured
        assert "alt2" in captured

    def test_backends_execute_concurrently(self):
        """Verify FAISS and FTS overlap in time via a barrier."""
        barrier = threading.Barrier(2, timeout=5.0)
        order = []

        def faiss_fn(q):
            order.append("faiss_start")
            barrier.wait()  # will only release when FTS also reaches barrier
            order.append("faiss_end")
            return [{"uid": "a::0", "source": "a", "chunk_id": 0, "text": "x"}]

        def fts_fn(q, top_k=5, db_path=None):
            order.append("fts_start")
            barrier.wait()
            order.append("fts_end")
            return [{"uid": "b::0", "source": "b", "chunk_id": 0, "text": "y"}]

        expand_fn = lambda q: [q]
        out = retrieve_hybrid(
            "q", expand_fn=expand_fn, faiss_fn=faiss_fn, fts_fn=fts_fn
        )
        # If both reached the barrier, they ran concurrently.
        assert "faiss_start" in order and "fts_start" in order
        assert len(out) >= 1

    def test_partial_failure_faiss_raises_keeps_fts(self):
        def boom(q):
            raise RuntimeError("faiss exploded")

        def fts_fn(q, top_k=5, db_path=None):
            return [{"uid": "b::0", "source": "b", "chunk_id": 0, "text": "fts only"}]

        out = retrieve_hybrid(
            "q", expand_fn=lambda q: [q], faiss_fn=boom, fts_fn=fts_fn
        )
        assert len(out) == 1
        assert out[0]["uid"] == "b::0"

    def test_partial_failure_fts_raises_keeps_faiss(self):
        def faiss_fn(q):
            return [{"uid": "a::0", "source": "a", "chunk_id": 0, "text": "faiss only"}]

        def boom(q, top_k=5, db_path=None):
            raise RuntimeError("fts exploded")

        out = retrieve_hybrid(
            "q", expand_fn=lambda q: [q], faiss_fn=faiss_fn, fts_fn=boom
        )
        assert len(out) == 1
        assert out[0]["uid"] == "a::0"

    def test_partial_failure_faiss_empty_keeps_fts(self):
        def faiss_fn(q):
            return []

        def fts_fn(q, top_k=5, db_path=None):
            return [{"uid": "b::0", "source": "b", "chunk_id": 0, "text": "y"}]

        out = retrieve_hybrid(
            "q", expand_fn=lambda q: [q], faiss_fn=faiss_fn, fts_fn=fts_fn
        )
        assert len(out) == 1
        assert out[0]["uid"] == "b::0"

    def test_both_backends_empty_returns_empty(self):
        out = retrieve_hybrid(
            "q",
            expand_fn=lambda q: [q],
            faiss_fn=lambda q: [],
            fts_fn=lambda q, top_k=5, db_path=None: [],
        )
        assert out == []

    def test_both_backends_raise_returns_empty(self):
        out = retrieve_hybrid(
            "q",
            expand_fn=lambda q: [q],
            faiss_fn=lambda q: (_ for _ in ()).throw(RuntimeError("x")),
            fts_fn=lambda q, top_k=5, db_path=None: (_ for _ in ()).throw(RuntimeError("y")),
        )
        assert out == []

    def test_expansion_failure_falls_back_to_original(self):
        """If expand_fn raises, retrieve_hybrid should still use [query]."""
        captured = {"faiss": None, "fts": []}

        def boom(q):
            raise RuntimeError("expansion failed")

        def faiss_fn(q):
            captured["faiss"] = q
            return [{"uid": "a::0", "source": "a", "chunk_id": 0, "text": "x"}]

        def fts_fn(q, top_k=5, db_path=None):
            captured["fts"].append(q)
            return []

        out = retrieve_hybrid(
            "ORIG", expand_fn=boom, faiss_fn=faiss_fn, fts_fn=fts_fn
        )
        # FAISS must have received the original query
        assert captured["faiss"] == "ORIG"
        # FTS must have been called with at least the original query
        assert "ORIG" in captured["fts"]
        assert len(out) == 1

    def test_expansion_returns_single_query(self):
        """If expansion returns only [query], FTS runs on [query]."""
        def faiss_fn(q):
            return [{"uid": "a::0", "source": "a", "chunk_id": 0, "text": "x"}]

        def fts_fn(q, top_k=5, db_path=None):
            return [{"uid": "a::0", "source": "a", "chunk_id": 0, "text": "x"}]

        out = retrieve_hybrid(
            "q", expand_fn=lambda q: [q], faiss_fn=faiss_fn, fts_fn=fts_fn
        )
        # a::0 appears in both rankings -> deduped
        assert len(out) == 1
        # score = 1/(60+0) + 1/(60+0)
        expected = 1.0 / (60 + 0) + 1.0 / (60 + 0)
        assert round(out[0]["rrf_score"], 6) == round(expected, 6)

    def test_dedup_across_backends(self):
        def faiss_fn(q):
            return [
                {"uid": "shared::0", "source": "s", "chunk_id": 0, "text": "x"},
                {"uid": "faiss_only::0", "source": "f", "chunk_id": 0, "text": "y"},
            ]

        def fts_fn(q, top_k=5, db_path=None):
            return [
                {"uid": "shared::0", "source": "s", "chunk_id": 0, "text": "x"},
                {"uid": "fts_only::0", "source": "t", "chunk_id": 0, "text": "z"},
            ]

        out = retrieve_hybrid(
            "q", expand_fn=lambda q: [q], faiss_fn=faiss_fn, fts_fn=fts_fn
        )
        uids = {d["uid"] for d in out}
        assert uids == {"shared::0", "faiss_only::0", "fts_only::0"}
        # shared::0 has the highest score (appears rank 0 in both)
        assert out[0]["uid"] == "shared::0"

    def test_top_k_truncation(self):
        def faiss_fn(q):
            return [{"uid": f"a::{i}", "source": "a", "chunk_id": i, "text": str(i)} for i in range(10)]

        def fts_fn(q, top_k=5, db_path=None):
            return []

        out = retrieve_hybrid(
            "q", expand_fn=lambda q: [q], faiss_fn=faiss_fn, fts_fn=fts_fn, top_k=5
        )
        assert len(out) == 5

    def test_uses_default_top_k_when_none(self):
        from config import HYBRID_TOP_K

        def faiss_fn(q):
            return [{"uid": f"a::{i}", "source": "a", "chunk_id": i, "text": str(i)} for i in range(20)]

        def fts_fn(q, top_k=5, db_path=None):
            return []

        out = retrieve_hybrid(
            "q", expand_fn=lambda q: [q], faiss_fn=faiss_fn, fts_fn=fts_fn, top_k=None
        )
        assert len(out) == HYBRID_TOP_K
