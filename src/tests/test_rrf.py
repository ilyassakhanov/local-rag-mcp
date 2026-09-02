"""Tests for rag.hybrid.rrf_fuse: Reciprocal Rank Fusion correctness."""

from rag.hybrid import rrf_fuse, RRF_K


class TestRrfFuse:
    def test_single_ranking_preserves_order(self):
        ranking = [
            {"uid": "a::0", "text": "x"},
            {"uid": "b::0", "text": "y"},
            {"uid": "c::0", "text": "z"},
        ]
        fused = rrf_fuse([ranking])
        assert [d["uid"] for d in fused] == ["a::0", "b::0", "c::0"]

    def test_score_formula_single_ranking(self):
        ranking = [{"uid": "a::0"}, {"uid": "a::1"}, {"uid": "a::2"}]
        fused = rrf_fuse([ranking])
        assert round(fused[0]["rrf_score"], 6) == round(1.0 / (RRF_K + 0), 6)
        assert round(fused[1]["rrf_score"], 6) == round(1.0 / (RRF_K + 1), 6)
        assert round(fused[2]["rrf_score"], 6) == round(1.0 / (RRF_K + 2), 6)

    def test_two_rankings_sum_scores(self):
        r1 = [{"uid": "a::0"}, {"uid": "b::0"}]  # ranks 0, 1
        r2 = [{"uid": "b::0"}, {"uid": "c::0"}]  # ranks 0, 1
        fused = rrf_fuse([r1, r2])
        scores = {d["uid"]: d["rrf_score"] for d in fused}
        # a::0 only in r1 at rank 0
        assert round(scores["a::0"], 6) == round(1.0 / (RRF_K + 0), 6)
        # b::0 in r1 rank 1 + r2 rank 0
        expected_b = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 0)
        assert round(scores["b::0"], 6) == round(expected_b, 6)
        # c::0 only in r2 rank 1
        assert round(scores["c::0"], 6) == round(1.0 / (RRF_K + 1), 6)

    def test_dedup_by_uid(self):
        r1 = [{"uid": "a::0", "text": "from faiss"}]
        r2 = [{"uid": "a::0", "text": "from fts"}]
        fused = rrf_fuse([r1, r2])
        assert len(fused) == 1
        assert fused[0]["uid"] == "a::0"
        # score = 1/(60+0) + 1/(60+0)
        expected = 1.0 / (RRF_K + 0) + 1.0 / (RRF_K + 0)
        assert round(fused[0]["rrf_score"], 6) == round(expected, 6)

    def test_sort_by_score_descending(self):
        r1 = [{"uid": "a::0"}, {"uid": "b::0"}, {"uid": "c::0"}]
        r2 = [{"uid": "b::0"}, {"uid": "a::0"}, {"uid": "d::0"}]
        fused = rrf_fuse([r1, r2])
        scores = [d["rrf_score"] for d in fused]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncation_in_retrieve_hybrid(self):
        from rag.hybrid import retrieve_hybrid
        faiss_fn = lambda q: [{"uid": f"a::{i}", "text": str(i)} for i in range(10)]
        fts_fn = lambda q, top_k=5, db_path=None: [{"uid": f"b::{i}", "text": str(i)} for i in range(10)]
        expand_fn = lambda q: [q]
        out = retrieve_hybrid(
            "q", expand_fn=expand_fn, faiss_fn=faiss_fn, fts_fn=fts_fn, top_k=3
        )
        assert len(out) == 3

    def test_custom_rrf_k(self):
        r1 = [{"uid": "a::0"}]
        fused = rrf_fuse([r1], k=100)
        assert round(fused[0]["rrf_score"], 6) == round(1.0 / (100 + 0), 6)

    def test_empty_rankings(self):
        fused = rrf_fuse([[], []])
        assert fused == []

    def test_no_rankings(self):
        fused = rrf_fuse([])
        assert fused == []

    def test_one_backend_empty_other_has_results(self):
        r1 = [{"uid": "a::0", "text": "x"}, {"uid": "b::0", "text": "y"}]
        r2 = []
        fused = rrf_fuse([r1, r2])
        assert len(fused) == 2
        assert {d["uid"] for d in fused} == {"a::0", "b::0"}

    def test_preserves_chunk_metadata(self):
        r1 = [{"uid": "a::0", "source": "a.md", "chunk_id": 0, "text": "hello"}]
        fused = rrf_fuse([r1])
        assert fused[0]["source"] == "a.md"
        assert fused[0]["chunk_id"] == 0
        assert fused[0]["text"] == "hello"

    def test_synthesises_uid_when_missing(self):
        r1 = [{"source": "a.md", "chunk_id": 5, "text": "x"}]
        fused = rrf_fuse([r1])
        assert fused[0]["uid"] == "a.md::5"

    def test_synthesised_uid_dedup_consistent(self):
        # Same chunk without uid in both rankings should dedup to one entry.
        r1 = [{"source": "a.md", "chunk_id": 0, "text": "x"}]
        r2 = [{"source": "a.md", "chunk_id": 0, "text": "x"}]
        fused = rrf_fuse([r1, r2])
        assert len(fused) == 1
