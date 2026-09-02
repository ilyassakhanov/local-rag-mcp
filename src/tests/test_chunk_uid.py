"""Tests for rag.chunk: stable, globally-unique chunk uid."""

from rag.chunk import chunk_documents, make_uid


class TestMakeUid:
    def test_format(self):
        assert make_uid("docs/leave.md", 3) == "docs/leave.md::3"

    def test_different_sources_same_index_unique(self):
        assert make_uid("a.md", 0) != make_uid("b.md", 0)

    def test_same_source_different_index_unique(self):
        assert make_uid("a.md", 0) != make_uid("a.md", 1)


class TestChunkDocumentsUid:
    def test_chunks_have_uid(self):
        docs = [{"path": "a.md", "text": "hello world " * 200}]
        chunks = chunk_documents(docs)
        assert len(chunks) > 0
        for c in chunks:
            assert "uid" in c
            assert c["uid"].startswith("a.md::")
            assert "::" in c["uid"]

    def test_uids_globally_unique_across_documents(self):
        docs = [
            {"path": "a.md", "text": "alpha " * 500},
            {"path": "b.md", "text": "beta " * 500},
        ]
        chunks = chunk_documents(docs)
        uids = [c["uid"] for c in chunks]
        assert len(uids) == len(set(uids)), "uids must be globally unique"

    def test_uid_matches_source_and_chunk_id(self):
        docs = [{"path": "doc.md", "text": "word " * 1000}]
        chunks = chunk_documents(docs)
        for c in chunks:
            assert c["uid"] == f"{c['source']}::{c['chunk_id']}"

    def test_preserves_existing_fields(self):
        docs = [{"path": "a.md", "text": "hello world " * 200}]
        chunks = chunk_documents(docs)
        for c in chunks:
            assert "text" in c
            assert "source" in c
            assert "chunk_id" in c
            assert c["source"] == "a.md"
