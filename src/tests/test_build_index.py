"""Tests for rag.build_index: verifies FTS is built alongside FAISS on rebuild."""

from unittest.mock import patch, MagicMock
from pathlib import Path

import rag.build_index as build_index_mod


class TestBuildIndex:
    def test_calls_build_fts_after_pickle_dump(self, tmp_path, monkeypatch):
        chunks = [
            {"text": "a", "source": "a.md", "chunk_id": 0, "uid": "a.md::0"},
            {"text": "b", "source": "b.md", "chunk_id": 0, "uid": "b.md::0"},
        ]
        embeddings = MagicMock()
        embeddings.shape = (2, 384)

        fts_calls = {"n": 0, "args": None}

        def fake_build_fts(chunks_arg, db_path=None):
            fts_calls["n"] += 1
            fts_calls["args"] = (chunks_arg, db_path)
            return None

        faiss_calls = {"write": 0}

        def fake_write_index(idx, path):
            faiss_calls["write"] += 1

        # Patch the names used inside build_index.py
        monkeypatch.setattr(build_index_mod, "ingest_documents", lambda: [{"path": "a.md", "text": "a"}])
        monkeypatch.setattr(build_index_mod, "chunk_documents", lambda docs: chunks)
        monkeypatch.setattr(build_index_mod, "embed_chunks", lambda c: embeddings)
        monkeypatch.setattr(build_index_mod, "build_fts", fake_build_fts)
        monkeypatch.setattr(build_index_mod.faiss, "IndexFlatIP", lambda dim: MagicMock())
        monkeypatch.setattr(build_index_mod.faiss, "normalize_L2", lambda x: None)
        monkeypatch.setattr(build_index_mod.faiss, "write_index", fake_write_index)
        # pickle.dump is real; just need a writable path
        monkeypatch.chdir(tmp_path)

        build_index_mod.build_index()

        assert fts_calls["n"] == 1
        assert fts_calls["args"][0] is chunks
        assert "chunks.sqlite" in str(fts_calls["args"][1])

    def test_fts_failure_does_not_break_build(self, tmp_path, monkeypatch):
        chunks = [{"text": "a", "source": "a.md", "chunk_id": 0, "uid": "a.md::0"}]
        embeddings = MagicMock()
        embeddings.shape = (1, 384)

        def boom(chunks_arg, db_path=None):
            raise RuntimeError("fts build failed")

        monkeypatch.setattr(build_index_mod, "ingest_documents", lambda: [{"path": "a.md", "text": "a"}])
        monkeypatch.setattr(build_index_mod, "chunk_documents", lambda docs: chunks)
        monkeypatch.setattr(build_index_mod, "embed_chunks", lambda c: embeddings)
        monkeypatch.setattr(build_index_mod, "build_fts", boom)
        monkeypatch.setattr(build_index_mod.faiss, "IndexFlatIP", lambda dim: MagicMock())
        monkeypatch.setattr(build_index_mod.faiss, "normalize_L2", lambda x: None)
        monkeypatch.setattr(build_index_mod.faiss, "write_index", lambda idx, path: None)
        monkeypatch.chdir(tmp_path)

        # Should not raise
        build_index_mod.build_index()

    def test_no_documents_returns_early(self, monkeypatch):
        monkeypatch.setattr(build_index_mod, "ingest_documents", lambda: [])
        # Should return without calling chunk/embed
        monkeypatch.setattr(build_index_mod, "chunk_documents", lambda d: (_ for _ in ()).throw(AssertionError("should not chunk")))
        build_index_mod.build_index()
