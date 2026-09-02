"""Tests for the assistant.query pipeline wiring: hybrid retrieval -> MCP
decision -> prompt -> answer. Verifies existing MCP behavior is preserved and
the hybrid retrieve output flows into the prompt/answer.

Uses the stubbed `ollama` from conftest and mocks retrieve/ask_llm.
"""

from unittest.mock import patch

import assistant
from assistant import CompanyKBAssistant


def _make_assistant():
    """Build an assistant without starting the real MCP subprocess."""
    with patch.object(CompanyKBAssistant, "_init_mcp", lambda self: setattr(self, "mcp", None)):
        return CompanyKBAssistant()


class TestPipelineWiring:
    def test_retrieve_called_with_user_query(self):
        a = _make_assistant()
        captured = {}

        def fake_retrieve(q):
            captured["query"] = q
            return [{"uid": "a::0", "source": "a.md", "chunk_id": 0, "text": "ctx"}]

        with patch.object(assistant, "retrieve", fake_retrieve):
            with patch.object(assistant, "ask_llm", lambda p: "ANSWER"):
                result = a.query("my question")

        assert captured["query"] == "my question"
        assert result["answer"] == "ANSWER"
        assert result["sources"] == ["a.md"]
        assert result["mcp_used"] is False

    def test_prompt_includes_fused_context(self):
        a = _make_assistant()
        captured = {}

        def fake_retrieve(q):
            return [
                {"uid": "a::0", "source": "a.md", "chunk_id": 0, "text": "ALPHA CONTEXT"},
                {"uid": "b::0", "source": "b.md", "chunk_id": 0, "text": "BETA CONTEXT"},
            ]

        def fake_ask_llm(prompt):
            captured["prompt"] = prompt
            return "FINAL ANSWER"

        with patch.object(assistant, "retrieve", fake_retrieve):
            with patch.object(assistant, "ask_llm", fake_ask_llm):
                result = a.query("question?")

        assert "ALPHA CONTEXT" in captured["prompt"]
        assert "BETA CONTEXT" in captured["prompt"]
        assert "a.md" in captured["prompt"]
        assert result["answer"] == "FINAL ANSWER"

    def test_mcp_decision_receives_contexts(self):
        """_llm_decide_mcp_usage must receive the hybrid retrieved contexts."""
        a = _make_assistant()
        a.mcp = object()  # pretend MCP exists
        captured = {}

        def fake_decide(query, contexts):
            captured["query"] = query
            captured["contexts"] = contexts
            return None, None  # decide not to use MCP

        with patch.object(a, "_llm_decide_mcp_usage", fake_decide):
            with patch.object(assistant, "retrieve", lambda q: [
                {"uid": "a::0", "source": "a.md", "chunk_id": 0, "text": "ctx"}
            ]):
                with patch.object(assistant, "ask_llm", lambda p: "ans"):
                    a.query("Q")

        assert captured["query"] == "Q"
        assert len(captured["contexts"]) == 1
        assert captured["contexts"][0]["text"] == "ctx"

    def test_mcp_used_flag_set_when_tool_called(self):
        a = _make_assistant()
        a.mcp = object()

        with patch.object(a, "_llm_decide_mcp_usage", lambda q, c: ("read_document", {"file_path": "docs/x.md"})):
            with patch.object(a, "_call_mcp_tool", lambda name, args: "MCP RESULT"):
                with patch.object(assistant, "retrieve", lambda q: [
                    {"uid": "a::0", "source": "a.md", "chunk_id": 0, "text": "ctx"}
                ]):
                    with patch.object(assistant, "ask_llm", lambda p: "ans"):
                        result = a.query("Q")

        assert result["mcp_used"] is True
        assert result["mcp_tool"] == "read_document"

    def test_mcp_result_appended_to_prompt(self):
        a = _make_assistant()
        a.mcp = object()
        captured = {}

        with patch.object(a, "_llm_decide_mcp_usage", lambda q, c: ("read_document", {"file_path": "docs/x.md"})):
            with patch.object(a, "_call_mcp_tool", lambda name, args: "EXTRA MCP INFO"):
                with patch.object(assistant, "retrieve", lambda q: [
                    {"uid": "a::0", "source": "a.md", "chunk_id": 0, "text": "ctx"}
                ]):
                    def fake_ask_llm(p):
                        captured["prompt"] = p
                        return "ans"
                    with patch.object(assistant, "ask_llm", fake_ask_llm):
                        a.query("Q")

        assert "EXTRA MCP INFO" in captured["prompt"]

    def test_no_contexts_still_works(self):
        a = _make_assistant()
        with patch.object(assistant, "retrieve", lambda q: []):
            with patch.object(assistant, "ask_llm", lambda p: "generic"):
                result = a.query("Q")
        assert result["answer"] == "generic"
        assert result["sources"] == []
        assert result["mcp_used"] is False

    def test_close_without_mcp(self):
        a = _make_assistant()
        a.close()  # must not raise


class TestMcpDecisionFallback:
    def test_no_mcp_returns_none(self):
        a = _make_assistant()
        assert a.mcp is None
        tool, args = a._llm_decide_mcp_usage("q", [])
        assert tool is None and args is None

    def test_llm_decision_exception_returns_none(self):
        a = _make_assistant()
        a.mcp = object()

        # The stubbed ollama.Client returns {"use_mcp": false}, so decision is None.
        # Force an exception path by patching the chat to raise.
        class BoomClient:
            def chat(self, *a, **kw):
                raise RuntimeError("ollama down")

        a.llm_client = BoomClient()
        tool, args = a._llm_decide_mcp_usage("q", [])
        assert tool is None and args is None


class TestHybridRetrieveWired:
    """Verify rag.query.retrieve now routes through hybrid, with FAISS fallback."""

    def test_retrieve_imports_retrieve_hybrid(self):
        # Importing rag.query triggers _ensure_index_exists via stubs (no real
        # faiss/model). The module must expose both retrieve and retrieve_faiss.
        from rag import query as query_mod
        assert hasattr(query_mod, "retrieve")
        assert hasattr(query_mod, "retrieve_faiss")

    def test_retrieve_falls_back_to_faiss_on_hybrid_failure(self, monkeypatch):
        from rag import query as query_mod

        # Populate module globals so the index/chunks guard passes.
        monkeypatch.setattr(query_mod, "index", object())
        monkeypatch.setattr(query_mod, "chunks", [{"uid": "a::0"}])
        monkeypatch.setattr(query_mod, "_fts_ready", True)

        # Force hybrid to raise -> fallback to retrieve_faiss
        def boom_retrieve_hybrid(q):
            raise RuntimeError("hybrid failed")

        monkeypatch.setattr("rag.hybrid.retrieve_hybrid", boom_retrieve_hybrid)

        faiss_calls = {"n": 0}

        def fake_faiss(q):
            faiss_calls["n"] += 1
            return [{"uid": "a::0", "source": "a.md", "chunk_id": 0, "text": "x"}]

        monkeypatch.setattr(query_mod, "retrieve_faiss", fake_faiss)

        out = query_mod.retrieve("q")
        assert faiss_calls["n"] == 1
        assert len(out) == 1
        assert out[0]["uid"] == "a::0"

    def test_retrieve_returns_hybrid_results_when_available(self, monkeypatch):
        from rag import query as query_mod

        monkeypatch.setattr(query_mod, "index", object())
        monkeypatch.setattr(query_mod, "chunks", [{"uid": "a::0"}])
        monkeypatch.setattr(query_mod, "_fts_ready", True)

        def fake_hybrid(q):
            return [{"uid": "h::0", "source": "h.md", "chunk_id": 0, "text": "hybrid"}]

        monkeypatch.setattr("rag.hybrid.retrieve_hybrid", fake_hybrid)

        out = query_mod.retrieve("q")
        assert len(out) == 1
        assert out[0]["uid"] == "h::0"

    def test_retrieve_falls_back_when_hybrid_empty(self, monkeypatch):
        from rag import query as query_mod

        monkeypatch.setattr(query_mod, "index", object())
        monkeypatch.setattr(query_mod, "chunks", [{"uid": "a::0"}])
        monkeypatch.setattr(query_mod, "_fts_ready", True)

        def fake_hybrid(q):
            return []

        def fake_faiss(q):
            return [{"uid": "f::0", "source": "f.md", "chunk_id": 0, "text": "faiss"}]

        monkeypatch.setattr("rag.hybrid.retrieve_hybrid", fake_hybrid)
        monkeypatch.setattr(query_mod, "retrieve_faiss", fake_faiss)

        out = query_mod.retrieve("q")
        assert len(out) == 1
        assert out[0]["uid"] == "f::0"
