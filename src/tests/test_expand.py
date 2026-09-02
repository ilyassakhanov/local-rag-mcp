"""Tests for rag.expand: query expansion parsing and fallback behavior."""

from rag import expand as expand_mod
from rag.expand import expand_query, _parse_alternatives


# ---------- _parse_alternatives ----------

class TestParseAlternatives:
    def test_plain_lines(self):
        raw = "vacation days\npaid time off\nleave policy"
        assert _parse_alternatives(raw, 5) == [
            "vacation days", "paid time off", "leave policy"
        ]

    def test_bullets_stripped(self):
        raw = "- alpha\n* beta\n+ gamma\n• delta"
        assert _parse_alternatives(raw, 5) == ["alpha", "beta", "gamma", "delta"]

    def test_numbering_stripped(self):
        raw = "1. alpha\n2) beta\n3. gamma"
        assert _parse_alternatives(raw, 5) == ["alpha", "beta", "gamma"]

    def test_quotes_stripped(self):
        raw = '"alpha"\n\'beta\'\n"gamma"'
        assert _parse_alternatives(raw, 5) == ["alpha", "beta", "gamma"]

    def test_markdown_fence_stripped(self):
        raw = "```\nalpha\nbeta\n```"
        assert _parse_alternatives(raw, 5) == ["alpha", "beta"]

    def test_markdown_fence_json(self):
        raw = "```json\nalpha\nbeta\n```"
        assert _parse_alternatives(raw, 5) == ["alpha", "beta"]

    def test_dedup_case_insensitive(self):
        raw = "alpha\nAlpha\nALPHA\nbeta"
        assert _parse_alternatives(raw, 5) == ["alpha", "beta"]

    def test_limit_applied(self):
        raw = "a\nb\nc\nd\ne\nf"
        assert _parse_alternatives(raw, 3) == ["a", "b", "c"]

    def test_empty_raw(self):
        assert _parse_alternatives("", 5) == []

    def test_blank_lines_skipped(self):
        raw = "\n\nalpha\n\n  \nbeta\n"
        assert _parse_alternatives(raw, 5) == ["alpha", "beta"]


# ---------- expand_query ----------

class TestExpandQuery:
    def test_original_always_first(self):
        gen = lambda p: "alt1\nalt2"
        out = expand_query("what is PTO?", generate_fn=gen)
        assert out[0] == "what is PTO?"
        assert "alt1" in out and "alt2" in out

    def test_alternatives_after_original(self):
        gen = lambda p: "alpha\nbeta"
        out = expand_query("query", generate_fn=gen)
        assert out == ["query", "alpha", "beta"]

    def test_caps_at_num_queries(self):
        gen = lambda p: "a\nb\nc\nd\ne"
        out = expand_query("q", generate_fn=gen, num_queries=2)
        assert out == ["q", "a", "b"]

    def test_drops_alternative_identical_to_original(self):
        gen = lambda p: "query\nalt1"
        out = expand_query("query", generate_fn=gen)
        assert out == ["query", "alt1"]

    def test_empty_output_falls_back(self):
        gen = lambda p: ""
        out = expand_query("q", generate_fn=gen)
        assert out == ["q"]

    def test_whitespace_only_output_falls_back(self):
        gen = lambda p: "   \n\n  \n"
        out = expand_query("q", generate_fn=gen)
        assert out == ["q"]

    def test_exception_falls_back_silently(self):
        def boom(p):
            raise RuntimeError("ollama down")
        out = expand_query("q", generate_fn=boom)
        assert out == ["q"]

    def test_timeout_falls_back_silently(self):
        class TimeoutErr(Exception):
            pass
        def slow(p):
            raise TimeoutErr("timed out")
        out = expand_query("q", generate_fn=slow)
        assert out == ["q"]

    def test_json_error_falls_back(self):
        def bad(p):
            raise ValueError("not json")
        out = expand_query("q", generate_fn=bad)
        assert out == ["q"]

    def test_empty_query(self):
        assert expand_query("", generate_fn=lambda p: "x") == [""]

    def test_none_query(self):
        assert expand_query(None, generate_fn=lambda p: "x") == []

    def test_default_uses_requests_when_no_fn(self, monkeypatch):
        # Stub requests.post to return a canned response.
        import sys
        import types
        fake_resp = types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": "alt one\nalt two"},
        )
        fake_requests = types.ModuleType("requests")
        called = {"count": 0}

        def fake_post(url, json=None, timeout=None):
            called["count"] += 1
            assert timeout == expand_mod.EXPANSION_TIMEOUT
            assert json["stream"] is False
            assert json["model"] == expand_mod.OLLAMA_MODEL
            assert json["options"]["temperature"] == expand_mod.EXPANSION_TEMPERATURE
            return fake_resp

        fake_requests.post = fake_post
        monkeypatch.setitem(sys.modules, "requests", fake_requests)

        out = expand_query("q")
        assert called["count"] == 1
        assert out == ["q", "alt one", "alt two"]

    def test_prompt_contains_question_and_limit(self):
        captured = {}

        def gen(prompt):
            captured["prompt"] = prompt
            return "alt1"

        expand_query("my question", generate_fn=gen, num_queries=4)
        assert "my question" in captured["prompt"]
        assert "4" in captured["prompt"]
