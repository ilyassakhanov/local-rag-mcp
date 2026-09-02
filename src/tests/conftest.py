"""Pytest configuration: stub heavy/external modules so tests run without
Ollama, model downloads, faiss, or sentence_transformers.

Tests that need the *real* sqlite3 / rag.fts / rag.expand / rag.hybrid logic
import those modules directly (they only use stdlib + config). The stubs here
only short-circuit the module-level side effects of `rag.query` (which normally
loads a SentenceTransformer + FAISS index at import time) and `assistant`
(which constructs an `ollama.Client`).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make `src/` importable as the project does (flat imports like
# `from rag.query import ...` and `from config import ...`).
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# --- Stub modules that have heavy/external side effects ---

def _install_stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    """Install (or augment) a stub module in sys.modules and return it."""
    if name in sys.modules and not attrs:
        return sys.modules[name]
    mod = sys.modules.get(name) or types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _make_faiss_stub() -> types.ModuleType:
    """A stub `faiss` module exposing the functions used by rag.query/build_index."""
    import numpy as np

    class _Index:
        def __init__(self, dim: int):
            self.dim = dim
            self._vectors: list[np.ndarray] = []

        def add(self, x):
            self._vectors.extend(np.asarray(v) for v in np.asarray(x))

        def search(self, q, k: int):
            # Return identity-ish ranking so tests are deterministic.
            import numpy as np
            n = len(self._vectors)
            if n == 0:
                return np.zeros((1, 0), dtype="float32"), np.zeros((1, 0), dtype="int64")
            k = min(k, n)
            scores = np.array([[1.0 - i * 0.1 for i in range(k)]], dtype="float32")
            ids = np.array([[i for i in range(k)]], dtype="int64")
            return scores, ids

    return _install_stub("faiss", {
        "IndexFlatIP": _Index,
        "normalize_L2": lambda x: None,
        "read_index": lambda p: _Index(384),
        "write_index": lambda idx, p: None,
        "__version__": "stub",
    })


def _make_sentence_transformers_stub() -> types.ModuleType:
    """Stub SentenceTransformer that returns fixed-size zero embeddings."""
    import numpy as np

    class _Encoder:
        def __init__(self, *args, **kwargs):
            self.dim = 384

        def encode(self, texts, show_progress_bar=False, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return np.zeros((len(texts), self.dim), dtype="float32")

    mod = _install_stub("sentence_transformers")
    mod.SentenceTransformer = _Encoder
    return mod


def _make_ollama_stub() -> types.ModuleType:
    """Stub the `ollama` package so `assistant.py` can import without a server."""
    class _Client:
        def __init__(self, *args, **kwargs):
            self.calls: list[dict] = []

        def chat(self, model, messages):
            self.calls.append({"model": model, "messages": messages})
            return {"message": {"content": '{"use_mcp": false, "tool": null, "args": {}}'}}

    mod = _install_stub("ollama")
    mod.Client = _Client
    return mod


# Install stubs *before* any test module imports rag.query / assistant.
_make_faiss_stub()
_make_sentence_transformers_stub()
_make_ollama_stub()


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    """A fresh sqlite path unique to each test."""
    return tmp_path / "chunks_test.sqlite"


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "text": "Company vacation policy: employees receive 20 paid days off per year.",
            "source": "docs/leave.md",
            "chunk_id": 0,
            "uid": "docs/leave.md::0",
        },
        {
            "text": "Sick leave requires a doctor's note submitted within three days.",
            "source": "docs/leave.md",
            "chunk_id": 1,
            "uid": "docs/leave.md::1",
        },
        {
            "text": "Remote work is allowed up to three days per week for all staff.",
            "source": "docs/remote.md",
            "chunk_id": 0,
            "uid": "docs/remote.md::0",
        },
    ]
