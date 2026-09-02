# Configuration for Company Knowledge Base Assistant

# Document directory - update this to point to your company documentation
DOCUMENTS_DIR = "./docs"

# Chunking configuration
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# FAISS index paths (relative to src directory)
FAISS_INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"

# SQLite FTS5 full-text index path (relative to src directory).
# Built/updated alongside the FAISS index during `build-index`.
FTS_DB_PATH = "chunks.sqlite"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:0.6b"

# RAG retrieval configuration
TOP_K = 5

# ---- Hybrid search & query expansion ----

# Final number of chunks to return from fused (FAISS + FTS) retrieval.
HYBRID_TOP_K = 5

# Reciprocal Rank Fusion constant: score(d) = sum( 1 / (RRF_K + rank(d)) )
RRF_K = 60

# Number of alternative queries / keyword phrases to request from the LLM.
EXPANSION_NUM_QUERIES = 3

# Sampling temperature for the expansion LLM call (keep ~0 for determinism).
EXPANSION_TEMPERATURE = 0.0

# Per-call timeout (seconds) for the Ollama expansion request.
EXPANSION_TIMEOUT = 15
