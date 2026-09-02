# Company Knowledge Base Assistant

An intelligent Q&A system that answers questions about company documentation using RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol) tools.

## Features

- **Hybrid search**: Combines FAISS semantic search with SQLite FTS5 full-text search, fused via Reciprocal Rank Fusion (RRF)
- **Query expansion**: An Ollama LLM expands the query into alternatives before retrieval (fail-safe, always retains the original)
- **MCP tools**: Dynamic document reading and management
- **Local LLM**: Privacy-preserving answers using Ollama

## Architecture

```
User query
   │
   ▼
Query expansion (Ollama, temperature 0, 2–3 alternatives, fail-safe)
   │  always retains the original query first
   ▼
Hybrid retrieval (ThreadPoolExecutor, concurrent)
   ├─ FAISS  ← original query (semantic)
   └─ FTS5   ← expanded queries (lexical, BM25)
   │
   ▼
RRF fusion: score(d) = Σ 1 / (60 + rank(d))
   │  dedup by stable chunk uid, sort, top-K
   ▼
Existing pipeline: MCP decision → build_prompt → ask_llm
```

### Index files

All index artifacts live in `src/` next to `main.py`:

| File | Purpose |
|------|---------|
| `index.faiss` | FAISS `IndexFlatIP` (L2-normalized) embeddings index |
| `chunks.pkl` | Pickled list of chunk dicts (text, source, chunk_id, uid) |
| `chunks.sqlite` | SQLite FTS5 full-text index over the same chunks |

`build-index` rebuilds **all three** from the same ingested + chunked corpus; FTS reuses the existing chunks so there is no duplicate ingestion or embedding work.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Documents

Create a `docs/` directory and add your company documentation files (`.txt`, `.md`, `.pdf`, `.docx`):

```bash
mkdir docs
# Add your company documentation files here
```

### 3. Configure

Edit `config.py` to set:
- `DOCUMENTS_DIR`: Path to your documentation directory
- `OLLAMA_MODEL`: Local LLM model to use (default: "llama3")
- Other settings as needed

### 4. Build Index (Optional)

The index will be built automatically on first use. To manually build it:

```bash
python main.py build-index
```

Or directly:

```bash
python -m rag.build_index
```

## Usage

### Interactive CLI

Run the interactive assistant:

```bash
python main.py
```

Then ask questions about your company documentation!

## Project Structure

```
src/
├── config.py              # Configuration
├── main.py                # CLI entry point
├── assistant.py           # Main assistant class
├── rag/                   # RAG components
│   ├── ingest.py         # Document ingestion
│   ├── chunk.py          # Text chunking (emits stable chunk uid)
│   ├── embed.py          # Embedding generation
│   ├── build_index.py    # FAISS + FTS index building
│   ├── fts.py            # SQLite FTS5 full-text index
│   ├── expand.py         # Query expansion (Ollama, fail-safe)
│   ├── hybrid.py         # Concurrent FAISS+FTS retrieval + RRF fusion
│   └── query.py          # Query and retrieval (hybrid by default)
├── mcp/                   # MCP components
│   ├── server.py         # MCP server with tools
│   └── client.py         # MCP client
├── tests/                # Pytest test suite
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## How It Works

1. **Document Ingestion**: Loads documents from the `docs/` directory
2. **Chunking**: Splits documents into smaller chunks with overlap; each chunk gets a stable, globally-unique `uid` (`"{source}::{chunk_id}"`)
3. **Embedding**: Generates embeddings using SentenceTransformers
4. **Indexing**: Builds a FAISS vector index AND a SQLite FTS5 full-text index over the same chunks
5. **Query**:
   - **Expansion**: The original query is expanded into 2–3 alternatives via Ollama (temperature 0). The original is always retained. On any failure (empty output, timeout, LLM error) it silently falls back to `[query]`.
   - **Hybrid retrieval**: FAISS (semantic, on the original query) and FTS5 (lexical/BM25, on the expanded queries) run **concurrently** via `ThreadPoolExecutor`. A failure in one backend does not discard results from the other.
   - **RRF fusion**: `score(d) = Σ 1 / (60 + rank(d))` over both rankings, deduplicated by `uid`, sorted by score, Top-K returned.
   - **Answer**: Optionally uses MCP tools for document access, then generates an answer using the local LLM (Ollama)

## Configuration (`config.py`)

| Key | Default | Description |
|-----|---------|-------------|
| `DOCUMENTS_DIR` | `"./docs"` | Source documents directory |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 700 / 100 | Token-based chunking |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | SentenceTransformer model |
| `FAISS_INDEX_PATH` | `"index.faiss"` | FAISS index file (in `src/`) |
| `CHUNKS_PATH` | `"chunks.pkl"` | Pickled chunks (in `src/`) |
| `FTS_DB_PATH` | `"chunks.sqlite"` | SQLite FTS5 database (in `src/`) |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `OLLAMA_MODEL` | `"qwen3:0.6b"` | LLM model (pull with `ollama pull qwen3:0.6b`) |
| `TOP_K` | 5 | FAISS-only retrieval depth (`retrieve_faiss`) |
| `HYBRID_TOP_K` | 5 | Final fused Top-K returned by hybrid retrieval |
| `RRF_K` | 60 | RRF denominator constant |
| `EXPANSION_NUM_QUERIES` | 3 | Max alternative queries requested |
| `EXPANSION_TEMPERATURE` | 0.0 | LLM sampling temperature for expansion |
| `EXPANSION_TIMEOUT` | 15 | Per-call timeout (seconds) for the expansion LLM request |

## MCP Tools

The MCP server provides:
- `read_document`: Read a specific document
- `list_documents`: List all available documents
- `search_documents`: Search documents by name

## Troubleshooting

**Index not found**: Run `python main.py build-index` first. This builds `index.faiss`, `chunks.pkl`, and `chunks.sqlite` together.

**Ollama not responding**: Make sure Ollama is running and the model is installed:
```bash
ollama pull qwen3:0.6b
```

**No documents found**: Check that `DOCUMENTS_DIR` in `config.py` points to your documents

## Tests

The test suite uses `pytest` and runs **without** Ollama, model downloads, or a live FAISS index — heavy/external modules (`faiss`, `sentence_transformers`, `ollama`) are stubbed in `tests/conftest.py`.

```bash
cd src
venv/bin/python -m pytest tests/ -v
```

## License

MIT
