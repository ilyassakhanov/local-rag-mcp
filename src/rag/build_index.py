import faiss
import pickle
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.ingest import ingest_documents
from rag.chunk import chunk_documents
from rag.embed import embed_chunks
from rag.fts import build_fts
from config import FAISS_INDEX_PATH, CHUNKS_PATH, FTS_DB_PATH


def build_index():
    """Build FAISS index from documents."""
    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH
    fts_path = src_dir / FTS_DB_PATH

    print("📥 Loading documents...")
    documents = ingest_documents()

    if not documents:
        print("❌ No documents found. Please add documents to the docs directory.")
        return

    print("✂️ Chunking...")
    chunks = chunk_documents(documents)

    print("🧠 Generating embeddings...")
    embeddings = embed_chunks(chunks)

    print("📦 Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print("💾 Saving...")
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    # Build/update the FTS5 full-text index alongside FAISS. Reuses the same
    # `chunks` list so there is no duplicated ingestion/embedding work.
    print("🔍 Building FTS5 index...")
    try:
        build_fts(chunks, fts_path)
        print(f"   FTS index saved to: {fts_path}")
    except Exception as e:
        print(f"⚠️  Warning: FTS index build failed: {e}")

    print(f"✅ Indexing complete: {len(chunks)} chunks indexed")
    print(f"   Index saved to: {index_path}")
    print(f"   Chunks saved to: {chunks_path}")


if __name__ == "__main__":
    build_index()
