import tiktoken
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP

encoder = tiktoken.get_encoding("cl100k_base")


def make_uid(source: str, chunk_id: int) -> str:
    """Build a stable, globally-unique chunk identifier.

    `chunk_id` in `chunk_documents` is per-document, so it is combined with the
    document `source` path to remain unique across the whole corpus. This uid is
    the key used for RRF deduplication and FTS rows.
    """
    return f"{source}::{chunk_id}"


def chunk_text(text: str):
    """Split text into chunks with overlap."""
    tokens = encoder.encode(text)
    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]
        chunks.append(encoder.decode(chunk_tokens))

    return chunks


def chunk_documents(documents):
    """Chunk all documents into smaller pieces."""
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["path"],
                "chunk_id": idx,
                "uid": make_uid(doc["path"], idx),
            })

    return all_chunks


if __name__ == "__main__":
    # Test chunking
    test_doc = {"text": "This is a test document. " * 100, "path": "test.txt"}
    chunks = chunk_documents([test_doc])
    print(f"Created {len(chunks)} chunks")
