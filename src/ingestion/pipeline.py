from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_chunks
from src.ingestion.loader import load_documents
from src.retrieval.vector_store import VectorStore


def run_ingestion(recreate_collection: bool = False):
    print("Loading documents...")
    docs = load_documents()
    print(f"  Loaded {len(docs)} documents")

    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"  Produced {len(chunks)} chunks")

    print("Embedding...")
    embedded = embed_chunks(chunks)
    print(f"  Embedded {len(embedded)} chunks")

    print("Upserting to Qdrant...")
    store = VectorStore()
    store.create_collection(recreate=recreate_collection)
    store.upsert(embedded)
    print("  Done.")


if __name__ == "__main__":
    run_ingestion(recreate_collection=True)
