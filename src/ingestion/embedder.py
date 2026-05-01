from langchain.schema import Document
from sentence_transformers import SentenceTransformer

from src.utils import load_config


def embed_chunks(chunks: list[Document]) -> list[tuple[Document, list[float]]]:
    cfg = load_config()
    model_name = cfg["embedding"]["model"]
    batch_size = cfg["embedding"]["batch_size"]

    model = SentenceTransformer(model_name)
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False)
        for chunk, embedding in zip(batch, embeddings):
            results.append((chunk, embedding.tolist()))
    return results
