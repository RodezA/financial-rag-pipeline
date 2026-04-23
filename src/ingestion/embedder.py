from langchain.schema import Document
from openai import OpenAI

from src.utils import load_config


def embed_chunks(chunks: list[Document]) -> list[tuple[Document, list[float]]]:
    cfg = load_config()
    client = OpenAI()
    model = cfg["embedding"]["model"]
    batch_size = cfg["embedding"]["batch_size"]

    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        response = client.embeddings.create(model=model, input=texts)
        for chunk, embedding_obj in zip(batch, response.data):
            results.append((chunk, embedding_obj.embedding))
    return results
