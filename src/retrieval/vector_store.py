import os
import uuid

from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.utils import load_config

# all-MiniLM-L6-v2 output dimension
VECTOR_DIM = 384


class VectorStore:
    def __init__(self):
        cfg = load_config()
        qcfg = cfg["qdrant"]
        api_key = os.environ.get("QDRANT_API_KEY")
        url = os.environ.get("QDRANT_URL") or qcfg.get("url")

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host=qcfg["host"], port=qcfg["port"])

        self.collection = qcfg["collection_name"]

    def create_collection(self, recreate: bool = False):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            if recreate:
                self.client.delete_collection(self.collection)
            else:
                return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

    def upsert(self, embedded_chunks: list[tuple[Document, list[float]]]):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                },
            )
            for doc, vector in embedded_chunks
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def similarity_search(self, query_vector: list[float], top_k: int) -> list[dict]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": r.payload["text"],
                "source": r.payload["source"],
                "chunk_id": r.payload["chunk_id"],
                "score": r.score,
            }
            for r in response.points
        ]
