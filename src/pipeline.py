import time

from openai import OpenAI

from src.generation.llm_client import LLMClient
from src.reranking.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.utils import load_config


class RAGPipeline:
    def __init__(self, use_reranker: bool = True):
        cfg = load_config()
        self.use_reranker = use_reranker
        self.top_k_retrieval = cfg["retrieval"]["top_k"]
        self.top_k_final = cfg["reranking"]["top_k_after_rerank"]
        self.embedding_model = cfg["embedding"]["model"]

        self._openai = OpenAI()
        self.vector_store = VectorStore()
        self.reranker = Reranker() if use_reranker else None
        self.llm = LLMClient()

    def query(self, question: str) -> dict:
        t0 = time.monotonic()

        query_vector = self._embed_query(question)
        retrieved = self.vector_store.similarity_search(query_vector, self.top_k_retrieval)

        if self.use_reranker and self.reranker:
            final_chunks = self.reranker.rerank(question, retrieved)
        else:
            final_chunks = retrieved[: self.top_k_final]

        generation = self.llm.generate(question, final_chunks)

        return {
            "answer": generation["answer"],
            "source_docs": final_chunks,
            "retrieval_scores": [c["score"] for c in retrieved],
            "rerank_scores": [c.get("rerank_score") for c in final_chunks],
            "token_usage": generation["usage"],
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    def _embed_query(self, text: str) -> list[float]:
        response = self._openai.embeddings.create(
            model=self.embedding_model, input=[text]
        )
        return response.data[0].embedding
