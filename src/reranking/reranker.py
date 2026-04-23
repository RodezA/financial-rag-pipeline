from sentence_transformers import CrossEncoder

from src.utils import load_config


class Reranker:
    def __init__(self):
        cfg = load_config()
        rcfg = cfg["reranking"]
        self.model = CrossEncoder(rcfg["model"])
        self.top_k = rcfg["top_k_after_rerank"]

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return chunks
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[: self.top_k]
