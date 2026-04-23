from unittest.mock import MagicMock, patch


@patch("src.pipeline.LLMClient")
@patch("src.pipeline.Reranker")
@patch("src.pipeline.VectorStore")
@patch("src.pipeline.OpenAI")
def test_pipeline_query_returns_required_keys(MockOpenAI, MockVectorStore, MockReranker, MockLLM):
    # OpenAI embedding stub
    embed_response = MagicMock()
    embed_response.data = [MagicMock(embedding=[0.1] * 1536)]
    MockOpenAI.return_value.embeddings.create.return_value = embed_response

    # VectorStore stub
    chunks = [{"text": "Revenue $412.7M", "source": "novatech_q1_2024.txt", "chunk_id": 0, "score": 0.92}]
    MockVectorStore.return_value.similarity_search.return_value = chunks

    # Reranker stub — adds rerank_score and returns truncated
    reranked = [{"text": "Revenue $412.7M", "source": "novatech_q1_2024.txt", "chunk_id": 0, "score": 0.92, "rerank_score": 8.4}]
    MockReranker.return_value.rerank.return_value = reranked
    MockReranker.return_value.top_k = 4

    # LLM stub
    MockLLM.return_value.generate.return_value = {
        "answer": "NovaTech's net revenue in Q1 2024 was $412.7 million.",
        "usage": {
            "input_tokens": 400,
            "output_tokens": 30,
            "cache_creation_input_tokens": 380,
            "cache_read_input_tokens": 0,
        },
    }

    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline(use_reranker=True)
    result = pipeline.query("What was NovaTech revenue Q1 2024?")

    required_keys = {"answer", "source_docs", "retrieval_scores", "rerank_scores", "token_usage", "latency_ms"}
    assert required_keys.issubset(result.keys())


@patch("src.pipeline.LLMClient")
@patch("src.pipeline.Reranker")
@patch("src.pipeline.VectorStore")
@patch("src.pipeline.OpenAI")
def test_pipeline_skips_reranker_when_disabled(MockOpenAI, MockVectorStore, MockReranker, MockLLM):
    embed_response = MagicMock()
    embed_response.data = [MagicMock(embedding=[0.1] * 1536)]
    MockOpenAI.return_value.embeddings.create.return_value = embed_response

    chunks = [
        {"text": f"chunk {i}", "source": "doc.txt", "chunk_id": i, "score": 0.9 - i * 0.1}
        for i in range(10)
    ]
    MockVectorStore.return_value.similarity_search.return_value = chunks
    MockLLM.return_value.generate.return_value = {
        "answer": "answer",
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }

    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline(use_reranker=False)
    pipeline.query("test question")

    MockReranker.return_value.rerank.assert_not_called()


@patch("src.pipeline.LLMClient")
@patch("src.pipeline.Reranker")
@patch("src.pipeline.VectorStore")
@patch("src.pipeline.OpenAI")
def test_pipeline_latency_ms_is_positive(MockOpenAI, MockVectorStore, MockReranker, MockLLM):
    embed_response = MagicMock()
    embed_response.data = [MagicMock(embedding=[0.1] * 1536)]
    MockOpenAI.return_value.embeddings.create.return_value = embed_response
    MockVectorStore.return_value.similarity_search.return_value = []
    MockReranker.return_value.rerank.return_value = []
    MockReranker.return_value.top_k = 4
    MockLLM.return_value.generate.return_value = {
        "answer": "answer",
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }

    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline(use_reranker=True)
    result = pipeline.query("query")

    assert result["latency_ms"] >= 0
