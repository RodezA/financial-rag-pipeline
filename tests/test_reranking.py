from unittest.mock import MagicMock, patch

from src.reranking.reranker import Reranker


@patch("src.reranking.reranker.CrossEncoder")
def test_reranker_sorts_by_score_descending(MockEncoder):
    MockEncoder.return_value.predict.return_value = [0.2, 0.9, 0.5]

    reranker = Reranker()
    chunks = [
        {"text": "chunk A", "source": "a.txt", "chunk_id": 0, "score": 0.8},
        {"text": "chunk B", "source": "b.txt", "chunk_id": 1, "score": 0.7},
        {"text": "chunk C", "source": "c.txt", "chunk_id": 2, "score": 0.6},
    ]

    result = reranker.rerank("test query", chunks)

    assert result[0]["text"] == "chunk B"
    assert result[1]["text"] == "chunk C"


@patch("src.reranking.reranker.CrossEncoder")
def test_reranker_truncates_to_top_k(MockEncoder):
    MockEncoder.return_value.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

    reranker = Reranker()
    chunks = [
        {"text": f"chunk {i}", "source": "x.txt", "chunk_id": i, "score": 0.5}
        for i in range(5)
    ]

    result = reranker.rerank("query", chunks)

    assert len(result) == reranker.top_k


@patch("src.reranking.reranker.CrossEncoder")
def test_reranker_adds_rerank_score_key(MockEncoder):
    MockEncoder.return_value.predict.return_value = [0.75]

    reranker = Reranker()
    chunks = [{"text": "some text", "source": "doc.txt", "chunk_id": 0, "score": 0.6}]

    result = reranker.rerank("query", chunks)

    assert "rerank_score" in result[0]
    assert result[0]["rerank_score"] == 0.75


@patch("src.reranking.reranker.CrossEncoder")
def test_reranker_returns_empty_for_empty_input(MockEncoder):
    reranker = Reranker()
    result = reranker.rerank("query", [])
    assert result == []
    MockEncoder.return_value.predict.assert_not_called()
