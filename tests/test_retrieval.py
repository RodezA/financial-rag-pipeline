from unittest.mock import MagicMock, patch

from langchain.schema import Document

from src.retrieval.vector_store import VectorStore


@patch("src.retrieval.vector_store.QdrantClient")
def test_create_collection_when_not_exists(MockQdrant):
    mock_client = MockQdrant.return_value
    mock_client.get_collections.return_value.collections = []

    store = VectorStore()
    store.create_collection()

    mock_client.create_collection.assert_called_once()


@patch("src.retrieval.vector_store.QdrantClient")
def test_create_collection_skips_if_exists(MockQdrant):
    mock_client = MockQdrant.return_value
    existing = MagicMock()
    existing.name = "financial_docs"
    mock_client.get_collections.return_value.collections = [existing]

    store = VectorStore()
    store.create_collection(recreate=False)

    mock_client.create_collection.assert_not_called()


@patch("src.retrieval.vector_store.QdrantClient")
def test_create_collection_recreates_when_flag_set(MockQdrant):
    mock_client = MockQdrant.return_value
    existing = MagicMock()
    existing.name = "financial_docs"
    mock_client.get_collections.return_value.collections = [existing]

    store = VectorStore()
    store.create_collection(recreate=True)

    mock_client.delete_collection.assert_called_once_with("financial_docs")
    mock_client.create_collection.assert_called_once()


@patch("src.retrieval.vector_store.QdrantClient")
def test_upsert_sends_correct_point_structure(MockQdrant):
    mock_client = MockQdrant.return_value

    doc = Document(page_content="Revenue was $412.7M", metadata={"source": "report.txt", "chunk_id": 0})
    vector = [0.1] * 1536

    store = VectorStore()
    store.upsert([(doc, vector)])

    mock_client.upsert.assert_called_once()
    call_kwargs = mock_client.upsert.call_args
    points = call_kwargs.kwargs.get("points") or call_kwargs.args[1] if call_kwargs.args else call_kwargs.kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["text"] == "Revenue was $412.7M"
    assert points[0].payload["source"] == "report.txt"


@patch("src.retrieval.vector_store.QdrantClient")
def test_similarity_search_returns_correct_shape(MockQdrant):
    mock_client = MockQdrant.return_value
    mock_point = MagicMock()
    mock_point.payload = {"text": "Net revenue $412.7M", "source": "novatech_q1_2024.txt", "chunk_id": 3}
    mock_point.score = 0.92
    mock_client.query_points.return_value.points = [mock_point]

    store = VectorStore()
    results = store.similarity_search([0.1] * 1536, top_k=5)

    assert len(results) == 1
    assert results[0]["text"] == "Net revenue $412.7M"
    assert results[0]["source"] == "novatech_q1_2024.txt"
    assert results[0]["score"] == 0.92
