import tempfile
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents


def test_loader_reads_txt_files(tmp_path):
    (tmp_path / "doc1.txt").write_text("Hello world. This is a test document.")
    (tmp_path / "doc2.txt").write_text("Another document with financial data.")

    docs = load_documents(data_dir=str(tmp_path))

    assert len(docs) == 2
    sources = {d.metadata["source"] for d in docs}
    assert sources == {"doc1.txt", "doc2.txt"}


def test_loader_sets_source_metadata(tmp_path):
    (tmp_path / "report.txt").write_text("Revenue: $100M")

    docs = load_documents(data_dir=str(tmp_path))

    assert docs[0].metadata["source"] == "report.txt"
    assert "source_path" in docs[0].metadata


def test_chunker_splits_large_document(tmp_path):
    long_text = "This is a sentence. " * 200
    (tmp_path / "long.txt").write_text(long_text)

    docs = load_documents(data_dir=str(tmp_path))
    chunks = chunk_documents(docs)

    assert len(chunks) > 1


def test_chunker_preserves_source_metadata(tmp_path):
    (tmp_path / "financials.txt").write_text("Revenue: $412.7M\n\n" * 50)

    docs = load_documents(data_dir=str(tmp_path))
    chunks = chunk_documents(docs)

    for chunk in chunks:
        assert chunk.metadata["source"] == "financials.txt"
        assert "chunk_id" in chunk.metadata


def test_chunker_assigns_sequential_chunk_ids(tmp_path):
    (tmp_path / "test.txt").write_text("Paragraph one.\n\n" * 60)

    docs = load_documents(data_dir=str(tmp_path))
    chunks = chunk_documents(docs)

    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))
