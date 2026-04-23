from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.utils import load_config


def chunk_documents(docs: list[Document]) -> list[Document]:
    cfg = load_config()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
        # "━" is ━ — section divider used in the financial docs
        separators=["\n\n", "\n", "━", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
