from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

from src.utils import load_config


def load_documents(data_dir: Optional[str] = None) -> list[Document]:
    cfg = load_config()
    dir_path = Path(data_dir or cfg["paths"]["raw_data"])
    docs = []
    for txt_file in sorted(dir_path.glob("*.txt")):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        for doc in loader.load():
            doc.metadata["source"] = txt_file.name
            doc.metadata["source_path"] = str(txt_file)
            docs.append(doc)
    return docs
