from typing import TypedDict

from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    item_results: list[Document]
    document_results: list[Document]
    answer: str