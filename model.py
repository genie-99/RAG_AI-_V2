from typing import TypedDict, Annotated

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# 앞에서 만든 고정된 순서의 RAG에서 사용하는 상태
class RAGState(TypedDict):
    question: str
    item_results: list[Document]
    document_results: list[Document]
    answer: str

# 에이전트에서 사용하는 상태
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

