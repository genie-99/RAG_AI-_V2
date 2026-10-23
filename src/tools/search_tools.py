from langchain_core.tools import Tool
from src.rag.retriever import search_items, search_documents

search_tools = [
    Tool.from_function(
        func=search_items,
        name="search_items",
        description=(
            "내부 JSON에서 식물 후보와 기본 관리 정보를 검색합니다."
            "입력은 조건이나 식물 이름이 포함된 검색 질문 문자열입니다."
        ),
    ),
    Tool.from_function(
        func=search_documents,
        name="search_documents",
        description=(
            "저장된 외부 PDF에서 식물의 상세 관리 방법과 근거를 검색합니다."
            "입력은 식물 이름과 확인할 내용이 포함된 검색 질문 문자열입니다."
        ),
    ),
]