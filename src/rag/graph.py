from langgraph.graph import StateGraph, START, END

from model import RAGState

from src.rag.retriever import search_items, search_documents
from src.rag.generate import generate_answer

#첫 번째 작업: RAG1 검색
def search_items_node(state: RAGState):
    results = search_items(state["question"], k=3)

    return {"item_results": results}

# 두 번째 작업 : RAG2 검색
def search_documents_node(state: RAGState):
    results = search_documents(state["question"], k=3)

    return {"document_results" : results}

# 세 번째 작업: 두 결과로 답변 생성
def generate_answer_node(state: RAGState):
    results = state["item_results"]+ state["document_results"]

    answer = generate_answer(state["question"], results)

    return {"answer" : answer}

# model.py에서 가져온 데이터 구조 사용
builder = StateGraph(RAGState)

#그래프에 작업 등록
builder.add_node("search_items",search_items_node)
builder.add_node("search_documents", search_documents_node)
builder.add_node("generate_answer",generate_answer_node)

#작업 실행 순서 연결
builder.add_edge(START, "search_items")
builder.add_edge("search_items", "search_documents")
builder.add_edge("search_documents", "generate_answer")
builder.add_edge("generate_answer", END)

#실행할 수 있는 그래프로 만들기
graph = builder.compile()

if __name__ == "__main__":
    question = "빛이 적은 실내에서 키울 수 있는 식물을 찾고 있어"

    result = graph.invoke({"question": question})

    print(f"RAG1 검색 결과 수: {len(result['item_results'])}")
    print(f"RAG2 검색 결과 수: {len(result['document_results'])}")
    print(result["answer"])