from src.vectorstore.store import get_store

def search_items(question, k=3):
    #저장해 둔 식물 DB 열기
    store = get_store()

    #질문과 가까운 식물 정보 검색 # L2 거리를 사
    results = store.similarity_search(
        query=question,
        k=k,
    )

    return results # Document라는 객체를 반환

# RAG2: 외부 문서의 조각 검색
def search_documents(question, k=3):
    store = get_store("rag2_documents")

    results = store.similarity_search(
        query=question,
        k=k,
    )

    return results

if __name__ == "__main__":
    question = "빛이 적은 실내에서 키울 수 있는 식물을 찾고 있어"

    results = search_items(question, k=3)

    print(f"검색된 식물 수: {len(results)}")

    for result in results:
        print(result.page_content)
        print()