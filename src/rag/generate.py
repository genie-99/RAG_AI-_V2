from langchain_openai import ChatOpenAI

from config import GMS_KEY, OPENAI_BASE_URL, CHAT_MODEL, PROJECT_ROOT
from src.rag.retriever import search_items, search_documents

def generate_answer(question, results):
    #검색 결과에서 원문 문자열 꺼내기
    texts =[]

    for result in results:
        texts.append(result.page_content)

    #검색된 원문들을 빈 줄로 구분해서 묶기
    context = "\n\n".join(texts)

    #답변을 작성할 모델 선정
    chat_model = ChatOpenAI(
        api_key=GMS_KEY,
        base_url=OPENAI_BASE_URL,
        model=CHAT_MODEL,
    )

    #프롬프트 파일 읽기
    prompt_path = PROJECT_ROOT / "prompt" / "answer_prompt.txt"
    template = prompt_path.read_text(encoding="utf-8")

    #빈자리에 실제 참고 자료와 질문 넣기
    prompt = template.format(
        context = context,
        question= question,
    )
    print(prompt)
    print()
    #모델에 답변 요청
    response = chat_model.invoke(prompt)

    return response.content

def answer_question(question):
    # RAG1: 내부 JSON에서 식물 정보 검색
    item_results = search_items(question, k=3)

    # RAG2: 외부 문서에서 관련 문서 조각 검색
    document_results = search_documents(question, k=3)

    # 두 검색 결과 합치기
    results = item_results + document_results

    # 검색한 자료를 바탕으로 답변 생성
    answer = generate_answer(question, results)

    return answer

if __name__ == "__main__":
    question = "빛이 적은 실내에서 키울 수 있는 식물을 찾고 있어"

    answer = answer_question(question)

    print(answer)
