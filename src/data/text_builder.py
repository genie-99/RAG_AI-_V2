import json

from src.data.data_loader import load_items as load_items
from src.data.data_loader import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_item_texts(items):
    texts = []

    for item in items:
        text = json.dumps(item, ensure_ascii=False)
        texts.append(text)

    return texts

def split_documents(documents):
    # 글을 어떤 크기로 나눌지 설정
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap=50,
    )

    #페이지별 Documents를 조각별 Document로 나누기
    chunks = splitter.split_documents(documents)

    return chunks

if __name__ == "__main__":
    items = load_items()
    texts = build_item_texts(items)

    print(f"불러온 식물 수: {len(items)}")
    print(f"변환한 문자열 수: {len(texts)}")
    print(texts[0])  # 확인용으로 첫 번째 결과만 출력

    documents = load_documents()
    chunks = split_documents(documents)

    print(f"불러온 페이지 수: {len(documents)}")
    print(f"나눈 조각 수: {len(chunks)}")

    # 처음 두 조각의 내용과 출처 확인
    for chunk in chunks[:2]:
        print(chunk.metadata)
        print(chunk.page_content)
        print()