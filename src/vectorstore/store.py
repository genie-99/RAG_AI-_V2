from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from functools import lru_cache

from config import GMS_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL, DB_DIR
from src.data.data_loader import load_items as loader
from src.data.text_builder import build_item_texts as builder

@lru_cache(maxsize=1)
def get_stores() :
    #어떤 API와 모델로 문자열로 임베딩을 할지 설정
    embeddings = OpenAIEmbeddings(
        api_key=GMS_KEY,
        base_url=OPENAI_BASE_URL,
        model=EMBEDDING_MODEL,
        check_embedding_ctx_length=False, #문자열을 그대로 API에 전달
    )

    stores = {}

    #식물 정보를 보관할 DB 열기
    for name in ["rag1_items","rag2_documents"]:
        stores[name] = Chroma(
            collection_name=name, # collection이름
            embedding_function=embeddings, # embedding에 대한 정보
            persist_directory=str(DB_DIR), # DB존재 경로
        )

    return stores

def get_store(collection_name="rag1_items"):
    stores = get_stores()

    return stores[collection_name]

def save_items(items, texts):
    store = get_store("rag1_items")

    ids = []

    for item in items:
        ids.append(item["plant_id"])

    saved_ids = store.add_texts(
        texts=texts,
        ids=ids,
    )

    return saved_ids


def save_documents(chunks):
    store = get_store("rag2_documents")

    ids = []
    page_counts = {}

    for chunk in chunks:
        source = chunk.metadata["source"]
        page = chunk.metadata["page"]

        page_key = f"{source}:{page}"
        chunk_number = page_counts.get(page_key, 0)

        chunk_id = f"{page_key}:{chunk_number}"
        ids.append(chunk_id)

        page_counts[page_key] = chunk_number + 1

    saved_ids = store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return saved_ids

if __name__ == "__main__":
    from src.data.data_loader import load_documents
    from src.data.text_builder import split_documents

    print("PDF에서 OCR로 글자를 읽고 있습니다.")
    documents = load_documents()
    print(f"불러온 페이지 수: {len(documents)}")

    chunks = split_documents(documents)
    print(f"나눈 조각 수: {len(chunks)}")

    if chunks:
        saved_ids = save_documents(chunks)
        print(f"저장한 문서 조각 수: {len(saved_ids)}")
    else:
        print("OCR에서 본문을 읽었는지 확인해 주세요.")