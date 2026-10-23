import json

from config import DATA_DIR
from pypdf import PdfReader
from langchain_core.documents import Document
import easyocr
import pymupdf


# JSON 파일을 읽어서 리스트와 딕셔너리로 변환해서 전달하는 메서드
def load_items():
    # 원본 JSON 파일의 위치
    file_path= DATA_DIR / "raw" / "items.json"

    # 파일 내용을 문자 열로
    text = file_path.read_text(encoding="utf-8-sig")

    # JSON 문자열을 Python 리스트와 딕셔너리로 변환
    items = json.loads(text)

    return items

def load_documents():
    folder_path = DATA_DIR / "raw" / "documents"
    file_paths = sorted(folder_path.glob("*.pdf"))

    # OCR은 한 번 준비해서 모든 페이지에 사용
    ocr = easyocr.Reader(["ko", "en"], gpu=False)

    documents = []

    for file_path in file_paths:
        plant_id = file_path.name.split("_")[0]

        # OCR에 전달할 이미지를 만들 수 있도록 PyMuPDF로 열기
        with pymupdf.open(file_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):

                # 여기서 우리가 만든 OCR 함수를 호출
                text = extract_page_text(page, ocr)

                document = Document(
                    page_content=text,
                    metadata={
                        "plant_id": plant_id,
                        "source": file_path.name,
                        "page": page_number,
                    },
                )

                documents.append(document)

    # 모든 파일을 처리한 뒤 반환
    return documents

def extract_page_text(page, ocr):
    # PDF 한 페이지를 이미지로 변환
    image = page.get_pixmap(dpi=200)

    # OCR에 전달할 PNG 이미지 데이터
    image_bytes = image.tobytes("png")

    # 이미지에서 읽은 글자들을 문자열 리스트로 반환
    lines = ocr.readtext(image_bytes, detail=0)

    # 한 페이지의 내용을 문자열 하나로 합치기
    text = "\n".join(lines)

    return text


if __name__ == "__main__" :
    
    items = load_items()

    print(f"불러온 식물 수: {len(items)}")

    documents = load_documents()

    print(f"불러온 페이지 수: {len(documents)}")
    print(documents[0].metadata)
    print(documents[0].page_content[:500])
    