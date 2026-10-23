import os
from pathlib import Path

from dotenv import load_dotenv

# 현재 config.py 파일이 있는 폴더를 프로젝트 기준 위치로 사용합니다.
PROJECT_ROOT = Path(__file__).resolve().parent

# .env의 설정을 환경변수로 읽어옵니다.
load_dotenv(PROJECT_ROOT / ".env")

# 환경변수에서 값을 꺼내 다른 파일에서도 사용할 수 있게 이름을 붙입니다.
GMS_KEY = os.getenv("GMS_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
HTTP_USER_AGENT = os.getenv("HTTP_USER_AGENT")

# 식물 정보와 문서를 보관할 위치입니다.
DATA_DIR = PROJECT_ROOT / "data"

# 임베딩한 데이터를 저장할 벡터 DB 위치입니다.
DB_DIR = PROJECT_ROOT / "storage" / "chroma"
