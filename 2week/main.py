import os
import pandas as pd
from qdrant_client import QdrantClient
import openai  # 최신 OpenAI 라이브러리 사용
from dotenv import load_dotenv
from qdrant_client.models import VectorParams, Distance

# .env 파일에서 API 키 로드
env_path = os.path.join(os.path.dirname(__file__), "../env.env")
load_dotenv(env_path)
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

# FAQ.xlsx 파일 읽기
faq_df = pd.read_excel("FAQ.xlsx")

# 1열은 질문, 2열은 답변 추출
questions = faq_df.iloc[:, 0].tolist()
answers = faq_df.iloc[:, 1].tolist()

# 질문과 답변을 하나의 텍스트로 결합 (문맥 보강)
faq_texts = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]

# 전체 FAQ 텍스트에 대해 한 번의 배치 요청으로 임베딩 생성
response = openai.embeddings.create(
    input=faq_texts,
    model="text-embedding-3-small"
)

# 각 텍스트에 대한 임베딩 리스트 생성 (속성 접근 방식 사용)
embeddings_list = [data.embedding for data in response.data]
embedding_size = len(embeddings_list[0]) ## 임베딩 크기는 모델마다 고유한 값으로 보통 고정되어 첫번째 값 추출하는 것 괜춘

# Qdrant 클라이언트 생성 (Docker 컨테이너가 localhost:6333에서 실행 중이어야 함)
qdrant_client = QdrantClient(host="localhost", port=6333)

# 컬렉션 이름 설정 및 재생성 (최신 API에 맞게 vectors_config 사용)
collection_name = "QNA_Collection"

# 컬렉션이 존재하지 않을 때만 새로 생성
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )

# 각 FAQ 항목에 대해 임베딩과 텍스트 페이로드를 Qdrant에 업로드
points = []
for idx, (text, embedding) in enumerate(zip(faq_texts, embeddings_list)):
    points.append({
        "id": idx,
        "vector": embedding,
        "payload": {"text": text}
    })

qdrant_client.upsert(
    collection_name=collection_name,
    points=points
)

print("FAQ 데이터가 Qdrant 벡터 DB에 저장되었습니다!")

