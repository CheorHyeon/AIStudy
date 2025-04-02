import os
import pandas as pd
from qdrant_client import QdrantClient
import openai  # 최신 OpenAI 라이브러리 사용
from dotenv import load_dotenv
from qdrant_client.models import VectorParams, Distance

## Open AI
from langchain_openai import OpenAI  # 최신 방식으로 import
## 프롬프트 템플릿 -> 어떻게 질문을 보여줄 지 미리 작성하는 방법
from langchain.prompts import PromptTemplate
## env 파일을 읽어올 수 있는 도구
from dotenv import load_dotenv
## 컴퓨터 안의 환경 변수나 파일 경로 등을 다루기 위한 Python의 기본 모듈
import os

# ------------------------------
# RAG: 사용자 질문에 대해 유사 FAQ 검색 후 답변 생성
# ------------------------------

# 현재 파일(main.py)이 있는 폴더 기준으로 상위 폴더의 env.env 경로 계산
# load_dotenv() 함수는 .env 파일에 저장된 환경변수를 읽어서 os.environ dictionary에 추가해줌
env_path = os.path.join(os.path.dirname(__file__), "../env.env")
load_dotenv(env_path)

# OpenAI API 키 설정 (환경변수 사용)
## OpenAI 모델에서는 별도 API키 매개변수로 넘기지 않아도 내부에서 자동으로 읽어 사용하는 경우가 많지만 에러나서 명시적 호출
api_key = os.environ.get("OPENAI_API_KEY") # 위에서 추가한거 꺼내오기

# 1. 사용자 질문 입력
user_question = input("질문을 입력하세요: ")

# 2. 사용자 질문의 임베딩 생성 (배열로 전달)
query_response = openai.embeddings.create(
    input=user_question,
    model="text-embedding-3-small"
)
query_embedding = query_response.data[0].embedding

# Qdrant 클라이언트 생성 (Docker 컨테이너가 localhost:6333에서 실행 중이어야 함)
qdrant_client = QdrantClient(host="localhost", port=6333)

# 컬렉션 이름 설정 및 재생성 (최신 API에 맞게 vectors_config 사용)
collection_name = "QNA_Collection"

# 3. Qdrant에서 유사 FAQ 항목 5개 검색
hits_response = qdrant_client.query_points(
    collection_name=collection_name,
    query=query_embedding,  # query 로 변경
    limit=5
)
hits = hits_response.points  # 결과 객체의 result 속성에서 실제 결과 리스트를 추출

# 검색 결과에서 페이로드(텍스트) 추출
context_texts = [hit.payload.get("text", "") for hit in hits]
context = "\n\n".join(context_texts)
print("\n[검색된 유사 FAQ 컨텍스트]:\n", context)

# 4. LangChain의 프롬프트 템플릿 구성
prompt_template = """
당신은 KT M모바일 고객 상담원입니다. 아래 Context를 참고하여 사용자에게 응답해주시길 바랍니다.

만일 모른다면 모른다고 답변 하시길 바랍니다.

Context:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# 5. LangChain의 LLM 체인 생성 (OpenAI LLM 사용)
# OpenAI의 LLM 인스턴스 생성
# temperature 값은 모델 답변의 창의성(랜덤성)을 조절합니다. (낮은 값은 결정적인 답변, 높은 값은 다채로운 답변)
# 낮은 온도 초기화 시 응답의 무작위성 줄일 수 있음
llm = OpenAI(temperature=0.1, api_key = api_key)
chain = prompt | llm

# 6. 체인을 실행하여 최종 답변 생성
result = chain.invoke({"context": context, "question": user_question})
print("\n[답변]:\n", result)