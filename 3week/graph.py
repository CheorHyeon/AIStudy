import os
from qdrant_client import QdrantClient
import openai  # 최신 OpenAI 라이브러리 사용
from dotenv import load_dotenv
from qdrant_client.models import VectorParams, Distance
from langchain import LLMChain, PromptTemplate
## Open AI
from langchain_openai import OpenAI  # 최신 방식으로 import
import numpy as np
from numpy.linalg import norm
from langchain.llms import OpenAI as LLM_OpenAI

# ============================================================
# Step 0. 환경 변수 로드 및 API 키 설정
# ============================================================
env_path = os.path.join(os.path.dirname(__file__), "../env.env")
load_dotenv(env_path)
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

# ============================================================
# Step 1. 사원 상태(State)에서 부서 정보 추출 (여기서는 가상 상태 사용)
# ============================================================
state = {
    "user_id": 123,
    "department": "developer"  # 현재는 developer 부서로 가정
}
department = state["department"]

# ============================================================
# Step 2. 부서에 따른 온보딩 문제 선택
#     - 현재 developer 부서의 문제를 Qdrant에서 불러옵니다.
# ============================================================
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "developer_onboarding_quiz"

# Qdrant에서 문제들을 스크롤로 가져오고, 문제 번호(quiz_id) 순으로 정렬
scroll_result = qdrant_client.scroll(collection_name=collection_name, limit=100)
# 각 포인트의 id는 quiz 번호로 설정되어 있다고 가정
problems = sorted(scroll_result.points, key=lambda p: p.id)

# ============================================================
# Step 3. 문제 순서대로 사용자에게 질문 및 응답 수집
# ============================================================
user_responses = {}  # 각 문제의 응답 저장 딕셔너리
print("온보딩 평가를 시작합니다.\n")
for point in problems:
    payload = point.payload
    quiz_id = payload.get("quiz_id")
    question = payload.get("question")
    correct_answer = payload.get("answer")
    explanation = payload.get("explanation")

    # 사용자에게 질문 출력 (실제 운영 시 챗봇 인터페이스나 웹폼 등 활용)
    print(f"문제 {quiz_id}: {question}")
    user_answer = input("답변을 입력해주세요: ")

    user_responses[quiz_id] = {
        "question": question,
        "correct_answer": correct_answer,
        "explanation": explanation,
        "user_answer": user_answer
    }
    print("\n--------------------------------------\n")

# --------------------------------------------------
# Step 4. 사용자 답변을 임베딩하기 (유사도 비교를 위해)
#         - 여기서는 사용자가 입력한 답변을 임베딩해요.
# --------------------------------------------------
user_answer_texts = [data["user_answer"] for data in user_responses.values()]
user_embed_response = openai.embeddings.create(
    input=user_answer_texts,
    model="text-embedding-3-small"
)
user_embeddings = [data.embedding for data in user_embed_response.data]


# --------------------------------------------------
# Step 5. 임베딩 유사도 계산하여 정답 여부 판별하기
#         - 정답 임베딩과 사용자의 답변 임베딩을 비교해요.
# --------------------------------------------------
def cosine_similarity(vec1, vec2):
    # 두 벡터가 얼마나 비슷한지 계산해요. 값이 1에 가까울수록 비슷해요.
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


embedding_threshold = 0.8  # 유사도가 0.8 이상이면 정답으로 판단해요.
embedding_results = {}  # 각 문제별 유사도와 정답 여부 저장

# 개발 문제 데이터셋 순서와 동일하게 맞춰서 진행해요.
total_correct = 0  # 맞은 문제 수

# user_responses를 quiz_id 순서대로 돌면서 처리해요.
for idx, (quiz_id, data) in enumerate(sorted(user_responses.items())):
    # Qdrant에서 해당 quiz_id에 저장된 정답 임베딩을 가져와요.
    point = qdrant_client.get_point(collection_name=collection_name, point_id=quiz_id)
    correct_embedding = point.vector  # Qdrant에 저장된 정답 임베딩 벡터예요.
    user_embedding = user_embeddings[idx]  # 사용자가 입력한 답의 임베딩 벡터예요.
    similarity = cosine_similarity(correct_embedding, user_embedding)  # 두 벡터가 얼마나 비슷한지 계산해요.

    # 유사도가 threshold 이상이면 정답으로 간주해요.
    is_correct = similarity >= embedding_threshold
    if is_correct:
        total_correct += 1

    # 각 문제별 결과를 저장해요.
    embedding_results[quiz_id] = {
        "similarity": similarity,
        "is_correct": is_correct,
        "data": data  # 문제 정보와 사용자의 답변을 함께 저장해요.
    }
# ============================================================
# Step 6. 통합 LLM 채점 프롬프트 구성하기
# ============================================================
# 이제 모든 문제에 대한 정보를 한 번에 LLM에 전달해서 피드백을 받으려고 해요.
# 먼저, 전체 결과(맞은 개수와 틀린 개수)를 계산해요.
total_questions = len(user_responses)
total_wrong = total_questions - total_correct

# 통합 프롬프트 메시지를 만듭니다.
# 틀린 문제에 대한 정보를 나열해요.
mistake_details = ""
for quiz_id, res in sorted(embedding_results.items()):
    # 틀린 문제만 추가해요.
    if not res["is_correct"]:
        data = res["data"]
        mistake_details += f"{quiz_id}번:\n"
        mistake_details += f"정답: {data['correct_answer']}\n"
        mistake_details += f"해설: {data['explanation']}\n"
        mistake_details += f"사용자 답변: {data['user_answer']}\n\n"

# 통합 프롬프트 메시지 작성
final_prompt = f"""
당신은 00회사의 신입사원 온보딩 문제 채점 및 해설 제공사 입니다.

000 신입사원 분의 온보딩 결과는 총 {total_questions}문제 중에 {total_correct}문제만 정답으로 판단했습니다.
틀린 문제는 총 {total_wrong}개 입니다.

아래 틀린 문제들에 대해, 각 문제에서 간략한 해설을 바탕으로 추가 설명과 관련된 학습 포인트를 문항별로 자세히 설명해주시기 바랍니다.

{mistake_details}

추가로, 총평으로 해당 신입 사원에게 한마디 짧게, 예를 들어 "업무에 필요한 JPA, 도커 등을 학습해두시면 업무에 큰 도움이 될 것입니다."와 같이 학습 포인트를 추천해 주세요.
"""

# ============================================================
# Step 7. LLM에 통합 프롬프트 전달하고 피드백 받기
# ============================================================
# LLM 체인을 사용해서 통합 피드백을 받아요.
llm = OpenAI(model="gpt-4o-mini", temperature=0)
# 단일 프롬프트이므로 LLM 체인을 사용할 수도 있고, 바로 호출할 수도 있어요.
final_feedback = llm(final_prompt)

# ============================================================
# Step 8. 최종 결과 출력하기
# ============================================================
print("\n===============================")
print(f"총 {total_questions}문제 중 {total_correct}문제 맞으셨습니다.\n")
print("통합 피드백 결과:")
print(final_feedback)