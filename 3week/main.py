import os
from qdrant_client import QdrantClient
import openai  # 최신 OpenAI 라이브러리 사용
from dotenv import load_dotenv
from qdrant_client.models import VectorParams, Distance

# .env 파일에서 API 키 로드
env_path = os.path.join(os.path.dirname(__file__), "../env.env")
load_dotenv(env_path)
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

# 주관식 정답 데이터셋 생성
developer_quiz_dataset = [
    {
        "id": 1,
        "question": "슬랙에 출근체크를 해야 하는 시간은?",
        "correct_answer": "슬랙 출근체크는 10시 전으로 이루어져야 합니다.",
        "explanation": "정확한 시간 준수가 업무 효율성을 높이는 중요한 요소입니다."
    },
    {
        "id": 2,
        "question": "우리 서비스의 핵심 기능은 무엇인가요?",
        "correct_answer": "AI를 활용한 이력서 분석 및 매칭 시스템입니다.",
        "explanation": "이 시스템은 지원자와 기업 간의 최적의 매칭을 도와줍니다."
    },
    {
        "id": 3,
        "question": "개발팀 내 코드 리뷰 과정에서 중요한 요소는 무엇인가요?",
        "correct_answer": "코드 품질, 기능 검증, 보안 점검 등 모든 측면을 꼼꼼히 확인하는 것입니다.",
        "explanation": "종합적인 리뷰가 소프트웨어 안정성과 품질 향상에 기여합니다."
    },
    {
        "id": 4,
        "question": "Git에서 feature 브랜치를 사용하는 주된 이유는 무엇인가요?",
        "correct_answer": "독립적인 기능 개발 및 테스트를 위해서입니다.",
        "explanation": "분리된 브랜치에서 작업함으로써 메인 코드의 안정성을 유지할 수 있습니다."
    },
    {
        "id": 5,
        "question": "Jenkins의 주요 역할은 무엇인가요?",
        "correct_answer": "지속적 통합과 배포 자동화를 지원하는 것입니다.",
        "explanation": "이를 통해 개발 프로세스가 원활하고 효율적으로 진행됩니다."
    },
    {
        "id": 6,
        "question": "Kubernetes의 핵심 역할은 무엇인가요?",
        "correct_answer": "컨테이너화된 애플리케이션의 자동 배포와 관리를 수행하는 것입니다.",
        "explanation": "효율적인 인프라 운영을 위해 Kubernetes가 사용됩니다."
    },
    {
        "id": 7,
        "question": "AWS를 활용한 배포 전략에서 중요한 요소는 무엇인가요?",
        "correct_answer": "인스턴스 선택, 네트워크 구성, 오토스케일링 등 모든 요소를 종합적으로 고려해야 합니다.",
        "explanation": "전체 시스템의 안정성을 위해 다각도로 접근해야 합니다."
    },
    {
        "id": 8,
        "question": "서비스 안정화를 위해 주로 사용하는 모니터링 도구는 무엇인가요?",
        "correct_answer": "Grafana를 활용하여 시각화 및 모니터링을 수행합니다.",
        "explanation": "Grafana는 실시간 모니터링 데이터의 시각화에 탁월합니다."
    },
    {
        "id": 9,
        "question": "개발팀에서 최신 기술 동향을 파악하는 가장 효과적인 방법은 무엇인가요?",
        "correct_answer": "기술 블로그와 컨퍼런스 참여를 통해 최신 정보를 얻는 것입니다.",
        "explanation": "업계 트렌드를 파악하고 네트워킹에도 도움이 됩니다."
    },
    {
        "id": 10,
        "question": "신기술 도입 시 고려해야 할 가장 중요한 요소는 무엇인가요?",
        "correct_answer": "비용 효율성, 기술 적합성, 유지보수 용이성을 모두 고려해야 합니다.",
        "explanation": "종합적인 검토를 통해 최적의 도입 결정을 내려야 합니다."
    }
]

# 정답만 임베딩 대상으로 생성 (주관식 정답)
text_to_embed = []
for quiz in developer_quiz_dataset:
    text_to_embed.append(quiz["correct_answer"])

# 전체 텍스트에 대해 한 번의 배치 요청으로 임베딩 생성
response = openai.embeddings.create(
    input=text_to_embed,
    model="text-embedding-3-small"
)

# 각 텍스트에 대한 임베딩 리스트 생성
embeddings_list = [data.embedding for data in response.data]
embedding_size = len(embeddings_list[0])  # 첫 번째 임베딩 벡터의 길이를 사용

# Qdrant 클라이언트 생성 (Docker 컨테이너가 localhost:6333에서 실행 중이어야 함)
qdrant_client = QdrantClient(host="localhost", port=6333)

# 컬렉션 이름 설정 및 컬렉션 생성 (이미 존재하지 않을 경우)
collection_name = "developer_onboarding_quiz"
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )

# 각 항목에 대해 임베딩과 함께 텍스트 및 기타 메타데이터를 Qdrant에 업로드
points = []
for quiz, embedding in zip(developer_quiz_dataset, embeddings_list):
    points.append({
        "id": quiz["id"],
        "vector": embedding,
        "payload": {
            "quiz_id": quiz["id"],
            "question": quiz["question"],
            "answer": quiz["correct_answer"],
            "explanation": quiz["explanation"]
        }
    })

qdrant_client.upsert(
    collection_name=collection_name,
    points=points
)

print("온보딩 문제 데이터가 Qdrant 벡터 DB에 저장되었습니다!")
