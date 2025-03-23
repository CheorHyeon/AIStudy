# AI Study Project

이 프로젝트는 AI 기술을 학습하고 활용하는 스터디 기록용 저장소입니다.

프로젝트 메인 홈에 나오는 해당 Readme 에서는 개념 부분을 종합하여 정리하고, 단계별로 학습한 스터디 기록을 readme로 남겨둘 계획입니다.

---

## 파인튜닝과 RAG

- **파인튜닝(Fine-tuning):**

  - 기존의 대형 언어 모델(LLM)을 특정 도메인이나 작업에 맞게 일부를 재학습 시키는 방법
  - 어떤 사람을 디지털화하는 동작에 굉장히 잘 작동함
    - 철학자가 쓴 책을 학습시켜서 철학자에게 심리 상담을 받는 듯한 느낌을 받는 서비스 등 존재

- **놀리지 베이스**
  - 가진 지식을 벡터 데이터베이스 형태로 변경
  - 언어 모델이 만들어 둔 벡터 데이터베이스를 찾아보도록 만드는 방식
  - 벡터 DB에 저장되어 있는 정보를 기반으로 답변하도록 만들 수 있음

[파인튜닝, 놀리지베이스 출처 영상](https://www.youtube.com/watch?v=SKFHCdkrqUA)

- **RAG (Retrieval-Augmented Generation):**
  - 외부 지식을 검색하여 생성 모델의 답변에 반영하는 기법
    - Retrieval(검색) : 외부 데이터, 소스를 **검색**
    - Augmented(증강) : 나의 질문을 **보강해서**
    - Generation(생성) : 더 좋은 답변을 **생성**
  - 답변할 때 확실한 출처로 말하기

[RAG 설명 출처 유튜브 영상 + GPT](https://www.youtube.com/watch?v=6m_bntSp21I)

## 실습 1. Lang chain과 Open AI API를 이용한 간단한 질의응답 코드

- Python, Lang chain, Open AI API Key 활용 간단 질의응답 서비스[]()

---

(..추가 중)
