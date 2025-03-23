# 필요한 라이브러리 임포트
## Open AI
from langchain_openai import OpenAI  # 최신 방식으로 import
## 프롬프트 템플릿 -> 어떻게 질문을 보여줄 지 미리 작성하는 방법
from langchain.prompts import PromptTemplate 
## env 파일을 읽어올 수 있는 도구
from dotenv import load_dotenv 
## 컴퓨터 안의 환경 변수나 파일 경로 등을 다루기 위한 Python의 기본 모듈
import os

# 현재 파일(main.py)이 있는 폴더 기준으로 상위 폴더의 env.env 경로 계산
# load_dotenv() 함수는 .env 파일에 저장된 환경변수를 읽어서 os.environ dictionary에 추가해줌
env_path = os.path.join(os.path.dirname(__file__), "../env.env") 
load_dotenv(env_path)

# OpenAI API 키 설정 (환경변수 사용)
## OpenAI 모델에서는 별도 API키 매개변수로 넘기지 않아도 내부에서 자동으로 읽어 사용하는 경우가 많지만 에러나서 명시적 호출
api_key = os.environ.get("OPENAI_API_KEY") # 위에서 추가한거 꺼내오기


# 프롬프트 템플릿 생성: 사용자가 입력한 질문에 대해 "Q: {question}\nA:" 형식의 프롬프트를 만듭니다.
prompt = PromptTemplate(
    input_variables=["question"], # 질문을 입력할 때 질문의 이름 "question" 으로 사용하겠단 의미
    template="Q: {question}\nA:"  # Q : 질문이 들어감, A 뒤에서 답변 시작되도록 만들어진 틀
)

# OpenAI의 LLM 인스턴스 생성
# temperature 값은 모델 답변의 창의성(랜덤성)을 조절합니다. (낮은 값은 결정적인 답변, 높은 값은 다채로운 답변)
# 낮은 온도 초기화 시 응답의 무작위성 줄일 수 있음
llm = OpenAI(temperature=0.5, api_key = api_key)

# LLMChain 생성: 프롬프트와 LLM을 연결하여 질문에 따른 답변을 생성하는 체인입니다.
chain = prompt | llm

# 프로그램 실행: 사용자에게 질문을 입력받고, 해당 질문에 대한 답변을 출력합니다.
def ask_question():
    question = input("질문을 입력하세요: ")
    # 최신 API에서는 chain.invoke()를 사용하여 실행합니다.
    result = chain.invoke({"question": question})
    print("답변:", result)

if __name__ == "__main__": # Python에서 파일을 직접 실행할 때 __name__ 변수가 "__main__" 으로 설정됨 -> 직접 실행때만 함수 실행
    ask_question()