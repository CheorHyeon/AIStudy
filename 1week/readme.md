# LangChain 이용 : 간단한 Q&A 프로그램

이 프로젝트는 LangChain과 OpenAI를 사용하여 간단한 질문-답변(Q&A) 프로그램을 구현해본 예제입니다.  
프로젝트에서는 최신 방식의 `langchain-openai` 패키지와 환경변수를 로드하기 위한 `python-dotenv`를 사용합니다.

## 기능 개요

- **환경변수 로드:** `.env` 파일에서 OpenAI API 키를 불러와 사용합니다.
- **프롬프트 템플릿:** 사용자의 질문을 정해진 형식(`Q: {question}\nA:`)으로 구성합니다.
- **OpenAI 모델 호출:** 최신 `langchain-openai` 패키지를 사용하여 OpenAI LLM을 초기화합니다.
- **체인 구성:** 프롬프트와 LLM을 파이프(`|`) 연산자를 통해 연결하여 질문을 처리합니다.
- **질문-답변 실행:** 사용자가 입력한 질문에 대해 모델이 답변을 생성합니다.

## 설치 및 설정

1. **Python 환경 준비:** Python 3.2 버전을 설치합니다.

2. **필요 패키지 설치:** 터미널에서 아래 명령어를 실행합니다.

   ```bash
   pip install python-dotenv
   pip install langchain
   pip install langchain-openai
   ```

3. **env 파일 생성:** 프로젝트 루트 디렉토리에 `env.env` 파일을 생성하고 API 키를 입력합니다. (open AI)

```env
OPENAI_API_KEY=***************************
```

4. **프로그램 실행:** 터미널에서 아래 명령어로 프로그램 실행

```bash
python main.py
```
