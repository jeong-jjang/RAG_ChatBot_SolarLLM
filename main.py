"""
RAG ChatBot만들기
- Solar LLM
- Langchain

"""

from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage

import os
from dotenv import load_dotenv

# 디버그: dotenv 파일을 로드하고 API 키를 확인
print("dotenv 파일 로드 중...")
load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")

if api_key is None:
    print("API 키를 찾을 수 없습니다. .env 파일을 확인하세요.")
else:
    print("API 키가 성공적으로 로드되었습니다.")

try:
    print("ChatUpstage 인스턴스 생성 중...")
    chat = ChatUpstage(api_key=api_key)
    print("ChatUpstage 인스턴스 생성 완료")
    
    messages = [
        SystemMessage(
            content="You are a helpful assistant."
        ),
        HumanMessage(
            content="Hi, how are you?"
        )
    ]

    # 디버그: 메시지가 제대로 생성되는지 확인
    print("생성된 메시지:", messages)

    print("메시지 전송 중...")
    response = chat.invoke(messages)

    # 디버그: 응답이 제대로 반환되는지 확인
    print("받은 응답:", response)

except Exception as e:
    print("오류가 발생했습니다:", e)
